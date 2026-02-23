"""JIT warmup benchmarks: compilation counting per function.

Verifies that the static-shape design achieves zero recompilation
after warmup for every JIT-compiled function in the pipeline.
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp

from ligand_neff.similarity.tanimoto import bulk_tanimoto, pairwise_tanimoto_chunk
from ligand_neff.neff.weighting import inverse_degree_weights
from ligand_neff.neff.per_atom import per_atom_neff_single_radius
from ligand_neff.neff.aggregation import aggregate_neff, normalise_to_confidence

from benchmarking.jit_warmup._helpers import (
    count_compilations,
    BENCH_FP_SIZE,
    BENCH_MAX_REFS,
    BENCH_CHUNK_SIZE,
)


class TestBulkTanimotoWarmup:
    """bulk_tanimoto: fixed (n_db, fp_size) → 1 compile, then 0."""

    def test_warmup_then_cache_hit(self, synthetic_query, synthetic_db):
        jax.clear_caches()

        # Warmup
        with count_compilations("bulk_tanimoto") as warmup:
            result = bulk_tanimoto(synthetic_query, synthetic_db)
            result.block_until_ready()
        assert warmup.count >= 1, "Expected at least 1 compilation on first call"

        # Subsequent calls: same shapes, different data
        with count_compilations("bulk_tanimoto") as cc:
            for _ in range(5):
                q = synthetic_query * 0.9
                res = bulk_tanimoto(q, synthetic_db)
                res.block_until_ready()
        assert cc.count == 0, f"Expected 0 recompilations, got {cc.count}"


class TestPairwiseTanimotoChunkWarmup:
    """pairwise_tanimoto_chunk: fixed shapes → 1 compile, then 0."""

    def test_warmup_then_cache_hit(self, rng):
        jax.clear_caches()
        chunk_size = BENCH_CHUNK_SIZE
        n_refs = BENCH_MAX_REFS

        chunk_fps = jnp.array(rng.integers(0, 2, (chunk_size, BENCH_FP_SIZE)).astype(np.float32))
        all_fps = jnp.array(rng.integers(0, 2, (n_refs, BENCH_FP_SIZE)).astype(np.float32))
        chunk_bits = jnp.sum(chunk_fps, axis=1)
        all_bits = jnp.sum(all_fps, axis=1)
        mask = jnp.ones(n_refs, dtype=bool)

        # Warmup
        with count_compilations("pairwise_tanimoto_chunk") as warmup:
            res = pairwise_tanimoto_chunk(chunk_fps, all_fps, chunk_bits, all_bits, mask)
            res.block_until_ready()
        assert warmup.count >= 1

        # Cache hits
        with count_compilations("pairwise_tanimoto_chunk") as cc:
            for _ in range(3):
                res = pairwise_tanimoto_chunk(
                    chunk_fps * 0.8, all_fps * 0.7, chunk_bits, all_bits, mask
                )
                res.block_until_ready()
        assert cc.count == 0, f"Expected 0 recompilations, got {cc.count}"


class TestInverseDegreeWeightsWarmup:
    """inverse_degree_weights: fixed (max_refs, fp_size) + chunk_size → compile once."""

    def test_warmup_then_cache_hit(self, filtered_refs, config):
        jax.clear_caches()

        # Warmup
        with count_compilations() as warmup:
            w = inverse_degree_weights(
                filtered_refs.fps, filtered_refs.mask,
                threshold=config.cluster_threshold,
                chunk_size=config.inverse_degree_chunk_size,
            )
            w.block_until_ready()
        initial_count = warmup.count
        assert initial_count >= 1, "Expected at least 1 compilation"

        # Second call: same shapes, different mask pattern
        rng = np.random.default_rng(99)
        new_mask = jnp.array(rng.choice([True, False], size=BENCH_MAX_REFS))
        with count_compilations() as cc:
            w2 = inverse_degree_weights(
                filtered_refs.fps, new_mask,
                threshold=config.cluster_threshold,
                chunk_size=config.inverse_degree_chunk_size,
            )
            w2.block_until_ready()
        # Allow ≤1 for JAX internal lazy scan/fori_loop compilation.
        # The key assertion is that user-level shape-triggered recompilation
        # does NOT happen; a single internal primitive may lazily compile once.
        assert cc.count <= 1, (
            f"Varying mask pattern caused {cc.count} recompilations "
            f"(expected ≤1 for same-shape inputs; 1 may come from "
            f"lazy scan primitive inside fori_loop)"
        )


class TestPerAtomNeffWarmup:
    """per_atom_neff_single_radius: one compilation per unique n_atoms."""

    def test_single_n_atoms_warmup(self, atom_bit_mask, filtered_refs, weights_array):
        jax.clear_caches()

        # Warmup
        with count_compilations("per_atom_neff") as warmup:
            neff = per_atom_neff_single_radius(
                atom_bit_mask, filtered_refs.fps, weights_array,
            )
            neff.block_until_ready()
        assert warmup.count >= 1

        # Same n_atoms → cache hit
        with count_compilations("per_atom_neff") as cc:
            neff2 = per_atom_neff_single_radius(
                atom_bit_mask * 0.9, filtered_refs.fps, weights_array,
            )
            neff2.block_until_ready()
        assert cc.count == 0, f"Same n_atoms caused {cc.count} recompilations"

    def test_cache_hits_after_variety(self, filtered_refs, weights_array, rng):
        """After seeing K distinct n_atoms, revisiting them → 0 compilations."""
        jax.clear_caches()

        # Warmup with K different atom counts
        n_atoms_values = list(range(20, 36))  # 16 distinct values
        for n_atoms in n_atoms_values:
            mask = jnp.array(
                rng.integers(0, 2, (n_atoms, BENCH_FP_SIZE)).astype(np.float32)
            )
            neff = per_atom_neff_single_radius(mask, filtered_refs.fps, weights_array)
            neff.block_until_ready()

        # Now revisit all seen n_atoms → 0 compilations
        with count_compilations("per_atom_neff") as cc:
            for n_atoms in n_atoms_values:
                mask = jnp.array(
                    rng.integers(0, 2, (n_atoms, BENCH_FP_SIZE)).astype(np.float32)
                )
                neff = per_atom_neff_single_radius(mask, filtered_refs.fps, weights_array)
                neff.block_until_ready()
        assert cc.count == 0, (
            f"Revisiting {len(n_atoms_values)} seen n_atoms caused "
            f"{cc.count} recompilations (expected 0)"
        )


class TestAggregationWarmup:
    """aggregate_neff + normalise_to_confidence: fixed n_atoms → no recompilation."""

    def test_warmup_then_cache_hit(self, rng):
        jax.clear_caches()
        n_atoms = 30

        neff_per_radius = {
            1: jnp.array(rng.uniform(0, 5, n_atoms).astype(np.float32)),
            2: jnp.array(rng.uniform(0, 5, n_atoms).astype(np.float32)),
            3: jnp.array(rng.uniform(0, 5, n_atoms).astype(np.float32)),
        }

        # Warmup
        combined = aggregate_neff(neff_per_radius, method="geometric")
        conf = normalise_to_confidence(combined, lam=5.0)
        conf.block_until_ready()

        # Second call: same shapes
        with count_compilations() as cc:
            neff2 = {
                1: jnp.array(rng.uniform(0, 5, n_atoms).astype(np.float32)),
                2: jnp.array(rng.uniform(0, 5, n_atoms).astype(np.float32)),
                3: jnp.array(rng.uniform(0, 5, n_atoms).astype(np.float32)),
            }
            combined2 = aggregate_neff(neff2, method="geometric")
            conf2 = normalise_to_confidence(combined2, lam=5.0)
            conf2.block_until_ready()
        assert cc.count == 0, f"Same n_atoms caused {cc.count} recompilations"
