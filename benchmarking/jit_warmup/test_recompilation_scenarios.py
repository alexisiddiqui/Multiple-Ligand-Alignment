"""Recompilation regression tests.

Tests that specific design decisions (JAX scalar n_valid, static
padding, varying mask patterns) do NOT cause spurious recompilation.

Note: `inverse_degree_weights` is NOT `@jax.jit` decorated — it's a
plain Python function that calls internal JIT'd ops. Individual JAX
primitives (scan, scatter, broadcast_in_dim) lazily compile on each
eager call. Compilation counting for this function uses timing-based
verification instead of raw backend compile events.
"""

import time
import pytest
import numpy as np
import jax
import jax.numpy as jnp

from ligand_neff.similarity.filtering import FilteredReferences
from ligand_neff.neff.weighting import inverse_degree_weights
from ligand_neff.neff.per_atom import per_atom_neff_single_radius

from benchmarking.jit_warmup._helpers import (
    count_compilations,
    BENCH_FP_SIZE,
    BENCH_MAX_REFS,
    BENCH_CHUNK_SIZE,
)


def _timed_call(fn, *args, **kwargs):
    """Time a single call including block_until_ready()."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    result.block_until_ready()
    return time.perf_counter() - t0


class TestMaskVariation:
    """Varying mask contents (not shape) must NOT cause shape-level recompilation.

    Since `inverse_degree_weights` is not `@jax.jit`, we verify via
    consistent execution time (no compilation spikes) after warmup.
    """

    def test_varying_n_valid_consistent_timing(self, filtered_refs, config, rng):
        """10 calls with different mask patterns: execution time stays stable."""
        jax.clear_caches()

        # Thorough warmup: 3 calls to fully warm all lazy paths
        for n_valid in [200, 100, 300]:
            mask = jnp.zeros(BENCH_MAX_REFS, dtype=bool).at[:n_valid].set(True)
            w = inverse_degree_weights(
                filtered_refs.fps, mask,
                threshold=config.cluster_threshold,
                chunk_size=config.inverse_degree_chunk_size,
            )
            w.block_until_ready()

        # Measure: 10 calls with varied masks
        times = []
        for n_valid in [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]:
            mask = jnp.zeros(BENCH_MAX_REFS, dtype=bool).at[:n_valid].set(True)
            t = _timed_call(
                inverse_degree_weights,
                filtered_refs.fps, mask,
                threshold=config.cluster_threshold,
                chunk_size=config.inverse_degree_chunk_size,
            )
            times.append(t)

        # No call should take more than 5× the median (would indicate recompilation)
        median_time = np.median(times)
        max_time = max(times)
        ratio = max_time / median_time if median_time > 0 else float("inf")
        assert ratio < 5.0, (
            f"Max time ({max_time:.4f}s) is {ratio:.1f}× median ({median_time:.4f}s). "
            f"This suggests shape-triggered recompilation during mask variation."
        )


class TestNValidAsJaxScalar:
    """n_valid stored as jnp.int32 → consistent timing (no recompilation)."""

    def test_different_n_valid_values(self, rng):
        """Build FilteredReferences with varying n_valid; timing stays stable."""
        jax.clear_caches()

        refs_list = []
        for n_valid in [50, 100, 150, 200, 250, 300]:
            fps = np.zeros((BENCH_MAX_REFS, BENCH_FP_SIZE), dtype=np.float32)
            fps[:n_valid] = rng.integers(0, 2, (n_valid, BENCH_FP_SIZE)).astype(np.float32)
            mask = np.zeros(BENCH_MAX_REFS, dtype=bool)
            mask[:n_valid] = True

            refs = FilteredReferences(
                fps=jnp.array(fps),
                mask=jnp.array(mask),
                similarities=jnp.zeros(BENCH_MAX_REFS),
                n_valid=jnp.array(n_valid, dtype=jnp.int32),
            )
            refs_list.append(refs)

        # Thorough warmup: 3 calls
        for refs in refs_list[:3]:
            w = inverse_degree_weights(
                refs.fps, refs.mask,
                threshold=0.7, chunk_size=BENCH_CHUNK_SIZE,
            )
            w.block_until_ready()

        # Measure remaining calls
        times = []
        for refs in refs_list[3:]:
            t = _timed_call(
                inverse_degree_weights,
                refs.fps, refs.mask,
                threshold=0.7, chunk_size=BENCH_CHUNK_SIZE,
            )
            times.append(t)

        # All calls should be within 5× of each other (no compilation spikes)
        if len(times) >= 2:
            median_time = np.median(times)
            max_time = max(times)
            ratio = max_time / median_time if median_time > 0 else float("inf")
            assert ratio < 5.0, (
                f"Max time ({max_time:.4f}s) is {ratio:.1f}× median ({median_time:.4f}s). "
                f"This suggests n_valid value change triggered recompilation."
            )


class TestFpSizeChange:
    """Changing fp_size is a legitimate shape change → triggers new compilation."""

    def test_fp_size_2048_to_4096(self, rng):
        jax.clear_caches()

        # Warmup at fp_size=2048
        q_2048 = jnp.array(rng.integers(0, 2, 2048).astype(np.float32))
        db_2048 = jnp.array(rng.integers(0, 2, (100, 2048)).astype(np.float32))

        from ligand_neff.similarity.tanimoto import bulk_tanimoto

        _ = bulk_tanimoto(q_2048, db_2048).block_until_ready()

        # Switch to fp_size=4096 → should trigger exactly 1 new compilation
        q_4096 = jnp.array(rng.integers(0, 2, 4096).astype(np.float32))
        db_4096 = jnp.array(rng.integers(0, 2, (100, 4096)).astype(np.float32))

        with count_compilations("bulk_tanimoto") as cc:
            _ = bulk_tanimoto(q_4096, db_4096).block_until_ready()
        assert cc.count >= 1, (
            f"Changing fp_size should trigger recompilation, got {cc.count}"
        )

        # Second call at 4096 → cache hit
        with count_compilations("bulk_tanimoto") as cc2:
            _ = bulk_tanimoto(q_4096 * 0.9, db_4096).block_until_ready()
        assert cc2.count == 0, (
            f"Second call at fp_size=4096 should be cache hit, got {cc2.count}"
        )


class TestAtomCountVariety:
    """per_atom_neff: K unique n_atoms → exactly K compilations, then 0."""

    def test_exact_compilation_count(self, filtered_refs, weights_array, rng):
        jax.clear_caches()

        n_atoms_values = list(range(20, 35))  # 15 distinct values
        K = len(n_atoms_values)

        # Warmup phase: expect exactly K compilations
        with count_compilations("per_atom_neff") as cc:
            for n_atoms in n_atoms_values:
                mask = jnp.array(
                    rng.integers(0, 2, (n_atoms, BENCH_FP_SIZE)).astype(np.float32)
                )
                neff = per_atom_neff_single_radius(mask, filtered_refs.fps, weights_array)
                neff.block_until_ready()

        assert cc.count == K, (
            f"Expected exactly {K} compilations for {K} distinct n_atoms, "
            f"got {cc.count}"
        )

        # Revisit all → 0 compilations
        with count_compilations("per_atom_neff") as cc2:
            for n_atoms in n_atoms_values:
                mask = jnp.array(
                    rng.integers(0, 2, (n_atoms, BENCH_FP_SIZE)).astype(np.float32)
                )
                neff = per_atom_neff_single_radius(mask, filtered_refs.fps, weights_array)
                neff.block_until_ready()
        assert cc2.count == 0, (
            f"Revisiting seen n_atoms caused {cc2.count} recompilations (expected 0)"
        )
