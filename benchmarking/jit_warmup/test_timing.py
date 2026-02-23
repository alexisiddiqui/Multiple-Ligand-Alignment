"""Wall-clock timing benchmarks for JIT warmup.

Measures first-call (compile+run) vs subsequent-call (run-only) times
to verify that compilation overhead is amortised after warmup.
"""

import time
import pytest
import numpy as np
import jax
import jax.numpy as jnp

from ligand_neff.similarity.tanimoto import bulk_tanimoto
from ligand_neff.neff.weighting import inverse_degree_weights
from ligand_neff.neff.per_atom import per_atom_neff_single_radius

from benchmarking.jit_warmup._helpers import BENCH_FP_SIZE, BENCH_MAX_REFS, BENCH_CHUNK_SIZE


def _timed_call(fn, *args):
    """Time a single call including block_until_ready()."""
    t0 = time.perf_counter()
    result = fn(*args)
    result.block_until_ready()
    return time.perf_counter() - t0, result


class TestFirstVsSubsequentTiming:
    """First call includes compilation; subsequent calls should be faster."""

    def test_bulk_tanimoto(self, synthetic_query, synthetic_db):
        jax.clear_caches()

        # First call: compile + run
        t_first, _ = _timed_call(bulk_tanimoto, synthetic_query, synthetic_db)

        # Subsequent calls: run only
        times = []
        for _ in range(10):
            t, _ = _timed_call(bulk_tanimoto, synthetic_query * 0.9, synthetic_db)
            times.append(t)

        t_mean_subsequent = np.mean(times)
        ratio = t_first / t_mean_subsequent if t_mean_subsequent > 0 else float("inf")

        # First call should be at least 2× slower (compilation overhead)
        assert ratio > 2.0, (
            f"First call ({t_first:.4f}s) should be >2× slower than "
            f"subsequent ({t_mean_subsequent:.4f}s), ratio={ratio:.1f}×"
        )

    def test_inverse_degree_weights(self, filtered_refs, config):
        jax.clear_caches()

        t_first, _ = _timed_call(
            inverse_degree_weights,
            filtered_refs.fps, filtered_refs.mask,
            config.cluster_threshold, config.inverse_degree_chunk_size,
        )

        times = []
        for _ in range(5):
            t, _ = _timed_call(
                inverse_degree_weights,
                filtered_refs.fps, filtered_refs.mask,
                config.cluster_threshold, config.inverse_degree_chunk_size,
            )
            times.append(t)

        t_mean_subsequent = np.mean(times)
        ratio = t_first / t_mean_subsequent if t_mean_subsequent > 0 else float("inf")

        assert ratio > 2.0, (
            f"First call ({t_first:.4f}s) should be >2× slower than "
            f"subsequent ({t_mean_subsequent:.4f}s), ratio={ratio:.1f}×"
        )

    def test_per_atom_neff(self, atom_bit_mask, filtered_refs, weights_array):
        jax.clear_caches()

        t_first, _ = _timed_call(
            per_atom_neff_single_radius,
            atom_bit_mask, filtered_refs.fps, weights_array,
        )

        times = []
        for _ in range(10):
            t, _ = _timed_call(
                per_atom_neff_single_radius,
                atom_bit_mask * 0.9, filtered_refs.fps, weights_array,
            )
            times.append(t)

        t_mean_subsequent = np.mean(times)
        ratio = t_first / t_mean_subsequent if t_mean_subsequent > 0 else float("inf")

        assert ratio > 2.0, (
            f"First call ({t_first:.4f}s) should be >2× slower than "
            f"subsequent ({t_mean_subsequent:.4f}s), ratio={ratio:.1f}×"
        )


class TestBatchAmortisation:
    """Pipeline becomes faster after initial compilations are cached."""

    def test_per_atom_neff_batch(self, filtered_refs, weights_array, rng):
        """Run 20 queries with varying n_atoms. Last 10 should be faster on average."""
        jax.clear_caches()

        # 20 queries with n_atoms in [20, 50] (some overlap expected)
        n_atoms_list = rng.integers(20, 50, size=20)
        times = []

        for n_atoms in n_atoms_list:
            mask = jnp.array(
                rng.integers(0, 2, (int(n_atoms), BENCH_FP_SIZE)).astype(np.float32)
            )
            t, _ = _timed_call(
                per_atom_neff_single_radius,
                mask, filtered_refs.fps, weights_array,
            )
            times.append(t)

        mean_first_5 = np.mean(times[:5])
        mean_last_10 = np.mean(times[10:])

        assert mean_last_10 < mean_first_5, (
            f"Mean time of last 10 ({mean_last_10:.4f}s) should be less than "
            f"mean of first 5 ({mean_first_5:.4f}s) due to cache warmup"
        )
