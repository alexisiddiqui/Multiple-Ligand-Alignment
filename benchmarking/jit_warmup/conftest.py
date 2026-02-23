"""Shared pytest fixtures for JIT warmup benchmarks.

Uses small array sizes (max_references=512, fp_size=2048) to keep
benchmarks fast (~seconds). The goal is to test compilation behaviour,
not throughput.
"""

import pytest
import numpy as np
import jax.numpy as jnp

from ligand_neff.config import NeffConfig
from ligand_neff.similarity.filtering import FilteredReferences

from benchmarking.jit_warmup._helpers import (
    BENCH_FP_SIZE,
    BENCH_MAX_REFS,
    BENCH_CHUNK_SIZE,
    BENCH_N_DB,
    BENCH_N_ATOMS_DEFAULT,
)


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def rng():
    """Seeded random generator for reproducibility."""
    return np.random.default_rng(42)


@pytest.fixture
def config():
    """Small NeffConfig for fast benchmarks."""
    return NeffConfig(
        fp_radii=(1, 2, 3),
        fp_size=BENCH_FP_SIZE,
        max_references=BENCH_MAX_REFS,
        inverse_degree_chunk_size=BENCH_CHUNK_SIZE,
        weighting="inverse_degree",
    )


@pytest.fixture
def synthetic_query(rng):
    """Single (fp_size,) float32 query fingerprint."""
    fp = rng.integers(0, 2, size=BENCH_FP_SIZE).astype(np.float32)
    return jnp.array(fp)


@pytest.fixture
def synthetic_db(rng):
    """(n_db, fp_size) float32 database fingerprints."""
    db = rng.integers(0, 2, size=(BENCH_N_DB, BENCH_FP_SIZE)).astype(np.float32)
    return jnp.array(db)


@pytest.fixture
def filtered_refs(rng):
    """Pre-built FilteredReferences with known n_valid=200, padded to max_refs."""
    n_valid = 200
    fps = np.zeros((BENCH_MAX_REFS, BENCH_FP_SIZE), dtype=np.float32)
    fps[:n_valid] = rng.integers(0, 2, size=(n_valid, BENCH_FP_SIZE)).astype(np.float32)

    mask = np.zeros(BENCH_MAX_REFS, dtype=bool)
    mask[:n_valid] = True

    sims = np.zeros(BENCH_MAX_REFS, dtype=np.float32)
    sims[:n_valid] = rng.uniform(0.3, 1.0, size=n_valid).astype(np.float32)

    return FilteredReferences(
        fps=jnp.array(fps),
        mask=jnp.array(mask),
        similarities=jnp.array(sims),
        n_valid=jnp.array(n_valid, dtype=jnp.int32),
    )


@pytest.fixture
def atom_bit_mask(rng):
    """(n_atoms, fp_size) float32 atom bit mask."""
    mask = rng.integers(0, 2, size=(BENCH_N_ATOMS_DEFAULT, BENCH_FP_SIZE)).astype(np.float32)
    return jnp.array(mask)


@pytest.fixture
def weights_array(rng):
    """(max_refs,) float32 weights for per-atom neff."""
    w = np.zeros(BENCH_MAX_REFS, dtype=np.float32)
    w[:200] = rng.uniform(0.0, 1.0, size=200).astype(np.float32)
    return jnp.array(w)
