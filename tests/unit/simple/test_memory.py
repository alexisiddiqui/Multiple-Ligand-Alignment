import pytest
import jax.numpy as jnp
import jax
from ligand_neff.neff.weighting import inverse_degree_weights

def test_inverse_degree_memory_25k(monkeypatch):
    """
    Simulates max_refs=25K to ensure it doesn't OOM due to chunking.
    We don't need real footprints, just large arrays of ones/zeros.
    """
    # 25,000 refs * 2048 bits
    max_refs = 25_000
    fp_size = 2048
    chunk_size = 2048
    
    # Mock data
    fps = jnp.zeros((max_refs, fp_size), dtype=jnp.float32)
    mask = jnp.ones(max_refs, dtype=bool)
    
    # This should run in < 500 MB VRAM due to chunk_size=2048.
    # Without chunking, pairwise (25k, 25k) is ~2.5GB float32.
    weights = inverse_degree_weights(
        fps=fps, 
        mask=mask, 
        threshold=0.7, 
        chunk_size=chunk_size
    )
    
    # Force evaluation
    _ = weights.block_until_ready()
    
    # If we made it here without getting Jax OOM (RESOURCE_EXHAUSTED), 
    # the chunking logic worked.
    assert weights.shape == (max_refs,)
