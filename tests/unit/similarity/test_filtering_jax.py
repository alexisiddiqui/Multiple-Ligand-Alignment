import pytest
import jax.numpy as jnp
import numpy as np
from ligand_neff.similarity.filtering_jax import filter_references_topk
from ligand_neff.similarity.filtering import filter_references

def test_filter_references_topk_matches_legacy():
    # Synthetic data
    n_db = 1000
    fp_size = 64
    max_refs = 50
    threshold = 0.5
    
    rng = np.random.default_rng(42)
    q_fp = (rng.random(fp_size) > 0.5).astype(np.float32)
    db_fps = (rng.random((n_db, fp_size)) > 0.8).astype(np.float32)
    
    # Add a few exact matches to guarantee strong overlaps above threshold
    for i in range(10):
        db_fps[i] = q_fp
        
    # Legacy filter
    legacy_res = filter_references(jnp.array(q_fp), jnp.array(db_fps), threshold, max_refs)
    
    # JAX filter
    jax_res = filter_references_topk(jnp.array(q_fp), jnp.array(db_fps), threshold, max_refs)
    
    # Compare number of valid references
    assert legacy_res.n_valid == jax_res.n_valid
    
    # Compare selected fingerprints
    # Tie breaking might order them differently, so we check sets
    # The similarities should be exactly the same when sorted
    legacy_sims = np.sort(np.asarray(legacy_res.similarities))[::-1]
    jax_sims = np.sort(np.asarray(jax_res.similarities))[::-1]
    
    np.testing.assert_allclose(legacy_sims, jax_sims, rtol=1e-5, atol=1e-5)
    
def test_filter_references_topk_small_db():
    n_db = 10  # Smaller than max_refs
    fp_size = 32
    max_refs = 20
    threshold = 0.5
    
    rng = np.random.default_rng(42)
    q_fp = (rng.random(fp_size) > 0.5).astype(np.float32)
    db_fps = (rng.random((n_db, fp_size)) > 0.5).astype(np.float32)
    
    jax_res = filter_references_topk(jnp.array(q_fp), jnp.array(db_fps), threshold, max_refs)
    
    # Output shapes should still be `max_refs`
    assert jax_res.fps.shape == (max_refs, fp_size)
    assert jax_res.similarities.shape == (max_refs,)
    assert jax_res.mask.shape == (max_refs,)
    
    # The mask should have at most `n_db` valid entries
    assert jax_res.n_valid <= n_db
