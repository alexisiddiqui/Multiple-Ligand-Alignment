import pytest
import jax.numpy as jnp
import numpy as np

from ligand_neff.neff.pipeline import single_radius_pipeline_unweighted, single_radius_pipeline_inverse_degree

def test_single_radius_pipeline_unweighted():
    fp_size = 64
    n_db = 100
    n_atoms = 10
    max_refs = 20
    
    rng = np.random.default_rng(42)
    q_fp = (rng.random(fp_size) > 0.5).astype(np.float32)
    db_fps = (rng.random((n_db, fp_size)) > 0.8).astype(np.float32)
    atom_mask = (rng.random((n_atoms, fp_size)) > 0.5).astype(np.float32)
    
    # 5 matching references
    for i in range(5):
        db_fps[i] = q_fp
        
    neff_r, n_valid = single_radius_pipeline_unweighted(
        jnp.array(q_fp),
        jnp.array(db_fps),
        jnp.array(atom_mask),
        threshold=0.5,
        max_refs=max_refs,
        min_overlap=1.0,
        atom_norm="q_length"
    )
    
    assert neff_r.shape == (n_atoms,)
    assert n_valid.shape == () 
    # n_valid is at least 5 because we injected 5 identical fingerprints
    assert int(np.asarray(n_valid)) >= 5

def test_single_radius_pipeline_inverse_degree():
    fp_size = 64
    n_db = 100
    n_atoms = 10
    max_refs = 20
    
    rng = np.random.default_rng(42)
    q_fp = (rng.random(fp_size) > 0.5).astype(np.float32)
    db_fps = (rng.random((n_db, fp_size)) > 0.8).astype(np.float32)
    atom_mask = (rng.random((n_atoms, fp_size)) > 0.5).astype(np.float32)
    
    for i in range(8):
        db_fps[i] = q_fp
        
    neff_r, n_valid = single_radius_pipeline_inverse_degree(
        jnp.array(q_fp),
        jnp.array(db_fps),
        jnp.array(atom_mask),
        threshold=0.5,
        cluster_threshold=0.8,
        max_refs=max_refs,
        min_overlap=1.0,
        chunk_size=16,
        atom_norm="q_length"
    )
    
    assert neff_r.shape == (n_atoms,)
    assert n_valid.shape == ()
    assert int(np.asarray(n_valid)) >= 8
