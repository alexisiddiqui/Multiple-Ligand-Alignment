import pytest
import jax.numpy as jnp
import numpy as np

from ligand_neff.neff.pipeline import compute_neff_core, single_radius_pipeline_unweighted

def test_compute_neff_core_vmap_equivalent():
    fp_size = 64
    n_db = 100
    n_atoms = 10
    max_refs = 20
    n_radii = 3
    
    rng = np.random.default_rng(42)
    q_fps_stacked = (rng.random((n_radii, fp_size)) > 0.5).astype(np.float32)
    db_fps_stacked = (rng.random((n_radii, n_db, fp_size)) > 0.8).astype(np.float32)
    atom_masks_stacked = (rng.random((n_radii, n_atoms, fp_size)) > 0.5).astype(np.float32)
    
    # Inject exact matches to ensure some Neff > 0
    for r in range(n_radii):
        for i in range(5):
            db_fps_stacked[r, i] = q_fps_stacked[r]
            
    # Calculate one by one manually using single radius unweighted pipeline
    neff_stack = []
    n_valids = []
    for r in range(n_radii):
        neff_r, n_valid = single_radius_pipeline_unweighted(
            jnp.array(q_fps_stacked[r]),
            jnp.array(db_fps_stacked[r]),
            jnp.array(atom_masks_stacked[r]),
            threshold=0.5,
            max_refs=max_refs,
            min_overlap=1.0,
            atom_norm="q_length"
        )
        neff_stack.append(neff_r)
        n_valids.append(n_valid)
        
    neff_stack = jnp.stack(neff_stack, axis=0) # (3, 10)
    
    radius_weights = (0.2, 0.5, 0.3)
    
    # Call the JIT core
    combined_neff, confidence, neff_stack_core, lam_core, n_refs_used = compute_neff_core(
        jnp.array(q_fps_stacked),
        jnp.array(db_fps_stacked),
        jnp.array(atom_masks_stacked),
        threshold=0.5,
        cluster_threshold=0.8,
        lambda_quantile=0.9,
        lambda_fixed=10.0,
        radius_weights=radius_weights,
        weighting="none",
        max_refs=max_refs,
        min_overlap=1.0,
        chunk_size=16,
        atom_norm="q_length",
        aggregation="geometric",
        lambda_mode="adaptive",
    )
    
    # Verify outputs match manual loop
    np.testing.assert_allclose(np.asarray(neff_stack_core), np.asarray(neff_stack), rtol=1e-5, atol=1e-5)
    
    # Check max refs
    assert int(np.asarray(n_refs_used)) == int(jnp.max(jnp.array(n_valids)))
    
    # Ensure shapes are strictly consistent
    assert combined_neff.shape == (n_atoms,)
    assert confidence.shape == (n_atoms,)
    assert neff_stack_core.shape == (n_radii, n_atoms)
