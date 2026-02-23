import pytest
import jax.numpy as jnp
import numpy as np

from ligand_neff.neff.aggregation import aggregate_neff, aggregate_neff_stacked

def test_aggregate_neff_stacked_matches_legacy():
    n_radii = 3
    n_atoms = 15
    
    rng = np.random.default_rng(42)
    radius_data = {
        1: jnp.array(rng.uniform(0.1, 5.0, size=(n_atoms,)).astype(np.float32)),
        2: jnp.array(rng.uniform(0.1, 5.0, size=(n_atoms,)).astype(np.float32)),
        3: jnp.array(rng.uniform(0.1, 5.0, size=(n_atoms,)).astype(np.float32)),
    }
    
    weights = (0.2, 0.5, 0.3)
    
    # Check all methods
    for method in ["geometric", "mean", "minimum"]:
        res_dict = aggregate_neff(radius_data, method=method, radius_weights=weights)
        
        # Stacked version
        stacked_data = jnp.stack([radius_data[1], radius_data[2], radius_data[3]], axis=0)
        res_stacked = aggregate_neff_stacked(stacked_data, method=method, radius_weights=weights)
        
        np.testing.assert_allclose(np.asarray(res_dict), np.asarray(res_stacked), rtol=1e-6, atol=1e-6)

