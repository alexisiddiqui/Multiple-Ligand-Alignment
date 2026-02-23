import pytest
import jax.numpy as jnp
import numpy as np

from ligand_neff.io.db_cache import DbCache, build_db_cache

def test_build_db_cache_valid():
    radii = [1, 2, 3]
    fp_size = 32
    n_db = 10
    
    # Create somewhat realistic dictionary
    precomputed = {}
    for r in radii:
        precomputed[f"radius_{r}"] = np.ones((n_db, fp_size), dtype=np.float32) * r
        
    cache = build_db_cache(precomputed, fp_radii=radii, dtype=jnp.float32)
    
    assert isinstance(cache, DbCache)
    assert cache.fp_radii == tuple(radii)
    assert cache.fp_size == fp_size
    assert cache.n_db == n_db
    assert cache.db_fps_stacked.shape == (3, n_db, fp_size)
    assert cache.db_fps_stacked.dtype == jnp.float32
    
    # Check that per-radius dict works and values match
    for i, r in enumerate(radii):
        assert r in cache.db_fps_per_radius
        arr = cache.db_fps_per_radius[r]
        assert arr.shape == (n_db, fp_size)
        assert jnp.all(arr == r)

def test_build_db_cache_missing_radius():
    radii = [1, 2, 3]
    precomputed = {
        "radius_1": np.zeros((10, 32)),
        "radius_2": np.zeros((10, 32)),
    }
    with pytest.raises(KeyError, match="Missing radius 3"):
        build_db_cache(precomputed, fp_radii=radii)

def test_build_db_cache_mismatched_db_size():
    radii = [1, 2]
    precomputed = {
        "radius_1": np.zeros((10, 32)),
        "radius_2": np.zeros((12, 32)), # different n_db
    }
    with pytest.raises(ValueError, match="Mismatched database size"):
        build_db_cache(precomputed, fp_radii=radii)

def test_build_db_cache_mismatched_fp_size():
    radii = [1, 2]
    precomputed = {
        "radius_1": np.zeros((10, 32)),
        "radius_2": np.zeros((10, 64)), # different fp_size
    }
    with pytest.raises(ValueError, match="Mismatched fp_size"):
        build_db_cache(precomputed, fp_radii=radii)
