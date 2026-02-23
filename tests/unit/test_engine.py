import pytest
import numpy as np
from rdkit import Chem

from ligand_neff.config import NeffConfig
from ligand_neff.engine import NeffEngine
from ligand_neff.io.db_cache import build_db_cache

def test_engine_init_and_compute():
    config = NeffConfig(fp_radii=(1, 2, 3), fp_size=2048, max_references=2)
    n_db = 10
    
    rng = np.random.default_rng(42)
    precomputed = {
        "radius_1": (rng.random((n_db, 2048)) > 0.5).astype(np.float32),
        "radius_2": (rng.random((n_db, 2048)) > 0.5).astype(np.float32),
        "radius_3": (rng.random((n_db, 2048)) > 0.5).astype(np.float32),
    }
    
    # 1. Test engine initialization with dict
    engine = NeffEngine(config, precomputed_db=precomputed)
    assert engine.db_cache.n_db == n_db
    
    # 2. Test engine initialize with DbCache
    cache = build_db_cache(precomputed, (1, 2, 3))
    engine2 = NeffEngine(config, precomputed_db=cache, compile_on_init=False)
    assert engine2.db_cache.db_fps_stacked.shape == (3, 10, 2048)
    
    # Create simple dummy molecule to test compute
    mol = Chem.MolFromSmiles("CC")
    
    res = engine.compute(mol)
    assert res.atom_neff.shape == (2,)
    assert res.atom_confidence.shape == (2,)
    assert res.config == config
    assert res.query_mol == mol
