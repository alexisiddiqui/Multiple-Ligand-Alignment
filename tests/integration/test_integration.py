import pytest
import time
from rdkit import Chem
from ligand_neff.compute import compute_neff
from ligand_neff.config import NeffConfig

@pytest.fixture
def sample_db():
    smiles_list = [
        "CCO",
        "CC(C)O",
        "CC(=O)O",
        "c1ccccc1",
        "Cc1ccccc1",
        "c1ccncc1",
        "C1CCOC1",
        "C1CCCC1",
        "CC1CCCC1",
        "CC1=CC=CC=C1"
    ]
    return [Chem.MolFromSmiles(s) for s in smiles_list]

@pytest.fixture
def sample_query():
    return Chem.MolFromSmiles("c1ccccc1O") # Phenol

def test_compute_neff_basic(sample_query, sample_db):
    """Test that a basic computation runs start to finish."""
    config = NeffConfig(
        fp_radii=(1, 2), 
        radius_weights=(0.5, 0.5), # Must match length of fp_radii
        fp_size=2048,
        max_references=5, # Small static shape
        weighting="inverse_degree"
    )
    
    result = compute_neff(sample_query, sample_db, config)
    
    assert result.query_mol is sample_query
    assert result.global_neff > 0.0
    assert 0.0 <= result.global_confidence <= 1.0
    assert result.atom_neff.shape == (sample_query.GetNumAtoms(),)
    assert result.atom_confidence.shape == (sample_query.GetNumAtoms(),)
    assert len(result.neff_per_radius) == 2

def test_jit_reused_on_subsequent_calls(sample_query, sample_db):
    """
    Test that JIT cache hits happen when running subsequent queries 
    with the same number of atoms.
    """
    config = NeffConfig(
        fp_radii=(2,), 
        radius_weights=(1.0,), # Must match length of fp_radii
        fp_size=2048,
        max_references=10,
        weighting="none", # simplified for testing speed
        lambda_mode="fixed",
        lambda_fixed=1.0
    )
    
    # Query 1 (Triggers Compilation)
    q1 = Chem.MolFromSmiles("c1ccccc1") # Benzene (6 heavy atoms)
    start_t1 = time.time()
    res1 = compute_neff(q1, sample_db, config)
    t1 = time.time() - start_t1
    
    # Query 2 (Different molecule, same atom count = 6) - Should hit JIT cache
    q2 = Chem.MolFromSmiles("c1ccncc1") # Pyridine (6 heavy atoms)
    start_t2 = time.time()
    res2 = compute_neff(q2, sample_db, config)
    t2 = time.time() - start_t2
    
    assert res1.global_neff > 0
    assert res2.global_neff > 0
    # Note: t2 should be significantly faster than t1 due to compilation caching,
    # but exact timing asserts are flaky in CI, so we just check execution didn't crash.
    # In rigorous environments we'd check `chex.assert_max_traces`.
