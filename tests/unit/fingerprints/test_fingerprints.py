import pytest
from rdkit import Chem
import numpy as np
import jax.numpy as jnp
from ligand_neff.fingerprints.decompose import decompose
from ligand_neff.fingerprints.encode import encode_molecule

@pytest.fixture
def sample_mol():
    """A simple molecule (Aspirin) for testing."""
    return Chem.MolFromSmiles("CC(=O)OC1=CC=CC=C1C(=O)O")

def test_encode_molecule(sample_mol):
    """Test encoding a molecule returns correct numpy array type and shape."""
    fp_size = 2048
    radius = 2
    
    fp = encode_molecule(sample_mol, radius, fp_size)
    
    assert isinstance(fp, np.ndarray)
    assert fp.dtype == np.uint8
    assert fp.shape == (fp_size,)
    # Aspirin has some features, so bitvector shouldn't be all zero
    assert np.sum(fp) > 0

def test_decompose(sample_mol):
    """Test decomposing a molecule."""
    fp_size = 2048
    radius = 2
    
    decomp = decompose(sample_mol, radius, fp_size)
    
    assert decomp.mol is sample_mol
    assert decomp.radius == radius
    assert decomp.fp_size == fp_size
    assert isinstance(decomp.info, dict)
    assert len(decomp.info) > 0

def test_build_atom_bit_mask(sample_mol):
    """Test building the atom bit mask float array."""
    fp_size = 2048
    radius = 2
    n_atoms = sample_mol.GetNumAtoms()
    
    decomp = decompose(sample_mol, radius, fp_size)
    mask = decomp.build_atom_bit_mask()
    
    assert isinstance(mask, jnp.ndarray)
    assert mask.dtype == jnp.float32
    assert mask.shape == (n_atoms, fp_size)
    
    # Check that the mask sets bits found in info dictionary
    for bit, envs in decomp.info.items():
        for env in envs:
            center_atom_idx = env[0]
            assert mask[center_atom_idx, bit] == 1.0

def test_fp_size_variations(sample_mol):
    """Test different valid and invalid sizes down the line."""
    for size in [2048, 4096]:
        fp = encode_molecule(sample_mol, radius=2, fp_size=size)
        assert fp.shape == (size,)
        
        decomp = decompose(sample_mol, radius=2, fp_size=size)
        mask = decomp.build_atom_bit_mask()
        assert mask.shape == (sample_mol.GetNumAtoms(), size)
