import pytest
import os
from rdkit import Chem
from ligand_neff.io.database import load_database

@pytest.fixture
def temp_database_files(tmp_path):
    """Create temporary SDF and SMILES files."""
    sdf_path = str(tmp_path / "test_db.sdf")
    smi_path = str(tmp_path / "test_db.smi")
    
    # Create valid molecules
    mol1 = Chem.MolFromSmiles("CCO")  # Ethanol
    mol2 = Chem.MolFromSmiles("c1ccccc1")  # Benzene
    
    # Write SDF
    writer = Chem.SDWriter(sdf_path)
    writer.write(mol1)
    writer.write(mol2)
    writer.close()
    
    # Write SMILES
    with open(smi_path, "w") as f:
        f.write("CCO\n")
        f.write("c1ccccc1\n")
        
    return sdf_path, smi_path
        
def test_load_database_sdf(temp_database_files):
    sdf_path, _ = temp_database_files
    mols = load_database(sdf_path)
    assert len(mols) == 2
    assert isinstance(mols[0], Chem.Mol)

def test_load_database_smi(temp_database_files):
    _, smi_path = temp_database_files
    mols = load_database(smi_path)
    assert len(mols) == 2
    assert isinstance(mols[0], Chem.Mol)

def test_load_database_not_found():
    with pytest.raises(FileNotFoundError):
        load_database("does_not_exist.sdf")

def test_load_database_invalid_format(tmp_path):
    invalid_path = tmp_path / "test.txt"
    with open(invalid_path, "w") as f:
        f.write("test")
        
    with pytest.raises(ValueError, match="Unsupported database format"):
        load_database(str(invalid_path))
