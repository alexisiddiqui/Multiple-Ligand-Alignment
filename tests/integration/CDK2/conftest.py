import pytest
from pathlib import Path
import numpy as np
from rdkit import Chem
from ligand_neff.config import load_config
from ligand_neff.io.database import load_database

@pytest.fixture(scope="session")
def cdk2_dir():
    # Return the examples/LOSO directory base path
    return Path(__file__).parent.parent.parent.parent / "examples" / "LOSO"

@pytest.fixture(scope="session")
def cdk2_data_dir(cdk2_dir):
    data_dir = cdk2_dir / "A_plot_global_neff" / "data"
    if not data_dir.exists():
        pytest.skip(f"CDK2 data directory missing: {data_dir}")
    return data_dir

@pytest.fixture(scope="session")
def cdk2_mols(cdk2_data_dir):
    smi_path = cdk2_data_dir / "cdk2_ligands.smi"
    if not smi_path.exists():
        pytest.skip(f"CDK2 SMILES file missing: {smi_path}")
    return load_database(smi_path)

@pytest.fixture(scope="session")
def cdk2_precomputed(cdk2_data_dir):
    db_path = cdk2_data_dir / "cdk2_db.npz"
    if not db_path.exists():
        pytest.skip(f"CDK2 precomputed database missing: {db_path}")
    return dict(np.load(db_path))

@pytest.fixture(scope="session")
def cdk2_config(cdk2_dir):
    config_path = cdk2_dir / "common" / "cdk2_config.yaml"
    if not config_path.exists():
        pytest.skip(f"CDK2 config file missing: {config_path}")
    return load_config(config_path)
