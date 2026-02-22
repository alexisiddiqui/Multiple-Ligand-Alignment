import os
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import rdmolfiles


def load_database(path: str | Path) -> list[Chem.Mol]:
    """
    Load reference database from SDF or SMILES (.smi) file.
    Only basic loading for Phase 1. Precomputations and npz caching are Phase 3.
    """
    path_str = str(path)
    
    if not os.path.exists(path_str):
        raise FileNotFoundError(f"Database not found at {path_str}")
        
    mols = []
    
    if path_str.endswith('.sdf'):
        suppl = Chem.SDMolSupplier(path_str)
        for i, mol in enumerate(suppl):
            if mol is not None:
                mols.append(mol)
            else:
                pass # Skipping invalid molecules
                
    elif path_str.endswith('.smi'):
        suppl = Chem.SmilesMolSupplier(path_str, titleLine=False)
        for i, mol in enumerate(suppl):
            if mol is not None:
                mols.append(mol)
            else:
                pass # Skipping invalid molecules
    else:
        raise ValueError(f"Unsupported database format. Must be .sdf or .smi, got: {path_str}")
        
    if not mols:
        raise ValueError(f"No valid molecules loaded from {path_str}")
        
    return mols
