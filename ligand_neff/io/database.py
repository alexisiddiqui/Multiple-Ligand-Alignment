import os
from pathlib import Path
from rdkit import Chem
import numpy as np

from ligand_neff.config import NeffConfig
from ligand_neff.fingerprints.encode import encode_molecule

def precompute_database(path: str | Path, out_path: str | Path, config: NeffConfig) -> None:
    """
    Precompute fingerprints for all references in the database to speed up Neff pipeline.
    Saves an .npz file containing the database molecules.
    """
    mols = load_database(path) # load from raw sdf/smi
    
    # We save a dictionary mapping radius -> array of fingerprints (N_db, fp_size)
    data_to_save = {}
    for r in config.fp_radii:
        fps = np.zeros((len(mols), config.fp_size), dtype=np.uint8)
        for i, mol in enumerate(mols):
            fps[i] = encode_molecule(mol, radius=r, fp_size=config.fp_size,
                                     use_chirality=config.use_chirality,
                                     use_features=config.use_features)
        data_to_save[f"radius_{r}"] = fps
        
    np.savez_compressed(out_path, **data_to_save)
    

def load_database(path: str | Path) -> list[Chem.Mol]:
    """
    Load reference database from SDF or SMILES (.smi) file.
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
                
    elif path_str.endswith('.smi'):
        suppl = Chem.SmilesMolSupplier(path_str, titleLine=False)
        for i, mol in enumerate(suppl):
            if mol is not None:
                mols.append(mol)
    else:
        raise ValueError(f"Unsupported database format. Must be .sdf or .smi, got: {path_str}")
        
    if not mols:
        raise ValueError(f"No valid molecules loaded from {path_str}")
        
    return mols
