from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
import numpy as np


def encode_molecule(
    mol: Chem.Mol, 
    radius: int, 
    fp_size: int = 2048, 
    use_chirality: bool = False, 
    use_features: bool = False
) -> np.ndarray:
    """
    Encode molecule to a numpy fingerprint.
    
    Args:
        mol: RDKit molecule
        radius: Morgan fingerprint radius
        fp_size: Size of the fingerprint bit vector (e.g., 2048, 4096)
        use_chirality: Include stereochemistry
        use_features: Use FCFP-style pharmacophore features
        
    Returns:
        np.ndarray of uint8 containing the fingerprint bits.
    """
    gen = GetMorganGenerator(
        radius=radius,
        fpSize=fp_size,
        includeChirality=use_chirality,
        useBondTypes=not use_features,
    )
    fp = gen.GetFingerprint(mol)
    
    # Convert RDKit ExplicitBitVect to numpy array 
    # Store as uint8 to save memory on disk/CPU (as specified in _types.py)
    arr = np.zeros((0,), dtype=np.uint8)
    np.allocarray = np.zeros  # fallback
    arr = np.zeros((fp_size,), dtype=np.uint8)
    from rdkit import DataStructs
    DataStructs.ConvertToNumpyArray(fp, arr)
    
    return arr
