import jax.numpy as jnp
import numpy as np
from rdkit import Chem
from ligand_neff.config import NeffConfig
from ligand_neff.fingerprints.encode import encode_molecule
from ligand_neff.similarity.tanimoto import bulk_tanimoto

def load_mols_from_smi(smi_path) -> list[Chem.Mol]:
    """Load molecules from a SMILES file."""
    mols = []
    with open(smi_path, "r") as f:
        for line in f:
            smi = line.strip().split()[0]
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                mols.append(mol)
    return mols

def create_loso_db(precomputed_db: dict, fp_radii: tuple[int, ...], exclude_idx: int) -> dict:
    """
    Simulate leave-one-out by creating a local 'precomputed' dict 
    where the exclude_idx row is zeroed out for all radii.
    """
    loso_db = {}
    for r in fp_radii:
        arr = precomputed_db[f"radius_{r}"].copy()
        arr[exclude_idx, :] = 0
        loso_db[f"radius_{r}"] = arr
    return loso_db

def compute_mean_tanimotos(mols: list[Chem.Mol], config: NeffConfig) -> np.ndarray:
    """Compute mean Tanimoto similarity for each molecule against all others."""
    n = len(mols)
    if n == 0:
        return np.array([])
        
    # We'll use radius 2 for the correlation check
    fps = []
    for mol in mols:
        fp = encode_molecule(mol, radius=2, fp_size=config.fp_size).astype(np.float32)
        fps.append(fp)
    
    fps_jax = jnp.array(fps)
    mean_sims = []
    
    for i in range(n):
        q = fps_jax[i]
        # bulk_tanimoto handles 1 vs N
        sims = bulk_tanimoto(q, fps_jax)
        # Exclude self-similarity (1.0)
        mean_sim = (jnp.sum(sims) - 1.0) / max(1, n - 1)
        mean_sims.append(float(mean_sim))
        
    return np.array(mean_sims)
