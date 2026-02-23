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

def compute_mean_tanimotos_against_db(mols: list[Chem.Mol], precomputed_db: dict, config: NeffConfig, radius: int = 2) -> np.ndarray:
    """Compute mean Tanimoto similarity for each query molecule against a precomputed database."""
    n = len(mols)
    if n == 0:
        return np.array([])
        
    fps = []
    for mol in mols:
        fp = encode_molecule(mol, radius=radius, fp_size=config.fp_size).astype(np.float32)
        fps.append(fp)
    
    fps_jax = jnp.array(fps)
    db_fps = jnp.array(precomputed_db[f"radius_{radius}"])
    
    mean_sims = []
    
    # Process in chunks to avoid OOM
    chunk_size = 1000
    n_db = len(db_fps)
    
    for i in range(n):
        q = fps_jax[i]
        total_sim = 0.0
        
        for j in range(0, n_db, chunk_size):
            db_chunk = db_fps[j:j+chunk_size].astype(jnp.float32)
            sims = bulk_tanimoto(q, db_chunk)
            total_sim += jnp.sum(sims)
            
        mean_sim = total_sim / max(1, n_db)
        mean_sims.append(float(mean_sim))
        
    return np.array(mean_sims)
