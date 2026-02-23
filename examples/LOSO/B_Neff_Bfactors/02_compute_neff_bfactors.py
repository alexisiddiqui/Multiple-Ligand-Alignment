import pickle
import pandas as pd
from pathlib import Path
from scipy import stats
from tqdm import tqdm
from ligand_neff.config import load_config
from ligand_neff.io.database import precompute_database
from ligand_neff.compute import compute_neff
from examples.LOSO.common.utils import create_loso_db
import numpy as np

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
COMMON_DIR = BASE_DIR.parent / "common"

def run_neff_bfactors():
    """Load PDBs, compute per-atom Neff, and correlate with B-factors."""
    print("Running Neff B-factor correlations...")
    
    pkl_file = DATA_DIR / "pdb_ligands.pkl"
    if not pkl_file.exists():
        print("PDB ligands not found. Run 01_fetch_pdb_ligands.py first.")
        return
        
    with open(pkl_file, "rb") as f:
        pdb_data = pickle.load(f)
        
    print(f"Loaded {len(pdb_data)} PDB ligands.")
    
    # Extract only the RDKit molecules for DB precomputation
    mols = [d["mol"] for d in pdb_data]
    
    # Save a temporary SMI file to satisfy precompute_database
    smi_path = DATA_DIR / "temp_pdb_ligands.smi"
    with open(smi_path, "w") as f:
        for i, d in enumerate(pdb_data):
            f.write(f"{d['smiles']} mol_{i}\n")
            
    # Precompute DB for these specific ligands
    db_path = DATA_DIR / "pdb_ligands_db.npz"
    config = load_config(COMMON_DIR / "cdk2_config.yaml")
    
    print("Precomputing fingerprint database for PDBs...")
    precompute_database(smi_path, db_path, config)
    precomputed = dict(np.load(db_path))
    smi_path.unlink() # cleanup
    
    print("Initializing NeffEngine for PDBs...")
    from ligand_neff.engine import NeffEngine
    engine = NeffEngine(config, precomputed_db=precomputed)
    
    print("Precomputing query data...")
    all_prepared_queries = [engine.prepare_query(mol) for mol in tqdm(mols, desc="Precomputing")]
    
    results = []
    
    for i in tqdm(range(len(pdb_data)), desc="LOSO Analysis"):
        record = pdb_data[i]
        query = record["mol"]
        prepared = all_prepared_queries[i]
        b_factors = record["b_factors"]
        
        # Compute Neff
        res = engine.compute_prepared(prepared)
        
        # Extract per-atom Neff (from the returned fields)
        atom_neffs = res.atom_neff
        
        # Ensure lengths match
        if len(atom_neffs) != len(b_factors):
            print(f"Mismatch in atom count for {record['pdb_id']}:{record['ligand_id']}. Skipping.")
            continue
            
        # Correlate
        spearman_r, spearman_p = stats.spearmanr(atom_neffs, b_factors)
        
        results.append({
            "pdb_id": record["pdb_id"],
            "ligand_id": record["ligand_id"],
            "global_neff": res.global_neff,
            "global_confidence": res.global_confidence,
            "n_refs": res.n_references_used,
            "n_atoms": len(atom_neffs),
            "spearman_r": spearman_r,
            "spearman_p": spearman_p
        })
        
    df = pd.DataFrame(results)
    out_csv = DATA_DIR / "bfactor_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"Correlation results saved to {out_csv}")
    
    # Quick summary
    valid = df.dropna(subset=["spearman_r"])
    if not valid.empty:
        mean_r = valid["spearman_r"].mean()
        print(f"\nMean Spearman Correlation: {mean_r:.3f}")
        print("Note: Negative correlation means higher Neff/Representation correlates with lower B-factor/flexibility.")

if __name__ == "__main__":
    run_neff_bfactors()
