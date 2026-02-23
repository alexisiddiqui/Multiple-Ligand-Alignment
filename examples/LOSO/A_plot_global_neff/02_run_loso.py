import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from ligand_neff.compute import compute_neff
from ligand_neff.config import load_config
from examples.LOSO.common.utils import load_mols_from_smi, create_loso_db

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
COMMON_DIR = BASE_DIR.parent / "common"

def run_loso():
    smi_path = DATA_DIR / "cdk2_ligands.smi"
    db_path = DATA_DIR / "cdk2_db.npz"
    config_path = COMMON_DIR / "cdk2_config.yaml"
    out_path = DATA_DIR / "cdk2_results.csv"
    
    if not smi_path.exists() or not db_path.exists():
        print("Data files not found. Run 01_fetch_and_precompute.py first.")
        return

    print("Loading data...")
    mols = load_mols_from_smi(smi_path)
    precomputed = dict(np.load(db_path))
    config = load_config(config_path)
    
    print("Initializing NeffEngine for LOSO...")
    from ligand_neff.engine import NeffEngine
    from ligand_neff.io.db_cache import build_db_cache
    engine = NeffEngine(config, precomputed_db=precomputed)
    
    n_mols = len(mols)
    processed_results = []
    
    print(f"Precomputing query data for {n_mols} molecules...")
    all_prepared_queries = [engine.prepare_query(mol) for mol in tqdm(mols, desc="Precomputing")]
    
    print(f"Starting LOSO for {n_mols} molecules...")
    
    for i in tqdm(range(n_mols), desc="LOSO Progress"):
        query = mols[i]
        prepared = all_prepared_queries[i]
        
        # Exclude the query from the precomputed db
        loso_db_dict = create_loso_db(precomputed, config.fp_radii, i)
        loso_cache = build_db_cache(loso_db_dict, config.fp_radii)
        
        # We can dynamically swap the DB cache in the engine (cache array shape is static)
        engine.db_cache = loso_cache
        res = engine.compute_prepared(prepared)
        
        processed_results.append({
            "idx": i,
            "id": query.GetProp("_Name") if query.HasProp("_Name") else f"mol_{i}",
            "global_neff": res.global_neff,
            "global_confidence": res.global_confidence,
            "n_refs": res.n_references_used
        })

    df = pd.DataFrame(processed_results)
    df.to_csv(out_path, index=False)
    print(f"Results saved to {out_path}")

if __name__ == "__main__":
    run_loso()
