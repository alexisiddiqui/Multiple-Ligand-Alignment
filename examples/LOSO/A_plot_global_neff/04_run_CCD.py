import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from ligand_neff.compute import compute_neff, prepare_query_data
from ligand_neff.config import load_config
from ligand_neff.io.database import precompute_database
from examples.LOSO.common.utils import load_mols_from_smi

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
COMMON_DIR = BASE_DIR.parent / "common"

def run_ccd():
    smi_path = DATA_DIR / "cdk2_ligands.smi"
    config_path = COMMON_DIR / "cdk2_config.yaml"
    out_path = DATA_DIR / "cdk2_ccd_results.csv"
    
    ccd_sdf_path = COMMON_DIR / "components-pub.sdf"
    ccd_db_path = COMMON_DIR / "components-pub.npz"

    if not smi_path.exists():
        print("Data files not found. Run 01_fetch_and_precompute.py first.")
        return
        
    config = load_config(config_path)

    if not ccd_db_path.exists():
        print(f"Precomputing CCD database from {ccd_sdf_path}...")
        precompute_database(ccd_sdf_path, ccd_db_path, config)
    
    print("Loading query ligands and CCD database...")
    mols = load_mols_from_smi(smi_path)
    precomputed_ccd = dict(np.load(ccd_db_path))
    
    print("Initializing NeffEngine for CCD database...")
    from ligand_neff.engine import NeffEngine
    engine = NeffEngine(config, precomputed_db=precomputed_ccd)
    
    n_mols = len(mols)
    processed_results = []
    
    print(f"Precomputing query data for {n_mols} molecules...")
    all_prepared_queries = [engine.prepare_query(mol) for mol in tqdm(mols, desc="Precomputing")]
    
    print(f"Starting Neff computation against CCD db for {n_mols} molecules...")
    
    for i in tqdm(range(n_mols), desc="CCD Progress"):
        query = mols[i]
        prepared = all_prepared_queries[i]
        
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
    run_ccd()