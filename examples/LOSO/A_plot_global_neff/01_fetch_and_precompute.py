import urllib.request
import json
import time
from pathlib import Path
from rdkit import Chem
from ligand_neff.config import load_config
from ligand_neff.io.database import precompute_database

# CDK2 Target ChEMBL ID
CDK2_TARGET_ID = "CHEMBL301"
BASE_URL = "https://www.ebi.ac.uk/chembl/api/data/activity"

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
COMMON_DIR = BASE_DIR.parent / "common"

def fetch_chembl_cdk2(min_pchembl: float = 5.0) -> list[dict]:
    """Fetch CDK2 ligands from ChEMBL REST API."""
    print(f"Fetching CDK2 ligands from ChEMBL (min pChEMBL={min_pchembl})...")
    
    params = f"target_chembl_id={CDK2_TARGET_ID}&pchembl_value__gte={min_pchembl}&format=json&limit=1000"
    records = []
    url = f"{BASE_URL}?{params}"
    
    while url:
        print(f"  Fetching: {url}")
        try:
            with urllib.request.urlopen(url) as response:
                data = json.loads(response.read().decode())
        except Exception as e:
            print(f"Error fetching data: {e}")
            break
            
        for act in data["activities"]:
            smiles = act.get("canonical_smiles")
            cid = act.get("molecule_chembl_id")
            if smiles and cid:
                records.append({"smiles": smiles, "id": cid})
        
        url = data["page_meta"]["next"]
        if url:
            if not url.startswith("http"):
                url = "https://www.ebi.ac.uk" + url
            time.sleep(0.5)
        
    return records

def smiles_to_smi_file(records: list[dict], out_path: Path) -> int:
    """Validate and deduplicate, write to .smi file."""
    unique_mols = {} 
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    valid_count = 0
    for rec in records:
        smiles = rec["smiles"]
        if smiles in unique_mols:
            continue
            
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            can_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
            if can_smiles not in unique_mols:
                unique_mols[can_smiles] = rec["id"]
                valid_count += 1
                
    print(f"Deduplicated to {valid_count} unique molecules.")
    
    with open(out_path, "w") as f:
        for smiles, cid in unique_mols.items():
            f.write(f"{smiles} {cid}\n")
            
    return valid_count

def run_fetch_and_precompute():
    records = fetch_chembl_cdk2()
    
    smi_path = DATA_DIR / "cdk2_ligands.smi"
    n_valid = smiles_to_smi_file(records, smi_path)
    
    print(f"Precomputing database for {n_valid} molecules...")
    db_path = DATA_DIR / "cdk2_db.npz"
    config = load_config(COMMON_DIR / "cdk2_config.yaml")
    precompute_database(smi_path, db_path, config)
    print(f"Done! Data saved to {DATA_DIR}")

if __name__ == "__main__":
    run_fetch_and_precompute()
