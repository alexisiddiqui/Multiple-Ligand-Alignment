import urllib.request
import json
import time
import csv
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
    """Fetch CDK2 ligands from ChEMBL REST API, including pChEMBL activity values."""
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
            pchembl = act.get("pchembl_value")
            if smiles and cid:
                records.append({
                    "smiles": smiles,
                    "id": cid,
                    "pchembl_value": float(pchembl) if pchembl is not None else None,
                })
        
        url = data["page_meta"]["next"]
        if url:
            if not url.startswith("http"):
                url = "https://www.ebi.ac.uk" + url
            time.sleep(0.5)
        
    return records

def smiles_to_smi_file(records: list[dict], out_path: Path) -> tuple[int, dict[str, float | None]]:
    """Validate and deduplicate, write to .smi file.

    Returns (n_valid, activity_map) where activity_map maps canonical_smiles id
    to the *maximum* pChEMBL value seen across all duplicate activity entries.
    """
    # smiles → {"id": str, "pchembl": float | None}
    unique_mols: dict[str, dict] = {}
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for rec in records:
        smiles = rec["smiles"]
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        can_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        pchembl = rec["pchembl_value"]

        if can_smiles not in unique_mols:
            unique_mols[can_smiles] = {"id": rec["id"], "pchembl": pchembl}
        else:
            # Keep the maximum pChEMBL value (most potent measurement)
            existing = unique_mols[can_smiles]["pchembl"]
            if pchembl is not None:
                if existing is None or pchembl > existing:
                    unique_mols[can_smiles]["pchembl"] = pchembl

    valid_count = len(unique_mols)
    print(f"Deduplicated to {valid_count} unique molecules.")

    with open(out_path, "w") as f:
        for smiles, info in unique_mols.items():
            f.write(f"{smiles} {info['id']}\n")

    # Build id → pchembl_value map (keyed by ChEMBL molecule ID)
    activity_map = {info["id"]: info["pchembl"] for info in unique_mols.values()}
    return valid_count, activity_map

def save_activity_csv(activity_map: dict[str, float | None], out_path: Path) -> None:
    """Save a CSV mapping molecule ID → pChEMBL activity value."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "pchembl_value"])
        writer.writeheader()
        for mol_id, pchembl in activity_map.items():
            writer.writerow({"id": mol_id, "pchembl_value": pchembl})
    print(f"Activity data saved to {out_path}")

def run_fetch_and_precompute():
    records = fetch_chembl_cdk2()

    smi_path = DATA_DIR / "cdk2_ligands.smi"
    activity_path = DATA_DIR / "cdk2_activity.csv"

    n_valid, activity_map = smiles_to_smi_file(records, smi_path)
    save_activity_csv(activity_map, activity_path)
    
    print(f"Precomputing database for {n_valid} molecules...")
    db_path = DATA_DIR / "cdk2_db.npz"
    config = load_config(COMMON_DIR / "cdk2_config.yaml")
    precompute_database(smi_path, db_path, config)
    print(f"Done! Data saved to {DATA_DIR}")

if __name__ == "__main__":
    run_fetch_and_precompute()
