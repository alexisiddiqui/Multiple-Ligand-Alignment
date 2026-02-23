"""
Sensitivity Example — Step 1: Fetch Receptor Ligands
=====================================================
Downloads active ligands for the Progesterone Receptor (PR) and Androgen Receptor (AR)
from ChEMBL, deduplicates by canonical SMILES, and saves .smi files.

Usage:
    python 01_fetch_receptor_ligands.py
"""

import json
import time
import urllib.request
from pathlib import Path

from rdkit import Chem

# ── Target Definitions ────────────────────────────────────────────────────────
# PR = Progesterone Receptor (nuclear receptor for progesterone / progestins)
# AR = Androgen Receptor    (nuclear receptor for testosterone / androgens)
TARGETS = {
    "PR": "CHEMBL208",
    "AR": "CHEMBL1871",
}

BASE_URL = "https://www.ebi.ac.uk/chembl/api/data/activity"
MIN_PCHEMBL = 6.0    # IC50 ≤ 1 µM
MAX_LIGANDS = 1000    # Cap per receptor to keep runtimes manageable

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"


# ── ChEMBL Fetching ───────────────────────────────────────────────────────────

def fetch_chembl_activities(target_id: str, min_pchembl: float = MIN_PCHEMBL) -> list[dict]:
    """Fetch activity records for a ChEMBL target."""
    print(f"  Fetching activities for {target_id} (pChEMBL ≥ {min_pchembl})...")
    params = (
        f"target_chembl_id={target_id}"
        f"&pchembl_value__gte={min_pchembl}"
        f"&assay_type=B"           # binding assays only
        f"&format=json&limit=1000"
    )
    records = []
    url = f"{BASE_URL}?{params}"

    while url:
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                data = json.loads(response.read().decode())
        except Exception as e:
            print(f"    HTTP error: {e}. Stopping pagination.")
            break

        for act in data.get("activities", []):
            smiles = act.get("canonical_smiles")
            cid = act.get("molecule_chembl_id")
            if smiles and cid:
                records.append({"smiles": smiles, "id": cid})

        next_url = data.get("page_meta", {}).get("next")
        if next_url:
            url = next_url if next_url.startswith("http") else "https://www.ebi.ac.uk" + next_url
            time.sleep(0.5)
        else:
            url = None

    print(f"    Retrieved {len(records)} raw activity records.")
    return records


def deduplicate_to_smi(records: list[dict], out_path: Path, cap: int = MAX_LIGANDS) -> int:
    """
    Deduplicate records by canonical SMILES (via RDKit), write a .smi file.
    Returns the number of unique, valid molecules saved.
    """
    unique: dict[str, str] = {}  # canonical_smiles → chembl_id

    for rec in records:
        if len(unique) >= cap:
            break
        mol = Chem.MolFromSmiles(rec["smiles"])
        if mol is None:
            continue
        can_smi = Chem.MolToSmiles(mol, isomericSmiles=True)
        if can_smi not in unique:
            unique[can_smi] = rec["id"]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for smi, cid in unique.items():
            f.write(f"{smi} {cid}\n")

    print(f"    Deduplicated → {len(unique)} unique molecules saved to {out_path.name}")
    return len(unique)


# ── Main ──────────────────────────────────────────────────────────────────────

def fetch_receptor_ligands() -> None:
    """Download PR and AR ligands from ChEMBL and save as .smi files."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}

    for receptor, target_id in TARGETS.items():
        print(f"\n── {receptor} ({target_id}) ──")
        records = fetch_chembl_activities(target_id)
        out_path = DATA_DIR / f"{receptor.lower()}_ligands.smi"
        n = deduplicate_to_smi(records, out_path)
        counts[receptor] = n

    print("\n── Summary ──")
    for receptor, n in counts.items():
        print(f"  {receptor}: {n} ligands → data/{receptor.lower()}_ligands.smi")
    print("\nDone. Run 02_compute_cross_neff.py next.")


if __name__ == "__main__":
    fetch_receptor_ligands()
