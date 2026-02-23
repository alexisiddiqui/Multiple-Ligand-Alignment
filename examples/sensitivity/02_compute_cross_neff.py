"""
Sensitivity Example — Step 2: Cross-Receptor Neff Computation
=============================================================
Precomputes fingerprint databases for PR and AR, then computes global and
per-atom Neff for every ligand queried against BOTH receptor reference sets.
Runs across all 9 config variants (fp_size × aggregation grid).

Usage:
    # Run all 9 configs:
    python 02_compute_cross_neff.py

    # Run a single config:
    python 02_compute_cross_neff.py --config configs/base.yaml
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from ligand_neff.compute import compute_neff
from ligand_neff.config import NeffConfig, load_config
from ligand_neff.io.database import precompute_database, load_database

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CONFIGS_DIR = BASE_DIR / "configs"

# These match the .smi files produced by script 01
RECEPTORS = ["PR", "AR"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_all_configs() -> list[Path]:
    """Return all YAML configs sorted by name."""
    return sorted(CONFIGS_DIR.glob("*.yaml"))


def config_name(config_path: Path) -> str:
    return config_path.stem  # e.g. "base", "fp4096_geometric"


def ensure_db(smi_path: Path, db_path: Path, config: NeffConfig) -> None:
    """Build .npz fingerprint DB if it doesn't exist yet."""
    if db_path.exists():
        return
    print(f"    Precomputing DB: {db_path.name} ...")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    precompute_database(smi_path, db_path, config)


def run_config(config_path: Path) -> None:
    """
    For one config variant:
      1. Precompute fingerprint DBs for PR and AR.
      2. Compute Neff for every PR ligand vs PR-DB and vs AR-DB.
      3. Compute Neff for every AR ligand vs PR-DB and vs AR-DB.
      4. Save per-molecule results to CSV.
    """
    cfg_name = config_name(config_path)
    config = load_config(config_path)
    out_dir = DATA_DIR / cfg_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "cross_neff_results.csv"

    if out_csv.exists():
        print(f"  [{cfg_name}] Results already exist — skipping. Delete to rerun.")
        return

    print(f"\n{'='*60}")
    print(f"  Config: {cfg_name}  (fp_size={config.fp_size}, agg={config.aggregation})")
    print(f"{'='*60}")

    # ── 1. Load mols + precompute DBs ────────────────────────────────────────
    receptor_mols: dict[str, list] = {}
    precomputed_dbs: dict[str, dict] = {}

    for receptor in RECEPTORS:
        smi_path = DATA_DIR / f"{receptor.lower()}_ligands.smi"
        if not smi_path.exists():
            raise FileNotFoundError(
                f"Ligand file not found: {smi_path}\n"
                "Did you run 01_fetch_receptor_ligands.py first?"
            )
        db_path = out_dir / f"{receptor.lower()}_db.npz"
        ensure_db(smi_path, db_path, config)

        mols = load_database(smi_path)
        receptor_mols[receptor] = mols
        precomputed_dbs[receptor] = dict(np.load(db_path))
        print(f"    {receptor}: {len(mols)} ligands loaded.")

    # ── 2. Cross-compute Neff ─────────────────────────────────────────────────
    results = []
    
    print("  Initializing NeffEngines...")
    from ligand_neff.engine import NeffEngine
    engines = {
        receptor: NeffEngine(config, precomputed_db=precomputed_dbs[receptor]) 
        for receptor in RECEPTORS
    }
    
    receptor_prepared_queries: dict[str, list] = {}
    for receptor in RECEPTORS:
        receptor_prepared_queries[receptor] = [
            engines[receptor].prepare_query(mol) 
            for mol in tqdm(receptor_mols[receptor], desc=f"  Precomputing {receptor}")
        ]

    for source_rec in RECEPTORS:
        mols = receptor_mols[source_rec]
        prepared_queries = receptor_prepared_queries[source_rec]
        desc = f"  {source_rec} ligands"

        for i, (query_mol, prepared) in enumerate(tqdm(zip(mols, prepared_queries), desc=desc, total=len(mols))):
            mol_id = (
                query_mol.GetProp("_Name")
                if query_mol.HasProp("_Name")
                else f"{source_rec}_mol_{i}"
            )

            for ref_rec in RECEPTORS:
                # Use the fast device engine associated with the reference DB
                res = engines[ref_rec].compute_prepared(prepared)

                results.append({
                    "config_name": cfg_name,
                    "fp_size": config.fp_size,
                    "aggregation": config.aggregation,
                    "source_receptor": source_rec,
                    "query_idx": i,
                    "query_id": mol_id,
                    "ref_receptor": ref_rec,
                    "global_neff": res.global_neff,
                    "global_confidence": res.global_confidence,
                    "n_refs_used": res.n_references_used,
                    "lambda_value": res.lambda_value,
                    # Store atom Neff as compact JSON string
                    "atom_neff_json": json.dumps([round(float(x), 4) for x in res.atom_neff]),
                })

    df = pd.DataFrame(results)
    df.to_csv(out_csv, index=False)
    print(f"  Saved {len(df)} rows → {out_csv.relative_to(BASE_DIR)}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute cross-receptor Neff sensitivity analysis."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a specific YAML config. Omit to run all 9 configs.",
    )
    args = parser.parse_args()

    configs = [args.config] if args.config else get_all_configs()
    print(f"Running {len(configs)} config(s)...")

    for cfg_path in configs:
        run_config(cfg_path)

    print("\nAll configs complete. Run 03_plot_sensitivity.py next.")


if __name__ == "__main__":
    main()
