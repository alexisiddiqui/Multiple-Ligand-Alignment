"""
Sensitivity Example — Step 4: Filter PDB Ligands
================================================
Filters PR and AR ligands based on their alignment confidence against the
PDB Chemical Component Dictionary (CCD) dataset. Ligands with a global 
confidence score >= THRESHOLD are kept as "drug-like" PDB ligands.

Usage:
    python 04_filter_PDB_ligands.py

    # With a specific config (though base is fine):
    python 04_filter_PDB_ligands.py --config configs/base.yaml
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from ligand_neff.config import load_config
from ligand_neff.engine import NeffEngine
from ligand_neff.io.database import precompute_database

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
COMMON_DIR = BASE_DIR.parent / "LOSO" / "common"

RECEPTORS = ["PR", "AR"]
CONFIDENCE_THRESHOLD = 0.5


def load_mols_from_smi(smi_path: Path) -> list:
    """Helper to load molecules from SMILES maintaining original IDs."""
    from rdkit import Chem

    mols = []
    if not smi_path.exists():
        return mols

    with open(smi_path, "r") as f:
        for idx, line in enumerate(f):
            parts = line.strip().split()
            if not parts:
                continue
            smi = parts[0]
            mol_id = parts[1] if len(parts) > 1 else f"mol_{idx}"

            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                mol.SetProp("_Name", mol_id)
                mol.SetProp("SMILES", smi)
                mols.append(mol)

    return mols


def save_filtered_mols(
    original_mols: list, filtered_indices: set[int], out_path: Path
) -> None:
    """Save the filtered subset of molecules to a new .smi file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for idx in sorted(filtered_indices):
            mol = original_mols[idx]
            smi = mol.GetProp("SMILES")
            mol_id = mol.GetProp("_Name")
            f.write(f"{smi} {mol_id}\n")


def plot_distributions(results_df: pd.DataFrame, out_path: Path) -> None:
    """Plot the Neff global confidence distributions for both receptors."""
    plt.figure(figsize=(8, 5))
    
    sns.kdeplot(
        data=results_df,
        x="global_confidence",
        hue="receptor",
        common_norm=False,
        fill=True,
        alpha=0.5,
        palette=["#2196F3", "#4CAF50"]  # Blue for PR, Green for AR
    )

    plt.axvline(
        x=CONFIDENCE_THRESHOLD,
        color="red",
        linestyle="--",
        label=f"Cutoff = {CONFIDENCE_THRESHOLD}",
    )
    
    plt.title("Distribution of Neff Global Confidence vs PDB CCD")
    plt.xlabel("Global Confidence")
    plt.ylabel("Density")
    plt.xlim(0, 1.0)
    plt.legend(title="Receptor / Threshold")
    
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    
    print(f"  Distribution plot saved to {out_path.relative_to(BASE_DIR)}")


def run_filter(config_path: Path) -> None:
    """Evaluate and filter ligands against CCD."""
    config = load_config(config_path)

    ccd_sdf_path = COMMON_DIR / "components-pub.sdf"
    ccd_db_path = COMMON_DIR / "components-pub.npz"

    # Precompute CCD db if it doesn't exist
    if not ccd_db_path.exists():
        if not ccd_sdf_path.exists():
            print(f"Error: CCD components file not found at {ccd_sdf_path}")
            print("Please run examples/LOSO/common/download_ccd.py first.")
            sys.exit(1)
            
        print(f"  Precomputing CCD database from {ccd_sdf_path}...")
        precompute_database(ccd_sdf_path, ccd_db_path, config)

    print("  Loading CCD database...")
    precomputed_ccd = dict(np.load(ccd_db_path))

    print("  Initializing NeffEngine for CCD database...")
    engine = NeffEngine(config, precomputed_db=precomputed_ccd, max_atoms=100)

    all_results = []

    for receptor in RECEPTORS:
        print(f"\n── {receptor} ──")
        smi_path = DATA_DIR / f"{receptor.lower()}_ligands.smi"
        if not smi_path.exists():
            print(f"  Ligand file not found: {smi_path}")
            print("  Skipping. Run 01_fetch_receptor_ligands.py first.")
            continue

        print(f"  Loading query ligands from {smi_path.name}...")
        mols = load_mols_from_smi(smi_path)
        n_mols = len(mols)

        print(f"  Precomputing local variables for {n_mols} ligands...")
        prepared_queries = [engine.prepare_query(mol) for mol in tqdm(mols, desc="Precomp")]

        print("  Computing Neff against PDB CCD...")
        receptor_results = []
        filtered_indices = set()

        for i in tqdm(range(n_mols), desc="Compute"):
            query_mol = mols[i]
            prepared = prepared_queries[i]
            res = engine.compute_prepared(prepared)
            
            receptor_results.append(
                {
                    "receptor": receptor,
                    "idx": i,
                    "id": query_mol.GetProp("_Name"),
                    "global_confidence": res.global_confidence,
                    "global_neff": res.global_neff,
                }
            )

            if res.global_confidence >= CONFIDENCE_THRESHOLD:
                filtered_indices.add(i)

        # Save filtered ligands
        out_smi_path = DATA_DIR / f"{receptor.lower()}_ligands_pdb.smi"
        save_filtered_mols(mols, filtered_indices, out_smi_path)
        
        n_filtered = len(filtered_indices)
        pct = (n_filtered / n_mols) * 100 if n_mols > 0 else 0
        print(f"  Retained {n_filtered}/{n_mols} ligands ({pct:.1f}%) ≥ threshold.")
        print(f"  Filtered file saved to {out_smi_path.relative_to(BASE_DIR)}")
        
        all_results.extend(receptor_results)

    if all_results:
        df = pd.DataFrame(all_results)
        
        # Save raw scores for inspection
        csv_path = DATA_DIR / "pdb_filter_scores.csv"
        df.to_csv(csv_path, index=False)
        
        print("\n── Plotting Distributions ──")
        plot_path = DATA_DIR / "pdb_ligands_filter_dist.png"
        plot_distributions(df, plot_path)

    print("\nDone.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter PR & AR ligands based on Neff global confidence against the PDB CCD dataset."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=BASE_DIR / "configs" / "base.yaml",
        help="Path to YAML config (default: configs/base.yaml).",
    )
    args = parser.parse_args()

    print(f"Running filtering with config: {args.config.name}")
    print(f"Global Confidence Threshold: {CONFIDENCE_THRESHOLD}")
    
    run_filter(args.config)


if __name__ == "__main__":
    main()