"""
Sensitivity Example — Step 5: Plot Filter Distributions
========================================================
Standalone plotting script that reads pre-computed filter scores and regenerates
the 3-panel distribution plot without re-running the expensive filtering step.

Usage:
    python 05_plot_filter.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

CONFIDENCE_THRESHOLD = 0.5
TANIMOTO_THRESHOLD = 0.9
palette = ["#2196F3", "#4CAF50"]  # Blue for PR, Green for AR


def plot_distributions(results_df: pd.DataFrame, out_path: Path) -> None:
    """Plot the Neff global confidence, global Neff, and max Tanimoto distributions for both receptors."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 5))

    # Plot 1: Global Confidence
    sns.kdeplot(
        data=results_df,
        x="global_confidence",
        hue="receptor",
        common_norm=False,
        fill=True,
        alpha=0.5,
        palette=palette,
        ax=ax1,
    )

    ax1.axvline(
        x=CONFIDENCE_THRESHOLD,
        color="red",
        linestyle="--",
        label=f"Cutoff = {CONFIDENCE_THRESHOLD}",
    )

    ax1.set_title("Distribution of Neff Global Confidence vs PDB CCD")
    ax1.set_xlabel("Global Confidence")
    ax1.set_ylabel("Density")
    ax1.set_xlim(0, 1.0)
    ax1.legend(title="Receptor / Threshold")

    # Plot 2: Global Neff (log scale)
    for receptor in results_df["receptor"].unique():
        receptor_data = results_df[results_df["receptor"] == receptor]
        color = "#2196F3" if receptor == "PR" else "#4CAF50"
        ax2.hist(
            receptor_data["global_neff"],
            bins=50,
            alpha=0.6,
            label=receptor,
            color=color,
            edgecolor="black",
            linewidth=0.5,
        )

    ax2.set_yscale("log")
    ax2.set_title("Distribution of Neff Global Neff vs PDB CCD")
    ax2.set_xlabel("Global Neff")
    ax2.set_ylabel("Count (log scale)")
    ax2.legend(title="Receptor")
    ax2.grid(True, alpha=0.3, which="both")

    # Plot 3: Max Tanimoto similarity to CCD
    if "max_tanimoto" in results_df.columns:
        sns.kdeplot(
            data=results_df,
            x="max_tanimoto",
            hue="receptor",
            common_norm=False,
            fill=True,
            alpha=0.5,
            palette=palette,
            ax=ax3,
        )
        ax3.axvline(
            x=TANIMOTO_THRESHOLD,
            color="red",
            linestyle="--",
            label=f"Cutoff = {TANIMOTO_THRESHOLD}",
        )
        ax3.set_title("Distribution of Max Tanimoto Similarity vs PDB CCD")
        ax3.set_xlabel("Max Tanimoto Similarity")
        ax3.set_ylabel("Density")
        ax3.set_xlim(0, 1.0)
        ax3.legend(title="Receptor / Threshold")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Distribution plot saved to {out_path.relative_to(BASE_DIR)}")


def main() -> None:
    csv_path = DATA_DIR / "pdb_filter_scores.csv"

    if not csv_path.exists():
        print(f"Error: Filter scores CSV not found at {csv_path}")
        print("Please run 04_filter_PDB_ligands.py first.")
        return

    print(f"Loading filter scores from {csv_path.name}...")
    df = pd.read_csv(csv_path)

    print("Generating distribution plot...")
    plot_path = DATA_DIR / "pdb_ligands_filter_dist.png"
    plot_distributions(df, plot_path)

    print("Done.")


if __name__ == "__main__":
    main()
