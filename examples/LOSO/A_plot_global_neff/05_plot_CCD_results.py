import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
from ligand_neff.config import load_config
from examples.LOSO.common.utils import compute_mean_tanimotos_against_db, load_mols_from_smi
import numpy as np

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
COMMON_DIR = BASE_DIR.parent / "common"


def _add_corr_annotation(ax, x, y):
    """Annotate an axes with Pearson and Spearman correlations."""
    pearson_r, pearson_p = stats.pearsonr(x, y)
    spearman_r, spearman_p = stats.spearmanr(x, y)
    corr_text = (
        f"Pearson  r = {pearson_r:.3f}  (p = {pearson_p:.2e})\n"
        f"Spearman ρ = {spearman_r:.3f}  (p = {spearman_p:.2e})"
    )
    ax.annotate(
        corr_text,
        xy=(0.05, 0.95), xycoords="axes fraction",
        verticalalignment="top",
        fontsize=10, fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.8),
    )


def plot_ccd_results():
    results_path = DATA_DIR / "cdk2_ccd_results.csv"
    activity_path = DATA_DIR / "cdk2_activity.csv"
    smi_path = DATA_DIR / "cdk2_ligands.smi"
    config_path = COMMON_DIR / "cdk2_config.yaml"

    if not results_path.exists():
        print(f"Results CSV not found at {results_path}. Run 04_run_CCD.py first.")
        return

    df = pd.read_csv(results_path)

    # The utils load_mols_from_smi strips the ID, resulting in "mol_0", "mol_1" in results.
    # We parse the IDs manually and inject them back into the dataframe to allow joining with activity.
    actual_ids = []
    with open(smi_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) > 1:
                actual_ids.append(parts[1])
            else:
                actual_ids.append(f"mol_{len(actual_ids)}")
    if len(actual_ids) == len(df):
        df["id"] = actual_ids


    # ── Tanimoto similarity ──────────────────────────────────────────────────
    if "mean_tanimoto" in df.columns:
        df = df.drop(columns=["mean_tanimoto"])
        
    if "mean_tanimoto" not in df.columns:
        print("Computing mean Tanimoto similarities (against CCD data)...")
        config = load_config(config_path)
        mols = load_mols_from_smi(smi_path)
        
        ccd_db_path = COMMON_DIR / "components-pub.npz"
        precomputed_ccd = dict(np.load(ccd_db_path))

        if "idx" in df.columns:
            subset_mols = [mols[i] for i in df["idx"]]
        else:
            subset_mols = mols[: len(df)]

        df["mean_tanimoto"] = compute_mean_tanimotos_against_db(subset_mols, precomputed_ccd, config)
        df.to_csv(results_path, index=False)  # save for next time

    # ── Activity (pChEMBL) ───────────────────────────────────────────────────
    if "pchembl_value" in df.columns:
        df = df.drop(columns=["pchembl_value"])

    if "pchembl_value" not in df.columns:
        if activity_path.exists():
            print("Merging pChEMBL activity values from cdk2_activity.csv...")

            act_df = pd.read_csv(activity_path)
            df = df.merge(act_df, on="id", how="left")
            df.to_csv(results_path, index=False)  # save for next time
        else:
            print(
                f"Activity CSV not found at {activity_path}. "
                "Re-run 01_fetch_and_precompute.py to generate it. "
                "Activity vs Neff plot will be skipped."
            )

    # ── Plot 1: Confidence vs Tanimoto ───────────────────────────────────────
    out_scatter = DATA_DIR / "cdk2_ccd_scatter.png"
    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(
        df["global_neff"],
        df["mean_tanimoto"],
        c=df["mean_tanimoto"],
        cmap="viridis",
        alpha=0.6,
        edgecolors="w",
        linewidths=0.5,
    )
    fig.colorbar(scatter, ax=ax, label="Mean Tanimoto Similarity")
    ax.set_xlabel("Global Neff Confidence (0–1)")
    ax.set_ylabel("Mean Tanimoto Similarity to Target Dataset")
    ax.set_title("CDK2 vs CCD: Confidence vs CCD Dataset Similarity")
    ax.grid(True, linestyle="--", alpha=0.3)
    _add_corr_annotation(ax, df["global_neff"], df["mean_tanimoto"])
    fig.tight_layout()
    fig.savefig(out_scatter, dpi=300)
    plt.close(fig)
    print(f"Scatter plot saved to {out_scatter}")

    # ── Plot 2: Activity vs Neff ─────────────────────────────────────────────
    if "pchembl_value" in df.columns:
        act_df_plot = df.dropna(subset=["pchembl_value"]).copy()
        if len(act_df_plot) < 2:
            print("Not enough activity data to plot activity vs Neff.")
        else:
            out_activity = DATA_DIR / "cdk2_ccd_activity_vs_neff.png"
            fig, axes = plt.subplots(1, 3, figsize=(24, 7))

            # --- 2a: global_neff vs pChEMBL ---
            ax = axes[0]
            sc = ax.scatter(
                act_df_plot["global_neff"],
                act_df_plot["pchembl_value"],
                c=act_df_plot["pchembl_value"],
                cmap="plasma",
                alpha=0.6,
                edgecolors="w",
                linewidths=0.5,
            )
            fig.colorbar(sc, ax=ax, label="pChEMBL Value")
            ax.set_xlabel("Global Neff Confidence (0–1)")
            ax.set_ylabel("Activity (pChEMBL)")
            ax.set_title("CDK2 vs CCD: Neff Confidence vs Activity")
            ax.grid(True, linestyle="--", alpha=0.3)
            _add_corr_annotation(ax, act_df_plot["global_neff"], act_df_plot["pchembl_value"])

            # --- 2b: global_confidence vs pChEMBL (if different from global_neff) ---
            ax = axes[1]
            x_col = "global_confidence" if "global_confidence" in act_df_plot.columns else "global_neff"
            sc2 = ax.scatter(
                act_df_plot[x_col],
                act_df_plot["pchembl_value"],
                c=act_df_plot["pchembl_value"],
                cmap="plasma",
                alpha=0.6,
                edgecolors="w",
                linewidths=0.5,
            )
            fig.colorbar(sc2, ax=ax, label="pChEMBL Value")
            ax.set_xlabel("Global Confidence (normalised)")
            ax.set_ylabel("Activity (pChEMBL)")
            ax.set_title("CDK2 vs CCD: Global Confidence vs Activity")
            ax.grid(True, linestyle="--", alpha=0.3)
            _add_corr_annotation(ax, act_df_plot[x_col], act_df_plot["pchembl_value"])

            # --- 2c: mean_tanimoto vs pChEMBL ---
            ax = axes[2]
            sc3 = ax.scatter(
                act_df_plot["mean_tanimoto"],
                act_df_plot["pchembl_value"],
                c=act_df_plot["pchembl_value"],
                cmap="plasma",
                alpha=0.6,
                edgecolors="w",
                linewidths=0.5,
            )
            fig.colorbar(sc3, ax=ax, label="pChEMBL Value")
            ax.set_xlabel("Mean Tanimoto Similarity to CCD")
            ax.set_ylabel("Activity (pChEMBL)")
            ax.set_title("CDK2 vs CCD: Dataset Tanimoto vs Activity")
            ax.grid(True, linestyle="--", alpha=0.3)
            _add_corr_annotation(ax, act_df_plot["mean_tanimoto"], act_df_plot["pchembl_value"])

            fig.suptitle("CDK2 Activity vs Neff Scores (from CCD Database)", fontsize=14, fontweight="bold")
            fig.tight_layout()
            fig.savefig(out_activity, dpi=300)
            plt.close(fig)
            print(f"Activity vs Neff plot saved to {out_activity}")

            # Print summary statistics
            pearson_r, pearson_p = stats.pearsonr(act_df_plot["global_neff"], act_df_plot["pchembl_value"])
            spearman_r, spearman_p = stats.spearmanr(act_df_plot["global_neff"], act_df_plot["pchembl_value"])
            print(f"\nActivity vs Neff correlations against CCD (n={len(act_df_plot)}):")
            print(f"  Pearson  r = {pearson_r:.3f}  (p = {pearson_p:.2e})")
            print(f"  Spearman ρ = {spearman_r:.3f}  (p = {spearman_p:.2e})")


if __name__ == "__main__":
    plot_ccd_results()
