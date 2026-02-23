"""
Sensitivity Example — Step 3: Visualization
============================================
Reads cross-Neff results and produces:
  A) 2×2 heatmap of mean global confidence (source × reference receptor)
  B) Violin plots of per-atom Neff distributions (temporarily removed).

One set of plots is generated per config variant; two summary panels combine
results for fp/agg and for chirality/features.

Usage:
    python 03_plot_sensitivity.py

    # Single config only:
    python 03_plot_sensitivity.py --config configs/base.yaml
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from ligand_neff.config import load_config

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CONFIGS_DIR = BASE_DIR / "configs"

RECEPTORS = ["PR", "AR"]

# ── Colour palette ────────────────────────────────────────────────────────────
PAIR_COLORS = {
    ("PR", "PR"): "#2196F3",  # Blue  — cognate
    ("PR", "AR"): "#F44336",  # Red   — off-target
    ("AR", "AR"): "#4CAF50",  # Green — cognate
    ("AR", "PR"): "#FF9800",  # Orange — off-target
}
PAIR_LABELS = {
    ("PR", "PR"): "PR → PR (cognate)",
    ("PR", "AR"): "PR → AR (off-target)",
    ("AR", "AR"): "AR → AR (cognate)",
    ("AR", "PR"): "AR → PR (off-target)",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_results(config_name: str) -> pd.DataFrame | None:
    csv_path = DATA_DIR / config_name / "cross_neff_results.csv"
    if not csv_path.exists():
        print(f"  [{config_name}] No results yet — skipping. Run 02 first.")
        return None
    return pd.read_csv(csv_path)


def plot_heatmap(df: pd.DataFrame, out_path: Path, title: str) -> None:
    """
    2×2 heatmap: rows = source receptor, cols = reference receptor.
    Cell value = mean global_confidence across all ligands in that (source, ref) cell.
    """
    matrix = np.zeros((2, 2))
    for ri, src in enumerate(RECEPTORS):
        for ci, ref in enumerate(RECEPTORS):
            mask = (df["source_receptor"] == src) & (df["ref_receptor"] == ref)
            vals = df.loc[mask, "global_confidence"]
            matrix[ri, ci] = vals.mean() if not vals.empty else float("nan")

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(RECEPTORS, fontsize=12)
    ax.set_yticklabels(RECEPTORS, fontsize=12)
    ax.set_xlabel("Reference DB Receptor", fontsize=12)
    ax.set_ylabel("Query Receptor (source)", fontsize=12)
    ax.set_title(title, fontsize=11, pad=8)

    for ri in range(2):
        for ci in range(2):
            val = matrix[ri, ci]
            text = f"{val:.3f}" if not np.isnan(val) else "N/A"
            # Choose text colour for readability
            text_col = "black" if 0.3 < val < 0.75 else "white"
            ax.text(ci, ri, text, ha="center", va="center",
                    fontsize=13, fontweight="bold", color=text_col)

    plt.colorbar(im, ax=ax, label="Mean Global Confidence")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_fp_agg_summary_panel(loaded_results: list[tuple[str, pd.DataFrame, any]], out_path: Path) -> None:
    """
    Summary panel: 9 mini-heatmaps arranged in a 3×3 grid
    (rows = aggregation method, cols = fp_size).
    """
    AGGREGATIONS = ["geometric", "minimum", "mean"]
    FP_SIZES = [2048, 4096, 8192]

    lookup = {}
    for cfg_name, df, cfg in loaded_results:
        if not cfg.use_chirality and not cfg.use_features:
            lookup[(cfg.fp_size, cfg.aggregation)] = df

    if not lookup:
        return

    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    fig.suptitle(
        "Sensitivity Analysis: Mean Global Confidence\n(rows = aggregation, cols = fp_size)",
        fontsize=13, y=1.01,
    )

    for ri, agg in enumerate(AGGREGATIONS):
        for ci, fp in enumerate(FP_SIZES):
            ax = axes[ri][ci]
            ax.set_title(f"fp={fp} | {agg}", fontsize=8)
            ax.set_xticks([0, 1]); ax.set_xticklabels(RECEPTORS, fontsize=7)
            ax.set_yticks([0, 1]); ax.set_yticklabels(RECEPTORS, fontsize=7)

            if (fp, agg) not in lookup:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9, color="gray")
                continue

            df = lookup[(fp, agg)]
            matrix = np.zeros((2, 2))
            for r_i, src in enumerate(RECEPTORS):
                for c_i, ref in enumerate(RECEPTORS):
                    mask = (df["source_receptor"] == src) & (df["ref_receptor"] == ref)
                    vals = df.loc[mask, "global_confidence"]
                    matrix[r_i, c_i] = vals.mean() if not vals.empty else float("nan")

            im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1)
            for r_i in range(2):
                for c_i in range(2):
                    val = matrix[r_i, c_i]
                    text = f"{val:.2f}" if not np.isnan(val) else "N/A"
                    text_col = "black" if 0.3 < val < 0.75 else "white"
                    ax.text(c_i, r_i, text, ha="center", va="center",
                            fontsize=9, fontweight="bold", color=text_col)

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    norm = Normalize(vmin=0, vmax=1)
    fig.colorbar(ScalarMappable(norm=norm, cmap="RdYlGn"),
                 cax=cbar_ax, label="Mean Global Confidence")

    for ri, agg in enumerate(AGGREGATIONS):
        axes[ri][0].set_ylabel(agg, fontsize=9, rotation=90, labelpad=4)
    for ci, fp in enumerate(FP_SIZES):
        axes[2][ci].set_xlabel(f"fp={fp}", fontsize=9)

    plt.tight_layout(rect=[0, 0, 0.91, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"FP/Agg Summary panel saved → {out_path.relative_to(BASE_DIR)}")


def plot_feat_chir_summary_panel(loaded_results: list[tuple[str, pd.DataFrame, any]], out_path: Path) -> None:
    """
    Summary panel: 2x2 grid for use_chirality vs use_features
    (rows = chirality, cols = features).
    """
    CHIRALITIES = [False, True]
    FEATURES = [False, True]

    lookup = {}
    for cfg_name, df, cfg in loaded_results:
        if cfg.fp_size == 2048 and cfg.aggregation == "geometric":
            lookup[(cfg.use_chirality, cfg.use_features)] = df

    if not lookup:
        return

    fig, axes = plt.subplots(2, 2, figsize=(8, 7))
    fig.suptitle(
        "Sensitivity Analysis: Mean Global Confidence\n(rows = chirality, cols = features)\n[fp_size=2048, agg=geometric]",
        fontsize=13, y=1.01,
    )

    for ri, chir in enumerate(CHIRALITIES):
        for ci, feat in enumerate(FEATURES):
            ax = axes[ri][ci]
            ax.set_title(f"chirality={chir} | features={feat}", fontsize=8)
            ax.set_xticks([0, 1]); ax.set_xticklabels(RECEPTORS, fontsize=7)
            ax.set_yticks([0, 1]); ax.set_yticklabels(RECEPTORS, fontsize=7)

            if (chir, feat) not in lookup:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9, color="gray")
                continue

            df = lookup[(chir, feat)]
            matrix = np.zeros((2, 2))
            for r_i, src in enumerate(RECEPTORS):
                for c_i, ref in enumerate(RECEPTORS):
                    mask = (df["source_receptor"] == src) & (df["ref_receptor"] == ref)
                    vals = df.loc[mask, "global_confidence"]
                    matrix[r_i, c_i] = vals.mean() if not vals.empty else float("nan")

            im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1)
            for r_i in range(2):
                for c_i in range(2):
                    val = matrix[r_i, c_i]
                    text = f"{val:.2f}" if not np.isnan(val) else "N/A"
                    text_col = "black" if 0.3 < val < 0.75 else "white"
                    ax.text(c_i, r_i, text, ha="center", va="center",
                            fontsize=9, fontweight="bold", color=text_col)

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    norm = Normalize(vmin=0, vmax=1)
    fig.colorbar(ScalarMappable(norm=norm, cmap="RdYlGn"),
                 cax=cbar_ax, label="Mean Global Confidence")

    for ri, chir in enumerate(CHIRALITIES):
        axes[ri][0].set_ylabel(f"Chirality {chir}", fontsize=9, rotation=90, labelpad=4)
    for ci, feat in enumerate(FEATURES):
        axes[1][ci].set_xlabel(f"Features {feat}", fontsize=9)

    plt.tight_layout(rect=[0, 0, 0.91, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Feat/Chir Summary panel saved → {out_path.relative_to(BASE_DIR)}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot sensitivity analysis results."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a specific YAML config. Omit to plot all available results.",
    )
    args = parser.parse_args()

    if args.config:
        config_names = [args.config.stem]
    else:
        config_names = [p.stem for p in sorted(CONFIGS_DIR.glob("*.yaml"))]

    loaded: list[tuple[str, pd.DataFrame, any]] = []

    for cfg_name in config_names:
        df = load_results(cfg_name)
        if df is None:
            continue

        cfg_path = CONFIGS_DIR / f"{cfg_name}.yaml"
        cfg = load_config(cfg_path)

        out_dir = DATA_DIR / cfg_name
        title = f"{cfg_name}  (fp={cfg.fp_size}, agg={cfg.aggregation}, chir={cfg.use_chirality}, feat={cfg.use_features})"

        print(f"[{cfg_name}] Plotting heatmap...")
        plot_heatmap(df, out_dir / "heatmap.png", title)
        # Violin plots removed as requested
        loaded.append((cfg_name, df, cfg))
        print(f"  Saved to {out_dir.relative_to(BASE_DIR)}/")

    if len(loaded) > 1:
        print("\nGenerating summary panels...")
        plot_fp_agg_summary_panel(loaded, DATA_DIR / "summary_panel_fp_agg.png")
        plot_feat_chir_summary_panel(loaded, DATA_DIR / "summary_panel_feat_chir.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
