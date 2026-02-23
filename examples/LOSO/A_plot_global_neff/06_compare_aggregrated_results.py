"""
Plots activity vs a range of ways to aggregate Neff scores.

This example creates two sets of neff scores - one against the ligand set and another against the PDB CCD.

The ligand-set (LOSO) Neff should be *positively* correlated with activity: molecules
that look like known actives tend to be active themselves.  The CCD Neff should be
*negatively* correlated: high similarity to the entire chemical universe implies a
generic scaffold rather than a selective one.

We exploit this sign difference via *contrastive* aggregation strategies:
  LOSO − α·CCD   (difference)
  LOSO / CCD     (selectivity ratio)
  optimal α, β   (grid-searched linear combination)

Look at scripts 03 and 05 to see the per-source plots.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
from itertools import product

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"


# ── Aggregation strategies ────────────────────────────────────────────────────
# Each callable takes (loso_array, ccd_array) -> combined_array.
# "Naïve" strategies treat both sources equally; "contrastive" strategies
# exploit the opposite correlation directions.

_EPS = 1e-8  # avoid division by zero

NAIVE_AGGREGATIONS: dict[str, callable] = {
    "LOSO Only": lambda a, _b: a,
    "CCD Only": lambda _a, b: b,
    "Mean": lambda a, b: (a + b) / 2,
    "Max": lambda a, b: np.maximum(a, b),
    "Min": lambda a, b: np.minimum(a, b),
    "Geometric Mean": lambda a, b: np.sqrt(np.clip(a, 0, None) * np.clip(b, 0, None)),
}

CONTRASTIVE_AGGREGATIONS: dict[str, callable] = {
    "LOSO − CCD": lambda a, b: a - b,
    "LOSO − 0.5·CCD": lambda a, b: a - 0.5 * b,
    "LOSO − 2·CCD": lambda a, b: a - 2.0 * b,
    "LOSO / CCD": lambda a, b: a / (b + _EPS),
    "log(LOSO/CCD)": lambda a, b: np.log((a + _EPS) / (b + _EPS)),
    "Rank(LOSO) − Rank(CCD)": None,  # handled specially below
}


# ── Helpers ───────────────────────────────────────────────────────────────────

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
        fontsize=8, fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="gray", alpha=0.8),
    )


def _rankdata(arr: np.ndarray) -> np.ndarray:
    """Rank-transform (ascending, average ties)."""
    return stats.rankdata(arr, method="average")


def _apply_aggregation(name: str, fn, loso: np.ndarray, ccd: np.ndarray) -> np.ndarray:
    """Apply a named aggregation, handling rank-based strategies specially."""
    if name == "Rank(LOSO) − Rank(CCD)":
        return _rankdata(loso) - _rankdata(ccd)
    return fn(loso, ccd)


def _load_and_merge() -> pd.DataFrame:
    """Load LOSO and CCD result CSVs and merge on molecule ID.

    Returns a combined DataFrame with suffixed columns for each source
    (``_loso`` and ``_ccd``) plus the shared ``pchembl_value`` column.
    """
    loso_path = DATA_DIR / "cdk2_results.csv"
    ccd_path = DATA_DIR / "cdk2_ccd_results.csv"

    if not loso_path.exists():
        raise FileNotFoundError(
            f"LOSO results not found at {loso_path}. Run 02_run_loso.py first."
        )
    if not ccd_path.exists():
        raise FileNotFoundError(
            f"CCD results not found at {ccd_path}. Run 04_run_CCD.py first."
        )

    loso_df = pd.read_csv(loso_path)
    ccd_df = pd.read_csv(ccd_path)

    merged = loso_df.merge(
        ccd_df,
        on="id",
        suffixes=("_loso", "_ccd"),
        how="inner",
    )

    # Resolve pchembl_value: take whichever is non-NaN, preferring LOSO
    if "pchembl_value_loso" in merged.columns:
        merged["pchembl_value"] = merged["pchembl_value_loso"].fillna(
            merged.get("pchembl_value_ccd", np.nan)
        )
        merged.drop(
            columns=[c for c in ("pchembl_value_loso", "pchembl_value_ccd")
                     if c in merged.columns],
            inplace=True,
        )

    print(f"Merged {len(merged)} molecules from LOSO and CCD result sets.")
    return merged


def _corr_summary(agg_vals, activity):
    """Return a dict of Pearson / Spearman stats."""
    pr, pp = stats.pearsonr(agg_vals, activity)
    sr, sp = stats.spearmanr(agg_vals, activity)
    return {"Pearson r": pr, "Pearson p": pp, "Spearman ρ": sr, "Spearman p": sp}


# ── Generic grid plotter ─────────────────────────────────────────────────────

def _plot_aggregation_grid(
    df: pd.DataFrame,
    loso_col: str,
    ccd_col: str,
    aggregations: dict[str, callable],
    title_prefix: str,
    score_label: str,
    out_name: str,
    section_label: str,
) -> list[dict]:
    """Plot a grid of scatter plots (aggregated score vs activity).

    Returns a list of summary rows for tabular output.
    """
    act_df = df.dropna(subset=["pchembl_value"]).copy()
    if len(act_df) < 2:
        print(f"Not enough activity data for {section_label}.")
        return []

    if loso_col not in act_df.columns or ccd_col not in act_df.columns:
        print(f"Columns {loso_col} / {ccd_col} not found – skipping {section_label}.")
        return []

    n_agg = len(aggregations)
    ncols = 3
    nrows = max(1, (n_agg + ncols - 1) // ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 7 * nrows))
    axes = np.asarray(axes).ravel()
    summary_rows = []

    loso_vals = act_df[loso_col].values
    ccd_vals = act_df[ccd_col].values
    activity = act_df["pchembl_value"].values

    for i, (name, fn) in enumerate(aggregations.items()):
        ax = axes[i]
        agg_vals = _apply_aggregation(name, fn, loso_vals, ccd_vals)

        sc = ax.scatter(
            agg_vals, activity,
            c=activity, cmap="plasma", alpha=0.6,
            edgecolors="w", linewidths=0.5,
        )
        fig.colorbar(sc, ax=ax, label="pChEMBL Value")
        ax.set_xlabel(f"{score_label} ({name})")
        ax.set_ylabel("Activity (pChEMBL)")
        ax.set_title(f"CDK2: {name}")
        ax.grid(True, linestyle="--", alpha=0.3)
        _add_corr_annotation(ax, agg_vals, activity)

        row = {"Aggregation": name, **_corr_summary(agg_vals, activity)}
        summary_rows.append(row)

    for j in range(n_agg, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"{title_prefix} vs Activity", fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = DATA_DIR / out_name
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"{section_label} plot saved to {out}")

    _print_summary_table(section_label, summary_rows)
    return summary_rows


def _print_summary_table(label: str, rows: list[dict]):
    summary = pd.DataFrame(rows)
    print(f"\n{'=' * 72}")
    print(label)
    print("=" * 72)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()


# ── Optimal weight search ────────────────────────────────────────────────────

def _find_optimal_weights(
    df: pd.DataFrame,
    loso_col: str,
    ccd_col: str,
    metric: str = "spearman",
) -> tuple[float, float, float]:
    """Grid-search for α, β that maximise corr(α·LOSO + β·CCD, activity).

    Searches α ∈ [0, 2] and β ∈ [−2, 0] (contrastive) at 0.05 resolution.
    Returns (best_α, best_β, best_correlation).
    """
    act_df = df.dropna(subset=["pchembl_value"]).copy()
    loso = act_df[loso_col].values
    ccd = act_df[ccd_col].values
    activity = act_df["pchembl_value"].values

    alphas = np.arange(0.0, 2.05, 0.05)
    betas = np.arange(-2.0, 0.05, 0.05)

    best_corr = -np.inf
    best_a, best_b = 1.0, 0.0

    for a, b in product(alphas, betas):
        combined = a * loso + b * ccd
        if metric == "spearman":
            r, _ = stats.spearmanr(combined, activity)
        else:
            r, _ = stats.pearsonr(combined, activity)
        if r > best_corr:
            best_corr = r
            best_a, best_b = a, b

    return best_a, best_b, best_corr


def _plot_weight_heatmap(
    df: pd.DataFrame,
    loso_col: str,
    ccd_col: str,
    score_label: str,
    out_name: str,
):
    """Plot a heatmap of Spearman ρ as a function of (α, β) in α·LOSO + β·CCD."""
    act_df = df.dropna(subset=["pchembl_value"]).copy()
    if len(act_df) < 2:
        return

    if loso_col not in act_df.columns or ccd_col not in act_df.columns:
        return

    loso = act_df[loso_col].values
    ccd = act_df[ccd_col].values
    activity = act_df["pchembl_value"].values

    alphas = np.arange(0.0, 2.05, 0.1)
    betas = np.arange(-2.0, 0.15, 0.1)
    corr_grid = np.zeros((len(betas), len(alphas)))

    for ia, a in enumerate(alphas):
        for ib, b in enumerate(betas):
            combined = a * loso + b * ccd
            r, _ = stats.spearmanr(combined, activity)
            corr_grid[ib, ia] = r

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(
        corr_grid,
        aspect="auto",
        origin="lower",
        extent=[alphas[0], alphas[-1], betas[0], betas[-1]],
        cmap="RdYlGn",
    )
    fig.colorbar(im, ax=ax, label="Spearman ρ")
    ax.set_xlabel("α  (LOSO weight)")
    ax.set_ylabel("β  (CCD weight, negative = contrastive)")
    ax.set_title(f"CDK2: Optimal α·{score_label}_LOSO + β·{score_label}_CCD")
    ax.grid(True, linestyle="--", alpha=0.2)

    # Mark the optimum
    best_idx = np.unravel_index(corr_grid.argmax(), corr_grid.shape)
    best_b = betas[best_idx[0]]
    best_a = alphas[best_idx[1]]
    best_r = corr_grid[best_idx]
    ax.plot(best_a, best_b, "k*", markersize=15)
    ax.annotate(
        f"α={best_a:.2f}, β={best_b:.2f}\nρ={best_r:.3f}",
        xy=(best_a, best_b),
        xytext=(best_a + 0.15, best_b + 0.15),
        fontsize=10, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="black"),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.9),
    )

    fig.tight_layout()
    out = DATA_DIR / out_name
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Weight heatmap saved to {out}")


# ── Plot: LOSO vs CCD direct comparison scatter ──────────────────────────────

def _plot_loso_vs_ccd(df: pd.DataFrame) -> None:
    """Direct scatter of LOSO scores vs CCD scores (global_neff and confidence)."""
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))

    ax = axes[0]
    sc = ax.scatter(
        df["global_neff_loso"], df["global_neff_ccd"],
        c=df.get("pchembl_value", pd.Series(dtype=float)),
        cmap="coolwarm", alpha=0.6, edgecolors="w", linewidths=0.5,
    )
    if "pchembl_value" in df.columns:
        fig.colorbar(sc, ax=ax, label="pChEMBL Value")
    ax.set_xlabel("Global Neff (LOSO)")
    ax.set_ylabel("Global Neff (CCD)")
    ax.set_title("LOSO vs CCD: Global Neff")
    ax.grid(True, linestyle="--", alpha=0.3)
    _add_corr_annotation(ax, df["global_neff_loso"], df["global_neff_ccd"])

    ax = axes[1]
    conf_loso = "global_confidence_loso" if "global_confidence_loso" in df.columns else "global_neff_loso"
    conf_ccd = "global_confidence_ccd" if "global_confidence_ccd" in df.columns else "global_neff_ccd"
    sc2 = ax.scatter(
        df[conf_loso], df[conf_ccd],
        c=df.get("pchembl_value", pd.Series(dtype=float)),
        cmap="coolwarm", alpha=0.6, edgecolors="w", linewidths=0.5,
    )
    if "pchembl_value" in df.columns:
        fig.colorbar(sc2, ax=ax, label="pChEMBL Value")
    ax.set_xlabel("Global Confidence (LOSO)")
    ax.set_ylabel("Global Confidence (CCD)")
    ax.set_title("LOSO vs CCD: Global Confidence")
    ax.grid(True, linestyle="--", alpha=0.3)
    _add_corr_annotation(ax, df[conf_loso], df[conf_ccd])

    ax = axes[2]
    has_tan = "mean_tanimoto_loso" in df.columns and "mean_tanimoto_ccd" in df.columns
    if has_tan:
        sc3 = ax.scatter(
            df["mean_tanimoto_loso"], df["mean_tanimoto_ccd"],
            c=df.get("pchembl_value", pd.Series(dtype=float)),
            cmap="coolwarm", alpha=0.6, edgecolors="w", linewidths=0.5,
        )
        if "pchembl_value" in df.columns:
            fig.colorbar(sc3, ax=ax, label="pChEMBL Value")
        ax.set_xlabel("Mean Tanimoto (LOSO)")
        ax.set_ylabel("Mean Tanimoto (CCD)")
        ax.set_title("LOSO vs CCD: Mean Tanimoto")
        ax.grid(True, linestyle="--", alpha=0.3)
        _add_corr_annotation(ax, df["mean_tanimoto_loso"], df["mean_tanimoto_ccd"])
    else:
        ax.set_visible(False)

    fig.suptitle("CDK2: LOSO vs CCD Direct Comparison", fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = DATA_DIR / "cdk2_loso_vs_ccd.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"LOSO vs CCD comparison plot saved to {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def compare_aggregated_results():
    """Load LOSO + CCD results, aggregate in multiple ways, and plot comparisons."""
    df = _load_and_merge()

    # ── 1. Direct LOSO vs CCD scatter ────────────────────────────────────────
    _plot_loso_vs_ccd(df)

    # ── 2. Naïve aggregation grid (Neff) ─────────────────────────────────────
    _plot_aggregation_grid(
        df, "global_neff_loso", "global_neff_ccd",
        NAIVE_AGGREGATIONS, "CDK2: Naïve Neff Aggregation",
        "Neff", "cdk2_naive_neff_vs_activity.png",
        "Naïve aggregation – Neff vs Activity",
    )

    # ── 3. Contrastive aggregation grid (Neff) ───────────────────────────────
    _plot_aggregation_grid(
        df, "global_neff_loso", "global_neff_ccd",
        CONTRASTIVE_AGGREGATIONS,
        "CDK2: Contrastive Neff (LOSO − CCD)",
        "Neff", "cdk2_contrastive_neff_vs_activity.png",
        "Contrastive aggregation – Neff vs Activity",
    )

    # ── 4. Contrastive aggregation grid (Confidence) ─────────────────────────
    conf_loso = "global_confidence_loso" if "global_confidence_loso" in df.columns else "global_neff_loso"
    conf_ccd = "global_confidence_ccd" if "global_confidence_ccd" in df.columns else "global_neff_ccd"
    _plot_aggregation_grid(
        df, conf_loso, conf_ccd,
        CONTRASTIVE_AGGREGATIONS,
        "CDK2: Contrastive Confidence (LOSO − CCD)",
        "Confidence", "cdk2_contrastive_confidence_vs_activity.png",
        "Contrastive aggregation – Confidence vs Activity",
    )

    # ── 5. Contrastive aggregation grid (Tanimoto) ───────────────────────────
    if "mean_tanimoto_loso" in df.columns and "mean_tanimoto_ccd" in df.columns:
        _plot_aggregation_grid(
            df, "mean_tanimoto_loso", "mean_tanimoto_ccd",
            CONTRASTIVE_AGGREGATIONS,
            "CDK2: Contrastive Tanimoto (LOSO − CCD)",
            "Tanimoto", "cdk2_contrastive_tanimoto_vs_activity.png",
            "Contrastive aggregation – Tanimoto vs Activity",
        )

    # ── 6. Weight-optimisation heatmaps ──────────────────────────────────────
    print("\n" + "━" * 72)
    print("Optimal weight search: α·LOSO + β·CCD  (grid Δ=0.05)")
    print("━" * 72)

    for label, loso_c, ccd_c, out in [
        ("Neff", "global_neff_loso", "global_neff_ccd",
         "cdk2_weight_heatmap_neff.png"),
        ("Confidence", conf_loso, conf_ccd,
         "cdk2_weight_heatmap_confidence.png"),
        ("Tanimoto", "mean_tanimoto_loso", "mean_tanimoto_ccd",
         "cdk2_weight_heatmap_tanimoto.png"),
    ]:
        if loso_c not in df.columns or ccd_c not in df.columns:
            continue
        _plot_weight_heatmap(df, loso_c, ccd_c, label, out)
        a, b, r = _find_optimal_weights(df, loso_c, ccd_c, metric="spearman")
        print(f"  {label:12s}:  α = {a:.2f},  β = {b:.2f}  →  Spearman ρ = {r:.4f}")


if __name__ == "__main__":
    compare_aggregated_results()