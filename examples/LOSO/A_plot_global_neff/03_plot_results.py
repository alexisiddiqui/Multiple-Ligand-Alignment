import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
from ligand_neff.config import load_config
from examples.LOSO.common.utils import compute_mean_tanimotos, load_mols_from_smi

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
COMMON_DIR = BASE_DIR.parent / "common"

def plot_results():
    results_path = DATA_DIR / "cdk2_results.csv"
    out_path = DATA_DIR / "cdk2_scatter.png"
    smi_path = DATA_DIR / "cdk2_ligands.smi"
    config_path = COMMON_DIR / "cdk2_config.yaml"
    
    if not results_path.exists():
        print(f"Results CSV not found at {results_path}. Run 02_run_loso.py first.")
        return
        
    df = pd.read_csv(results_path)
    
    if "mean_tanimoto" not in df.columns:
        print("Computing mean Tanimoto similarities...")
        config = load_config(config_path)
        mols = load_mols_from_smi(smi_path)
        
        if "idx" in df.columns:
            subset_mols = [mols[i] for i in df["idx"]]
        else:
            subset_mols = mols[:len(df)]
            
        df["mean_tanimoto"] = compute_mean_tanimotos(subset_mols, config)
        df.to_csv(results_path, index=False) # save for next time
    
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(
        df["global_neff"], 
        df["mean_tanimoto"], 
        c=df["mean_tanimoto"], 
        cmap="viridis", 
        alpha=0.6,
        edgecolors="w",
        linewidths=0.5
    )
    
    plt.colorbar(scatter, label="Mean Tanimoto Similarity")
    plt.xlabel("Global Neff Confidence (0-1)")
    plt.ylabel("Mean Tanimoto Similarity to Reference Set")
    plt.title("CDK2 LOSO: Confidence vs Dataset Similarity")
    plt.grid(True, linestyle="--", alpha=0.3)
    
    pearson_r, pearson_p = stats.pearsonr(df["global_neff"], df["mean_tanimoto"])
    spearman_r, spearman_p = stats.spearmanr(df["global_neff"], df["mean_tanimoto"])
    corr_text = (
        f"Pearson  r = {pearson_r:.3f}  (p = {pearson_p:.2e})\\n"
        f"Spearman ρ = {spearman_r:.3f}  (p = {spearman_p:.2e})"
    )
    plt.annotate(
        corr_text,
        xy=(0.05, 0.95), xycoords="axes fraction",
        verticalalignment="top",
        fontsize=10, fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.8),
    )
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Scatter plot saved to {out_path}")

if __name__ == "__main__":
    plot_results()
