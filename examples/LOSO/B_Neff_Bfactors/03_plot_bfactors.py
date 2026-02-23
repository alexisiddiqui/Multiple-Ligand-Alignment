import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

def plot_neff_vs_bfactors():
    """Plot Global Neff Confidence vs Spearman Correlation of Per-Atom Neff and B-factors."""
    print("Plotting Neff vs B-factors...")
    
    results_csv = DATA_DIR / "bfactor_results.csv"
    if not results_csv.exists():
        print("Results CSV not found. Run 02_compute_neff_bfactors.py first.")
        return
        
    df = pd.read_csv(results_csv)
    
    # Drop rows where correlation couldn't be computed
    df = df.dropna(subset=["spearman_r", "global_confidence"])
    
    if df.empty:
        print("No valid data to plot.")
        return
        
    plt.figure(figsize=(10, 7))
    
    # Scatter plot
    scatter = plt.scatter(
        df["global_confidence"], 
        df["spearman_r"], 
        c=df["global_confidence"], # Color by confidence also
        cmap="plasma", 
        alpha=0.7,
        edgecolors="w",
        linewidths=0.5,
        s=100
    )
    
    # Add annotations for each point
    for i, row in df.iterrows():
        plt.annotate(
            f"{row['pdb_id']}:{row['ligand_id']}",
            (row["global_confidence"], row["spearman_r"]),
            xytext=(5, 5), textcoords="offset points",
            fontsize=8, alpha=0.8
        )
    
    plt.colorbar(scatter, label="Global Neff Confidence")
    plt.xlabel("Global Neff Confidence (0-1)")
    plt.ylabel("Spearman Correlation (Atom Neff vs B-factor)")
    plt.title("CDK2 PDB Ligands: Confidence vs Flexibility Correlation")
    
    # Add horizontal line at 0 for reference
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    
    plt.grid(True, linestyle="--", alpha=0.3)
    
    # Optional Trendline
    if len(df) > 1:
        z = np.polyfit(df["global_confidence"], df["spearman_r"], 1)
        p = np.poly1d(z)
        plt.plot(df["global_confidence"], p(df["global_confidence"]), "r--", alpha=0.5, label=f"Trendline")
        plt.legend(loc="lower left")
    
    out_path = DATA_DIR / "cdk2_bfactor_scatter.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Scatter plot saved to {out_path}")

if __name__ == "__main__":
    import numpy as np
    plot_neff_vs_bfactors()
