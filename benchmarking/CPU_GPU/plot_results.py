#!/usr/bin/env python3
"""Plot CPU vs GPU benchmark results from a CSV produced by benchmark_cpu_gpu.py.

Generates:
  1. Grouped bar chart      — per-stage mean time, CPU vs GPU, for each (n_db, fp_size).
  2. Line chart             — total pipeline time vs n_db, separately for each fp_size.
  3. Speedup heatmap        — GPU speedup factor (cpu/gpu) across (n_db × stage).
  4. Abs-improvement heatmap — GPU time saving in ms (cpu − gpu) across (n_db × stage),
                               colour scale centred at 0 ms.

Usage
-----
    # Use default CSV path
    uv run python benchmarking/CPU_GPU/plot_results.py

    # Specify CSV and output directory
    uv run python benchmarking/CPU_GPU/plot_results.py \\
        --csv benchmarking/CPU_GPU/results/cpu_gpu_benchmark.csv \\
        --out-dir benchmarking/CPU_GPU/results/plots
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


# ── Helpers ───────────────────────────────────────────────────────────────────

PHASE_ORDER = [
    "1_fingerprint", "2_decompose", "3_filter",
    "4_weight", "5_per_atom", "6_aggregate", "7_normalise",
]

STAGE_LABEL = {
    "1_fingerprint": "Fingerprint",
    "2_decompose":   "Decompose",
    "3_filter":      "Filter",
    "4_weight":      "Weight",
    "5_per_atom":    "Per-atom",
    "6_aggregate":   "Aggregate",
    "7_normalise":   "Normalise",
}

DEVICE_COLOR = {
    "cpu":    "#4C72B0",
    "gpu":    "#DD8452",
    "metal":  "#DD8452",
}

DEVICE_LABEL = {
    "cpu":   "CPU",
    "gpu":   "GPU (Metal)",
    "metal": "GPU (Metal)",
}


def _device_color(dev: str) -> str:
    return DEVICE_COLOR.get(dev.lower(), "#7f7f7f")


def _device_label(dev: str) -> str:
    return DEVICE_LABEL.get(dev.lower(), dev.upper())


def _phase_from_stage(stage: str) -> str:
    """'4_weight_r1' → '4_weight'"""
    parts = stage.split("_r")
    return parts[0]


# ── Data loading & aggregation ────────────────────────────────────────────────

def load_and_aggregate(csv_path: Path) -> pd.DataFrame:
    """Load CSV; roll up per-radius stages into phase totals per (device, n_db, fp_size)."""
    df = pd.read_csv(csv_path)
    df["phase"] = df["stage"].apply(_phase_from_stage)

    # Sum mean_ms across radii for the same phase in the same iteration context.
    # Since each row is already aggregated over n_iters we sum across radii.
    phase_df = (
        df.groupby(["device", "n_db", "fp_size", "max_refs", "chunk_size", "phase"])
        .agg(mean_ms=("mean_ms", "sum"), std_ms=("std_ms", "sum"))
        .reset_index()
    )

    phase_df["phase_order"] = phase_df["phase"].map(
        {p: i for i, p in enumerate(PHASE_ORDER)}
    ).fillna(99)
    phase_df = phase_df.sort_values(["n_db", "fp_size", "device", "phase_order"])
    return phase_df


# ── Plot 1: Grouped bar chart per config ──────────────────────────────────────

def plot_stage_bars(df: pd.DataFrame, out_dir: Path) -> None:
    configs = df[["n_db", "fp_size"]].drop_duplicates().sort_values(["fp_size", "n_db"])
    devices = sorted(df["device"].unique())
    n_devices = len(devices)
    phases = [p for p in PHASE_ORDER if p in df["phase"].values]

    for _, row in configs.iterrows():
        n_db, fp_size = int(row["n_db"]), int(row["fp_size"])
        sub = df[(df["n_db"] == n_db) & (df["fp_size"] == fp_size)]

        fig, ax = plt.subplots(figsize=(12, 5))
        x = np.arange(len(phases))
        width = 0.75 / n_devices
        offsets = np.linspace(-0.375 + width / 2, 0.375 - width / 2, n_devices)

        for dev, offset in zip(devices, offsets):
            dev_data = sub[sub["device"] == dev].set_index("phase")
            means = [dev_data.loc[p, "mean_ms"] if p in dev_data.index else 0.0 for p in phases]
            stds  = [dev_data.loc[p, "std_ms"]  if p in dev_data.index else 0.0 for p in phases]
            bars = ax.bar(
                x + offset, means, width,
                label=_device_label(dev),
                color=_device_color(dev),
                yerr=stds, capsize=4, alpha=0.92,
            )

        ax.set_xticks(x)
        ax.set_xticklabels([STAGE_LABEL.get(p, p) for p in phases], fontsize=10)
        ax.set_ylabel("Mean time (ms)", fontsize=11)
        ax.set_title(
            f"Per-stage CPU vs GPU  |  n_db={n_db}, fp_size={fp_size}",
            fontsize=13, fontweight="bold",
        )
        ax.legend(fontsize=10)
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
        ax.grid(axis="y", which="major", linestyle="--", alpha=0.4)
        ax.grid(axis="y", which="minor", linestyle=":", alpha=0.2)
        fig.tight_layout()

        fname = out_dir / f"stage_bars_ndb{n_db}_fp{fp_size}.png"
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        print(f"  Saved: {fname}")


# ── Plot 2: Total time vs n_db line chart ─────────────────────────────────────

def plot_total_time(df: pd.DataFrame, out_dir: Path) -> None:
    totals = (
        df.groupby(["device", "n_db", "fp_size"])["mean_ms"]
        .sum()
        .reset_index()
        .rename(columns={"mean_ms": "total_ms"})
    )
    fp_sizes = sorted(totals["fp_size"].unique())
    devices  = sorted(totals["device"].unique())

    for fp_size in fp_sizes:
        sub = totals[totals["fp_size"] == fp_size].sort_values("n_db")
        fig, ax = plt.subplots(figsize=(9, 5))

        for dev in devices:
            dev_data = sub[sub["device"] == dev]
            ax.plot(
                dev_data["n_db"], dev_data["total_ms"],
                marker="o", linewidth=2.0,
                color=_device_color(dev),
                label=_device_label(dev),
            )

        ax.set_xlabel("Database size (n_db)", fontsize=11)
        ax.set_ylabel("Total pipeline time (ms)", fontsize=11)
        ax.set_title(f"Total time vs DB size  |  fp_size={fp_size}", fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(linestyle="--", alpha=0.4)
        fig.tight_layout()

        fname = out_dir / f"total_time_fp{fp_size}.png"
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        print(f"  Saved: {fname}")


# ── Plot 3: Speedup heatmap ───────────────────────────────────────────────────

def plot_speedup_heatmap(df: pd.DataFrame, out_dir: Path) -> None:
    """For each (fp_size), plot a heatmap of GPU speedup (cpu_time / gpu_time)
    with axes n_db × phase."""
    devices = df["device"].unique()
    gpu_devs = [d for d in devices if d.lower() in ("gpu", "metal")]
    if not gpu_devs:
        print("  No GPU device in data — skipping speedup heatmap.")
        return

    gpu_dev = gpu_devs[0]
    cpu_dev = "cpu"
    fp_sizes = sorted(df["fp_size"].unique())
    phases = [p for p in PHASE_ORDER if p in df["phase"].values]

    for fp_size in fp_sizes:
        sub = df[df["fp_size"] == fp_size]
        n_dbs = sorted(sub["n_db"].unique())

        speedup_matrix = np.zeros((len(n_dbs), len(phases)))
        for i, n_db in enumerate(n_dbs):
            for j, phase in enumerate(phases):
                cpu_row = sub[(sub["device"] == cpu_dev) & (sub["n_db"] == n_db) & (sub["phase"] == phase)]
                gpu_row = sub[(sub["device"] == gpu_dev) & (sub["n_db"] == n_db) & (sub["phase"] == phase)]
                if not cpu_row.empty and not gpu_row.empty:
                    cpu_t = float(cpu_row["mean_ms"].values[0])
                    gpu_t = float(gpu_row["mean_ms"].values[0])
                    speedup_matrix[i, j] = cpu_t / gpu_t if gpu_t > 0 else 0.0

        fig, ax = plt.subplots(figsize=(10, max(3, len(n_dbs) * 0.7 + 2)))
        im = ax.imshow(speedup_matrix, aspect="auto", cmap="RdYlGn", vmin=0.5, vmax=max(2.0, speedup_matrix.max()))
        ax.set_xticks(np.arange(len(phases)))
        ax.set_xticklabels([STAGE_LABEL.get(p, p) for p in phases], rotation=30, ha="right", fontsize=9)
        ax.set_yticks(np.arange(len(n_dbs)))
        ax.set_yticklabels([f"n_db={n}" for n in n_dbs], fontsize=9)
        ax.set_title(f"GPU Speedup (CPU time / GPU time)  |  fp_size={fp_size}", fontsize=12, fontweight="bold")

        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Speedup (×)", fontsize=10)

        # Annotate cells
        for i in range(len(n_dbs)):
            for j in range(len(phases)):
                v = speedup_matrix[i, j]
                text = f"{v:.2f}×" if v > 0 else "N/A"
                ax.text(j, i, text, ha="center", va="center", fontsize=8,
                        color="black" if 0.4 < v < 3 else "white")

        fig.tight_layout()
        fname = out_dir / f"speedup_heatmap_fp{fp_size}.png"
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        print(f"  Saved: {fname}")


# ── Plot 4: Absolute time-improvement heatmap ────────────────────────────────

def plot_abs_improvement_heatmap(df: pd.DataFrame, out_dir: Path) -> None:
    """For each fp_size, plot a heatmap of absolute GPU time saving in ms.

    Value = cpu_time - gpu_time.  Positive (blue) means GPU was faster;
    negative (red) means GPU was slower.  Colour scale is always centred at 0.
    """
    devices = df["device"].unique()
    gpu_devs = [d for d in devices if d.lower() in ("gpu", "metal")]
    if not gpu_devs:
        print("  No GPU device in data — skipping abs-improvement heatmap.")
        return

    gpu_dev = gpu_devs[0]
    cpu_dev = "cpu"
    fp_sizes = sorted(df["fp_size"].unique())
    phases = [p for p in PHASE_ORDER if p in df["phase"].values]

    for fp_size in fp_sizes:
        sub = df[df["fp_size"] == fp_size]
        n_dbs = sorted(sub["n_db"].unique())

        diff_matrix = np.zeros((len(n_dbs), len(phases)))
        for i, n_db in enumerate(n_dbs):
            for j, phase in enumerate(phases):
                cpu_row = sub[
                    (sub["device"] == cpu_dev) & (sub["n_db"] == n_db) & (sub["phase"] == phase)
                ]
                gpu_row = sub[
                    (sub["device"] == gpu_dev) & (sub["n_db"] == n_db) & (sub["phase"] == phase)
                ]
                if not cpu_row.empty and not gpu_row.empty:
                    cpu_t = float(cpu_row["mean_ms"].values[0])
                    gpu_t = float(gpu_row["mean_ms"].values[0])
                    diff_matrix[i, j] = cpu_t - gpu_t  # positive = GPU faster

        # Symmetric colour scale centred at 0
        abs_max = np.abs(diff_matrix).max()
        abs_max = abs_max if abs_max > 0 else 1.0  # avoid zero-range scale

        fig, ax = plt.subplots(figsize=(10, max(3, len(n_dbs) * 0.7 + 2)))
        im = ax.imshow(
            diff_matrix, aspect="auto", cmap="RdBu",
            vmin=-abs_max, vmax=abs_max,
        )
        ax.set_xticks(np.arange(len(phases)))
        ax.set_xticklabels(
            [STAGE_LABEL.get(p, p) for p in phases], rotation=30, ha="right", fontsize=9
        )
        ax.set_yticks(np.arange(len(n_dbs)))
        ax.set_yticklabels([f"n_db={n}" for n in n_dbs], fontsize=9)
        ax.set_title(
            f"GPU Absolute Time Saving (CPU − GPU, ms)  |  fp_size={fp_size}",
            fontsize=12, fontweight="bold",
        )

        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Time saving (ms)  [blue = GPU faster]", fontsize=10)

        # Annotate cells
        for i in range(len(n_dbs)):
            for j in range(len(phases)):
                v = diff_matrix[i, j]
                sign = "+" if v >= 0 else ""
                ax.text(
                    j, i, f"{sign}{v:.1f}",
                    ha="center", va="center", fontsize=8,
                    color="white" if abs(v) > 0.6 * abs_max else "black",
                )

        fig.tight_layout()
        fname = out_dir / f"abs_improvement_heatmap_fp{fp_size}.png"
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        print(f"  Saved: {fname}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot CPU vs GPU benchmark results.")
    p.add_argument("--csv", type=str,
                   default="benchmarking/CPU_GPU/results/cpu_gpu_benchmark.csv",
                   help="Input CSV from benchmark_cpu_gpu.py")
    p.add_argument("--out-dir", type=str,
                   default="benchmarking/CPU_GPU/results/plots",
                   help="Directory to save PNG plots")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    out_dir  = Path(args.out_dir)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV not found: {csv_path}\n"
            "Run benchmark_cpu_gpu.py first to generate results."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Reading: {csv_path}")
    df = load_and_aggregate(csv_path)

    print("\n── Stage bar charts ────────────────────────────────────────────────────")
    plot_stage_bars(df, out_dir)

    print("\n── Total time line charts ──────────────────────────────────────────────")
    plot_total_time(df, out_dir)

    print("\n── Speedup heatmap ─────────────────────────────────────────────────────")
    plot_speedup_heatmap(df, out_dir)

    print("\n── Absolute time-improvement heatmap ───────────────────────────────────")
    plot_abs_improvement_heatmap(df, out_dir)

    print(f"\nAll plots saved to: {out_dir}")


if __name__ == "__main__":
    main()
