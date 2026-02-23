#!/usr/bin/env python3
"""Compare CPU vs GPU (Metal) runtimes for the `compute_neff` pipeline.

Sweeps over a grid of (n_db, fp_size, max_refs, chunk_size) and runs the full
pipeline on each available JAX device, collecting per-stage wall-clock timings.

Usage
-----
    # Default sweep (CPU + GPU if available)
    uv run python benchmarking/CPU_GPU/benchmark_cpu_gpu.py

    # Custom sweep
    uv run python benchmarking/CPU_GPU/benchmark_cpu_gpu.py \\
        --n-db 100,500,1000 --fp-size 2048,4096 --n-iters 5 \\
        --out-csv benchmarking/CPU_GPU/results/my_run.csv

    # CPU only (skip GPU even if available)
    uv run python benchmarking/CPU_GPU/benchmark_cpu_gpu.py --cpu-only
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
import warnings
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from rdkit import Chem

# ── Project imports ───────────────────────────────────────────────────────────
from ligand_neff.config import NeffConfig
from ligand_neff.fingerprints.encode import encode_molecule
from ligand_neff.fingerprints.decompose import decompose
from ligand_neff.similarity.filtering import filter_references
from ligand_neff.neff.weighting import inverse_degree_weights
from ligand_neff.neff.per_atom import per_atom_neff_single_radius
from ligand_neff.neff.aggregation import aggregate_neff, normalise_to_confidence


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class StageResult:
    name: str
    times_ms: list[float] = field(default_factory=list)

    @property
    def mean_ms(self) -> float:
        return float(np.mean(self.times_ms)) if self.times_ms else 0.0

    @property
    def std_ms(self) -> float:
        return float(np.std(self.times_ms)) if len(self.times_ms) > 1 else 0.0


@dataclass
class BenchmarkRow:
    device: str
    n_db: int
    fp_size: int
    max_refs: int
    chunk_size: int
    stage: str
    mean_ms: float
    std_ms: float
    n_iters: int


# ── Timing helper ─────────────────────────────────────────────────────────────

def time_stage(name: str, fn: Callable, block_jax: bool = True) -> tuple[StageResult, object]:
    t0 = time.perf_counter()
    result = fn()
    if block_jax:
        if hasattr(result, "block_until_ready"):
            result.block_until_ready()
        elif isinstance(result, (tuple, list)):
            for r in result:
                if hasattr(r, "block_until_ready"):
                    r.block_until_ready()
    t1 = time.perf_counter()
    sr = StageResult(name)
    sr.times_ms.append((t1 - t0) * 1000)
    return sr, result


# ── Synthetic data ────────────────────────────────────────────────────────────

QUERY_SMILES = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"  # caffeine

def make_query_mol() -> Chem.Mol:
    mol = Chem.MolFromSmiles(QUERY_SMILES)
    return Chem.AddHs(mol)


def make_synthetic_db(n_db: int, fp_size: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.random((n_db, fp_size)) < 0.10).astype(np.float32)


# ── Pipeline runner ───────────────────────────────────────────────────────────

def run_single_iteration(
    query_mol: Chem.Mol,
    db_fps_per_radius: dict[int, jnp.ndarray],   # already on the current device
    config: NeffConfig,
) -> dict[str, StageResult]:
    """Run one full iteration, return per-stage timings."""
    stages: dict[str, StageResult] = {}
    neff_per_radius: dict[int, jnp.ndarray] = {}

    # 0. Prepare Query Data (CPU)
    from ligand_neff.compute import prepare_query_data
    s0, query_data = time_stage(
        "0_prepare_query_data",
        lambda: prepare_query_data(query_mol, config)
    )
    stages[s0.name] = s0

    for r in config.fp_radii:
        suffix = f"_r{r}"

        # 1. Fingerprint (from precomputed data)
        s1, q_fp = time_stage(
            f"1_fingerprint{suffix}",
            lambda r=r: jnp.array(query_data.fps[r])
        )
        stages[s1.name] = s1

        # 2. Atom decomposition (from precomputed data)
        s2, atom_mask = time_stage(
            f"2_decompose{suffix}", 
            lambda r=r: jnp.array(query_data.atom_masks[r])
        )
        stages[s2.name] = s2

        # 3. Filter references (JAX)
        db_fps = db_fps_per_radius[r]

        def _filter(q_fp=q_fp, db_fps=db_fps):
            return filter_references(
                query_fp=q_fp,
                db_fps=db_fps,
                threshold=config.tanimoto_inclusion,
                max_refs=config.max_references,
            )

        s3, filtered = time_stage(f"3_filter{suffix}", _filter)
        stages[s3.name] = s3

        # 4. Weighting (JAX)
        def _weight(filtered=filtered):
            return inverse_degree_weights(
                fps=filtered.fps,
                mask=filtered.mask,
                threshold=config.cluster_threshold,
                chunk_size=config.inverse_degree_chunk_size,
            )

        s4, weights = time_stage(f"4_weight{suffix}", _weight)
        stages[s4.name] = s4

        # 5. Per-atom Neff (JAX)
        def _per_atom(atom_mask=atom_mask, filtered=filtered, weights=weights):
            return per_atom_neff_single_radius(
                atom_bit_mask=atom_mask,
                ref_fps=filtered.fps,
                weights=weights,
                min_overlap=config.min_overlap,
                atom_norm=config.atom_norm,
            )

        s5, neff_r = time_stage(f"5_per_atom{suffix}", _per_atom)
        stages[s5.name] = s5
        neff_per_radius[r] = neff_r

    # 6. Aggregation (JAX)
    def _aggregate():
        return aggregate_neff(neff_per_radius, method=config.aggregation, radius_weights=config.radius_weights)

    s6, combined_neff = time_stage("6_aggregate", _aggregate)
    stages[s6.name] = s6

    # 7. Normalisation (JAX)
    lam = (
        config.lambda_fixed
        if config.lambda_mode == "fixed"
        else float(jnp.maximum(jnp.quantile(combined_neff, config.lambda_quantile), 1e-3))
    )

    def _normalise(combined_neff=combined_neff, lam=lam):
        return normalise_to_confidence(combined_neff, lam)

    s7, _ = time_stage("7_normalise", _normalise)
    stages[s7.name] = s7

    return stages


def merge_stages(all_runs: list[dict[str, StageResult]]) -> list[StageResult]:
    merged: dict[str, StageResult] = {}
    for run in all_runs:
        for name, sr in run.items():
            if name not in merged:
                merged[name] = StageResult(name)
            merged[name].times_ms.extend(sr.times_ms)
    return sorted(merged.values(), key=lambda s: s.name)


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_table(stages: list[StageResult], title: str, device: str, config_label: str) -> None:
    total = sum(s.mean_ms for s in stages)
    header = f"  [{device.upper()}] {title}  |  {config_label}"
    print(f"\n{'=' * 76}")
    print(header)
    print(f"{'=' * 76}")
    print(f"  {'Stage':<38} {'Mean (ms)':>10} {'Std (ms)':>10} {'% Total':>8}")
    print(f"  {'-' * 68}")
    for s in stages:
        pct = (s.mean_ms / total * 100) if total > 0 else 0
        print(f"  {s.name:<38} {s.mean_ms:>10.2f} {s.std_ms:>10.2f} {pct:>7.1f}%")
    print(f"  {'-' * 68}")
    print(f"  {'TOTAL':<38} {total:>10.2f}")
    print(f"{'=' * 76}\n")


def print_comparison_table(
    cpu_stages: list[StageResult],
    gpu_stages: list[StageResult],
    config_label: str,
) -> None:
    # Align by stage name
    gpu_map = {s.name: s for s in gpu_stages}
    total_cpu = sum(s.mean_ms for s in cpu_stages)
    total_gpu = sum(gpu_map[s.name].mean_ms for s in cpu_stages if s.name in gpu_map)

    print(f"\n{'=' * 88}")
    print(f"  CPU vs GPU Comparison  |  {config_label}")
    print(f"{'=' * 88}")
    print(f"  {'Stage':<38} {'CPU (ms)':>10} {'GPU (ms)':>10} {'Speedup':>9}")
    print(f"  {'-' * 70}")
    for s in cpu_stages:
        g = gpu_map.get(s.name)
        if g and g.mean_ms > 0:
            speedup = s.mean_ms / g.mean_ms
            print(f"  {s.name:<38} {s.mean_ms:>10.2f} {g.mean_ms:>10.2f} {speedup:>8.2f}x")
        else:
            print(f"  {s.name:<38} {s.mean_ms:>10.2f} {'N/A':>10}")
    print(f"  {'-' * 70}")
    total_speedup = total_cpu / total_gpu if total_gpu > 0 else float("nan")
    print(f"  {'TOTAL':<38} {total_cpu:>10.2f} {total_gpu:>10.2f} {total_speedup:>8.2f}x")
    print(f"{'=' * 88}\n")


# ── Device detection ──────────────────────────────────────────────────────────

def detect_devices(cpu_only: bool) -> list[jax.Device]:
    """Return a list of JAX devices to benchmark on.

    JAX 0.5.0 on macOS sets Metal as the *default* backend, meaning:
    - `jax.devices()` → Metal devices only
    - `jax.devices('metal')` → [] (empty — already selected as default)
    - `jax.devices('cpu')` → CPU devices

    Strategy:
    1. Always grab a CPU device via jax.devices('cpu').
    2. If not cpu_only, look at the default backend: if it's Metal/GPU use
       it directly via `jax.devices()`.  Otherwise try explicit probes.
    """
    selected: list[jax.Device] = []

    # Always try to get a CPU device
    try:
        cpu_devs = jax.devices("cpu")
        if cpu_devs:
            selected.append(cpu_devs[0])
    except RuntimeError:
        pass

    if not cpu_only:
        # Check default backend first (handles Metal on macOS)
        default_devs = jax.devices()
        gpu_from_default = [d for d in default_devs if d.platform.lower() in ("gpu", "metal")]
        if gpu_from_default:
            selected.append(gpu_from_default[0])
        else:
            # Explicit probes for CUDA etc.
            for backend in ("gpu",):
                try:
                    gpu_devs = jax.devices(backend)
                    if gpu_devs:
                        selected.append(gpu_devs[0])
                        break
                except RuntimeError:
                    continue

    return selected


# ── Main sweep ────────────────────────────────────────────────────────────────

def run_benchmark(
    devices: list[jax.Device],
    n_db_list: list[int],
    fp_size_list: list[int],
    max_refs: int,
    chunk_size: int,
    n_iters: int,
    query_mol: Chem.Mol,
) -> list[BenchmarkRow]:
    rows: list[BenchmarkRow] = []

    for n_db, fp_size in product(n_db_list, fp_size_list):
        config = NeffConfig(
            fp_radii=(1, 2, 3),
            fp_size=fp_size,
            max_references=max_refs,
            inverse_degree_chunk_size=chunk_size,
            weighting="inverse_degree",
        )
        config_label = f"n_db={n_db}, fp_size={fp_size}, max_refs={max_refs}, chunk={chunk_size}"
        print(f"\n{'─' * 76}")
        print(f"  Config: {config_label}")
        print(f"{'─' * 76}")

        # Build DB fingerprints in CPU numpy memory (shared across devices)
        db_fps_np: dict[int, np.ndarray] = {
            r: make_synthetic_db(n_db, fp_size, seed=42 + r)
            for r in config.fp_radii
        }

        device_stage_results: dict[str, list[StageResult]] = {}

        for device in devices:
            dev_name = f"{device.platform}"
            print(f"\n  ▶ Device: {device} ({dev_name.upper()})")

            with jax.default_device(device):
                # Move DB fps to this device once
                db_fps_jax: dict[int, jnp.ndarray] = {
                    r: jnp.array(fp) for r, fp in db_fps_np.items()
                }

                # Warm-up (JIT compilation)
                print(f"    Warming up (JIT)…", end="", flush=True)
                t0 = time.perf_counter()
                _ = run_single_iteration(query_mol, db_fps_jax, config)
                warmup_ms = (time.perf_counter() - t0) * 1000
                print(f" {warmup_ms:.0f} ms")

                # Timed iterations
                all_runs: list[dict[str, StageResult]] = []
                for i in range(n_iters):
                    stages = run_single_iteration(query_mol, db_fps_jax, config)
                    all_runs.append(stages)
                    iter_total = sum(s.mean_ms for s in stages.values())
                    print(f"    Iter {i + 1}/{n_iters}: {iter_total:.1f} ms")

            merged = merge_stages(all_runs)
            print_table(merged, title="Per-Stage Timing", device=dev_name, config_label=config_label)
            device_stage_results[dev_name] = merged

            # Collect rows for CSV
            for s in merged:
                rows.append(BenchmarkRow(
                    device=dev_name,
                    n_db=n_db,
                    fp_size=fp_size,
                    max_refs=max_refs,
                    chunk_size=chunk_size,
                    stage=s.name,
                    mean_ms=s.mean_ms,
                    std_ms=s.std_ms,
                    n_iters=n_iters,
                ))

        # Cross-device comparison if both CPU and GPU ran
        if "cpu" in device_stage_results and len(device_stage_results) > 1:
            gpu_key = [k for k in device_stage_results if k != "cpu"][0]
            print_comparison_table(
                device_stage_results["cpu"],
                device_stage_results[gpu_key],
                config_label,
            )

    return rows


# ── CSV output ────────────────────────────────────────────────────────────────

def save_csv(rows: list[BenchmarkRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["device", "n_db", "fp_size", "max_refs", "chunk_size",
                         "stage", "mean_ms", "std_ms", "n_iters"])
        for r in rows:
            writer.writerow([r.device, r.n_db, r.fp_size, r.max_refs, r.chunk_size,
                              r.stage, f"{r.mean_ms:.4f}", f"{r.std_ms:.4f}", r.n_iters])
    print(f"  Results saved to: {path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark compute_neff pipeline on CPU vs GPU (Metal).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--n-db", type=str, default="1000,5000",
                   help="Comma-separated DB sizes to sweep. Default: 100,500")
    p.add_argument("--fp-size", type=str, default="2048,4096,8192",
                   help="Comma-separated fingerprint sizes (2048|4096|8192). Default: 2048")
    p.add_argument("--max-refs", type=int, default=1000,
                   help="Max reference slots (static JAX shape). Default: 512")
    p.add_argument("--chunk-size", type=int, default=256,
                   help="Chunk size for inverse-degree weighting. Default: 256")
    p.add_argument("--n-iters", type=int, default=5,
                   help="Number of timed iterations per config. Default: 5")
    p.add_argument("--cpu-only", action="store_true",
                   help="Skip GPU even if available.")
    p.add_argument("--out-csv", type=str,
                   default="benchmarking/CPU_GPU/results/cpu_gpu_benchmark.csv",
                   help="Output CSV path.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    n_db_list  = [int(x) for x in args.n_db.split(",")]
    fp_size_list = [int(x) for x in args.fp_size.split(",")]

    devices = detect_devices(args.cpu_only)
    if not devices:
        print("ERROR: No JAX devices found.", file=sys.stderr)
        sys.exit(1)

    print("=" * 76)
    print("  ligand-neff  —  CPU vs GPU Benchmark")
    print("=" * 76)
    print(f"  Devices      : {[str(d) for d in devices]}")
    print(f"  DB sizes     : {n_db_list}")
    print(f"  FP sizes     : {fp_size_list}")
    print(f"  max_refs     : {args.max_refs}")
    print(f"  chunk_size   : {args.chunk_size}")
    print(f"  Iterations   : 1 warm-up + {args.n_iters} timed")

    query_mol = make_query_mol()
    print(f"  Query        : {Chem.MolToSmiles(Chem.RemoveHs(query_mol))}  "
          f"({query_mol.GetNumAtoms()} atoms incl. H)")

    rows = run_benchmark(
        devices=devices,
        n_db_list=n_db_list,
        fp_size_list=fp_size_list,
        max_refs=args.max_refs,
        chunk_size=args.chunk_size,
        n_iters=args.n_iters,
        query_mol=query_mol,
    )

    save_csv(rows, Path(args.out_csv))
    print("\nDone.")


if __name__ == "__main__":
    main()
