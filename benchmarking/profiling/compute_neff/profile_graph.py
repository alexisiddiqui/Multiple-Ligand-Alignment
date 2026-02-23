#!/usr/bin/env python3
"""Profile the computational graph of the `compute_neff` pipeline.

Decomposes `compute_neff` into individually-timed stages:
    1. Fingerprint encoding   (CPU / RDKit)
    2. Atom decomposition     (CPU / RDKit)
    3. Reference filtering    (mixed: JAX bulk_tanimoto + NumPy padding)
    4. Inverse-degree weights  (JAX JIT, chunked fori_loop)
    5. Per-atom Neff           (JAX JIT)
    6. Aggregation             (JAX)
    7. Normalisation           (JAX)

Usage
-----
    # Basic timing (5 timed iterations, 500 DB molecules)
    uv run python benchmarking/profiling/compute_neff/profile_graph.py

    # Custom parameters
    uv run python benchmarking/profiling/compute_neff/profile_graph.py \
        --n-iters 10 --n-db 2000 --fp-size 4096 --max-refs 1024

    # Dump JAX HLO computational graphs
    uv run python benchmarking/profiling/compute_neff/profile_graph.py --dump-hlo

    # Generate a JAX profiler trace (open with Perfetto / TensorBoard)
    uv run python benchmarking/profiling/compute_neff/profile_graph.py \
        --profile-dir /tmp/jax_profile
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from rdkit import Chem

# ── Project imports ──────────────────────────────────────────────────────────
from ligand_neff.config import NeffConfig
from ligand_neff.fingerprints.encode import encode_molecule
from ligand_neff.fingerprints.decompose import decompose
from ligand_neff.similarity.filtering import filter_references
from ligand_neff.neff.weighting import inverse_degree_weights
from ligand_neff.neff.per_atom import per_atom_neff_single_radius
from ligand_neff.neff.aggregation import aggregate_neff, normalise_to_confidence


# ── Timing infrastructure ───────────────────────────────────────────────────

@dataclass
class StageResult:
    """Timing result for a single pipeline stage."""
    name: str
    times_ms: list[float] = field(default_factory=list)

    @property
    def mean_ms(self) -> float:
        return float(np.mean(self.times_ms))

    @property
    def std_ms(self) -> float:
        return float(np.std(self.times_ms)) if len(self.times_ms) > 1 else 0.0


def time_stage(name: str, fn: Callable, block_jax: bool = False) -> tuple[StageResult, object]:
    """Run a callable once for timing. Returns (StageResult with 1 sample, return value)."""
    t0 = time.perf_counter()
    result = fn()
    if block_jax and hasattr(result, "block_until_ready"):
        result.block_until_ready()
    elif block_jax and isinstance(result, (tuple, list)):
        for r in result:
            if hasattr(r, "block_until_ready"):
                r.block_until_ready()
    t1 = time.perf_counter()
    sr = StageResult(name)
    sr.times_ms.append((t1 - t0) * 1000)
    return sr, result


# ── Synthetic data generation ────────────────────────────────────────────────

# Real molecules for realistic atom counts & fingerprint patterns
QUERY_SMILES = [
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",          # caffeine (14 atoms)
    "CC(=O)OC1=CC=CC=C1C(=O)O",                # aspirin (13 atoms)
    "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",     # testosterone (23 atoms)
    "C1=CC=C(C=C1)C(=O)O",                      # benzoic acid (9 atoms)
]


def make_query_mol(smiles: str | None = None) -> Chem.Mol:
    """Return a real RDKit mol (defaults to caffeine)."""
    smiles = smiles or QUERY_SMILES[0]
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    return mol


def make_synthetic_db(n_db: int, fp_size: int, seed: int = 42) -> np.ndarray:
    """Generate random binary fingerprints as a database."""
    rng = np.random.default_rng(seed)
    # Sparse binary fps — ~10% density is realistic for Morgan FPs
    db = (rng.random((n_db, fp_size)) < 0.10).astype(np.float32)
    return db


# ── Profile runner ───────────────────────────────────────────────────────────

def run_single_iteration(
    query_mol: Chem.Mol,
    db_fps_per_radius: dict[int, np.ndarray],
    config: NeffConfig,
) -> dict[str, StageResult]:
    """Run one full pipeline iteration and return per-stage timings."""

    import chex
    chex.clear_trace_counter()

    stages: dict[str, StageResult] = {}
    neff_per_radius: dict[int, jnp.ndarray] = {}

    # Stage 0: Prepare Query Data
    from ligand_neff.compute import prepare_query_data
    s0, query_data = time_stage(
        "0_prepare_query_data",
        lambda: prepare_query_data(query_mol, config)
    )
    stages[s0.name] = s0

    for r in config.fp_radii:
        suffix = f"_r{r}"

        # Stage 1: Fingerprint encoding (already done in prepare_query_data, 
        # but we keep it for per-radius profiling)
        s1, q_fp = time_stage(
            f"1_fingerprint{suffix}",
            lambda r=r: query_data.fps[r],
        )
        stages[s1.name] = s1

        # Stage 2: Atom decomposition (already done in prepare_query_data)
        s2, atom_mask = time_stage(
            f"2_decompose{suffix}", 
            lambda r=r: jnp.array(query_data.atom_masks[r]), 
            block_jax=True
        )
        stages[s2.name] = s2

        # Stage 3: Filtering & padding
        db_fps = db_fps_per_radius[r]

        def _filter(q_fp=q_fp, db_fps=db_fps):
            return filter_references(
                query_fp=jnp.array(q_fp),
                db_fps=jnp.array(db_fps),
                threshold=config.tanimoto_inclusion,
                max_refs=config.max_references,
            )

        s3, filtered = time_stage(f"3_filter{suffix}", _filter, block_jax=True)
        stages[s3.name] = s3

        # Stage 4: Weighting
        if config.weighting == "inverse_degree":
            def _weight(filtered=filtered):
                return inverse_degree_weights(
                    fps=filtered.fps,
                    mask=filtered.mask,
                    threshold=config.cluster_threshold,
                    chunk_size=config.inverse_degree_chunk_size,
                )

            s4, weights = time_stage(f"4_weight{suffix}", _weight, block_jax=True)
        else:
            def _weight_none(filtered=filtered):
                return jnp.where(filtered.mask, 1.0, 0.0)

            s4, weights = time_stage(f"4_weight{suffix}", _weight_none, block_jax=True)
        stages[s4.name] = s4

        # Stage 5: Per-atom Neff
        def _per_atom(atom_mask=atom_mask, filtered=filtered, weights=weights):
            return per_atom_neff_single_radius(
                atom_bit_mask=atom_mask,
                ref_fps=filtered.fps,
                weights=weights,
                min_overlap=config.min_overlap,
                atom_norm=config.atom_norm,
            )

        s5, neff_r = time_stage(f"5_per_atom{suffix}", _per_atom, block_jax=True)
        stages[s5.name] = s5

        neff_per_radius[r] = neff_r

    # Stage 6: Aggregation (across radii)
    def _aggregate():
        return aggregate_neff(
            neff_per_radius,
            method=config.aggregation,
            radius_weights=config.radius_weights,
        )

    s6, combined_neff = time_stage("6_aggregate", _aggregate, block_jax=True)
    stages[s6.name] = s6

    # Stage 7: Normalisation
    if config.lambda_mode == "fixed":
        lam = config.lambda_fixed
    else:
        lam = float(jnp.maximum(jnp.quantile(combined_neff, config.lambda_quantile), 1e-3))

    def _normalise(combined_neff=combined_neff, lam=lam):
        return normalise_to_confidence(combined_neff, lam)

    s7, _ = time_stage("7_normalise", _normalise, block_jax=True)
    stages[s7.name] = s7

    return stages


def merge_stages(all_runs: list[dict[str, StageResult]]) -> list[StageResult]:
    """Merge per-stage timings across multiple iterations."""
    merged: dict[str, StageResult] = {}
    for run in all_runs:
        for name, sr in run.items():
            if name not in merged:
                merged[name] = StageResult(name)
            merged[name].times_ms.extend(sr.times_ms)
    # Return sorted by stage name so the output is ordered
    return sorted(merged.values(), key=lambda s: s.name)


def aggregate_by_phase(stages: list[StageResult]) -> list[StageResult]:
    """Roll up per-radius stages into phase-level summaries."""
    phases: dict[str, StageResult] = {}
    for s in stages:
        # Extract phase prefix: "1_fingerprint_r1" -> "1_fingerprint"
        parts = s.name.split("_r")
        phase = parts[0]  # e.g. "1_fingerprint"
        if phase not in phases:
            phases[phase] = StageResult(phase)
        # Sum per-radius times per iteration — simplify: just extend
        phases[phase].times_ms.extend(s.times_ms)
    return sorted(phases.values(), key=lambda s: s.name)


# ── Reporting ────────────────────────────────────────────────────────────────

def print_table(stages: list[StageResult], title: str = "Per-Stage Timing") -> None:
    """Pretty-print a timing table."""
    total_mean = sum(s.mean_ms for s in stages)

    print(f"\n{'=' * 72}")
    print(f"  {title}")
    print(f"{'=' * 72}")
    print(f"  {'Stage':<35} {'Mean (ms)':>10} {'Std (ms)':>10} {'% Total':>8}")
    print(f"  {'-' * 65}")

    for s in stages:
        pct = (s.mean_ms / total_mean * 100) if total_mean > 0 else 0
        print(f"  {s.name:<35} {s.mean_ms:>10.2f} {s.std_ms:>10.2f} {pct:>7.1f}%")

    print(f"  {'-' * 65}")
    print(f"  {'TOTAL':<35} {total_mean:>10.2f}")
    print(f"{'=' * 72}\n")


# ── HLO dump ─────────────────────────────────────────────────────────────────

def dump_hlo(config: NeffConfig, fp_size: int, max_refs: int, n_atoms: int) -> None:
    """Print the JAX HLO (jaxpr) for JIT'd functions in the pipeline."""
    print(f"\n{'=' * 72}")
    print("  JAX Computation Graphs (jaxpr)")
    print(f"{'=' * 72}")

    # Create abstract shapes
    ref_fps = jnp.zeros((max_refs, fp_size), dtype=jnp.float32)
    mask = jnp.ones(max_refs, dtype=bool)
    weights = jnp.ones(max_refs, dtype=jnp.float32)
    atom_mask = jnp.zeros((n_atoms, fp_size), dtype=jnp.float32)

    # bulk_tanimoto
    from ligand_neff.similarity.tanimoto import bulk_tanimoto
    query_fp = jnp.zeros(fp_size, dtype=jnp.float32)
    db_fps = jnp.zeros((100, fp_size), dtype=jnp.float32)

    print(f"\n── bulk_tanimoto ──")
    print(f"   Shapes: query ({fp_size},), db (100, {fp_size})")
    try:
        jaxpr = jax.make_jaxpr(bulk_tanimoto)(query_fp, db_fps)
        print(jaxpr)
    except Exception as e:
        print(f"   [Could not trace: {e}]")

    # per_atom_neff_single_radius  (unwrap jit to get the jaxpr of the inner fn)
    print(f"\n── per_atom_neff_single_radius ──")
    print(f"   Shapes: atom_mask ({n_atoms}, {fp_size}), ref_fps ({max_refs}, {fp_size}), weights ({max_refs},)")
    try:
        # The function is jit'd with static_argnames, so we need to trace the
        # underlying function directly
        inner = per_atom_neff_single_radius.__wrapped__ if hasattr(per_atom_neff_single_radius, "__wrapped__") else per_atom_neff_single_radius
        jaxpr = jax.make_jaxpr(
            lambda a, r, w: inner(a, r, w, min_overlap=config.min_overlap, atom_norm=config.atom_norm)
        )(atom_mask, ref_fps, weights)
        print(jaxpr)
    except Exception as e:
        print(f"   [Could not trace: {e}]")

    # aggregate_neff
    print(f"\n── aggregate_neff ──")
    neff_dict = {r: jnp.zeros(n_atoms, dtype=jnp.float32) for r in config.fp_radii}
    try:
        jaxpr = jax.make_jaxpr(
            lambda d: aggregate_neff(d, method=config.aggregation, radius_weights=config.radius_weights)
        )(neff_dict)
        print(jaxpr)
    except Exception as e:
        print(f"   [Could not trace: {e}]")

    # normalise_to_confidence
    print(f"\n── normalise_to_confidence ──")
    neff_arr = jnp.zeros(n_atoms, dtype=jnp.float32)
    try:
        jaxpr = jax.make_jaxpr(lambda n: normalise_to_confidence(n, 10.0))(neff_arr)
        print(jaxpr)
    except Exception as e:
        print(f"   [Could not trace: {e}]")

    print(f"\n{'=' * 72}\n")


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Profile the compute_neff computational graph.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--n-iters", type=int, default=5,
                    help="Number of timed iterations (after warm-up). Default: 5")
    p.add_argument("--n-db", type=int, default=500,
                    help="Number of synthetic database molecules. Default: 500")
    p.add_argument("--fp-size", type=int, default=2048, choices=[2048, 4096, 8192],
                    help="Fingerprint bit-vector length. Default: 2048")
    p.add_argument("--max-refs", type=int, default=512,
                    help="Max reference slots (static JAX shape). Default: 512")
    p.add_argument("--chunk-size", type=int, default=128,
                    help="Chunk size for inverse-degree weighting. Default: 128")
    p.add_argument("--query-smiles", type=str, default=None,
                    help=f"SMILES for the query molecule. Default: caffeine")
    p.add_argument("--dump-hlo", action="store_true",
                    help="Print JAX computation graphs (jaxpr) for JIT'd functions.")
    p.add_argument("--profile-dir", type=str, default=None,
                    help="Directory for JAX profiler trace output (Perfetto/TensorBoard).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Config ───────────────────────────────────────────────────────────
    config = NeffConfig(
        fp_radii=(1, 2, 3),
        fp_size=args.fp_size,
        max_references=args.max_refs,
        inverse_degree_chunk_size=args.chunk_size,
        weighting="inverse_degree",
    )

    query_mol = make_query_mol(args.query_smiles)
    n_atoms = query_mol.GetNumAtoms()

    print(f"  Query: {Chem.MolToSmiles(Chem.RemoveHs(query_mol))}  ({n_atoms} atoms incl. H)")
    print(f"  DB size: {args.n_db}")
    print(f"  Config: fp_size={config.fp_size}, max_refs={config.max_references}, "
          f"chunk_size={config.inverse_degree_chunk_size}")
    print(f"  Radii: {config.fp_radii}")
    print(f"  Iterations: 1 warm-up + {args.n_iters} timed")

    # ── Precompute DB fingerprints (not part of profiled pipeline) ────
    print("\n  Generating synthetic database fingerprints …")
    db_fps_per_radius: dict[int, np.ndarray] = {}
    for r in config.fp_radii:
        db_fps_per_radius[r] = make_synthetic_db(args.n_db, config.fp_size, seed=42 + r)

    # ── Warm-up run (includes JIT compilation) ───────────────────────
    print("  Running warm-up iteration (JIT compilation) …")
    t_warmup_start = time.perf_counter()
    warmup_stages = run_single_iteration(query_mol, db_fps_per_radius, config)
    t_warmup = (time.perf_counter() - t_warmup_start) * 1000
    print(f"  Warm-up complete: {t_warmup:.0f} ms (includes compilation)\n")

    print_table(
        sorted(warmup_stages.values(), key=lambda s: s.name),
        title="Warm-Up (includes JIT compilation)",
    )

    # ── Optional: JAX profiler trace ─────────────────────────────────
    profiler_ctx = None
    if args.profile_dir:
        profile_dir = Path(args.profile_dir)
        profile_dir.mkdir(parents=True, exist_ok=True)
        print(f"  JAX profiler trace will be saved to: {profile_dir}")
        jax.profiler.start_trace(str(profile_dir))

    # ── Timed iterations ─────────────────────────────────────────────
    print(f"  Running {args.n_iters} timed iterations …")
    all_runs: list[dict[str, StageResult]] = []
    for i in range(args.n_iters):
        stages = run_single_iteration(query_mol, db_fps_per_radius, config)
        all_runs.append(stages)
        iter_total = sum(s.mean_ms for s in stages.values())
        print(f"    Iteration {i + 1}/{args.n_iters}: {iter_total:.1f} ms")

    if args.profile_dir:
        jax.profiler.stop_trace()
        print(f"  Profiler trace saved to: {args.profile_dir}")

    # ── Results ──────────────────────────────────────────────────────
    merged = merge_stages(all_runs)
    print_table(merged, title=f"Detailed Per-Stage Timing ({args.n_iters} iterations)")

    phase_summary = aggregate_by_phase(merged)
    print_table(phase_summary, title="Phase Summary (aggregated across radii)")

    # ── Optional: HLO dump ───────────────────────────────────────────
    if args.dump_hlo:
        dump_hlo(config, args.fp_size, args.max_refs, n_atoms)

    print("Done.")


if __name__ == "__main__":
    main()
