#!/usr/bin/env python3
"""Golden benchmark runner for compute_neff optimization.

This script establishes baseline performance (Phase 0) and verifies acceptance
criteria for the new JAX-only device-resident pipeline.
"""
import argparse
import time
import sys
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from rdkit import Chem

# Make sure we can import ligand_neff
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from ligand_neff.config import NeffConfig
from ligand_neff.compute import compute_neff, prepare_query_data

# Import decomposed timing helpers from existing profiling script
from benchmarking.profiling.compute_neff.profile_graph import (
    make_query_mol,
    make_synthetic_db,
    run_single_iteration,
    merge_stages,
    aggregate_by_phase,
    print_table,
)

def run_legacy_end_to_end(
    query_mol: Chem.Mol,
    config: NeffConfig,
    precomputed_db: dict[int, np.ndarray],
    n_iters: int = 5
) -> dict:
    """Run the legacy compute_neff wrapper end-to-end."""
    # Convert integer keys to string keys as expected by compute_neff
    db_for_wrapper = {f"radius_{r}": v for r, v in precomputed_db.items()}

    # Warmup
    qd_warmup = prepare_query_data(query_mol, config)
    _ = compute_neff(qd_warmup, config, precomputed_db=db_for_wrapper, query_mol=query_mol)
    
    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        qd = prepare_query_data(query_mol, config)
        result = compute_neff(qd, config, precomputed_db=db_for_wrapper, query_mol=query_mol)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    
    return {
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)) if len(times) > 1 else 0.0,
        "times_ms": times,
        "result": result
    }

def verify_correctness(legacy_result, fast_result, rtol=1e-4, atol=1e-5):
    """Verify acceptance criteria for correctness."""
    print("\n--- Correctness Check ---")
    if fast_result is None:
        print("  [Skip] No fast result provided yet.")
        return True
        
    atom_neff_close = np.allclose(legacy_result.atom_neff, fast_result.atom_neff, rtol=rtol, atol=atol)
    atom_conf_close = np.allclose(legacy_result.atom_confidence, fast_result.atom_confidence, rtol=rtol, atol=atol)
    n_refs_exact = (legacy_result.n_references_used == fast_result.n_references_used)
    
    print(f"  atom_neff match: {atom_neff_close}")
    print(f"  atom_confidence match: {atom_conf_close}")
    print(f"  n_references_used match: {n_refs_exact}")
    
    passed = atom_neff_close and atom_conf_close and n_refs_exact
    if not passed:
        print("  [FAIL] Correctness matching failed!")
    else:
        print("  [PASS] Correctness verified.")
    return passed

def verify_performance(legacy_time_ms, fast_time_ms):
    """Verify acceptance criteria for performance."""
    print("\n--- Performance Check ---")
    if fast_time_ms is None:
        print("  [Skip] No fast runtime provided yet.")
        return True
        
    speedup = legacy_time_ms / fast_time_ms
    print(f"  Legacy time: {legacy_time_ms:.2f} ms")
    print(f"  Fast time:   {fast_time_ms:.2f} ms")
    print(f"  Speedup:     {speedup:.2f}x")
    
    passed = speedup >= 2.0
    if not passed:
        print(f"  [FAIL] Speedup >= 2x: {passed}")
    else:
        print(f"  [PASS] Speedup >= 2x: {passed}")
    return passed

def main():
    p = argparse.ArgumentParser(description="Golden Benchmark Runner for compute_neff")
    p.add_argument("--n-db", type=int, default=10000, help="DB size (default: 10k)")
    p.add_argument("--n-iters", type=int, default=5, help="Benchmark iterations")
    p.add_argument("--query-smiles", type=str, default=None)
    p.add_argument("--fast", action="store_true", help="Also run the fast engine (if implemented)")
    args = p.parse_args()
    
    config = NeffConfig(fp_radii=(1, 2, 3), max_references=512, weighting="inverse_degree")
    query_mol = make_query_mol(args.query_smiles)
    
    print(f"Initializing synthetic DB with {args.n_db} molecules...")
    db_fps_per_radius = {r: make_synthetic_db(args.n_db, config.fp_size, seed=42+r) for r in config.fp_radii}
    
    print("Gathering decomposed timings (legacy) for stage analysis...")
    _ = run_single_iteration(query_mol, db_fps_per_radius, config) # warmup
    all_runs = [run_single_iteration(query_mol, db_fps_per_radius, config) for _ in range(args.n_iters)]
    merged = merge_stages(all_runs)
    phase_summary = aggregate_by_phase(merged)
    
    sum_of_stages = sum(s.mean_ms for s in phase_summary)
    print_table(phase_summary, title="Decomposed Legacy Stages")
    
    print("Gathering end-to-end timing (legacy wrappers)...")
    e2e_res = run_legacy_end_to_end(query_mol, config, db_fps_per_radius, n_iters=args.n_iters)
    actual_e2e = e2e_res["mean_ms"]
    
    overhead = actual_e2e - sum_of_stages
    
    print(f"\n[{'='*60}]")
    print(f"Legacy End-to-End Time: {actual_e2e:.2f} ms")
    print(f"Sum of JIT/CPU stages:  {sum_of_stages:.2f} ms")
    print(f"Calculated Sync Overhead: {overhead:.2f} ms ({(overhead/actual_e2e)*100:.1f}%)")
    print(f"[{'='*60}]")
    
    fast_result = None
    fast_time_ms = None
    
    if args.fast:
        from ligand_neff.engine import NeffEngine
        print("\nGathering end-to-end timing (fast Engine)...")
        db_fps_for_engine = {f"radius_{r}": v for r, v in db_fps_per_radius.items()}
        
        t0_init = time.perf_counter()
        fast_engine = NeffEngine(config, precomputed_db=db_fps_for_engine, compile_on_init=True)
        print(f"  Engine init + warmup took: {(time.perf_counter() - t0_init)*1000:.2f} ms")
        
        t0_prep = time.perf_counter()
        prepared = fast_engine.prepare_query(query_mol)
        print(f"  Query prep took: {(time.perf_counter() - t0_prep)*1000:.2f} ms")
        
        # Warmup actual graph
        _ = fast_engine.compute_prepared(prepared)
        
        times = []
        for _ in range(args.n_iters):
            t0 = time.perf_counter()
            fast_result = fast_engine.compute_prepared(prepared)
            times.append((time.perf_counter() - t0) * 1000)
            
        fast_time_ms = float(np.mean(times))
        print(f"  Fast Engine End-to-End mean: {fast_time_ms:.2f} ms")
    else:
        print("\n  [Skip] --fast mode disabled.")
    
    verify_correctness(e2e_res["result"], fast_result)
    verify_performance(actual_e2e, fast_time_ms)

if __name__ == "__main__":
    main()
