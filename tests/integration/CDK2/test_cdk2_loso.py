import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from ligand_neff.compute import compute_neff
from tqdm import tqdm

@pytest.mark.slow
def test_data_loaded(cdk2_mols):
    """Smoke test: ensure we have a reasonable amount of data."""
    assert len(cdk2_mols) >= 100
    print(f"Loaded {len(cdk2_mols)} CDK2 ligands.")

@pytest.mark.slow
def test_precomputed_db_shape(cdk2_precomputed, cdk2_config):
    """Verify the precomputed database matches the config."""
    for r in cdk2_config.fp_radii:
        key = f"radius_{r}"
        assert key in cdk2_precomputed
        assert cdk2_precomputed[key].shape[1] == cdk2_config.fp_size

@pytest.mark.slow
def test_leave_one_out_neff(cdk2_mols, cdk2_precomputed, cdk2_config, cdk2_data_dir):
    """
    Main LOSO loop. 
    For each molecule, we simulate its absence from the database.
    We do this by zeroing out its row in the precomputed fingerprints.
    """
    results = []
    
    # Limit to 100 for verification speed; set to None or 0 for full run
    max_iter = 100
    n_mols = min(len(cdk2_mols), max_iter) if max_iter else len(cdk2_mols)
    
    # Pre-allocate results list to store tuples
    processed_results = []
    
    print(f"Starting LOSO for {n_mols} molecules (max_iter={max_iter})...")
    

    for i in tqdm(range(n_mols), desc="LOSO Progress"):
        query = cdk2_mols[i]

        
        # Simulate leave-one-out by creating a local 'precomputed' dict 
        # where row i is zeroed out for all radii.
        from examples.LOSO.common.utils import create_loso_db
        loso_db = create_loso_db(cdk2_precomputed, cdk2_config.fp_radii, i)
            
        from ligand_neff.compute import prepare_query_data
        query_data = prepare_query_data(query, cdk2_config)
        
        # Run pipeline
        res = compute_neff(
            query_data=query_data,
            db_mols=None, # use precomputed
            config=cdk2_config,
            precomputed_db=loso_db,
            query_mol=query
        )
        
        processed_results.append({
            "idx": i,
            "id": query.GetProp("_Name") if query.HasProp("_Name") else f"mol_{i}",
            "global_neff": res.global_neff,
            "global_confidence": res.global_confidence,
            "n_refs": res.n_references_used
        })
        
        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{n_mols}...")

    # Assertions
    df = pd.DataFrame(processed_results)
    assert not df["global_confidence"].isna().any()
    assert (df["global_confidence"] >= 0).all()
    assert (df["global_confidence"] <= 1.0).all()
    
    # Save results for analysis
    out_path = cdk2_data_dir / "cdk2_results.csv"
    df.to_csv(out_path, index=False)
    print(f"Results saved to {out_path}")

@pytest.mark.slow
def test_neff_tanimoto_correlation(cdk2_data_dir):
    """
    Verify that global confidence correlates positively with mean Tanimoto similarity.
    This test runs AFTER test_leave_one_out_neff produces the CSV.
    """
    from examples.LOSO.common.utils import compute_mean_tanimotos, load_mols_from_smi
    from rdkit import Chem
    from ligand_neff.config import load_config
    
    results_path = cdk2_data_dir / "cdk2_results.csv"
    if not results_path.exists():
        pytest.fail("Results CSV missing - did test_leave_one_out_neff run?")
        
    df = pd.DataFrame(pd.read_csv(results_path))
    
    # Need to compute mean Tanimotos for the molecules
    config_path = cdk2_data_dir.parent.parent / "common" / "cdk2_config.yaml"
    config = load_config(config_path)
    
    smi_path = cdk2_data_dir / "cdk2_ligands.smi"
    mols = load_mols_from_smi(smi_path)
    
    # Only use molecules that appear in results (handles partial LOSO runs)
    if "idx" in df.columns:
        subset_mols = [mols[i] for i in df["idx"]]
    else:
        subset_mols = mols[:len(df)]
    
    mean_tanimotos = compute_mean_tanimotos(subset_mols, config)
    df["mean_tanimoto"] = mean_tanimotos
    
    # Calculate correlation
    corr = df["global_neff"].corr(df["mean_tanimoto"])
    print(f"Correlation (Confidence vs Mean Tanimoto): {corr:.4f}")
    
    # Higher Tanimoto should generally mean higher confidence
    assert corr > 0.1, f"Expected positive correlation between neff and similarity, got {corr:.4f}"
