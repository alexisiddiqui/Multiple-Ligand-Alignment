import pytest
from rdkit import Chem
import numpy as np
from typing import Dict, List

from ligand_neff.config import NeffConfig
from ligand_neff.compute import compute_neff

@pytest.fixture
def congeneric_library() -> List[Chem.Mol]:
    """Generates an 8-member congeneric series based on a benzamide core."""
    core_smiles = "O=C(Nc1cccc([*:1])c1)c1cc([*:2])cc([*:3])c1"
    
    r1_groups = ["F"]
    r2_groups = ["Cl", "Br"]
    r3_groups = ["C", "OC", "N(C)C", "C(F)(F)F"]
    
    mols = []
    for r1 in r1_groups:
        for r2 in r2_groups:
            for r3 in r3_groups:
                smi = core_smiles.replace("[*:1]", r1)
                smi = smi.replace("[*:2]", r2)
                smi = smi.replace("[*:3]", r3)
                
                mol = Chem.MolFromSmiles(smi)
                assert mol is not None, f"Failed to parse SMILES: {smi}"
                mols.append(mol)
                
    assert len(mols) == 8, f"Expected 8 library members, got {len(mols)}"
    return mols

def test_r_group_neff_scaling(congeneric_library: List[Chem.Mol]):
    """
    Validates that Neff scales correctly across a congeneric series:
    - High Neff for the conserved scaffold (present in 8/8)
    - Moderate Neff for R2 (present in 4/8)
    - Low Neff for R3 (present in 2/8)
    """
    
    # Configuration tuned for sensitivity to small structural changes
    config = NeffConfig(
        fp_radii=(1, 2, 3), 
        fp_size=4096,               # High precision
        tanimoto_inclusion=0.2,     # Allow inclusion of entire library as references
        max_references=100,         # Plenty for an 8-member library
        weighting="none",           # Disable inverse degree to observe pure frequency scaling
        coverage_metric="overlap",
        min_overlap=0.7,            # Strict overlap to only capture exact or near-exact matches
        aggregation="mean",         # Simple mean aggregation to track variations
        lambda_mode="fixed",        # Use fixed lambda to avoid adaptive scaling interference
        lambda_fixed=10.0           # Reasonable fixed lambda
    )
    
    from ligand_neff.compute import prepare_query_data
    results = []
    for query_mol in congeneric_library:
        query_data = prepare_query_data(query_mol, config)
        res = compute_neff(
            query_data=query_data,
            config=config,
            db_mols=congeneric_library,
            precomputed_db=None,
            query_mol=query_mol
        )
        results.append(res)
    
    assert len(results) == 8
    
    # We will analyze the first molecule in the library
    test_mol = congeneric_library[0]
    test_result = results[0]
    
    # We need to map atom indices to structural regions.
    # The first molecule is: O=C(Nc1cccc(F)c1)c1cc(Cl)cc(C)c1 
    # (R1 = F, R2 = Cl, R3 = C)
    
    # Define SMARTS patterns to find the specific regions
    # 1. Conserved Scaffold (Benzamide core without any R-groups)
    scaffold_smarts = "O=C(Nc1ccccc1)c1ccccc1"
    
    # Note: precise mapping of regions depends on atom indexing in the specific SMILES
    scaffold_pattern = Chem.MolFromSmarts("O=C(Nc1cccc(F)c1)c1cc(Cl)cc(C)c1")
    # Actually, let's just use simple SMARTS match to find the specific substitutent atoms
    
    # R1: The Fluoro group on the aniline ring
    r1_pattern = Chem.MolFromSmarts("Fc1cccc(NC=O)c1")
    
    # R2: The Chloro group on the benzoyl ring
    r2_pattern = Chem.MolFromSmarts("Clc1cc(C)cc(C(=O)N)c1")
    
    # R3: The Methyl group on the benzoyl ring
    r3_pattern = Chem.MolFromSmarts("Cc1cc(Cl)cc(C(=O)N)c1")
    
    # This might be tricky because overlapping SMARTS matches. 
    # A cleaner way is to track atoms by atomic number or specific patterns.
    
    # For O=C(Nc1cccc(F)c1)c1cc(Cl)cc(C)c1 :
    # F has atomic number 9
    # Cl has atomic number 17
    # C of the methyl group needs to be differentiated from the ring carbons
    # It's an aliphatic carbon (aromatic carbons are aromatic)
    
    atom_neffs = test_result.atom_neff
    
    f_idx = -1
    cl_idx = -1
    methyl_c_idx = -1
    scaffold_indices = []
    
    for atom in test_mol.GetAtoms():
        idx = atom.GetIdx()
        atomic_num = atom.GetAtomicNum()
        is_aromatic = atom.GetIsAromatic()
        
        if atomic_num == 9: # F
            f_idx = idx
        elif atomic_num == 17: # Cl
            cl_idx = idx
        elif atomic_num == 6 and not is_aromatic:
            # Check if it's the methyl (not the carbonyl C which is not aromatic but is C=O)
            # Carbonyl C has a double bond to O (atomic num 8)
            has_double_bond_to_o = any(
                bond.GetBondType() == Chem.BondType.DOUBLE and 
                bond.GetOtherAtom(atom).GetAtomicNum() == 8 
                for bond in atom.GetBonds()
            )
            if not has_double_bond_to_o:
                methyl_c_idx = idx
            else:
                scaffold_indices.append(idx)
        else:
            scaffold_indices.append(idx)
            
    assert f_idx != -1, "Fluorine not found"
    assert cl_idx != -1, "Chlorine not found"
    assert methyl_c_idx != -1, "Methyl carbon not found"
    assert len(scaffold_indices) > 0, "Scaffold atoms not found"
    
    # Calculate average Neff for each region
    scaffold_neff = float(np.mean([atom_neffs[i] for i in scaffold_indices]))
    r1_neff = float(atom_neffs[f_idx])
    r2_neff = float(atom_neffs[cl_idx])
    r3_neff = float(atom_neffs[methyl_c_idx])
    
    # We expect:
    # Scaffold and R1 (F) are present in 8/8 molecules. Their environments are highly conserved.
    # R2 (Cl) is present in 4/8 molecules.
    # R3 (C) is present in 2/8 molecules.
    
    print(f"Scaffold Neff: {scaffold_neff:.3f}")
    print(f"R1 (F, 8/8) Neff: {r1_neff:.3f}")
    print(f"R2 (Cl, 4/8) Neff: {r2_neff:.3f}")
    print(f"R3 (CH3, 2/8) Neff: {r3_neff:.3f}")
    
    # The core assertion: scaling by frequency
    # Note: R1 might be slightly lower than some deep scaffold atoms because its radius 3
    # environment captures the variation at R2 and R3.
    # But R1 > R2 > R3 should absolutely hold.
    
    assert r1_neff > r2_neff, f"R1 (fixed) Neff {r1_neff:.3f} should be > R2 (4/8) Neff {r2_neff:.3f}"
    assert r2_neff > r3_neff, f"R2 (4/8) Neff {r2_neff:.3f} should be > R3 (2/8) Neff {r3_neff:.3f}"
    
    # And Scaffold should be very high
    assert scaffold_neff > r2_neff, f"Scaffold Neff {scaffold_neff:.3f} should be > R2 Neff {r2_neff:.3f}"
