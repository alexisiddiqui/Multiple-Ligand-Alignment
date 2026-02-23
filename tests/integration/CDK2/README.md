2. Leave-one-series-out on PDB ligands
Take a target with many co-crystallized ligands (e.g., CDK2 has ~400 in the PDB). For each ligand, compute per-atom Neff using all other PDB ligands for that target as the reference set. 

Plot the global Neff / confidence for each ligand with mean tanimoto similarity between the query and the ligand 

1. Download the SMILES for all unique CDK2 ligands from ChEMBL - convert to SDF
2. Pre compute the reference database using ligand_neff.io.database.precompute_database
3. For each ligand, compute per-atom Neff using ligand_neff.compute.compute_neff
4. Plot the global Neff / confidence for each ligand with mean tanimoto similarity between the query and the ligand as the color of the point.