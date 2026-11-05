"""
Performs a sensitivity analysis similar to 02_compute_cross_neff.py for testosterone and progesterone against the angrogen and the progesterone receptor, respectively using the ligands dataset generated in 04_filter_PDB_ligands.py using a 0.9 tanimoto cutoff. Uses the base config.
These are compared against the global_neff scores of the other ligands.
Plots the per-atom neff scores onto the ligand structures for each 2x2 cognate-off target receptors.
Plots violin plots of the per atom neff scores.
Plots whether the global_neff correlates with activity.


"""
