"""Shared type aliases used across the package."""

import jax.numpy as jnp
from jaxtyping import Float, Bool, Int, UInt8, Array
import chex

# ── Core array types ──────────────────────────────────────────────
# Fingerprints are stored on disk/CPU as uint8 (0 or 1 per bit position).
# They are cast to float32 inside JAX kernels that use dot-products.
Fingerprint      = UInt8[Array, "fp_size"]       # storage dtype
FingerprintF32   = Float[Array, "fp_size"]       # computation dtype (cast inside kernels)
FingerprintBatch = UInt8[Array, "batch fp_size"] # storage dtype

import numpy as np
# Per-atom structures
AtomBitMask      = Float[Array, "n_atoms fp_size"]  # always float32 (used in dots)
# We use numpy arrays for the final result object
AtomScores       = np.ndarray

# Padded reference arrays (static shape from config.max_references)
# Stored as uint8 in the database, converted to float32 for JAX ops.
PaddedRefs       = Float[Array, "max_refs fp_size"]  # float32 inside JIT
RefMask          = Bool[Array, "max_refs"]
RefWeights       = Float[Array, "max_refs"]
RefSimilarities  = Float[Array, "max_refs"]

# Similarity matrices
SimMatrix        = Float[Array, "rows cols"]

# Scalars
Scalar           = Float[Array, ""]
IntScalar        = Int[Array, ""]   # e.g. n_valid stored as JAX int32


@chex.dataclass(frozen=True)
class QueryData:
    """
    Precomputed fingerprints and atom masks for a query molecule.
    Using dicts for radii to keep it flexible, but could be a stacked tensor.
    """
    fps: dict[int, np.ndarray]        # radius -> fingerprint (1D)
    atom_masks: dict[int, np.ndarray] # radius -> atom mask (2D: n_atoms x fp_size)
    n_atoms: int


@chex.dataclass(frozen=True, mappable_dataclass=False)
class NeffResult:
    """
    Immutable output of a Neff computation.

    frozen=True because results should not be mutated after creation.
    mappable_dataclass=False because this is a user-facing result object,
    not something we pass through jax.tree_util — it contains RDKit Mol
    objects which are not JAX-compatible leaves.
    """
    query_mol: object              # Chem.Mol — not a JAX type
    config: object                 # NeffConfig
    atom_neff: AtomScores          # (n_atoms,)
    atom_confidence: AtomScores    # (n_atoms,) in [0, 1]
    neff_per_radius: dict          # {int: AtomScores}
    global_neff: float
    global_confidence: float
    n_references_used: int
    lambda_value: float

    def to_sdf(self, path: str) -> None:
        """Write query mol with Neff/confidence as atom properties."""
        from rdkit import Chem
        mol = Chem.Mol(self.query_mol) # copy
        
        # We store the arrays as formatted strings so external tools can read them
        mol.SetProp("MLA_Neff_Global", f"{self.global_neff:.4f}")
        mol.SetProp("MLA_Confidence_Global", f"{self.global_confidence:.4f}")
        mol.SetProp("MLA_Lambda", f"{self.lambda_value:.4f}")
        
        # Atom properties usually space-separated or comma-separated strings
        neff_str = " ".join([f"{x:.4f}" for x in self.atom_neff])
        conf_str = " ".join([f"{x:.4f}" for x in self.atom_confidence])
        
        mol.SetProp("MLA_Neff_Per_Atom", neff_str)
        mol.SetProp("MLA_Confidence_Per_Atom", conf_str)
        
        writer = Chem.SDWriter(str(path))
        writer.write(mol)
        writer.close()

    def to_csv(self, path: str) -> None:
        """Per-atom table: idx, element, neff_r1, ..., combined, confidence."""
        import csv
        
        radii = sorted(self.neff_per_radius.keys())
        headers = ["Atom_Idx", "Element"] + [f"Neff_R{r}" for r in radii] + ["Neff_Combined", "Confidence"]
        
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            
            for i in range(self.query_mol.GetNumAtoms()):
                atom = self.query_mol.GetAtomWithIdx(i)
                row = [i, atom.GetSymbol()]
                
                for r in radii:
                    row.append(f"{self.neff_per_radius[r][i]:.4f}")
                    
                row.append(f"{self.atom_neff[i]:.4f}")
                row.append(f"{self.atom_confidence[i]:.4f}")
                
                writer.writerow(row)

    def plot(self, **kwargs):
        """2D depiction coloured by atom_confidence (always normalised)."""
        from ligand_neff.vis.plot import plot_atom_neff
        return plot_atom_neff(self.query_mol, self.atom_confidence, **kwargs)

    def plot_breakdown(self, **kwargs):
        """Bar chart with per-radius breakdown."""
        from ligand_neff.vis.plot import plot_confidence_bar
        return plot_confidence_bar(self.query_mol, self.atom_confidence, self.neff_per_radius)
