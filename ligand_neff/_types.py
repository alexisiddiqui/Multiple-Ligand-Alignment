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
        pass # TODO: implementation

    def to_csv(self, path: str) -> None:
        """Per-atom table: idx, element, neff_r1, ..., combined, confidence."""
        pass # TODO: implementation

    def plot(self, **kwargs):
        """2D depiction coloured by atom_confidence (always normalised)."""
        pass # TODO: implementation

    def plot_breakdown(self, **kwargs):
        """Bar chart with per-radius breakdown."""
        pass # TODO: implementation
