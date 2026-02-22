import chex
from jaxtyping import Float, Array
from ligand_neff._types import PaddedRefs, RefMask, RefWeights, AtomBitMask

@chex.dataclass
class NeffState:
    """
    Intermediate computation state. Passed through JAX pipeline.
    chex.dataclass makes this a PyTree, so jax.jit can trace through
    functions that accept/return NeffState.

    fp_radii is passed separately as a static Python tuple (not a field here)
    so it never becomes a traced leaf and changes in radii are handled via
    static_argnames on any jit-compiled callers.
    """
    ref_fps: PaddedRefs                        # (max_refs, fp_size)
    ref_mask: RefMask                          # (max_refs,)
    ref_weights: RefWeights                    # (max_refs,)
    atom_masks: Float[Array, "n_radii n_atoms fp_size"]  # stacked over radii
    # replaces the previous `dict` field. Ordering matches the fp_radii
    # tuple passed as a static argument to callers. Using a stacked tensor
    # keeps the PyTree structure fully static and enables vmap/lax.map over
    # radii without recompilation.
