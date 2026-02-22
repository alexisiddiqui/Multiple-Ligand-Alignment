import jax
import jax.numpy as jnp
import chex
from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker
from functools import partial


@partial(jax.jit, static_argnames=("min_overlap",))
@chex.assert_max_traces(n=50)
@jaxtyped(typechecker=typechecker)
def per_atom_neff_single_radius(
    atom_bit_mask: Float[Array, "n_atoms fp_size"],
    ref_fps: Float[Array, "max_refs fp_size"],
    weights: Float[Array, " max_refs"],
    min_overlap: float = 0.5,
) -> Float[Array, " n_atoms"]:
    """
    Per-atom Neff at a single Morgan radius.

    jaxtyping enforces:
    - atom_bit_mask and ref_fps share fp_size dimension
    - ref_fps and weights share max_refs dimension
    - Output has n_atoms from atom_bit_mask

    chex guards:
    - No NaN/Inf in intermediate results
    - weights are non-negative
    """
    chex.assert_rank([atom_bit_mask, ref_fps], 2)
    chex.assert_rank(weights, 1)

    atom_counts = jnp.sum(atom_bit_mask, axis=1, keepdims=True)
    atom_counts = jnp.maximum(atom_counts, 1.0)

    intersection = jnp.dot(atom_bit_mask, ref_fps.T)
    overlap = intersection / atom_counts

    gated = jnp.where(overlap >= min_overlap, overlap, 0.0)
    neff = jnp.dot(gated, weights)

    return neff
