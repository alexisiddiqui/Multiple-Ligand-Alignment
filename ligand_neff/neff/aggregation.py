from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker
import jax.numpy as jnp
import chex


@jaxtyped(typechecker=typechecker)
def aggregate_neff(
    neff_per_radius: dict[int, Float[Array, " n_atoms"]],
    method: str = "geometric",
    radius_weights: tuple[float, ...] = (0.2, 0.5, 0.3),
) -> Float[Array, " n_atoms"]:
    """
    Combine per-atom Neff across radii.
    jaxtyping enforces all radius arrays share n_atoms dimension.
    """
    radii = sorted(neff_per_radius.keys())
    stacked = jnp.stack([neff_per_radius[r] for r in radii], axis=0)

    chex.assert_equal_shape(list(neff_per_radius.values()))

    w = jnp.array(radius_weights)

    if method == "geometric":
        shifted = jnp.log1p(stacked)
        weighted = jnp.sum(w[:, None] * shifted, axis=0)
        combined = jnp.expm1(weighted / jnp.sum(w))
    elif method == "minimum":
        combined = jnp.min(stacked, axis=0)
    elif method == "mean":
        combined = jnp.sum(w[:, None] * stacked, axis=0) / jnp.sum(w)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")

    chex.assert_tree_all_finite(combined)
    return combined


@jaxtyped(typechecker=typechecker)
def normalise_to_confidence(
    neff: Float[Array, " n_atoms"],
    lam: float,
) -> Float[Array, " n_atoms"]:
    """Map raw Neff → [0, 1] confidence."""
    chex.assert_scalar_positive(lam)
    confidence = 1.0 - jnp.exp(-neff / lam)
    return confidence
