from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker
import jax.numpy as jnp
import jax
import chex


@jaxtyped(typechecker=typechecker)
def aggregate_neff_stacked(
    neff_stacked: Float[Array, "n_radii n_atoms"],
    method: str = "geometric",
    radius_weights: tuple[float, ...] = (0.2, 0.5, 0.3),
) -> Float[Array, " n_atoms"]:
    """
    Combine per-atom Neff across a stacked array of radii.
    """
    w = jnp.array(radius_weights)

    if method == "geometric":
        shifted = jnp.log1p(neff_stacked)
        weighted = jnp.sum(w[:, None] * shifted, axis=0)
        combined = jnp.expm1(weighted / jnp.sum(w))
    elif method == "minimum":
        combined = jnp.min(neff_stacked, axis=0)
    elif method == "mean":
        combined = jnp.sum(w[:, None] * neff_stacked, axis=0) / jnp.sum(w)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")

    return combined


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

    return aggregate_neff_stacked(stacked, method, radius_weights)


@jaxtyped(typechecker=typechecker)
def normalise_to_confidence(
    neff: Float[Array, " n_atoms"],
    lam: float | jax.Array,
) -> Float[Array, " n_atoms"]:
    """Map raw Neff → [0, 1] confidence."""
    confidence = 1.0 - jnp.exp(-neff / lam)
    return confidence
