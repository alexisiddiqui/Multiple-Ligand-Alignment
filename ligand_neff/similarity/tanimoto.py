import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Bool, jaxtyped
from beartype import beartype as typechecker
from functools import partial


@jaxtyped(typechecker=typechecker)
@partial(jax.jit)
def bulk_tanimoto(
    query: Float[Array, " fp_size"],
    database: Float[Array, "n_refs fp_size"],
) -> Float[Array, " n_refs"]:
    """
    Tanimoto between one query and N database fingerprints.

    jaxtyping enforces:
    - Both arrays are float dtype
    - query is 1D, database is 2D
    - Their last dimensions (fp_size) match
    """
    intersection = jnp.dot(database, query)
    query_bits = jnp.sum(query)
    db_bits = jnp.sum(database, axis=1)
    union = query_bits + db_bits - intersection
    return jnp.where(union > 0, intersection / union, 0.0)


@jaxtyped(typechecker=typechecker)
@jax.jit
def pairwise_tanimoto_chunk(
    chunk_fps: Float[Array, "chunk fp_size"],
    all_fps: Float[Array, "n_refs fp_size"],
    chunk_bits: Float[Array, " chunk"],
    all_bits: Float[Array, " n_refs"],
    mask: Bool[Array, " n_refs"],
) -> Float[Array, "chunk n_refs"]:
    """Tanimoto between a chunk of references and all references."""
    intersection = jnp.dot(chunk_fps, all_fps.T)
    union = chunk_bits[:, None] + all_bits[None, :] - intersection
    sim = jnp.where(union > 0, intersection / union, 0.0)
    return sim * mask[None, :]
