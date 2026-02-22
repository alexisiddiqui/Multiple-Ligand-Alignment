import numpy as np
import jax.numpy as jnp
import chex
from jaxtyping import Array, Float, Bool, Int
from ligand_neff._types import PaddedRefs, RefMask, RefSimilarities
from ligand_neff.similarity.tanimoto import bulk_tanimoto


@chex.dataclass
class FilteredReferences:
    """
    Static-shape container for filtered reference ligands.
    All array fields have leading dimension = max_refs.
    Registered as a JAX PyTree via chex.dataclass, so this
    entire struct can cross jax.jit boundaries.

    IMPORTANT — n_valid is a JAX scalar (int32), NOT a Python int.
    If it were a Python int leaf inside a chex.dataclass PyTree, JAX would
    treat it as static data and trigger recompilation whenever its value
    changed. As a jnp.array scalar it is a traced leaf: JAX propagates it
    dynamically without recompiling. Callers that need the count as a
    Python int (e.g. for np.ndarray slicing on CPU) should call
    `int(refs.n_valid)` explicitly — that happens outside jit.
    """
    fps: PaddedRefs               # (max_refs, fp_size)
    mask: RefMask                 # (max_refs,)
    similarities: RefSimilarities # (max_refs,)
    n_valid: Int[Array, ""]       # JAX int32 scalar — safe across jit boundaries


def filter_references(
    query_fp: Float[Array, " fp_size"],
    db_fps: Float[Array, "n_db fp_size"],
    threshold: float,
    max_refs: int,
) -> FilteredReferences:
    """
    Filter + pad references to static shape.

    Note: This function is NOT @jax.jit because it uses dynamic
    numpy indexing. The jaxtyping annotations still check shapes
    at the input boundary.
    """
    sims = bulk_tanimoto(query_fp, db_fps)

    # CPU-side dynamic filtering
    sims_np = np.asarray(sims)
    passing_idx = np.where(sims_np >= threshold)[0]

    if len(passing_idx) > max_refs:
        top_k = np.argpartition(sims_np[passing_idx], -max_refs)[-max_refs:]
        passing_idx = passing_idx[top_k]

    n_valid = len(passing_idx)

    # Build padded static-shape arrays
    fp_size = db_fps.shape[1]
    padded_fps = np.zeros((max_refs, fp_size), dtype=np.float32)
    padded_sims = np.zeros(max_refs, dtype=np.float32)
    mask = np.zeros(max_refs, dtype=bool)

    if n_valid > 0:
        selected_fps = np.asarray(db_fps[passing_idx])
        padded_fps[:n_valid] = selected_fps
        padded_sims[:n_valid] = sims_np[passing_idx]
        mask[:n_valid] = True

    result = FilteredReferences(
        fps=jnp.array(padded_fps),
        mask=jnp.array(mask),
        similarities=jnp.array(padded_sims),
        n_valid=jnp.array(n_valid, dtype=jnp.int32),  # JAX scalar, not Python int
    )

    # Validate output shapes — chex catches any padding bugs
    chex.assert_shape(result.fps, (max_refs, fp_size))
    chex.assert_shape(result.mask, (max_refs,))
    chex.assert_shape(result.similarities, (max_refs,))

    return result
