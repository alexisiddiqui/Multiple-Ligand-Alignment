import jax
import jax.numpy as jnp
import chex
from jaxtyping import Array, Float, Bool, jaxtyped
from beartype import beartype as typechecker


@jaxtyped(typechecker=typechecker)
def inverse_degree_weights(
    fps: Float[Array, "max_refs fp_size"],
    mask: Bool[Array, " max_refs"],
    threshold: float = 0.7,
    chunk_size: int = 2048,
) -> Float[Array, " max_refs"]:
    """
    Inverse Degree-style weights with memory-safe chunked pairwise computation.

    jaxtyping enforces fps and mask share the max_refs dimension.
    Output has same max_refs dimension as input.

    Peak memory: O(chunk_size × max_refs) instead of O(max_refs²).

    OOB-safety note: `lax.dynamic_slice` requires
        start + slice_size <= array_length
    for every loop iteration. Without padding, the last chunk starting at
    `(n_chunks-1)*chunk_size` would need `start + chunk_size <= max_refs`,
    which fails unless `max_refs` is a multiple of `chunk_size`
    (e.g. max_refs=10000, chunk_size=2048 → last start=8192,
    8192+2048=10240 > 10000). We fix this by padding fps/fp_bits/mask to
    the next multiple of chunk_size. Padding rows get mask=False so they
    contribute 0 to all neighbour counts; the extra output rows are cropped
    before returning.
    """
    chex.assert_rank(fps, 2)
    chex.assert_rank(mask, 1)
    chex.assert_equal_shape_prefix([fps, mask[:, None]], prefix_len=1)

    max_refs, fp_size = fps.shape
    fp_bits = jnp.sum(fps, axis=1)  # float32 dot products — cast already done by caller

    # ── Pad to exact multiple of chunk_size to avoid OOB dynamic_slice ──────
    pad_len = (-max_refs) % chunk_size   # 0 if already a multiple
    if pad_len > 0:
        fps_p     = jnp.pad(fps,     [(0, pad_len), (0, 0)])
        fp_bits_p = jnp.pad(fp_bits, [(0, pad_len)])
        mask_p    = jnp.pad(mask,    [(0, pad_len)])  # pads with False
    else:
        fps_p, fp_bits_p, mask_p = fps, fp_bits, mask

    neighbor_counts_p = _chunked_neighbor_count(
        fps_p, fp_bits_p, mask_p, threshold, chunk_size
    )
    # Crop back to original max_refs (extra pad rows are irrelevant)
    neighbor_counts = neighbor_counts_p[:max_refs]

    weights = jnp.where(
        mask,
        1.0 / jnp.maximum(neighbor_counts, 1.0),
        0.0,
    )

    chex.assert_tree_all_finite(weights)
    return weights


def _chunked_neighbor_count(
    fps: jnp.ndarray,         # (max_refs_padded, fp_size)  ← PADDED to chunk multiple
    fp_bits: jnp.ndarray,     # (max_refs_padded,)
    mask: jnp.ndarray,        # (max_refs_padded,)  — False on padding
    threshold: float,
    chunk_size: int,           # STATIC — must divide max_refs_padded exactly
) -> jnp.ndarray:             # (max_refs_padded,)
    """
    Count neighbors per reference using chunked pairwise Tanimoto.

    OOB-safety: `lax.dynamic_slice` requires
        start + slice_size <= array_length
    for ALL i in the loop, including the last chunk. We guarantee this by
    padding `fps` (and `fp_bits`, `mask`) to the next multiple of chunk_size
    BEFORE calling this function. The extra pad rows have mask=False, so
    they contribute 0 to all counts and are safe to read.

    The caller (`inverse_degree_weights`) is responsible for the padding:

        pad_len = (-max_refs) % chunk_size   # 0 if already a multiple
        fps     = jnp.pad(fps,     [(0, pad_len), (0, 0)])
        fp_bits = jnp.pad(fp_bits, [(0, pad_len)])
        mask    = jnp.pad(mask,    [(0, pad_len)])  # pads with False
        # Now fps.shape[0] == n_chunks * chunk_size exactly.

    Processes chunk_size rows at a time:
    1. dynamic_slice chunk rows out of fps        → (chunk_size, fp_size)
    2. Compute Tanimoto against ALL padded refs   → (chunk_size, max_refs_padded)
    3. Mask out padded columns + padded rows
    4. Count neighbours above threshold           → (chunk_size,)
    5. dynamic_update_slice counts back in
    """
    n_rows = fps.shape[0]                     # == n_chunks * chunk_size
    n_chunks = n_rows // chunk_size           # exact integer (no remainder by construction)
    neighbor_counts = jnp.zeros(n_rows)

    def chunk_body(i, counts):
        start = i * chunk_size
        # dynamic_slice is safe: start + chunk_size == (i+1)*chunk_size <= n_rows
        chunk_fps  = jax.lax.dynamic_slice(fps,     (start, 0), (chunk_size, fps.shape[1]))
        chunk_bits = jax.lax.dynamic_slice(fp_bits, (start,),   (chunk_size,))
        chunk_mask = jax.lax.dynamic_slice(mask,    (start,),   (chunk_size,))

        # Tanimoto: chunk vs all refs → (chunk_size, n_rows)
        intersection = jnp.dot(chunk_fps, fps.T)
        union = chunk_bits[:, None] + fp_bits[None, :] - intersection
        sim = jnp.where(union > 0, intersection / union, 0.0)

        # Zero out padded columns (invalid refs) and padded rows (invalid chunk rows)
        sim = sim * mask[None, :]        # (chunk_size, n_rows)
        sim = sim * chunk_mask[:, None]  # (chunk_size, n_rows)

        chunk_counts = jnp.sum(sim >= threshold, axis=1)  # (chunk_size,)
        counts = jax.lax.dynamic_update_slice(counts, chunk_counts.astype(counts.dtype), (start,))
        return counts

    neighbor_counts = jax.lax.fori_loop(0, n_chunks, chunk_body, neighbor_counts)
    return neighbor_counts
