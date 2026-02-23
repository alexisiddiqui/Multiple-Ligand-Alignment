import collections
from functools import partial

import jax
import jax.numpy as jnp

from ligand_neff.similarity.tanimoto import bulk_tanimoto
from ligand_neff.similarity.filtering import FilteredReferences

@partial(jax.jit, static_argnames=("max_refs",))
def filter_references_topk(
    query_fp: jnp.ndarray, 
    db_fps: jnp.ndarray, 
    threshold: float | jax.Array, 
    max_refs: int
) -> FilteredReferences:
    """
    Pure JAX filtering pipeline using lax.top_k.
    Calculates similarities and returns the top max_refs references.
    
    If fewer than max_refs molecules exceed the threshold, the mask will be false for the rest.
    If n_db < max_refs, it pads the output arrays.
    """
    n_db = db_fps.shape[0]
    
    # Pad db_fps if necessary to ensure lax.top_k doesn't fail
    # Note: JAX slice operations need static shapes. Since n_db might be static but variable
    # per invocation, this padding approach only works cleanly if n_db is a static dimension
    # (which it is for DbCache) or we just pad during the DbCache build. But since n_db is 
    # typically > max_refs, we'll use a shape check here for safety.
    # To keep it JAX-traceable, we can conditionally pad if n_db < max_refs using a generic shape padded.
    # Actually, JAX dimensions can't be dynamically sized anyway, so if n_db < max_refs the pad 
    # must be static. But what if n_db is passed dynamically? It's not, JIT bakes it in.
    
    # We'll just define the logic for static n_db tracing
    sims = bulk_tanimoto(query_fp, db_fps)                   # (n_db,)
    
    # If the database is smaller than max_refs, we pad sims to shape max_refs to allow top_k
    pad_len = max(0, max_refs - n_db)
    if pad_len > 0:
        sims = jnp.pad(sims, (0, pad_len), constant_values=-1.0)
        # Also need to pad db_fps so indexing works with padded indices
        db_fps = jnp.pad(db_fps, ((0, pad_len), (0, 0)), constant_values=0.0)
    
    # Compute top k using jax.lax.top_k
    top_sims, top_idx = jax.lax.top_k(sims, max_refs)        # (max_refs,)
    
    # Thresholding mask
    mask = top_sims >= threshold                             # (max_refs,)
    
    # Gather reference fingerprints
    ref_fps = jnp.take(db_fps, top_idx, axis=0)              # (max_refs, fp_size)
    
    # Zero out padded rows for determinism and debugging
    ref_fps = ref_fps * mask[:, None]
    top_sims = jnp.where(mask, top_sims, 0.0)
    
    n_valid = jnp.sum(mask, dtype=jnp.int32)
    
    return FilteredReferences(
        fps=ref_fps,
        mask=mask,
        similarities=top_sims,
        n_valid=n_valid
    )
