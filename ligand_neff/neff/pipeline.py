from functools import partial

import jax
import jax.numpy as jnp

from ligand_neff.similarity.filtering_jax import filter_references_topk
from ligand_neff.neff.weighting import inverse_degree_weights
from ligand_neff.neff.per_atom import per_atom_neff_single_radius


@partial(jax.jit, static_argnames=("max_refs", "min_overlap", "atom_norm"))
def single_radius_pipeline_unweighted(
    q_fp: jnp.ndarray,
    db_fps: jnp.ndarray,
    atom_mask: jnp.ndarray,
    threshold: float | jax.Array,
    max_refs: int,
    min_overlap: float,
    atom_norm: str = "q_length",
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Single radius fusion pipeline computing Neff with uniform weights (unweighted).
    Combines filtering, weighting, and per-atom score computation.
    """
    filtered = filter_references_topk(q_fp, db_fps, threshold, max_refs)
    
    weights = jnp.where(filtered.mask, 1.0, 0.0)
    
    neff_r = per_atom_neff_single_radius(
        atom_bit_mask=atom_mask,
        ref_fps=filtered.fps,
        weights=weights,
        min_overlap=min_overlap,
        atom_norm=atom_norm
    )
    
    return neff_r, filtered.n_valid


@partial(jax.jit, static_argnames=("max_refs", "min_overlap", "atom_norm", "chunk_size"))
def single_radius_pipeline_inverse_degree(
    q_fp: jnp.ndarray,
    db_fps: jnp.ndarray,
    atom_mask: jnp.ndarray,
    threshold: float | jax.Array,
    cluster_threshold: float | jax.Array,
    max_refs: int,
    min_overlap: float,
    chunk_size: int,
    atom_norm: str = "q_length",
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Single radius fusion pipeline computing Neff using inverse-degree weighting.
    """
    filtered = filter_references_topk(q_fp, db_fps, threshold, max_refs)
    
    weights = inverse_degree_weights(
        fps=filtered.fps,
        mask=filtered.mask,
        threshold=cluster_threshold,
        chunk_size=chunk_size
    )
    
    neff_r = per_atom_neff_single_radius(
        atom_bit_mask=atom_mask,
        ref_fps=filtered.fps,
        weights=weights,
        min_overlap=min_overlap,
        atom_norm=atom_norm
    )
    
    return neff_r, filtered.n_valid


from ligand_neff.neff.aggregation import aggregate_neff_stacked, normalise_to_confidence

@partial(jax.jit, static_argnames=(
    "weighting", "max_refs", "min_overlap", "chunk_size", 
    "atom_norm", "aggregation", "lambda_mode", "radius_weights"
))
def compute_neff_core(
    q_fps_stacked: jnp.ndarray,
    db_fps_stacked: jnp.ndarray,
    atom_masks_stacked: jnp.ndarray,
    threshold: float | jax.Array,
    cluster_threshold: float | jax.Array,
    lambda_quantile: float | jax.Array,
    lambda_fixed: float | jax.Array,
    radius_weights: tuple[float, ...],
    weighting: str,
    max_refs: int,
    min_overlap: float,
    chunk_size: int,
    atom_norm: str,
    aggregation: str,
    lambda_mode: str,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Multi-radius fused JIT pipeline.
    Maps/vmaps over radii and returns only pure JAX arrays.
    """
    def _single_radius_fn(q, db, am):
        # We use python control flow here, perfectly fine under JIT since `weighting` is static
        if weighting == "inverse_degree":
            return single_radius_pipeline_inverse_degree(
                q, db, am, threshold, cluster_threshold, max_refs, min_overlap, chunk_size, atom_norm
            )
        else:
            return single_radius_pipeline_unweighted(
                q, db, am, threshold, max_refs, min_overlap, atom_norm
            )
            
    # Vectorize the pipeline over the n_radii dimension (axis 0 of inputs)
    neff_stack, n_valids = jax.vmap(_single_radius_fn)(q_fps_stacked, db_fps_stacked, atom_masks_stacked)
    
    combined_neff = aggregate_neff_stacked(
        neff_stack, 
        method=aggregation, 
        radius_weights=radius_weights
    )
    
    # We use python control flow here too since `lambda_mode` is static
    if lambda_mode == "fixed":
        lam = jnp.asarray(lambda_fixed, dtype=jnp.float32)
    else:
        # Avoid quantile on zero-size arrays (could crash if n_atoms=0, but assume n_atoms>0)
        lam_q = jnp.nanquantile(combined_neff, lambda_quantile)
        lam = jnp.maximum(lam_q, 1e-3)
        
    confidence = normalise_to_confidence(combined_neff, lam)
    
    n_refs_used = jnp.max(n_valids)
    
    return combined_neff, confidence, neff_stack, lam, n_refs_used
