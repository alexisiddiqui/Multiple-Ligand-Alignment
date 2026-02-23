from typing import Sequence
import numpy as np
import chex
import jax.numpy as jnp
from rdkit import Chem

from ligand_neff.config import NeffConfig
from ligand_neff._types import NeffResult, QueryData
from ligand_neff.fingerprints.encode import encode_molecule
from ligand_neff.fingerprints.decompose import decompose
from ligand_neff.similarity.filtering import filter_references
from ligand_neff.neff.weighting import inverse_degree_weights
from ligand_neff.neff.per_atom import per_atom_neff_single_radius
from ligand_neff.neff.aggregation import aggregate_neff, normalise_to_confidence

def prepare_query_data(query: Chem.Mol, config: NeffConfig) -> QueryData:
    """
    Precompute fingerprints and atom masks for a query molecule.
    This can be done once and reused for multiple Neff computations.
    """
    query_fps = {}
    atom_masks = {}
    
    for r in config.fp_radii:
        # Query Fingerprint
        q_fp = encode_molecule(query, radius=r, fp_size=config.fp_size, 
                              use_chirality=config.use_chirality, 
                              use_features=config.use_features).astype(np.float32)
        query_fps[r] = q_fp
        
        # Atom Decomposition
        decomp = decompose(query, r, config.fp_size, config.use_chirality, config.use_features)
        atom_mask = decomp.build_atom_bit_mask()
        atom_masks[r] = np.asarray(atom_mask)
        
    return QueryData(
        fps=query_fps,
        atom_masks=atom_masks,
        n_atoms=query.GetNumAtoms()
    )

def compute_neff(
    query_data: QueryData, 
    config: NeffConfig,
    db_mols: Sequence[Chem.Mol] | None = None, 
    precomputed_db: str | dict | None = None,
    query_mol: Chem.Mol | None = None
) -> NeffResult:
    """
    Main pipeline for computing Neff and confidence scores for a query against a reference database.
    
    Architecture handles dynamic indexing (filtering) on CPU and then passes padded, 
    static-shaped PyTrees into JAX to avoid recompilations downstream.
    """
    # Each query molecule may have a different atom count, causing JAX to retrace
    # JIT-compiled functions with new static shapes. Clear the trace counter so
    # the per-function limit only fires on pathological single-call retracing.
    chex.clear_trace_counter()
    
    # ── 0. Handle Precomputed Database ───────────────────────────
    if isinstance(precomputed_db, str):
        precomputed_db = np.load(precomputed_db)
        
    # ── 1. CPU Stage: Fingerprints & Filtering ───────────────────
    neff_per_radius = {}
    max_refs_used = 0
    
    # We loop over radii. Fingerprint and filter for each radius.
    for i, r in enumerate(config.fp_radii):
        # Query Fingerprint (from precomputed data)
        q_fp = query_data.fps[r]
                              
        # DB Fingerprints (load from precomputed if available)
        if precomputed_db is not None:
            db_fps = precomputed_db[f"radius_{r}"].astype(np.float32)
        else:
            if db_mols is None:
                raise ValueError("Must provide either db_mols or precomputed_db")
            db_fps = np.zeros((len(db_mols), config.fp_size), dtype=np.float32)
            for j, mol in enumerate(db_mols):
                db_fps[j] = encode_molecule(mol, radius=r, fp_size=config.fp_size,
                                            use_chirality=config.use_chirality, 
                                            use_features=config.use_features)
                                        
        # Filtering & Padding on CPU
        filtered = filter_references(
            query_fp=jnp.array(q_fp),
            db_fps=jnp.array(db_fps),
            threshold=config.tanimoto_inclusion,
            max_refs=config.max_references
        )
        
        # Atom Mask (from precomputed data)
        atom_mask = jnp.array(query_data.atom_masks[r])
        
        max_refs_used = max(max_refs_used, int(filtered.n_valid))
        
        # ── 2. GPU/JAX Stage: Static Shape Math ────────────────────
        
        # Calculate weights based on filtering mask (Weighting can be "none")
        if config.weighting == "inverse_degree":
            weights = inverse_degree_weights(
                fps=filtered.fps,
                mask=filtered.mask,
                threshold=config.cluster_threshold,
                chunk_size=config.inverse_degree_chunk_size
            )
        else:
            # If no weighting, use ones for masked refs and zeros for padded ones
            weights = jnp.where(filtered.mask, 1.0, 0.0)
            
        # Per Atom Neff for this radius
        neff_r = per_atom_neff_single_radius(
            atom_bit_mask=atom_mask,
            ref_fps=filtered.fps,
            weights=weights,
            min_overlap=config.min_overlap,
        )
        
        neff_per_radius[r] = np.asarray(neff_r)
        
    # ── 3. Aggregation across Radii ──────────────────────────────
    # Convert dict of numpy arrays back to JAX arrays for aggregation
    neff_per_radius_jax = {r: jnp.array(v) for r, v in neff_per_radius.items()}
    
    combined_neff = aggregate_neff(
        neff_per_radius_jax, 
        method=config.aggregation, 
        radius_weights=config.radius_weights
    )
    
    # ── 4. Normalise to Confidence ───────────────────────────────
    if config.lambda_mode == "fixed":
        lam = config.lambda_fixed
    else:
        # Adaptive Lambda based on quantile of the query's neff scores
        lam = float(jnp.maximum(jnp.quantile(combined_neff, config.lambda_quantile), 1e-3))
        
    confidence = normalise_to_confidence(combined_neff, lam)
    
    # Global scores are just scalar means of the atom scores for now
    global_neff = float(jnp.mean(combined_neff))
    global_confidence = float(jnp.mean(confidence))

    return NeffResult(
        query_mol=query_mol,
        config=config,
        atom_neff=np.asarray(combined_neff),
        atom_confidence=np.asarray(confidence),
        neff_per_radius=neff_per_radius,
        global_neff=global_neff,
        global_confidence=global_confidence,
        n_references_used=max_refs_used,
        lambda_value=lam
    )
