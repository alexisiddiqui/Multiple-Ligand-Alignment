import jax
import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass
from typing import Sequence, Optional, Union
from rdkit import Chem

from ligand_neff.config import NeffConfig
from ligand_neff._types import QueryData, NeffResult
from ligand_neff.io.db_cache import DbCache, build_db_cache, load_precomputed_npz
from ligand_neff.compute import prepare_query_data
from ligand_neff.neff.pipeline import compute_neff_core

@dataclass(frozen=True)
class PreparedQuery:
    """Query data prepared and stacked on device for the Neff core kernel."""
    q_fps: jnp.ndarray          # (n_radii, fp_size)
    atom_masks: jnp.ndarray     # (n_radii, n_atoms, fp_size)
    n_atoms: int
    query_mol: Chem.Mol | None = None

class NeffEngine:
    """
    Stateful object providing a fast path for computing Neff.
    Loads DB into device memory once and JIT-compiles a static scoring graph.
    """
    def __init__(
        self,
        config: NeffConfig,
        *,
        precomputed_db: Union[str, dict, DbCache, None] = None,
        db_mols: Optional[Sequence[Chem.Mol]] = None,
        dtype=jnp.float32,
        compile_on_init: bool = True,
    ):
        self.config = config
        self.fp_radii = tuple(config.fp_radii)
        
        # 1. Build or extract DbCache
        if isinstance(precomputed_db, DbCache):
            self.db_cache = precomputed_db
        elif isinstance(precomputed_db, str):
            precomputed_dict = load_precomputed_npz(precomputed_db)
            self.db_cache = build_db_cache(precomputed_dict, self.fp_radii, dtype=dtype)
        elif isinstance(precomputed_db, dict):
            self.db_cache = build_db_cache(precomputed_db, self.fp_radii, dtype=dtype)
        elif db_mols is not None:
            raise NotImplementedError("Direct caching from db_mols not implemented yet. Pass precomputed_db.")
        else:
            raise ValueError("Must provide either precomputed_db or db_mols")

        # 3. Optional warmup
        if compile_on_init:
            self.warmup(n_atoms_hint=32)

    def warmup(self, n_atoms_hint: int = 32):
        q = jnp.zeros((len(self.fp_radii), self.config.fp_size), dtype=jnp.float32)
        am = jnp.zeros((len(self.fp_radii), n_atoms_hint, self.config.fp_size), dtype=jnp.float32)
        
        # Warmup by executing the JIT kernel with dummy data
        _ = compute_neff_core(
            q_fps_stacked=q,
            db_fps_stacked=self.db_cache.db_fps_stacked,
            atom_masks_stacked=am,
            threshold=self.config.tanimoto_inclusion,
            cluster_threshold=self.config.cluster_threshold,
            lambda_quantile=self.config.lambda_quantile,
            lambda_fixed=self.config.lambda_fixed,
            radius_weights=tuple(self.config.radius_weights),
            weighting=self.config.weighting,
            max_refs=self.config.max_references,
            min_overlap=float(self.config.min_overlap),
            chunk_size=self.config.inverse_degree_chunk_size,
            atom_norm=self.config.atom_norm,
            aggregation=self.config.aggregation,
            lambda_mode=self.config.lambda_mode,
        )

    def prepare_query(self, query_mol: Chem.Mol, query_data: Optional[QueryData] = None) -> PreparedQuery:
        """Prepares a single query molecule into stacked JAX arrays."""
        if query_data is None:
            query_data = prepare_query_data(query_mol, self.config)
            
        q_fps = jnp.stack([jnp.asarray(query_data.fps[r], dtype=jnp.float32) for r in self.fp_radii], axis=0)
        atom_masks = jnp.stack([jnp.asarray(query_data.atom_masks[r], dtype=jnp.float32) for r in self.fp_radii], axis=0)
        
        return PreparedQuery(q_fps=q_fps, atom_masks=atom_masks, n_atoms=query_data.n_atoms, query_mol=query_mol)

    def compute_prepared(self, prepared: PreparedQuery) -> NeffResult:
        """Runs the fully JITted JAX pipeline on a prepared query."""
        
        combined, conf, per_r, lam, nrefs = compute_neff_core(
            q_fps_stacked=prepared.q_fps,
            db_fps_stacked=self.db_cache.db_fps_stacked,
            atom_masks_stacked=prepared.atom_masks,
            threshold=self.config.tanimoto_inclusion,
            cluster_threshold=self.config.cluster_threshold,
            lambda_quantile=self.config.lambda_quantile,
            lambda_fixed=self.config.lambda_fixed,
            radius_weights=tuple(self.config.radius_weights),
            weighting=self.config.weighting,
            max_refs=self.config.max_references,
            min_overlap=float(self.config.min_overlap),
            chunk_size=self.config.inverse_degree_chunk_size,
            atom_norm=self.config.atom_norm,
            aggregation=self.config.aggregation,
            lambda_mode=self.config.lambda_mode,
        )

        combined_np = np.asarray(combined)
        conf_np = np.asarray(conf)
        per_r_np = np.asarray(per_r)

        neff_dict = {r: per_r_np[i] for i, r in enumerate(self.fp_radii)}

        return NeffResult(
            query_mol=prepared.query_mol,
            config=self.config,
            atom_neff=combined_np,
            atom_confidence=conf_np,
            neff_per_radius=neff_dict,
            global_neff=float(combined_np.mean()),
            global_confidence=float(conf_np.mean()),
            n_references_used=int(np.asarray(nrefs)),
            lambda_value=float(np.asarray(lam)),
        )

    def compute(self, query_mol: Chem.Mol) -> NeffResult:
        """Helper that prepares and computes in one step."""
        prepared = self.prepare_query(query_mol)
        return self.compute_prepared(prepared)
