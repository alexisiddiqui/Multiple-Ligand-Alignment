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
    atom_masks: jnp.ndarray     # (n_radii, max_atoms, fp_size)  -- padded
    n_atoms: int                # true atom count before padding
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
        max_atoms: int = 64,
    ):
        self.config = config
        self.fp_radii = tuple(config.fp_radii)
        # Pad all atom_masks to this fixed size so JAX never retraces for new n_atoms shapes.
        self.max_atoms = max_atoms
        
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

        # 3. Optional warmup — compile with the fixed max_atoms shape
        if compile_on_init:
            self.warmup(n_atoms_hint=self.max_atoms)

    def warmup(self, n_atoms_hint: int | None = None):
        n_atoms_hint = n_atoms_hint if n_atoms_hint is not None else self.max_atoms
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
        """Prepares a single query molecule into stacked JAX arrays, padded to max_atoms."""
        if query_data is None:
            query_data = prepare_query_data(query_mol, self.config)

        n_atoms = query_data.n_atoms
        if n_atoms > self.max_atoms:
            raise ValueError(
                f"Query molecule has {n_atoms} atoms, which exceeds max_atoms={self.max_atoms}. "
                "Increase max_atoms when constructing NeffEngine."
            )

        q_fps = jnp.stack([jnp.asarray(query_data.fps[r], dtype=jnp.float32) for r in self.fp_radii], axis=0)

        # Pad each atom_mask to (max_atoms, fp_size) so JAX always sees the same shape
        padded_masks = []
        for r in self.fp_radii:
            mask = jnp.asarray(query_data.atom_masks[r], dtype=jnp.float32)  # (n_atoms, fp_size)
            pad_rows = self.max_atoms - n_atoms
            mask_padded = jnp.pad(mask, ((0, pad_rows), (0, 0)))              # (max_atoms, fp_size)
            padded_masks.append(mask_padded)
        atom_masks = jnp.stack(padded_masks, axis=0)                          # (n_radii, max_atoms, fp_size)

        return PreparedQuery(q_fps=q_fps, atom_masks=atom_masks, n_atoms=n_atoms, query_mol=query_mol)

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

        n = prepared.n_atoms  # true atom count; slice off padding
        combined_np = np.asarray(combined)[:n]
        conf_np     = np.asarray(conf)[:n]
        per_r_np    = np.asarray(per_r)[:, :n]

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
