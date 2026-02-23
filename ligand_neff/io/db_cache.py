"""
Device-resident DB caching for fast repeated queries.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass
from typing import Mapping, Sequence

import jax.numpy as jnp
import numpy as np


@dataclass
class DbCache:
    """Stores precomputed database fingerprints on device."""
    fp_radii: tuple[int, ...]
    fp_size: int
    n_db: int
    db_fps_stacked: jnp.ndarray
    db_fps_per_radius: dict[int, jnp.ndarray]


def load_precomputed_npz(path: str) -> dict[str, np.ndarray]:
    """Load precomputed fingerprints from an npz file."""
    with np.load(path) as data:
        return {k: v for k, v in data.items()}


def build_db_cache(
    precomputed: Mapping[str, np.ndarray], 
    fp_radii: Sequence[int], 
    dtype=jnp.float32
) -> DbCache:
    """Build a DbCache from a precomputed dictionary mapping 'radius_{r}' to arrays."""
    stacked = []
    n_db = None
    fp_size = None
    
    for r in fp_radii:
        key = f"radius_{r}"
        if key not in precomputed:
            raise KeyError(f"Missing radius {r} in precomputed db (expected key: '{key}')")
        
        arr = precomputed[key]
        if n_db is None:
            n_db = arr.shape[0]
            fp_size = arr.shape[1]
        
        if arr.shape[0] != n_db:
             raise ValueError(f"Mismatched database size for radius {r}: expected {n_db}, got {arr.shape[0]}")
        if arr.shape[1] != fp_size:
             raise ValueError(f"Mismatched fp_size for radius {r}: expected {fp_size}, got {arr.shape[1]}")
             
        stacked.append(jnp.asarray(arr, dtype=dtype))
        
    db_fps_stacked = jnp.stack(stacked, axis=0) # (n_radii, n_db, fp_size)
    
    # Store per_radius too as views
    db_fps_per_radius = {r: db_fps_stacked[i] for i, r in enumerate(fp_radii)}
    
    return DbCache(
        fp_radii=tuple(fp_radii),
        fp_size=fp_size,
        n_db=n_db,
        db_fps_stacked=db_fps_stacked,
        db_fps_per_radius=db_fps_per_radius
    )
