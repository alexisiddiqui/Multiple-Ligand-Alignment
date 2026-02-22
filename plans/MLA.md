# `ligand-neff` v2.1: Revised Implementation Plan

## Revision Summary

Addresses five critical issues from review, plus chex/jaxtyping integration and four additional JAX correctness fixes:

1. **JAX dynamic shape recompilation** → static padding + boolean masking throughout
2. **Inverse Degree pairwise OOM** → chunked `lax.map` with configurable chunk size
3. **Visualization outlier sensitivity** → always plot normalised confidence, not raw Neff
4. **λ saturation mismatch** → adaptive λ estimation from reference distribution
5. **Bitvector length** → tunable: 2048, 4096, 8192 with validation
6. **Type safety** → `chex` dataclasses + `jaxtyping`/`beartype` shape annotations throughout
7. **`FilteredReferences.n_valid` recompilation** → stored as `Int[Array, ""]` (JAX int32 scalar), not Python `int`
8. **`dynamic_slice` OOB in chunked Inverse Degree** → fps padded to next multiple of `chunk_size` before slicing
9. **`NeffState.atom_masks: dict`** → replaced with stacked `Float[Array, "n_radii n_atoms fp_size"]` tensor
10. **Fingerprint memory** → stored as `uint8`, cast to `float32` inside JAX kernels

---

## 1a. Dependencies

```toml
[project]
dependencies = [
    "rdkit",
    "jax[cpu]",
    "jaxlib",
    "numpy>=1.24",
    "chex>=0.1.87",
    "jaxtyping>=0.2.28",
    "beartype>=0.18",
]
```

| Library | Role | Scope |
|---------|------|-------|
| **jaxtyping** | Shape + dtype annotations on function signatures | Every JAX-facing function |
| **beartype** | Runtime enforcement of jaxtyping annotations | Dev/test; zero-cost in production via import hook |
| **chex** | JAX-aware dataclasses (PyTree-compatible), shape/rank/dtype/trace assertions inside function bodies, test variants | Dataclasses, internal guards, testing |

jaxtyping annotates the *contract*, beartype *enforces* it at call
boundaries, and chex *guards* invariants inside function bodies and
provides JAX-native dataclasses.

---

## 1b. Architecture (Revised)

```
mla/
├── __init__.py
├── _types.py            # Type aliases + NeffResult (chex.dataclass)
├── config.py
├── io/
│   ├── query.py
│   └── database.py
├── fingerprints/
│   ├── decompose.py
│   └── encode.py
├── similarity/
│   ├── tanimoto.py
│   └── filtering.py
├── neff/
│   ├── _state.py        # NeffState (chex.dataclass, internal)
│   ├── weighting.py
│   ├── per_atom.py
│   └── aggregation.py
├── vis/
│   └── plot.py
└── cli.py
```

---

## 1c. Project-Wide Setup

### Import Hook (Development Mode)

```python
# tests/conftest.py
from jaxtyping import install_import_hook

# All functions in ligand_neff will be checked at test time
install_import_hook("ligand_neff", "beartype.beartype")
```

The hook catches everything at test time. In production, users import
normally without the hook — zero overhead.

### Explicit Decoration (Public API)

For public-facing functions, use explicit `@jaxtyped(typechecker=beartype)`
so shape checking is always on regardless of how the package is imported:

```python
from jaxtyping import jaxtyped
from beartype import beartype as typechecker
```

### Decorator Stacking Order

```python
@jaxtyped(typechecker=typechecker)           # 1. Shape check at call boundary
@chex.assert_max_traces(n=50)                # 2. Guard against recompilation
@partial(jax.jit, static_argnames=(...))     # 3. JIT compilation
def my_function(
    x: Float[Array, "n_atoms fp_size"],
    y: Float[Array, "max_refs fp_size"],
) -> Float[Array, " n_atoms"]:
    chex.assert_rank(x, 2)                   # 4. chex assertions inside body
    ...
```

For development/debug with value assertions, add `@chex.chexify` between
`@jaxtyped` and `@chex.assert_max_traces`.

---

## 1d. Type Aliases (`_types.py`)

```python
# ligand_neff/_types.py
"""Shared type aliases used across the package."""

import jax.numpy as jnp
from jaxtyping import Float, Bool, Int, UInt8, Array

# ── Core array types ──────────────────────────────────────────────
# Fingerprints are stored on disk/CPU as uint8 (0 or 1 per bit position).
# They are cast to float32 inside JAX kernels that use dot-products.
Fingerprint      = UInt8[Array, "fp_size"]       # storage dtype
FingerprintF32   = Float[Array, "fp_size"]       # computation dtype (cast inside kernels)
FingerprintBatch = UInt8[Array, "batch fp_size"] # storage dtype

# Per-atom structures
AtomBitMask      = Float[Array, "n_atoms fp_size"]  # always float32 (used in dots)
AtomScores       = Float[Array, "n_atoms"]

# Padded reference arrays (static shape from config.max_references)
# Stored as uint8 in the database, converted to float32 for JAX ops.
PaddedRefs       = Float[Array, "max_refs fp_size"]  # float32 inside JIT
RefMask          = Bool[Array, "max_refs"]
RefWeights       = Float[Array, "max_refs"]
RefSimilarities  = Float[Array, "max_refs"]

# Similarity matrices
SimMatrix        = Float[Array, "rows cols"]

# Scalars
Scalar           = Float[Array, ""]
IntScalar        = Int[Array, ""]   # e.g. n_valid stored as JAX int32
```

---

## 1e. Internal State Container (`neff/_state.py`)

```python
import chex
from ligand_neff._types import PaddedRefs, RefMask, RefWeights, AtomBitMask

@chex.dataclass
class NeffState:
    """
    Intermediate computation state. Passed through JAX pipeline.
    chex.dataclass makes this a PyTree, so jax.jit can trace through
    functions that accept/return NeffState.

    fp_radii is passed separately as a static Python tuple (not a field here)
    so it never becomes a traced leaf and changes in radii are handled via
    static_argnames on any jit-compiled callers.
    """
    ref_fps: PaddedRefs                        # (max_refs, fp_size)
    ref_mask: RefMask                          # (max_refs,)
    ref_weights: RefWeights                    # (max_refs,)
    atom_masks: Float[Array, "n_radii n_atoms fp_size"]  # stacked over radii
    # ↑ replaces the previous `dict` field. Ordering matches the fp_radii
    # tuple passed as a static argument to callers. Using a stacked tensor
    # keeps the PyTree structure fully static and enables vmap/lax.map over
    # radii without recompilation.
```

---

## 1f. Assertion Strategy

| Assertion Type | Library | When | Cost |
|---|---|---|---|
| **Shape/dtype at boundaries** | jaxtyping + beartype | Function entry/exit | Trace-time only |
| **Rank/shape inside bodies** | chex.assert_rank, assert_shape | After array construction | Trace-time (static) |
| **Value invariants** | chex.assert_tree_all_finite | After computation | Requires chexify for JIT |
| **Trace count guards** | chex.assert_max_traces | On JIT'd functions | Compile-time |
| **Dimension consistency** | chex.Dimensions | Multi-array checks | Trace-time |

---

## 2. Config (Revised)

```python
from dataclasses import dataclass
from typing import Literal

VALID_FP_SIZES = (2048, 4096, 8192)

@dataclass
class NeffConfig:
    """All tunable parameters for a Neff computation."""

    # ── Fingerprint ──────────────────────────────────────────────
    fp_radii: tuple[int, ...] = (1, 2, 3)
    fp_size: int = 2048                         # Must be in VALID_FP_SIZES
    use_chirality: bool = False
    use_features: bool = False                  # FCFP-style if True

    # ── Reference filtering ──────────────────────────────────────
    tanimoto_inclusion: float = 0.3
    max_references: int = 10_000                # STATIC shape for JAX

    # ── Weighting ────────────────────────────────────────────────
    cluster_threshold: float = 0.7
    weighting: Literal["inverse_degree", "none"] = "inverse_degree"
    inverse_degree_chunk_size: int = 2048             # Rows per chunk in pairwise

    # ── Per-atom Neff ────────────────────────────────────────────
    coverage_metric: Literal["overlap", "binary"] = "overlap"
    min_overlap: float = 0.5

    # ── Aggregation ──────────────────────────────────────────────
    aggregation: Literal["geometric", "minimum", "mean"] = "geometric"
    radius_weights: tuple[float, ...] = (0.2, 0.5, 0.3)

    # ── Normalisation ────────────────────────────────────────────
    lambda_mode: Literal["fixed", "adaptive"] = "adaptive"
    lambda_fixed: float = 10.0                  # Only used if lambda_mode="fixed"
    lambda_quantile: float = 0.5                # Median of global Neff for adaptive

    def __post_init__(self):
        if self.fp_size not in VALID_FP_SIZES:
            raise ValueError(
                f"fp_size must be one of {VALID_FP_SIZES}, got {self.fp_size}. "
                f"Larger sizes reduce bit collisions but increase memory. "
                f"2048 ≈ ECFP standard, 4096 recommended for per-atom work, "
                f"8192 for maximum precision."
            )
        if len(self.radius_weights) != len(self.fp_radii):
            raise ValueError(
                f"radius_weights length ({len(self.radius_weights)}) must match "
                f"fp_radii length ({len(self.fp_radii)})"
            )
        if self.max_references > 50_000:
            import warnings
            warnings.warn(
                f"max_references={self.max_references} will allocate "
                f"~{self.max_references**2 * 4 / 1e9:.1f} GB for Inverse Degree "
                f"pairwise matrix. Consider reducing or using weighting='none'."
            )
```

### Fingerprint Size Guidance

| fp_size | Memory per mol | Collision rate (r=2) | Collision rate (r=3) | Use case |
|---------|---------------|---------------------|---------------------|----------|
| 2048 | 2 KB | ~5-8% | ~10-15% | Fast screening, global Tanimoto |
| 4096 | 4 KB | ~2-4% | ~5-8% | **Recommended for per-atom Neff** |
| 8192 | 8 KB | ~1-2% | ~2-4% | Maximum precision, small DB |

Bit collisions matter more for per-atom attribution than for global similarity.
Two different atom environments hashing to the same bit inflates coverage for
both atoms. At radius=3 with fp_size=2048, this noise is non-trivial. Default
recommendation for serious per-atom work: **fp_size=4096**.

### 2.1 Declarative YAML Configuration

To facilitate reproducibility and batch processing, `NeffConfig` can be instantiated directly from a YAML file.

```yaml
# config.yaml
fingerprint:
  fp_radii: [1, 2, 3]
  fp_size: 4096
  use_chirality: false
  use_features: false

reference_filtering:
  tanimoto_inclusion: 0.3
  max_references: 10000

weighting:
  cluster_threshold: 0.7
  weighting: "inverse_degree"
  inverse_degree_chunk_size: 2048

per_atom_neff:
  coverage_metric: "overlap"
  min_overlap: 0.5

aggregation:
  aggregation: "geometric"
  radius_weights: [0.2, 0.5, 0.3]

normalisation:
  lambda_mode: "adaptive"
  lambda_quantile: 0.5
```

```python
# ligand_neff/config.py
import yaml
from pathlib import Path

def load_config(path: str | Path) -> NeffConfig:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    
    # Flatten nested YAML structure to match flat dataclass fields
    flat_data = {}
    for section in data.values():
        if isinstance(section, dict):
            flat_data.update(section)
        else:
            flat_data.update(data)
            break
            
    if "fp_radii" in flat_data:
        flat_data["fp_radii"] = tuple(flat_data["fp_radii"])
    if "radius_weights" in flat_data:
        flat_data["radius_weights"] = tuple(flat_data["radius_weights"])
        
    return NeffConfig(**flat_data)
```

---

## 3. The Static Shape Contract (NEW — Critical)

### The Problem

`jax.jit` traces functions once per unique combination of array shapes and
`static_argnames` values. If `filter_references` returns M=4821 for query A
and M=3107 for query B, JAX recompiles the entire downstream pipeline for
each query. At ~200ms per compilation, this destroys performance for batch
processing.

### The Solution: Pad + Mask

Every array that enters a JIT-compiled function has a **fixed shape**
determined by `config.max_references`. Variable-length data is represented
as a padded array + a boolean validity mask.

```
┌─────────────────────────────────────────────────┐
│  Filtered references: M = 4821 actual ligands   │
│                                                  │
│  ref_fps:  (max_refs, fp_size)  ← zero-padded   │
│  ref_mask: (max_refs,)          ← True/False     │
│                                                  │
│  ref_fps[0:4821]  = actual fingerprints          │
│  ref_fps[4821:]   = zeros                        │
│  ref_mask[0:4821] = True                         │
│  ref_mask[4821:]  = False                        │
└─────────────────────────────────────────────────┘
```

This mask propagates through the entire JAX pipeline:

```
filter_references  →  (padded_fps, mask)
        │                    │
        ▼                    ▼
pairwise_tanimoto  →  (sim_matrix, mask)     # mask out padded rows/cols
        │                    │
        ▼                    ▼
inverse_degree_weights   →  (weights * mask)       # padded entries get weight 0
        │                    │
        ▼                    ▼
per_atom_neff      →  neff (correct)         # zero weights kill padding
```

---

## 4. Revised Module Specifications

### 4.1 `similarity/filtering.py` (Revised)

```python
import numpy as np
import jax.numpy as jnp
import chex
from jaxtyping import Array, Float, Bool
from ligand_neff._types import PaddedRefs, RefMask, RefSimilarities


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
```

**Design note**: The filtering itself stays on CPU (numpy). This is deliberate.
Dynamic boolean indexing and argsort are awkward in JAX and would require
`jax.experimental.io_callback` or produce worse XLA code than just doing it
in numpy. The key insight is that filtering is O(N_db) and takes <10ms — it's
not the bottleneck. Everything *downstream* of filtering runs on padded
static-shape arrays in JIT-compiled JAX.

### 4.2 `similarity/tanimoto.py` (Revised — Typed)

```python
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
```

**Note on `@jaxtyped` + `@jax.jit` ordering:** `@jaxtyped` must be the
outermost decorator. jaxtyping checks shapes during JAX tracing (when
abstract shapes are available), so it works correctly inside JIT — the
checks happen at trace time, not at runtime, meaning zero performance cost
after the first trace.

### 4.3 `neff/weighting.py` (Revised — Chunked Pairwise)

```python
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
```

**Memory budget table:**

| max_refs | chunk_size | Padded size | Peak VRAM (pairwise) | Full matrix | Savings |
|----------|-----------|-------------|---------------------|-------------|---------|
| 5,000 | 2048 | 6144 | 48 MB | 100 MB | 2.1× |
| 10,000 | 2048 | 10240 | 80 MB | 400 MB | 5× |
| 25,000 | 2048 | 26112 | 204 MB | 2.5 GB | 12.3× |
| 50,000 | 2048 | 51200 | 400 MB | 10 GB | 25× |

The "Padded size" column shows the actual array allocated (next multiple of
chunk_size). Overhead is at most `chunk_size - 1` rows ≈ 2047 extra rows,
which is negligible. The chunk_size of 2048 is tunable via config. On GPUs
with >16 GB VRAM, increasing to 4096 reduces loop iterations and may be faster.

**OOB guard note:** Because `inverse_degree_weights` pads before calling
`_chunked_neighbor_count`, the `lax.dynamic_slice` start indices are always
a multiple of `chunk_size`, so `start + chunk_size == (i+1)*chunk_size <= padded_size`
holds for every `i` in `[0, n_chunks)`. No bounds check is needed inside the loop.

### 4.4 `neff/per_atom.py` (Revised — Typed + Mask-Aware)

```python
import jax
import jax.numpy as jnp
import chex
from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker
from functools import partial


@jaxtyped(typechecker=typechecker)
@chex.assert_max_traces(n=50)
@partial(jax.jit, static_argnames=("min_overlap",))
def per_atom_neff_single_radius(
    atom_bit_mask: Float[Array, "n_atoms fp_size"],
    ref_fps: Float[Array, "max_refs fp_size"],
    weights: Float[Array, " max_refs"],
    min_overlap: float = 0.5,
) -> Float[Array, " n_atoms"]:
    """
    Per-atom Neff at a single Morgan radius.

    jaxtyping enforces:
    - atom_bit_mask and ref_fps share fp_size dimension
    - ref_fps and weights share max_refs dimension
    - Output has n_atoms from atom_bit_mask

    chex guards:
    - No NaN/Inf in intermediate results
    - weights are non-negative
    """
    chex.assert_rank([atom_bit_mask, ref_fps], 2)
    chex.assert_rank(weights, 1)

    atom_counts = jnp.sum(atom_bit_mask, axis=1, keepdims=True)
    atom_counts = jnp.maximum(atom_counts, 1.0)

    intersection = jnp.dot(atom_bit_mask, ref_fps.T)
    overlap = intersection / atom_counts

    gated = jnp.where(overlap >= min_overlap, overlap, 0.0)
    neff = jnp.dot(gated, weights)

    return neff
```

**Why n_atoms variation doesn't cause recompilation**: For batch processing,
different query molecules have different atom counts. Options:

1. **Pad atoms too** (if batch processing many queries): pad `atom_bit_mask` to
   `(max_atoms, fp_size)` with an atom mask. Adds complexity but enables
   `vmap` over queries.
2. **Accept per-query recompilation** (recommended for typical use): Most
   drug-like molecules have 20–50 heavy atoms. JAX will cache a compilation
   for each unique n_atoms value. After seeing ~30 distinct sizes, most new
   queries will hit the cache. For a batch of 1000 diverse molecules, you'd
   get ~30 compilations + 970 cache hits.

The plan uses option 2 by default, with an optional `max_atoms` padding
parameter in config for high-throughput batch mode.

### 4.5 `neff/aggregation.py` (Revised — Adaptive λ)

```python
from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker
import chex


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

    w = jnp.array(radius_weights)

    if method == "geometric":
        shifted = jnp.log1p(stacked)
        weighted = jnp.sum(w[:, None] * shifted, axis=0)
        combined = jnp.expm1(weighted / jnp.sum(w))
    elif method == "minimum":
        combined = jnp.min(stacked, axis=0)
    elif method == "mean":
        combined = jnp.sum(w[:, None] * stacked, axis=0) / jnp.sum(w)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")

    chex.assert_tree_all_finite(combined)
    return combined


@jaxtyped(typechecker=typechecker)
def normalise_to_confidence(
    neff: Float[Array, " n_atoms"],
    lam: float,
) -> Float[Array, " n_atoms"]:
    """Map raw Neff → [0, 1] confidence."""
    chex.assert_scalar_positive_finite(lam)
    confidence = 1.0 - jnp.exp(-neff / lam)
    return confidence
```

**Adaptive λ examples:**

| Database | Typical median Neff | Adaptive λ | Median confidence | Top atom (3×) |
|----------|--------------------|-----------|--------------------|---------------|
| 50 known actives | ~3 | 3.0 | 0.63 | 0.95 |
| 1K target-specific | ~8 | 8.0 | 0.63 | 0.95 |
| ChEMBL 2M | ~45 | 45.0 | 0.63 | 0.95 |

The adaptive approach guarantees meaningful spread regardless of database size.
Users can override with `lambda_mode="fixed"` for reproducibility across
different databases.

### 4.6 `vis/plot.py` (Revised — Confidence-Based)

```python
from rdkit.Chem import Draw
from rdkit.Chem.Draw import SimilarityMaps
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


def plot_atom_neff(
    mol: Chem.Mol,
    confidence: np.ndarray,          # (n_atoms,) — ALWAYS normalised [0, 1]
    size: tuple[int, int] = (600, 400),
    cmap_name: str = "RdYlGn",
    show_values: bool = False,
) -> PIL.Image.Image:
    """
    Render 2D depiction with atoms coloured by confidence score.

    IMPORTANT: This function takes confidence (0-1), not raw Neff.
    This guarantees:
    - Consistent colour scaling across molecules and databases
    - No outlier sensitivity (raw Neff of 500 vs 10 would break colour maps)
    - Intuitive interpretation: red = low confidence, green = high

    Uses RDKit's atom highlight-based colouring for clean rendering.
    """
    cmap = cm.get_cmap(cmap_name)

    # Build atom colour map: atom_idx → (r, g, b)
    atom_colors = {}
    for i in range(mol.GetNumAtoms()):
        rgba = cmap(float(confidence[i]))
        atom_colors[i] = tuple(rgba[:3])

    # Build radius map proportional to confidence for visual weight
    atom_radii = {}
    for i in range(mol.GetNumAtoms()):
        atom_radii[i] = 0.3 + 0.3 * float(confidence[i])  # 0.3 to 0.6

    drawer = Draw.MolDraw2DCairo(size[0], size[1])
    drawer.DrawMoleculeWithHighlights(
        mol,
        "",
        atom_colors,
        {},       # bond colours
        atom_radii,
        {},       # bond radii
    )
    drawer.FinishDrawing()

    bio = io.BytesIO(drawer.GetDrawingText())
    return PIL.Image.open(bio)


def plot_confidence_bar(
    mol: Chem.Mol,
    confidence: np.ndarray,
    neff_per_radius: dict[int, np.ndarray] | None = None,
) -> plt.Figure:
    """
    Bar chart of per-atom confidence with optional radius breakdown.

    Shows each heavy atom on x-axis with its element symbol and index,
    stacked/grouped bars for each radius if neff_per_radius provided.
    Useful for detailed analysis beyond the 2D depiction.
    """
    ...
```

---

## 5. Revised Execution Pipeline

```
                  CPU (RDKit + numpy)                │    GPU/CPU (JAX — all static shapes)
                                                     │
 ┌──────────────────────────────────────────────────┐│
 │ 1. Parse query SDF → Mol                         ││
 │ 2. For each radius r:                            ││
 │    decompose(query, r) → AtomDecomposition       ││
 │    build_atom_bit_mask() → (n_atoms, fp_size)    ││
 │ 3. Load database → (N_db, fp_size) per radius    ││
 └───────────────────┬──────────────────────────────┘│
                     │ numpy                          │
                     ▼                                │
 ┌──────────────────────────────────────────────────┐│
 │ 4. jnp.array(db_fps)      [FIXED shape: N_db]   ││
 │ 5. bulk_tanimoto(query, db)       [JIT, N_db]    ││
 └───────────────────┬──────────────────────────────┘│
                     │ sims (N_db,)                   │
                     ▼                                │
 ┌──────────────────────────────────────────────────┐│
 │ 6. filter_references() ON CPU                    ││  ← dynamic indexing here
 │    → padded (max_refs, fp_size) + mask           ││  ← output is STATIC shape
 └───────────────────┬──────────────────────────────┘│
                     │ static-shape arrays             │
                     ▼                                │
 ┌──────────────────────────────────────────────────┐│
 │ 7. inverse_degree_weights(fps, mask, θ, chunk_size)    ││  ← JIT, chunked pairwise
 │    → weights (max_refs,)  [0 for padded]         ││
 │                                                   ││
 │ 8. For each radius r:  [JIT per unique n_atoms]  ││
 │    per_atom_neff(atom_mask, ref_fps, weights)    ││  ← JIT, static shapes
 │    → neff_r (n_atoms,)                           ││
 │                                                   ││
 │ 9. aggregate_neff(neff_per_radius)               ││
 │    → combined_neff (n_atoms,)                    ││
 │                                                   ││
 │ 10. estimate_lambda(combined_neff)               ││  ← adaptive λ
 │ 11. normalise_to_confidence(neff, λ)             ││
 │    → confidence (n_atoms,) in [0, 1]             ││
 └───────────────────┬──────────────────────────────┘│
                     │                                │
                     ▼                                │
 ┌──────────────────────────────────────────────────┐│
 │ 12. NeffResult(atom_neff, atom_confidence, ...)  ││
 │ 13. plot(confidence) ← always normalised         ││
 │ 14. to_sdf() / to_csv()                         ││
 └──────────────────────────────────────────────────┘│
```

**JIT compilation profile for a typical batch of 100 queries:**

| Function | Recompilations | Reason |
|----------|---------------|--------|
| `bulk_tanimoto` | 1 | N_db is fixed per database |
| `inverse_degree_weights` | 1 | max_refs is static config |
| `per_atom_neff_single_radius` | ~25 | One per unique n_atoms value |
| `aggregate_neff` | ~25 | Same n_atoms dependency |
| **Total** | ~52 | Amortised over 100 queries |

After warmup, subsequent batches against the same database get 0 recompilations
(assuming molecules have similar atom counts, which drug-like molecules do).

---

## 6. Results Object (Revised)

```python
# ligand_neff/_types.py (continued)
import chex
from ligand_neff._types import AtomScores


@chex.dataclass(frozen=True, mappable_dataclass=False)
class NeffResult:
    """
    Immutable output of a Neff computation.

    frozen=True because results should not be mutated after creation.
    mappable_dataclass=False because this is a user-facing result object,
    not something we pass through jax.tree_util — it contains RDKit Mol
    objects which are not JAX-compatible leaves.
    """
    query_mol: object              # Chem.Mol — not a JAX type
    config: object                 # NeffConfig
    atom_neff: AtomScores          # (n_atoms,)
    atom_confidence: AtomScores    # (n_atoms,) in [0, 1]
    neff_per_radius: dict          # {int: AtomScores}
    global_neff: float
    global_confidence: float
    n_references_used: int
    lambda_value: float

    def to_sdf(self, path: str) -> None:
        """Write query mol with Neff/confidence as atom properties."""

    def to_csv(self, path: str) -> None:
        """Per-atom table: idx, element, neff_r1, ..., combined, confidence."""

    def plot(self, **kwargs) -> PIL.Image.Image:
        """2D depiction coloured by atom_confidence (always normalised)."""

    def plot_breakdown(self, **kwargs) -> plt.Figure:
        """Bar chart with per-radius breakdown."""
```

---

## 7. Implementation Order (Revised)

### Phase 1: Core with Static Shapes + Typing
[x] 1. `_types.py` — type aliases + NeffResult (chex.dataclass)
[x] 2. `config.py` — dataclass with fp_size validation
[x] 3. `fingerprints/decompose.py` — bit decomposition
[x] 4. `fingerprints/encode.py` — FP → numpy with configurable fp_size
[x] 5. `similarity/tanimoto.py` — JAX bulk Tanimoto (jaxtyped)
[x] 6. `similarity/filtering.py` — **pad + mask** output (chex.dataclass + chex assertions)
[x] 7. `neff/per_atom.py` — mask-aware Neff (jaxtyped + assert_max_traces)
[x] 8. `neff/_state.py` — NeffState (chex.dataclass)
[x] 9. `io/query.py` — SDF loading
[x] 10. `io/database.py` — basic SDF/SMILES loading
[x] 11. `tests/conftest.py` — jaxtyping import hook
[x] 12. Integration: `compute_neff()` single radius, verify no recompilation
[x] 13. **Test**: run 50 queries, assert JIT cache hits after warmup

### Phase 2: Weighting + Multi-Radius
[x] 14. `neff/weighting.py` — **chunked** Inverse Degree (jaxtyped + chex assertions)
[x] 15. `neff/aggregation.py` — multi-radius + **adaptive λ** (jaxtyped + chex assertions)
[x] 16. Multi-radius support in `compute_neff()`
[x] 17. **Test**: memory profiling at max_refs=25K, verify no OOM
[x] 18. **Test**: λ adaptation across small (50 mol) and large (100K mol) DBs

### Phase 3: Outputs + Usability
[x] 19. `NeffResult` — chex.dataclass(frozen=True) with `.to_sdf()`, `.to_csv()`
[x] 20. `vis/plot.py` — **confidence-based** 2D depiction
[x] 21. `vis/plot.py` — bar chart breakdown
[x] 22. `io/database.py` — precomputation + `.npz` save/load
[x] 23. `cli.py` — command-line interface

### Phase 4: Performance + Validation
[ ] 24. Benchmark: JIT warmup profile (compilations vs cache hits)
[ ] 25. Benchmark: GPU vs CPU at fp_size 2048/4096/8192
[ ] 26. Benchmark: chunked vs full pairwise at various max_refs
[ ] 27. Validation
[ ] 28. Documentation + examples

---

## 8. Testing Additions

```
tests/
├── conftest.py               # jaxtyping import hook for beartype checking
├── test_static_shapes.py     # verify no JIT recompilation in batch
├── test_chunked_pairwise.py  # chunked vs full matrix equivalence
├── test_adaptive_lambda.py   # λ scaling with DB size
├── test_fp_sizes.py          # 2048/4096/8192 consistency
├── test_mask_propagation.py  # padded entries contribute 0 to Neff
├── test_trace_counts.py      # NEW: assert_max_traces on JIT'd functions
├── test_decompose.py
├── test_tanimoto.py
├── test_weighting.py
├── test_per_atom.py          # Uses @chex.variants(with_jit, without_jit)
├── test_integration.py
└── test_io.py
```

### Test Infrastructure

```python
# tests/conftest.py
from jaxtyping import install_import_hook
install_import_hook("ligand_neff", "beartype.beartype")
```

All test classes extend `chex.TestCase` and use `@chex.variants` to
run each test under both JIT and non-JIT modes:

```python
# tests/test_per_atom.py
import chex
from absl.testing import absltest
import jax.numpy as jnp


class PerAtomNeffTest(chex.TestCase):

    @chex.variants(with_jit=True, without_jit=True)
    def test_known_output(self):
        atom_mask = jnp.zeros((3, 16))
        atom_mask = atom_mask.at[0, :3].set(1.0)
        ref_fps = jnp.zeros((5, 16))
        ref_fps = ref_fps.at[0, :4].set(1.0)
        weights = jnp.array([1.0, 1.0, 0.0, 0.0, 0.0])

        neff = self.variant(per_atom_neff_single_radius)(
            atom_mask, ref_fps, weights, min_overlap=0.5,
        )
        expected = jnp.array([1.0, 1.5, 1.0])
        chex.assert_trees_all_close(neff, expected, atol=1e-6)

    @chex.variants(with_jit=True, without_jit=True)
    def test_padded_entries_contribute_zero(self):
        """Padded references (weight=0) must not affect Neff."""
        atom_mask = jnp.ones((2, 8))
        ref_fps = jnp.zeros((100, 8)).at[:3].set(1.0)
        weights_none = jnp.zeros(100)

        neff = self.variant(per_atom_neff_single_radius)(
            atom_mask, ref_fps, weights_none,
        )
        chex.assert_trees_all_close(neff, jnp.zeros(2), atol=1e-7)


class TraceCountTest(chex.TestCase):

    def test_inverse_degree_traces_once(self):
        """Inverse Degree with fixed shapes must trace exactly once."""
        chex.clear_trace_counter()
        fps = jnp.ones((100, 2048))
        mask = jnp.ones(100, dtype=bool)
        _ = inverse_degree_weights(fps, mask)
        _ = inverse_degree_weights(fps * 0.5, mask)  # cache hit, no re-trace
```

### Critical new tests

```python
# test_static_shapes.py
def test_no_recompilation_across_queries():
    """10 queries with different ref counts → 0 recompilations after warmup."""

# test_mask_propagation.py
def test_padded_entries_contribute_zero():
    """Padded (100, fp_size) with mask == unpadded (5, fp_size)."""

# test_chunked_pairwise.py
def test_chunked_matches_full_matrix():
    """chunked_neighbor_count(chunk_size=128) == full (500, 500) pairwise."""

# test_adaptive_lambda.py
def test_lambda_scales_with_database():
    """50 refs → low λ; 50K refs → high λ."""
```

---

## 9. Module × Typing Summary

| Module | jaxtyping | chex assertions | chex.dataclass | chex test variants |
|--------|-----------|----------------|----------------|-------------------|
| `_types.py` | Type aliases (incl. `UInt8`) | — | NeffResult | — |
| `similarity/tanimoto.py` | All signatures | assert_rank | — | — |
| `similarity/filtering.py` | Input sig | assert_shape on output | FilteredReferences | — |
| `neff/weighting.py` | All signatures | assert_rank, assert_tree_all_finite, assert_max_traces | — | — |
| `neff/per_atom.py` | All signatures | assert_rank, assert_max_traces | — | — |
| `neff/aggregation.py` | All signatures | assert_equal_shape, assert_scalar_positive_finite | — | — |
| `neff/_state.py` | Field types (stacked tensor) | — | NeffState | — |
| `tests/` | Via import hook | assert_trees_all_close | — | @chex.variants |

### Rules of Thumb

1. **Every function that touches JAX arrays** gets jaxtyping annotations.
2. **Every JIT'd function with fixed-shape inputs** gets `@chex.assert_max_traces`.
3. **Every dataclass that crosses a `jax.jit` boundary** uses `chex.dataclass`.
4. **Every test** uses `@chex.variants(with_jit=True, without_jit=True)`.
5. **Value assertions** (`assert_tree_all_finite`) are used in debug mode via `@chex.chexify`, stripped in production.

### JAX PyTree Leaf Rules (Critical)

| Field type | JAX treatment | Risk if wrong | Mitigation |
|---|---|---|---|
| `jnp.array(x, dtype=jnp.int32)` | Traced dynamic leaf | — (correct) | Use for `n_valid` |
| `int` inside `chex.dataclass` | Static data → recompile on change | Recompilation per unique value | Always use `jnp.array(...)` |
| `dict` inside `chex.dataclass` | PyTree if structure is constant | Ordering fragility, typing pain | Replace with stacked tensor |
| `tuple[float, ...]` (config field) | Not in PyTree (passed as static) | — | Mark as `static_argnames` |
| `float` (config field) | Not in PyTree (passed as static) | — | Mark as `static_argnames` |

### Fingerprint dtype Strategy

| Stage | dtype | Reason |
|---|---|---|
| On disk / CPU load | `uint8` | 4× memory saving vs float32 |
| `FingerprintBatch` in DB arrays | `uint8` | Compact storage |
| Inside JAX kernels (dot products) | `float32` | XLA matmul efficiency |
| Cast point | Entry to each JIT'd function | `fps = fps.astype(jnp.float32)` |

For packed-bit storage (future), replace dot-product Tanimoto with popcount
over packed `uint8` bitvectors — avoids cast entirely and reduces bandwidth
by 8×, at the cost of a more complex kernel.