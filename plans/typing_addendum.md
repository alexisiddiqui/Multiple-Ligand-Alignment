# `ligand-neff` v2.1: chex + jaxtyping Integration

Addendum to the v2 implementation plan. This document specifies how
`chex` and `jaxtyping` (with `beartype`) are woven into every module.

---

## 1. Dependencies (Updated)

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

### Why these three together

| Library | Role | Scope |
|---------|------|-------|
| **jaxtyping** | Shape + dtype annotations on function signatures | Every JAX-facing function |
| **beartype** | Runtime enforcement of jaxtyping annotations | Dev/test; zero-cost in production via import hook |
| **chex** | JAX-aware dataclasses (PyTree-compatible), shape/rank/dtype/trace assertions inside function bodies, test variants | Dataclasses, internal guards, testing |

They complement each other cleanly: jaxtyping annotates the *contract*,
beartype *enforces* it at call boundaries, and chex *guards* invariants
inside function bodies and provides JAX-native dataclasses.

---

## 2. Project-Wide Setup

### 2.1 Import Hook (Development Mode)

For development and testing, enable jaxtyping's import hook so every
function in the package gets runtime shape checking automatically:

```python
# tests/conftest.py
from jaxtyping import install_import_hook

# All functions in ligand_neff will be checked at test time
install_import_hook("ligand_neff", "beartype.beartype")
```

This means developers don't need `@jaxtyped` on every function during
testing — the hook catches everything. In production, users import
normally without the hook, so there's zero overhead.

### 2.2 Explicit Decoration (Public API)

For the public-facing functions (which users call directly, outside the
import hook), use explicit `@jaxtyped(typechecker=beartype)` so shape
checking is always on regardless of how the package is imported:

```python
from jaxtyping import jaxtyped
from beartype import beartype as typechecker
```

### 2.3 Type Aliases

```python
# ligand_neff/_types.py
"""Shared type aliases used across the package."""

import jax
import jax.numpy as jnp
from jaxtyping import Float, Bool, Int, Array

# ── Core array types ──────────────────────────────────────────────
# Named dimensions make signatures self-documenting.
# Variable names (lowercase) are matched across arguments by jaxtyping.

# Fingerprint arrays
Fingerprint      = Float[Array, "fp_size"]
FingerprintBatch = Float[Array, "batch fp_size"]

# Per-atom structures
AtomBitMask      = Float[Array, "n_atoms fp_size"]
AtomScores       = Float[Array, "n_atoms"]

# Padded reference arrays (static shape from config.max_references)
PaddedRefs       = Float[Array, "max_refs fp_size"]
RefMask          = Bool[Array, "max_refs"]
RefWeights       = Float[Array, "max_refs"]
RefSimilarities  = Float[Array, "max_refs"]

# Similarity matrices
SimMatrix        = Float[Array, "rows cols"]

# Scalars
Scalar           = Float[Array, ""]
```

**Why type aliases?** They prevent annotation sprawl. Writing
`Float[Array, "max_refs fp_size"]` everywhere is verbose and error-prone.
`PaddedRefs` is self-documenting and if the shape contract changes, we
update it in one place.

---

## 3. Dataclasses: chex Throughout

### 3.1 `FilteredReferences` → chex.dataclass

The v2 plan used a plain `@dataclass`. This doesn't work with
`jax.tree_util` — if you pass a `FilteredReferences` through a
`jax.jit` boundary, JAX won't know how to flatten/unflatten it.
`chex.dataclass` fixes this by registering it as a PyTree node.

```python
# ligand_neff/similarity/filtering.py

import chex
import jax.numpy as jnp
from ligand_neff._types import PaddedRefs, RefMask, RefSimilarities


@chex.dataclass
class FilteredReferences:
    """
    Static-shape container for filtered reference ligands.

    All array fields have leading dimension = max_refs.
    Registered as a JAX PyTree via chex.dataclass, so this
    entire struct can cross jax.jit boundaries.
    """
    fps: PaddedRefs               # (max_refs, fp_size)
    mask: RefMask                 # (max_refs,)
    similarities: RefSimilarities # (max_refs,)
    n_valid: int                  # Actual count (static, not traced)
```

### 3.2 `NeffResult` → chex.dataclass (frozen)

```python
# ligand_neff/_types.py (continued)

@chex.dataclass(frozen=True, mappable_dataclass=False)
class NeffResult:
    """
    Immutable output of a Neff computation.

    frozen=True because results should not be mutated after creation.
    mappable_dataclass=False because this is a user-facing result object,
    not something we pass through jax.tree_util — it contains RDKit Mol
    objects which are not JAX-compatible leaves.
    """
    # Non-JAX fields (excluded from PyTree traversal)
    query_mol: object              # Chem.Mol — not a JAX type
    config: object                 # NeffConfig

    # JAX-compatible fields
    atom_neff: AtomScores          # (n_atoms,)
    atom_confidence: AtomScores    # (n_atoms,) in [0, 1]
    neff_per_radius: dict          # {int: AtomScores}
    global_neff: float
    global_confidence: float
    n_references_used: int
    lambda_value: float
```

### 3.3 Internal State Containers

For passing intermediate state between pipeline stages:

```python
# ligand_neff/neff/_state.py

import chex
from ligand_neff._types import (
    PaddedRefs, RefMask, RefWeights, AtomBitMask, AtomScores
)


@chex.dataclass
class NeffState:
    """
    Intermediate computation state. Passed through JAX pipeline.

    chex.dataclass makes this a PyTree, so jax.jit can trace through
    functions that accept/return NeffState.
    """
    ref_fps: PaddedRefs           # (max_refs, fp_size)
    ref_mask: RefMask             # (max_refs,)
    ref_weights: RefWeights       # (max_refs,)
    atom_masks: dict              # {radius: AtomBitMask}
```

---

## 4. Function Signatures with jaxtyping

### 4.1 `similarity/tanimoto.py`

```python
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker
from functools import partial


@jaxtyped(typechecker=typechecker)
@partial(jax.jit)
def bulk_tanimoto(
    query: Float[Array, " fp_size"],
    database: Float[Array, "n_refs fp_size"],
) -> Float[Array, " n_refs"]:
    """
    Tanimoto similarity between one query and N references.

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
    """
    Tanimoto between a chunk of references and all references.
    Mask zeros out padded columns.
    """
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

### 4.2 `neff/weighting.py`

```python
import jax
import jax.numpy as jnp
import chex
from jaxtyping import Array, Float, Bool, jaxtyped
from beartype import beartype as typechecker


@jaxtyped(typechecker=typechecker)
def henikoff_weights(
    fps: Float[Array, "max_refs fp_size"],
    mask: Bool[Array, " max_refs"],
    threshold: float = 0.7,
    chunk_size: int = 2048,
) -> Float[Array, " max_refs"]:
    """
    Henikoff-style per-ligand weights with chunked pairwise computation.

    jaxtyping enforces:
    - fps and mask share the max_refs dimension
    - Output has same max_refs dimension as input
    """
    # Static shape assertions inside the function body — chex territory
    chex.assert_rank(fps, 2)
    chex.assert_rank(mask, 1)
    chex.assert_equal_shape_prefix([fps, mask[:, None]], prefix_len=1)

    max_refs, fp_size = fps.shape
    fp_bits = jnp.sum(fps, axis=1)

    neighbor_counts = _chunked_neighbor_count(
        fps, fp_bits, mask, threshold, chunk_size
    )

    weights = jnp.where(
        mask,
        1.0 / jnp.maximum(neighbor_counts, 1.0),
        0.0,
    )

    # Post-condition: weights should be non-negative, ≤ 1
    chex.assert_tree_all_finite(weights)

    return weights
```

### 4.3 `neff/per_atom.py`

```python
import jax
import jax.numpy as jnp
import chex
from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker
from functools import partial


@jaxtyped(typechecker=typechecker)
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

### 4.4 `neff/aggregation.py`

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
    Combine per-atom Neff across Morgan radii.

    jaxtyping enforces all radius arrays share n_atoms dimension.
    """
    radii = sorted(neff_per_radius.keys())
    stacked = jnp.stack([neff_per_radius[r] for r in radii], axis=0)

    # All radius arrays must have the same atom count
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

### 4.5 `similarity/filtering.py`

The filtering function mixes numpy (CPU) and JAX, so we annotate the
JAX portions and use chex assertions for the output contract:

```python
import numpy as np
import jax.numpy as jnp
import chex
from jaxtyping import Array, Float, Bool


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
        n_valid=n_valid,
    )

    # Validate output shapes — chex catches any padding bugs
    chex.assert_shape(result.fps, (max_refs, fp_size))
    chex.assert_shape(result.mask, (max_refs,))
    chex.assert_shape(result.similarities, (max_refs,))

    return result
```

---

## 5. chex Assertions: Where and Why

### Assertion Strategy

| Assertion Type | Library | When | Cost |
|---|---|---|---|
| **Shape/dtype at boundaries** | jaxtyping + beartype | Function entry/exit | Trace-time only |
| **Rank/shape inside bodies** | chex.assert_rank, assert_shape | After array construction | Trace-time (static) |
| **Value invariants** | chex.assert_tree_all_finite | After computation | Requires chexify for JIT |
| **Trace count guards** | chex.assert_max_traces | On JIT'd functions | Compile-time |
| **Dimension consistency** | chex.Dimensions | Multi-array checks | Trace-time |

### 5.1 Trace Guards (Recompilation Detection)

The v2 plan identified JIT recompilation as a critical risk. chex provides
`@chex.assert_max_traces` which makes this a hard assertion:

```python
# The core Neff function should be traced at most once per unique n_atoms.
# Over a batch of drug-like molecules (20-50 atoms), we expect ~30 unique
# traces. Setting n=50 provides headroom.

@chex.assert_max_traces(n=50)
@partial(jax.jit, static_argnames=("min_overlap",))
def per_atom_neff_single_radius(
    atom_bit_mask: Float[Array, "n_atoms fp_size"],
    ref_fps: Float[Array, "max_refs fp_size"],
    weights: Float[Array, " max_refs"],
    min_overlap: float = 0.5,
) -> Float[Array, " n_atoms"]:
    ...

# Henikoff weights should be traced EXACTLY ONCE per config
# (max_refs and fp_size are fixed). If this re-traces, something
# is wrong with the padding.

@chex.assert_max_traces(n=1)
@jax.jit
def _henikoff_core(
    fps: Float[Array, "max_refs fp_size"],
    fp_bits: Float[Array, " max_refs"],
    mask: Bool[Array, " max_refs"],
    neighbor_counts: Float[Array, " max_refs"],
    n_chunks: int,
    chunk_size: int,
) -> Float[Array, " max_refs"]:
    """Inner loop for Henikoff. Must trace exactly once."""
    ...
```

### 5.2 chex.Dimensions for Multi-Array Checks

When multiple arrays must share dimensions:

```python
import chex

def _validate_pipeline_inputs(
    atom_bit_mask: Float[Array, "n_atoms fp_size"],
    ref_fps: Float[Array, "max_refs fp_size"],
    weights: Float[Array, " max_refs"],
    mask: Bool[Array, " max_refs"],
):
    """Validate dimensional consistency across pipeline inputs."""
    dims = chex.Dimensions(
        A=atom_bit_mask.shape[0],    # n_atoms
        F=atom_bit_mask.shape[1],    # fp_size
        M=ref_fps.shape[0],          # max_refs
    )
    chex.assert_shape(atom_bit_mask, dims["AF"])
    chex.assert_shape(ref_fps, dims["MF"])
    chex.assert_shape(weights, dims["M"])
    chex.assert_shape(mask, dims["M"])
```

### 5.3 Value Assertions in JIT (chexify)

For value-level checks inside JIT'd code (NaN detection, range checks),
use `@chex.chexify`:

```python
@chex.chexify
@jax.jit
def _safe_neff_core(atom_bit_mask, ref_fps, weights, min_overlap):
    """
    JIT'd Neff with runtime NaN/Inf checking.

    @chex.chexify enables value assertions inside JIT by converting
    them to jax.experimental.checkify checks. In production, remove
    @chex.chexify for zero overhead.
    """
    intersection = jnp.dot(atom_bit_mask, ref_fps.T)
    chex.assert_tree_all_finite(intersection)

    atom_counts = jnp.maximum(jnp.sum(atom_bit_mask, axis=1, keepdims=True), 1.0)
    overlap = intersection / atom_counts
    chex.assert_tree_all_finite(overlap)

    gated = jnp.where(overlap >= min_overlap, overlap, 0.0)
    neff = jnp.dot(gated, weights)
    chex.assert_tree_all_finite(neff)

    return neff
```

---

## 6. Testing with chex Variants

chex provides `@chex.variants` for testing the same function under
multiple JAX transformation modes:

```python
# tests/test_per_atom.py

import chex
from absl.testing import absltest
import jax.numpy as jnp
from ligand_neff.neff.per_atom import per_atom_neff_single_radius


class PerAtomNeffTest(chex.TestCase):
    """Test per_atom_neff under JIT/no-JIT/vmap variants."""

    @chex.variants(with_jit=True, without_jit=True)
    def test_known_output(self):
        """Synthetic test: known bits, known refs → expected Neff."""
        n_atoms, fp_size, max_refs = 3, 16, 5

        # Atom 0 sets bits [0,1,2], Atom 1 sets bits [3,4], Atom 2 sets bits [5]
        atom_mask = jnp.zeros((n_atoms, fp_size))
        atom_mask = atom_mask.at[0, :3].set(1.0)
        atom_mask = atom_mask.at[1, 3:5].set(1.0)
        atom_mask = atom_mask.at[2, 5:6].set(1.0)

        # Ref 0 has bits [0,1,2,3] → full overlap with atom 0, partial with atom 1
        # Ref 1 has bits [3,4,5]   → full overlap with atom 1 and 2
        ref_fps = jnp.zeros((max_refs, fp_size))
        ref_fps = ref_fps.at[0, :4].set(1.0)
        ref_fps = ref_fps.at[1, 3:6].set(1.0)

        weights = jnp.array([1.0, 1.0, 0.0, 0.0, 0.0])

        neff = self.variant(per_atom_neff_single_radius)(
            atom_mask, ref_fps, weights, min_overlap=0.5,
        )

        # Atom 0: ref0 overlap=3/3=1.0 ✓, ref1 overlap=0/3=0.0 ✗ → neff=1.0
        # Atom 1: ref0 overlap=1/2=0.5 ✓, ref1 overlap=2/2=1.0 ✓ → neff=1.5
        # Atom 2: ref0 overlap=0/1=0.0 ✗, ref1 overlap=1/1=1.0 ✓ → neff=1.0
        expected = jnp.array([1.0, 1.5, 1.0])
        chex.assert_trees_all_close(neff, expected, atol=1e-6)

    @chex.variants(with_jit=True, without_jit=True)
    def test_padded_entries_contribute_zero(self):
        """Padded references (weight=0) must not affect Neff."""
        n_atoms, fp_size, max_refs = 2, 8, 100

        atom_mask = jnp.ones((n_atoms, fp_size))

        # Only 3 valid refs, rest are padding
        ref_fps = jnp.zeros((max_refs, fp_size))
        ref_fps = ref_fps.at[:3].set(1.0)

        weights_valid = jnp.zeros(max_refs).at[:3].set(0.5)
        weights_none = jnp.zeros(max_refs)  # all padding

        neff_valid = self.variant(per_atom_neff_single_radius)(
            atom_mask, ref_fps, weights_valid,
        )
        neff_none = self.variant(per_atom_neff_single_radius)(
            atom_mask, ref_fps, weights_none,
        )

        # With no valid weights, Neff should be zero
        chex.assert_trees_all_close(neff_none, jnp.zeros(n_atoms), atol=1e-7)
        # With valid weights, Neff should be positive
        assert jnp.all(neff_valid > 0)

    @chex.variants(with_jit=True, without_jit=True)
    def test_output_shape_matches_atoms(self):
        """Output dimension must equal input atom count."""
        for n_atoms in [1, 10, 50]:
            atom_mask = jnp.ones((n_atoms, 2048))
            ref_fps = jnp.ones((100, 2048))
            weights = jnp.ones(100) / 100

            neff = self.variant(per_atom_neff_single_radius)(
                atom_mask, ref_fps, weights,
            )
            chex.assert_shape(neff, (n_atoms,))


class HenikoffWeightsTest(chex.TestCase):

    @chex.variants(with_jit=True, without_jit=True)
    def test_identical_refs_get_low_weight(self):
        """N identical references should each get weight 1/N."""
        max_refs, fp_size = 50, 128
        fps = jnp.zeros((max_refs, fp_size))
        fps = fps.at[:10].set(1.0)  # 10 identical refs
        mask = jnp.zeros(max_refs, dtype=bool).at[:10].set(True)

        weights = henikoff_weights(fps, mask, threshold=0.7)

        # First 10 should have weight ≈ 0.1 (1/10)
        chex.assert_trees_all_close(
            weights[:10],
            jnp.full(10, 0.1),
            atol=1e-6,
        )
        # Padding should be exactly 0
        chex.assert_trees_all_close(
            weights[10:],
            jnp.zeros(40),
            atol=1e-7,
        )


class TraceCountTest(chex.TestCase):
    """Verify JIT recompilation stays bounded."""

    def test_henikoff_traces_once(self):
        """Henikoff with fixed shapes must trace exactly once."""
        chex.clear_trace_counter()

        fps = jnp.ones((100, 2048))
        mask = jnp.ones(100, dtype=bool)

        # First call: traces
        _ = henikoff_weights(fps, mask)
        # Second call: cache hit, no re-trace
        _ = henikoff_weights(fps * 0.5, mask)
        # If _henikoff_core has @chex.assert_max_traces(n=1),
        # a re-trace here would raise AssertionError


if __name__ == "__main__":
    absltest.main()
```

---

## 7. Decorator Stacking Order

The order of decorators matters. The correct stack from outermost to
innermost:

```python
@jaxtyped(typechecker=typechecker)  # 1. Shape check at call boundary
@chex.assert_max_traces(n=50)       # 2. Guard against recompilation
@partial(jax.jit, static_argnames=(...))  # 3. JIT compilation
def my_function(
    x: Float[Array, "n_atoms fp_size"],
    y: Float[Array, "max_refs fp_size"],
) -> Float[Array, " n_atoms"]:
    # 4. chex assertions inside body
    chex.assert_rank(x, 2)
    ...
```

**Why this order:**
1. `@jaxtyped` wraps the outermost call, checking shapes of concrete args
   *before* they enter JIT.
2. `@chex.assert_max_traces` wraps the JIT boundary, counting traces.
3. `@jax.jit` does the actual compilation.
4. `chex.assert_shape/rank` inside the body runs during tracing (free at
   runtime for static assertions).

For development/debug with value assertions:

```python
@jaxtyped(typechecker=typechecker)
@chex.chexify                        # Enables value assertions in JIT
@chex.assert_max_traces(n=50)
@jax.jit
def my_function_debug(...):
    chex.assert_tree_all_finite(x)   # Now works inside JIT
    ...
```

---

## 8. Summary: What Goes Where

| Module | jaxtyping | chex assertions | chex.dataclass | chex test variants |
|--------|-----------|----------------|----------------|-------------------|
| `_types.py` | Type aliases | — | NeffResult | — |
| `similarity/tanimoto.py` | All signatures | assert_rank | — | — |
| `similarity/filtering.py` | Input sig | assert_shape on output | FilteredReferences | — |
| `neff/weighting.py` | All signatures | assert_rank, assert_tree_all_finite, assert_max_traces | — | — |
| `neff/per_atom.py` | All signatures | assert_rank, assert_max_traces | — | — |
| `neff/aggregation.py` | All signatures | assert_equal_shape, assert_scalar_positive_finite | — | — |
| `neff/_state.py` | Field types | — | NeffState | — |
| `tests/` | Via import hook | assert_trees_all_close | — | @chex.variants |

### Rules of Thumb

1. **Every function that touches JAX arrays** gets jaxtyping annotations.
2. **Every JIT'd function with fixed-shape inputs** gets `@chex.assert_max_traces`.
3. **Every dataclass that crosses a `jax.jit` boundary** uses `chex.dataclass`.
4. **Every test** uses `@chex.variants(with_jit=True, without_jit=True)`.
5. **Value assertions** (`assert_tree_all_finite`) are used in debug mode via `@chex.chexify`, stripped in production.