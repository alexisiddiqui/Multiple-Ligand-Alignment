# implementation_plan.MD — Speeding up `compute_neff` (device-resident, fused, vmapped)

## Context: what’s slow today (and why)

The current `compute_neff` pipeline intentionally does **dynamic filtering on CPU**, then hands **padded static-shaped** arrays to JAX. That design avoids recompilations downstream, but it causes **host↔device ping‑pong** and repeated CPU work:

- `filter_references()` runs `bulk_tanimoto()` on device, then immediately pulls `sims` to CPU (`np.asarray(sims)`) to threshold + argpartition + pad, then pushes padded arrays back to device.  
  (See `ligand_neff/similarity/filtering.py`.) fileciteturn14file1L61-L64
- `compute_neff()` casts/loads DB fingerprints per radius (`.astype(np.float32)`) and converts per‑radius Neff results back to NumPy (`np.asarray(neff_r)`), then pushes back to JAX for aggregation.  
  (See `ligand_neff/compute.py`.) fileciteturn13file13L11-L13 fileciteturn13file13L57-L62

These crossings force device synchronization and destroy throughput. The goal of this plan is to make the **inner method** as strong as possible (single query), and then unlock batching later.

---

## Goals

1. **Zero host↔device round-trips inside the main math path** (filtering → weighting → per-atom Neff → aggregation → normalization).
2. **Load/convert DB once** and keep it device-resident across queries.
3. Reduce Python dispatch overhead by **fusing** into a small number of `@jax.jit` entrypoints.
4. Compute all radii with **`lax.map` first** (safe memory), then optionally switch to **`vmap`** for parallelism.
5. Introduce an **Engine API** that makes the fast path easy and amortizes compilation/caching.

Non-goals (for now): bit-packed/popcount fingerprints, approximate nearest neighbor search, multi-query batching over `n_atoms` via padding, or shape polymorphism.

---

## Proposed architecture at a glance

### Data representations
- **DB**: device-resident `db_fps_stacked` with shape `(n_radii, n_db, fp_size)` and dtype `float32` (or optional `float16`).
- **Query**: `q_fps_stacked` with shape `(n_radii, fp_size)`; `atom_masks_stacked` with shape `(n_radii, n_atoms, fp_size)`.

### Core device pipeline
1. `bulk_tanimoto(q_fp, db_fps)` → `sims` (on device)
2. `lax.top_k(sims, max_refs)` → `(top_sims, top_idx)` (static)
3. threshold mask → `mask = top_sims >= threshold`
4. gather fps → `ref_fps = take(db_fps, top_idx)`
5. weights:
   - `inverse_degree_weights(ref_fps, mask, ...)` **or**
   - `weights = where(mask, 1, 0)`
6. `per_atom_neff_single_radius(atom_mask, ref_fps, weights, ...)`
7. aggregate radii (stacked aggregator)
8. lambda + normalize to confidence
9. return JAX arrays; only convert to NumPy once at the wrapper boundary.

---

## Phases and atomic steps

### Phase 0 — Baseline + acceptance criteria (1 day)
**Outcome:** You can quantify wins and avoid regressions.

1. Add a single “golden” benchmark runner that measures:
   - end-to-end runtime of `compute_neff()`
   - per-stage timings (filtering, weighting, per-atom, aggregation, normalization)
   - host-device sync counts (implicitly via timing gaps)
2. Use existing profiling helpers as scaffolding (there’s already a profiling script under `benchmarking/profiling/compute_neff/`). fileciteturn13file10L65-L70
3. Define acceptance criteria (suggested):
   - **Correctness:** outputs match legacy within tolerance:
     - `atom_neff`, `atom_confidence` close (`rtol=1e-4`, `atol=1e-5`)
     - `n_references_used` identical or explainable (top‑k tie ordering)
   - **Performance:** for a representative DB size (e.g. 10k–100k),
     - at least **2× faster** than legacy for the same config after warm-up
     - no per-radius CPU sync in the hot path (verify by removing `np.asarray` in new path)

---

### Phase 1 — Device-resident DB cache (1–2 days)
**Outcome:** DB is loaded once and stays on device; `compute_neff` no longer `.astype()`s per call.

Atomic steps:
1. Create `ligand_neff/io/db_cache.py`:
   - `load_precomputed_npz(path) -> dict[int, np.ndarray]` (existing behavior)
   - `prepare_db_device(precomputed: dict, fp_radii, dtype=jnp.float32) -> jnp.ndarray`
     - returns `db_fps_stacked: (n_radii, n_db, fp_size)`
2. Add a small `DbCache` dataclass that stores:
   - `fp_radii`, `fp_size`, `n_db`
   - `db_fps_stacked` (device)
   - optional `db_fps_per_radius` (dict) if you want both views
3. Update `compute_neff()` signature (non-breaking):
   - allow `precomputed_db` to be either:
     - path/npz/dict (legacy)
     - **DbCache** (fast path)
4. Add tests:
   - `DbCache` shapes and dtype are correct.
   - `DbCache` rejects mismatched `fp_size` / radii.

Notes:
- If GPU memory is tight, allow `dtype=float16` storage but cast to float32 inside kernels (Phase 8 optional).

---

### Phase 2 — Pure JAX filtering (`lax.top_k`) (1–2 days)
**Outcome:** filtering becomes a JAX kernel; no `np.asarray(sims)` or CPU padding.

Atomic steps:
1. Add a new function (keep the old one for compatibility initially):
   - `ligand_neff/similarity/filtering_jax.py::filter_references_topk(...)`
2. Implementation sketch:

```python
import jax
import jax.numpy as jnp
from functools import partial
from ligand_neff.similarity.tanimoto import bulk_tanimoto
from ligand_neff.similarity.filtering import FilteredReferences

@partial(jax.jit, static_argnames=("max_refs",))
def filter_references_topk(query_fp, db_fps, threshold: float, max_refs: int):
    sims = bulk_tanimoto(query_fp, db_fps)                    # (n_db,)
    top_sims, top_idx = jax.lax.top_k(sims, max_refs)         # (max_refs,)
    mask = top_sims >= threshold                              # (max_refs,)
    ref_fps = jnp.take(db_fps, top_idx, axis=0)               # (max_refs, fp_size)

    # Optional: zero out padded rows for determinism/debugging
    ref_fps = ref_fps * mask[:, None]
    top_sims = jnp.where(mask, top_sims, 0.0)

    return FilteredReferences(
        fps=ref_fps,
        mask=mask,
        similarities=top_sims,
        n_valid=jnp.sum(mask, dtype=jnp.int32),
    )
```

3. Edge-case handling:
   - `n_db < max_refs`: either (a) assert, or (b) pad DB once in `DbCache` to `max_refs`.
   - ties in `top_k`: accept nondeterministic ordering (but stable count/scores).
4. Tests to add:
   - Compare `n_valid` with legacy `filter_references` on random data.
   - Ensure selected sims are among the global top‑k and obey threshold.
   - Ensure output shapes are static and JIT compiles once for fixed shapes.

---

### Phase 3 — Stacked aggregation + keep everything in JAX until the end (1 day)
**Outcome:** remove dict construction + repeated conversions; make aggregation JAX-friendly.

Atomic steps:
1. Add `aggregate_neff_stacked(neff_stacked, method, radius_weights)`:
   - `neff_stacked: (n_radii, n_atoms)`
   - reuse the existing math from `aggregate_neff()` but avoid dict sorting/stacking.
2. Update the legacy `aggregate_neff()` to call the stacked version internally (optional).
3. Update `compute_neff()` wrapper logic:
   - stop doing `np.asarray(neff_r)` per radius
   - stop converting back to JAX dict for aggregation
4. Add tests:
   - `aggregate_neff_stacked` matches `aggregate_neff(dict)` bit-for-bit (or within tolerance).

---

### Phase 4 — Single-radius pipeline kernel (1–2 days)
**Outcome:** one jitted function for a single radius = filter → weights → per-atom.

Atomic steps:
1. Create `_single_radius_pipeline(...)`:
   - inputs: `q_fp (fp_size,)`, `db_fps (n_db, fp_size)`, `atom_mask (n_atoms, fp_size)`
   - outputs: `neff_r (n_atoms,)`, `n_valid scalar`
2. Implement weighting as **compile-time selection**:
   - easiest: build two pipelines:
     - `_single_radius_pipeline_unweighted`
     - `_single_radius_pipeline_inverse_degree`
   - and let the engine select which to call (avoids `lax.cond` + extra compiled branches).
3. Validate that `inverse_degree_weights` remains stable under the new ref_fps/mask and that it doesn’t recompile.

---

### Phase 5 — Multi-radius device pipeline (fused JIT + `lax.map` then `vmap`) (2–4 days)
**Outcome:** radii loop leaves Python entirely; pipeline becomes a single jitted graph per query.

Atomic steps:
1. Implement `_compute_neff_core(...)` returning **JAX outputs only**:

Inputs:
- `q_fps_stacked: (n_radii, fp_size)`
- `atom_masks_stacked: (n_radii, n_atoms, fp_size)`
- `db_fps_stacked: (n_radii, n_db, fp_size)`
- config scalars/tuples (passed as static args where appropriate)

Outputs:
- `combined_neff: (n_atoms,)`
- `confidence: (n_atoms,)`
- `neff_per_radius: (n_radii, n_atoms)`
- `lam: scalar`
- `n_refs_used: scalar` (max n_valid across radii)

2. Start with `lax.map` (safe memory) over radii:

```python
def body(i):
    q = q_fps_stacked[i]
    db = db_fps_stacked[i]
    am = atom_masks_stacked[i]
    neff_r, n_valid = _single_radius_pipeline(q, db, am)
    return neff_r, n_valid

(neff_stack, n_valids) = jax.lax.map(body, jnp.arange(n_radii))
```

3. Once correct and stable, optionally switch to `vmap`:

```python
neff_stack, n_valids = jax.vmap(_single_radius_pipeline)(q_fps_stacked, db_fps_stacked, atom_masks_stacked)
```

4. Replace lambda logic with a fully-JAX scalar:
   - fixed: `lam = jnp.asarray(lambda_fixed, jnp.float32)`
   - adaptive: `lam = jnp.maximum(jnp.quantile(combined, q), 1e-3)`
5. Ensure the only host conversions are at the end of the Python wrapper:
   - `np.asarray(combined_neff)`
   - `np.asarray(confidence)`
   - `float(jnp.mean(...))` (or compute on device then `float(...)`)

---

### Phase 6 — Engine API (2–3 days)
**Outcome:** users can do: “load DB once → run many queries fast”.

#### Public API design
Create `ligand_neff/engine.py`:

```python
from dataclasses import dataclass
from typing import Sequence, Optional
import numpy as np
import jax
import jax.numpy as jnp
from rdkit import Chem

from ligand_neff.config import NeffConfig
from ligand_neff._types import QueryData, NeffResult

@dataclass(frozen=True)
class PreparedQuery:
    # device arrays, ready for the core kernel
    q_fps: jnp.ndarray          # (n_radii, fp_size)
    atom_masks: jnp.ndarray     # (n_radii, n_atoms, fp_size)
    n_atoms: int

class NeffEngine:
    def __init__(
        self,
        config: NeffConfig,
        *,
        precomputed_db: str | dict | None = None,
        db_mols: Optional[Sequence[Chem.Mol]] = None,
        dtype=jnp.float32,
        device=None,
        compile_on_init: bool = True,
    ):
        self.config = config
        self.fp_radii = tuple(config.fp_radii)
        self.device = device or jax.devices()[0]

        # 1) build DbCache on device
        self.db_cache = build_db_cache(config, precomputed_db=precomputed_db, db_mols=db_mols, dtype=dtype, device=self.device)

        # 2) select a core function variant (weighting/mode/aggregation)
        self._core = build_compiled_core(config) if compile_on_init else build_core_uncompiled(config)

        # 3) optional warmup (compile for a typical n_atoms)
        if compile_on_init:
            self.warmup(n_atoms_hint=32)

    def warmup(self, n_atoms_hint: int = 32):
        q = jnp.zeros((len(self.fp_radii), self.config.fp_size), dtype=jnp.float32)
        am = jnp.zeros((len(self.fp_radii), n_atoms_hint, self.config.fp_size), dtype=jnp.float32)
        _ = self._core(q, am, self.db_cache.db_fps_stacked).block_until_ready()

    def prepare_query(self, query_mol: Chem.Mol) -> PreparedQuery:
        # leverage existing prepare_query_data() then stack + device_put
        qd = prepare_query_data(query_mol, self.config)
        q_fps = jnp.stack([jnp.asarray(qd.fps[r], dtype=jnp.float32) for r in self.fp_radii], axis=0)
        atom_masks = jnp.stack([jnp.asarray(qd.atom_masks[r], dtype=jnp.float32) for r in self.fp_radii], axis=0)
        return PreparedQuery(q_fps=q_fps, atom_masks=atom_masks, n_atoms=qd.n_atoms)

    def compute_prepared(self, prepared: PreparedQuery, *, query_mol=None) -> NeffResult:
        combined, conf, per_r, lam, nrefs = self._core(prepared.q_fps, prepared.atom_masks, self.db_cache.db_fps_stacked)

        # host conversion boundary
        combined_np = np.asarray(combined)
        conf_np = np.asarray(conf)
        per_r_np = np.asarray(per_r)   # (n_radii, n_atoms)

        neff_dict = {r: per_r_np[i] for i, r in enumerate(self.fp_radii)}

        return NeffResult(
            query_mol=query_mol,
            config=self.config,
            atom_neff=combined_np,
            atom_confidence=conf_np,
            neff_per_radius=neff_dict,
            global_neff=float(combined.mean()),
            global_confidence=float(conf.mean()),
            n_references_used=int(np.asarray(nrefs)),
            lambda_value=float(np.asarray(lam)),
        )

    def compute(self, query_mol: Chem.Mol) -> NeffResult:
        prepared = self.prepare_query(query_mol)
        return self.compute_prepared(prepared, query_mol=query_mol)
```

Atomic steps:
1. Implement `DbCache` (Phase 1) and expose `build_db_cache(...)`.
2. Implement `PreparedQuery` and conversion helpers:
   - `stack_query_data(query_data, fp_radii) -> PreparedQuery`
3. Implement `build_compiled_core(config)`:
   - returns a `jax.jit`’d function with static args baked-in
   - choose specialized variants by:
     - weighting mode (`inverse_degree` vs `none`)
     - lambda mode (`fixed` vs `adaptive`)
     - aggregation method (`geometric`/`minimum`/`mean`)
4. Ensure `compute_neff()` remains supported:
   - legacy path stays as-is
   - optionally, if `precomputed_db` is a `DbCache`, route to `NeffEngine.compute_prepared` internally

---

### Phase 7 — Correctness, regression tests, and benchmarks (1–2 days)
**Outcome:** confidence to ship.

Atomic steps:
1. Add a “fast vs legacy” test:
   - small synthetic DB
   - random query (or RDKit mol)
   - compare outputs (tolerances; handle top‑k tie reorder)
2. Add a “no ping-pong” smoke test:
   - run `engine.compute_prepared(...)` and ensure it doesn’t call `np.asarray` internally (use monkeypatch or code inspection).
3. Add benchmark suite entries:
   - legacy `compute_neff`
   - engine fast path (warm + cold)
   - report speedup and memory

---

## Phase 8 — Optional next-level optimizations (backlog)
These can be tackled after the JAX-only pipeline lands.

1. **Bit-packed fingerprints + popcount** (huge speed potential):
   - store fingerprints as `uint32` blocks
   - intersection via `bitwise_and` + `popcount`
   - union via counts
2. **Streaming top‑k over chunked DB**:
   - handle massive `n_db` without materializing full `sims`
3. **Batch queries (`vmap` over queries)**:
   - requires padding `n_atoms` or using shape polymorphism
4. **Mixed precision**:
   - store DB in `float16` and cast to float32 at dot boundaries
5. **Approximate candidate pruning**:
   - two-stage: coarse hash → exact tanimoto on reduced set

---

## Checklist for “Done”

- [ ] `filter_references_topk` is in-tree and covered by tests.
- [ ] Engine loads DB once, keeps it on device, and runs without CPU sync in hot path.
- [ ] `compute_neff` wrapper can use Engine without breaking CLI / examples.
- [ ] Benchmarks show meaningful speedup after warm-up on representative workloads.
- [ ] Code paths remain maintainable (clear split between “public API wrapper” and “compiled core”).
