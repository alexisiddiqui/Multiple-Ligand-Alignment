"""Shared utilities for JIT warmup benchmarks.

Contains CompilationCounter, count_compilations context manager,
and benchmark constants used across test modules.
"""

from contextlib import contextmanager


# ── Benchmark Constants ─────────────────────────────────────────────────────

BENCH_FP_SIZE = 2048
BENCH_MAX_REFS = 512       # power of 2, divides chunk_size evenly
BENCH_CHUNK_SIZE = 128
BENCH_N_DB = 1000
BENCH_N_ATOMS_DEFAULT = 30


# ── Compilation Counter ─────────────────────────────────────────────────────

class CompilationCounter:
    """Counts JAX backend compilations using the monitoring API.

    Filters on `/jax/core/compile/backend_compile_duration` events,
    optionally matching a specific `fun_name` pattern.
    """

    def __init__(self, fun_name_contains: str | None = None):
        self.fun_name_contains = fun_name_contains
        self.count = 0
        self._listener_id = None

    def _listener(self, event: str, duration: float, **kwargs):
        if event != "/jax/core/compile/backend_compile_duration":
            return
        if self.fun_name_contains is not None:
            fun_name = kwargs.get("fun_name", "")
            if self.fun_name_contains not in fun_name:
                return
        self.count += 1

    def start(self):
        from jax._src import monitoring
        self.count = 0
        monitoring.register_event_duration_secs_listener(self._listener)
        self._listener_id = self._listener

    def stop(self):
        from jax._src import monitoring
        if self._listener_id is not None:
            monitoring.unregister_event_duration_listener(self._listener_id)
            self._listener_id = None

    def reset(self):
        self.count = 0


@contextmanager
def count_compilations(fun_name_contains: str | None = None):
    """Context manager that yields a CompilationCounter.

    Usage::

        with count_compilations("bulk_tanimoto") as cc:
            _ = bulk_tanimoto(q, db)
            _ = bulk_tanimoto(q2, db)
        assert cc.count == 1  # compiled once, second call was cache hit
    """
    cc = CompilationCounter(fun_name_contains)
    cc.start()
    try:
        yield cc
    finally:
        cc.stop()
