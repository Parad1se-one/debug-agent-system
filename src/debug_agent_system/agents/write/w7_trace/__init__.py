"""W7 multi-decision-agent shadow pipeline.

The package is intentionally fail-closed.  Model stages emit evidence-bounded
decisions; only :class:`TraceCompiler` may materialize execution semantics.
"""

from .contracts import W7_MODES, resolve_w7_mode
from .batch_orchestrator import W7BatchShadowOrchestrator
from .batch_candidate import build_w7_batch_typed_candidate
from .correction_compiler import (
    compile_trace_corrections,
    materialize_corrected_typed_candidate,
)
from .trace_compiler import TraceCompiler

__all__ = [
    "TraceCompiler",
    "W7BatchShadowOrchestrator",
    "build_w7_batch_typed_candidate",
    "W7_MODES",
    "compile_trace_corrections",
    "materialize_corrected_typed_candidate",
    "resolve_w7_mode",
]
