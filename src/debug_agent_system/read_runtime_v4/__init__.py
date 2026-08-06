"""Evidence Investigation Runtime v4.

v4 is an additive, shadow-first read-side runtime.  It reuses the v3
read-only providers and evidence fabric, but makes investigation state the
primary output instead of concatenating a frozen answer with provider output.
"""

from .config import ReadRuntimeV4Options, load_options
from .runtime import ReadRuntimeV4

__all__ = ["ReadRuntimeV4", "ReadRuntimeV4Options", "load_options"]
