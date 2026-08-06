"""Small helpers for reproducible trace fields."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator


@contextmanager
def timer_ms() -> Iterator[dict[str, int]]:
    start = time.time()
    box = {"latency_ms": 0}
    try:
        yield box
    finally:
        box["latency_ms"] = int((time.time() - start) * 1000)
