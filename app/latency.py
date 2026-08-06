"""Tiny latency logger — always prints to stdout so LiveKit can't hide it."""

from __future__ import annotations

import sys
import time
from contextlib import contextmanager
from typing import Iterator

from app.logging_setup import get_logger

log = get_logger(__name__)


def log_latency(label: str, ms: float, **extra: object) -> None:
    """Emit one LATENCY line to stdout + logging."""
    extras = " ".join(f"{k}={v}" for k, v in extra.items())
    line = f"LATENCY {label}={ms:.1f}ms" + (f" {extras}" if extras else "")
    # Force visible in worker/API terminals even if logging is reconfigured.
    print(line, flush=True, file=sys.stdout)
    log.info(line)


@contextmanager
def timed(label: str, **extra: object) -> Iterator[None]:
    t0 = time.perf_counter()
    try:
        yield
    finally:
        log_latency(label, (time.perf_counter() - t0) * 1000.0, **extra)
