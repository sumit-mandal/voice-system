"""Debug-first logging setup used by API and LiveKit worker."""

from __future__ import annotations

import logging
import sys


def setup_logging(level: str = "DEBUG") -> None:
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level.upper())
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root.addHandler(handler)
    root.setLevel(level.upper())

    # Keep third-party noise down unless we are debugging our own code.
    logging.getLogger("httpx").setLevel(logging.INFO)
    logging.getLogger("httpcore").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)

    logging.getLogger(__name__).debug("Logging configured at level=%s", level.upper())


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
