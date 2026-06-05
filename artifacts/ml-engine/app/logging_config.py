"""Structured JSON logging for the ML engine.

Every request produces one log record with coin, timeframe, duration, and
feature vector hash so later phases can audit calls.
"""
from __future__ import annotations

import logging
import sys

from pythonjsonlogger.json import JsonFormatter


def configure_logging() -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        JsonFormatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s",
            rename_fields={"asctime": "ts", "levelname": "level", "name": "logger"},
        )
    )
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    # uvicorn has its own loggers; route them through our handler.
    for noisy in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        log = logging.getLogger(noisy)
        log.handlers.clear()
        log.propagate = True


logger = logging.getLogger("ml-engine")
