"""Logging strutturato (NFR4).

Espone ``setup_logging`` e un helper ``turn_logger`` per loggare le metriche di
ogni turno (testo STT, tool usati, latenze per stadio) in modo strutturato.
"""

from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from typing import Any, Iterator

import structlog


def setup_logging(level: str = "INFO") -> None:
    """Configura structlog con output leggibile su console."""
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level.upper())
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="%H:%M:%S", utc=False),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(level.upper())
        ),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = "jarvis") -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)


class TurnTimer:
    """Accumula le latenze per stadio di un turno e le logga alla fine."""

    def __init__(self, log: structlog.stdlib.BoundLogger) -> None:
        self._log = log
        self._stages: dict[str, float] = {}
        self._meta: dict[str, Any] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self._stages[name] = round((time.perf_counter() - start) * 1000, 1)

    def set(self, **meta: Any) -> None:
        self._meta.update(meta)

    def emit(self) -> None:
        self._log.info("turn", **self._meta, latency_ms=self._stages)


@contextmanager
def turn_logger(log: structlog.stdlib.BoundLogger) -> Iterator[TurnTimer]:
    timer = TurnTimer(log)
    try:
        yield timer
    finally:
        timer.emit()
