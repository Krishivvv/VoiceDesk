"""Centralised logging configuration for the audio support agent.

Call :func:`configure_logging` once at process start (FastAPI startup,
Streamlit boot, or a script's ``__main__``) to install a consistent,
structured log format across every module. Library code should never call
``logging.basicConfig`` itself — it only acquires a module logger via
``logging.getLogger(__name__)`` and emits records.
"""

from __future__ import annotations

import logging
import os

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_configured = False


def configure_logging(level: str | int | None = None) -> None:
    """Install the root logging handler/format exactly once.

    Args:
        level: Log level as a name (``"INFO"``) or numeric value. When
            ``None`` the ``LOG_LEVEL`` environment variable is used, falling
            back to ``INFO``.
    """
    global _configured
    if _configured:
        return

    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO")
    if isinstance(level, str):
        level = logging.getLevelName(level.upper())

    logging.basicConfig(level=level, format=_DEFAULT_FORMAT)

    # Quiet down noisy third-party loggers that would otherwise drown signal.
    for noisy in ("httpx", "httpcore", "urllib3", "chromadb", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a module logger, ensuring logging is configured first."""
    configure_logging()
    return logging.getLogger(name)
