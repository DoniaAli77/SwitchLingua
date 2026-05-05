"""Logging utilities for the pipeline."""

from __future__ import annotations

import logging


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger instance for the given component name.

    Logging configuration details are intentionally deferred.
    """

    logger = logging.getLogger(name)
    raise NotImplementedError("Logger configuration is not implemented yet.")
