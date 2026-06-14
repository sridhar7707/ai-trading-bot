"""Shared exception logging for the recommendation engine and bot core."""
import logging
import traceback

logger = logging.getLogger("tradegenie.core")


def setup_logger(name: str) -> logging.Logger:
    """Return a named logger for the given module."""
    return logging.getLogger(name)


def log_exception(fn_name: str, exc: Exception) -> None:
    """Log a caught exception with function context."""
    logger.error(f"{fn_name} error: {exc}\n{traceback.format_exc()}")
