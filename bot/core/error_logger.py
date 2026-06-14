"""Shared exception logging, timing, and render safety for the bot and dashboard."""
import functools
import logging
import time
import traceback
from typing import Optional

_default_logger = logging.getLogger("tradegenie.core")


def setup_logger(name: str) -> logging.Logger:
    """Return a named standard-library logger."""
    return logging.getLogger(name)


def log_exception(
    logger_obj,
    fn_name: str,
    exc: Exception,
    ctx: Optional[dict] = None,
) -> None:
    """Log exception with full traceback. Works with both loguru and stdlib loggers."""
    ctx_str = f" | ctx={ctx}" if ctx else ""
    msg = f"{fn_name} error: {exc}{ctx_str}"
    tb = traceback.format_exc()
    try:
        # loguru loggers expose .opt(exception=...)
        logger_obj.opt(exception=exc).error(msg)
    except AttributeError:
        logger_obj.error(f"{msg}\n{tb}")


def timed(logger_obj, warn_sec: float = 2.0, err_sec: float = 10.0):
    """Decorator that logs a WARNING if function exceeds warn_sec, ERROR if err_sec."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - t0
            if elapsed >= err_sec:
                logger_obj.error(
                    f"{func.__name__} took {elapsed:.2f}s (limit {err_sec}s)"
                )
            elif elapsed >= warn_sec:
                logger_obj.warning(
                    f"{func.__name__} took {elapsed:.2f}s (limit {warn_sec}s)"
                )
            return result
        return wrapper
    return decorator


def safe_render(fallback_title: str = "Panel"):
    """
    Decorator for render_* functions in dashboard/app.py.

    On any unhandled exception:
    - Logs full traceback to the function's named logger
    - Returns a visible red error card so the user sees what failed
    - Never crashes the Gradio dashboard
    """
    def decorator(func):
        _fn_logger = setup_logger(f"dashboard.{func.__name__}")

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                log_exception(
                    _fn_logger, func.__name__, exc,
                    {"panel": fallback_title},
                )
                # Hardcoded design-system colors to avoid circular import
                return (
                    f'<div style="'
                    f'background:#171a21;'
                    f'border:1px solid #ff5252;'
                    f'border-left:3px solid #ff5252;'
                    f'border-radius:8px;'
                    f'padding:16px 20px;'
                    f'margin:8px 0;'
                    f'">'
                    f'<div style="'
                    f'font-size:15px;font-weight:700;'
                    f'color:#ff5252;margin-bottom:6px;'
                    f'">&#9888; {fallback_title} unavailable</div>'
                    f'<div style="'
                    f'font-size:11px;color:#b0b7c3;line-height:1.7;'
                    f'">'
                    f'{type(exc).__name__}: {str(exc)[:120]}<br>'
                    f'Full details in logs/errors.log'
                    f'</div>'
                    f'</div>'
                )
        return wrapper
    return decorator
