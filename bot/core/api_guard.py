"""API guard: retry with exponential backoff for rate-limited 3rd-party calls.

Usage:
    from bot.core.api_guard import call_with_retry
    result = call_with_retry(fn, *args, _label="Alpaca/get_bars", **kwargs)

All retried failures are logged with the [API RATE LIMIT] prefix so they are
grep-able in CI logs: grep "[API RATE LIMIT]" logs/trading.log
"""
from __future__ import annotations

import time
from typing import Any, Callable, TypeVar
from loguru import logger

T = TypeVar("T")

_RETRYABLE_HTTP   = {429, 503, 502}
_RETRYABLE_PHRASES = frozenset({
    "rate limit", "too many requests", "connection reset",
    "connection timed out", "server disconnected", "ratelimitexceeded",
})


def _is_retryable(exc: Exception, rate_limit_only: bool) -> bool:
    """Classify whether an exception warrants a retry."""
    msg = str(exc).lower()
    is_rate_limit = (
        any(p in msg for p in _RETRYABLE_PHRASES)
        or type(exc).__name__.lower() in {"ratelimitexceeded", "ratelimiterror"}
    )

    # alpaca_trade_api.rest.APIError / requests.HTTPError expose status codes
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if not is_rate_limit and status is not None:
        is_rate_limit = is_rate_limit or (status == 429)

    resp = getattr(exc, "response", None)
    if not is_rate_limit and resp is not None:
        is_rate_limit = is_rate_limit or (getattr(resp, "status_code", 0) == 429)

    if rate_limit_only:
        return is_rate_limit

    if is_rate_limit:
        return True

    # Non-429 transient: 502/503 or network errors (only when rate_limit_only=False)
    if status in {502, 503}:
        return True
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    try:
        import requests as _req
        if isinstance(exc, (_req.exceptions.ConnectionError, _req.exceptions.Timeout)):
            return True
    except ImportError:
        pass

    return False


def call_with_retry(
    fn: Callable[..., T],
    *args: Any,
    _label: str = "",
    _max_retries: int = 3,
    _base_delay: float = 1.0,
    _rate_limit_only: bool = False,
    **kwargs: Any,
) -> T:
    """
    Call fn(*args, **kwargs) with exponential back-off on rate-limit / transient errors.

    Guard-specific kwargs use underscore prefixes to avoid colliding with fn's own kwargs:
      _label           — human-readable name for log lines (e.g. "Alpaca/submit_order")
      _max_retries     — number of retry attempts (default 3)
      _base_delay      — initial back-off in seconds (doubles each attempt)
      _rate_limit_only — when True, only retry confirmed HTTP 429 (use for order submissions)

    Raises the original exception after all retries are exhausted.
    """
    _name = _label or getattr(fn, "__name__", repr(fn))
    last_exc: Exception | None = None

    for attempt in range(_max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            if attempt >= _max_retries:
                break
            if not _is_retryable(exc, _rate_limit_only):
                raise
            delay = _base_delay * (2 ** attempt)
            logger.warning(
                f"[API RATE LIMIT] {_name} — "
                f"attempt {attempt + 1}/{_max_retries} — "
                f"back-off {delay:.1f}s | {type(exc).__name__}: {exc}"
            )
            time.sleep(delay)

    logger.error(
        f"[API RATE LIMIT] {_name} failed after {_max_retries} retries — "
        f"{type(last_exc).__name__}: {last_exc}"
    )
    raise last_exc  # type: ignore[misc]
