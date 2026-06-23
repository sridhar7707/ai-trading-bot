import logging
import pytest
from bot.core.error_logger import log_exception, timed, safe_render, setup_logger


def test_log_exception_stdlib_logger(caplog):
    logger = logging.getLogger("test.error_logger")
    with caplog.at_level(logging.ERROR, logger="test.error_logger"):
        log_exception(logger, "test_fn", ValueError("boom"))
    assert "test_fn error" in caplog.text


def test_log_exception_with_ctx(caplog):
    logger = logging.getLogger("test.ctx")
    with caplog.at_level(logging.ERROR, logger="test.ctx"):
        log_exception(logger, "fn", RuntimeError("x"), ctx={"key": "val"})
    assert "ctx=" in caplog.text


def test_timed_decorator_passes_return_value():
    logger = logging.getLogger("test.timed")

    @timed(logger, warn_sec=999)
    def add(a, b):
        return a + b

    assert add(2, 3) == 5


def test_timed_decorator_logs_warning(caplog):
    logger = logging.getLogger("test.timed.warn")

    @timed(logger, warn_sec=0.0, err_sec=999)
    def noop():
        pass

    with caplog.at_level(logging.WARNING, logger="test.timed.warn"):
        noop()
    assert "noop" in caplog.text


def test_safe_render_returns_error_card_on_exception():
    @safe_render("TestPanel")
    def broken():
        raise RuntimeError("oops")

    result = broken()
    assert "TestPanel unavailable" in result
    assert "RuntimeError" in result


def test_safe_render_passes_through_normal_result():
    @safe_render("OK")
    def fine():
        return "good"

    assert fine() == "good"


def test_setup_logger_returns_named_logger():
    lg = setup_logger("tradegenie.test")
    assert lg.name == "tradegenie.test"
