"""Unit tests for dashboard.registry — independent of Gradio and the full dashboard."""
from __future__ import annotations

import sys
import pytest


def _fresh_registry():
    """Return a clean registry module with empty _specs and _widgets dicts."""
    # Remove any cached import so each test gets an isolated state
    for key in list(sys.modules):
        if key == "dashboard.registry":
            del sys.modules[key]
    import dashboard.registry as reg
    return reg


def test_register_and_retrieve():
    reg = _fresh_registry()
    spec = reg.ComponentSpec("foo", reg.RefreshGroup.FAST, lambda: "html", priority=10)
    reg.register(spec)
    assert "foo" in reg._specs
    assert reg._specs["foo"].key == "foo"
    assert reg._specs["foo"].group == reg.RefreshGroup.FAST


def test_empty_key_raises():
    reg = _fresh_registry()
    with pytest.raises(ValueError, match="non-empty"):
        reg.ComponentSpec("", reg.RefreshGroup.FAST, lambda: "")

    with pytest.raises(ValueError, match="non-empty"):
        reg.ComponentSpec("   ", reg.RefreshGroup.FAST, lambda: "")


def test_duplicate_key_raises():
    reg = _fresh_registry()
    reg.register(reg.ComponentSpec("dup", reg.RefreshGroup.FAST, lambda: ""))
    with pytest.raises(ValueError, match="Duplicate"):
        reg.register(reg.ComponentSpec("dup", reg.RefreshGroup.SLOW, lambda: ""))


def test_mount_binds_output():
    reg = _fresh_registry()
    reg.register(reg.ComponentSpec("hero", reg.RefreshGroup.SLOW, lambda: ""))
    sentinel = object()
    returned = reg.mount("hero", sentinel)
    assert returned is sentinel
    assert reg._specs["hero"].output is sentinel
    assert reg._widgets["hero"] is sentinel


def test_mount_non_spec_key_stored():
    """mount() on a key with no registered spec just stores in _widgets."""
    reg = _fresh_registry()
    obj = object()
    reg.mount("perf_tabs", obj)
    assert reg.widget("perf_tabs") is obj
    assert "perf_tabs" not in reg._specs


def test_by_group_filters_and_sorts():
    reg = _fresh_registry()
    reg.register(reg.ComponentSpec("a", reg.RefreshGroup.FAST, lambda: "", priority=20))
    reg.register(reg.ComponentSpec("b", reg.RefreshGroup.FAST, lambda: "", priority=10))
    reg.register(reg.ComponentSpec("c", reg.RefreshGroup.SLOW, lambda: "", priority=5))
    reg.mount("a", object())
    reg.mount("b", object())
    reg.mount("c", object())

    fast = reg.by_group(reg.RefreshGroup.FAST)
    assert [s.key for s in fast] == ["b", "a"]   # sorted by priority

    slow = reg.by_group(reg.RefreshGroup.SLOW)
    assert [s.key for s in slow] == ["c"]


def test_by_group_excludes_unmounted():
    reg = _fresh_registry()
    reg.register(reg.ComponentSpec("mounted",   reg.RefreshGroup.FAST, lambda: "", priority=1))
    reg.register(reg.ComponentSpec("unmounted", reg.RefreshGroup.FAST, lambda: "", priority=2))
    reg.mount("mounted", object())
    # "unmounted" never gets mount() called

    fast = reg.by_group(reg.RefreshGroup.FAST)
    assert len(fast) == 1
    assert fast[0].key == "mounted"


def test_by_group_excludes_disabled():
    reg = _fresh_registry()
    reg.register(reg.ComponentSpec("on",  reg.RefreshGroup.FAST, lambda: "", enabled=True))
    reg.register(reg.ComponentSpec("off", reg.RefreshGroup.FAST, lambda: "", enabled=False))
    reg.mount("on",  object())
    reg.mount("off", object())

    fast = reg.by_group(reg.RefreshGroup.FAST)
    assert len(fast) == 1
    assert fast[0].key == "on"


def test_validate_passes_when_all_mounted():
    reg = _fresh_registry()
    reg.register(reg.ComponentSpec("x", reg.RefreshGroup.FAST, lambda: ""))
    reg.mount("x", object())
    reg.validate()  # should not raise


def test_validate_raises_on_unbound():
    reg = _fresh_registry()
    reg.register(reg.ComponentSpec("unbound", reg.RefreshGroup.SLOW, lambda: ""))
    with pytest.raises(RuntimeError, match="unbound"):
        reg.validate()


def test_validate_skips_disabled():
    reg = _fresh_registry()
    reg.register(reg.ComponentSpec("disabled", reg.RefreshGroup.FAST, lambda: "", enabled=False))
    reg.validate()  # disabled spec without output should not raise


def test_render_fn_called_in_batch():
    reg = _fresh_registry()
    calls = []

    def _render():
        calls.append(1)
        return "html"

    reg.register(reg.ComponentSpec("r", reg.RefreshGroup.FAST, _render))
    reg.mount("r", object())

    specs = reg.by_group(reg.RefreshGroup.FAST)
    result = tuple(s.render_fn() for s in specs)
    assert result == ("html",)
    assert len(calls) == 1


def test_non_callable_render_fn_raises():
    reg = _fresh_registry()
    with pytest.raises(ValueError, match="callable"):
        reg.ComponentSpec("bad", reg.RefreshGroup.FAST, "not_a_function")


def test_require_widgets_caught_by_validate():
    reg = _fresh_registry()
    reg.require_widgets("must_exist")
    with pytest.raises(RuntimeError, match="must_exist"):
        reg.validate()


def test_require_widgets_passes_when_mounted():
    reg = _fresh_registry()
    reg.require_widgets("will_be_mounted")
    reg.mount("will_be_mounted", object())
    reg.validate()  # should not raise
