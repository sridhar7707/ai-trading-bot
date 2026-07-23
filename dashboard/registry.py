"""Component registry: each component declares its own refresh group and render function.

timer discovery flow:
  1. Importing a component module executes its module-level register() call.
  2. app.py mounts each Gradio widget via registry.mount(key, widget).
  3. register_all_timers() calls by_group() to build one batched tick per group.

This eliminates the 40-key dict that previously wired app.py ↔ timers.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Any


class RefreshGroup(Enum):
    FAST = 60    # DB reads only — fires every 60 s
    SLOW = 300   # yfinance + charts + AI — fires every 300 s


@dataclass
class ComponentSpec:
    key: str
    group: RefreshGroup
    render_fn: Callable
    priority: int = 100   # lower = runs earlier in the batch
    enabled: bool = True
    output: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.key or not self.key.strip():
            raise ValueError("ComponentSpec key must be a non-empty string")
        if not callable(self.render_fn):
            raise ValueError(f"ComponentSpec render_fn must be callable, got {type(self.render_fn)!r}")


_specs: dict[str, ComponentSpec] = {}
_widgets: dict[str, Any] = {}           # all mounted Gradio widgets, including inputs
_required_widgets: set[str] = set()     # non-spec keys that must be mounted before validate()


def register(spec: ComponentSpec) -> ComponentSpec:
    """Register a ComponentSpec. Raises ValueError on duplicate key."""
    if spec.key in _specs:
        raise ValueError(f"Duplicate ComponentSpec key: {spec.key!r}")
    _specs[spec.key] = spec
    return spec


def mount(key: str, w: Any) -> Any:
    """Bind a Gradio widget to its key. Returns the widget for inline assignment.

    If a ComponentSpec with this key exists, its output is set automatically.
    Non-spec keys (inputs, special widgets) are still stored for later lookup.
    """
    _widgets[key] = w
    if key in _specs:
        _specs[key].output = w
    return w


def widget(key: str) -> Any:
    """Return the mounted Gradio widget for key (inputs and special widgets)."""
    return _widgets[key]


def require_widgets(*keys: str) -> None:
    """Declare that these non-spec widget keys must be mounted before validate()."""
    _required_widgets.update(keys)


def by_group(group: RefreshGroup) -> list[ComponentSpec]:
    """Return specs for the group, sorted by priority, that have been mounted."""
    return sorted(
        [s for s in _specs.values() if s.group == group and s.enabled and s.output is not None],
        key=lambda s: s.priority,
    )


def validate() -> None:
    """Raise RuntimeError if any enabled spec or required widget was never mounted."""
    unbound = [k for k, s in _specs.items() if s.enabled and s.output is None]
    if unbound:
        raise RuntimeError(f"ComponentSpecs registered but never mounted: {unbound}")
    missing = [k for k in _required_widgets if k not in _widgets]
    if missing:
        raise RuntimeError(f"Required widgets never mounted: {missing}")
