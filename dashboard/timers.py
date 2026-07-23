"""Centralized Gradio timer callback registration (req 7.1).

Two timers keep the UI responsive without hammering external APIs:
  timer_ui   (60 s)  — lightweight DB reads only; no yfinance calls
  timer_data (300 s) — heavy: yfinance (15 s daemon-thread timeout), charts, AI; batched + stateful

Component ownership: each component module registers a ComponentSpec that declares its
RefreshGroup and render function. register_all_timers() discovers specs via by_group()
and builds one batched tick per group — no central mapping required.

Batching all callbacks into a single timer.tick() per timer prevents Gradio 5
from firing N separate sequential SSE events (one per registration), which
causes components to show loading indicators one-by-one and looks like a
continuous page refresh on slow servers.
"""
from __future__ import annotations

import gradio as gr

from dashboard.registry import RefreshGroup, by_group, widget, require_widgets
from dashboard.components.history import render_portfolio_performance, perf_choices, PERF_SEP
from dashboard.components.symbol_detail import render_symbol_detail

require_widgets("sim_sym_dd", "perf_tabs", "perf_out", "symbol_selector", "symbol_detail_out")


def register_all_timers(timer_ui: gr.Timer, timer_data: gr.Timer) -> None:
    """Register batched timer.tick() callbacks using ComponentSpec discovery."""
    _register_ui_tick(timer_ui)
    _register_data_tick(timer_data)


# ── Shared helper ─────────────────────────────────────────────────────────────

def _batch_tick(timer: gr.Timer, group: RefreshGroup) -> None:
    """Register one batched tick for all specs in *group*.

    Deduplicates calls to the same render_fn within a single tick so widgets
    sharing a function (e.g. pos_brief_out and pos_out both using render_positions)
    only trigger one DB call per cycle.
    """
    specs   = by_group(group)
    fns     = [s.render_fn for s in specs]
    outputs = [s.output    for s in specs]

    def _tick():
        cache: dict = {}
        results = []
        for fn in fns:
            if fn not in cache:
                cache[fn] = fn()
            results.append(cache[fn])
        return tuple(results)

    timer.tick(fn=_tick, outputs=outputs)


# ── Fast (60 s) — DB reads only, no external API calls ────────────────────────

def _register_ui_tick(timer: gr.Timer) -> None:
    """Batched FAST tick, plus a separate choices-update for the sim dropdown."""
    _batch_tick(timer, RefreshGroup.FAST)

    # sim_sym_dd returns gr.update() (not HTML) — kept as a separate tick.
    # symbol_selector is NOT refreshed: writing to a Dropdown with a registered
    # .change() triggers Gradio 5.9's feedback-loop bug.
    def _sim_choices():
        from dashboard.data import get_data as _gd
        choices = sorted(_gd().get("prices", {}).keys()) or []
        return gr.update(choices=choices)

    timer.tick(fn=_sim_choices, outputs=[widget("sim_sym_dd")])


# ── Slow (300 s) — yfinance (15 s timeout) + charts + AI ──────────────────────

def _register_data_tick(timer: gr.Timer) -> None:
    """Batched SLOW tick, plus stateful callbacks for perf and symbol detail."""
    _batch_tick(timer, RefreshGroup.SLOW)

    # Read perf_tabs label directly. Do NOT write back to perf_tabs — writing to
    # a Radio that has a .change() handler registered causes Gradio 5.9 to fire
    # that handler (Radio.svelte:39 → handle_change), which then sends both a
    # trigger value and an input value to a 1-param endpoint → "Too many arguments".
    def _refresh_perf(current_label: str):
        current_key = current_label.split(PERF_SEP)[0].strip() if isinstance(current_label, str) and current_label else "1M"
        choices = perf_choices()
        matched = next((ch for ch in choices if ch.split(PERF_SEP)[0].strip() == current_key), None)
        val = matched or (choices[2] if len(choices) > 2 else choices[0] if choices else None)
        return render_portfolio_performance(val or "1M")

    def _sym_detail(sel: str):
        return render_symbol_detail(sel)

    timer.tick(fn=_refresh_perf, inputs=[widget("perf_tabs")],       outputs=[widget("perf_out")])
    timer.tick(fn=_sym_detail,   inputs=[widget("symbol_selector")], outputs=[widget("symbol_detail_out")])
