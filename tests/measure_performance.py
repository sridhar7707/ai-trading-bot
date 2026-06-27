#!/usr/bin/env python3
"""Measure actual non-functional performance metrics for docs/NFR.md.

Usage:
    python tests/measure_performance.py          # run and print
    python tests/measure_performance.py --json   # output JSON
    python tests/measure_performance.py --update # write results to docs/NFR.md

Results are point-in-time. Re-run quarterly or after infrastructure changes.
"""
from __future__ import annotations
import argparse
import datetime
import importlib
import json
import os
import sys
import time
import tracemalloc
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# -- Helpers -------------------------------------------------------------------

def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)


def _measure_import(module_path: str) -> float:
    """Time a module import (cold, not cached)."""
    if module_path in sys.modules:
        del sys.modules[module_path]
    t = time.perf_counter()
    importlib.import_module(module_path)
    return _ms(t)


def _measure_fn(fn, *args, repeat: int = 3, **kwargs) -> tuple[float, float, float]:
    """Run fn repeat times; return (min_ms, avg_ms, max_ms)."""
    times = []
    for _ in range(repeat):
        t = time.perf_counter()
        fn(*args, **kwargs)
        times.append(_ms(t))
    return min(times), round(sum(times) / len(times), 1), max(times)


# -- Measurement functions -----------------------------------------------------

def measure_db_queries() -> dict:
    """Time common SQLite reads against the local trades.db."""
    results: dict = {}
    db_path = ROOT / "trades.db"
    if not db_path.exists():
        return {"error": "trades.db not found --- run the bot first"}

    import sqlite3
    con = sqlite3.connect(str(db_path))

    queries = {
        "count_trades":         "SELECT COUNT(*) FROM trades",
        "latest_10_trades":     "SELECT * FROM trades ORDER BY id DESC LIMIT 10",
        "open_positions":       "SELECT * FROM position_state",
        "portfolio_snapshot":   "SELECT * FROM portfolio_snapshots ORDER BY timestamp DESC LIMIT 1",
        "recommendations_14d":  "SELECT * FROM recommendations WHERE prediction_date >= date('now', '-14 days')",
    }

    for name, sql in queries.items():
        mn, avg, mx = _measure_fn(lambda q=sql: con.execute(q).fetchall(), repeat=5)
        results[name] = {"min_ms": mn, "avg_ms": avg, "max_ms": mx}

    con.close()
    return results


def measure_dashboard_imports() -> dict:
    """Time first-import of the dashboard data layer."""
    results: dict = {}
    for mod in ["dashboard.design_system", "dashboard.data", "dashboard.builders"]:
        try:
            ms = _measure_import(mod)
            results[mod] = {"import_ms": ms}
        except Exception as exc:
            results[mod] = {"error": str(exc)}
    return results


def measure_render_functions() -> dict:
    """Time key render functions (requires trades.db to be present)."""
    results: dict = {}
    db_path = ROOT / "trades.db"
    if not db_path.exists():
        return {"error": "trades.db not found --- skipping render benchmarks"}

    render_fns = [
        ("dashboard.components.overview", "render_metrics"),
        ("dashboard.components.history",  "render_portfolio_performance"),
        ("dashboard.components.portfolio", "render_positions"),
    ]

    for mod_path, fn_name in render_fns:
        try:
            mod = importlib.import_module(mod_path)
            fn = getattr(mod, fn_name)
            mn, avg, mx = _measure_fn(fn, repeat=3)
            results[f"{mod_path}.{fn_name}"] = {"min_ms": mn, "avg_ms": avg, "max_ms": mx}
        except Exception as exc:
            results[f"{mod_path}.{fn_name}"] = {"error": str(exc)[:120]}

    return results


def measure_memory() -> dict:
    """Peak RSS memory used by importing the full dashboard stack."""
    tracemalloc.start()
    snapshot_before = tracemalloc.take_snapshot()

    try:
        import dashboard.app  # noqa: F401
    except Exception:
        pass

    snapshot_after = tracemalloc.take_snapshot()
    tracemalloc.stop()

    stats = snapshot_after.compare_to(snapshot_before, "lineno")
    total_kb = sum(s.size_diff for s in stats) / 1024

    try:
        import psutil
        rss_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except ImportError:
        rss_mb = None

    return {
        "dashboard_stack_heap_kb": round(total_kb, 1),
        "process_rss_mb": round(rss_mb, 1) if rss_mb else "psutil not installed",
    }


def measure_test_suite() -> dict:
    """Time the full pytest run."""
    import subprocess
    t = time.perf_counter()
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q", "--tb=no", "--no-header"],
        capture_output=True, text=True, cwd=str(ROOT)
    )
    elapsed = _ms(t)
    lines = r.stdout.strip().splitlines()
    summary = lines[-1] if lines else r.stdout[:200]
    return {"total_ms": elapsed, "total_s": round(elapsed / 1000, 1), "summary": summary}


# -- NFR.md writer -------------------------------------------------------------

def _write_nfr(results: dict) -> None:
    nfr_path = ROOT / "docs" / "NFR.md"
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    db = results.get("db_queries", {})
    render = results.get("render_functions", {})
    mem = results.get("memory", {})
    tests = results.get("test_suite", {})

    def _row(label: str, data: dict, key: str = "avg_ms", unit: str = "ms") -> str:
        val = data.get(key, data.get("error", "---"))
        return f"| {label} | {val} {unit if isinstance(val, (int, float)) else ''} |"

    lines = [
        "# TradeGenius AI --- Non-Functional Requirements",
        "",
        f"Last measured: {now}",
        "Generated by: `python tests/measure_performance.py --update`",
        "",
        "## Database Query Latency (SQLite, local trades.db)",
        "",
        "| Query | Avg (ms) |",
        "|-------|----------|",
    ]
    for name, data in db.items():
        if isinstance(data, dict) and "avg_ms" in data:
            lines.append(f"| {name} | {data['avg_ms']} |")
    lines += [
        "",
        "All queries run against the local `trades.db` (no network). Target: < 50 ms per query.",
        "",
        "## Dashboard Render Latency",
        "",
        "| Component | Avg (ms) |",
        "|-----------|----------|",
    ]
    for name, data in render.items():
        short = name.split(".")[-1]
        if isinstance(data, dict) and "avg_ms" in data:
            lines.append(f"| {short} | {data['avg_ms']} |")
    lines += [
        "",
        "Render functions run synchronously inside Gradio's 60-second timer.",
        "Target: each render function < 2000 ms (worst case, cold cache).",
        "",
        "## Memory Usage",
        "",
        f"| Metric | Value |",
        "|--------|-------|",
        f"| Dashboard stack heap (tracemalloc) | {mem.get('dashboard_stack_heap_kb', '---')} KB |",
        f"| Process RSS (full bot + dashboard) | {mem.get('process_rss_mb', '---')} MB |",
        "",
        "Target: process RSS < 512 MB on HuggingFace free-tier Space (2 CPU, 16 GB RAM).",
        "",
        "## Test Suite Performance",
        "",
        f"| Metric | Value |",
        "|--------|-------|",
        f"| Total test time | {tests.get('total_s', '---')} s |",
        f"| Result | {tests.get('summary', '---')} |",
        "",
        "Target: full test suite < 120 s.",
        "",
        "## Dashboard Refresh Cycle",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
        "| Gradio Timer interval | 60 s |",
        "| DB cache TTL | 55 s (module-level `_CACHE`) |",
        "| News feed cache TTL | 1800 s (30 min) |",
        "| Benchmark price cache TTL | 3600 s (1 hr) |",
        "| yfinance per-stock cache TTL | 3600 s (1 hr) |",
        "",
        "## Availability Targets",
        "",
        "| Component | Target | Notes |",
        "|-----------|--------|-------|",
        "| Trading bot (GitHub Actions) | 99% of market-hours cycles | Misses < 1 cycle / week acceptable |",
        "| Dashboard (HuggingFace Spaces) | 95% uptime | HF free-tier may sleep; restarts in < 30 s |",
        "| DB sync to HF | ≤ 15 min lag | `push_db()` runs each cycle |",
        "",
    ]

    nfr_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Written: {nfr_path}")


# -- Main ----------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Measure TradeGenius performance metrics")
    ap.add_argument("--json",   action="store_true", help="Output JSON")
    ap.add_argument("--update", action="store_true", help="Write results to docs/NFR.md")
    args = ap.parse_args()

    print("Measuring...")
    results: dict = {}

    print("  DB queries...")
    results["db_queries"] = measure_db_queries()

    print("  Dashboard imports...")
    results["dashboard_imports"] = measure_dashboard_imports()

    print("  Render functions...")
    results["render_functions"] = measure_render_functions()

    print("  Memory...")
    results["memory"] = measure_memory()

    print("  Test suite (this may take ~40s)...")
    results["test_suite"] = measure_test_suite()

    results["measured_at"] = datetime.datetime.now().isoformat()

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print("\n-- DB Query Latency -------------------------------------")
        for name, data in results["db_queries"].items():
            if isinstance(data, dict) and "avg_ms" in data:
                print(f"  {name:35s}  avg={data['avg_ms']}ms  min={data['min_ms']}ms  max={data['max_ms']}ms")

        print("\n-- Render Function Latency ------------------------------")
        for name, data in results["render_functions"].items():
            short = name.split(".")[-1]
            if isinstance(data, dict) and "avg_ms" in data:
                print(f"  {short:35s}  avg={data['avg_ms']}ms")
            elif "error" in data:
                print(f"  {short:35s}  ERROR: {data['error'][:60]}")

        print("\n-- Memory -----------------------------------------------")
        mem = results["memory"]
        print(f"  Heap delta (tracemalloc): {mem.get('dashboard_stack_heap_kb', '---')} KB")
        print(f"  Process RSS:              {mem.get('process_rss_mb', '---')} MB")

        print("\n-- Test Suite -------------------------------------------")
        ts = results["test_suite"]
        print(f"  Duration: {ts.get('total_s', '---')} s")
        print(f"  Result:   {ts.get('summary', '---')}")

    if args.update:
        _write_nfr(results)


if __name__ == "__main__":
    main()
