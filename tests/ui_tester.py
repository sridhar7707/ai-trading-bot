"""
ui_tester.py — Dashboard render function compliance checks.

Usage:
  python tests/ui_tester.py
  python tests/ui_tester.py --verbose
  python tests/ui_tester.py --group 7
"""
from __future__ import annotations

import re
import sys
import argparse
import ast
import os

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ── Design system allowed values ─────────────────────────────────────────────
_ALLOWED_FONT_SIZES   = {"36px", "20px", "15px", "11px"}
_ALLOWED_TEXT_COLORS  = {"#ffffff", "#b0b7c3", "#7f8896"}
_ALLOWED_ACTION_COLORS = {"#00c853", "#ff5252", "#ffb300", "#64b5f6", "#ab47bc"}
_ALLOWED_BG_COLORS    = {
    "#0f1115", "#171a21", "#222733", "#2d3445",
    "#00200d", "#200808", "#1f1500", "#081428", "#150820",
}
_ALL_ALLOWED_COLORS = _ALLOWED_TEXT_COLORS | _ALLOWED_ACTION_COLORS | _ALLOWED_BG_COLORS

_OLD_COLORS = {
    "#0e0e0e": "old BG (use #0f1115)",
    "#1b1b1b": "old SURFACE (use #171a21)",
    "#252525": "old SURFACE2 (use #222733)",
    "#2a2a2a": "old BORDER (use #2d3445)",
    "#a0a0a0": "old TEXT2 (use #b0b7c3)",
    "#00c805": "old PRIMARY/GAIN (use #00c853)",
    "#ff5000": "old LOSS (use #ff5252)",
    "#9d4edd": "old NEURAL (use #ab47bc)",
}


# ── Collect render functions from all dashboard source files ─────────────────
def _collect_render_fn_names() -> list[str]:
    dash_dir = os.path.join(os.path.dirname(__file__), "..", "dashboard")
    names = []
    for root, _, files in os.walk(dash_dir):
        for fname in files:
            if not fname.endswith(".py"):
                continue
            fpath = os.path.join(root, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    src = f.read()
                tree = ast.parse(src)
                names += [
                    node.name for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef)
                    and node.name.startswith("render_")
                ]
            except Exception:
                pass
    return list(dict.fromkeys(names))  # deduplicate, preserve order


def _read_app_source() -> str:
    """Read all dashboard Python source files combined (app.py + modules)."""
    dash_dir = os.path.join(os.path.dirname(__file__), "..", "dashboard")
    parts = []
    for root, _, files in os.walk(dash_dir):
        for fname in sorted(files):
            if not fname.endswith(".py"):
                continue
            fpath = os.path.join(root, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    parts.append(f.read())
            except Exception:
                pass
    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# GROUP 7: DESIGN SYSTEM COMPLIANCE
# ─────────────────────────────────────────────────────────────────────────────

def test_design_system_compliance(verbose: bool = False) -> tuple[int, int, list[str]]:
    """
    Scans app.py source for design system violations.
    Returns (fail_count, warn_count, messages).
    Zero FAIL allowed before going live.
    """
    src = _read_app_source()
    fn_names = _collect_render_fn_names()

    failures:  list[str] = []
    warnings:  list[str] = []
    passes:    list[str] = []

    # --- Check 1: old colors not present anywhere in source ------------------
    for old_color, description in _OLD_COLORS.items():
        # Find all lines containing the old color
        lines_found = [
            i + 1 for i, line in enumerate(src.splitlines())
            if old_color in line.lower()
            # skip comment lines and the _OLD_COLORS dict itself
            and not line.strip().startswith("#")
            and "_OLD_COLORS" not in line
            and "old BG" not in line
            and "old SURFACE" not in line
            and "old TEXT2" not in line
            and "old PRIMARY" not in line
            and "old LOSS" not in line
            and "old NEURAL" not in line
        ]
        if lines_found:
            failures.append(
                f"FAIL  old color {old_color} ({description}) found on lines: "
                f"{lines_found[:5]}"
            )
        else:
            passes.append(f"PASS  old color {old_color} not present")

    # --- Check 2: design system constants defined ----------------------------
    required_constants = [
        "ACTION_BUY", "ACTION_SELL", "ACTION_TRIM", "ACTION_HOLD", "ACTION_WATCH",
        "FONT_HERO", "FONT_SECTION", "FONT_VALUE", "FONT_LABEL",
        "WEIGHT_BOLD", "WEIGHT_MEDIUM", "WEIGHT_NORMAL",
        "CARD_PADDING", "CARD_RADIUS", "TEXT3", "SYMBOL_STYLE",
    ]
    for const in required_constants:
        if re.search(rf'^{const}\s*=', src, re.MULTILINE):
            passes.append(f"PASS  constant {const} defined")
        else:
            failures.append(f"FAIL  constant {const} missing from app.py")

    # --- Check 3: design system helpers defined ------------------------------
    required_helpers = [
        "_card", "_label", "_hero_value", "_section_title", "_action_badge",
        "_symbol", "_confidence_bar", "_metric_row", "_progress_bar",
        "_divider", "_empty_state", "_action_row", "_table",
    ]
    for fn in required_helpers:
        if re.search(rf'^def {fn}\(', src, re.MULTILINE):
            passes.append(f"PASS  helper {fn}() defined")
        else:
            failures.append(f"FAIL  helper {fn}() missing from app.py")

    # --- Check 4: render functions don't use removed columns -----------------
    # Check render_positions doesn't contain "Shares", "Invested", "Cost Basis"
    pos_fn_match = re.search(
        r'def render_positions\(\).*?(?=\ndef |\Z)', src, re.DOTALL
    )
    if pos_fn_match:
        pos_src = pos_fn_match.group(0)
        removed_cols = ["Shares  ", "Invested  ", "Cost Basis", "Current Value  "]
        for col in removed_cols:
            if col in pos_src:
                failures.append(
                    f"FAIL  render_positions() still contains removed column: '{col.strip()}'"
                )
            else:
                passes.append(f"PASS  render_positions() removed column '{col.strip()}'")
    else:
        warnings.append("WARN  render_positions() not found in source")

    # --- Check 5: docs/DESIGN_SYSTEM.md exists -------------------------------
    ds_path = os.path.join(
        os.path.dirname(__file__), "..", "docs", "DESIGN_SYSTEM.md"
    )
    if os.path.exists(ds_path):
        passes.append("PASS  docs/DESIGN_SYSTEM.md exists")
    else:
        failures.append("FAIL  docs/DESIGN_SYSTEM.md missing — create it")

    # --- Check 6: render functions list present ------------------------------
    if fn_names:
        passes.append(f"PASS  {len(fn_names)} render functions found in app.py")
    else:
        warnings.append("WARN  no render_ functions found in app.py")

    # --- Check 7: mobile CSS present -----------------------------------------
    mobile_markers = ["max-width: 480px", "table-layout: fixed", "text-overflow: ellipsis"]
    for marker in mobile_markers:
        if marker in src:
            passes.append(f"PASS  mobile CSS: '{marker}' present")
        else:
            failures.append(f"FAIL  mobile CSS missing: '{marker}'")

    # --- Check 8: recommendation engine imported ----------------------------
    if "from bot.core.recommendation_engine import" in src:
        passes.append("PASS  recommendation_engine imported")
    else:
        failures.append("FAIL  recommendation_engine not imported in app.py")

    all_messages = failures + warnings
    if verbose:
        all_messages = passes + warnings + failures

    return len(failures), len(warnings), all_messages


# ─────────────────────────────────────────────────────────────────────────────
# GROUP 8: Exception handling
# ─────────────────────────────────────────────────────────────────────────────

def test_exception_handling(verbose: bool = False):
    files_to_check = [
        "dashboard/app.py",
        "bot/main.py",
        "bot/risk/risk_manager.py",
        "bot/execution/alpaca_client.py",
        "database/repositories/analytics_repository.py",
        "database/services/analytics_service.py",
    ]

    failures = []
    warnings = []
    all_messages = []

    for filepath in files_to_check:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except FileNotFoundError:
            continue

        for i, line in enumerate(lines, 1):
            stripped = line.strip()

            # CLASS A pattern: bare `except Exception:` or `except:` followed by pass
            # (narrow-type exceptions like ValueError/OperationalError are acceptable)
            is_generic_except = (
                stripped == "except Exception:" or
                stripped.startswith("except Exception as ") and stripped.endswith(":") or
                stripped == "except:"
            )
            if is_generic_except and i < len(lines):
                next_stripped = lines[i].strip() if i < len(lines) else ""
                if next_stripped == "pass":
                    msg = f"FAIL  CLASS A — {filepath}:{i} — bare pass after except Exception (no logging)"
                    failures.append(msg)
                    all_messages.append(msg)
                    continue

            # CLASS B: string-only error logging (loses traceback)
            if ("logger.error(str(e))" in stripped or
                    "logger.error(str(exc))" in stripped or
                    "logger.warning(str(e))" in stripped):
                msg = f"WARN  CLASS B — {filepath}:{i} — string-only error log (use log_exception)"
                warnings.append(msg)
                all_messages.append(msg)
                continue

            if verbose and not failures and not warnings:
                pass  # only report on problems

    files_checked = sum(1 for f in files_to_check if os.path.exists(f))
    summary = (f"PASS  CLASS A violations: {len(failures)}  "
               f"CLASS B warnings: {len(warnings)}  "
               f"Files checked: {files_checked}")
    all_messages.insert(0, summary)

    return len(failures), len(warnings), all_messages


def test_analytics_check_health(verbose: bool = False):
    """Verify AnalyticsService.check_health() exists and returns a valid structure."""
    failures = []
    warnings = []
    all_messages = []

    try:
        import sys, os as _os
        sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
        from database.services.analytics_service import AnalyticsService
        svc = AnalyticsService()

        if not hasattr(svc, "check_health"):
            msg = "FAIL  analytics: AnalyticsService missing check_health() method"
            failures.append(msg)
            all_messages.append(msg)
        else:
            health = svc.check_health()
            if health.get("overall") not in ("ok", "degraded"):
                msg = f"FAIL  analytics: check_health() returned unexpected overall={health.get('overall')!r}"
                failures.append(msg)
                all_messages.append(msg)
            elif "duckdb_connection" not in health:
                msg = "FAIL  analytics: check_health() missing duckdb_connection key"
                failures.append(msg)
                all_messages.append(msg)
            else:
                all_messages.append(f"PASS  analytics: check_health() overall={health['overall']!r}")
    except Exception as exc:
        msg = f"FAIL  analytics: check_health() raised {type(exc).__name__}: {exc}"
        failures.append(msg)
        all_messages.append(msg)

    return len(failures), len(warnings), all_messages


# ─────────────────────────────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TradeGenius UI compliance tester")
    parser.add_argument("--verbose", action="store_true", help="Show all checks including PASS")
    parser.add_argument("--group",   type=int,            help="Run only a specific group (e.g. 7)")
    args = parser.parse_args()

    total_fail = 0
    total_warn = 0
    all_msgs   = []

    if args.group is None or args.group == 7:
        print("\n── GROUP 7: DESIGN SYSTEM COMPLIANCE ──────────────────────────────────")
        fail, warn, msgs = test_design_system_compliance(verbose=args.verbose)
        for m in msgs:
            print(f"  {m}")
        total_fail += fail
        total_warn += warn
        all_msgs   += msgs

    if args.group is None or args.group == 8:
        print("\n── GROUP 8: EXCEPTION HANDLING ─────────────────────────────────────────")
        fail, warn, msgs = test_exception_handling(verbose=args.verbose)
        for m in msgs:
            print(f"  {m}")
        total_fail += fail
        total_warn += warn
        all_msgs   += msgs

        fail, warn, msgs = test_analytics_check_health(verbose=args.verbose)
        for m in msgs:
            print(f"  {m}")
        total_fail += fail
        total_warn += warn
        all_msgs   += msgs

    print(f"\n{'─'*70}")
    print(f"  Results: {total_fail} FAIL  {total_warn} WARN")
    if total_fail == 0:
        print("  ALL CHECKS: OK — zero violations")
    else:
        print(f"  VIOLATIONS: {total_fail} must be fixed before going live")
    print(f"{'─'*70}\n")

    sys.exit(0 if total_fail == 0 else 1)


if __name__ == "__main__":
    main()
