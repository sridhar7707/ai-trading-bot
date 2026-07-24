"""
Playwright visual UI check for TradeGenius dashboard.

Starts the Gradio server, opens each tab in a real browser, takes screenshots,
and asserts that key content is visible. Run this manually any time you want to
verify the look & feel and data is showing correctly.

Usage:
    python tests/check_ui.py                # headless
    python tests/check_ui.py --headed       # watch the browser live
    python tests/check_ui.py --no-server    # server already on :7860

Screenshots saved to:  tests/snapshots/playwright/<YYYYMMDD_HHMMSS>/
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DASH_URL  = "http://localhost:7860"
SNAP_ROOT = Path(__file__).parent / "snapshots" / "playwright"

# ── ANSI colours ─────────────────────────────────────────────────────────────
_GREEN = "\033[92m"
_RED   = "\033[91m"
_YELLOW = "\033[93m"
_RESET = "\033[0m"
_BOLD  = "\033[1m"


def _ok(msg: str)   -> None: print(f"  {_GREEN}✓{_RESET} {msg}")
def _fail(msg: str) -> None: print(f"  {_RED}✗{_RESET} {msg}")
def _warn(msg: str) -> None: print(f"  {_YELLOW}~{_RESET} {msg}")
def _head(msg: str) -> None: print(f"\n{_BOLD}{msg}{_RESET}")


# ── Per-tab check specs ───────────────────────────────────────────────────────
# Each entry: (display_name, tab_button_text, content_assertions, wait_secs)
# Assertions are (description, css_or_text_to_find).
TAB_SPECS: list[tuple[str, str, list[tuple[str, str]], float]] = [
    (
        "Brief",
        "Brief",
        [
            ("Executive summary rendered",    ".nt-wrap"),
            ("Decision bar rendered",         "text=AI"),
            ("No crash text",                 "not:text=Traceback"),
        ],
        3.0,
    ),
    (
        "Portfolio",
        "Portfolio",
        [
            ("Portfolio value shown",         "text=$"),
            ("Positions section rendered",    "text=Position"),
            ("No crash text",                 "not:text=Traceback"),
        ],
        3.0,
    ),
    (
        "Capital",
        "Capital",
        [
            ("Capital overview rendered",     "text=Initial Deposit"),
            ("Managed capital pool shown",    "text=Managed Capital Pool"),
            ("Tradeable cash shown",          "text=Tradeable"),
            ("Profit breakdown rendered",     "text=Realized"),
            ("No crash text",                 "not:text=Traceback"),
        ],
        3.0,
    ),
    (
        "Trades",
        "Trades",
        [
            ("Top picks rendered",            ".nt-wrap"),
            ("No crash text",                 "not:text=Traceback"),
        ],
        3.0,
    ),
    (
        "Performance",
        "Performance",
        [
            ("Institutional metrics rendered", "text=Win Rate"),
            ("Attribution by symbol shown",    "text=Attribution"),
            ("No crash text",                  "not:text=Traceback"),
        ],
        3.0,
    ),
    (
        "Settings",
        "Settings",
        [
            ("Risk tolerance radio shown",    "text=Risk Tolerance"),
            ("Stop-loss slider shown",        "text=Stop-Loss"),
            ("No crash text",                 "not:text=Traceback"),
        ],
        2.0,
    ),
]


# ── Server management ─────────────────────────────────────────────────────────

def _server_alive() -> bool:
    try:
        import urllib.request
        urllib.request.urlopen(f"{DASH_URL}/info", timeout=2)
        return True
    except Exception:
        return False


def start_server() -> subprocess.Popen:
    print(f"Starting dashboard server …")
    proc = subprocess.Popen(
        [sys.executable, "scripts/dashboard.py"],
        cwd=str(BASE_DIR),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return proc


def wait_for_server(timeout: int = 120) -> bool:
    print(f"Waiting for server on {DASH_URL} (up to {timeout}s) …", end="", flush=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _server_alive():
            print(f"  ready")
            return True
        time.sleep(2)
        print(".", end="", flush=True)
    print()
    return False


# ── Browser checks ────────────────────────────────────────────────────────────

def _check_assertion(page, description: str, selector: str) -> tuple[bool, str]:
    """Return (passed, error_detail)."""
    if selector.startswith("not:"):
        inner = selector[4:]
        try:
            loc = page.locator(inner)
            count = loc.count()
            if count > 0:
                return False, f"Unexpected content found: {inner!r}"
            return True, ""
        except Exception as exc:
            return True, ""  # selector failed → content absent → ok
    else:
        try:
            loc = page.locator(selector).first
            loc.wait_for(state="visible", timeout=5_000)
            return True, ""
        except Exception as exc:
            return False, str(exc)[:120]


def run_tab_checks(page, snap_dir: Path) -> list[dict]:
    results = []
    page.goto(DASH_URL, wait_until="networkidle", timeout=60_000)
    time.sleep(2)  # let Gradio finish mounting components

    for tab_name, btn_text, assertions, wait_secs in TAB_SPECS:
        _head(f"[{tab_name}]")
        tab_results: list[tuple[bool, str, str]] = []

        # Click the tab
        try:
            page.get_by_role("tab", name=btn_text).click(timeout=8_000)
            _ok(f"Tab '{tab_name}' clicked")
        except Exception:
            try:
                page.locator(f"button:has-text('{btn_text}')").first.click(timeout=5_000)
                _ok(f"Tab '{tab_name}' clicked (fallback selector)")
            except Exception as exc:
                _fail(f"Could not click tab '{tab_name}': {exc}")
                tab_results.append((False, f"Tab click failed", str(exc)[:100]))
                results.append({
                    "tab": tab_name, "passed": False,
                    "checks": tab_results, "screenshot": None,
                })
                continue

        time.sleep(wait_secs)

        # Run assertions
        for desc, selector in assertions:
            ok, detail = _check_assertion(page, desc, selector)
            tab_results.append((ok, desc, detail))
            if ok:
                _ok(desc)
            else:
                _fail(f"{desc} — {detail}")

        # Screenshot
        snap_path = snap_dir / f"{tab_name.lower()}.png"
        try:
            page.screenshot(path=str(snap_path), full_page=True)
            _ok(f"Screenshot → {snap_path.relative_to(BASE_DIR)}")
        except Exception as exc:
            _warn(f"Screenshot failed: {exc}")
            snap_path = None

        passed = all(ok for ok, _, _ in tab_results)
        results.append({
            "tab": tab_name,
            "passed": passed,
            "checks": tab_results,
            "screenshot": snap_path,
        })

    return results


# ── Report ────────────────────────────────────────────────────────────────────

def print_report(results: list[dict], snap_dir: Path) -> int:
    _head("=" * 60)
    _head("REPORT")
    print("=" * 60)
    failed_tabs = [r for r in results if not r["passed"]]
    passed_tabs = [r for r in results if r["passed"]]

    for r in results:
        icon = f"{_GREEN}PASS{_RESET}" if r["passed"] else f"{_RED}FAIL{_RESET}"
        print(f"  {icon}  {r['tab']}")
        if not r["passed"]:
            for ok, desc, detail in r["checks"]:
                if not ok:
                    print(f"         ✗ {desc}")
                    if detail:
                        print(f"           {detail}")

    print()
    print(f"  Tabs passed : {len(passed_tabs)}/{len(results)}")
    print(f"  Screenshots : {snap_dir}")
    print()

    if failed_tabs:
        print(f"{_RED}{_BOLD}  {len(failed_tabs)} tab(s) FAILED{_RESET}")
        return 1
    print(f"{_GREEN}{_BOLD}  All tabs passed ✓{_RESET}")
    return 0


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="TradeGenius Playwright UI check")
    parser.add_argument("--headed",    action="store_true",
                        help="Show browser window (default: headless)")
    parser.add_argument("--no-server", action="store_true",
                        help="Skip server startup — assume :7860 is already running")
    parser.add_argument("--timeout",   type=int, default=120,
                        help="Server startup timeout in seconds (default 120)")
    args = parser.parse_args()

    # ── Server ─────────────────────────────────────────────────────────────────
    proc = None
    if args.no_server:
        if not _server_alive():
            print(f"{_RED}Error: --no-server set but nothing is running on {DASH_URL}{_RESET}")
            sys.exit(1)
        print(f"Using existing server on {DASH_URL}")
    else:
        if _server_alive():
            print(f"Server already running on {DASH_URL} — skipping startup")
        else:
            proc = start_server()
            if not wait_for_server(timeout=args.timeout):
                print(f"{_RED}Server did not start within {args.timeout}s{_RESET}")
                if proc:
                    proc.terminate()
                sys.exit(1)

    # ── Snapshot directory ─────────────────────────────────────────────────────
    stamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
    snap_dir = SNAP_ROOT / stamp
    snap_dir.mkdir(parents=True, exist_ok=True)

    # ── Browser ────────────────────────────────────────────────────────────────
    exit_code = 1
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=not args.headed)
            ctx  = browser.new_context(viewport={"width": 1440, "height": 900})
            page = ctx.new_page()
            results  = run_tab_checks(page, snap_dir)
            browser.close()
        exit_code = print_report(results, snap_dir)
    except ImportError:
        print(f"{_RED}playwright not installed. Run: pip install playwright && playwright install chromium{_RESET}")
        exit_code = 1
    finally:
        if proc:
            proc.terminate()

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
