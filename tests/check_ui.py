"""
Playwright visual UI check for TradeGenius dashboard.

Starts the Gradio server, opens every tab, asserts every key section is visible,
and takes a full-page screenshot. Run this manually any time you want to verify
the look & feel and data is showing correctly.

Usage:
    python tests/check_ui.py                # headless
    python tests/check_ui.py --headed       # watch the browser live
    python tests/check_ui.py --no-server    # server already on :7860

Screenshots saved to:  tests/snapshots/playwright/<YYYYMMDD_HHMMSS>/
Update TAB_SPECS whenever a section heading or key text changes in the dashboard.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

BASE_DIR  = Path(__file__).parent.parent
DASH_URL  = "http://localhost:7860"
SNAP_ROOT = Path(__file__).parent / "snapshots" / "playwright"

_GREEN  = "\033[92m"
_RED    = "\033[91m"
_YELLOW = "\033[93m"
_RESET  = "\033[0m"
_BOLD   = "\033[1m"


def _ok(msg: str)   -> None: print(f"  {_GREEN}✓{_RESET} {msg}")
def _fail(msg: str) -> None: print(f"  {_RED}✗{_RESET} {msg}")
def _warn(msg: str) -> None: print(f"  {_YELLOW}~{_RESET} {msg}")
def _head(msg: str) -> None: print(f"\n{_BOLD}{msg}{_RESET}")


# ── Per-tab check specs ───────────────────────────────────────────────────────
# (display_name, tab_button_text, assertions, wait_secs)
# Assertions: (description, selector)
#   selector starts with "not:" → content must be ABSENT
#   "not:text=unavailable" is the universal crash detector (safe_render error card)
TAB_SPECS: list[tuple[str, str, list[tuple[str, str]], float]] = [
    (
        "Brief",
        "Brief",
        [
            # Universal crash detector — safe_render error cards contain "unavailable"
            ("No component crash cards",         "not:text=unavailable"),
            # Immediately-rendered sections (value=callable in app.py)
            ("Executive summary rendered",       ".nt-wrap"),
            ("Decision bar rendered",            "text=AI"),
            ("Morning brief rendered",           "text=Morning"),
            ("Positions snapshot rendered",      "text=Position"),
        ],
        4.0,
    ),
    (
        "Portfolio",
        "Portfolio",
        [
            ("No component crash cards",         "not:text=unavailable"),
            ("Portfolio value shown ($)",        "text=$"),
            ("Weekly summary rendered",          "text=Week"),
            ("Daily headline rendered",          "text=Portfolio"),
            ("Equity chart rendered",            "#equity-chart"),
            ("Open positions rendered",          "text=Position"),
            ("Trade log rendered",               "text=Trade"),
            ("Watchlist rendered",               "text=Watch"),
        ],
        4.0,
    ),
    (
        "Capital",
        "Capital",
        [
            ("No component crash cards",         "not:text=unavailable"),
            ("Capital overview rendered",        "text=Initial Deposit"),
            ("Managed Capital Pool shown",       "text=Managed Capital Pool"),
            ("Tradeable cash row shown",         "text=Tradeable"),
            ("Reserve row shown",                "text=Reserve"),
            ("Invested row shown",               "text=Invested"),
            ("Profit breakdown rendered",        "text=Realized"),
            ("Reinvestment toggle shown",        "text=Reinvest"),
        ],
        3.0,
    ),
    (
        "Trades",
        "Trades",
        [
            ("No component crash cards",         "not:text=unavailable"),
            ("Top picks rendered",               "text=Top Pick"),
            ("Trade frequency rendered",         "text=Frequency"),
            ("Buy candidates rendered",          "text=Buy"),
            ("Signal history rendered",          "text=Signal"),
            ("Recommendation history rendered",  "text=Recommendation"),
        ],
        3.0,
    ),
    (
        "Performance",
        "Performance",
        [
            ("No component crash cards",         "not:text=unavailable"),
            ("Institutional metrics rendered",   "text=Win Rate"),
            ("Attribution by symbol shown",      "text=Attribution"),
            ("Investor view rendered",           "text=Investor"),
        ],
        4.0,
    ),
    (
        "Settings",
        "Settings",
        [
            ("No component crash cards",         "not:text=unavailable"),
            ("Risk tolerance radio shown",       "text=Risk Tolerance"),
            ("Stop-loss slider shown",           "text=Stop-Loss"),
            ("Max position slider shown",        "text=Max Position"),
            ("Max drawdown slider shown",        "text=Drawdown"),
            ("Notifications checkbox shown",     "text=Notification"),
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
    print("Starting dashboard server …")
    return subprocess.Popen(
        [sys.executable, "scripts/dashboard.py"],
        cwd=str(BASE_DIR),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def wait_for_server(timeout: int = 120) -> bool:
    print(f"Waiting for server on {DASH_URL} (up to {timeout}s) …", end="", flush=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _server_alive():
            print("  ready")
            return True
        time.sleep(2)
        print(".", end="", flush=True)
    print()
    return False


# ── Assertion runner ──────────────────────────────────────────────────────────

def _check(page, description: str, selector: str) -> tuple[bool, str]:
    if selector.startswith("not:"):
        inner = selector[4:]
        try:
            if page.locator(inner).count() > 0:
                return False, f"Unexpected content found: {inner!r}"
            return True, ""
        except Exception:
            return True, ""
    try:
        page.locator(selector).first.wait_for(state="visible", timeout=5_000)
        return True, ""
    except Exception as exc:
        return False, str(exc)[:120]


def run_all_tabs(page, snap_dir: Path) -> list[dict]:
    results = []
    page.goto(DASH_URL, wait_until="networkidle", timeout=60_000)
    time.sleep(2)

    for tab_name, btn_text, assertions, wait_secs in TAB_SPECS:
        _head(f"[{tab_name}]")
        checks: list[tuple[bool, str, str]] = []

        # Click the tab
        clicked = False
        for selector in [f'[role="tab"]:has-text("{btn_text}")',
                         f'button:has-text("{btn_text}")',
                         f'text="{btn_text}"']:
            try:
                page.locator(selector).first.click(timeout=6_000)
                clicked = True
                break
            except Exception:
                continue
        if not clicked:
            _fail(f"Could not click tab '{tab_name}'")
            results.append({"tab": tab_name, "passed": False,
                            "checks": [(False, "Tab click failed", "")],
                            "screenshot": None})
            continue

        time.sleep(wait_secs)

        for desc, sel in assertions:
            ok, detail = _check(page, desc, sel)
            checks.append((ok, desc, detail))
            _ok(desc) if ok else _fail(f"{desc}  {detail}")

        snap_path = snap_dir / f"{tab_name.lower()}.png"
        try:
            page.screenshot(path=str(snap_path), full_page=True)
            _ok(f"Screenshot → {snap_path.relative_to(BASE_DIR)}")
        except Exception as exc:
            _warn(f"Screenshot failed: {exc}")
            snap_path = None

        results.append({
            "tab":        tab_name,
            "passed":     all(ok for ok, _, _ in checks),
            "checks":     checks,
            "screenshot": snap_path,
        })

    return results


# ── Report ────────────────────────────────────────────────────────────────────

def print_report(results: list[dict], snap_dir: Path) -> int:
    _head("=" * 60)
    _head("REPORT")
    print("=" * 60)
    for r in results:
        status = f"{_GREEN}PASS{_RESET}" if r["passed"] else f"{_RED}FAIL{_RESET}"
        print(f"  {status}  {r['tab']}")
        for ok, desc, detail in r["checks"]:
            if not ok:
                print(f"         ✗ {desc}")
                if detail:
                    print(f"           {detail}")

    failed = [r for r in results if not r["passed"]]
    print(f"\n  Tabs passed : {len(results) - len(failed)}/{len(results)}")
    print(f"  Screenshots : {snap_dir}\n")
    if failed:
        print(f"{_RED}{_BOLD}  {len(failed)} tab(s) FAILED{_RESET}")
        return 1
    print(f"{_GREEN}{_BOLD}  All {len(results)} tabs passed ✓{_RESET}")
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

    stamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
    snap_dir = SNAP_ROOT / stamp
    snap_dir.mkdir(parents=True, exist_ok=True)

    exit_code = 1
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=not args.headed)
            page    = browser.new_context(
                viewport={"width": 1440, "height": 900}
            ).new_page()
            results   = run_all_tabs(page, snap_dir)
            browser.close()
        exit_code = print_report(results, snap_dir)
    except ImportError:
        print(f"{_RED}playwright not installed. Run: pip install playwright && playwright install chromium{_RESET}")
    finally:
        if proc:
            proc.terminate()

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
