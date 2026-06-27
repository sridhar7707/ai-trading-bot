#!/usr/bin/env python3
"""Living requirements tracker for TradeGenius AI.

Maintains docs/REQUIREMENTS.md, docs/specs/ and docs/bugs/ from a JSON
state file.  Run after any project change to keep documentation current.

Usage:
  python tests/requirements_tracker.py              # scan + update
  python tests/requirements_tracker.py --dry-run   # preview, no writes
  python tests/requirements_tracker.py --status    # compact overview
  python tests/requirements_tracker.py --bug "desc" [--severity high]
  python tests/requirements_tracker.py --fix BUG-001
  python tests/requirements_tracker.py --complete SPEC-4
"""
import argparse, datetime, json, os, re, sys
from pathlib import Path

ROOT         = Path(__file__).resolve().parent.parent
STATE_FILE   = ROOT / "tests" / "req_snapshots" / "req_state.json"
SNAP_DIR     = ROOT / "tests" / "req_snapshots"
REQ_MD       = ROOT / "docs" / "REQUIREMENTS.md"
SPECS_DIR    = ROOT / "docs" / "specs"
BUGS_DIR     = ROOT / "docs" / "bugs"
REPORTS_DIR  = ROOT / "tests" / "reports"
UI_SNAP      = ROOT / "tests" / "snapshots" / "snapshot_latest.json"

_STATUS = {
    "complete":    "✅ Complete",
    "in_progress": "🔄 In Progress",
    "planned":     "⏳ Planned",
    "blocked":     "❌ Blocked",
    "has_bug":     "🐛 Has Bug",
}
_SEV = {
    "critical": "🔴 Critical",
    "high":     "🟠 High",
    "medium":   "🟡 Medium",
    "low":      "🟢 Low",
}

# ── Initial seed data ─────────────────────────────────────────────────────────

_FEATURES = [
    {"name": "Dashboard Hero & Health Cards",      "status": "complete",    "spec": "SPEC 1",  "last_updated": "2026-06-13", "notes": "Portfolio value, P&L, cash, VIX, health score cards"},
    {"name": "Portfolio Health Score",             "status": "complete",    "spec": "SPEC 1",  "last_updated": "2026-06-13", "notes": "Score 0-100 from VIX/cash/concentration/drawdown"},
    {"name": "Rich Telegram BUY/SELL Alerts",      "status": "complete",    "spec": "SPEC 2",  "last_updated": "2026-06-13", "notes": "Confidence %, SHAP drivers, sector %, cash after"},
    {"name": "Since Yesterday Panel",              "status": "complete",    "spec": "SPEC 3",  "last_updated": "2026-06-13", "notes": "render_whats_changed(): ensemble/regime/sentiment delta"},
    {"name": "AI Action Column (HOLD/TRIM/EXIT)",  "status": "complete",    "spec": "SPEC 4",  "last_updated": "2026-06-13", "notes": "Per-position sell score 0-100, sub-row with top reasons"},
    {"name": "Daily Summary Telegram Alert",       "status": "complete",    "spec": "SPEC 5",  "last_updated": "2026-06-13", "notes": "4:05pm ET: portfolio value, best/worst trade, health score"},
    {"name": "Portfolio Performance Periods",      "status": "complete",    "spec": "SPEC 6A", "last_updated": "2026-06-13", "notes": "1D/1W/1M/3M/YTD/1Y/All Time tabs with headline stats"},
    {"name": "Per-Stock Performance Columns",      "status": "complete",    "spec": "SPEC 6B", "last_updated": "2026-06-13", "notes": "yfinance 1D/1W/1M/1Y/All Time % per position (1h cache)"},
    {"name": "Sparkline Charts",                   "status": "complete",    "spec": "SPEC 6C", "last_updated": "2026-06-13", "notes": "80×32 SVG 30-day trend column in positions table"},
    {"name": "UI/UX Test Suite",                   "status": "complete",    "spec": "SPEC 7",  "last_updated": "2026-06-13", "notes": "14 test files in tests/ including test_dashboard_render.py"},
    {"name": "UI Change Log",                      "status": "complete",    "spec": "SPEC 8",  "last_updated": "2026-06-13", "notes": "tests/ui_changelog.py; 20 render_* components tracked"},
    {"name": "Living Requirements Tracker",        "status": "complete",    "spec": "SPEC 9",  "last_updated": "2026-06-26", "notes": "tests/requirements_tracker.py — scan, --status, --bug, --fix, --complete, --dry-run. SPEC 38/39 added."},
    {"name": "Rebalance Suggestions",              "status": "complete",    "spec": "SPEC 38", "last_updated": "2026-06-26", "notes": "render_rebalance_suggestions(): grouped action plan — reduce/exit rows, add rows, net cash Δ, sector shift. Portfolio tab below Rebalance."},
    {"name": "Paper Trading Scorecard",            "status": "complete",    "spec": "SPEC 39", "last_updated": "2026-06-26", "notes": "render_paper_trading_scorecard(): bot return vs SPY/QQQ, Sharpe, max DD, win rate, AI-follow rate (30d). Models tab, always visible."},
    # Backend features
    {"name": "5-min Trading Loop",                 "status": "complete",    "spec": "SPEC 10", "last_updated": "2026-06-13", "notes": "GitHub Actions cron; market-hours + holiday detection; HALT_TRADING emergency override", "category": "backend"},
    {"name": "Pre-market Screener",                "status": "complete",    "spec": "SPEC 11", "last_updated": "2026-06-13", "notes": "universe_today.json → screener_log; RL agent ranks candidates; separate premarket job", "category": "backend"},
    {"name": "Technical Feature Engineering",      "status": "complete",    "spec": "SPEC 12", "last_updated": "2026-06-13", "notes": "compute_features(): ATR, RSI, EMA, volume ratio, 15-min RSI via 5-min bars", "category": "backend"},
    {"name": "Market Regime Classifier",           "status": "complete",    "spec": "SPEC 13", "last_updated": "2026-06-13", "notes": "TRENDING / RANGING / BEARISH / VOLATILE labels; entry gated to TRENDING + RANGING only", "category": "backend"},
    {"name": "XGBoost Signal Model",               "status": "complete",    "spec": "SPEC 14", "last_updated": "2026-06-13", "notes": "Probability-calibrated; SHAP feature_drivers stored per trade; pre-market retrain", "category": "backend"},
    {"name": "LSTM Signal Model",                  "status": "complete",    "spec": "SPEC 15", "last_updated": "2026-06-13", "notes": "30-bar rolling window; loaded once in run_loop to avoid per-cycle startup cost", "category": "backend"},
    {"name": "Sentiment Pipeline",                 "status": "complete",    "spec": "SPEC 16", "last_updated": "2026-06-13", "notes": "FinBERT premarket batch (NewsAPI) + Reddit/WSB dynamic weighting (log1p mentions, 5-min cache)", "category": "backend"},
    {"name": "FRED Macro Signals",                 "status": "complete",    "spec": "SPEC 17", "last_updated": "2026-06-13", "notes": "VIX >= 40 halts all buys; macro score + size cap; 4-hour DB-backed cache", "category": "backend"},
    {"name": "Ensemble Signal",                    "status": "complete",    "spec": "SPEC 18", "last_updated": "2026-06-13", "notes": "Weighted: XGB + LSTM + sentiment + macro → STRONG_BUY / BUY / HOLD / SELL", "category": "backend"},
    {"name": "Entry Gate Suite",                   "status": "complete",    "spec": "SPEC 19", "last_updated": "2026-06-13", "notes": "10 gates: VIX halt / regime / volume / 15-min RSI / RS / open-order / earnings / correlation / wash-sale / stop re-entry + Kelly sizing", "category": "backend"},
    {"name": "Exit Logic Suite",                   "status": "complete",    "spec": "SPEC 20", "last_updated": "2026-06-13", "notes": "Gap-down floor → take-profit (3xATR, 6-8%) → ATR stop → trailing stop → drift trim → time-exit → ensemble sell", "category": "backend"},
    {"name": "Risk Manager",                       "status": "complete",    "spec": "SPEC 21", "last_updated": "2026-06-13", "notes": "Daily 2% / weekly 5% loss limits; PDT 3-trade gate; drawdown circuit-breaker; portfolio-high tracking", "category": "backend"},
    {"name": "Alpaca Execution Engine",            "status": "complete",    "spec": "SPEC 22", "last_updated": "2026-06-13", "notes": "Limit buy + fill confirmation; limit/market sell with stop-timeout escalation; slippage logging", "category": "backend"},
    {"name": "SQLite Data Layer",                  "status": "complete",    "spec": "SPEC 23", "last_updated": "2026-06-13", "notes": "8 tables: trades, position_state, risk_state, earnings_cache, macro_cache, portfolio_snapshots, signal_log, screener_log", "category": "backend"},
    {"name": "HuggingFace DB Bridge",              "status": "complete",    "spec": "SPEC 24", "last_updated": "2026-06-13", "notes": "sync_db.py pushes trades.db at most every 15 min; dashboard reads from HF dataset repo", "category": "backend"},
    {"name": "Telegram Alert System",              "status": "complete",    "spec": "SPEC 25", "last_updated": "2026-06-13", "notes": "BUY / SELL / stop-loss / risk-warning / VIX-halt / daily-summary / weekly-report alerts", "category": "backend"},
    {"name": "Position Reconciliation",            "status": "complete",    "spec": "SPEC 26", "last_updated": "2026-06-13", "notes": "Startup sync: removes stale DB entries, logs SELL_RECONCILE; seeds externally-opened positions", "category": "backend"},
]

_DECISIONS = [
    {"title": "One Gradio app not three",
     "body": "Single shared portfolio and AI committee view reduces maintenance overhead."},
    {"title": "Pure Python / Gradio over React",
     "body": "Matches ML codebase; all UI rendered as HTML strings inside gr.HTML components."},
    {"title": "yfinance for historical prices (SPEC 6B/6C)",
     "body": "Alpaca handles live intraday data; yfinance provides multi-year history for free."},
    {"title": "SQLite synced via HuggingFace dataset repo",
     "body": "Simple file-based persistence without external DB. trades.db pushed after each cycle."},
    {"title": "Module-level _CACHE with 55-second TTL",
     "body": "All render functions share one DB read per 60-second refresh to prevent N×DB calls."},
    {"title": "GitHub Actions cron + HF Spaces auto-deploy",
     "body": "Bot runs on scheduled GH Actions workflows. Dashboard auto-deploys from main branch."},
]

_LIMITATIONS = [
    "Day trading not implemented — intentional to avoid the PDT rule",
    "Historical performance shows '—' until enough bot history accumulates",
    "yfinance slow on first load with 10+ open positions (cached for 1 hour after)",
    "Dashboard refreshes every 60 seconds (Gradio Timer)",
    "Paper trading only — real money deployment gated behind confidence threshold",
    "SQLite not suitable for high-frequency writes; fine for swing-trading cadence",
]

_ENHANCEMENTS = [
    {"date": "2026-06-13", "title": "Portfolio Health Score (SPEC 1)",
     "lines": ["Added to render_dashboard_hero(); score from VIX, cash, concentration, drawdown",
               "Green ≥ 75, purple ≥ 50, red < 50 with progress bar and weakest-component subtitle"]},
    {"date": "2026-06-13", "title": "Rich Telegram Alerts (SPEC 2)",
     "lines": ["BUY: ensemble %, XGBoost/LSTM %, SHAP drivers in plain English, sector %, cash %",
               "SELL: exit reason label, freed cash %. Daily summary at 4:05pm ET (SPEC 5)"]},
    {"date": "2026-06-13", "title": "Since Yesterday Panel (SPEC 3)",
     "lines": ["render_whats_changed(): compares latest-per-symbol between today and yesterday",
               "Detects ensemble_score (>0.05), regime label, sentiment direction changes"]},
    {"date": "2026-06-13", "title": "AI Action Column (SPEC 4)",
     "lines": ["HOLD/WATCH/TRIM/EXIT badge from sell score (size risk + profit + confidence + DD)",
               "Sub-row below each position row shows top two scoring reasons in plain text"]},
    {"date": "2026-06-13", "title": "Portfolio Performance + yfinance + Sparklines (SPEC 6A/6B/6C)",
     "lines": ["1D/1W/1M/3M/YTD/1Y/All Time Radio tabs at top of Portfolio tab",
               "Per-stock % columns in positions table via yfinance with 1-hour module-level cache",
               "80×32px SVG sparkline (last 30 closes) as second column in positions table"]},
    {"date": "2026-06-13", "title": "UI Change Tracker (SPEC 8)",
     "lines": ["tests/ui_changelog.py: snapshots all render_* functions via ast, diffs on each run",
               "Appends entries to docs/UI_CHANGELOG.md; supports --diff, --history, --reset"]},
]

# ── State I/O ─────────────────────────────────────────────────────────────────

def _load_state() -> dict:
    if not STATE_FILE.exists():
        return {"version": [1, 0, 0], "last_updated": "",
                "features": _FEATURES, "bugs": [],
                "enhancements": _ENHANCEMENTS,
                "technical_decisions": _DECISIONS,
                "known_limitations": _LIMITATIONS}
    return json.loads(STATE_FILE.read_text(encoding="utf-8"))


def _save_state(state: dict) -> None:
    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    state["last_updated"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    v = state["version"]
    v[2] += 1
    data = json.dumps(state, indent=2)
    STATE_FILE.write_text(data, encoding="utf-8")
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    (SNAP_DIR / f"req_{ts}.json").write_text(data, encoding="utf-8")


# ── Project scanning ──────────────────────────────────────────────────────────

def _scan_ui_components() -> list:
    if not UI_SNAP.exists():
        return []
    try:
        return list(json.loads(UI_SNAP.read_text(encoding="utf-8")).get("components", {}).keys())
    except Exception:
        return []


def _scan_new_spec_files(state: dict) -> list:
    if not SPECS_DIR.exists():
        return []
    known = {f.get("spec", "") for f in state.get("features", [])}
    new = []
    for fp in sorted(SPECS_DIR.glob("SPEC_*.md")):
        m = re.match(r'SPEC_0*(\d+[A-Z]?)', fp.stem, re.IGNORECASE)
        if m:
            sid = f"SPEC {m.group(1).upper()}"
            if sid not in known:
                title = re.sub(r'^SPEC_\w+_', '', fp.stem).replace("_", " ").title()
                new.append({"name": title, "status": "planned", "spec": sid,
                            "last_updated": datetime.date.today().isoformat(),
                            "notes": "Auto-detected from docs/specs/"})
    return new


def _scan_test_reports(state: dict) -> list:
    if not REPORTS_DIR.exists():
        return []
    seen = {b.get("description", "").lower() for b in state.get("bugs", [])}
    new_descs = []
    for report in sorted(REPORTS_DIR.glob("*.txt"), key=lambda p: p.stat().st_mtime, reverse=True)[:5]:
        try:
            for m in re.finditer(r'FAIL[ED]?\s*[:\-]\s*([^\n]+)', report.read_text(errors="ignore"), re.IGNORECASE):
                desc = m.group(1).strip()[:100]
                if desc.lower() not in seen:
                    new_descs.append(desc)
                    seen.add(desc.lower())
        except Exception:
            pass
    return new_descs


# ── Markdown generation ───────────────────────────────────────────────────────

def _ver(state: dict) -> str:
    v = state.get("version", [1, 0, 0])
    return f"{v[0]}.{v[1]}.{v[2]}"


def _generate_markdown(state: dict) -> str:
    now = state.get("last_updated", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    HR  = "\n---\n"
    out = [
        "# TradeGenius — Living Requirements Document",
        "",
        "Auto-generated and auto-updated.",
        f"Last updated: {now}",
        f"Version: {_ver(state)}",
        HR,
        "## PROJECT OVERVIEW",
        "Name: TradeGenius AI",
        "Type: AI Portfolio Copilot",
        "Stack: Python, Gradio, Alpaca, HuggingFace Spaces",
        "Purpose: Long-term and swing investment guidance with AI recommendations,",
        "         portfolio health monitoring, and automated execution with full explainability.",
        HR,
        "## FEATURE STATUS",
        "",
    ]
    _CAT_LABELS = {"core": "### CORE FEATURES", "backend": "### BACKEND FEATURES"}
    _cats: dict[str, list] = {}
    for f in state.get("features", []):
        _cats.setdefault(f.get("category", "core"), []).append(f)
    for cat, label in _CAT_LABELS.items():
        feats = _cats.get(cat, [])
        if not feats:
            continue
        out += [label, "| Feature | Status | Spec | Last Updated | Notes |", "|---------|--------|------|--------------|-------|"]
        for f in feats:
            icon = _STATUS.get(f.get("status", "planned"), "⏳ Planned")
            out.append(f'| {f["name"]} | {icon} | {f["spec"]} | {f.get("last_updated","—")} | {f.get("notes","—")} |')
        out.append("")
    out += [
        "Status legend:",
        "✅ Complete — built and tested  🔄 In Progress — currently being worked on",
        "⏳ Planned — specified but not started  ❌ Blocked — cannot proceed",
        "🐛 Has Bug — working but known issue exists",
        HR,
        "## ENHANCEMENTS LOG",
        "Chronological list of all improvements:",
        "",
    ]
    for e in state.get("enhancements", []):
        out.append(f'### [{e["date"]}] {e["title"]}')
        for line in e.get("lines", []):
            out.append(f"- {line}")
        out.append("")
    out += [
        HR,
        "## BUG TRACKER",
        "",
        "| ID | Description | Severity | Status | File | Discovered | Fixed |",
        "|----|-------------|----------|--------|------|------------|-------|",
    ]
    bugs = state.get("bugs", [])
    if bugs:
        for b in bugs:
            out.append(
                f'| {b["id"]} | {b["description"]} '
                f'| {_SEV.get(b.get("severity","medium"),"🟡 Medium")} '
                f'| {_STATUS.get(b.get("status","open"),"🔄 In Progress")} '
                f'| {b.get("file","—")} | {b.get("discovered","—")} | {b.get("fixed","—")} |'
            )
    else:
        out.append("| — | No bugs recorded | — | — | — | — | — |")
    out += [
        "",
        "Severity: 🔴 Critical  🟠 High  🟡 Medium  🟢 Low",
        HR,
        "## TECHNICAL DECISIONS",
        "",
    ]
    for td in state.get("technical_decisions", []):
        out += [f'### {td["title"]}', td["body"], ""]
    out += [
        HR,
        "## KNOWN LIMITATIONS",
        "",
    ]
    for lim in state.get("known_limitations", []):
        out.append(f"- {lim}")
    # Next priorities
    items = [
        f'{f["spec"]} — {f["name"]} (in progress)'
        for f in state.get("features", []) if f.get("status") == "in_progress"
    ] + [
        f'{b["id"]} — {b["description"]}'
        for b in bugs if b.get("status") not in ("fixed", "complete")
    ] + [
        f'{f["spec"]} — {f["name"]}'
        for f in state.get("features", []) if f.get("status") == "planned"
    ]
    out += [HR, "## NEXT PRIORITIES", "Auto-updated based on planned specs and open bugs:", ""]
    if items:
        for i, item in enumerate(items[:10], 1):
            out.append(f"{i}. {item}")
    else:
        out.append("All planned features complete. 🎉")
    out.append("")
    return "\n".join(out)


# ── Spec file initialisation ──────────────────────────────────────────────────

_SPEC_DEFS = {
    "SPEC_01": ("Portfolio Health Score",        "complete",
                "Score 0-100 in dashboard hero from VIX, cash reserve, concentration, drawdown."),
    "SPEC_02": ("Rich Telegram Alerts",          "complete",
                "BUY/SELL alerts with ensemble confidence, SHAP drivers, sector %, cash after."),
    "SPEC_03": ("Since Yesterday Panel",         "complete",
                "render_whats_changed(): compares latest per-symbol ensemble/regime/sentiment."),
    "SPEC_04": ("AI Action Column",              "complete",
                "Per-position HOLD/WATCH/TRIM/EXIT badge with sell score 0-100 in positions table."),
    "SPEC_05": ("Daily Summary Alert",           "complete",
                "4:05pm ET Telegram: portfolio value, best/worst trade, health score, cash %."),
    "SPEC_06A": ("Portfolio Performance Periods", "complete",
                 "1D/1W/1M/3M/YTD/1Y/All Time Radio tabs at top of Portfolio tab."),
    "SPEC_06B": ("Per-Stock Performance Columns", "complete",
                 "yfinance 1D/1W/1M/1Y/All Time % columns in positions table (1-hour cache)."),
    "SPEC_06C": ("Sparkline Charts",             "complete",
                 "80×32 SVG 30-day price sparkline as second column in positions table."),
    "SPEC_07": ("UI/UX Test Suite",              "complete",
                "14-file automated test suite covering render functions and bot components."),
    "SPEC_08": ("UI Change Log",                 "complete",
                "tests/ui_changelog.py: snapshots render_* functions, diffs, updates docs/UI_CHANGELOG.md."),
    "SPEC_09": ("Living Requirements Tracker",   "complete",
                "tests/requirements_tracker.py maintains docs/REQUIREMENTS.md from JSON state. Full CLI: --status, --bug, --fix, --complete, --dry-run."),
    "SPEC_38": ("Rebalance Suggestions",         "complete",
                "render_rebalance_suggestions(): reduce/exit/add rows, net cash change, sector risk shift. Portfolio tab."),
    "SPEC_39": ("Paper Trading Scorecard",       "complete",
                "render_paper_trading_scorecard(): bot vs SPY/QQQ return, Sharpe, max DD, win rate, AI follow rate. Models tab."),
}


def _init_specs() -> None:
    SPECS_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.date.today().isoformat()
    for slug, (title, status, req) in _SPEC_DEFS.items():
        safe = re.sub(r'[^A-Za-z0-9]+', '_', title).lower()
        path = SPECS_DIR / f"{slug}_{safe}.md"
        if path.exists():
            continue
        completed = today if status == "complete" else "—"
        path.write_text(
            f"# {slug.replace('_', ' ')} — {title}\n\n"
            f"Status: {_STATUS.get(status, '⏳ Planned')}\n"
            f"Created: {today}\nCompleted: {completed}\n"
            f"Files changed: See git log\n\n"
            f"## Requirements\n{req}\n\n"
            f"## Implementation Notes\nSee dashboard/app.py and relevant bot files.\n\n"
            f"## Test Results\nRun: `python tests/ui_changelog.py --diff`\n",
            encoding="utf-8",
        )


# ── CLI commands ──────────────────────────────────────────────────────────────

def _next_bug_id(state: dict) -> str:
    ids = [int(b["id"].replace("BUG-", "")) for b in state.get("bugs", []) if b["id"].startswith("BUG-")]
    return f"BUG-{(max(ids, default=0) + 1):03d}"


def cmd_bug(state: dict, description: str, severity: str = "medium") -> None:
    bid   = _next_bug_id(state)
    today = datetime.date.today().isoformat()
    state["bugs"].append({"id": bid, "description": description, "severity": severity,
                           "status": "open", "file": "—", "discovered": today, "fixed": "—"})
    BUGS_DIR.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r'[^\w]+', '_', description.lower())[:40]
    (BUGS_DIR / f"{bid.replace('-', '_')}_{slug}.md").write_text(
        f"# {bid} — {description}\n\nSeverity: {_SEV.get(severity, severity)}\n"
        f"Status: 🔄 In Progress\nDiscovered: {today}\nFixed: —\nFile: —\n\n"
        f"## Description\n{description}\n\n## Root Cause\nTBD\n\n"
        f"## Fix Applied\nTBD\n\n## Verified By\nTBD\n",
        encoding="utf-8",
    )
    print(f"Added {bid}: {description}")


def cmd_fix(state: dict, bug_id: str) -> None:
    today = datetime.date.today().isoformat()
    uid   = bug_id.upper()
    for b in state.get("bugs", []):
        if b["id"].upper() == uid:
            b["status"] = "fixed"
            b["fixed"]  = today
            for fp in BUGS_DIR.glob(f"{uid.replace('-', '_')}*.md"):
                txt = fp.read_text(encoding="utf-8")
                fp.write_text(re.sub(r'Status: .*', "Status: ✅ Fixed",
                              re.sub(r'Fixed: —', f"Fixed: {today}", txt)), encoding="utf-8")
            print(f"Marked {uid} as fixed.")
            return
    print(f"Bug {bug_id} not found.")


def cmd_complete(state: dict, spec_arg: str) -> None:
    today  = datetime.date.today().isoformat()
    target = re.sub(r'\s+', ' ', spec_arg.upper().replace("SPEC-", "SPEC ").replace("SPEC", "SPEC ")).strip()
    for feat in state.get("features", []):
        if feat.get("spec", "").upper() == target:
            feat["status"]       = "complete"
            feat["last_updated"] = today
            print(f"Marked {feat['spec']} — {feat['name']} as complete.")
            return
    print(f"'{spec_arg}' not found. Known specs: {', '.join(f.get('spec','') for f in state['features'])}")


def cmd_status(state: dict) -> None:
    feats = state.get("features", [])
    bugs  = state.get("bugs", [])
    done  = sum(1 for f in feats if f.get("status") == "complete")
    wip   = sum(1 for f in feats if f.get("status") == "in_progress")
    plan  = sum(1 for f in feats if f.get("status") == "planned")
    pct   = int(done / len(feats) * 100) if feats else 0
    print("\nTradeGenius Requirements Status")
    print("================================")
    print(f"Features:  {done} complete / {wip} in progress / {plan} planned")
    print(f"Bugs:      {sum(1 for b in bugs if b.get('status')=='fixed')} fixed / "
          f"{sum(1 for b in bugs if b.get('status')!='fixed')} open")
    print(f"Coverage:  {pct}% of features complete")
    print(f"Version:   {_ver(state)}")
    print(f"Updated:   {state.get('last_updated', '—')}\n")


def cmd_update(state: dict, dry_run: bool = False) -> None:
    changed = False
    for feat in _scan_new_spec_files(state):
        state["features"].append(feat)
        print(f"  + New spec: {feat['spec']} — {feat['name']}")
        changed = True
    for desc in _scan_test_reports(state):
        cmd_bug(state, desc, "medium")
        changed = True
    if not changed:
        print("No changes detected.")
    if dry_run:
        print("(dry-run — no files written)")
        return
    _save_state(state)
    REQ_MD.parent.mkdir(parents=True, exist_ok=True)
    REQ_MD.write_text(_generate_markdown(state), encoding="utf-8")
    print(f"Updated {REQ_MD}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="TradeGenius requirements tracker")
    ap.add_argument("--dry-run",  action="store_true")
    ap.add_argument("--status",   action="store_true")
    ap.add_argument("--bug",      metavar="DESC")
    ap.add_argument("--severity", default="medium", choices=["critical", "high", "medium", "low"])
    ap.add_argument("--fix",      metavar="BUG-ID")
    ap.add_argument("--complete", metavar="SPEC-N")
    args = ap.parse_args()

    state = _load_state()

    if args.status:
        cmd_status(state)
        return

    if args.bug:
        cmd_bug(state, args.bug, args.severity)
        _save_state(state)
        REQ_MD.parent.mkdir(parents=True, exist_ok=True)
        REQ_MD.write_text(_generate_markdown(state), encoding="utf-8")
        return

    if args.fix:
        cmd_fix(state, args.fix)
        _save_state(state)
        REQ_MD.write_text(_generate_markdown(state), encoding="utf-8")
        return

    if args.complete:
        cmd_complete(state, args.complete)
        _save_state(state)
        REQ_MD.write_text(_generate_markdown(state), encoding="utf-8")
        return

    first_run = not STATE_FILE.exists()
    if first_run:
        _init_specs()
        BUGS_DIR.mkdir(parents=True, exist_ok=True)
        components = _scan_ui_components()
        _save_state(state)
        REQ_MD.parent.mkdir(parents=True, exist_ok=True)
        REQ_MD.write_text(_generate_markdown(state), encoding="utf-8")
        n_files = sum(1 for _ in ROOT.rglob("*.py") if ".git" not in str(_))
        print(f"TradeGenius requirements initialized.")
        print(f"Found {len(components)} UI components across {n_files} Python files.")
        print(f"Created {REQ_MD}")
        return

    cmd_update(state, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
