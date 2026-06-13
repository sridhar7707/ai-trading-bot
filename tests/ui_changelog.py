#!/usr/bin/env python3
"""Auto-documents UI/UX changes to dashboard/app.py.

Snapshots the source of every render_* function and compares it to
the previous run to detect additions, removals, and modifications.
All state is stored in tests/snapshots/; the living record is
docs/UI_CHANGELOG.md.

Usage:
  python tests/ui_changelog.py           # detect changes, update changelog
  python tests/ui_changelog.py --diff    # show diff without writing files
  python tests/ui_changelog.py --history # compact history to console
  python tests/ui_changelog.py --reset   # delete snapshots (asks confirmation)
"""
import ast
import hashlib
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT         = Path(__file__).resolve().parent.parent
APP_PY       = ROOT / "dashboard" / "app.py"
SNAP_DIR     = ROOT / "tests" / "snapshots"
LATEST_SNAP  = SNAP_DIR / "snapshot_latest.json"
CHANGELOG    = ROOT / "docs" / "UI_CHANGELOG.md"
REPORTS_DIR  = ROOT / "tests" / "reports"

_BADGES = {"HOLD", "TRIM", "EXIT", "WATCH", "BUY", "SELL"}


# ── Source extraction ─────────────────────────────────────────────────────────

def _extract_functions(source: str) -> dict:
    """Return {func_name: source_text} for every render_* function."""
    lines = source.splitlines(keepends=True)
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        print(f"WARNING: syntax error parsing app.py — {exc}")
        return {}
    funcs = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name.startswith("render_"):
            start = node.lineno - 1
            end   = getattr(node, "end_lineno", start + 100)
            funcs[node.name] = "".join(lines[start:end])
    return funcs


def _snap_func(src: str) -> dict:
    """Snapshot one render function: hash + colors + font-sizes + sections + badges."""
    h        = hashlib.md5(src.encode()).hexdigest()
    colors   = sorted(set(re.findall(r'#[0-9a-fA-F]{6}\b|#[0-9a-fA-F]{3}\b', src)))
    fsizes   = sorted(set(re.findall(r'font-size:\s*(\d+(?:\.\d+)?(?:px|em|rem))', src)))
    sections = sorted(set(
        re.findall(r'_section\(\s*["\'][^"\']*["\'],\s*["\']([^"\']+)["\']', src)
    ))
    badges = sorted(b for b in _BADGES if re.search(rf'\b{b}\b', src))
    return {
        "hash": h, "size": len(src),
        "colors": colors, "font_sizes": fsizes,
        "sections": sections, "badges": badges,
    }


def _snapshot() -> dict:
    """Full snapshot of all render_* functions in dashboard/app.py."""
    src   = APP_PY.read_text(encoding="utf-8")
    funcs = _extract_functions(src)
    return {
        "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "file_hash":  hashlib.md5(src.encode()).hexdigest(),
        "file_size":  len(src),
        "components": {name: _snap_func(body) for name, body in funcs.items()},
    }


# ── Snapshot I/O ──────────────────────────────────────────────────────────────

def _load_latest():
    if not LATEST_SNAP.exists():
        return None
    try:
        return json.loads(LATEST_SNAP.read_text(encoding="utf-8"))
    except Exception:
        return None


def _save_snapshot(snap: dict) -> None:
    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    slug = snap["timestamp"].replace(" ", "_").replace(":", "").replace("-", "")
    path = SNAP_DIR / f"snapshot_{slug}.json"
    if path.exists():
        slug += f"_{os.getpid()}"
        path = SNAP_DIR / f"snapshot_{slug}.json"
    data = json.dumps(snap, indent=2)
    path.write_text(data, encoding="utf-8")
    LATEST_SNAP.write_text(data, encoding="utf-8")


# ── Change detection ──────────────────────────────────────────────────────────

def _compare(old: dict, new: dict) -> list:
    old_c = old.get("components", {})
    new_c = new.get("components", {})
    changes = []

    for name, nw in new_c.items():
        if name not in old_c:
            changes.append({"name": name, "kind": "ADDED", "new": nw})
        elif nw["hash"] != old_c[name]["hash"]:
            ol    = old_c[name]
            delta = nw["size"] - ol["size"]
            changes.append({
                "name":             name,
                "kind":             "MODIFIED",
                "delta":            delta,
                "new_colors":       [c for c in nw["colors"]   if c not in ol["colors"]],
                "rem_colors":       [c for c in ol["colors"]   if c not in nw["colors"]],
                "new_sections":     [s for s in nw["sections"] if s not in ol["sections"]],
                "rem_sections":     [s for s in ol["sections"] if s not in nw["sections"]],
                "new_badges":       [b for b in nw["badges"]   if b not in ol["badges"]],
                "font_size_changes":[f for f in nw["font_sizes"] if f not in ol["font_sizes"]],
            })

    for name in old_c:
        if name not in new_c:
            changes.append({"name": name, "kind": "REMOVED", "old": old_c[name]})

    return changes


# ── Test result integration ───────────────────────────────────────────────────

def _read_test_results():
    if not REPORTS_DIR.exists():
        return None
    today = datetime.now().strftime("%Y%m%d")
    candidates = sorted(REPORTS_DIR.glob(f"*{today}*"), reverse=True)
    if not candidates:
        candidates = sorted(
            list(REPORTS_DIR.glob("*.txt")) + list(REPORTS_DIR.glob("*.md")),
            key=lambda p: p.stat().st_mtime, reverse=True,
        )
    if not candidates:
        return None
    try:
        text = candidates[0].read_text(encoding="utf-8", errors="ignore")
        m = re.search(r'SUMMARY[:\s]+(.+)', text, re.IGNORECASE)
        if m:
            return m.group(1).strip()
        p = re.search(r'(\d+)\s+PASS', text, re.IGNORECASE)
        w = re.search(r'(\d+)\s+WARN', text, re.IGNORECASE)
        f = re.search(r'(\d+)\s+FAIL', text, re.IGNORECASE)
        if p or f:
            parts = []
            if p: parts.append(f"{p.group(1)} PASS")
            if w: parts.append(f"{w.group(1)} WARN")
            if f: parts.append(f"{f.group(1)} FAIL")
            return " / ".join(parts)
    except Exception:
        pass
    return None


# ── Changelog formatting ──────────────────────────────────────────────────────

def _format_change(ch: dict) -> str:
    lines = [f"### {ch['name']} — {ch['kind']}"]

    if ch["kind"] == "ADDED":
        nw = ch["new"]
        lines.append("- New component (did not exist before)")
        lines.append(f"- Size: {nw['size']:,} chars")
        if nw["sections"]:
            lines.append(f"- Sections found: {', '.join(repr(s) for s in nw['sections'])}")
        if nw["colors"]:
            lines.append(f"- Colors: {', '.join(nw['colors'])}")
        if nw["badges"]:
            lines.append(f"- Badges: {', '.join(nw['badges'])}")

    elif ch["kind"] == "REMOVED":
        lines.append("- Component removed")
        lines.append(f"- Was: {ch['old']['size']:,} chars")

    else:  # MODIFIED
        d   = ch["delta"]
        adj = " slightly" if 0 < abs(d) < 100 else ""
        dir = ("grew" + adj) if d > 0 else (("shrank" + adj) if d < 0 else "unchanged")
        lines.append(f"- Size: {'+' if d > 0 else ''}{d:,} chars ({dir})")

        if ch["new_sections"]:
            lines.append(f"- New sections: {', '.join(repr(s) for s in ch['new_sections'])}")
        if ch["rem_sections"]:
            lines.append(f"- Removed sections: {', '.join(repr(s) for s in ch['rem_sections'])}")
        if not ch["new_sections"] and not ch["rem_sections"]:
            lines.append("- No section changes")

        if ch["new_badges"]:
            lines.append(f"- New badges: {', '.join(ch['new_badges'])}")

        if ch["new_colors"]:
            lines.append(f"- New colors: {', '.join(ch['new_colors'])}")
        else:
            lines.append("- No colors added")
        if ch["rem_colors"]:
            lines.append(f"- Colors removed: {', '.join(ch['rem_colors'])}")
        else:
            lines.append("- No colors removed")

        if ch["font_size_changes"]:
            lines.append(f"- New font sizes: {', '.join(ch['font_size_changes'])}")

        if d == 0 and not any([
            ch["new_sections"], ch["rem_sections"],
            ch["new_colors"],   ch["rem_colors"], ch["new_badges"],
        ]):
            lines.append("- Note: Minor text or formatting tweak")

    return "\n".join(lines)


def _format_entry(ts: str, changes: list, test_summary) -> str:
    n    = len(changes)
    word = "component" if n == 1 else "components"
    out  = [f"## [{ts}] — {n} {word} changed", ""]
    for ch in changes:
        out.append(_format_change(ch))
        out.append("")
    if test_summary:
        out.append(f"_Tests at time of change: {test_summary}_")
        out.append("")
    out.append("---")
    return "\n".join(out)


# ── Changelog I/O ─────────────────────────────────────────────────────────────

def _changelog_stats(content: str) -> tuple:
    """Return (total_entries, most_frequent_component) from existing markdown."""
    total = len(re.findall(r'^## \[', content, re.MULTILINE))
    names = re.findall(r'^### (\w+)', content, re.MULTILINE)
    if names:
        freq: dict = {}
        for n in names:
            freq[n] = freq.get(n, 0) + 1
        most_freq = max(freq, key=lambda k: freq[k])
    else:
        most_freq = "—"
    return total, most_freq


def _snapshot_date_range() -> tuple:
    snaps = sorted(f.name for f in SNAP_DIR.glob("snapshot_2*.json"))
    if not snaps:
        return "—", "—"

    def _parse(name: str) -> str:
        s = name.replace("snapshot_", "").replace(".json", "")
        try:
            return f"{s[:4]}-{s[4:6]}-{s[6:8]} {s[9:11]}:{s[11:13]}:{s[13:15]}"
        except Exception:
            return s

    return _parse(snaps[0]), _parse(snaps[-1])


def _update_changelog(entry: str, snap: dict) -> None:
    CHANGELOG.parent.mkdir(parents=True, exist_ok=True)
    existing_body = ""
    if CHANGELOG.exists():
        content = CHANGELOG.read_text(encoding="utf-8")
        m = re.search(r'\n---\n', content)
        if m:
            existing_body = content[m.end():]

    full_body    = entry + "\n" + existing_body
    total, freq  = _changelog_stats(full_body)
    first_dt, last_dt = _snapshot_date_range()

    header = (
        "# TradeGenius UI/UX Changelog\n\n"
        "Auto-generated. Do not edit manually.\n"
        f"Last updated: {snap['timestamp']}\n"
        f"Total changes recorded: {total}\n"
        f"Most changed component: {freq}\n"
        f"First snapshot: {first_dt}\n"
        f"Latest snapshot: {last_dt}\n"
        f"Components tracked: {len(snap['components'])}\n\n"
        "---\n\n"
    )
    CHANGELOG.write_text(header + full_body, encoding="utf-8")


# ── CLI helpers ───────────────────────────────────────────────────────────────

def _print_diff(changes: list) -> None:
    if not changes:
        print("No UI changes since last snapshot.")
        return
    n = len(changes)
    print(f"\n{n} component{'s' if n != 1 else ''} changed:\n")
    for ch in changes:
        print(f"  [{ch['kind']}] {ch['name']}")
        if ch["kind"] == "MODIFIED":
            d = ch["delta"]
            print(f"          size {'+' if d > 0 else ''}{d:,} chars")
            if ch["new_colors"]:
                print(f"          + colors: {', '.join(ch['new_colors'])}")
            if ch["new_sections"]:
                print(f"          + sections: {', '.join(ch['new_sections'])}")
            if ch["new_badges"]:
                print(f"          + badges: {', '.join(ch['new_badges'])}")
        print()


def _print_history() -> None:
    if not CHANGELOG.exists():
        print("No changelog yet. Run without flags to create one.")
        return
    content = CHANGELOG.read_text(encoding="utf-8")
    entries = re.finditer(
        r'^## \[(.+?)\] — (\d+) components? changed\s*\n(.*?)(?=^## \[|\Z)',
        content, re.MULTILINE | re.DOTALL,
    )
    rows = []
    for m in entries:
        ts    = m.group(1)
        count = int(m.group(2))
        names = re.findall(r'^### (\w+)', m.group(3), re.MULTILINE)
        short = [n.replace("render_", "") for n in names]
        if len(short) > 3:
            label = ", ".join(short[:3]) + ", ..."
        else:
            label = ", ".join(short)
        rows.append(f"  {ts}  {count} change{'s' if count != 1 else ' '}  ({label})")
    if not rows:
        print("No history entries found.")
        return
    print("\nUI change history (newest first):\n")
    for row in rows:
        print(row)
    print()


def _reset() -> None:
    snaps = list(SNAP_DIR.glob("snapshot*.json")) if SNAP_DIR.exists() else []
    if not snaps:
        print("No snapshots found. Nothing to reset.")
        return
    print(f"Found {len(snaps)} snapshot(s) in {SNAP_DIR}")
    ans = input("Reset all UI snapshots? This cannot be undone. (yes/no): ").strip().lower()
    if ans != "yes":
        print("Aborted.")
        return
    for f in snaps:
        f.unlink()
    print(f"Deleted {len(snaps)} snapshot(s). Run without flags to start fresh.")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    args = set(sys.argv[1:])

    if "--reset" in args:
        _reset()
        return

    if "--history" in args:
        _print_history()
        return

    if not APP_PY.exists():
        print(f"ERROR: {APP_PY} not found.")
        sys.exit(1)

    old = _load_latest()
    new = _snapshot()

    if "--diff" in args:
        if old is None:
            print("No baseline yet. Run without --diff first.")
            return
        _print_diff(_compare(old, new))
        return

    if old is None:
        _save_snapshot(new)
        print(f"Baseline created — tracking {len(new['components'])} components.")
        print(f"Snapshot: {LATEST_SNAP}")
        return

    changes = _compare(old, new)
    if not changes:
        print("No UI changes since last snapshot.")
        return

    entry = _format_entry(new["timestamp"], changes, _read_test_results())
    _save_snapshot(new)
    _update_changelog(entry, new)
    n = len(changes)
    print(f"Recorded {n} change{'s' if n != 1 else ''} → {CHANGELOG}")


if __name__ == "__main__":
    main()
