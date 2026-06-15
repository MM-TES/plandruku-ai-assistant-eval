"""Aggregate the 4-class eval reports into the production-readiness gates table.

Reads ``R_and_D/assistant_production_eval/gates.json``, resolves every gate's value
from the latest LABELED report (``reports/red_team/*.json`` / ``reports/pii/*.json`` /
``reports/sci/history.csv``) or from a pinned cited m2 number, applies the target op,
and writes ``gates_table.md`` + ``gates_resolved.json``. Gates whose fresh run has not
happened yet resolve to PENDING (the script is safe to run at any phase). The
ship/not-ship narrative stays HAND-written in REPORT.md — this script only refreshes
the auditable numbers it cites.

    python scripts/production_readiness_report.py
    python scripts/production_readiness_report.py --gates R_and_D/assistant_production_eval/gates.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from src.utils.logger import setup_logger  # noqa: E402

_logger = setup_logger("production_readiness_report")
_REPORT_DIRS = {"red_team": "reports/red_team", "pii": "reports/pii"}


def _latest_labeled_json(area: str, label: str) -> tuple[dict | None, str | None]:
    """Newest ``<stamp>_<label>.json`` in the area's reports dir (stamp-sorted)."""
    folder = _ROOT / _REPORT_DIRS[area]
    hits = sorted(folder.glob(f"*_{label}.json")) if folder.is_dir() else []
    if not hits:
        return None, None
    payload = json.loads(hits[-1].read_text(encoding="utf-8"))
    return payload, hits[-1].name


def _sci_history_row(label: str) -> dict | None:
    """LAST row of reports/sci/history.csv with the given label."""
    path = _ROOT / "reports" / "sci" / "history.csv"
    if not path.is_file():
        return None
    row = None
    with path.open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            if r.get("label") == label:
                row = r
    return row


def _to_num(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _resolve_fresh(source: dict) -> tuple[Any, str | None, Any]:
    """→ (value, source_note, n) for the gate's fresh measurement (None = pending)."""
    stype = source["type"]
    if stype == "cited":
        return None, None, None
    field = source["field"]
    fields = field if isinstance(field, list) else [field]
    if stype in ("red_team", "pii"):
        payload, fname = _latest_labeled_json(stype, source["label"])
        if payload is None:
            return None, None, None
        scope = payload.get("summary", payload)
        vals = [scope.get(f) for f in fields]
        n = scope.get("n")  # pii summary carries n; red_team gates are absolute counts
    elif stype == "sci_history":
        row = _sci_history_row(source["label"])
        if row is None:
            return None, None, None
        vals = [_to_num(row.get(f)) for f in fields]
        fname = f"sci/history.csv [{row.get('timestamp')}]"
        n = _to_num(row.get("n"))
    else:
        raise ValueError(f"unknown source type: {stype}")
    agg = source.get("agg")
    if agg == "sum":
        value: Any = sum(_to_num(v) or 0.0 for v in vals)
    elif agg == "all":
        value = [bool(v) for v in vals]
    else:
        value = vals[0]
    return value, fname, n


def _check(target: str, value: Any) -> bool | None:
    """Apply a ``>=x`` / ``<=x`` / ``==x`` / ``all_true`` target. None = not resolvable."""
    if value is None:
        return None
    if target == "all_true":
        return isinstance(value, list) and all(value)
    num = _to_num(value)
    thr = _to_num(target[2:])  # all ops are 2-char: ">=x" / "<=x" / "==x"
    if num is None or thr is None:
        return None
    if target.startswith(">="):
        return num >= thr
    if target.startswith("<="):
        return num <= thr
    if target.startswith("=="):
        return num == thr
    raise ValueError(f"unknown target op: {target}")


def _status(target: str, value: Any, near_margin: float) -> str:
    ok = _check(target, value)
    if ok is None:
        return "PENDING"
    if ok:
        return "PASS"
    num, thr = _to_num(value), _to_num(target[2:])
    if num is not None and thr is not None and thr != 0:
        if target.startswith(">=") and num >= thr * (1 - near_margin):
            return "NEAR"
        if target.startswith("<=") and num <= thr * (1 + near_margin):
            return "NEAR"
    return "ESCALATE"


def _fmt(value: Any, n: Any = None) -> str:
    if value is None:
        return "—"
    if isinstance(value, list):
        s = "/".join(str(bool(v)).lower() for v in value)
    elif isinstance(value, float):
        s = f"{value:.3f}".rstrip("0").rstrip(".") if value % 1 else f"{value:.0f}"
    else:
        s = str(value)
    n_num = _to_num(n)
    return f"{s} (n={n_num:.0f})" if n_num else s


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Refresh the production-readiness gates table.")
    ap.add_argument("--gates", default="R_and_D/assistant_production_eval/gates.json")
    args = ap.parse_args(argv)
    cfg = json.loads((_ROOT / args.gates).read_text(encoding="utf-8"))
    near = float(cfg.get("near_margin", 0.10))

    resolved: list[dict] = []
    for gate in cfg["gates"]:
        src = gate["source"]
        fresh_val, fresh_src, fresh_n = _resolve_fresh(src)
        cited = gate.get("cited") or (src if src["type"] == "cited" else None)
        cited_val = cited.get("value") if cited else None
        # status from the FRESH number when it exists, else from the cited one
        basis_val = fresh_val if fresh_val is not None else cited_val
        status = _status(gate["target"], basis_val, near)
        if fresh_val is None and cited_val is not None and status != "PENDING":
            status += " (cited)"
        resolved.append({
            "id": gate["id"], "class": gate["class"], "name": gate["name"],
            "target": gate["target"], "fresh": fresh_val, "fresh_n": fresh_n,
            "fresh_source": fresh_src, "cited": cited_val,
            "cited_note": (f'{cited.get("label")} {cited.get("date")}'
                           + (f' n={cited["n"]}' if cited and cited.get("n") else ""))
                          if cited else None,
            "status": status,
        })

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%MZ")
    lines = [
        f"<!-- AUTO-GENERATED by scripts/production_readiness_report.py @ {stamp} -->",
        "",
        "| # | Клас | Метрика | Target | Fresh | Cited (m2) | Статус |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in resolved:
        lines.append(
            f"| {r['id']} | {r['class']} | {r['name']} | `{r['target']}` "
            f"| {_fmt(r['fresh'], r['fresh_n'])} "
            f"| {_fmt(r['cited'])}{' — ' + r['cited_note'] if r['cited_note'] else ''} "
            f"| **{r['status']}** |")
    lines += ["", "Джерела fresh-чисел:"]
    for r in resolved:
        if r["fresh_source"]:
            lines.append(f"- {r['id']}: `{r['fresh_source']}`")
    out_md = _ROOT / cfg["output_md"]
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (_ROOT / cfg["output_json"]).write_text(
        json.dumps({"generated": stamp, "gates": resolved}, ensure_ascii=False, indent=2),
        encoding="utf-8")

    for r in resolved:
        print(f"  {r['id']:>4} {r['status']:<14} {r['name']}: "
              f"fresh={_fmt(r['fresh'], r['fresh_n'])} cited={_fmt(r['cited'])} "
              f"target={r['target']}")
    print(f"  → {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
