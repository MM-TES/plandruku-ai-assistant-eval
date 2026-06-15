"""Extra C (capstone) — quality / drift dashboard for the RAG assistant.

A dependency-free, file-based take on the course's monitoring lessons (lesson-12 Langfuse
scoring, lesson-15 Prometheus drift): instead of standing up a metrics stack, it reads the
eval-harness history CSVs (``reports/sci/history.csv``, ``reports/kbdepth/history.csv``),
the judge-calibration reports, and (best-effort) operator feedback, and renders a single
Markdown dashboard with per-run trend tables and a DRIFT section (latest vs the first/
baseline run, with ▲/▼ direction). No runtime coupling — it only reads what eval runs and
the live feedback table already produce, so it is safe to regenerate any time.

``render_markdown`` is pure (rows in → markdown out) and unit-tested; ``build`` wires the
file reads + optional feedback.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[3]
_SCI_HISTORY = _ROOT / "reports" / "sci" / "history.csv"
_KB_HISTORY = _ROOT / "reports" / "kbdepth" / "history.csv"
_CALIB_DIR = _ROOT / "reports" / "judge_calibration"
_OUT = _ROOT / "reports" / "quality_dashboard.md"

# (csv column, display label, higher_is_better) for the science trend + drift.
_SCI_METRICS = [
    ("mean_source_recall", "source-recall", True),
    ("mean_recall_at_1", "recall@1", True),
    ("mean_recall_at_5", "recall@5", True),
    ("mean_recall_at_10", "recall@10", True),
    ("mean_mrr", "MRR", True),
    ("mean_ndcg", "nDCG", True),
    ("route_accuracy", "route-acc", True),
    ("abstention_correctness", "abstention", True),
    ("mean_citation_recall", "cite-recall", True),
    ("mean_faithfulness", "faithfulness", True),
]


def read_history(path: str | Path) -> list[dict[str, Any]]:
    """Read a harness history.csv into a list of row dicts (newest last)."""
    p = Path(path)
    if not p.is_file():
        return []
    with p.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _fnum(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _fmt(v: Any) -> str:
    n = _fnum(v)
    return "—" if n is None else f"{n:.3f}"


def _drift_arrow(first: float | None, last: float | None, higher_is_better: bool) -> str:
    if first is None or last is None:
        return "—"
    d = round(last - first, 3)
    if d == 0:
        return f"→ 0.000"
    good = (d > 0) == higher_is_better
    return f"{'▲' if d > 0 else '▼'} {d:+.3f} {'✓' if good else '✗'}"


def _trend_table(rows: list[dict], metrics: list[tuple[str, str, bool]]) -> str:
    if not rows:
        return "_no runs recorded yet_\n"
    headers = ["run"] + [m[1] for m in metrics] + ["cost$"]
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        cells = [str(r.get("label", "?"))[:22]]
        cells += [_fmt(r.get(col)) for col, _, _ in metrics]
        cells.append(_fmt(r.get("total_cost_usd")))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _drift_table(rows: list[dict], metrics: list[tuple[str, str, bool]]) -> str:
    if len(rows) < 2:
        return "_need ≥2 runs to compute drift_\n"
    first, last = rows[0], rows[-1]
    lines = ["| metric | baseline | latest | drift |", "|---|---|---|---|"]
    for col, label, hib in metrics:
        fb, lt = _fnum(first.get(col)), _fnum(last.get(col))
        lines.append(f"| {label} | {_fmt(fb)} | {_fmt(lt)} | {_drift_arrow(fb, lt, hib)} |")
    return "\n".join(lines) + "\n"


def latest_calibration(calib_dir: str | Path = _CALIB_DIR) -> dict | None:
    """Most recent judge-calibration REPORT payload, if any. Skips non-report files in
    the same dir (e.g. the ``answers_cache*.json`` lists, which sort after the timestamped
    reports and would otherwise be picked as the 'latest')."""
    d = Path(calib_dir)
    if not d.is_dir():
        return None
    for f in sorted(d.glob("*.json"), reverse=True):
        if "cache" in f.name:
            continue
        try:
            payload = json.loads(f.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if isinstance(payload, dict) and "agreement" in payload:
            return payload
    return None


# m2: the business-uplift acceptance gates (01_TZ.md). value_of(latest_sci, calib,
# feedback) → current value; threshold; higher_is_better. 👍-rate / p95 rows appear as
# soon as their data exists — before that they render as "—" (honest unmeasured state).
_GATES: list[tuple[str, str, float, bool]] = [
    ("mean_source_recall", "source-recall", 0.90, True),
    ("mean_recall_at_1", "recall@1", 0.60, True),
    ("mean_faithfulness", "faithfulness", 0.80, True),
    ("abstention_correctness", "abstention-correctness", 0.85, True),
    ("mean_citation_recall", "citation_recall", 0.80, True),
    ("route_accuracy", "route_accuracy", 0.95, True),
    ("p95_wall_ms", "p95 latency (ms)", 6000.0, False),
]


def gates_table(sci_rows: list[dict], calib: dict | None = None,
                feedback: dict | None = None) -> str:
    """m2: metrics-vs-gates table — the mission contract at a glance (pure)."""

    def _latest(col: str) -> float | None:
        for r in reversed(sci_rows):
            v = _fnum(r.get(col))
            if v is not None:
                return v
        return None

    lines = ["| gate | target | latest | pass |", "|---|---|---|---|"]
    for col, label, thr, hib in _GATES:
        v = _latest(col)
        ok = "—" if v is None else ("✓" if ((v >= thr) if hib else (v <= thr)) else "✗")
        target = f"{'≥' if hib else '≤'} {thr:g}"
        lines.append(f"| {label} | {target} | {_fmt(v) if col != 'p95_wall_ms' else (v if v is not None else '—')} | {ok} |")
    calib = calib if isinstance(calib, dict) else {}
    feedback = feedback if isinstance(feedback, dict) else {}
    rho = _fnum(calib.get("agreement", {}).get("overall", {}).get("spearman"))
    rho_ok = "—" if rho is None else ("✓" if rho >= 0.70 else "✗")
    lines.append(f"| judge Spearman ρ | ≥ 0.7 | {_fmt(rho)} | {rho_ok} |")
    up, total = (feedback or {}).get("up", 0), (feedback or {}).get("total", 0)
    rate = (up / total) if total else None
    rate_ok = "—" if rate is None else ("✓" if rate >= 0.80 else "✗")
    lines.append(f"| online 👍-rate | ≥ 0.8 | {_fmt(rate)} | {rate_ok} |")
    return "\n".join(lines) + "\n"


def render_markdown(sci_rows: list[dict], kb_rows: list[dict],
                    calib: dict | None = None, feedback: dict | None = None) -> str:
    """Render the full dashboard markdown from already-loaded data (pure)."""
    out: list[str] = ["# RAG Assistant — Quality & Drift Dashboard", ""]
    out.append(f"_Science eval runs: {len(sci_rows)} · datasheet eval runs: {len(kb_rows)}_\n")

    out.append("## Metrics vs acceptance gates (m2 business-uplift contract)\n")
    out.append(gates_table(sci_rows, calib, feedback))

    out.append("\n## Science eval — trend (newest last)\n")
    out.append(_trend_table(sci_rows, _SCI_METRICS))
    out.append("\n## Science eval — drift (latest vs baseline)\n")
    out.append(_drift_table(sci_rows, _SCI_METRICS))

    if kb_rows:
        kb_metrics = [("mean_ctx_recall", "ctx-recall", True), ("mean_source_recall", "src-recall", True),
                      ("mean_recall_at_1", "recall@1", True), ("mean_mrr", "MRR", True),
                      ("mean_answer_recall", "ans-recall", True)]
        out.append("\n## Datasheet eval — trend\n")
        out.append(_trend_table(kb_rows, kb_metrics))

    out.append("\n## LLM-judge calibration (dual-judge agreement)\n")
    if calib:
        agr = calib.get("agreement", {}).get("overall", {})
        out.append(f"- judges: `{calib.get('judge_a_model')}` vs `{calib.get('judge_b_model')}` "
                   f"(n={agr.get('n', '?')})\n")
        out.append(f"- overall Spearman ρ: **{_fmt(agr.get('spearman'))}**\n")
        weak = calib.get("weak_criteria") or []
        out.append(f"- weak criteria (need rubric anchors): {', '.join(weak) if weak else 'none'}\n")
        out.append(f"- recommendation: {calib.get('recommendation', '')}\n")
    else:
        out.append("_no calibration run recorded_\n")

    out.append("\n## Operator feedback\n")
    if feedback:
        out.append(f"- 👍 {feedback.get('up', 0)} · 👎 {feedback.get('down', 0)} "
                   f"(total {feedback.get('total', 0)})\n")
        by_route = feedback.get("by_route") or {}
        if by_route:
            out.append("\n| route | 👍 | 👎 |\n|---|---|---|\n")
            for route, v in by_route.items():
                out.append(f"| {route} | {v.get('up', 0)} | {v.get('down', 0)} |\n")
    else:
        out.append("_feedback table unavailable (DB offline or no votes yet)_\n")
    return "".join(s if s.endswith("\n") else s + "\n" for s in out)


def build(sci_history: str | Path = _SCI_HISTORY, kb_history: str | Path = _KB_HISTORY,
          out: str | Path = _OUT, feedback: dict | None = None) -> Path:
    """Read the eval histories + latest calibration (+ optional feedback) and write the
    dashboard markdown to *out*."""
    md = render_markdown(read_history(sci_history), read_history(kb_history),
                         latest_calibration(), feedback)
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(md, encoding="utf-8")
    return p


__all__ = ["read_history", "render_markdown", "build", "latest_calibration", "gates_table"]
