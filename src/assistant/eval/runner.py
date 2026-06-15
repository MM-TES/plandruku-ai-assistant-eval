"""Full eval runner (operator-run; uses real models) + offline scoring helpers.

    python -m src.assistant.eval.runner manual [--limit N]

Runs the golden set through the orchestrator, scores every answer, prints a
summary and writes ``reports/assistant_experiments.csv``. The judge (LLM-as-judge
success_rate) is optional via --judge. The offline gate (tests/) uses the
deterministic evaluators only; this is the costed, end-to-end run.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT))

from src.assistant.data.text2sql import validate_sql  # noqa: E402
from src.assistant.eval import evaluators, synth  # noqa: E402
from src.assistant.llm import LLMUsage  # noqa: E402
from src.assistant.schema import AssistantRequest, PageContext  # noqa: E402
from src.utils.logger import setup_logger  # noqa: E402

_logger = setup_logger("assistant.eval.runner")
_REPORTS = _ROOT / "reports"


def _page_context(page: str | None) -> PageContext:
    if not page:
        return PageContext(route="/")
    route, _, stage = page.partition(":")
    return PageContext(route=route, stage=stage or None)


def _score_query_item(item: dict, *, judge: bool, judge_usage: LLMUsage) -> dict[str, Any]:
    from src.assistant import orchestrator

    req = AssistantRequest(message=item["query"], page_context=_page_context(item.get("page")))
    resp = orchestrator.answer(req)
    route = resp.route
    refused = route == "out_of_scope"
    result = {"refused": refused}
    metrics = {
        "intent_match": evaluators.intent_match(route, item)["score"],
        "tool_selection": evaluators.tool_selection_accuracy(resp.tool_trace, item)["score"],
        "groundedness": evaluators.groundedness(resp.text_md, resp.tool_trace, resp.evidence)["score"],
        "citation": evaluators.citation_grounding([c.model_dump() for c in resp.citations], item)["score"],
        "safety": evaluators.safety_refusal(result, item)["score"],
    }
    if judge:
        metrics["success"] = evaluators.success_rate(resp.text_md, resp.tool_trace, item, judge_usage)["score"]
    return {"query": item["query"], "route": route, "cost_usd": resp.usage.get("cost_usd", 0.0), **metrics}


def _score_sql_item(item: dict) -> dict[str, Any]:
    v = validate_sql(item["sql"])
    return {"query": item["sql"], "route": "text2sql", "safety": 1.0 if not v.ok else 0.0,
            "cost_usd": 0.0}


def run(limit: int | None = None, *, judge: bool = False) -> dict[str, Any]:
    items = synth.load_golden()
    if limit:
        # keep a mix: queries first, then a couple of safety items
        queries = [i for i in items if i.get("query")][:limit]
        unsafe = [i for i in items if i.get("safety_class") == "unsafe_sql"][:3]
        items = queries + unsafe
    judge_usage = LLMUsage()
    rows: list[dict[str, Any]] = []
    for item in items:
        if item.get("sql"):
            rows.append(_score_sql_item(item))
        else:
            rows.append(_score_query_item(item, judge=judge, judge_usage=judge_usage))

    summary = _summarise(rows)
    _write_csv(rows)
    _logger.info("Eval summary: %s", summary)
    return {"rows": rows, "summary": summary}


def _summarise(rows: list[dict]) -> dict[str, Any]:
    def avg(key: str) -> float | None:
        vals = [r[key] for r in rows if key in r]
        return round(sum(vals) / len(vals), 3) if vals else None

    return {
        "n": len(rows),
        "intent_match": avg("intent_match"),
        "tool_selection": avg("tool_selection"),
        "groundedness": avg("groundedness"),
        "citation": avg("citation"),
        "safety": avg("safety"),
        "success": avg("success"),
        "total_cost_usd": round(sum(r.get("cost_usd", 0.0) for r in rows), 4),
    }


def _write_csv(rows: list[dict]) -> None:
    _REPORTS.mkdir(parents=True, exist_ok=True)
    path = _REPORTS / "assistant_experiments.csv"
    keys = sorted({k for r in rows for k in r})
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    _logger.info("Wrote %s", path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Assistant eval runner")
    ap.add_argument("mode", choices=["manual"], help="manual = local run + CSV")
    ap.add_argument("--limit", type=int, default=None, help="cap number of query items (cheap run)")
    ap.add_argument("--judge", action="store_true", help="also run LLM-as-judge success_rate (costed)")
    args = ap.parse_args(argv)
    result = run(limit=args.limit, judge=args.judge)
    print(result["summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
