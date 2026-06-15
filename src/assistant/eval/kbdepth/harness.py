"""KB datasheet-depth eval harness (INC-0 / P0) — the keystone autonomous gate.

Layer A (PRIMARY · free · deterministic · no API): for each golden product, retrieve
from the ISOLATED eval index using the stored uk+en query variants (no translator)
and measure numeric context-recall (fraction of the datasheet's known spec numbers
present in the retrieved context) + source-recall. This is the shallowness detector
and the per-step acceptance gate.

Layer B (optional · costed · ``--answers``): run the orchestrator end-to-end in «kb»
mode and measure whether the final ANSWER contains the numbers (answer_recall) +
answer groundedness; ``--judge`` adds a focused LLM depth score. Skipped gracefully
when ``OPENROUTER_API_KEY`` is absent. ``--eval-index`` points answer() at the eval
index (via ``KB_PERSIST_DIR``) instead of the production index.

    python -m src.assistant.eval.kbdepth.harness --build --label baseline
    python -m src.assistant.eval.kbdepth.harness --label after_p13
    python -m src.assistant.eval.kbdepth.harness --answers --judge --eval-index --label ma_on

Env: miniconda base (full stack). Exit code ≠0 if any product is below its
``min_recall`` (so the harness doubles as a CI-style gate).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_ROOT))

from src.assistant import config  # noqa: E402
from src.assistant.eval.kbdepth import build_eval_index  # noqa: E402
from src.assistant.eval.kbdepth.golden import GoldenItem, load_golden, numbers_in, recall  # noqa: E402
from src.utils.logger import setup_logger  # noqa: E402

_logger = setup_logger("kbdepth.harness")
_REPORTS = _ROOT / "reports" / "kbdepth"
_HISTORY = _REPORTS / "history.csv"
# Science-eval reports live in a separate tree so the datasheet history.csv schema
# (numeric ctx-recall) is never mixed with the science metric columns.
_SCI_REPORTS = _ROOT / "reports" / "sci"
_SCI_HISTORY = _SCI_REPORTS / "history.csv"


# ── config snapshot (so each report records what produced its numbers) ──────────
def _flags_snapshot() -> dict[str, Any]:
    def _kb_enabled(name: str) -> bool:
        v = config.kb_param(name, {})
        return bool(v.get("enabled", False)) if isinstance(v, dict) else False

    def _feat(name: str) -> bool:
        try:
            return bool(config.feature(name))
        except Exception:  # noqa: BLE001 — net-new flags may be absent
            return False

    def _agent_flag(group: str, key: str) -> bool:
        try:
            return bool(config.agents_param(group, {}).get(key, False))
        except Exception:  # noqa: BLE001 — net-new agent flags may be absent
            return False

    return {
        "metadata_scope": _kb_enabled("metadata_scope"),
        "parent_merge": _kb_enabled("parent_merge"),
        "structured_pdf": _kb_enabled("structured_pdf"),
        "context_prefix": _kb_enabled("context_prefix"),
        "multi_agent": _feat("multi_agent"),
        # science levers (SA-N) — all default False until their increment ships:
        "rerank": _feat("rerank"),
        "kb_rerank": bool(config.kb_param("rerank", False)),  # kb.rerank is a bool, not a {enabled} dict
        "retrieve_before_refuse": _feat("retrieve_before_refuse"),
        "crag_gate": _feat("crag_gate"),
        "honest_marker": _feat("honest_marker"),
        "source_quotas": bool(config.kb_param("source_quotas_enabled", False)),
        "topic_scope": _kb_enabled("topic_scope"),
        # answer-layer levers (Blocks 4a/5) — default False until their flag is flipped:
        "claim_check": _agent_flag("answer_critic", "claim_check"),
        "reasoning": _agent_flag("controller", "reasoning"),
        "sci_full": _agent_flag("controller", "sci_full"),
        "embed_model": config.kb_param("embed_model", "?"),
        "top_k": int(config.kb_param("top_k", 10)),
    }


# ── Layer A: numeric context-recall (free, deterministic) ───────────────────────
def _retrieve_context(item: GoldenItem, top_k: int | None) -> tuple[str, list[str]]:
    from src.assistant.kb.index import KBRetriever

    retr = KBRetriever(persist_dir=build_eval_index.EVAL_PERSIST_DIR)
    if not retr.available:
        raise RuntimeError(
            f"Eval index missing at {build_eval_index.EVAL_PERSIST_DIR} — run with --build first."
        )
    hits = retr.retrieve_multi(item.queries, top_k)
    context = "\n\n".join(c.text for c, _ in hits)
    sources = [c.source for c, _ in hits]
    return context, sources


def _layer_a(items: list[GoldenItem], top_k: int | None) -> list[dict[str, Any]]:
    from src.assistant.eval import evaluators

    rows: list[dict[str, Any]] = []
    for it in items:
        ctx, sources = _retrieve_context(it, top_k)
        r, found, missing = recall(it.golden_set, ctx)
        src_hit = it.source in set(sources)
        row = {
            "product": it.product, "lang": it.lang,
            "ctx_recall": round(r, 3), "passed": bool(r >= it.min_recall),
            "min_recall": it.min_recall, "source_recall": 1.0 if src_hit else 0.0,
            "found": len(found), "total": len(it.golden_set),
            "missing": sorted(missing),
        }
        # rank-aware retrieval metrics over the SAME retrieved source order
        row.update(evaluators.ranking_metrics(sources, it.expected_sources))
        rows.append(row)
    return rows


# ── Layer B: answer-level depth (optional, costed) ──────────────────────────────
_DEPTH_JUDGE = """Оцінюєш ГЛИБИНУ відповіді про технічну плівку «{product}».
Питання оператора: {query}
Відповідь помічника:
\"\"\"{answer}\"\"\"
Відомі числові характеристики (еталон): {golden}
Поверни ЧИСТИЙ JSON {{"depth": x, "reason": "..."}} де depth 0..1 — частка еталонних
числових характеристик, реально наведених у відповіді (а не загальний опис."""


def _depth_judge(item: GoldenItem, answer: str, judge_usage) -> float:
    try:
        from src.assistant.llm import call_llm
        from src.assistant.tracing import parse_json_object

        payload = _DEPTH_JUDGE.format(
            product=item.product, query=item.queries[0], answer=(answer or "")[:1800],
            golden=", ".join(item.numbers),
        )
        resp = call_llm(
            agent_name="kbdepth_judge", role_key="judge",
            messages=[{"role": "user", "content": payload}],
            usage=judge_usage, temperature=0.0, max_tokens=300,
            response_format={"type": "json_object"},
        )
        parsed = parse_json_object(resp.choices[0].message.content or "{}")
        return round(float(parsed.get("depth", 0.0)), 3)
    except Exception as exc:  # noqa: BLE001 — judge is best-effort
        _logger.info("depth judge failed for %s: %s", item.product, str(exc)[:100])
        return -1.0


def _layer_b(items: list[GoldenItem], judge: bool) -> list[dict[str, Any]]:
    from src.assistant import orchestrator
    from src.assistant.eval import evaluators
    from src.assistant.llm import LLMUsage
    from src.assistant.schema import AssistantRequest, PageContext

    judge_usage = LLMUsage()
    rows: list[dict[str, Any]] = []
    for it in items:
        req = AssistantRequest(message=it.queries[0], page_context=PageContext(route="/"), mode="kb")
        t0 = time.perf_counter()
        try:
            resp = orchestrator.answer(req)
        except Exception as exc:  # noqa: BLE001 — one failed answer must not kill the batch
            _logger.warning("Layer-B answer failed for %s: %s", it.product, str(exc)[:120])
            rows.append({"product": it.product, "answer_recall": 0.0, "answer_grounded": 0.0,
                         "cost_usd": 0.0, "error": str(exc)[:120]})
            continue
        wall_ms = int((time.perf_counter() - t0) * 1000)
        ans = resp.text_md or ""
        present = numbers_in(ans)
        found = {g for g in it.golden_set if g in present}
        ans_recall = len(found) / len(it.golden_set) if it.golden_set else 1.0
        grounded = evaluators.groundedness(ans, resp.tool_trace, resp.evidence)["score"]
        row = {
            "product": it.product,
            "answer_recall": round(ans_recall, 3),
            "answer_grounded": grounded,
            "cost_usd": round(float(resp.usage.get("cost_usd", 0.0)), 5),
            "wall_ms": wall_ms,
        }
        if judge:
            row["depth_judge"] = _depth_judge(it, ans, judge_usage)
        rows.append(row)
    return rows


# ── summary + reports ───────────────────────────────────────────────────────────
def _mean(rows: list[dict], key: str) -> float | None:
    vals = [float(r[key]) for r in rows if isinstance(r.get(key), (int, float)) and r[key] >= 0]
    return round(sum(vals) / len(vals), 3) if vals else None


def _summarise(a_rows: list[dict], b_rows: list[dict]) -> dict[str, Any]:
    s: dict[str, Any] = {
        "n": len(a_rows),
        "mean_ctx_recall": _mean(a_rows, "ctx_recall"),
        "mean_source_recall": _mean(a_rows, "source_recall"),
        "mean_recall_at_1": _mean(a_rows, "recall_at_1"),
        "mean_recall_at_5": _mean(a_rows, "recall_at_5"),
        "mean_recall_at_10": _mean(a_rows, "recall_at_10"),
        "mean_mrr": _mean(a_rows, "mrr"),
        "mean_ndcg": _mean(a_rows, "ndcg"),
        "n_passed": sum(1 for r in a_rows if r.get("passed")),
        "mean_answer_recall": _mean(b_rows, "answer_recall") if b_rows else None,
        "mean_answer_grounded": _mean(b_rows, "answer_grounded") if b_rows else None,
        "mean_depth_judge": _mean(b_rows, "depth_judge") if b_rows else None,
        "total_cost_usd": round(sum(float(r.get("cost_usd", 0.0)) for r in b_rows), 5) if b_rows else 0.0,
    }
    return s


def _append_history_row(path: Path, row: dict[str, Any]) -> None:
    """Append *row* to a history.csv, AUTO-ROTATING when the column set changes.

    ``csv.DictWriter`` only writes a header when the file is new, so adding a metric
    column to an existing history would silently misalign old vs new rows. Here we
    compare the existing header line to the new field set and, on mismatch, rename the
    old file to ``<name>.<stamp>.csv`` and start a fresh file with the new header — so
    schema evolution never corrupts the trend. Same schema → plain append (unchanged)."""
    fieldnames = list(row.keys())
    new_header = ",".join(fieldnames)
    existing_header: str | None = None
    if path.is_file():
        with path.open("r", encoding="utf-8") as f:
            existing_header = (f.readline() or "").strip()
    if existing_header is not None and existing_header != new_header:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path.rename(path.with_name(f"{path.stem}.{stamp}{path.suffix}"))
        existing_header = None
    with path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if existing_header is None:
            w.writeheader()
        w.writerow(row)


def _write_reports(label: str, payload: dict[str, Any]) -> Path:
    _REPORTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = _REPORTS / f"{stamp}_{label}.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (_REPORTS / "latest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    flags = payload["flags"]
    s = payload["summary"]
    row = {
        "timestamp": stamp, "label": label, "n": s["n"],
        "mean_ctx_recall": s["mean_ctx_recall"], "mean_source_recall": s["mean_source_recall"],
        "mean_recall_at_1": s.get("mean_recall_at_1"), "mean_recall_at_5": s.get("mean_recall_at_5"),
        "mean_recall_at_10": s.get("mean_recall_at_10"), "mean_mrr": s.get("mean_mrr"),
        "mean_ndcg": s.get("mean_ndcg"),
        "n_passed": s["n_passed"], "mean_answer_recall": s["mean_answer_recall"],
        "mean_answer_grounded": s["mean_answer_grounded"], "mean_depth_judge": s["mean_depth_judge"],
        "total_cost_usd": s["total_cost_usd"],
        "metadata_scope": flags["metadata_scope"], "parent_merge": flags["parent_merge"],
        "structured_pdf": flags["structured_pdf"], "multi_agent": flags["multi_agent"],
        "kb_rerank": flags.get("kb_rerank"),
        "embed_model": flags["embed_model"], "top_k": flags["top_k"],
    }
    _append_history_row(_HISTORY, row)
    return out


def _print_table(a_rows: list[dict], b_rows: list[dict]) -> None:
    bmap = {r["product"]: r for r in b_rows}
    print("\n  product        lang  ctx_recall  src  found/total  pass" + ("  ans_recall  grnd" if b_rows else ""))
    print("  " + "-" * (62 + (20 if b_rows else 0)))
    for r in a_rows:
        line = (f"  {r['product']:<14} {r['lang']:<4}  {r['ctx_recall']:>9.2f}  "
                f"{int(r['source_recall']):>3}  {r['found']:>2}/{r['total']:<2}        "
                f"{'OK' if r['passed'] else 'FAIL':<4}")
        if b_rows:
            b = bmap.get(r["product"], {})
            line += f"  {b.get('answer_recall', 0):>9.2f}  {b.get('answer_grounded', 0):>4.2f}"
        print(line)


# ── Science eval: source-recall (free) + routing/abstention/citation/faithfulness ──
# Unlike the datasheet harness, science items query the FULL PRODUCTION index (the
# literature/standards/patents live there, not in the datasheet-only eval index).
def _sci_retrieve_sources(item: GoldenItem, top_k: int | None) -> list[str]:
    from src.assistant.kb.index import get_kb_retriever

    retr = get_kb_retriever()
    if not retr.available:
        raise RuntimeError(
            "Production KB index unavailable — build it first "
            "(python scripts/build_knowledge_base_rag.py)."
        )
    hits = retr.retrieve_multi(item.queries, top_k or int(config.kb_param("top_k", 10)))
    return [c.source for c, _ in hits]


def _sci_layer_a(items: list[GoldenItem], top_k: int | None) -> list[dict[str, Any]]:
    from src.assistant.eval import evaluators

    rows: list[dict[str, Any]] = []
    for it in items:
        sources = _sci_retrieve_sources(it, top_k)
        sr = evaluators.source_recall(sources, it.expected_sources)
        row = {
            "id": it.product, "lang": it.lang, "category": it.category,
            "abstain_expected": it.abstain_expected,
            "source_recall": sr.get("score"), "coverage": sr.get("coverage"),
            "matched": sr.get("matched", []), "top_sources": sources[:5],
        }
        row.update(evaluators.ranking_metrics(sources, it.expected_sources))
        rows.append(row)
    return rows


def _sci_layer_b(items: list[GoldenItem], judge: bool) -> list[dict[str, Any]]:
    from src.assistant import orchestrator
    from src.assistant.eval import evaluators
    from src.assistant.llm import LLMUsage
    from src.assistant.schema import AssistantRequest, PageContext

    judge_usage = LLMUsage()
    rows: list[dict[str, Any]] = []
    for it in items:
        # Default «hybrid» mode (NOT forced kb) so routing itself is under test.
        req = AssistantRequest(message=it.queries[0], page_context=PageContext(route="/"))
        t0 = time.perf_counter()
        try:
            resp = orchestrator.answer(req)
        except Exception as exc:  # noqa: BLE001 — one failed answer must not kill the batch
            _logger.warning("Layer-B answer failed for %s: %s", it.product, str(exc)[:120])
            rows.append({"id": it.product, "route": "error", "route_expected": it.route_expected or "any",
                         "route_match": 0.0, "abstained": False, "abstain_expected": it.abstain_expected,
                         "abstention_correct": 0.0, "error": str(exc)[:120],
                         "wall_ms": int((time.perf_counter() - t0) * 1000)})
            continue
        wall_ms = int((time.perf_counter() - t0) * 1000)
        text = resp.text_md or ""
        abstained = evaluators.is_abstention(text, resp.route)
        ab = evaluators.abstention_correctness(abstained, it.abstain_expected)
        route_match = 1.0 if (not it.route_expected or resp.route == it.route_expected) else 0.0
        cpr = evaluators.citation_pr(resp.citations, it.expected_sources)
        row: dict[str, Any] = {
            "id": it.product, "route": resp.route, "route_expected": it.route_expected or "any",
            "route_match": route_match,
            "abstained": abstained, "abstain_expected": it.abstain_expected,
            "abstention_correct": ab["score"],
            "citation_recall": cpr.get("recall"), "citation_precision": cpr.get("precision"),
            "n_citations": len(resp.citations or []),
            "cost_usd": round(float(resp.usage.get("cost_usd", 0.0)), 5),
            # m2 instrumentation: end-to-end latency + the answer itself, so every
            # costed run doubles as a latency sample (p95 gate) and a forensic asset.
            "wall_ms": wall_ms,
            "answer_head": text[:600],
        }
        # Faithfulness only meaningful when an answer was expected (not for abstain items).
        if judge and not it.abstain_expected:
            row["faithfulness"] = evaluators.faithfulness(text, resp.evidence, usage=judge_usage)["score"]
        rows.append(row)
    return rows


def _percentile(vals: list[float], pct: float) -> float | None:
    """Nearest-rank percentile (small-n friendly, no numpy dependency here): the value at
    1-based ordinal rank ceil(pct/100 · n). p95 of 1..100 → rank 95 → 95.0."""
    import math

    if not vals:
        return None
    vs = sorted(vals)
    k = max(1, min(len(vs), math.ceil(pct / 100.0 * len(vs))))
    return vs[k - 1]


def _sci_summarise(a_rows: list[dict], b_rows: list[dict]) -> dict[str, Any]:
    s: dict[str, Any] = {
        "n": len(a_rows),
        "mean_source_recall": _mean(a_rows, "source_recall"),
        "mean_coverage": _mean(a_rows, "coverage"),
        "mean_recall_at_1": _mean(a_rows, "recall_at_1"),
        "mean_recall_at_5": _mean(a_rows, "recall_at_5"),
        "mean_recall_at_10": _mean(a_rows, "recall_at_10"),
        "mean_mrr": _mean(a_rows, "mrr"),
        "mean_ndcg": _mean(a_rows, "ndcg"),
    }
    if b_rows:
        walls = [float(r["wall_ms"]) for r in b_rows if isinstance(r.get("wall_ms"), (int, float))]
        s.update({
            "route_accuracy": _mean(b_rows, "route_match"),
            "abstention_correctness": _mean(b_rows, "abstention_correct"),
            "mean_citation_recall": _mean(b_rows, "citation_recall"),
            "mean_citation_precision": _mean(b_rows, "citation_precision"),
            "mean_faithfulness": _mean(b_rows, "faithfulness"),
            "total_cost_usd": round(sum(float(r.get("cost_usd", 0.0)) for r in b_rows), 5),
            # m2: end-to-end latency percentiles (p95 gate ≤ 6000 ms) ride every run.
            "p50_wall_ms": _percentile(walls, 50),
            "p95_wall_ms": _percentile(walls, 95),
        })
    return s


def _write_sci_reports(label: str, payload: dict[str, Any]) -> Path:
    _SCI_REPORTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = _SCI_REPORTS / f"{stamp}_{label}.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (_SCI_REPORTS / "latest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    flags, s = payload["flags"], payload["summary"]
    row = {
        "timestamp": stamp, "label": label, "n": s["n"],
        "mean_source_recall": s.get("mean_source_recall"), "mean_coverage": s.get("mean_coverage"),
        "mean_recall_at_1": s.get("mean_recall_at_1"), "mean_recall_at_5": s.get("mean_recall_at_5"),
        "mean_recall_at_10": s.get("mean_recall_at_10"), "mean_mrr": s.get("mean_mrr"),
        "mean_ndcg": s.get("mean_ndcg"),
        "route_accuracy": s.get("route_accuracy"), "abstention_correctness": s.get("abstention_correctness"),
        "mean_citation_recall": s.get("mean_citation_recall"),
        "mean_citation_precision": s.get("mean_citation_precision"),
        "mean_faithfulness": s.get("mean_faithfulness"), "total_cost_usd": s.get("total_cost_usd"),
        "p50_wall_ms": s.get("p50_wall_ms"), "p95_wall_ms": s.get("p95_wall_ms"),
        "rerank": flags["rerank"], "kb_rerank": flags["kb_rerank"],
        "retrieve_before_refuse": flags["retrieve_before_refuse"], "crag_gate": flags["crag_gate"],
        "honest_marker": flags["honest_marker"], "source_quotas": flags["source_quotas"],
        "claim_check": flags.get("claim_check"), "reasoning": flags.get("reasoning"),
        "sci_full": flags.get("sci_full"),
        "multi_agent": flags["multi_agent"], "embed_model": flags["embed_model"], "top_k": flags["top_k"],
    }
    _append_history_row(_SCI_HISTORY, row)
    return out


def _sci_print_table(a_rows: list[dict], b_rows: list[dict]) -> None:
    bmap = {r["id"]: r for r in b_rows}
    head = "\n  id              lang  cat            src_rec"
    if b_rows:
        head += "  route(exp)            abst  cit_r"
    print(head)
    print("  " + "-" * (44 + (40 if b_rows else 0)))
    for r in a_rows:
        sr = r.get("source_recall")
        sr_s = " n/a " if sr is None else f"{sr:>4.1f}"
        line = f"  {str(r['id'])[:14]:<14} {str(r['lang']):<4}  {str(r['category'])[:12]:<12}  {sr_s:>5}"
        if b_rows:
            b = bmap.get(r["id"], {})
            rt = f"{str(b.get('route',''))[:10]}({str(b.get('route_expected',''))[:8]})"
            ab = "OK" if b.get("abstention_correct") else "x"
            cr = b.get("citation_recall")
            cr_s = "n/a" if cr is None else f"{cr:>3.1f}"
            line += f"  {rt:<20}  {ab:<4}  {cr_s}"
        print(line)


# Cheap routing-only probe (router.classify = Haiku) — measures T0.1/T0.2 misroute
# fixes without the costed Sonnet answer+judge.
def _sci_route_probe(items: list[GoldenItem]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from src.assistant import router
    from src.assistant.llm import LLMUsage
    from src.assistant.schema import PageContext

    usage = LLMUsage()
    rows: list[dict[str, Any]] = []
    for it in items:
        rr = router.classify(it.queries[0], PageContext(route="/"), usage=usage)
        match = 1.0 if (not it.route_expected or rr.route == it.route_expected) else 0.0
        rows.append({
            "id": it.product, "lang": it.lang, "category": it.category,
            "route": rr.route, "route_expected": it.route_expected or "any",
            "route_match": match, "source": rr.source,
            "abstain_expected": it.abstain_expected,
        })
    scored = [r for r in rows if r["route_expected"] != "any"]
    dist: dict[str, int] = {}
    for r in rows:
        dist[r["route"]] = dist.get(r["route"], 0) + 1
    summary = {
        "n": len(rows),
        "route_accuracy": round(sum(r["route_match"] for r in scored) / len(scored), 3) if scored else None,
        "n_instructions": dist.get("instructions", 0),
        "n_out_of_scope": dist.get("out_of_scope", 0),
        "n_data_routes": sum(dist.get(x, 0) for x in ("data_query", "analysis", "history")),
        "distribution": dist,
        "cost_usd": round(float(usage.as_dict().get("cost_usd", 0.0)), 5),
    }
    return rows, summary


# ── Operator-help eval (m2): Layer-A source-recall over the SEPARATE operator RAG ──
# The operator-help retriever (src/assistant/rag/index.py → models/assistant_rag) is a
# DIFFERENT index + embedder from the external KB and was previously UNMEASURED.
# Mirrors production: one raw uk query (no translator), rag_top_k default.
_OP_REPORTS = _ROOT / "reports" / "operator"
_OP_HISTORY = _OP_REPORTS / "history.csv"


def run_operator(*, label: str, top_k: int | None, golden: str | None) -> dict[str, Any]:
    from src.assistant.eval import evaluators
    from src.assistant.rag.index import get_retriever

    items = [it for it in (load_golden(golden) if golden else load_golden())
             if it.kind == "operator"]
    if not items:
        _logger.error('Operator golden set empty — need items with "kind": "operator" in %s',
                      golden or "the golden jsonl")
        return {"summary": {"n": 0}, "exit": 2}

    retr = get_retriever()
    if not retr.available:
        raise RuntimeError(
            "Operator RAG index unavailable — build it first (python scripts/build_assistant_rag.py)."
        )
    k = top_k or int(config.threshold("rag_top_k", 4))
    rows: list[dict[str, Any]] = []
    for it in items:
        hits = retr.retrieve(it.queries[0], top_k=k)
        sources = [c.source for c, _ in hits]
        sr = evaluators.source_recall(sources, it.expected_sources)
        row: dict[str, Any] = {
            "id": it.product, "lang": it.lang, "category": it.category,
            "source_recall": sr.get("score"), "coverage": sr.get("coverage"),
            "matched": sr.get("matched", []), "top_sources": sources[:5],
        }
        row.update(evaluators.ranking_metrics(sources, it.expected_sources))
        rows.append(row)

    summary = {
        "n": len(rows),
        "mean_source_recall": _mean(rows, "source_recall"),
        "mean_recall_at_1": _mean(rows, "recall_at_1"),
        "mean_mrr": _mean(rows, "mrr"),
        "mean_ndcg": _mean(rows, "ndcg"),
        "top_k": k,
    }
    payload = {"label": label, "mode": "operator",
               "timestamp": datetime.now(timezone.utc).isoformat(),
               "flags": _flags_snapshot(), "summary": summary, "layer_a": rows}
    _OP_REPORTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = _OP_REPORTS / f"{stamp}_{label}.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _append_history_row(_OP_HISTORY, {
        "timestamp": stamp, "label": label, "n": summary["n"],
        "mean_source_recall": summary["mean_source_recall"],
        "mean_recall_at_1": summary["mean_recall_at_1"], "mean_mrr": summary["mean_mrr"],
        "mean_ndcg": summary["mean_ndcg"], "top_k": k,
    })
    print("\noperator summary:", json.dumps(summary, ensure_ascii=False))
    print("report:", out)
    payload["exit"] = 0
    return payload


def run_science(*, label: str, answers: bool, judge: bool,
                top_k: int | None, golden: str | None, route_only: bool = False) -> dict[str, Any]:
    items = load_golden(golden) if golden else load_golden()
    items = [it for it in items if it.kind == "science"] or items
    if not items:
        _logger.error("Science golden set empty — check %s", golden or "config/sci_golden.jsonl")
        return {"summary": {"n": 0}, "exit": 2}

    if route_only:
        rows, summary = _sci_route_probe(items)
        payload = {"label": label, "mode": "route_probe",
                   "timestamp": datetime.now(timezone.utc).isoformat(),
                   "flags": _flags_snapshot(), "summary": summary, "rows": rows}
        _SCI_REPORTS.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        (_SCI_REPORTS / f"{stamp}_{label}_route.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print("\n  id              lang  cat            route        exp          match")
        print("  " + "-" * 64)
        for r in rows:
            print(f"  {str(r['id'])[:14]:<14} {str(r['lang']):<4}  {str(r['category'])[:12]:<12}  "
                  f"{str(r['route'])[:10]:<11}  {str(r['route_expected'])[:10]:<11}  "
                  f"{'OK' if r['route_match'] else 'x'}")
        print("\nroute summary:", json.dumps(summary, ensure_ascii=False))
        payload["exit"] = 0
        return payload

    a_rows = _sci_layer_a(items, top_k)
    b_rows: list[dict] = []
    if answers:
        if not config.openrouter_api_key():
            _logger.warning("OPENROUTER_API_KEY absent — science Layer B skipped (source-recall still computed).")
        else:
            b_rows = _sci_layer_b(items, judge)

    summary = _sci_summarise(a_rows, b_rows)
    payload = {
        "label": label, "mode": "science",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "flags": _flags_snapshot(), "summary": summary,
        "layer_a": a_rows, "layer_b": b_rows if answers else "skipped",
    }
    out = _write_sci_reports(label, payload)
    _sci_print_table(a_rows, b_rows)
    print("\nscience summary:", json.dumps(summary, ensure_ascii=False))
    print("report:", out)
    payload["exit"] = 0  # science is a measurement baseline, not a hard CI gate
    return payload


# ── CLI ─────────────────────────────────────────────────────────────────────────
def run(*, build: bool, label: str, answers: bool, judge: bool,
        eval_index: bool, top_k: int | None, golden: str | None = None) -> dict[str, Any]:
    if build:
        _logger.info("Building eval index…")
        info = build_eval_index.build()
        _logger.info("eval index: %s", info)
    if eval_index:
        os.environ["KB_PERSIST_DIR"] = str(build_eval_index.EVAL_PERSIST_DIR)

    items = load_golden(golden) if golden else load_golden()
    items = [it for it in items if it.kind == "datasheet"]
    if not items:
        _logger.error("Datasheet golden set empty — check %s", golden or "config/kb_depth_golden.jsonl")
        return {"summary": {"n": 0}, "exit": 2}

    a_rows = _layer_a(items, top_k)
    b_rows: list[dict] = []
    if answers:
        if not config.openrouter_api_key():
            _logger.warning("OPENROUTER_API_KEY absent — Layer B skipped (Layer A still gates).")
        else:
            b_rows = _layer_b(items, judge)

    summary = _summarise(a_rows, b_rows)
    payload = {
        "label": label,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "flags": _flags_snapshot(),
        "summary": summary,
        "layer_a": a_rows,
        "layer_b": b_rows if answers else "skipped",
    }
    out = _write_reports(label, payload)
    _print_table(a_rows, b_rows)
    print("\nsummary:", json.dumps(summary, ensure_ascii=False))
    print("report:", out)
    n_fail = sum(1 for r in a_rows if not r["passed"])
    payload["exit"] = 1 if n_fail else 0
    return payload


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="KB datasheet-depth eval harness")
    ap.add_argument("--build", action="store_true", help="(re)build the isolated eval index first")
    ap.add_argument("--label", default="run", help="report label (e.g. baseline, after_p13, ma_on)")
    ap.add_argument("--answers", action="store_true", help="Layer B: end-to-end answer() depth (costed)")
    ap.add_argument("--judge", action="store_true", help="add LLM depth judge to Layer B (costed)")
    ap.add_argument("--eval-index", action="store_true", help="point answer() at the eval index")
    ap.add_argument("--top-k", type=int, default=None, help="override retrieval top_k (default: config)")
    ap.add_argument("--golden", default=None,
                    help="golden jsonl path (default: config/kb_depth_golden.jsonl)")
    ap.add_argument("--science", action="store_true",
                    help="science eval: source-recall (free) + routing/abstention/citation/faithfulness "
                         "over the PROD index (use with --golden config/sci_golden.jsonl)")
    ap.add_argument("--route-only", action="store_true",
                    help="science: cheap router-only probe (Haiku) — route accuracy/distribution, no Sonnet")
    ap.add_argument("--operator", action="store_true",
                    help="operator-help eval: Layer-A source-recall over the SEPARATE operator RAG "
                         "(models/assistant_rag) for kind=operator golden items (free)")
    args = ap.parse_args(argv)
    if args.operator:
        res = run_operator(label=args.label, top_k=args.top_k, golden=args.golden)
    elif args.science:
        res = run_science(label=args.label, answers=args.answers, judge=args.judge,
                          top_k=args.top_k, golden=args.golden, route_only=args.route_only)
    else:
        res = run(build=args.build, label=args.label, answers=args.answers, judge=args.judge,
                  eval_index=args.eval_index, top_k=args.top_k, golden=args.golden)
    return int(res.get("exit", 0))


if __name__ == "__main__":
    raise SystemExit(main())
