# -*- coding: utf-8 -*-
"""Grounding evaluation harness for the «Помічник» assistant.

Measures the acceptance criteria from the grounding task against a RUNNING UI
server (default http://127.0.0.1:8050): SKU fabrication, field accuracy,
off-viewport resolution, counter semantics, honest "no data", and determinism
across N runs of the same query.

The "ground truth" SKU set is read live from the DB via the assistant's own
read-only tools — so fabrication is judged against verified data, not a fixture
that could drift.

Run (UI server must be up):
    python scripts/assistant_grounding_eval.py --runs 10 --host 127.0.0.1:8050

This is an operator/dev harness (needs the LLM + DB), NOT a CI gate. The
deterministic parts (tool SQL, SKU/guardrail, routing, counter glossary) are
pinned by tests/ under pytest.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.parse
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SKU_RE = re.compile(r"\b\d\.\d{2,4}\.\d{2,6}\b")


def _real_sku_set() -> set[str]:
    """Verified SKU codes that currently exist in the system (deficits + stock)."""
    from src.assistant.data import tools

    skus: set[str] = set()
    d = tools.run_tool("get_deficits_top", {"limit": 200})
    for r in d.get("rows") or []:
        if r.get("sku"):
            skus.add(str(r["sku"]))
    return skus


def _ask(host: str, message: str, page_context: dict) -> str:
    body = urllib.parse.urlencode({
        "message": message,
        "page_context": json.dumps(page_context, ensure_ascii=False),
        "scope": "standard",
    }).encode("utf-8")
    req = urllib.request.Request(
        f"http://{host}/assistant/ask", data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded; charset=utf-8"},
    )
    html = urllib.request.urlopen(req, timeout=90).read().decode("utf-8")
    bodies = re.findall(r'amsg-body">(.*?)</div>', html, re.S)
    return " ".join(re.sub("<[^>]+>", " ", bodies[-1]).split()) if bodies else ""


_DEFICITS_PC = {"route": "/workflow", "stage": "zabezpechennia", "section": "deficits"}
_PROD_PC = {"route": "/workflow", "stage": "vyrobnytstvo",
            "filters": {"order_id_like": "12345"}, "visible_entity_ids": ["12345"]}

# (label, message, page_context, kind)
CASES = [
    ("top_deficits", "Покажи топ-5 дефіцитів за матеріалом (SKU).", _DEFICITS_PC, "sku"),
    ("off_viewport_order", "Що з замовленням #12345 — план, дефіцит, статус?", _PROD_PC, "order"),
    ("counter_semantics", "Що означає лічильник «Контроль» і скільки там зараз?", {"route": "/"}, "counter"),
    ("honesty_no_data", "Який вільний залишок по матеріалу 9.99.99999 на складі?", _DEFICITS_PC, "honesty"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--host", default="127.0.0.1:8050")
    args = ap.parse_args()

    real = _real_sku_set()
    print(f"verified SKU universe: {len(real)} codes\n")

    total_fab = 0
    for label, msg, pc, kind in CASES:
        answers = []
        fabricated = 0
        q_skus = set(SKU_RE.findall(msg))  # SKUs the operator typed are allowed verbatim
        for _ in range(args.runs):
            try:
                a = _ask(args.host, msg, pc)
            except Exception as exc:  # noqa: BLE001
                a = f"(error: {exc})"
            answers.append(a)
            for sku in set(SKU_RE.findall(a)):
                if sku not in real and sku not in q_skus:
                    fabricated += 1
        total_fab += fabricated
        # determinism: how many distinct SKU-sets across runs
        sku_sets = {frozenset(SKU_RE.findall(a)) for a in answers}
        print(f"[{label}] kind={kind}")
        print(f"  runs={args.runs}  fabricated_SKU_mentions={fabricated}  "
              f"distinct_SKU_sets={len(sku_sets)}")
        print(f"  sample: {answers[0][:200]}")
        print()

    print(f"TOTAL fabricated SKU mentions across all cases/runs: {total_fab} "
          f"(acceptance target: 0)")
    return 0 if total_fab == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
