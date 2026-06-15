"""Synthetic operator-query generation + golden-set loading.

``generate_seed`` produces deterministic page×capability queries (no LLM — used
by the offline gate). ``expand_with_llm`` is an optional operator-run step that
paraphrases seeds via the answer model to grow coverage.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.assistant.llm import LLMUsage, call_llm
from src.assistant.tracing import parse_json_object

_ROOT = Path(__file__).resolve().parents[3]
GOLDEN_PATH = _ROOT / "config" / "assistant_golden.jsonl"

_PAGES = [
    "/", "/orders", "/kpi", "/schedule",
    "/workflow:prodazhi", "/workflow:materialy",
    "/workflow:vyrobnytstvo", "/workflow:zabezpechennia",
]
_TEMPLATES = {
    "instructions": ["що мені тут робити?", "як це працює на цій сторінці?", "для чого ця вкладка?"],
    "data_query": ["скільки тут позицій?", "покажи перелік", "які замовлення тут?"],
    "history": ["що змінилось за добу?", "що нового після оновлення даних?"],
}


def generate_seed() -> list[dict[str, Any]]:
    """Deterministic page×capability queries (no LLM)."""
    items: list[dict[str, Any]] = []
    for page in _PAGES:
        for route, templates in _TEMPLATES.items():
            for q in templates:
                items.append({
                    "query": q, "expected_route": route, "page": page,
                    "safety_class": "safe", "source": "synth",
                })
    return items


def load_golden(path: Path | str = GOLDEN_PATH) -> list[dict[str, Any]]:
    """Load the curated golden set (jsonl)."""
    path = Path(path)
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def expand_with_llm(base_query: str, n: int = 3, *, usage: LLMUsage | None = None) -> list[str]:
    """Operator-run: paraphrase a query into *n* realistic variants."""
    resp = call_llm(
        agent_name="synth",
        role_key="answer",
        messages=[{
            "role": "user",
            "content": (
                f"Згенеруй {n} різних реалістичних перефразувань запиту оператора друкарні "
                f"українською. Запит: «{base_query}». Поверни ЧИСТИЙ JSON: "
                '{"variants": ["...", "..."]}'
            ),
        }],
        usage=usage,
        temperature=0.7,
        max_tokens=400,
        response_format={"type": "json_object"},
    )
    return parse_json_object(resp.choices[0].message.content or "{}").get("variants", [])


__all__ = ["generate_seed", "load_golden", "expand_with_llm", "GOLDEN_PATH"]
