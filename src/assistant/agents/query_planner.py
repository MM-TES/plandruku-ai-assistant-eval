"""Query planner (INC-6): attribute expansion + comparison decomposition.

Cheap and best-effort. The deterministic part (product codes, comparison detection)
needs no LLM; one optional Haiku call expands a spec query into an English attribute
search string (товщина→thickness, бар'єр→OTR…) to strengthen dense/BM25 reach for
code-less / semantic queries. Returns extra retrieval variants — never raises.
"""
from __future__ import annotations

import re

from src.assistant import config
from src.assistant.agents.contracts import PlannerOut

_CODE_RE = re.compile(r"\b[A-Z][A-Z0-9]{2,}\b")
_COMPARE_RE = re.compile(r"порівн|різниц|кращ|найкращ|найниж|найвищ|найбіль|найменш|vs\b|compare|better|lowest|highest", re.IGNORECASE)


def plan(message: str, *, usage=None) -> PlannerOut:
    """Build a retrieval plan for *message*. No-op (empty) when the planner is off."""
    if not config.feature("multi_agent") or not config.agents_param("query_planner", {}).get("enabled", True):
        return PlannerOut(variants=[], products=[], comparison=False)
    msg = message or ""
    products = sorted(set(_CODE_RE.findall(msg)))
    comparison = len(products) >= 2 or bool(_COMPARE_RE.search(msg))
    variants: list[str] = [f"{p} technical specifications" for p in products]
    # One cheap attribute-expansion call (helps code-less / semantic queries reach
    # the numeric rows). Skipped silently on any error → degrades to deterministic part.
    try:
        from src.assistant.llm import call_llm

        resp = call_llm(
            agent_name="query_planner", role_key="router",
            messages=[
                {"role": "system", "content": config.prompt("agent_query_planner")},
                {"role": "user", "content": msg},
            ],
            usage=usage, temperature=0.0, max_tokens=80,
        )
        extra = (resp.choices[0].message.content or "").strip()
        if extra and extra.lower() not in (v.lower() for v in variants):
            variants.append(extra)
    except Exception:  # noqa: BLE001 — planner is a best-effort booster
        pass
    return PlannerOut(variants=variants, products=products, comparison=comparison)


__all__ = ["plan"]
