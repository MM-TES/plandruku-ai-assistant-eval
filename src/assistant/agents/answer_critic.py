"""Deterministic answer verifier for KB datasheet answers (free, no LLM).

Two checks: (1) no-invention — every number in the answer must appear in the
evidence; (2) for a full-spec query, the answer must cover >= coverage_min of the
evidence's numbers (a terse 2–5 sentence summary fails this and triggers one
bounded re-synthesis with the spec-complete prompt). The number matcher matches
plain integers/decimals whole (so ``2200`` is not split), comma→dot normalized.
"""
from __future__ import annotations

import re

from src.assistant import config
from src.assistant.agents.contracts import Critique

_NUM = re.compile(r"\d+(?:[.,]\d+)?")


def _nums(text: str) -> set[str]:
    return {m.replace(",", ".") for m in _NUM.findall(text or "")}


def assess(draft: str, evidence: str, *, full_spec: bool) -> Critique:
    """Verdict on a drafted answer against its grounding evidence."""
    ctx = _nums(evidence)
    ans = _nums(draft)
    invented = sorted(ans - ctx) if ctx else []
    coverage = (len(ans & ctx) / len(ctx)) if (full_spec and ctx) else 1.0
    min_cov = float(config.agents_param("answer_critic", {}).get("coverage_min", 0.6))
    problems: list[str] = []
    if invented:
        problems.append("invented_numbers")
    if full_spec and coverage < min_cov:
        problems.append("incomplete")
    return Critique(
        ok=not problems, coverage=round(coverage, 3), invented=invented,
        missing=sorted(ctx - ans) if full_spec else [], problems=",".join(problems),
    )


def better(a: Critique, b: Critique) -> bool:
    """True if verdict *a* is better than *b*: fewer invented numbers wins, then
    higher coverage. Used to keep the stronger of draft vs revision (never worse)."""
    if len(a.invented) != len(b.invented):
        return len(a.invented) < len(b.invented)
    return a.coverage > b.coverage


__all__ = ["assess", "better"]
