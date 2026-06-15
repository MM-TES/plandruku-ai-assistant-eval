"""Retrieval critic (INC-6): is the retrieved context sufficient to answer?

Deterministic and free. A code-matched datasheet (metadata scope pins the whole
product) is sufficient by construction. Otherwise sufficiency keys off whether any
knowledge came back above the gate score. On 'insufficient' the caller runs ONE
widened second pass (bigger top_k, relevance veto skipped) within budget.
"""
from __future__ import annotations

from src.assistant import config
from src.assistant.agents.contracts import PlannerOut, RetrievalJudgment


def assess(knowledge: str, best_score: float, plan: PlannerOut) -> RetrievalJudgment:
    if not config.feature("multi_agent") or not config.agents_param("retrieval_critic", {}).get("enabled", True):
        return RetrievalJudgment(sufficient=True, reason="off")
    if not (knowledge or "").strip():
        return RetrievalJudgment(sufficient=False, reason="empty")
    if plan.products:  # a specific datasheet was named → its chunks are pinned
        return RetrievalJudgment(sufficient=True, reason="product-matched")
    min_score = float(config.agents_param("retrieval_critic", {}).get("sufficient_score", 0.55))
    if float(best_score) >= min_score:
        return RetrievalJudgment(sufficient=True, reason=f"score>={min_score}")
    return RetrievalJudgment(sufficient=False, reason=f"weak score {float(best_score):.2f}")


__all__ = ["assess"]
