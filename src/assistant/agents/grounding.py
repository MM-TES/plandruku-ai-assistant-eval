"""Claim-level grounding gate (SA T1.4) for science KB answers.

The deterministic answer_critic only checks NUMBERS. A prose science answer can be
fluent yet contain claims taken from the model's parametric memory (Correctness ≠
Faithfulness — up to 57% of citations can be post-rationalized). This stage asks a
judge model to decompose the answer into atomic technical claims and list the ones
the retrieved evidence does NOT support, so the caller can refine or hedge.

Gated by ``agents.answer_critic.claim_check`` (default OFF) and best-effort: returns
``[]`` (treat as grounded) when disabled, on empty input, or on any judge error, so a
judge outage never blocks an answer. Makes one cheap judge call when enabled.
"""
from __future__ import annotations

from src.assistant import config
from src.assistant.llm import LLMUsage, call_llm
from src.assistant.tracing import parse_json_object
from src.utils.logger import setup_logger

_logger = setup_logger(__name__)


def enabled() -> bool:
    return bool(config.agents_param("answer_critic", {}).get("claim_check", False))


def unsupported_claims(answer: str, evidence: str, *, usage: LLMUsage | None = None) -> list[str]:
    """Atomic technical claims in *answer* that the *evidence* does NOT support.
    Empty when disabled / empty / on error (fail-open)."""
    if not enabled() or not (answer or "").strip() or not (evidence or "").strip():
        return []
    try:
        user = f"[КОНТЕКСТ]\n{evidence[:6000]}\n\n[ВІДПОВІДЬ]\n{answer[:2000]}"
        resp = call_llm(
            agent_name="grounding", role_key="judge",
            messages=[
                {"role": "system", "content": config.prompt("grounding_judge")},
                {"role": "user", "content": user},
            ],
            usage=usage, temperature=0.0, max_tokens=400,
            response_format={"type": "json_object"},
        )
        parsed = parse_json_object(resp.choices[0].message.content or "{}")
        ups = parsed.get("unsupported", [])
        return [str(u) for u in ups][:8] if isinstance(ups, list) else []
    except Exception as exc:  # noqa: BLE001 — grounding judge is best-effort (fail-open)
        _logger.info("claim grounding check failed (%s)", str(exc)[:120])
        return []


__all__ = ["enabled", "unsupported_claims"]
