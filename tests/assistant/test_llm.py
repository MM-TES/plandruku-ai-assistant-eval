"""Phase 1 gate: OpenRouter client wrapper — cost accounting + fallback."""
from __future__ import annotations

from src.assistant import config
from src.assistant.llm import LLMUsage, call_llm

from tests.assistant.conftest import FakeCompletion, FakeUsage


def test_call_llm_tracks_usage(patch_llm_client) -> None:
    patch_llm_client([FakeCompletion("hello", usage=FakeUsage(11, 22))])
    usage = LLMUsage()
    resp = call_llm(
        agent_name="router",
        role_key="router",
        messages=[{"role": "user", "content": "hi"}],
        usage=usage,
    )
    assert resp.choices[0].message.content == "hello"
    assert usage.input_tokens == 11
    assert usage.output_tokens == 22
    assert usage.calls == 1
    assert usage.by_agent["router"]["calls"] == 1
    assert usage.cost_usd > 0


def test_call_llm_falls_back_on_error(patch_llm_client) -> None:
    client = patch_llm_client(
        [RuntimeError("model unavailable"), FakeCompletion("ok", usage=FakeUsage(5, 7))]
    )
    usage = LLMUsage()
    resp = call_llm(
        agent_name="router",
        role_key="router",
        messages=[{"role": "user", "content": "hi"}],
        usage=usage,
    )
    assert resp.choices[0].message.content == "ok"
    # the second call must have switched to the configured fallback model
    primary = config.model_for("router")
    fallback = config.fallback_for(primary)
    assert len(client.calls) == 2
    assert client.calls[0]["model"] == primary
    assert client.calls[1]["model"] == fallback
    assert usage.calls == 1  # only the successful call is counted


def test_estimate_cost_uses_config_pricing() -> None:
    # claude-3.5-haiku priced at (0.80, 4.00) per 1M
    cost = config.estimate_cost("anthropic/claude-3.5-haiku", 1_000_000, 0)
    assert abs(cost - 0.80) < 1e-9
