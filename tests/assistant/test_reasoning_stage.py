"""Block 5 — reasoning / extended-thinking stage.

Offline: the controller flags COMPLEX KB queries for reasoning only when the flag is
on; llm.call_llm merges a per-call extra_body over the model's config extra_body; and
_synthesize forwards the reasoning extra_body only when reasoning=True.
"""
from __future__ import annotations

import src.assistant.config as cfg
from src.assistant import llm, orchestrator
from src.assistant.agents import controller
from src.assistant.llm import LLMUsage


def _patch_controller(monkeypatch, *, reasoning: bool) -> None:
    real_feature, real_ap = cfg.feature, cfg.agents_param
    monkeypatch.setattr(cfg, "feature", lambda n: True if n == "multi_agent" else real_feature(n))

    def ap(name, default=None):
        if name == "controller":
            return {"reasoning": reasoning, "sci_full": False, "max_extra_llm_calls": 3, "max_added_ms": 8000}
        if name == "answer_critic":
            return {"enabled": True}
        return real_ap(name, default)

    monkeypatch.setattr(cfg, "agents_param", ap)


def test_controller_flags_complex_only_when_enabled(monkeypatch):
    _patch_controller(monkeypatch, reasoning=True)
    # full-spec datasheet ask (code + "характеристики") → complex → reasoning on
    assert controller.classify("усі характеристики FXCMT", kb_used=True).reasoning is True
    # multi-code comparison → complex
    assert controller.classify("порівняй FXCMT та FXC", kb_used=True).reasoning is True
    # science/process question → complex
    assert controller.classify("чому виникає тунелювання при ламінуванні?", kb_used=True).reasoning is True
    # a simple lookup → NOT complex → no reasoning even with the flag on
    assert controller.classify("що це за сторінка?", kb_used=True).reasoning is False


def test_controller_reasoning_off_by_default(monkeypatch):
    _patch_controller(monkeypatch, reasoning=False)
    assert controller.classify("усі характеристики FXCMT", kb_used=True).reasoning is False
    assert controller.classify("порівняй FXCMT та FXC", kb_used=True).reasoning is False


def test_call_llm_merges_per_call_extra_body(monkeypatch, patch_llm_client):
    client = patch_llm_client(["ok"])
    llm.call_llm(agent_name="t", role_key="answer", messages=[{"role": "user", "content": "hi"}],
                 usage=LLMUsage(), extra_body={"reasoning": {"effort": "high"}})
    assert client.calls[-1].get("extra_body") == {"reasoning": {"effort": "high"}}


def test_call_llm_no_extra_body_by_default(monkeypatch, patch_llm_client):
    # The default answer model has no config extra_body → no extra_body key at all.
    client = patch_llm_client(["ok"])
    llm.call_llm(agent_name="t", role_key="answer", messages=[{"role": "user", "content": "hi"}],
                 usage=LLMUsage())
    assert "extra_body" not in client.calls[-1]


def test_synthesize_forwards_reasoning(monkeypatch, patch_llm_client):
    client = patch_llm_client(["відповідь з міркуванням"])
    orchestrator._synthesize("ctx", "складне питання", "instructions", LLMUsage(),
                             system_key="answer_sci_full", reasoning=True)
    # forwards the config-driven reasoning payload (effort vs max_tokens form is config's choice)
    assert client.calls[-1].get("extra_body") == orchestrator._reasoning_extra_body()


def test_synthesize_no_reasoning_by_default(monkeypatch, patch_llm_client):
    client = patch_llm_client(["звичайна відповідь"])
    orchestrator._synthesize("ctx", "просте питання", "instructions", LLMUsage(), system_key="answer")
    assert "extra_body" not in client.calls[-1]
