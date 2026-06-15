"""Block 4a — live grounding gate (bounded refine + budget + claim_check gating).

Offline: the gate refines once when the judge finds unsupported claims and keeps the
better draft; it skips the refine when the controller budget is exhausted; and it is a
no-op (zero LLM calls) when claim_check is off.
"""
from __future__ import annotations

import src.assistant.config as cfg
from src.assistant import orchestrator
from src.assistant.llm import LLMUsage


def _set_claim_check(monkeypatch, on: bool) -> None:
    real_ap = cfg.agents_param
    monkeypatch.setattr(cfg, "agents_param",
                        lambda n, d=None: {"claim_check": on} if n == "answer_critic" else real_ap(n, d))


def test_refines_when_judge_finds_unsupported(monkeypatch, patch_llm_client):
    _set_claim_check(monkeypatch, True)
    client = patch_llm_client([
        '{"unsupported": ["твердження X"]}',   # judge 1: draft has 1 unsupported claim
        "уточнена відповідь",                   # refine synthesis
        '{"unsupported": []}',                  # judge 2: refined draft is clean
    ])
    out = orchestrator._claim_grounded("чернетка", "контекст", "питання", "instructions",
                                       LLMUsage(), refine_key="answer_kb", budget_ok=lambda: True)
    assert out == "уточнена відповідь"
    assert len(client.calls) == 3


def test_keeps_draft_when_refine_not_better(monkeypatch, patch_llm_client):
    _set_claim_check(monkeypatch, True)
    client = patch_llm_client([
        '{"unsupported": ["a", "b"]}',          # judge 1: 2 unsupported
        "гірша відповідь",                       # refine
        '{"unsupported": ["a", "b", "c"]}',     # judge 2: worse (3) → keep original
    ])
    out = orchestrator._claim_grounded("чернетка", "контекст", "питання", "instructions",
                                       LLMUsage(), refine_key="answer_kb", budget_ok=lambda: True)
    assert out == "чернетка"


def test_budget_exhausted_skips_refine(monkeypatch, patch_llm_client):
    _set_claim_check(monkeypatch, True)
    client = patch_llm_client(['{"unsupported": ["твердження X"]}'])  # only the judge runs
    out = orchestrator._claim_grounded("чернетка", "контекст", "питання", "instructions",
                                       LLMUsage(), refine_key="answer_kb", budget_ok=lambda: False)
    assert out == "чернетка"
    assert len(client.calls) == 1  # judge only — no refine spent


def test_claim_check_off_is_noop(monkeypatch, patch_llm_client):
    _set_claim_check(monkeypatch, False)
    client = patch_llm_client([])
    out = orchestrator._claim_grounded("чернетка", "контекст", "питання", "instructions",
                                       LLMUsage(), refine_key="answer_kb", budget_ok=lambda: True)
    assert out == "чернетка"
    assert client.calls == []  # grounding.unsupported_claims returns [] without any LLM call
