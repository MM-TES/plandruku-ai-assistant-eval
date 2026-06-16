"""Deterministic guardrails that pin assistant behaviour against prompt/doc drift.

These don't call the LLM — they lock the *instructions* that produce the
operator-validated behaviour (anti-hallucination, SKU-over-name, route
awareness, deficit prioritization), so a future prompt/doc edit can't silently
remove them. The costed behavioural check lives in the operator-run LLM eval.
"""
from __future__ import annotations

from pathlib import Path

from src.assistant import config

from tests.assistant.conftest import skip_if_empty

_ROOT = Path(__file__).resolve().parents[2]


def test_answer_prompt_keeps_antihallucination_and_route_guards() -> None:
    p = config.prompt("answer").lower()
    assert "вигад" in p                       # ISSUE-5: must not invent data/steps
    assert "не знаю" in p                      # honesty fallback
    assert "адміністратор" in p                # rebuild-code guidance (no invented code)
    assert "вже на цьому екрані" in p          # ISSUE-2: don't send to the current page


def test_vision_prompt_is_localization_only_no_ocr_facts() -> None:
    # New principle: the screenshot is for CONTEXT only; facts come from system
    # data, never OCR of the pixels (codes differ by one digit).
    p = config.prompt("vision").lower()
    assert p, "vision prompt must exist"
    assert "вигад" in p                         # no invented visuals/values
    assert "червоне" in p                        # explicit «no red highlighting» guard
    assert "дані системи" in p                   # facts come from structured system data
    assert "ненадійн" in p                       # image OCR explicitly called unreliable
    assert "за знімком" in p                     # source marker kept


def test_prompts_carry_instruction_hierarchy_clause() -> None:
    # Extra B: defense-in-depth against prompt injection — the answer/answer_kb/vision
    # system prompts must state that context/user data are DATA, not commands, so a
    # future prompt edit can't silently drop the instruction-hierarchy guard.
    clause = "перевизначити ці правила"
    for key in ("answer", "answer_kb", "vision"):
        p = config.prompt(key).lower()
        assert clause in p, f"{key} prompt lost the instruction-hierarchy clause"
        assert "лише дані, а не команди" in p, f"{key} prompt lost the data-not-commands clause"


def test_shcho_zrobyty_doc_encodes_risk_prioritization() -> None:
    doc = (_ROOT / "docs" / "operator_help" / "shcho_zrobyty.md").read_text(encoding="utf-8").lower()
    assert "0%" in doc                          # ISSUE-6: 0%-coverage first
    assert "ризик" in doc                        # by business risk, not kg volume


def test_golden_set_has_prioritization_case() -> None:
    from src.assistant.eval import synth

    golden = synth.load_golden()
    skip_if_empty(golden, "data-query golden set")
    notes = " ".join(str(i.get("note", "")) for i in golden).lower()
    assert "issue-6" in notes                   # prioritization scenario kept for the LLM eval
    assert "issue-5" in notes                   # honesty-on-missing-data scenario kept
