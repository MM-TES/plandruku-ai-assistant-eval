"""Block 4b — dual-judge calibration: pure stats + scripted dual_judge.

Offline: spearman gives ±1 on monotone/anti-monotone series, agreement aggregates two
judges' score dicts, and dual_judge parses two scripted judge JSONs (judge A + judge B).
"""
from __future__ import annotations

from src.assistant.eval import judge_calibration as JC
from src.assistant.llm import LLMUsage


# ── pure stats ───────────────────────────────────────────────────────────────────
def test_spearman_perfect_and_inverse():
    assert JC.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0      # monotone increasing
    assert JC.spearman([1, 2, 3, 4], [40, 30, 20, 10]) == -1.0     # monotone decreasing
    assert JC.spearman([1, 2, 3, 4], [1, 3, 2, 4]) < 1.0           # one swap → < 1
    assert JC.spearman([], []) == 0.0                              # empty
    assert JC.spearman([5, 5, 5], [1, 2, 3]) == 0.0               # constant series


def test_agreement_aggregates_criteria():
    a = [{"complete": 1.0, "correct": 1.0, "grounded": 1.0, "useful": 1.0},
         {"complete": 0.0, "correct": 0.0, "grounded": 0.0, "useful": 0.0}]
    b = [{"complete": 0.9, "correct": 1.0, "grounded": 1.0, "useful": 1.0},
         {"complete": 0.1, "correct": 0.0, "grounded": 0.0, "useful": 0.0}]
    agr = JC.agreement(a, b)
    assert set(agr) == {"complete", "correct", "grounded", "useful", "overall"}
    assert agr["correct"]["spearman"] == 1.0 and agr["correct"]["mean_abs_diff"] == 0.0
    assert agr["complete"]["pct_within_0.2"] == 1.0     # diffs 0.1 each → within 0.2
    assert agr["overall"]["n"] == 2


# ── scripted dual judging (no network) ────────────────────────────────────────────
def test_dual_judge_parses_both_judges(patch_llm_client):
    client = patch_llm_client([
        '{"complete":0.8,"correct":0.9,"grounded":1.0,"useful":0.7}',   # judge A (Sonnet)
        '{"complete":0.7,"correct":0.9,"grounded":0.9,"useful":0.6}',   # judge B (gpt-4o-mini)
    ])
    dj = JC.dual_judge("питання?", "відповідь", [], usage=LLMUsage())
    assert dj["judge_a"]["correct"] == 0.9 and dj["judge_b"]["grounded"] == 0.9
    assert len(client.calls) == 2


def test_run_flags_weak_criteria_and_writes(tmp_path, monkeypatch, patch_llm_client):
    # Two examples; judges DISAGREE wildly on "useful" → it should be flagged weak.
    patch_llm_client([
        '{"complete":1.0,"correct":1.0,"grounded":1.0,"useful":1.0}',   # ex1 judge A
        '{"complete":1.0,"correct":1.0,"grounded":1.0,"useful":0.0}',   # ex1 judge B
        '{"complete":1.0,"correct":1.0,"grounded":1.0,"useful":0.0}',   # ex2 judge A
        '{"complete":1.0,"correct":1.0,"grounded":1.0,"useful":1.0}',   # ex2 judge B
    ])
    monkeypatch.setattr(JC, "_REPORTS", tmp_path)
    out = JC.run([{"query": "q1", "answer": "a1"}, {"query": "q2", "answer": "a2"}],
                 usage=LLMUsage(), label="t")
    assert out["n"] == 2
    assert "useful" in out["weak_criteria"]            # judges flip on useful → low agreement
    assert list(tmp_path.glob("*_t.json"))             # report written
