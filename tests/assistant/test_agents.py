"""INC-5 tests: multi-agent controller + answer_critic (offline, monkeypatched flags)."""
from __future__ import annotations

from src.assistant import config as C
from src.assistant.agents import answer_critic, controller, query_planner, retrieval_critic
from src.assistant.agents.contracts import Critique, PlannerOut


def _force_ma(monkeypatch, on=True, critic=True):
    monkeypatch.setattr(C, "feature", lambda name: on if name == "multi_agent" else False)
    monkeypatch.setattr(C, "agents_param", lambda name, default=None:
                        {"enabled": critic, "coverage_min": 0.6, "max_refine_iters": 1}
                        if name == "answer_critic" else default)


def _critic_cfg(monkeypatch):
    monkeypatch.setattr(C, "agents_param", lambda name, default=None:
                        {"coverage_min": 0.6} if name == "answer_critic" else default)


# ── controller ──────────────────────────────────────────────────────────────────
def test_controller_off_when_flag_off(monkeypatch):
    monkeypatch.setattr(C, "feature", lambda name: False)
    p = controller.classify("усі характеристики FXCMT", kb_used=True)
    assert not p.full_spec and not p.run_answer_critic


def test_controller_off_when_no_kb(monkeypatch):
    _force_ma(monkeypatch)
    p = controller.classify("усі характеристики FXCMT", kb_used=False)
    assert not p.full_spec and not p.run_answer_critic


def test_controller_full_spec(monkeypatch):
    _force_ma(monkeypatch)
    p = controller.classify("дай усі технічні характеристики плівки FXCMT", kb_used=True)
    assert p.full_spec and p.query_class == "full_spec" and p.run_answer_critic


def test_controller_specific_code_is_not_full_spec(monkeypatch):
    _force_ma(monkeypatch)
    p = controller.classify("яка товщина FXCMT", kb_used=True)
    assert not p.full_spec and p.query_class == "specific"


def test_controller_general(monkeypatch):
    _force_ma(monkeypatch)
    p = controller.classify("що таке anilox", kb_used=True)
    assert not p.full_spec and p.query_class == "general"


def test_controller_comparison_two_codes_is_full_spec(monkeypatch):
    _force_ma(monkeypatch)
    p = controller.classify("порівняй FXCMT і FXC", kb_used=True)
    assert p.full_spec and p.query_class == "full_spec"


# ── answer_critic ────────────────────────────────────────────────────────────────
def test_assess_passes_when_grounded(monkeypatch):
    _critic_cfg(monkeypatch)
    c = answer_critic.assess("товщина 20, вихід 56.8", "20 56.8 75.8 37.8", full_spec=False)
    assert c.ok and not c.invented


def test_assess_detects_invented_number(monkeypatch):
    _critic_cfg(monkeypatch)
    c = answer_critic.assess("COF 0.99", "0.27 56.8", full_spec=False)
    assert not c.ok and "0.99" in c.invented and "invented_numbers" in c.problems


def test_assess_full_spec_incomplete_flags(monkeypatch):
    _critic_cfg(monkeypatch)
    ctx = " ".join(str(n) for n in range(100, 120))   # 20 numbers
    c = answer_critic.assess("100 101", ctx, full_spec=True)   # 2/20 = 0.1
    assert not c.ok and "incomplete" in c.problems and c.coverage < 0.6


def test_assess_full_spec_complete_passes(monkeypatch):
    _critic_cfg(monkeypatch)
    ctx = " ".join(str(n) for n in range(100, 110))   # 10
    ans = " ".join(str(n) for n in range(100, 108))   # 8/10 = 0.8
    c = answer_critic.assess(ans, ctx, full_spec=True)
    assert c.ok and c.coverage >= 0.6


def test_assess_matcher_keeps_4digit_whole(monkeypatch):
    _critic_cfg(monkeypatch)
    # answer cites 2200; must count as grounded (not split into 220+0).
    c = answer_critic.assess("OTR 2200", "OTR 2200 1300", full_spec=False)
    assert c.ok and not c.invented


def test_better_prefers_fewer_invented_then_coverage():
    a = Critique(ok=True, coverage=0.8, invented=[])
    b = Critique(ok=False, coverage=0.9, invented=["9.9"])
    assert answer_critic.better(a, b)            # fewer invented wins over coverage
    c = Critique(ok=True, coverage=0.7, invented=[])
    assert answer_critic.better(a, c)            # then higher coverage


# ── INC-6: query_planner ──────────────────────────────────────────────────────
def _no_llm(monkeypatch):
    import src.assistant.llm as llm

    def _boom(**_):
        raise RuntimeError("no network in tests")
    monkeypatch.setattr(llm, "call_llm", _boom)


def test_planner_off_returns_empty(monkeypatch):
    monkeypatch.setattr(C, "feature", lambda name: False)
    p = query_planner.plan("усі характеристики FXCMT")
    assert p.variants == [] and not p.comparison and p.products == []


def test_planner_products_and_comparison(monkeypatch):
    monkeypatch.setattr(C, "feature", lambda name: name == "multi_agent")
    monkeypatch.setattr(C, "agents_param", lambda name, default=None:
                        {"enabled": True} if name == "query_planner" else default)
    _no_llm(monkeypatch)   # force the deterministic fallback (no LLM call)
    p = query_planner.plan("порівняй FXCMT і FXC за бар'єром")
    assert set(p.products) == {"FXCMT", "FXC"} and p.comparison
    assert any("FXCMT" in v for v in p.variants) and any("FXC" in v for v in p.variants)


# ── INC-6: retrieval_critic ─────────────────────────────────────────────────────
def _crit_on(monkeypatch):
    monkeypatch.setattr(C, "feature", lambda name: name == "multi_agent")
    monkeypatch.setattr(C, "agents_param", lambda name, default=None:
                        {"enabled": True, "sufficient_score": 0.55} if name == "retrieval_critic" else default)


def test_retrieval_critic_product_matched_is_sufficient(monkeypatch):
    _crit_on(monkeypatch)
    j = retrieval_critic.assess("some datasheet text", 0.1, PlannerOut(products=["FXCMT"]))
    assert j.sufficient   # pinned product → sufficient even at low dense score


def test_retrieval_critic_weak_vs_strong_score(monkeypatch):
    _crit_on(monkeypatch)
    assert not retrieval_critic.assess("k", 0.30, PlannerOut(products=[])).sufficient
    assert retrieval_critic.assess("k", 0.70, PlannerOut(products=[])).sufficient


def test_retrieval_critic_empty_insufficient(monkeypatch):
    _crit_on(monkeypatch)
    assert not retrieval_critic.assess("", 0.99, PlannerOut(products=[])).sufficient


def test_retrieval_critic_off_is_noop(monkeypatch):
    monkeypatch.setattr(C, "feature", lambda name: False)
    assert retrieval_critic.assess("", 0.0, PlannerOut()).sufficient
