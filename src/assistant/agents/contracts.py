"""Typed contracts for the multi-agent answer layer."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StagePlan:
    """What the controller decided to run for one request."""
    query_class: str            # "full_spec" | "specific" | "general" | "sci"
    full_spec: bool             # use the spec-complete answer prompt (datasheet)
    run_answer_critic: bool
    reason: str = ""
    sci_full: bool = False      # T1.6/T2.3: structured, cited, cross-source science answer
    reasoning: bool = False     # Block 5: extended-thinking for complex KB queries


@dataclass
class Critique:
    """Deterministic verdict on a drafted KB answer."""
    ok: bool
    coverage: float                                     # answer∩context nums / context nums
    invented: list[str] = field(default_factory=list)  # numbers in answer absent from context
    missing: list[str] = field(default_factory=list)   # context nums absent from a full-spec answer
    problems: str = ""


@dataclass
class PlannerOut:
    """Retrieval plan: extra query variants + detected product codes / comparison."""
    variants: list[str] = field(default_factory=list)
    products: list[str] = field(default_factory=list)
    comparison: bool = False


@dataclass
class RetrievalJudgment:
    """Verdict on whether retrieved context suffices to answer (INC-6)."""
    sufficient: bool
    reason: str = ""
