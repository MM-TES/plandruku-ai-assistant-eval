"""Per-query stage selection — the 'branched, variable' element of the answer layer.

Cheap (regex, no LLM). Multi-agent stages engage ONLY for KB-grounded datasheet
answers; data/instructions-only queries stay on the single-shot path so latency and
cost are unchanged for the common case.
"""
from __future__ import annotations

import re

from src.assistant import config
from src.assistant.agents.contracts import StagePlan

# "give me ALL specs / full characteristics / parameters / datasheet" cues (uk + en).
_FULL_SPEC_RE = re.compile(
    r"усі?\s+(тех\w*\s+)?характеристик|вс[іе]\s+характеристик|"
    r"повн\w*\s+(перелік|характеристик|специфікац|параметр)|"
    r"характеристик|параметр(и|ів|ах)?|специфікац|тех\w*\s+дан|"
    r"datasheet|data\s*sheet|specif|full\s+spec|all\s+spec|tech\w*\s+data",
    re.IGNORECASE,
)
# A product code in the query (UPPERCASE alnum, >=3) — distinctive of a datasheet ask.
_CODE_RE = re.compile(r"\b[A-Z][A-Z0-9]{2,}\b")
# A scientific / process / normative printing question (NOT a datasheet spec dump) —
# the cue for the structured, cited, cross-source science answer mode (T1.6/T2.3).
_SCI_RE = re.compile(
    r"анілокс|anilox|ракел|doctor.?blade|розтиск|dot gain|муар|каламут|haze|"
    r"ламінув|ламінац|тунелюв|бар.?єр|міграц|розчинник|10/2011|first 5|ffta|"
    r"corona|корон|dyne|поверхнев|глибок\w* друк|флексодрук|гравюр|gravure|"
    r"чому|поясни|порівня|вибра|підбер|який .* кращ|trade.?off",
    re.IGNORECASE,
)


def classify(message: str, *, kb_used: bool) -> StagePlan:
    """Decide the answer-layer plan. A no-op (all-False) plan when multi_agent is
    off or the answer isn't KB-grounded, so the caller falls back to single-shot."""
    if not config.feature("multi_agent") or not kb_used:
        return StagePlan(query_class="general", full_spec=False,
                         run_answer_critic=False, reason="off/no-kb")
    msg = message or ""
    codes = _CODE_RE.findall(msg)
    has_code = bool(codes)
    multi_code = len(set(codes)) >= 2          # a comparison across products
    wants_all = bool(_FULL_SPEC_RE.search(msg))
    # Full-spec = an explicit "all characteristics" ask about a specific product code,
    # OR a comparison naming >=2 codes (the operator wants both products' values).
    # A bare single code ("яка товщина FXCMT") stays specific — don't dump the table.
    full_spec = (wants_all and has_code) or multi_code
    # T1.6/T2.3: a NON-datasheet science/process/normative question → the structured,
    # cited, cross-source answer mode (outline-then-fill). Gated by agents.controller.sci_full.
    sci_full = (not full_spec and not has_code
                and bool(config.agents_param("controller", {}).get("sci_full", False))
                and bool(_SCI_RE.search(msg)))
    run_critic = bool(config.agents_param("answer_critic", {}).get("enabled", True))
    # Block 5: extended-thinking fires only for COMPLEX KB queries (full-spec dump,
    # multi-product comparison, or a science/process question) and only when the flag is on.
    reasoning = (bool(config.agents_param("controller", {}).get("reasoning", False))
                 and (full_spec or multi_code or sci_full or bool(_SCI_RE.search(msg))))
    return StagePlan(
        query_class=("full_spec" if full_spec else "sci" if sci_full
                     else "specific" if has_code else "general"),
        full_spec=full_spec,
        run_answer_critic=run_critic,
        reason=f"code={has_code} multi={multi_code} all={wants_all} sci={sci_full} think={reasoning}",
        sci_full=sci_full,
        reasoning=reasoning,
    )
