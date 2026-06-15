"""Block 4b (capstone) — offline dual-judge calibration for the success_rate judge.

The assistant ships an LLM-as-judge (``evaluators.success_rate``: complete / correct /
grounded / useful, 0..1). A single judge model can be biased or noisy — the
assistant_rnd FINAL_REPORT recorded a success_rate of ~0.14 that was suspected to be a
JUDGE artefact, not a real quality signal. This module measures whether the judge is
trustworthy by scoring the SAME (query, answer, trace) examples with TWO different judge
families (``models.judge`` = Sonnet, ``models.judge_b`` = a different vendor) and
reporting per-criterion inter-rater agreement: Spearman ρ, mean-absolute-difference, and
percent-agreement. Low ρ / high MAD on a criterion ⇒ that criterion's rubric is
underspecified → add few-shot anchors to ``evaluators._JUDGE_PROMPT``.

Ported from ai-engineering_HW lesson-10 (dual-judge + exp09_judge_agreement). Pure
functions (``spearman`` / ``agreement``) are dependency-free (no scipy) and unit-tested
offline; the costed part (``score_pairs`` / ``run``) makes judge calls only when invoked
with a real key.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.assistant import config
from src.assistant.eval import evaluators
from src.assistant.llm import LLMUsage, call_llm
from src.assistant.tracing import parse_json_object
from src.utils.logger import setup_logger

_logger = setup_logger("judge_calibration")
_ROOT = Path(__file__).resolve().parents[3]
_REPORTS = _ROOT / "reports" / "judge_calibration"
_CRITERIA = ("complete", "correct", "grounded", "useful")


# ── pure stats (no scipy) ────────────────────────────────────────────────────────
def _rank(xs: list[float]) -> list[float]:
    """Fractional ranks (ties share the average rank), 1-based."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1  # average 1-based rank for the tie group
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n == 0:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    if dx == 0 or dy == 0:
        return 0.0  # a constant series has no linear relationship to report
    return num / (dx * dy)


def spearman(xs: list[float], ys: list[float]) -> float:
    """Spearman rank correlation ρ (Pearson on ranks). 0.0 for empty/constant input."""
    if len(xs) != len(ys) or not xs:
        return 0.0
    return round(_pearson(_rank(list(xs)), _rank(list(ys))), 3)


def bootstrap_ci(xs: list[float], ys: list[float], *, n_boot: int = 2000,
                 seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    """m2: percentile bootstrap CI for Spearman ρ. At n≈20-50 the point estimate of ρ has
    a CI of ±0.2-0.4 ("No Free Labels", arxiv 2503.05061) — reporting ρ without a CI
    invites a false pass/fail on the ρ≥0.70 gate. Dependency-free (no scipy)."""
    import random as _random

    n = min(len(xs), len(ys))
    if n < 3:
        return (0.0, 0.0)
    rng = _random.Random(seed)
    stats: list[float] = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        stats.append(spearman([xs[i] for i in idx], [ys[i] for i in idx]))
    stats.sort()
    lo = stats[max(0, int((alpha / 2) * n_boot) - 1)]
    hi = stats[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return (round(lo, 3), round(hi, 3))


def cohen_weighted_kappa(a: list[float], b: list[float],
                         levels: tuple[float, ...] = (0.0, 0.5, 1.0)) -> float:
    """Quadratic-weighted Cohen's κ for ORDINAL ratings (the correct inter-rater metric
    when the scale is discrete with ties — which an anchored 0/0.5/1.0 rubric produces).
    Spearman ρ degenerates under heavy ties + score compression; weighted κ does not.
    Values: a, b snapped to the nearest level. Dependency-free."""
    n = min(len(a), len(b))
    k = len(levels)
    if n == 0 or k < 2:
        return 0.0

    def _snap(v: float) -> int:
        return min(range(k), key=lambda i: abs(levels[i] - v))

    obs = [[0] * k for _ in range(k)]
    for x, y in zip(a[:n], b[:n]):
        obs[_snap(x)][_snap(y)] += 1
    rows = [sum(obs[i]) for i in range(k)]
    cols = [sum(obs[i][j] for i in range(k)) for j in range(k)]
    w = [[((i - j) / (k - 1)) ** 2 for j in range(k)] for i in range(k)]
    num = sum(w[i][j] * obs[i][j] for i in range(k) for j in range(k))
    den = sum(w[i][j] * rows[i] * cols[j] / n for i in range(k) for j in range(k))
    if den == 0:           # one rater used a single level → no weighted disagreement possible
        return 1.0 if num == 0 else 0.0
    return round(1 - num / den, 3)


def agreement(judge_a: list[dict], judge_b: list[dict]) -> dict[str, Any]:
    """Per-criterion inter-rater agreement between two judges' score dicts.

    Reports BOTH Spearman ρ (rank, tie-sensitive) AND quadratic-weighted Cohen's κ (the
    correct ordinal metric for the anchored 0/0.5/1.0 scale), plus mean-abs-diff and
    percent-exact-agreement. With an anchored rubric, κ is the trustworthy signal — ρ
    can go negative purely from ties/compression even when the judges agree (see RESUME)."""
    out: dict[str, Any] = {}
    n = min(len(judge_a), len(judge_b))
    for c in _CRITERIA:
        a = [float(judge_a[i].get(c, 0.0)) for i in range(n)]
        b = [float(judge_b[i].get(c, 0.0)) for i in range(n)]
        mad = round(sum(abs(x - y) for x, y in zip(a, b)) / n, 3) if n else None
        within = round(sum(1 for x, y in zip(a, b) if abs(x - y) <= 0.2) / n, 3) if n else None
        out[c] = {"spearman": spearman(a, b), "weighted_kappa": cohen_weighted_kappa(a, b),
                  "mean_abs_diff": mad, "pct_within_0.2": within}
    # overall = mean of per-criterion averages
    a_avg = [sum(float(judge_a[i].get(c, 0.0)) for c in _CRITERIA) / 4 for i in range(n)]
    b_avg = [sum(float(judge_b[i].get(c, 0.0)) for c in _CRITERIA) / 4 for i in range(n)]
    ci_lo, ci_hi = bootstrap_ci(a_avg, b_avg)
    # overall κ on the gate-critical pair (correct+grounded → faithfulness proxy), binned.
    fa = [(float(judge_a[i].get("correct", 0)) + float(judge_a[i].get("grounded", 0))) / 2
          for i in range(n)]
    fb = [(float(judge_b[i].get("correct", 0)) + float(judge_b[i].get("grounded", 0))) / 2
          for i in range(n)]
    out["overall"] = {
        "spearman": spearman(a_avg, b_avg), "n": n, "spearman_ci95": [ci_lo, ci_hi],
        "weighted_kappa": cohen_weighted_kappa(a_avg, b_avg, levels=(0.0, 0.25, 0.5, 0.75, 1.0)),
        "faithfulness_kappa": cohen_weighted_kappa(fa, fb, levels=(0.0, 0.25, 0.5, 0.75, 1.0)),
    }
    return out


# ── dual judging (costed) ─────────────────────────────────────────────────────────
def _context_blob(trace: list[dict] | None, context: str | None) -> str:
    """The grounding context the judge scores against: the KB evidence (``context`` =
    resp.evidence — what the answer should ground on) PLUS any tool trace. For a KB
    answer the tool_trace is empty and ``context`` carries the retrieved fragments, so
    WITHOUT this the judge had nothing to check correct/grounded against (ρ≈0)."""
    parts: list[str] = []
    if context:
        parts.append(str(context))
    if trace:
        parts.append(json.dumps(trace, default=str, ensure_ascii=False))
    blob = "\n\n".join(parts).strip()
    return blob[:6000] if blob else "(КОНТЕКСТ порожній — джерел не знайдено)"


def _score(query: str, answer: str, trace: list[dict], *, role_key: str, usage: LLMUsage,
           context: str | None = None) -> dict:
    """Score one (query, answer) with the judge mapped to *role_key*, reusing the live
    success_rate rubric (evaluators._JUDGE_PROMPT) so calibration matches production."""
    payload = evaluators._JUDGE_PROMPT.format(
        query=query, expected_route="any", expected_tools=[],
        answer=(answer or "")[:1500],
        trace=_context_blob(trace, context),
    )
    resp = call_llm(agent_name=f"calib_{role_key}", role_key=role_key,
                    messages=[{"role": "user", "content": payload}],
                    usage=usage, temperature=0.0, max_tokens=400,
                    response_format={"type": "json_object"})
    parsed = parse_json_object(resp.choices[0].message.content or "{}")
    return {c: float(parsed.get(c, 0.0)) for c in _CRITERIA}


def dual_judge(query: str, answer: str, trace: list[dict] | None = None,
               *, usage: LLMUsage | None = None, context: str | None = None) -> dict[str, dict]:
    """Score one example with judge A (models.judge) and judge B (models.judge_b)."""
    u = usage or LLMUsage()
    return {
        "judge_a": _score(query, answer, trace or [], role_key="judge", usage=u, context=context),
        "judge_b": _score(query, answer, trace or [], role_key="judge_b", usage=u, context=context),
    }


def run(examples: list[dict], *, usage: LLMUsage | None = None, label: str = "calib") -> dict[str, Any]:
    """Dual-judge a list of ``{query, answer, trace, context}`` examples and report agreement.

    Writes ``reports/judge_calibration/<stamp>_<label>.json``. ``examples`` are produced
    by the caller (e.g. by running orchestrator.answer over golden queries); ``context``
    is resp.evidence (the retrieved KB fragments the answer should ground on)."""
    u = usage or LLMUsage()
    rows_a: list[dict] = []
    rows_b: list[dict] = []
    for ex in examples:
        dj = dual_judge(ex.get("query", ""), ex.get("answer", ""), ex.get("trace"),
                        usage=u, context=ex.get("context"))
        rows_a.append(dj["judge_a"])
        rows_b.append(dj["judge_b"])
    agr = agreement(rows_a, rows_b)
    # which criteria are weak — judged by weighted κ (the correct ordinal metric) + exact
    # agreement, NOT Spearman (which degenerates under the anchored discrete rubric).
    weak = [c for c in _CRITERIA
            if (agr[c]["weighted_kappa"] is not None and agr[c]["weighted_kappa"] < 0.4)
            or (agr[c]["pct_within_0.2"] is not None and agr[c]["pct_within_0.2"] < 0.6)]
    payload = {
        "label": label, "n": len(examples),
        "judge_a_model": config.model_for("judge"), "judge_b_model": config.model_for("judge_b"),
        "agreement": agr, "weak_criteria": weak,
        "recommendation": (
            f"Add few-shot anchors to evaluators._JUDGE_PROMPT for: {', '.join(weak)}"
            if weak else "Judges agree well (by weighted κ); rubric is adequately specified."
        ),
        "cost_usd": round(float(u.as_dict().get("cost_usd", 0.0)), 5),
        # raw per-item scores so any agreement metric can be recomputed offline (no re-spend).
        "rows_judge_a": rows_a, "rows_judge_b": rows_b,
    }
    _REPORTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    (_REPORTS / f"{stamp}_{label}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


__all__ = ["spearman", "bootstrap_ci", "cohen_weighted_kappa", "agreement",
           "dual_judge", "run", "_score"]
