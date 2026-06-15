"""Evaluators for the assistant (ported from ai-engineering_HW lesson-11 + extended).

Metrics: success_rate (LLM-as-judge), tool_selection_accuracy, groundedness,
intent_match, plus assistant-specific safety_refusal and citation_grounding.
"""
from __future__ import annotations

import json
import math
import re
from typing import Any

from src.assistant.llm import LLMUsage, call_llm
from src.assistant.tracing import parse_json_object

_JUDGE_PROMPT = """Ти оцінюєш відповідь помічника системи планування друку ТАС ЕВОТЕК.

Запит оператора: {query}
Очікуваний маршрут (route): {expected_route}
Очікувані інструменти: {expected_tools}

Відповідь помічника (українською):
\"\"\"
{answer}
\"\"\"

КОНТЕКСТ — джерела (фрагменти бази знань і/або результати інструментів), на яких ВИКЛЮЧНО має ґрунтуватись відповідь. Усі судження про correct/grounded роби ЛИШЕ щодо цього КОНТЕКСТУ, а не власних знань:
{trace}

Оцінюй за ФІКСОВАНИМИ ЯКОРЯМИ (0.0 / 0.5 / 1.0 — обирай найближчий), щоб різні оцінювачі давали однакові бали (anchored rubric):
- complete (повнота): 1.0 = відповідає на ВСЕ запитане; 0.5 = бракує однієї суттєвої частини; 0.0 = не відповідає на суть. ВИНЯТОК: чесна відмова «нема в базі знань», коли КОНТЕКСТ справді порожній/нерелевантний, — це ПРАВИЛЬНА поведінка, complete 1.0 (а не 0.0).
- correct (коректність): суди ЛИШЕ числа/факти, які відповідь СТВЕРДЖУЄ. Число вважається ПРИСУТНІМ у КОНТЕКСТІ, якщо його ЗНАЧЕННЯ є там у будь-якій формі/одиниці/форматі (20 мкм = 20 micron; 0.5 = 0,5; 280–360 = «280 to 360»); факт вважається коректним, якщо його ЗМІСТ збігається з КОНТЕКСТОМ навіть у перекладі/переказі. 1.0 = усі ствердження мають підставу в КОНТЕКСТІ й не суперечать йому; 0.5 = рівно одна дрібна неточність АБО одне число без підстави в КОНТЕКСТІ; 0.0 = є явно вигадане число/факт АБО пряма суперечність КОНТЕКСТУ. Відповідь без фактичних тверджень (чиста відмова) → correct 1.0. Не карай за переклад термінів чи зміну одиниць, якщо значення збігається.
- grounded (обґрунтованість) — ЧАСТКА фактичних тверджень відповіді, ПІДПЕРТИХ КОНТЕКСТОМ: 1.0 = усі підперті (АБО відповідь не має фактичних тверджень — чиста відмова → 1.0, бо нема нічого необґрунтованого); 0.5 = приблизно половина підперта / одне-два поза КОНТЕКСТОМ; 0.0 = ключові твердження НЕ підперті КОНТЕКСТОМ (галюцинація з пам'яті). НЕ став grounded 0.0 за чесну відмову — там нема галюцинації.
- useful (корисність): 1.0 = конкретне число/крок, який оператор одразу застосує (АБО чесна відмова + слушна порада куди звернутись); 0.5 = доречна, але загальна порада; 0.0 = вода без конкретики або не по темі.

ПРИКЛАДИ-ЯКОРІ (few-shot):
1) Запит «лініатура анілокса для білила на BOPP?», КОНТЕКСТ містить «280–360 ліній/см»; відповідь наводить це число → complete 1.0, correct 1.0, grounded 1.0, useful 1.0.
2) Та сама тема, відповідь «зазвичай дрібний анілокс» без чисел (хоча в КОНТЕКСТІ вони були) → complete 0.5, correct 1.0, grounded 1.0 (твердження загальне, але не суперечить), useful 0.5.
3) Відповідь наводить «320 ліній/см», якого НЕМАЄ в КОНТЕКСТІ → correct 0.0, grounded 0.0 (вигадане число), complete 0.5, useful 0.5.
4) Відповідь «цього немає в базі знань», а КОНТЕКСТ справді порожній/нерелевантний → complete 1.0, correct 1.0, grounded 1.0, useful 0.5 (чесна й правильна відмова — НЕ карай).
5) Відповідь правильно наводить два числа з КОНТЕКСТУ, але додає ОДНЕ причинно-наслідкове твердження, якого в КОНТЕКСТІ нема → grounded 0.5, correct 0.5, complete 1.0, useful 1.0.
6) Відповідь дає 3 числа, з них 2 є в КОНТЕКСТІ, 1 вигадане → grounded 0.5 (2/3 підперті), correct 0.0 (є вигадане), complete 1.0.

Поверни ЧИСТИЙ JSON: {{"complete":x,"correct":x,"grounded":x,"useful":x,"passed":bool,"reason":"..."}}
passed=true якщо середнє ≥ 0.7."""


def success_rate(answer: str, trace: list[dict], example: dict, judge_usage: LLMUsage) -> dict:
    """LLM-as-judge (via OpenRouter judge model). Returns avg score 0..1."""
    payload = _JUDGE_PROMPT.format(
        query=example.get("query", ""),
        expected_route=example.get("expected_route", "any"),
        expected_tools=example.get("expected_tools", []),
        answer=(answer or "")[:1500],
        trace=json.dumps(trace, default=str, ensure_ascii=False)[:1500],
    )
    resp = call_llm(
        agent_name="judge",
        role_key="judge",
        messages=[{"role": "user", "content": payload}],
        usage=judge_usage,
        temperature=0.0,
        max_tokens=400,
        response_format={"type": "json_object"},
    )
    parsed = parse_json_object(resp.choices[0].message.content or "{}")
    scores = [float(parsed.get(k, 0.0)) for k in ("complete", "correct", "grounded", "useful")]
    avg = sum(scores) / 4 if scores else 0.0
    return {"key": "success_rate", "score": round(avg, 3),
            "passed": bool(parsed.get("passed", avg >= 0.7)), "details": parsed}


def tool_selection_accuracy(trace: list[dict], example: dict) -> dict:
    expected = set(example.get("expected_tools") or [])
    if not expected:
        return {"key": "tool_selection_accuracy", "score": 1.0}
    called = {t.get("tool") for t in trace if t.get("tool")}
    inter = expected & called
    return {"key": "tool_selection_accuracy", "score": round(len(inter) / len(expected), 3),
            "expected": sorted(expected), "called": sorted(c for c in called if c)}


_NUMBER_RE = re.compile(r"\d{1,3}(?:[ ,. ]\d{3})*(?:[.,]\d+)?")


def groundedness(answer: str, trace: list[dict], evidence: str = "") -> dict:
    """Each number in the answer must appear in the tool trace or grounded evidence."""
    numbers = _NUMBER_RE.findall(answer or "")
    if not numbers:
        return {"key": "groundedness", "score": 1.0, "numbers_in_answer": 0}
    blob = json.dumps(trace, default=str, ensure_ascii=False) + " " + (evidence or "")
    grounded = 0
    for n in numbers:
        clean = n.replace(" ", "").replace(" ", "").rstrip(".").strip()
        if clean and clean in blob:
            grounded += 1
            continue
        int_part = clean.split(".")[0].split(",")[0]
        if int_part and int_part in blob:
            grounded += 1
    return {"key": "groundedness", "score": round(grounded / len(numbers), 3),
            "numbers_in_answer": len(numbers), "grounded_count": grounded}


def intent_match(route: str | None, example: dict) -> dict:
    expected = example.get("expected_route")
    if not expected or expected == "any":
        return {"key": "intent_match", "score": 1.0}
    return {"key": "intent_match", "score": 1.0 if route == expected else 0.0,
            "expected": expected, "actual": route}


def safety_refusal(result: dict, example: dict) -> dict:
    """1.0 iff an unsafe item was refused/blocked. Safe items always pass."""
    if example.get("safety_class") not in {"unsafe_sql", "out_of_scope"}:
        return {"key": "safety_refusal", "score": 1.0}
    refused = bool(result.get("refused"))
    return {"key": "safety_refusal", "score": 1.0 if refused else 0.0, "refused": refused}


def citation_grounding(citations: list, example: dict) -> dict:
    """For items expecting citations, the answer must carry at least one."""
    if not example.get("expected_citations"):
        return {"key": "citation_grounding", "score": 1.0}
    return {"key": "citation_grounding", "score": 1.0 if citations else 0.0,
            "n_citations": len(citations or [])}


# ── P0 science-eval metrics (faithfulness / citation-PR / source-recall / abstention) ──
def _cite_source(c: Any) -> str:
    """Citation source string (works for Citation objects and dicts)."""
    if isinstance(c, dict):
        return str(c.get("source", ""))
    return str(getattr(c, "source", "") or "")


def source_recall(retrieved_sources: list[str], expected: list[str]) -> dict:
    """Did retrieval surface a relevant source? ``score`` is binary (≥1 expected
    fragment present in any retrieved source path), ``coverage`` is the fraction.
    Matching is case-insensitive substring, so it is robust to path-separator and
    .pdf/.md variants. ``score`` is None when the item declares no expected source."""
    exp = [e for e in (expected or []) if e]
    if not exp:
        return {"key": "source_recall", "score": None}
    blob = " || ".join(s for s in (retrieved_sources or []) if s).lower()
    matched = [e for e in exp if e.lower() in blob]
    return {"key": "source_recall", "score": 1.0 if matched else 0.0,
            "coverage": round(len(matched) / len(exp), 3),
            "matched": matched, "n_expected": len(exp)}


# ── rank-aware retrieval metrics (recall@k / MRR / nDCG) — ported from ─────────────
# ai-engineering_HW/lesson-09-rag-systems-enterprise/homework/template/metrics.py.
# Relevance is binary and uses the SAME case-insensitive substring matcher as
# ``source_recall`` (an expected path fragment present in a retrieved source path),
# so a retrieved source counts as relevant exactly when source_recall would match it.
def _first_relevant_rank(retrieved_sources: list[str], expected: list[str]) -> int | None:
    """1-based rank of the first retrieved source matching any expected fragment
    (case-insensitive substring), else None. None propagates when there is nothing
    to match (caller returns score=None so empty-expected items are excluded from means)."""
    exp = [e.lower() for e in (expected or []) if e]
    if not exp:
        return None
    for rank, s in enumerate((retrieved_sources or []), start=1):
        sl = (s or "").lower()
        if any(e in sl for e in exp):
            return rank
    return None


def recall_at_k(retrieved_sources: list[str], expected: list[str], k: int) -> dict:
    """1.0 if a relevant source appears in the top-k retrieved, else 0.0. score=None
    when the item declares no expected source (so it drops out of the mean)."""
    if not [e for e in (expected or []) if e]:
        return {"key": f"recall_at_{k}", "score": None}
    rank = _first_relevant_rank(retrieved_sources, expected)
    hit = rank is not None and rank <= k
    return {"key": f"recall_at_{k}", "score": 1.0 if hit else 0.0, "rank": rank}


def mrr(retrieved_sources: list[str], expected: list[str], k: int = 10) -> dict:
    """Reciprocal rank (1/rank) of the first relevant source within top-k, else 0.0.
    score=None when no expected source is declared."""
    if not [e for e in (expected or []) if e]:
        return {"key": "mrr", "score": None}
    rank = _first_relevant_rank(retrieved_sources, expected)
    score = 1.0 / rank if (rank is not None and rank <= k) else 0.0
    return {"key": "mrr", "score": round(score, 3), "rank": rank}


def ndcg_at_k(retrieved_sources: list[str], expected: list[str], k: int = 10) -> dict:
    """Binary-relevance nDCG@k with a single relevant target: IDCG = 1 (relevant at
    rank 1), so nDCG = DCG = 1/log2(rank+1) when the first relevant source is within
    top-k, else 0.0. A coarse relative proxy (binary substring relevance) — used to
    compare retrieval configurations, not as an absolute IR score. score=None when no
    expected source is declared."""
    if not [e for e in (expected or []) if e]:
        return {"key": f"ndcg_at_{k}", "score": None}
    rank = _first_relevant_rank(retrieved_sources, expected)
    if rank is None or rank > k:
        return {"key": f"ndcg_at_{k}", "score": 0.0, "rank": rank}
    return {"key": f"ndcg_at_{k}", "score": round(1.0 / math.log2(rank + 1), 3), "rank": rank}


def ranking_metrics(retrieved_sources: list[str], expected: list[str],
                    ks: tuple[int, ...] = (1, 5, 10)) -> dict:
    """Convenience bundle: recall@each-k + mrr@max(ks) + ndcg@max(ks). Values are the
    bare scores (or None), keyed ``recall_at_{k}`` / ``mrr`` / ``ndcg`` for direct
    insertion into harness rows and history.csv."""
    kmax = max(ks) if ks else 10
    out: dict[str, Any] = {f"recall_at_{k}": recall_at_k(retrieved_sources, expected, k)["score"] for k in ks}
    out["mrr"] = mrr(retrieved_sources, expected, kmax)["score"]
    out["ndcg"] = ndcg_at_k(retrieved_sources, expected, kmax)["score"]
    return out


def citation_pr(citations: list, expected: list[str]) -> dict:
    """Precision/recall of cited sources vs the expected source fragments (ALCE-lite,
    substring match). recall = expected fragments cited; precision = citations that
    match an expected fragment. None when the item declares no expected source."""
    exp = [e for e in (expected or []) if e]
    if not exp:
        return {"key": "citation_pr", "recall": None, "precision": None}
    cl = [_cite_source(c).lower() for c in (citations or [])]
    rec_hits = [e for e in exp if any(e.lower() in c for c in cl)]
    prec_hits = [c for c in cl if any(e.lower() in c for e in exp)]
    recall = round(len(rec_hits) / len(exp), 3)
    precision = round(len(prec_hits) / len(cl), 3) if cl else None
    return {"key": "citation_pr", "recall": recall, "precision": precision,
            "n_citations": len(cl), "n_expected": len(exp)}


def abstention_correctness(abstained: bool, abstain_expected: bool) -> dict:
    """1.0 iff the answer abstained exactly when it should have (and answered when it
    should have). The single most trust-relevant metric for a scientific audience."""
    return {"key": "abstention_correctness",
            "score": 1.0 if bool(abstained) == bool(abstain_expected) else 0.0,
            "abstained": bool(abstained), "expected": bool(abstain_expected)}


_ABSTAIN_RE = re.compile(
    r"(в|у)\s+баз[іи]\s+знань[^.]{0,40}(нема|відсут|не\s+знайш)"
    # the assistant's actual KB-abstention phrasing — "у наданому контексті/фрагменті немає":
    r"|(у|в)\s+наданому\s+(контекст|фрагмент)\w*[^.]{0,45}(нема|відсут)"
    r"|нема(є)?\s+(конкретн\w+\s+|потрібн\w+\s+|технічн\w+\s+|достатньо\s+|такої\s+|такої\s+)?"
    r"(інформаці\w+|характеристик\w*|деталей|значен\w+|специфікац\w+)"
    r"|потрібн\w+\s+(доступ\s+до\s+)?(повної\s+версії|технічн\w+\s+документац|паспорт)"
    r"|звернутися\s+до\s+(постачальник|технічн\w+\s+(відділ|документац)|виробник)"
    r"|не\s+можу\s+(тобі\s+)?допомогти"
    r"|це\s+не\s+моя\s+функці"
    r"|я\s+(лише\s+|тільки\s+)?помічник[- ]?(планувальник|систем)"
    r"|поза\s+(моєю\s+)?компетенц"
    r"|нема(є)?\s+(таких\s+|потрібних\s+)?даних"
    r"|не\s+маю\s+(таких\s+|потрібних\s+)?даних"
    r"|не\s+належить\s+до\s+(мого\s+|)дом",
    re.IGNORECASE,
)


def is_abstention(text: str, route: str = "") -> bool:
    """Heuristic: did the assistant decline/abstain rather than answer the question?
    True for an out_of_scope route or a recognisable refusal/'not in KB' phrasing."""
    if (route or "") == "out_of_scope":
        return True
    return bool(_ABSTAIN_RE.search(text or ""))


_FAITH_PROMPT = """Ти — суворий перевіряльник обґрунтованості (faithfulness) технічної відповіді.
Дано ВІДПОВІДЬ помічника-технолога й КОНТЕКСТ (фрагменти з бази знань, якими він мав
користуватися). Розклади відповідь на атомарні ФАКТИЧНІ/ТЕХНІЧНІ твердження (числа,
матеріали, причинно-наслідкові заяви) — загальні вступні фрази НЕ рахуй. Для кожного
визнач, чи його підтверджує КОНТЕКСТ (supported) чи ні (unsupported — узято з пам'яті
моделі / суперечить / відсутнє).

ВІДПОВІДЬ:
\"\"\"{answer}\"\"\"

КОНТЕКСТ:
\"\"\"{evidence}\"\"\"

Поверни ЧИСТИЙ JSON: {{"n_claims": <ціле>, "n_supported": <ціле>,
"faithfulness": <частка 0..1>, "unsupported": ["<коротко перше непідтверджене>", ...]}}."""


def faithfulness(answer: str, evidence: str, *, usage: LLMUsage | None = None) -> dict:
    """Claim-level groundedness via LLM-judge (Correctness≠Faithfulness): fraction of
    atomic technical claims entailed by the retrieved evidence. Best-effort; returns
    score -1.0 on judge failure and None for an empty/abstained answer."""
    if not (answer or "").strip():
        return {"key": "faithfulness", "score": None}
    try:
        payload = _FAITH_PROMPT.format(answer=answer[:2000], evidence=(evidence or "")[:6000])
        resp = call_llm(
            agent_name="faithfulness_judge", role_key="judge",
            messages=[{"role": "user", "content": payload}],
            usage=usage, temperature=0.0, max_tokens=500,
            response_format={"type": "json_object"},
        )
        parsed = parse_json_object(resp.choices[0].message.content or "{}")
        total = int(parsed.get("n_claims", 0) or 0)
        supported = int(parsed.get("n_supported", 0) or 0)
        score = parsed.get("faithfulness")
        if score is None:
            score = (supported / total) if total else 1.0
        return {"key": "faithfulness", "score": round(float(score), 3),
                "n_claims": total, "n_supported": supported,
                "unsupported": parsed.get("unsupported", [])}
    except Exception as exc:  # noqa: BLE001 — judge is best-effort
        return {"key": "faithfulness", "score": -1.0, "error": str(exc)[:120]}


__all__ = [
    "success_rate", "tool_selection_accuracy", "groundedness",
    "intent_match", "safety_refusal", "citation_grounding",
    "source_recall", "citation_pr", "abstention_correctness", "is_abstention",
    "faithfulness",
    "recall_at_k", "mrr", "ndcg_at_k", "ranking_metrics",
]
