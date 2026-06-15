"""The assistant brain: ground → route → dispatch to a skill → synthesize.

Every request is grounded with structured page-context; the cheap router picks
a route; the route dispatches to the right skill (RAG instructions, data tools /
text2SQL, history, vision, schedule, refusal). The answer model writes the final
Ukrainian reply from the gathered evidence.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any

from src.assistant import config, router, vision
from src.assistant.cache import get_cache
from src.assistant.data import tools
from src.assistant.grounding import context_builder
from src.assistant.llm import LLMUsage, call_llm, call_llm_stream
from src.assistant.schema import AssistantRequest, AssistantResponse, Citation
from src.assistant.security import injection
from src.assistant.skills.schedule import run_schedule_command
from src.assistant.tracing import traceable
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

_DATA_ROUTES = {"data_query", "analysis", "history"}

# Operator mode toggle → forced route. hybrid keeps the router's choice; data
# forces the tools/DB path (no RAG/KB); kb forces the RAG/knowledge-base path
# (no tools). Camera (screen_vision) and schedule commands are never forced.
_MODE_FORCE = {"data": "data_query", "kb": "instructions"}


def _apply_mode(mode: str, route: str) -> str:
    if route in ("screen_vision", "schedule_action"):
        return route
    return _MODE_FORCE.get(mode, route)


def _trace_grounded(tool_trace: list[dict]) -> bool:
    """True iff at least one tool call SUCCEEDED (no error) — i.e. the answer can be
    backed by real system data. A trace of only failed calls (e.g. DB unreachable)
    must NOT earn the «📊 за даними системи» marker — that is exactly the audit #13
    confident-but-ungrounded trust failure (T0.5)."""
    return any(not (t or {}).get("error") for t in (tool_trace or []))


# An answer that abstains / says "not in the knowledge base" must not be cached —
# else a transient "не знаю" is replayed after the KB later gains the content (T0.5).
_UNSURE_RE = re.compile(
    r"(в|у)\s+баз[іи]\s+знань[^.]{0,40}(нема|відсут|не\s+знайш)"
    r"|(у|в)\s+наданому\s+(контекст|фрагмент)\w*[^.]{0,45}(нема|відсут)"
    r"|\bне\s+знаю\b|не\s+можу\s+(тобі\s+)?допомогти|це\s+не\s+моя\s+функці",
    re.IGNORECASE,
)


def _contextualize_query(req: AssistantRequest, usage: LLMUsage) -> AssistantRequest:
    """Rewrite a follow-up into a standalone question using the conversation
    history (history-aware retrieval / "condense question"). Returns *req*
    unchanged when there's no history, on a screenshot turn, or on any error;
    otherwise a copy whose ``message`` carries the resolved context so routing,
    tools and retrieval don't lose the thread (e.g. «дай повний перелік BOPP» →
    «… BOPP плівок виробництва ПЛАСТХІМ»). The original message is stored/shown
    by the web layer; only the internal working copy is rewritten.
    """
    history = req.history or []
    if not history or req.screenshot_b64 or not (req.message or "").strip():
        return req
    try:
        convo = "\n".join(
            f"{'Оператор' if h.get('role') == 'user' else 'Помічник'}: {str(h.get('text', ''))[:400]}"
            for h in history[-6:]
        )
        user = (
            f"[ДІАЛОГ]\n{convo}\n\n[ОСТАННЄ ЗАПИТАННЯ]\n{req.message}\n\n"
            "[САМОДОСТАТНЄ ЗАПИТАННЯ]:"
        )
        resp = call_llm(
            agent_name="contextualize", role_key="router",
            messages=[
                {"role": "system", "content": config.prompt("contextualize")},
                {"role": "user", "content": user},
            ],
            usage=usage, temperature=0.0, max_tokens=120,
        )
        rewritten = (resp.choices[0].message.content or "").strip().strip('"').strip()
        if rewritten and rewritten != req.message and len(rewritten) <= 400:
            logger.info("assistant contextualize: %r -> %r", req.message[:60], rewritten[:80])
            return req.model_copy(update={"message": rewritten})
    except Exception:  # noqa: BLE001 — contextualization is best-effort
        pass
    return req

# Nomenclature SKU like 2.01.51045 / 6.0809.133 — distinctive enough to guard
# without false-positives on dates (DD.MM.YYYY starts with two digits).
_SKU_RE = re.compile(r"\b\d\.\d{2,4}\.\d{2,6}\b")
# Order reference «#12345» — only the explicit-hash form (plain digits are kg/counts).
_ORDER_REF_RE = re.compile(r"#(\d{4,7})\b")


def _strip_unverified_ids(text: str, evidence: str) -> str:
    """Anti-invention guardrail: a SKU code or order «#N» may appear in the answer
    ONLY if it is present in the grounding evidence (tool rows / system data) or in
    the operator's own question. Otherwise it is almost certainly an OCR/LLM
    distortion (codes differ by one digit) → replace it rather than mislead.
    A real "order #N not found" answer is preserved (the # is in the question).
    """
    if not text:
        return text
    ev = evidence or ""

    def _repl_sku(m: "re.Match[str]") -> str:
        code = m.group(0)
        return code if code in ev else "(код уточніть у таблиці)"

    def _repl_order(m: "re.Match[str]") -> str:
        return m.group(0) if m.group(1) in ev else "(№ уточніть у таблиці)"

    return _ORDER_REF_RE.sub(_repl_order, _SKU_RE.sub(_repl_sku, text))


# Prompt-injection defense (Block 2) — gated by features.injection_guard (default OFF →
# byte-identical behaviour). Input side: a jailbreak/override/leak-request is refused
# before any LLM call is spent. Output side: a suspected system-prompt leak is suppressed.
def _injection_refusal(usage: LLMUsage, reason: str) -> AssistantResponse:
    logger.warning("assistant: input blocked by injection guard (%s)", reason)
    u = usage.as_dict()
    u["injection_suspected"] = reason
    return AssistantResponse(
        text_md=config.prompt("refusal"), route="out_of_scope", usage=u, evidence="",
    )


def _guard_output(text: str) -> str:
    """Suppress a suspected system-prompt leak (defense-in-depth, gated)."""
    if config.feature("injection_guard") and injection.check_output(text):
        logger.warning("assistant: output suppressed by injection guard (system-prompt leak)")
        return config.prompt("refusal")
    return text
# Only stable, page-deterministic answers are cached (NOT live-data routes).
_CACHEABLE = {"instructions", "out_of_scope"}


def _cache_lookup(req: AssistantRequest) -> dict | None:
    # The cache key is (message, page key) and ignores both the order in focus and
    # the conversation. Skip it for order-grounded answers AND for any follow-up
    # (history present) — otherwise a near-identical follow-up replays the prior
    # answer (cosine ≥ threshold), which is the «identical answer» bug in chats.
    if not config.feature("cache") or _effective_order_id(req) is not None or req.history:
        return None
    try:
        return get_cache().lookup(req.message, req.page_context.key())
    except Exception:  # noqa: BLE001 — cache is best-effort
        return None


def _cache_store(req: AssistantRequest, route: str, text: str, citations: list[Citation]) -> None:
    if (not config.feature("cache") or route not in _CACHEABLE
            or _effective_order_id(req) is not None or req.history):
        return
    # T0.5: never cache an abstention / "not in the knowledge base" — else a transient
    # "не знаю" is replayed after the KB later gains the content (cache-poisoning).
    if config.feature("honest_marker") and _UNSURE_RE.search(text or ""):
        return
    try:
        get_cache().store(req.message, req.page_context.key(), {
            "text_md": text, "route": route,
            "citations": [c.model_dump() for c in citations],
        })
    except Exception:  # noqa: BLE001
        pass


def _response_from_cache(hit: dict) -> AssistantResponse:
    return AssistantResponse(
        text_md=hit.get("text_md", ""),
        route=hit.get("route", "instructions"),
        citations=[Citation(**c) for c in hit.get("citations", [])],
        usage={"cache_hit": True},
        evidence="",
    )


def _route_tools(route: str, message: str) -> list[str]:
    """Deterministic default tool set per route (config-light heuristic).

    A named SKU → resolve the material from data; deficit data-queries → the
    top-deficits-by-SKU tool (real codes, not OCR). These run before the
    route-default tools so verified data backs the answer.
    """
    low = message.lower()
    lead: list[str] = []
    if context_builder.extract_sku(message):
        lead.append("get_material")
    if "дефіцит" in low and any(
        w in low for w in ("топ", "найбіль", "перш", "почати", "скільки", "як", "які")
    ):
        lead.append("get_deficits_top")
    if route == "history":
        if "постач" in low:
            base = ["supply_commitment_events_recent", "latest_etl_run"]
        elif "зробити" in low or "дії" in low or "дій" in low:
            base = ["pending_proposed_actions"]
        else:
            base = ["etl_diff_summary", "latest_etl_run"]
    elif route == "analysis":
        base = ["order_risk", "material_readiness_breakdown", "allocation_status_breakdown"]
    elif "дефіцит" in low or "ризик" in low:  # data_query
        base = ["order_risk"]
    elif "готовн" in low or "матеріал" in low:
        base = ["material_readiness_breakdown"]
    else:
        base = ["pending_orders_count"]
    return lead + [t for t in base if t not in lead]


def _effective_order_id(req: AssistantRequest) -> int | None:
    """The order this request is about: one explicitly named in the message wins
    over the open card (so the assistant isn't trapped on the drawer)."""
    return (
        context_builder.extract_order_id(req.message)
        or req.page_context.focus_order_id()
    )


def _tool_params(req: AssistantRequest) -> dict[str, Any]:
    # A named order in the question overrides the open card; otherwise the open
    # card; otherwise None. (Never the first visible id — that's just DOM order.)
    # A named SKU + a default top-N limit feed get_material / get_deficits_top.
    return {
        "order_id": _effective_order_id(req),
        "sku": context_builder.extract_sku(req.message),
        "limit": 10,
    }


def _augment_with_requested_order(req: AssistantRequest, grounded: str) -> str:
    """If the operator names an order other than the open card, add ITS live
    summary so the answer model can switch to it instead of refusing."""
    msg_oid = context_builder.extract_order_id(req.message)
    if msg_oid is None or msg_oid == req.page_context.focus_order_id():
        return grounded
    extra = context_builder.summarize_order(msg_oid, opened=False)
    return grounded + ("\n\n" + extra if extra else "")


def _reasoning_extra_body() -> dict | None:
    """Config-driven extra_body that turns on extended thinking (Block 5)."""
    eb = config.agents_param("controller", {}).get("reasoning_extra_body")
    return eb if isinstance(eb, dict) and eb else {"reasoning": {"effort": "high"}}


def _extract_reasoning(resp: Any) -> str:
    """Best-effort: pull the model's extended-thinking text from an answer response.

    OpenRouter returns it on ``message.reasoning`` (or ``reasoning_content`` / model_extra)
    when ``reasoning_extra_body`` is set. Display-only — never affects the answer. Capped.
    """
    try:
        msg = resp.choices[0].message
    except Exception:  # noqa: BLE001
        return ""
    val = getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None)
    if not isinstance(val, str):
        extra = getattr(msg, "model_extra", None) or {}
        val = extra.get("reasoning") or extra.get("reasoning_content")
    return val.strip()[:6000] if isinstance(val, str) and val.strip() else ""


def _synthesize(
    evidence: str, message: str, route: str, usage: LLMUsage, *, system_key: str = "answer",
    reasoning: bool = False, out: dict | None = None,
) -> str:
    user = (
        f"[КОНТЕКСТ]\n{evidence}\n\n[ЗАПИТ ОПЕРАТОРА ({route})]\n{message}\n\n"
        "Дай стислу, корисну відповідь українською. Використовуй конкретні числа з контексту, "
        "де вони є; не вигадуй даних."
    )
    resp = call_llm(
        agent_name="answer",
        role_key="answer",
        messages=[
            {"role": "system", "content": config.prompt(system_key)},
            {"role": "user", "content": user},
        ],
        usage=usage,
        temperature=float(config.threshold("answer_temperature", 0.3)),
        max_tokens=int(config.threshold("answer_max_tokens", 1024)),
        extra_body=_reasoning_extra_body() if reasoning else None,  # Block 5: extended thinking
    )
    if out is not None and reasoning:
        out["reasoning"] = _extract_reasoning(resp)  # display-only thinking text (UX)
    return resp.choices[0].message.content or ""


def _synthesize_kb_ma(
    evidence: str, message: str, route: str, usage: LLMUsage, *, kb_used: bool,
    out: dict | None = None,
) -> str:
    """Multi-agent KB answer (INC-5). A controller picks a spec-complete answer mode
    for datasheet "all characteristics" queries; the answer_critic then verifies
    completeness + no-invention and triggers ONE bounded re-synthesis if the draft is
    too terse. Reduces EXACTLY to ``_synthesize`` when multi_agent is off or the plan
    opts out — so the single-shot path is unchanged by default."""
    from src.assistant.agents import answer_critic, controller, grounding

    plan = controller.classify(message, kb_used=kb_used)
    if plan.full_spec:
        system_key = "answer_kb_full"
    elif plan.sci_full:           # T1.6/T2.3: structured, cited, cross-source science answer
        system_key = "answer_sci_full"
    else:
        system_key = _answer_key(kb_used)
    # Block 5: extended thinking on the (complex) draft synthesis only — refines stay cheap.
    draft = _synthesize(evidence, message, route, usage, system_key=system_key, reasoning=plan.reasoning, out=out)

    # Budget for the EXTRA answer-layer calls (Block 4a binds the previously-unenforced
    # controller budget): count only calls beyond the base draft + cap wall-time.
    ctrl = config.agents_param("controller", {})
    max_extra = int(ctrl.get("max_extra_llm_calls", 3))
    max_ms = int(ctrl.get("max_added_ms", 8000))
    base_calls = usage.calls
    start = time.time()

    def _budget_ok() -> bool:
        return (usage.calls - base_calls) < max_extra and (time.time() - start) * 1000 < max_ms

    if plan.run_answer_critic:
        crit = answer_critic.assess(draft, evidence, full_spec=plan.full_spec)
        iters = int(config.agents_param("answer_critic", {}).get("max_refine_iters", 1))
        if not crit.ok and iters >= 1 and _budget_ok():
            # One bounded refine; keep the stronger draft (never lowers grounding).
            hint = config.prompt("agent_answer_critic")
            refine_key = "answer_sci_full" if plan.sci_full else "answer_kb_full"
            draft2 = _synthesize(evidence, f"{message}\n\n{hint}", route, usage, system_key=refine_key)
            crit2 = answer_critic.assess(draft2, evidence, full_spec=plan.full_spec)
            if answer_critic.better(crit2, crit):
                draft = draft2
    # T1.4 claim-grounding gate — Block 4a broadens it from sci_full-only to ALL KB-grounded
    # answers (the deterministic number-critic above can't catch unsupported PROSE claims).
    # No-op unless agents.answer_critic.claim_check is on (unsupported_claims returns []).
    if kb_used and grounding.enabled():
        draft = _claim_grounded(draft, evidence, message, route, usage,
                                refine_key=system_key, budget_ok=_budget_ok)
    return draft


def _claim_grounded(draft: str, evidence: str, message: str, route: str, usage: LLMUsage,
                    *, refine_key: str = "answer_sci_full", budget_ok=None) -> str:
    """T1.4: if a KB answer carries claims the evidence does not support, refine once with a
    strict-grounding hint and keep the better draft (fewer unsupported). A no-op when
    ``agents.answer_critic.claim_check`` is off (unsupported_claims returns []). The refine is
    skipped when the controller budget is exhausted (``budget_ok`` False) — fail-safe to the
    draft. ``refine_key`` keeps the refine in the same prompt family as the draft."""
    from src.assistant.agents import grounding

    ups = grounding.unsupported_claims(draft, evidence, usage=usage)
    if not ups:
        return draft
    if budget_ok is not None and not budget_ok():
        return draft  # out of budget — keep the draft rather than spend another refine
    hint = config.prompt("agent_grounding")
    draft2 = _synthesize(evidence, f"{message}\n\n{hint}", route, usage, system_key=refine_key)
    ups2 = grounding.unsupported_claims(draft2, evidence, usage=usage)
    return draft2 if len(ups2) < len(ups) else draft


def _kb_relevant(query: str, res, usage: LLMUsage | None = None) -> bool:
    """Precise on/off-topic gate: a cheap yes/no check on the retrieved excerpts.

    The cosine score is a poor on/off-topic separator for short same-language
    queries (a Russian off-topic query floor-matches the Russian manuals as high
    as a real domain query). This asks the router model whether the excerpts
    actually answer the question. Fail-open (return True) on error so a judge
    outage degrades to the cosine decision rather than refusing everything.
    """
    if not config.kb_param("verify_relevance", True):
        return True
    try:
        n = int(config.kb_param("verify_max_chars", 1000))
        user = f"[ПИТАННЯ]\n{query}\n\n[ФРАГМЕНТИ З БАЗИ ЗНАНЬ]\n{res.knowledge[:n]}"
        resp = call_llm(
            agent_name="kb_verify",
            role_key="router",
            messages=[
                {"role": "system", "content": config.prompt("kb_verify")},
                {"role": "user", "content": user},
            ],
            usage=usage,
            temperature=0.0,
            max_tokens=8,
        )
        ans = (resp.choices[0].message.content or "").strip().lower()
        return ans.startswith(("так", "yes", "y"))
    except Exception:  # noqa: BLE001 — relevance check is best-effort (fail-open)
        return True


def _search_external_kb(
    query: str, usage: LLMUsage | None = None, *, force: bool = False,
    extra_variants: list[str] | None = None, top_k: int | None = None,
):
    """Return a KBResult if the external KB covers the query, else None.

    Two-stage gate: (1) cheap cosine pre-filter to skip obvious noise, then
    (2) an LLM relevance check that decides on/off-topic. ``force=True`` (the
    operator explicitly chose «База знань» mode) skips the relevance veto — the
    cosine pre-filter still applies, but the user has opted into KB answers.
    ``extra_variants`` / ``top_k`` carry the INC-6 planner variants and a widened
    second-pass budget.
    """
    from src.assistant.kb.search import search_kb

    res = search_kb(query, top_k=top_k, usage=usage, extra_variants=extra_variants)
    if not res.knowledge or res.best_score < config.kb_min_score():
        return None
    if not force and not _kb_relevant(query, res, usage):
        return None
    return res


def _kb_retrieve_ma(query: str, usage: LLMUsage | None = None, *, force: bool = False):
    """Multi-agent KB retrieval (INC-6): the query_planner adds attribute / per-product
    variants; the retrieval_critic checks sufficiency and triggers ONE widened second
    pass when the context is weak. Reduces to ``_search_external_kb`` when multi_agent
    is off. Never raises beyond what the callees handle."""
    if not config.feature("multi_agent"):
        return _search_external_kb(query, usage, force=force)
    from src.assistant.agents import query_planner, retrieval_critic

    plan = query_planner.plan(query, usage=usage)
    kb = _search_external_kb(query, usage, force=force, extra_variants=plan.variants)
    knowledge = kb.knowledge if kb is not None else ""
    score = kb.best_score if kb is not None else 0.0
    if not retrieval_critic.assess(knowledge, score, plan).sufficient:
        passes = int(config.agents_param("retrieval_critic", {}).get("max_retrieval_passes", 1))
        if passes >= 1:
            widen = int(config.kb_param("top_k", 10)) * 2
            kb2 = _search_external_kb(query, usage, force=True, extra_variants=plan.variants, top_k=widen)
            if kb2 is not None and (kb is None or len(kb2.knowledge) > len(kb.knowledge)):
                kb = kb2
    return kb


def _kb_answer_if_covered(req: AssistantRequest, grounded: str, usage: LLMUsage | None = None):
    """Route-agnostic *retrieve-before-refuse*: if the external KB confidently covers
    the question, return ``(evidence, citations)`` to answer from it; else ``None``.

    The operator's knowledge base (flexo/gravure literature, standards, patents,
    datasheets) often answers a technical question the router did not send to the
    KB path. This is the shared seam for the out_of_scope escalation (T0.1) and the
    data-route CRAG-gate (T0.3) — so both recover the same way instead of refusing
    or answering from the model's memory.
    """
    if not config.feature("external_kb"):
        return None
    try:
        kb = _kb_retrieve_ma(req.message, usage, force=False)
    except Exception:  # noqa: BLE001 — external KB is best-effort
        return None
    if kb is None:
        return None
    evidence = grounded + f"\n\n[ЗНАННЯ]\n[ЗОВНІШНЯ БАЗА ЗНАНЬ]\n{kb.knowledge}"
    return evidence, kb.citations


def _escalate_out_of_scope(req: AssistantRequest, grounded: str, usage: LLMUsage | None = None):
    """Off-domain query → consult the external KB before refusing (T0.1).

    The router flags equipment / domain-literature questions as ``out_of_scope``
    (they are outside print *planning*), but the external KB may answer them.
    Gated by ``features.retrieve_before_refuse`` (default ON — a named kill-switch
    for the already-shipped escalation); the real over-refusal fix is the sharper
    router scope (config ``prompts.router`` / ``heuristics``).
    """
    if not config.feature("retrieve_before_refuse"):
        return None
    return _kb_answer_if_covered(req, grounded, usage)


def _crag_recover(req: AssistantRequest, grounded: str, tool_trace: list[dict],
                  usage: LLMUsage | None = None):
    """CRAG-gate (T0.3): a recovery net for a query that landed on a data route but
    whose tools produced NO grounded data (every call errored). Instead of answering
    from the model's memory or refusing, consult the external KB (reusing the T0.1
    retrieve-before-refuse seam). Fires ONLY on an all-errored trace, so a genuine
    data query that legitimately returns 0 rows (a SUCCESSFUL call) is never diverted.
    Returns ``(evidence, citations)`` to answer from KB, else ``None``.
    Mostly belt-and-suspenders since T0.1 already routes science to instructions."""
    if not config.feature("crag_gate") or _trace_grounded(tool_trace):
        return None
    return _kb_answer_if_covered(req, grounded, usage)


def _gather_instructions(
    req: AssistantRequest, grounded: str, usage: LLMUsage | None = None,
) -> tuple[str, list[Citation], bool]:
    citations: list[Citation] = []
    knowledge = ""
    top_score = 0.0
    kb_used = False
    # «База знань» mode: the operator explicitly wants the knowledge base — always
    # consult it (not only when operator_help is weak) and skip the relevance veto.
    force_kb = req.normalised_mode() == "kb"
    if config.feature("rag"):
        try:
            from src.assistant.rag.index import get_retriever

            retr = get_retriever()  # shared singleton — model loads once, not per request
            if retr.available:
                hits = retr.retrieve(req.message)
                if hits:
                    top_score = float(hits[0][1])
                knowledge = "\n\n".join(f"[{c.source}] {c.text}" for c, _ in hits)
                citations = [
                    Citation(source=c.source, snippet=c.text[:180], url=c.url) for c, _ in hits
                ]
        except Exception:  # noqa: BLE001 — RAG is best-effort
            pass

    # ESCALATION: consult the external KB when the in-project instructions don't
    # cover the question (relevance below threshold) — OR always, in «База знань» mode.
    if config.feature("external_kb") and (force_kb or top_score < config.kb_escalation_threshold()):
        try:
            kb = _kb_retrieve_ma(req.message, usage, force=force_kb)
            if kb is not None:
                block = f"[ЗОВНІШНЯ БАЗА ЗНАНЬ]\n{kb.knowledge}"
                knowledge = f"{knowledge}\n\n{block}" if knowledge else block
                citations = citations + kb.citations
                kb_used = True
        except Exception:  # noqa: BLE001 — external KB is best-effort
            pass

    evidence = grounded + (f"\n\n[ЗНАННЯ]\n{knowledge}" if knowledge else "")
    return evidence, citations, kb_used


def _assistant_tool_msg(msg: Any) -> dict:
    """Re-serialize a model message that requested tool calls (to append back)."""
    return {
        "role": "assistant",
        "content": getattr(msg, "content", "") or "",
        "tool_calls": [
            {"id": tc.id, "type": "function",
             "function": {"name": tc.function.name, "arguments": tc.function.arguments or "{}"}}
            for tc in (getattr(msg, "tool_calls", None) or [])
        ],
    }


def _agentic_gather(req: AssistantRequest, grounded: str, usage: LLMUsage) -> tuple[str, list[dict]]:
    """Function-calling loop: the answer model CHOOSES typed read-only tools and
    args; we execute them and collect the rows into evidence. Facts therefore
    come from verified data, not the model's guess. Returns (evidence, trace)."""
    schemas = tools.tool_schemas()
    user = (
        f"[КОНТЕКСТ ЕКРАНА]\n{grounded}\n\n[ЗАПИТ ОПЕРАТОРА]\n{req.message}\n\n"
        "Виклич потрібні функції, щоб дістати точні дані для відповіді."
    )
    messages: list[dict] = [
        {"role": "system", "content": config.prompt("tool_router")},
        {"role": "user", "content": user},
    ]
    evidence = grounded
    trace: list[dict] = []
    rounds = int(config.threshold("tool_max_rounds", 3))
    for _ in range(max(1, rounds)):
        resp = call_llm(
            agent_name="tool_router", role_key="answer", messages=messages,
            tools=schemas, usage=usage, temperature=0.0,
            max_tokens=int(config.threshold("tool_gather_max_tokens", 600)),
        )
        msg = resp.choices[0].message
        calls = getattr(msg, "tool_calls", None) or []
        if not calls:
            break
        messages.append(_assistant_tool_msg(msg))
        for tc in calls:
            name = getattr(tc.function, "name", "")
            try:
                args = json.loads(tc.function.arguments or "{}")
            except (ValueError, TypeError):
                args = {}
            res = tools.run_tool(name, args)
            trace.append({"tool": name, "args": args, "error": res.get("error")})
            blob = json.dumps((res.get("rows") or [])[:30], ensure_ascii=False, default=str)
            evidence += f"\n\n[{name} {args}] {blob}"
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": blob[:4000]})
    return evidence, trace


def _gather_data(req: AssistantRequest, grounded: str, route: str, usage: LLMUsage) -> tuple[str, list[dict]]:
    tool_trace: list[dict] = []
    # Live counters are relevant ONLY for data/analysis/history questions — they
    # are added here (not in the base grounding) so they don't leak into
    # instructions/how-to answers and don't cost a DB roundtrip there.
    live = context_builder.live_summary()
    evidence = grounded + (f"\n\n{live}" if live else "")
    if config.feature("tools") and config.feature("tool_calling"):
        # Agentic: the model picks tools + args (function-calling).
        ev2, tool_trace = _agentic_gather(req, evidence, usage)
        return ev2, tool_trace
    if config.feature("tools"):
        params = _tool_params(req)
        for name in _route_tools(route, req.message):
            res = tools.run_tool(name, params)
            tool_trace.append({k: res.get(k) for k in ("tool", "columns", "error")})
            if res.get("rows"):
                blob = json.dumps(res["rows"][:30], ensure_ascii=False, default=str)
                evidence += f"\n\n[{name}] {blob}"
    elif config.feature("text2sql"):
        from src.assistant.data.engine import ro_available
        from src.assistant.data.text2sql import run_text2sql

        if ro_available():
            r = run_text2sql(req.message, usage=usage)
            tool_trace.append({"tool": "text2sql", "sql": r.get("sql"), "error": r.get("error")})
            if r.get("ok") and r.get("rows"):
                blob = json.dumps(r["rows"][:30], ensure_ascii=False, default=str)
                evidence += f"\n\n[SQL] {blob}"
    return evidence, tool_trace


def _answer_key(kb_used: bool) -> str:
    """KB-grounded answers use the cross-lingual prompt (uk output, en evidence)."""
    return "answer_kb" if kb_used else "answer"


def _instructions(req: AssistantRequest, grounded: str, usage: LLMUsage) -> tuple[str, list[Citation], str]:
    evidence, citations, kb_used = _gather_instructions(req, grounded, usage)
    text = _synthesize_kb_ma(evidence, req.message, "instructions", usage, kb_used=kb_used)
    return text, citations, evidence


_STATUS = {
    "instructions": "Шукаю у довідці…",
    "data_query": "Аналізую дані…",
    "analysis": "Аналізую стан системи…",
    "history": "Переглядаю історію подій…",
    "schedule_action": "Звертаюся до планувальника розкладу…",
    "screen_vision": "Дивлюся на екран…",
    "clarify": "Уточнюю запит…",
    "out_of_scope": "Перевіряю запит…",
}


def _status_for(route: str) -> str:
    return _STATUS.get(route, "Готую відповідь…")


def _stage(key: str, label: str) -> dict:
    """A 'thinking step' event for the live progress panel (perceived-latency UX).

    Additive to the existing status/delta/done SSE protocol — clients that don't
    know the ``stage`` type simply ignore it; no answer logic is affected.
    """
    return {"type": "stage", "key": key, "label": label}


def _synthesize_stream(
    evidence: str, message: str, route: str, usage: LLMUsage, *, system_key: str = "answer",
):
    user = (
        f"[КОНТЕКСТ]\n{evidence}\n\n[ЗАПИТ ОПЕРАТОРА ({route})]\n{message}\n\n"
        "Дай стислу, корисну відповідь українською. Використовуй конкретні числа з контексту, "
        "де вони є; не вигадуй даних."
    )
    yield from call_llm_stream(
        agent_name="answer",
        role_key="answer",
        messages=[
            {"role": "system", "content": config.prompt(system_key)},
            {"role": "user", "content": user},
        ],
        usage=usage,
        temperature=float(config.threshold("answer_temperature", 0.3)),
        max_tokens=int(config.threshold("answer_max_tokens", 1024)),
    )


def stream_answer(req: AssistantRequest, *, my_schedule: Any = None):
    """Generator yielding event dicts: {type: status|stage|reasoning|delta|done, ...}.

    NOTE: deliberately NOT @traceable — langsmith's generator wrapper terminates the
    SSE stream early. LangSmith still captures every LLM call via wrap_openai (non-stream)
    + the @traceable router/answer/text2sql spans.

    Emits an immediate status (so the operator sees work starting), a route-aware
    status after classification, then streams the answer tokens, then a final
    ``done`` with citations/route/tool_trace/usage.
    """
    usage = LLMUsage()
    if config.feature("injection_guard"):
        clean, reason = injection.check_input(req.message)
        if not clean:
            logger.warning("assistant stream: input blocked by injection guard (%s)", reason)
            text = config.prompt("refusal")
            yield {"type": "delta", "text": text}
            yield {"type": "done", "route": "out_of_scope", "text": text,
                   "citations": [], "tool_trace": [], "usage": {"injection_suspected": reason}}
            return
    yield {"type": "status", "text": "Аналізую запит…"}
    yield _stage("analyze", "Аналізую запит")
    req = _contextualize_query(req, usage)  # follow-up → standalone (history-aware)
    cached = _cache_lookup(req)
    if cached is not None:
        yield {"type": "status", "text": "Знайшов готову відповідь…"}
        text = cached.get("text_md", "")
        yield {"type": "delta", "text": text}
        yield {
            "type": "done", "route": cached.get("route", "instructions"), "text": text,
            "citations": cached.get("citations", []), "tool_trace": [], "usage": {"cache_hit": True},
        }
        return
    grounded = _augment_with_requested_order(
        req, context_builder.build(req.page_context, include_live=False)
    )
    yield _stage("route", "Визначаю напрям запиту")
    rr = router.classify(
        req.message, req.page_context, has_screenshot=bool(req.screenshot_b64), usage=usage
    )
    route = _apply_mode(req.normalised_mode(), rr.route)
    yield {"type": "status", "text": _status_for(route), "route": route}

    citations: list[Citation] = []
    tool_trace: list[dict] = []

    def _done(text: str) -> dict:
        logger.info(
            "assistant stream: route=%s tool_calling=%s tools_called=%d",
            route, config.feature("tool_calling"), len(tool_trace),
        )
        return {
            "type": "done", "route": route, "text": text,
            "citations": [c.model_dump() for c in citations],
            "tool_trace": tool_trace, "usage": usage.as_dict(),
        }

    # Non-streamable routes: emit the whole answer as one delta.
    if route == "out_of_scope":
        yield _stage("kb_search", "Шукаю в базі знань")
        kb = _escalate_out_of_scope(req, grounded, usage)
        if kb is None:
            text = rr.refusal or config.prompt("refusal")
            _cache_store(req, route, text, citations)
            yield {"type": "delta", "text": text}
            yield _done(text)
            return
        # External KB covers this off-domain question → answer for real.
        route = "instructions"
        evidence, citations = kb
        yield {"type": "status", "text": _status_for("instructions"), "route": route}
        if config.feature("multi_agent"):
            yield _stage("reason", "Аналізую джерела й формую відповідь")
            _rinfo: dict = {}
            full = _guard_output(_strip_unverified_ids(
                _synthesize_kb_ma(evidence, req.message, "instructions", usage, kb_used=True, out=_rinfo),
                (evidence or "") + "\n" + req.message))
            if _rinfo.get("reasoning"):
                yield {"type": "reasoning", "text": _rinfo["reasoning"]}
            yield {"type": "delta", "text": full}
        else:
            yield _stage("compose", "Формую відповідь")
            acc = []
            for token in _synthesize_stream(evidence, req.message, "instructions", usage, system_key="answer_kb"):
                acc.append(token)
                yield {"type": "delta", "text": token}
            full = _guard_output(_strip_unverified_ids("".join(acc), (evidence or "") + "\n" + req.message))
        _cache_store(req, route, full, citations)
        yield _done(full)
        return
    if route == "schedule_action":
        text = run_schedule_command(req.message, scope=req.normalised_scope(), my_schedule=my_schedule)
        yield {"type": "delta", "text": text}
        yield _done(text)
        return
    if route == "screen_vision":
        text = "Щоб я подивився на екран, натисніть 📷 «Прочитати екран» у панелі помічника."
        yield {"type": "delta", "text": text}
        yield _done(text)
        return

    # Gather evidence (non-streamed), then stream the synthesis.
    system_key = "answer"
    kb_used = False
    if route == "instructions":
        yield _stage("kb_search", "Шукаю в базі знань")
        evidence, citations, kb_used = _gather_instructions(req, grounded, usage)
        system_key = _answer_key(kb_used)
    elif route in _DATA_ROUTES:
        yield _stage("data_query", "Звертаюся до даних системи")
        evidence, tool_trace = _gather_data(req, grounded, route, usage)
        kb = _crag_recover(req, grounded, tool_trace, usage)  # T0.3 recovery net
        if kb is not None:
            evidence, citations = kb
            route, kb_used, system_key = "instructions", True, _answer_key(True)
            yield {"type": "status", "text": _status_for("instructions"), "route": route}
    elif route == "clarify":
        evidence = grounded
        req = req.model_copy(update={
            "message": f"{req.message}\n\n(Запит неоднозначний — постав ОДНЕ коротке уточнювальне питання.)"
        })
    else:
        yield _stage("kb_search", "Шукаю в базі знань")
        evidence, citations, kb_used = _gather_instructions(req, grounded, usage)
        system_key = _answer_key(kb_used)

    # MA (INC-5): for KB-grounded answers, compute the (possibly refined, spec-complete)
    # answer off-stream and emit it once — token-streaming is preserved for every other
    # path and whenever multi_agent is off.
    if config.feature("multi_agent") and kb_used:
        yield _stage("reason", "Аналізую джерела й формую відповідь")
        _rinfo: dict = {}
        full = _guard_output(_strip_unverified_ids(
            _synthesize_kb_ma(evidence, req.message, route, usage, kb_used=kb_used, out=_rinfo),
            (evidence or "") + "\n" + req.message))
        if _rinfo.get("reasoning"):
            yield {"type": "reasoning", "text": _rinfo["reasoning"]}
        yield {"type": "delta", "text": full}
        _cache_store(req, route, full, citations)
        yield _done(full)
        return

    yield _stage("compose", "Формую відповідь")
    acc: list[str] = []
    for token in _synthesize_stream(evidence, req.message, route, usage, system_key=system_key):
        acc.append(token)
        yield {"type": "delta", "text": token}
    full = _strip_unverified_ids("".join(acc), (evidence or "") + "\n" + req.message)
    guarded = _guard_output(full)  # best-effort: protects cache/record (tokens already streamed)
    if guarded != full:
        full = guarded
    elif tool_trace and (_trace_grounded(tool_trace) or not config.feature("honest_marker")):
        _m = context_builder.data_freshness_marker()  # §4 / T0.5: only on grounded data
        if _m:
            full = f"{full}\n\n{_m}"
    _cache_store(req, route, full, citations)
    yield _done(full)


@traceable(name="assistant.answer")
def answer(req: AssistantRequest, *, my_schedule: Any = None) -> AssistantResponse:
    """Main entry point: produce an AssistantResponse for a request."""
    usage = LLMUsage()
    if config.feature("injection_guard"):
        clean, reason = injection.check_input(req.message)
        if not clean:
            return _injection_refusal(usage, reason or "blocked")
    req = _contextualize_query(req, usage)  # follow-up → standalone (history-aware)
    cached = _cache_lookup(req)
    if cached is not None:
        return _response_from_cache(cached)
    grounded = _augment_with_requested_order(
        req, context_builder.build(req.page_context, include_live=False)
    )
    rr = router.classify(
        req.message,
        req.page_context,
        has_screenshot=bool(req.screenshot_b64),
        usage=usage,
    )
    route = _apply_mode(req.normalised_mode(), rr.route)
    citations: list[Citation] = []
    tool_trace: list[dict] = []
    evidence = grounded
    clarify = False

    if route == "out_of_scope":
        kb = _escalate_out_of_scope(req, grounded, usage)
        if kb is not None:
            evidence, citations = kb
            route = "instructions"  # answered from external KB, not a refusal
            text = _synthesize_kb_ma(evidence, req.message, "instructions", usage, kb_used=True)
        else:
            text = rr.refusal or config.prompt("refusal")
    elif route == "screen_vision":
        if req.screenshot_b64 and config.feature("vision"):
            # Facts from data, screenshot for context: gather via the tool loop
            # FIRST, then let vision answer from that evidence (not from pixels).
            evidence, tool_trace = _gather_data(req, grounded, "data_query", usage)
            text = vision.describe(
                req.screenshot_b64, req.page_context,
                message=req.message, grounded=evidence, usage=usage,
            )
        else:
            text = (
                "Щоб я подивився на екран, увімкніть перегляд екрана та натисніть «Прочитати екран» "
                "(📷) у панелі помічника."
            )
    elif route == "schedule_action":
        text = run_schedule_command(req.message, scope=req.normalised_scope(), my_schedule=my_schedule)
    elif route == "instructions":
        text, citations, evidence = _instructions(req, grounded, usage)
    elif route in _DATA_ROUTES:
        evidence, tool_trace = _gather_data(req, grounded, route, usage)
        kb = _crag_recover(req, grounded, tool_trace, usage)  # T0.3 recovery net
        if kb is not None:
            evidence, citations = kb
            route = "instructions"  # recovered from KB, not a memory/refusal answer
            text = _synthesize_kb_ma(evidence, req.message, "instructions", usage, kb_used=True)
        else:
            text = _synthesize(evidence, req.message, route, usage)
    elif route == "clarify":
        clarify = True
        text = _synthesize(
            grounded,
            f"{req.message}\n\n(Запит неоднозначний — постав ОДНЕ коротке уточнювальне питання.)",
            "clarify",
            usage,
        )
    else:  # safety net
        text, citations, evidence = _instructions(req, grounded, usage)

    text = _strip_unverified_ids(text, (evidence or "") + "\n" + req.message)
    guarded = _guard_output(text)
    if guarded != text:
        text = guarded  # leak suppressed → refusal, no freshness marker
    elif tool_trace and (_trace_grounded(tool_trace) or not config.feature("honest_marker")):
        _m = context_builder.data_freshness_marker()  # §4 / T0.5: only on grounded data
        if _m:
            text = f"{text}\n\n{_m}"
    logger.info(
        "assistant answer: route=%s tool_calling=%s tools_called=%d",
        route, config.feature("tool_calling"), len(tool_trace),
    )
    _cache_store(req, route, text, citations)
    return AssistantResponse(
        text_md=text,
        route=route,
        citations=citations,
        tool_trace=tool_trace,
        usage=usage.as_dict(),
        clarify=clarify,
        evidence=evidence,
    )


__all__ = ["answer", "stream_answer"]
