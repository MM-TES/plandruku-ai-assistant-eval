"""Intent router: cheap-model JSON classifier with a heuristic fast-path.

70–80% of obvious operator queries are routed by regex (config ``heuristics``)
without an LLM call; the rest fall through to a cheap model (config
``models.router``) that returns a strict JSON ``{route, reason, refusal?}``.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from src.assistant import config
from src.assistant.llm import LLMUsage, call_llm
from src.assistant.schema import PageContext
from src.assistant.tracing import parse_json_object, traceable

_EXEMPLARS_PATH = Path(__file__).resolve().parents[2] / "config" / "router_exemplars.jsonl"
_EXEMPLAR_CACHE: tuple[Any, list[str]] | None = None

ROUTES = {
    "instructions",
    "data_query",
    "analysis",
    "history",
    "screen_vision",
    "schedule_action",
    "clarify",
    "out_of_scope",
}
_DEFAULT_ROUTE = "instructions"


@dataclass
class RouteResult:
    route: str
    reason: str = ""
    refusal: Optional[str] = None
    source: str = "llm"  # heuristic | llm | forced


def _compiled_patterns() -> list[tuple[str, re.Pattern]]:
    out: list[tuple[str, re.Pattern]] = []
    for route, pattern in config.heuristics().items():
        if route.startswith("_") or not isinstance(pattern, str):
            continue
        try:
            out.append((route, re.compile(pattern, re.IGNORECASE)))
        except re.error:
            continue
    return out


def heuristic_route(message: str) -> str | None:
    """Return a route iff exactly one heuristic pattern matches (else None)."""
    text = (message or "").lower()
    hits = {route for route, rx in _compiled_patterns() if rx.search(text)}
    if len(hits) == 1:
        return next(iter(hits))
    return None


def _load_exemplars() -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    try:
        for line in _EXEMPLARS_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            d = json.loads(line)
            if d.get("route") in ROUTES and d.get("text"):
                out.append((d["route"], d["text"]))
    except Exception:  # noqa: BLE001 — exemplar router is an optional booster
        pass
    return out


def _exemplars() -> tuple[Any, list[str]]:
    """(L2-normalized exemplar embedding matrix, parallel route labels), cached.
    Uses the shared local MiniLM embedder — NO API call."""
    global _EXEMPLAR_CACHE
    if _EXEMPLAR_CACHE is None:
        ex = _load_exemplars()
        if not ex:
            _EXEMPLAR_CACHE = (None, [])
            return _EXEMPLAR_CACHE
        try:
            from src.assistant.rag.index import get_embedder

            mat = np.asarray(get_embedder().encode([t for _, t in ex]), dtype=np.float32)
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            _EXEMPLAR_CACHE = (mat / norms, [r for r, _ in ex])
        except Exception:  # noqa: BLE001
            _EXEMPLAR_CACHE = (None, [])
    return _EXEMPLAR_CACHE


def semantic_route(message: str) -> str | None:
    """Route by cosine to per-route exemplars (local embedder, no API). Returns a route
    ONLY when the best route's score >= router_semantic_min AND beats the runner-up route
    by >= router_tie_margin; else None (defer to the LLM classifier). Conservative by
    design so it never overrides the LLM on an ambiguous query."""
    mat, routes = _exemplars()
    if mat is None or not (message or "").strip():
        return None
    try:
        from src.assistant.rag.index import get_embedder

        qv = np.asarray(get_embedder().encode([message]), dtype=np.float32)[0]
        qv = qv / (np.linalg.norm(qv) or 1.0)
        sims = mat @ qv
        best_by_route: dict[str, float] = {}
        for r, s in zip(routes, sims):
            best_by_route[r] = max(best_by_route.get(r, -1.0), float(s))
        ranked = sorted(best_by_route.items(), key=lambda kv: kv[1], reverse=True)
        if not ranked:
            return None
        top_route, top_score = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else -1.0
        min_score = float(config.threshold("router_semantic_min", 0.5))
        margin = float(config.threshold("router_tie_margin", 0.08))
        if top_score >= min_score and (top_score - second) >= margin:
            return top_route
    except Exception:  # noqa: BLE001 — fail-open to the LLM classifier
        return None
    return None


@traceable(name="assistant.router")
def classify(
    message: str,
    page_context: PageContext | None = None,
    *,
    has_screenshot: bool = False,
    usage: LLMUsage | None = None,
) -> RouteResult:
    """Classify *message* into one of ``ROUTES``."""
    if has_screenshot:
        return RouteResult("screen_vision", "до запиту додано скріншот", source="forced")

    fast = heuristic_route(message)
    if fast:
        return RouteResult(fast, "правило fast-path", source="heuristic")

    if config.feature("semantic_router"):
        sem = semantic_route(message)
        if sem:
            return RouteResult(sem, "семантичний роутер (exemplar)", source="semantic")

    ctx = ""
    if page_context is not None:
        ctx = f"\n\n[Контекст сторінки] route={page_context.key()}"
        focus = page_context.focus_order_id()
        if focus is not None:
            ctx += f", відкрита картка замовлення #{focus}"
        if page_context.visible_entity_ids:
            ctx += f", видимі ID: {', '.join(page_context.visible_entity_ids[:10])}"

    resp = call_llm(
        agent_name="router",
        role_key="router",
        messages=[
            {"role": "system", "content": config.prompt("router")},
            {"role": "user", "content": f"{message}{ctx}"},
        ],
        usage=usage,
        temperature=float(config.threshold("router_temperature", 0.0)),
        max_tokens=int(config.threshold("router_max_tokens", 400)),
        response_format={"type": "json_object"},
    )
    parsed = parse_json_object(resp.choices[0].message.content or "{}")
    route = parsed.get("route", _DEFAULT_ROUTE)
    if route not in ROUTES:
        route = _DEFAULT_ROUTE
    return RouteResult(
        route=route,
        reason=parsed.get("reason", ""),
        refusal=parsed.get("refusal"),
        source="llm",
    )


__all__ = ["ROUTES", "RouteResult", "classify", "heuristic_route"]
