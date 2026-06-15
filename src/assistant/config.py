"""Typed accessors over ``config/assistant.json`` + assistant env secrets.

Single source of truth for models-per-role, pricing, feature flags, thresholds,
prompts, page descriptions and heuristics. No assistant behaviour is hardcoded
in ``.py`` — it all flows through here (project rule #1).
"""
from __future__ import annotations

import functools
import os
from typing import Any

from src.utils.config_loader import load_config

_DEFAULT_PRICE: tuple[float, float] = (1.0, 5.0)
_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


@functools.lru_cache(maxsize=1)
def _cfg() -> dict[str, Any]:
    return load_config("assistant")


def reload() -> None:
    """Drop the cached config (used by tests / after a config edit)."""
    _cfg.cache_clear()


# --- master switch + feature flags ------------------------------------------
def is_enabled() -> bool:
    return bool(_cfg().get("enabled", False))


def feature(name: str) -> bool:
    return bool(_cfg().get("features", {}).get(name, False))


# --- models / pricing / fallback --------------------------------------------
def model_for(role: str) -> str:
    return _cfg()["models"][role]


def fallback_for(model: str) -> str | None:
    return _cfg().get("fallback", {}).get(model)


def extra_body_for(model: str) -> dict[str, Any] | None:
    return _cfg().get("extra_body", {}).get(model)


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    in_p, out_p = _cfg().get("pricing_usd_per_m", {}).get(model, _DEFAULT_PRICE)
    return (input_tokens * in_p + output_tokens * out_p) / 1_000_000


# --- thresholds / prompts / pages / heuristics ------------------------------
def threshold(name: str, default: Any = None) -> Any:
    return _cfg().get("thresholds", {}).get(name, default)


def prompt(name: str) -> str:
    return _cfg().get("prompts", {}).get(name, "")


def pages() -> dict[str, Any]:
    return _cfg().get("pages", {})


def heuristics() -> dict[str, str]:
    return _cfg().get("heuristics", {})


def links() -> list[dict[str, Any]]:
    """Internal destinations the assistant may turn into clickable links."""
    items = _cfg().get("links", {}).get("items", [])
    return [i for i in items if isinstance(i, dict)]


# --- external knowledge base (escalation tier) ------------------------------
def kb() -> dict[str, Any]:
    return _cfg().get("kb", {})


def kb_path() -> str:
    return kb().get("path", "")


def kb_embed_model() -> str:
    return kb().get("embed_model", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


def kb_param(name: str, default: Any = None) -> Any:
    return kb().get(name, default)


def kb_escalation_threshold() -> float:
    """If the operator-help top relevance is BELOW this, escalate to the KB."""
    return float(kb().get("escalation_threshold", 0.5))


def kb_min_score() -> float:
    """The KB top relevance must be at least this to answer from the KB."""
    return float(kb().get("min_score", 0.3))


# --- multi-agent answer layer (INC-5/6) -------------------------------------
def agents() -> dict[str, Any]:
    return _cfg().get("agents", {})


def agents_param(name: str, default: Any = None) -> Any:
    """Read an ``agents.<name>`` sub-config (e.g. answer_critic, controller)."""
    return agents().get(name, default)


# --- OpenRouter --------------------------------------------------------------
def openrouter_api_key() -> str | None:
    return os.getenv("OPENROUTER_API_KEY")


def openrouter_base_url() -> str:
    return _cfg().get("openrouter", {}).get("base_url", _DEFAULT_BASE_URL)


def openrouter_headers() -> dict[str, str]:
    o = _cfg().get("openrouter", {})
    headers: dict[str, str] = {}
    if o.get("referer"):
        headers["HTTP-Referer"] = o["referer"]
    if o.get("title"):
        headers["X-Title"] = o["title"]
    return headers


# --- LangSmith ---------------------------------------------------------------
def langsmith_project() -> str:
    return _cfg().get("langsmith", {}).get("project", "plandruku-assistant")


def langsmith_api_key() -> str | None:
    return os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")


def langsmith_tracing_enabled() -> bool:
    """Tracing is governed at runtime by the LANGSMITH_TRACING env var.

    Reading env (not a cached config value) keeps it test-friendly: a test can
    monkeypatch the env to force tracing off without touching the config cache.
    """
    return os.getenv("LANGSMITH_TRACING", "false").strip().lower() == "true" and bool(
        langsmith_api_key()
    )
