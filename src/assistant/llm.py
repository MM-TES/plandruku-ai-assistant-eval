"""OpenRouter client (OpenAI SDK) + LLM usage/cost tracking.

Ported from the ai-engineering course (lesson-11 ``app/llm.py``) and adapted to
read everything from ``config/assistant.json`` via ``src.assistant.config`` and
to log through the project logger instead of ``print``.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

from src.assistant import config
from src.utils.logger import setup_logger

_logger = setup_logger(__name__)


def get_client(*, wrap: bool = True) -> Any:
    """Build an OpenAI-SDK client pointed at OpenRouter.

    Args:
        wrap: when True (default) and LangSmith tracing is on, wrap the client with
            ``wrap_openai`` for per-call LLM spans. Streaming callers pass ``wrap=False``
            so the SSE token stream is never routed through the tracing wrapper.

    Raises:
        RuntimeError: if OPENROUTER_API_KEY is not configured.
    """
    api_key = config.openrouter_api_key()
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set in .env")
    from openai import OpenAI

    client = OpenAI(
        api_key=api_key,
        base_url=config.openrouter_base_url(),
        default_headers=config.openrouter_headers(),
    )
    # LangSmith: per-call LLM spans (tokens/latency) when tracing is on. No-op
    # otherwise; never let tracing break the client.
    if wrap and config.langsmith_tracing_enabled():
        try:
            from langsmith.wrappers import wrap_openai

            client = wrap_openai(client)
        except Exception:  # noqa: BLE001
            _logger.info("langsmith wrap_openai unavailable — proceeding untraced")
    return client


@dataclass
class LLMUsage:
    """Accumulates token/cost across calls, with a per-agent breakdown."""

    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    duration_ms: int = 0
    calls: int = 0
    by_agent: dict[str, dict[str, Any]] = field(default_factory=dict)

    def add(self, agent: str, input_t: int, output_t: int, cost: float, ms: int) -> None:
        self.input_tokens += input_t
        self.output_tokens += output_t
        self.cost_usd += cost
        self.duration_ms += ms
        self.calls += 1
        bucket = self.by_agent.setdefault(
            agent, {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "calls": 0}
        )
        bucket["input_tokens"] += input_t
        bucket["output_tokens"] += output_t
        bucket["cost_usd"] += cost
        bucket["calls"] += 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.input_tokens + self.output_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "duration_ms": self.duration_ms,
            "calls": self.calls,
            "by_agent": self.by_agent,
        }


def call_llm(
    *,
    agent_name: str,
    role_key: str,
    messages: list[dict],
    tools: Optional[list[dict]] = None,
    usage: Optional[LLMUsage] = None,
    temperature: float = 0.3,
    max_tokens: int = 1024,
    response_format: Optional[dict] = None,
    extra_body: Optional[dict] = None,
) -> Any:
    """Call the model mapped to *role_key*, with cost tracking and one-shot fallback.

    Args:
        agent_name: label for the usage breakdown (e.g. "router", "judge").
        role_key: key into ``models`` in config (router/answer/vision/judge).
        messages: OpenAI-style messages (content may include image blocks).
        tools: optional tool schema (enables tool_choice="auto").
        usage: optional accumulator; updated in place.
        response_format: e.g. ``{"type": "json_object"}``.
        extra_body: per-call extra body (e.g. ``{"reasoning": {"effort": "high"}}`` for
            extended thinking) MERGED OVER the model's config ``extra_body`` (Block 5).

    Returns:
        The raw OpenAI-SDK chat completion response.
    """
    client = get_client()
    model = config.model_for(role_key)
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    if response_format is not None:
        kwargs["response_format"] = response_format

    # config extra_body (per-model) first, then the per-call extra_body overrides it (Block 5).
    extra = {**(config.extra_body_for(model) or {}), **(extra_body or {})}
    if extra:
        kwargs["extra_body"] = extra
    # Per-request timeout so a stalled upstream connection fails fast (→ fallback) instead of
    # hanging the request forever (caught a 30-min eval hang). Config-driven (no hardcode).
    kwargs["timeout"] = float(config.threshold("llm_timeout_s", 60))

    start = time.time()
    try:
        response = client.chat.completions.create(**kwargs)
    except Exception as exc:
        fallback_model = config.fallback_for(model)
        if not fallback_model:
            raise
        _logger.warning("%s/%s failed (%s) — falling back to %s", agent_name, model, exc, fallback_model)
        kwargs["model"] = fallback_model
        fb_extra = config.extra_body_for(fallback_model)
        if fb_extra is not None:
            kwargs["extra_body"] = {**(kwargs.get("extra_body") or {}), **fb_extra}
        elif "extra_body" in kwargs and extra:
            kwargs.pop("extra_body", None)
        response = client.chat.completions.create(**kwargs)
        model = fallback_model

    duration_ms = int((time.time() - start) * 1000)
    resp_usage = getattr(response, "usage", None)
    if resp_usage is None:
        _logger.warning("%s/%s: response.usage is None — token/cost counted as 0", agent_name, model)
    in_t = resp_usage.prompt_tokens if resp_usage else 0
    out_t = resp_usage.completion_tokens if resp_usage else 0
    cost = config.estimate_cost(model, in_t, out_t)

    if usage is not None:
        usage.add(agent_name, in_t, out_t, cost, duration_ms)

    return response


def call_llm_stream(
    *,
    agent_name: str,
    role_key: str,
    messages: list[dict],
    usage: Optional[LLMUsage] = None,
    temperature: float = 0.3,
    max_tokens: int = 1024,
):
    """Stream the answer model token-by-token. Yields content deltas (str).

    Usage is captured from the final chunk when the provider returns it
    (``stream_options.include_usage``); otherwise it is left at zero.
    """
    client = get_client(wrap=False)  # streaming: raw client (no wrap_openai) to keep the SSE token stream intact
    model = config.model_for(role_key)
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    extra = config.extra_body_for(model)
    if extra:
        kwargs["extra_body"] = dict(extra)
    kwargs["timeout"] = float(config.threshold("llm_timeout_s", 60))

    start = time.time()
    try:
        stream = client.chat.completions.create(**kwargs)
    except Exception as exc:
        fallback_model = config.fallback_for(model)
        if not fallback_model:
            raise
        _logger.warning("%s/%s stream failed (%s) — fallback %s", agent_name, model, exc, fallback_model)
        kwargs["model"] = fallback_model
        kwargs.pop("extra_body", None)
        model = fallback_model
        stream = client.chat.completions.create(**kwargs)

    final_usage = None
    for chunk in stream:
        choices = getattr(chunk, "choices", None) or []
        if choices:
            delta = getattr(choices[0], "delta", None)
            content = getattr(delta, "content", None) if delta else None
            if content:
                yield content
        if getattr(chunk, "usage", None):
            final_usage = chunk.usage

    if usage is not None and final_usage is not None:
        in_t = getattr(final_usage, "prompt_tokens", 0) or 0
        out_t = getattr(final_usage, "completion_tokens", 0) or 0
        usage.add(agent_name, in_t, out_t, config.estimate_cost(model, in_t, out_t),
                  int((time.time() - start) * 1000))


__all__ = ["get_client", "LLMUsage", "call_llm", "call_llm_stream"]
