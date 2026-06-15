"""LangSmith tracing helper + robust JSON parsing.

``traceable`` is a real-or-no-op decorator: when LANGSMITH_TRACING is off it is
a transparent pass-through (no langsmith import, no network); when on it lazily
wraps the function with ``langsmith.traceable``. The decision is made per call
(not at import time) so tests can force it off via env.
"""
from __future__ import annotations

import functools
import json
import re
from typing import Any, Callable

from src.assistant import config


def _tracing_enabled() -> bool:
    return config.langsmith_tracing_enabled()


def traceable(*dargs: Any, **dkwargs: Any) -> Any:
    """Decorator usable bare (``@traceable``) or parametrised (``@traceable(name=...)``)."""

    def decorator(fn: Callable) -> Callable:
        wrapped_cache: dict[str, Callable] = {}

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not _tracing_enabled():
                return fn(*args, **kwargs)
            ls = wrapped_cache.get("fn")
            if ls is None:
                try:
                    from langsmith import traceable as _ls_traceable

                    ls = _ls_traceable(*dargs, **dkwargs)(fn)
                except Exception:
                    ls = fn
                wrapped_cache["fn"] = ls
            return ls(*args, **kwargs)

        return wrapper

    # bare @traceable
    if dargs and callable(dargs[0]) and not dkwargs:
        fn = dargs[0]
        return decorator(fn)
    return decorator


def parse_json_object(text: str) -> dict[str, Any]:
    """Robustly extract the first JSON object from a model response."""
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return {}


__all__ = ["traceable", "parse_json_object"]
