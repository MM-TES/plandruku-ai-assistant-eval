"""Maps a PageContext to its human description + suggested prompts (config-driven)."""
from __future__ import annotations

from src.assistant import config
from src.assistant.schema import PageContext

_FALLBACK = {"desc": "", "prompts": []}


def describe(page_context: PageContext) -> dict:
    """Return ``{desc, prompts}`` for the page, falling back route → default."""
    pages = config.pages()
    return (
        pages.get(page_context.key())
        or pages.get(page_context.route)
        or pages.get("default", _FALLBACK)
    )


def suggested_prompts(page_context: PageContext) -> list[str]:
    return list(describe(page_context).get("prompts", []))


def page_description(page_context: PageContext) -> str:
    return describe(page_context).get("desc", "")


__all__ = ["describe", "suggested_prompts", "page_description"]
