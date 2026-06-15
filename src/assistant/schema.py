"""Pydantic request/response contracts for the assistant."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

_VALID_SCOPES = {"standard", "my", "all"}
# Answer mode (operator-selectable toggle):
#   hybrid — system data first, then escalate to RAG/KB (default, current behaviour)
#   data   — only verified system data (tools/DB); RAG/KB disabled
#   kb     — only the knowledge base (RAG); system data/tools disabled
_VALID_MODES = {"hybrid", "data", "kb"}


class PageContext(BaseModel):
    """Structured "what's on screen" context, injected on every request."""

    route: str = "/"
    stage: str | None = None
    section: str | None = None
    selected_order: str | None = None
    visible_text: str | None = None
    # Server-authoritative summary of the active schedule (which plan is active,
    # order count, horizon). Set by the web layer, never by the client — so the
    # assistant reports the real plan state, not the pending-orders count.
    plan_context: str | None = None
    visible_entity_ids: list[str] = Field(default_factory=list)
    filters: dict[str, Any] = Field(default_factory=dict)

    def key(self) -> str:
        """Page-registry key: ``route`` or ``route:stage`` (used for descriptions)."""
        if self.stage:
            return f"{self.route}:{self.stage}"
        return self.route

    def focus_order_id(self) -> int | None:
        """The order the operator is looking at: ``selected_order`` (open card),
        else first numeric visible id, else a single-order screen filter
        (``?order_id_like=12345``). ``None`` when none applies.

        The order_id_like fallback matters on stage tables that filter to one
        order but don't carry per-row ``data-order-id`` (e.g. Виробництво).
        """
        raw: Any = self.selected_order
        if raw is None and self.visible_entity_ids:
            raw = self.visible_entity_ids[0]
        if raw is None:
            oil = (self.filters or {}).get("order_id_like")
            if isinstance(oil, str) and oil.isdigit() and 4 <= len(oil) <= 7:
                raw = oil
        try:
            return int(str(raw))
        except (TypeError, ValueError):
            return None


class AssistantRequest(BaseModel):
    message: str
    page_context: PageContext = Field(default_factory=PageContext)
    scope: str = "standard"
    mode: str = "hybrid"
    # Recent conversation turns ([{role, text}, …], oldest→newest), used to rewrite
    # a follow-up into a standalone question before routing/retrieval.
    history: list[dict[str, Any]] = Field(default_factory=list)
    screenshot_b64: str | None = None
    session_id: str = "default"

    def normalised_scope(self) -> str:
        return self.scope if self.scope in _VALID_SCOPES else "standard"

    def normalised_mode(self) -> str:
        return self.mode if self.mode in _VALID_MODES else "hybrid"


class Citation(BaseModel):
    source: str
    snippet: str = ""
    url: str | None = None


class AssistantResponse(BaseModel):
    text_md: str
    route: str
    citations: list[Citation] = Field(default_factory=list)
    tool_trace: list[dict] = Field(default_factory=list)
    usage: dict = Field(default_factory=dict)
    feedback_id: str | None = None
    clarify: bool = False
    error: str | None = None
    # All evidence the answer was grounded on (page-context + tool rows + RAG),
    # used by the groundedness evaluator and for debugging.
    evidence: str = ""


__all__ = ["PageContext", "AssistantRequest", "Citation", "AssistantResponse"]
