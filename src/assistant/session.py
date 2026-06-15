"""Per-session conversational state for the assistant.

Mirrors the chat planner's session pattern but is independent so the old chat
keeps working untouched (rollback safety). The web layer stores an instance on
``SessionState.assistant_session`` and uses the swap-and-restore idiom from
``src/web/routers/chat.py``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.assistant import config


@dataclass
class AssistantSession:
    """Rolling conversation history + last-turn metadata."""

    history: list[dict[str, Any]] = field(default_factory=list)
    last_route: str | None = None
    scope: str = "standard"

    def add_turn(self, role: str, text: str) -> None:
        """Append a turn and trim to the configured history window."""
        self.history.append({"role": role, "text": text})
        max_msgs = int(config.threshold("history_messages", 6)) * 2  # user+assistant pairs
        if len(self.history) > max_msgs:
            self.history = self.history[-max_msgs:]

    def recent(self, limit: int | None = None) -> list[dict[str, Any]]:
        if limit is None:
            return list(self.history)
        return self.history[-limit:]

    def clear(self) -> None:
        self.history.clear()
        self.last_route = None


__all__ = ["AssistantSession"]
