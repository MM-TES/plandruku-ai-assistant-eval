"""Schedule-action skill: a thin wrapper around the existing module_08 chat planner.

Reuses the 10 schedule tools + mutation guards unchanged — the assistant simply
routes ``schedule_action`` requests here. module_08 itself is not modified.
"""
from __future__ import annotations

from typing import Any

from src.utils.logger import setup_logger

_logger = setup_logger(__name__)


def run_schedule_command(message: str, *, scope: str = "standard", my_schedule: Any = None) -> str:
    """Run a schedule command through the existing planner; return markdown text."""
    try:
        from src.module_08_chat_planner import ChatContext
        from src.web.chat_planner_singleton import _get_chat_planner

        planner = _get_chat_planner()
        old_ctx = planner._context
        planner._context = ChatContext()
        planner._context.scope = scope
        planner._context.my_schedule = my_schedule
        try:
            resp = planner.process_command(message)
            text = resp.text
            extras: list[str] = []
            if getattr(resp, "actions_executed", None):
                extras.append("інструменти: " + ", ".join(resp.actions_executed))
            if getattr(resp, "rebuilt", False):
                extras.append("розклад перебудовано")
            if extras:
                text += f"\n\n_({' · '.join(extras)})_"
            return text
        finally:
            planner._context = old_ctx
    except Exception as exc:  # noqa: BLE001 — never crash the assistant
        _logger.exception("schedule skill failed")
        return (
            "Не вдалося виконати дію з розкладом зараз. Спробуйте через вкладку розкладу "
            f"або сформулюйте інакше. ({type(exc).__name__})"
        )


__all__ = ["run_schedule_command"]
