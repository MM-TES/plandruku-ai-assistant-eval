"""Prompt-injection / jailbreak detection on input + output.

Ported from ai-engineering_HW/lesson-10-api-layer-ai-systems/.../app/security/injection.py
and extended for this assistant:

* the original English jailbreak regexes are kept and **Ukrainian + Russian** patterns
  are added (operators type uk/ru), all anchored to injection-like phrasing so benign
  shop queries («яка лініатура анілокса для білила?») are not blocked;
* ``MAX_INPUT_LENGTH`` is read from ``config.threshold`` (project rule #1 — no hardcode);
* ``OUTPUT_LEAK_INDICATORS`` are built at call time from distinctive openings of THIS
  app's own system prompts (``config.prompt(...)``), so a leak is detected against the
  real prompts, not the course's 12-Factor strings.

Scope (honest limitation): the input regex catches *plain-text* override / role-hijack /
system-leak / disregard attempts. Obfuscated payloads (base64, rot13, leetspeak) are NOT
matched at input — they are caught defensively at output (system-prompt-leak detection)
and by the instruction-hierarchy clause hardened into the system prompts. This is the
defense-in-depth posture the course's EXP-07 measured (``blocked_at_input`` vs
``defended_at_output``).
"""
from __future__ import annotations

import re

from src.assistant import config

_DEFAULT_MAX_INPUT_LENGTH = 4000

# ── jailbreak / override / leak request patterns (input side) ────────────────────
# English (course-ported) + Ukrainian + Russian. Compiled case-insensitively.
_INJECTION_PATTERNS: list[re.Pattern[str]] = [re.compile(p, re.IGNORECASE) for p in (
    # — English —
    # allow 1-3 stacked modifiers between the verb and the noun ("ignore ALL PRIOR rules")
    r"ignore\s+(?:(?:previous|all|above|prior|any|the)\s+){1,3}(instructions|rules|prompts?|guidelines)",
    r"forget\s+(?:(?:all|prior|previous|the|any)\s+){1,3}(rules|instructions|guidelines)",
    r"disregard\s+.{0,40}(rules|instructions|prompt|guidelines)",
    r"override\s+.{0,30}(instructions|rules|prompt)",
    r"(^|\s)(system|admin)\s*:\s*",
    r"<\|im_(start|end)\|>",
    r"</?s>",
    # "you are now AN unrestricted AI" — handle a/an + optional filler before the trait
    r"you\s+are\s+(now\s+)?(an?\s+)?(dan|jailbroken|unrestricted|gpt-unrestricted)",
    r"(developer|dev)\s+mode",
    r"(reveal|show|repeat|print|output|quote|list)\s+.{0,40}"
    r"(system\s+(prompt|message|section|instructions|rules|role|text)|"
    r"your\s+(instructions|rules|guidelines)|hidden\s+(prompt|rules|instructions)|"
    r"initialization\s+text)",
    # — Ukrainian —
    r"ігнор\w*\s+(попередн\w+|вс[іе]\s|усі\s|всі\s|будь-?як).{0,30}"
    r"(інструкц|правил|вказівк|настанов|промпт)",
    r"знехт\w+\s+.{0,30}(інструкц|правил|вказівк|промпт|настанов)",
    r"забудь\s+(усі\s+|вс[іе]\s+|попередн\w+\s+)?(правил|інструкц|вказівк|обмеженн|настанов)",
    r"(^|\s)(система|систем|адміністратор|адмін)\s*:\s*",
    r"ти\s+(тепер|відтепер)\s+.{0,30}(dan|без\s+обмеж|вільн|інш\w+\s+(асистент|режим)|"
    r"не\s+помічник)",
    r"(покажи|виведи|надрукуй|розкрий|повтори|процитуй|дай)\s+.{0,40}"
    r"(системн\w+\s+(промпт|підказк|інструкц|повідомл)|свій\s+промпт|свої\s+(приховані\s+)?інструкц|"
    r"початков\w+\s+інструкц)",
    r"(дій|працюй|поводься|відповідай)\s+(як|в\s+ролі)\s+.{0,25}"
    r"(хакер|зловмисник|dan|без\s+цензур|без\s+обмеж)",
    r"режим\s+розробник",
    # — Russian —
    r"игнор\w*\s+(предыдущ\w+|все\s|всех\s|любы).{0,30}(инструкц|правил|указани|промпт)",
    r"забудь\s+(все\s+)?(правил|инструкц|указани|ограничен)",
    r"(^|\s)(система|систем|администратор|админ)\s*:\s*",
    r"ты\s+(теперь|отныне)\s+.{0,30}(dan|без\s+огранич|свободн|не\s+помощник)",
    r"(покажи|выведи|раскрой|повтори|процитируй)\s+.{0,40}"
    r"(системн\w+\s+(промпт|подсказк|инструкц|сообщени)|свой\s+промпт|свои\s+(скрыт\w+\s+)?инструкц)",
    r"режим\s+разработчик",
)]


def _max_input_length() -> int:
    try:
        return int(config.threshold("max_input_chars", _DEFAULT_MAX_INPUT_LENGTH))
    except Exception:  # noqa: BLE001 — config may be absent in unit context
        return _DEFAULT_MAX_INPUT_LENGTH


def check_input(message: str) -> tuple[bool, str | None]:
    """Return ``(clean, reason_if_blocked)``.

    ``clean`` is False when the message exceeds the length cap or matches a known
    injection pattern; ``reason`` names the trigger (for the trace / logs)."""
    msg = message or ""
    cap = _max_input_length()
    if len(msg) > cap:
        return False, f"input_too_long ({len(msg)} > {cap})"
    for pat in _INJECTION_PATTERNS:
        if pat.search(msg):
            return False, f"injection_pattern: {pat.pattern[:48]}"
    return True, None


# ── output-side system-prompt-leak detection ─────────────────────────────────────
def _leak_indicators() -> list[str]:
    """Distinctive opening fragments of THIS assistant's own system prompts. If the
    model echoes ≥2 of them verbatim, it is almost certainly leaking its instructions."""
    frags: list[str] = []
    for key in ("answer", "router", "answer_kb", "answer_sci_full", "vision", "text2sql"):
        try:
            p = (config.prompt(key) or "").strip()
        except Exception:  # noqa: BLE001
            p = ""
        if p:
            opening = p.splitlines()[0][:60].strip()
            if len(opening) >= 14:
                frags.append(opening)
    return frags


def check_output(response: str, indicators: list[str] | None = None) -> bool:
    """True iff the response looks like a system-prompt leak (≥2 distinctive prompt
    openings echoed verbatim). ``indicators`` may be injected for testing."""
    text = response or ""
    inds = indicators if indicators is not None else _leak_indicators()
    matches = sum(1 for ind in inds if ind and ind in text)
    return matches >= 2


__all__ = ["check_input", "check_output"]
