"""Enumerate + chunk the assistant knowledge corpus.

Sources: ``docs/*.md``, ``CLAUDE.md``, the assistant schema-card, the operator
glossary, and per-page help from ``config/assistant.json``. Live operational
rows are deliberately NOT indexed (they change daily → text2SQL/tools instead).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.assistant import config

ROOT = Path(__file__).resolve().parents[3]
_MAX_CHARS = 900
_MIN_CHARS = 80


@dataclass
class Chunk:
    id: str
    source: str
    url: str | None
    text: str


def _title_of(text: str, fallback: str) -> str:
    """First markdown H1 (``# Title``) → human-friendly citation source."""
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("# "):
            return s[2:].strip()
    return fallback


def _iter_source_texts() -> list[tuple[str, str | None, str]]:
    """Yield ``(source_name, url, raw_text)`` for the OPERATOR knowledge base.

    Indexes only ``docs/operator_help/*.md`` (plain-language guides using the
    exact UI terminology) + per-page help. Developer docs / CLAUDE.md /
    schema-card are deliberately EXCLUDED — they contain internal codes, table
    names and phase labels the operator must never see. ``source_name`` is the
    doc's friendly H1 title, shown as the citation chip.
    """
    out: list[tuple[str, str | None, str]] = []

    help_dir = ROOT / "docs" / "operator_help"
    if help_dir.is_dir():
        for p in sorted(help_dir.glob("*.md")):
            try:
                text = p.read_text(encoding="utf-8")
            except OSError:
                continue
            title = _title_of(text, p.stem)
            out.append((title, str(p.relative_to(ROOT)).replace("\\", "/"), text))

    # per-page help from assistant config (already operator-friendly)
    page_lines: list[str] = []
    for key, page in config.pages().items():
        desc = page.get("desc", "")
        prompts = "; ".join(page.get("prompts", []))
        page_lines.append(f"Сторінка {key}: {desc} Приклади запитів: {prompts}")
    if page_lines:
        out.append(("Довідка по сторінці", None, "\n".join(page_lines)))

    return out


def _chunk_text(text: str) -> list[str]:
    """Greedily pack paragraphs into <= _MAX_CHARS chunks."""
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    buf = ""
    for para in paras:
        if len(para) > _MAX_CHARS:
            if buf:
                chunks.append(buf)
                buf = ""
            # hard-split a very long paragraph
            for i in range(0, len(para), _MAX_CHARS):
                chunks.append(para[i:i + _MAX_CHARS])
            continue
        if len(buf) + len(para) + 2 <= _MAX_CHARS:
            buf = f"{buf}\n\n{para}" if buf else para
        else:
            chunks.append(buf)
            buf = para
    if buf:
        chunks.append(buf)
    return [c for c in chunks if len(c) >= _MIN_CHARS or len(chunks) == 1]


def build_chunks() -> list[Chunk]:
    """Return the full list of indexable chunks with provenance metadata."""
    chunks: list[Chunk] = []
    for source, url, text in _iter_source_texts():
        for n, body in enumerate(_chunk_text(text)):
            chunks.append(Chunk(id=f"{source}#{n}", source=source, url=url, text=body))
    return chunks


__all__ = ["Chunk", "build_chunks"]
