"""High-level KB search used by the orchestrator escalation.

Cross-lingual: operator queries are uk/ru but the knowledge base is
English-dominant. A Cyrillic query is translated to English (cheap router
model) and BOTH the original and the translation are retrieved + fused, so the
lexical (BM25) half of the hybrid can reach the English PDFs while the
multilingual dense half bridges the languages semantically. Gated by
``kb.cross_lingual``; never raises (escalation is best-effort).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from src.assistant import config
from src.assistant.llm import LLMUsage
from src.assistant.schema import Citation
from src.utils.logger import setup_logger

_logger = setup_logger(__name__)

_CYRILLIC = re.compile(r"[Ѐ-ӿ]")


@dataclass
class KBResult:
    knowledge: str = ""
    citations: list[Citation] = field(default_factory=list)
    best_score: float = 0.0


def _label(chunk) -> str:
    loc = f" · {chunk.locator}" if chunk.locator else ""
    return f"{chunk.source}{loc}"


def _translate_to_english(query: str, usage: Optional[LLMUsage] = None) -> Optional[str]:
    """uk/ru query → concise English search query (best-effort, cheap model)."""
    try:
        from src.assistant.llm import call_llm

        resp = call_llm(
            agent_name="kb_translate",
            role_key="router",
            messages=[
                {"role": "system", "content": config.kb_param("translate_prompt", "")},
                {"role": "user", "content": query},
            ],
            usage=usage,
            temperature=0.0,
            max_tokens=120,
        )
        out = (resp.choices[0].message.content or "").strip()
        return out or None
    except Exception as exc:  # noqa: BLE001 — translation is a best-effort booster
        _logger.info("KB query translation failed (%s) — original query only", exc)
        return None


def _query_variants(query: str, usage: Optional[LLMUsage] = None) -> list[str]:
    """Return [original] plus an English translation when the query is Cyrillic."""
    variants = [query]
    if config.kb_param("cross_lingual", True) and _CYRILLIC.search(query):
        en = _translate_to_english(query, usage)
        if en and en.lower() != query.lower() and en not in variants:
            variants.append(en)
    return variants


def _doc_loc_range(chunks) -> str:
    """A ``<prefix> <min>–<max>`` page label from a document's chunk locators."""
    nums: list[int] = []
    prefix = ""
    for c in chunks:
        loc = c.locator or ""
        m = re.search(r"\d+", loc)
        if m:
            nums.append(int(m.group()))
            if not prefix:
                pm = re.match(r"\s*([^\d]+)", loc)
                prefix = pm.group(1).strip() if pm else ""
    if not nums:
        return chunks[0].locator if chunks and chunks[0].locator else ""
    lo, hi = min(nums), max(nums)
    span = f"{lo}" if lo == hi else f"{lo}–{hi}"
    return f"{prefix} {span}".strip()


def _assemble(hits) -> tuple[str, list[Citation]]:
    """Build the knowledge block + citations from retrieved hits.

    P1.3 (``kb.parent_merge.enabled``): group hits by document (source == parent_id)
    into ONE coherent, locator-ordered block per datasheet with a single page-range
    header and one citation — so the LLM sees the whole spec table, not fragments.
    When off, behaviour is byte-identical to the previous per-chunk listing.
    """
    pm = config.kb_param("parent_merge", {})
    if not (isinstance(pm, dict) and pm.get("enabled", False)):
        knowledge = "\n\n".join(f"[{_label(c)}] {c.text}" for c, _ in hits)
        citations = [
            Citation(source=f"База знань: {_label(c)}", snippet=c.text[:180], url=None)
            for c, _ in hits
        ]
        return knowledge, citations

    max_chars = int(pm.get("max_parent_chars", 6000))
    groups: dict[str, list] = {}
    order: list[str] = []
    for c, _ in hits:
        if c.source not in groups:
            groups[c.source] = []
            order.append(c.source)
        groups[c.source].append(c)
    blocks: list[str] = []
    citations: list[Citation] = []
    for src in order:
        cs = sorted(groups[src], key=lambda c: int(c.id) if str(c.id).isdigit() else 0)
        text = "\n".join(c.text for c in cs)
        if len(text) > max_chars:
            text = text[:max_chars]
        loc = _doc_loc_range(cs)
        header = f"{src}" + (f" · {loc}" if loc else "")
        blocks.append(f"[{header}]\n{text}")
        citations.append(Citation(source=f"База знань: {header}", snippet=cs[0].text[:180], url=None))
    return "\n\n".join(blocks), citations


def search_kb(
    query: str, top_k: int | None = None, *, usage: Optional[LLMUsage] = None,
    extra_variants: list[str] | None = None,
) -> KBResult:
    """Retrieve from the external knowledge base (cross-lingual). Never raises.

    *extra_variants* (INC-6 query planner) are appended to the uk+en variants — extra
    attribute / per-product search strings that strengthen reach to numeric rows.
    """
    try:
        from src.assistant.kb.index import get_kb_retriever

        retr = get_kb_retriever()
        if not retr.available:
            return KBResult()
        variants = _query_variants(query, usage)
        for v in (extra_variants or []):
            if v and v not in variants:
                variants.append(v)
        hits = retr.retrieve_multi(variants, top_k)
        if not hits:
            return KBResult()
        knowledge, citations = _assemble(hits)
        # MAX dense sim across hits (not hits[0]): a code-boosted chunk may be
        # pinned first with a low/zero dense score, but the gate should reflect
        # the strongest match present, not the pinned one.
        best = max((s for _, s in hits), default=0.0)
        return KBResult(knowledge=knowledge, citations=citations, best_score=float(best))
    except Exception:  # noqa: BLE001 — escalation is best-effort
        return KBResult()


__all__ = ["KBResult", "search_kb"]
