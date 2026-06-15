"""Golden-set loader + spec-number matcher for KB datasheet-depth eval.

The PRIMARY metric (numeric context-recall) is computed locally with NO LLM: for a
product query, what fraction of the datasheet's known spec numbers appear in the
retrieved context.

The matcher intentionally differs from ``evaluators._NUMBER_RE`` (which is tuned for
localized THOUSANDS separators in free-text answers, and therefore splits a bare
4-digit spec like ``2200`` into ``220`` + ``0``). Datasheet spec values are plain
integers/decimals without grouping, so we match ``\\d+(?:[.,]\\d+)?`` whole and
compare as a normalized set (decimal comma → dot, so the Ukrainian ``67,6`` and the
English ``67.6`` unify).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_GOLDEN = _ROOT / "config" / "kb_depth_golden.jsonl"

# Spec numbers: integer or decimal (dot OR comma), no thousands grouping.
_SPEC_NUM = re.compile(r"\d+(?:[.,]\d+)?")


def normalize_num(tok: str) -> str:
    """Canonical numeric form: strip, unify decimal comma→dot. ``'67,6'`` → ``'67.6'``."""
    return (tok or "").strip().replace(",", ".")


def numbers_in(text: str) -> set[str]:
    """All spec numbers present in *text*, normalized for set membership."""
    return {normalize_num(m) for m in _SPEC_NUM.findall(text or "")}


@dataclass
class GoldenItem:
    product: str = ""                 # datasheet code OR science item id
    source: str = ""                  # KB-relative path, e.g. sites/plastchim.ua/pdf/datasheet/FXCMT.pdf
    queries: list[str] = field(default_factory=list)  # [uk, en, …] — used directly (no translator → deterministic)
    numbers: list[str] = field(default_factory=list)  # known spec values (datasheet items)
    min_recall: float = 0.8
    material: str = ""
    lang: str = ""
    raw: dict = field(default_factory=dict)
    # ── P0 science-eval extensions (all defaulted → the datasheet jsonl still loads) ──
    kind: str = "datasheet"           # "datasheet" | "science"
    route_expected: str = ""          # expected router route (e.g. "instructions", "out_of_scope")
    abstain_expected: bool = False    # the CORRECT behaviour is to abstain (data genuinely absent)
    source_paths: list[str] = field(default_factory=list)  # distinctive relpath fragments (substring match)
    reference_answer: str = ""        # technologist reference (depth/faithfulness anchor)
    key_claims: list[str] = field(default_factory=list)    # atomic claims a good answer should ground
    category: str = ""                # anilox | doctor_blade | lamination | film | patent | standard | tradeoff …

    @property
    def golden_set(self) -> set[str]:
        return {normalize_num(n) for n in self.numbers}

    @property
    def expected_sources(self) -> list[str]:
        """Distinctive relpath fragments expected in retrieval. Science items use
        ``source_paths`` (a list, any-of); datasheet items fall back to the single
        ``source`` — so both kinds share one accessor for source-recall/citation-PR."""
        return list(self.source_paths) or ([self.source] if self.source else [])


def load_golden(path: str | Path = DEFAULT_GOLDEN) -> list[GoldenItem]:
    """Load the datasheet-depth golden set (jsonl). Lines starting with ``//`` skipped."""
    p = Path(path)
    out: list[GoldenItem] = []
    if not p.is_file():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        d = json.loads(line)
        nums = [str(n["value"]) if isinstance(n, dict) else str(n) for n in d.get("numbers", [])]
        out.append(GoldenItem(
            product=d.get("product") or d.get("id", ""), source=d.get("source", ""),
            queries=list(d.get("queries") or []),
            numbers=nums, min_recall=float(d.get("min_recall", 0.8)),
            material=d.get("material", ""), lang=d.get("lang", ""), raw=d,
            kind=d.get("kind", "datasheet"), route_expected=d.get("route_expected", ""),
            abstain_expected=bool(d.get("abstain_expected", False)),
            source_paths=list(d.get("source_paths") or []),
            reference_answer=d.get("reference_answer", ""),
            key_claims=list(d.get("key_claims") or []), category=d.get("category", ""),
        ))
    return out


def recall(golden: set[str], context: str) -> tuple[float, set[str], set[str]]:
    """(recall, found, missing) of *golden* numbers within *context*."""
    present = numbers_in(context)
    found = {g for g in golden if g in present}
    missing = golden - found
    r = len(found) / len(golden) if golden else 1.0
    return r, found, missing


__all__ = ["GoldenItem", "load_golden", "numbers_in", "normalize_num", "recall", "DEFAULT_GOLDEN"]
