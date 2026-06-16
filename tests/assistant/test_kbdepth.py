"""Offline tests for the KB datasheet-depth eval (INC-0).

Pure-logic only: the spec-number matcher, recall, golden-set integrity, and the
`only_paths` ingest filter. No embedder / FAISS / API — fast and deterministic.
"""
from __future__ import annotations

from pathlib import Path

from src.assistant.eval.kbdepth import golden as G

from tests.assistant.conftest import skip_if_empty


# ── spec-number matcher ─────────────────────────────────────────────────────────
def test_matcher_keeps_bare_4digit_whole():
    # The whole point: 2200 must NOT split into 220+0 (the evaluators._NUMBER_RE bug
    # this matcher deliberately avoids).
    nums = G.numbers_in("OTR 2200 2100 1300 cc/m2/day")
    assert {"2200", "2100", "1300"} <= nums
    assert "220" not in nums


def test_matcher_decimals_dot_and_comma():
    assert G.numbers_in("yield 75.8 m2/kg") >= {"75.8"}
    # Ukrainian comma decimal unifies with dot form.
    assert G.numbers_in("вихід 67,6 м2/кг") >= {"67.6"}
    assert G.normalize_num("67,6") == "67.6"
    assert G.normalize_num("  140 ") == "140"


def test_recall_found_missing():
    golden = {"75.8", "2200", "0.27", "240"}
    r, found, missing = G.recall(golden, "context yield 75.8 OTR 2200 tensile 240")
    assert found == {"75.8", "2200", "240"}
    assert missing == {"0.27"}
    assert r == 0.75


def test_recall_empty_golden_is_one():
    r, _, _ = G.recall(set(), "anything")
    assert r == 1.0


# ── golden-set integrity ────────────────────────────────────────────────────────
def test_golden_loads_varied_products():
    items = G.load_golden()
    skip_if_empty(items, "kbdepth golden set")
    assert len(items) >= 6
    by = {it.product: it for it in items}
    # six base datasheets + a comparison item
    for code in ("FXCMT", "FXC", "FXCW", "PLC", "PNRA21P", "TATRAFAN_SHT"):
        assert code in by, code
    # The reference datasheet carries its distinctive numbers.
    assert {"75.8", "2200"} <= by["FXCMT"].golden_set
    # Cross-lingual coverage: at least one Ukrainian datasheet.
    assert any(it.lang == "uk" for it in items)
    # A comparison item spanning two products.
    assert any(it.product.startswith("CMP_") for it in items)


def test_golden_every_item_well_formed():
    for it in G.load_golden():
        assert it.source.endswith(".pdf") and it.source.startswith("sites/")
        assert len(it.queries) >= 2          # uk + en, used without a translator
        assert len(it.golden_set) >= 8       # enough numbers for a meaningful 0.8 gate
        assert 0.0 < it.min_recall <= 1.0


# ── only_paths ingest filter (no langchain/embedder needed) ─────────────────────
def test_only_paths_restricts_iter_files(tmp_path: Path):
    from src.assistant.kb import corpus

    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    (tmp_path / "b.txt").write_text("bravo", encoding="utf-8")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "c.txt").write_text("charlie", encoding="utf-8")

    all_files = list(corpus.iter_files(tmp_path, include_doc=False, ocr=False))
    assert len(all_files) == 3

    only = list(corpus.iter_files(
        tmp_path, include_doc=False, ocr=False, only_paths=[tmp_path / "a.txt", sub / "c.txt"]))
    names = sorted(p.name for p in only)
    assert names == ["a.txt", "c.txt"]
