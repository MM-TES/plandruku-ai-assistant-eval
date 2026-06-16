"""INC-1/INC-2 tests: KBChunk metadata derivation + product/parent matching (offline)."""
from __future__ import annotations

from src.assistant.kb import corpus
from src.assistant.kb.index import _product_candidates, _product_keys

from tests.assistant.conftest import requires_import


# ── path → metadata (P1.2) ──────────────────────────────────────────────────────
def test_chunk_meta_datasheet_product():
    m = corpus._chunk_meta("sites/plastchim.ua/pdf/datasheet/FXCMT.pdf")
    assert m == {"product": "FXCMT", "doc_type": "datasheet", "supplier": "plastchim.ua"}


def test_chunk_meta_underscore_code_product():
    m = corpus._chunk_meta("sites/plastchim.ua/pdf/datasheet/TATRAFAN_SHT.pdf")
    assert m["product"] == "TATRAFAN_SHT" and m["doc_type"] == "datasheet"


def test_chunk_meta_products_dir_is_datasheet():
    m = corpus._chunk_meta("sites/www.flexfilm.com/pdf/Products/BX100.pdf")
    assert m["doc_type"] == "datasheet" and m["product"] == "BX100"
    assert m["supplier"] == "www.flexfilm.com"


def test_chunk_meta_page_has_no_product():
    m = corpus._chunk_meta("sites/plastchim.ua/pages/bopp-films.md")
    assert m["product"] is None and m["doc_type"] == "page" and m["supplier"] == "plastchim.ua"


def test_chunk_meta_scanned_and_literature():
    assert corpus._chunk_meta("scanned/Operator Manual.pdf")["doc_type"] == "scanned"
    lit = corpus._chunk_meta("Література/flexo_handbook.pdf")
    assert lit["doc_type"] == "literature" and lit["product"] is None


def test_looks_like_code():
    assert corpus._looks_like_code("FXCMT")
    assert corpus._looks_like_code("TATRAFAN_SHT")
    assert not corpus._looks_like_code("ab")            # too short
    assert not corpus._looks_like_code("bopp films")    # has space


# ── product matching (P1.2 scope) ───────────────────────────────────────────────
def test_product_keys():
    assert _product_keys("TATRAFAN_SHT") == {"tatrafan_sht", "tatrafansht"}
    assert _product_keys("FXCMT") == {"fxcmt"}


def test_product_candidates_single_code():
    assert "fxcmt" in _product_candidates(["усі технічні характеристики FXCMT"])


def test_product_candidates_hyphenated_code():
    # flexfilm codes are hyphenated; the candidate set must match the product key
    # (a plain split would drop the F- prefix -> HSP and miss it).
    cands = _product_candidates(["усі технічні характеристики плівки F-HSP"])
    assert _product_keys("F-HSP") & cands       # {"f-hsp","fhsp"} intersects


def test_chunk_meta_flexfilm_nested_products():
    m = corpus._chunk_meta("sites/www.flexfilm.com/pdf/Products/BOPET/F-HSP.pdf")
    assert m["doc_type"] == "datasheet" and m["product"] == "F-HSP"
    assert m["supplier"] == "www.flexfilm.com"


def test_product_candidates_joins_adjacent_but_not_bare_prefix():
    cands = _product_candidates(["характеристики TATRAFAN SHT плівки"])
    # joined form matches the full product key…
    assert "tatrafansht" in cands or "tatrafan_sht" in cands
    keys = _product_keys("TATRAFAN_SHT")
    assert keys & cands                       # at least one key matches
    # …but the bare prefix must not equal the full key (avoids pulling TATRAFAN_*)
    assert "tatrafan" not in keys


# ── metadata migration (no re-embed) ─────────────────────────────────────────────
def test_migrate_kb_metadata_adds_fields(tmp_path):
    requires_import("scripts.migrate_kb_metadata")
    import json

    from scripts.migrate_kb_metadata import migrate

    chunks = [
        {"id": "0", "source": "sites/plastchim.ua/pdf/datasheet/FXCMT.pdf",
         "path": "x", "locator": "стор. 1", "text": "a"},
        {"id": "1", "source": "sites/plastchim.ua/pages/bopp.md",
         "path": "y", "locator": "", "text": "b"},
    ]
    (tmp_path / "kb_chunks.json").write_text(json.dumps(chunks), encoding="utf-8")
    out = migrate(tmp_path)
    assert out["ok"] and out["n"] == 2 and out["with_product"] == 1
    migrated = json.loads((tmp_path / "kb_chunks.json").read_text(encoding="utf-8"))
    assert migrated[0]["product"] == "FXCMT" and migrated[0]["parent_id"] == migrated[0]["source"]
    assert migrated[1]["product"] is None and migrated[1]["doc_type"] == "page"
    assert (tmp_path / "kb_chunks.json.premeta.bak").exists()   # backup written
