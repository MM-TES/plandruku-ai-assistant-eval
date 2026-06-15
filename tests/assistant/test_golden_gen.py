"""Block 1 — KB golden generator: stratified sampling, number verification, round-trip.

Offline: no real KB / no API. The LLM draft is scripted via patch_llm_client, and the
sampler runs on a tiny synthetic chunk list. Proves the anti-hallucinated-golden
contract (a fabricated number is dropped) and that emitted JSONL loads back into
well-formed GoldenItems via the existing harness loader.
"""
from __future__ import annotations

import json

from src.assistant.eval.kbdepth import golden as G
from src.assistant.eval.kbdepth import golden_gen as GG


def _chunks() -> list[dict]:
    long = "x" * 250  # padding so the chunk passes the _MIN_CHUNK_CHARS gate
    out: list[dict] = []
    for i in range(5):
        out.append({"id": f"d{i}", "source": f"sites/plastchim.ua/pdf/datasheet/P{i}.pdf",
                    "text": f"yield 75.8 m2/kg OTR 2200 tensile 140 240 haze 62 COF 0.27 {long}",
                    "doc_type": "datasheet", "product": f"P{i}", "parent_id": f"sites/plastchim.ua/pdf/datasheet/P{i}.pdf"})
    for i in range(20):
        out.append({"id": f"l{i}", "source": f"Література/lit_{i}.pdf", "text": f"anilox line screen {long}",
                    "doc_type": "literature", "product": None})
    for i in range(8):
        out.append({"id": f"s{i}", "source": f"scanned/patent_{i}.pdf", "text": f"peel strength {long}",
                    "doc_type": "scanned", "product": None})
    for i in range(4):
        out.append({"id": f"p{i}", "source": f"sites/x/page_{i}.md", "text": f"catalog {long}",
                    "doc_type": "page", "product": None})
    for i in range(10):
        out.append({"id": f"o{i}", "source": f"WORD_MD/rec_{i}.md", "text": f"key value {long}",
                    "doc_type": None, "product": None})
    out.append({"id": "thin", "source": "Література/thin.pdf", "text": "too short", "doc_type": "literature"})
    return out


# ── stratified sampling ──────────────────────────────────────────────────────────
def test_stratified_sample_buckets_and_determinism():
    chunks = _chunks()
    s1 = GG.stratified_sample(chunks, seed=0)
    s2 = GG.stratified_sample(chunks, seed=0)
    assert set(s1) == set(GG.DEFAULT_PLAN)
    # datasheets are grouped per-parent (5 distinct sources)
    assert len(s1["datasheet"]) == 5
    # buckets present; thin (<200 chars) literature chunk excluded
    assert all(len(c["text"]) >= GG._MIN_CHUNK_CHARS for cs in s1.values() for c in cs)
    # deterministic for the same seed
    assert [c["id"] for c in s1["literature"]] == [c["id"] for c in s2["literature"]]


def test_parent_text_concatenates_one_source():
    chunks = [
        {"source": "d.pdf", "locator": "p2", "text": "BBB"},
        {"source": "d.pdf", "locator": "p1", "text": "AAA"},
        {"source": "other.pdf", "locator": "p1", "text": "ZZZ"},
    ]
    blob = GG.parent_text(chunks, "d.pdf")
    assert "AAA" in blob and "BBB" in blob and "ZZZ" not in blob
    assert blob.index("AAA") < blob.index("BBB")  # locator-ordered


# ── grounded drafting + mechanical number verification ───────────────────────────
def test_draft_drops_fabricated_numbers(patch_llm_client):
    # The model returns a real number (75.8, present) and a fabricated one (999, absent).
    scripted = json.dumps({
        "uk_query": "які характеристики плівки?", "en_query": "film specs?",
        "reference_answer": "вихід 75.8 m2/kg", "key_claims": ["вихід 75.8"],
        "numbers": [{"value": "75.8", "attr": "yield", "unit": "m2/kg"},
                    {"value": "999", "attr": "fake", "unit": ""}],
    })
    patch_llm_client([scripted])
    chunk = {"source": "Література/x.pdf", "text": "yield 75.8 m2/kg measured", "doc_type": "literature"}
    item = GG.draft_item(chunk, bucket="literature")
    assert item is not None
    vals = [n["value"] for n in item.get("numbers", [])]
    assert "75.8" in vals and "999" not in vals          # fabricated number dropped
    assert item["source_paths"] == ["Література/x.pdf"]   # source pinned from the chunk
    assert item["kind"] == "science" and item["route_expected"] == "instructions"


def test_datasheet_kind_requires_enough_verified_numbers(patch_llm_client):
    eight = [{"value": v, "attr": "a", "unit": "u"} for v in
             ["75.8", "2200", "140", "240", "62", "0.27", "56.8", "26.4"]]
    scripted = json.dumps({"uk_query": "повні характеристики P0", "en_query": "P0 full specs",
                           "reference_answer": "...", "key_claims": ["a"], "numbers": eight})
    patch_llm_client([scripted])
    text = "yield 75.8 56.8 26.4 OTR 2200 tensile 140 240 haze 62 COF 0.27"
    chunk = {"source": "sites/plastchim.ua/pdf/datasheet/P0.pdf", "text": text,
             "doc_type": "datasheet", "product": "P0"}
    item = GG.draft_item(chunk, bucket="datasheet", all_chunks=[chunk])
    assert item is not None and item["kind"] == "datasheet"
    assert len(item["numbers"]) >= GG._MIN_DATASHEET_NUMBERS


def test_draft_returns_none_on_bad_json(patch_llm_client):
    patch_llm_client(["not json at all"])
    chunk = {"source": "x.pdf", "text": "y" * 250, "doc_type": "literature"}
    assert GG.draft_item(chunk, bucket="literature") is None


# ── out-of-scope + JSONL round-trip through the harness loader ───────────────────
def test_out_of_scope_items_abstain():
    oos = GG.out_of_scope_items()
    assert len(oos) == len(GG._OUT_OF_SCOPE) >= 4   # m2 expanded the fixed off-domain pool
    assert GG.out_of_scope_items(4) == oos[:4]      # n-cap selects a prefix
    assert all(o["abstain_expected"] and o["route_expected"] == "out_of_scope" and not o["source_paths"]
               for o in oos)


def test_emit_roundtrips_through_golden_loader(tmp_path, patch_llm_client):
    items = [
        {"product": "P0", "kind": "science", "category": "literature", "lang": "uk",
         "source": "Література/a.pdf", "source_paths": ["Література/a.pdf"],
         "queries": ["питання?", "question?"], "key_claims": ["факт"], "reference_answer": "відповідь",
         "route_expected": "instructions", "abstain_expected": False, "min_recall": 0.8,
         "numbers": [{"value": "75.8", "attr": "yield", "unit": "m2/kg"}]},
    ] + GG.out_of_scope_items()
    out = GG.emit_jsonl(items, tmp_path / "g.jsonl")
    loaded = G.load_golden(out)  # // lines (header + REVIEW) must be skipped
    assert len(loaded) == len(items)
    first = loaded[0]
    assert first.kind == "science" and first.source_paths == ["Література/a.pdf"]
    assert "75.8" in first.golden_set
    assert any(it.abstain_expected for it in loaded)
