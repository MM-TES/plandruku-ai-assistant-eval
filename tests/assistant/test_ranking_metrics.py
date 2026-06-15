"""Block 3 — rank-aware retrieval metrics (recall@k / MRR / nDCG).

Pure-logic, offline. Relevance is binary and uses the SAME case-insensitive substring
matcher as ``source_recall`` (an expected fragment present in a retrieved source path),
so these metrics and ``source_recall`` always agree on what counts as relevant.
Ported from ai-engineering_HW/lesson-09-rag-systems-enterprise/.../template/metrics.py.
"""
from __future__ import annotations

import math

from src.assistant.eval import evaluators as E

# A retrieval where the relevant doc (a plastchim FXCMT datasheet) lands at rank 3.
_SOURCES = [
    "sites/www.flexfilm.com/pdf/Products/BX100.pdf",   # 1 — irrelevant
    "Література/anilox_white_ink.pdf",                  # 2 — irrelevant
    "sites/plastchim.ua/pdf/datasheet/FXCMT.pdf",       # 3 — RELEVANT
    "sites/plastchim.ua/pdf/datasheet/FXC.pdf",         # 4
]
_EXPECTED = ["datasheet/FXCMT.pdf"]


# ── first-relevant-rank (shared matcher) ─────────────────────────────────────────
def test_first_relevant_rank_position():
    assert E._first_relevant_rank(_SOURCES, _EXPECTED) == 3
    assert E._first_relevant_rank(_SOURCES, ["FXCW"]) is None         # not retrieved
    assert E._first_relevant_rank(_SOURCES, []) is None               # nothing expected


def test_first_relevant_rank_case_insensitive():
    assert E._first_relevant_rank(["SITES/PLASTCHIM.UA/FXCMT.PDF"], ["fxcmt.pdf"]) == 1


def test_matcher_mirrors_source_recall():
    # Whatever source_recall calls "matched", the rank matcher must also find.
    sr = E.source_recall(_SOURCES, _EXPECTED)
    assert sr["score"] == 1.0
    assert E._first_relevant_rank(_SOURCES, _EXPECTED) is not None
    # And vice-versa for a miss.
    assert E.source_recall(_SOURCES, ["nope"])["score"] == 0.0
    assert E._first_relevant_rank(_SOURCES, ["nope"]) is None


# ── recall@k ─────────────────────────────────────────────────────────────────────
def test_recall_at_k_threshold():
    assert E.recall_at_k(_SOURCES, _EXPECTED, 1)["score"] == 0.0   # rank 3 not in top-1
    assert E.recall_at_k(_SOURCES, _EXPECTED, 2)["score"] == 0.0   # not in top-2
    assert E.recall_at_k(_SOURCES, _EXPECTED, 3)["score"] == 1.0   # exactly at top-3
    assert E.recall_at_k(_SOURCES, _EXPECTED, 10)["score"] == 1.0


def test_recall_at_k_none_when_no_expected():
    assert E.recall_at_k(_SOURCES, [], 5)["score"] is None


# ── MRR ──────────────────────────────────────────────────────────────────────────
def test_mrr_reciprocal_rank():
    assert E.mrr(_SOURCES, _EXPECTED, 10)["score"] == round(1 / 3, 3)
    # relevant at rank 2 → 0.5
    assert E.mrr(["x", "datasheet/FXCMT.pdf"], _EXPECTED, 10)["score"] == 0.5
    # relevant at rank 1 → 1.0
    assert E.mrr(["datasheet/FXCMT.pdf"], _EXPECTED, 10)["score"] == 1.0
    # not found → 0.0; beyond k → 0.0
    assert E.mrr(_SOURCES, ["FXCW"], 10)["score"] == 0.0
    assert E.mrr(_SOURCES, _EXPECTED, 2)["score"] == 0.0
    assert E.mrr(_SOURCES, [], 10)["score"] is None


# ── nDCG@k ───────────────────────────────────────────────────────────────────────
def test_ndcg_single_relevant():
    # rank 1 → 1/log2(2) = 1.0
    assert E.ndcg_at_k(["datasheet/FXCMT.pdf"], _EXPECTED, 10)["score"] == 1.0
    # rank 3 → 1/log2(4) = 0.5
    assert E.ndcg_at_k(_SOURCES, _EXPECTED, 10)["score"] == round(1 / math.log2(4), 3)
    # beyond k → 0.0; not found → 0.0; no expected → None
    assert E.ndcg_at_k(_SOURCES, _EXPECTED, 2)["score"] == 0.0
    assert E.ndcg_at_k(_SOURCES, ["FXCW"], 10)["score"] == 0.0
    assert E.ndcg_at_k(_SOURCES, [], 10)["score"] is None


# ── bundle (what the harness inserts into rows + history.csv) ────────────────────
def test_ranking_metrics_bundle_keys_and_values():
    b = E.ranking_metrics(_SOURCES, _EXPECTED, ks=(1, 5, 10))
    assert set(b) == {"recall_at_1", "recall_at_5", "recall_at_10", "mrr", "ndcg"}
    assert b["recall_at_1"] == 0.0 and b["recall_at_5"] == 1.0 and b["recall_at_10"] == 1.0
    assert b["mrr"] == round(1 / 3, 3)
    assert b["ndcg"] == round(1 / math.log2(4), 3)


def test_ranking_metrics_bundle_all_none_when_no_expected():
    b = E.ranking_metrics(_SOURCES, [], ks=(1, 5, 10))
    assert all(v is None for v in b.values())
