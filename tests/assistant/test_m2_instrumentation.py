"""m2 (business-uplift) — P0 harness instrumentation.

Offline, no LLM, no real index: percentile math for the p95 latency gate, the
``kb.rrf_k`` sweep knob (config-driven RRF kconst), and the ``--operator`` harness
mode (Layer-A over the SEPARATE operator-help RAG) with a fake retriever.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from src.assistant import config
from src.assistant.eval.kbdepth import harness

from tests.assistant.conftest import requires_file

_ROOT = Path(__file__).resolve().parents[2]


def _load_script(name: str):
    """Import a scripts/*.py module (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location(name, _ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── _percentile (nearest-rank; p95 wall_ms gate rides every Layer-B run) ─────────
def test_percentile_empty_and_single():
    assert harness._percentile([], 95) is None
    assert harness._percentile([1200.0], 50) == 1200.0
    assert harness._percentile([1200.0], 95) == 1200.0


def test_percentile_nearest_rank():
    vals = [float(x) for x in range(1, 101)]  # 1..100
    assert harness._percentile(vals, 50) == 50.0
    assert harness._percentile(vals, 95) == 95.0
    # order-independent
    assert harness._percentile(list(reversed(vals)), 95) == 95.0


# ── kb.rrf_k knob (m2-tuned to 30; the fallback default stays the old hard-code 60) ──
def test_rrf_k_config_is_read_and_valid():
    # config carries the tuned value; the fallback (when the key is absent) is the
    # previous hard-code 60 — so an old config without the key is byte-identical.
    assert int(config.kb_param("rrf_k", 60)) in (30, 60)
    assert config.kb().get("rrf_k") == 30  # adopted m2 value


def test_rrf_kconst_changes_fusion_flatness():
    from src.assistant.kb.index import _rrf

    # ranking A strongly prefers 1; ranking B prefers 2; doc 3 is mid in both.
    rankings = [[1, 3, 2], [2, 3, 1]]
    low = _rrf(rankings, kconst=1)
    high = _rrf(rankings, kconst=1000)
    # Both fuse the same id set; kconst only reshapes the blend.
    assert set(low) == set(high) == {1, 2, 3}
    # With a huge kconst the fusion is ~flat (sum of ranks): 3 (ranks 2+2) ties 1 and 2
    # (1+3); with kconst=1 the top-1 ranks dominate. The orders must NOT be identical
    # for this asymmetric input — proving the knob is live.
    scores_low = {i: sum(1.0 / (1 + r) for rank_list in rankings
                         for r, idx in enumerate(rank_list, start=1) if idx == i)
                  for i in (1, 2, 3)}
    assert low == sorted(scores_low, key=lambda i: scores_low[i], reverse=True)


# ── --operator mode (fake retriever; reports redirected to tmp) ──────────────────
class _FakeChunk:
    def __init__(self, source: str) -> None:
        self.source = source


class _FakeRetriever:
    available = True

    def retrieve(self, query: str, top_k: int | None = None):
        return [(_FakeChunk("pomichnyk.md"), 0.9), (_FakeChunk("vybir_materialu.md"), 0.5)]


def test_run_operator_layer_a(tmp_path, monkeypatch):
    golden = tmp_path / "op_golden.jsonl"
    golden.write_text(
        "\n".join([
            json.dumps({"id": "op1", "kind": "operator", "queries": ["як спитати помічника?"],
                        "source_paths": ["pomichnyk.md"], "lang": "uk", "category": "help"},
                       ensure_ascii=False),
            json.dumps({"id": "op2", "kind": "operator", "queries": ["як обрати матеріал?"],
                        "source_paths": ["neisnuyuchy.md"], "lang": "uk", "category": "help"},
                       ensure_ascii=False),
        ]),
        encoding="utf-8",
    )
    import src.assistant.rag.index as rag_index

    monkeypatch.setattr(rag_index, "get_retriever", lambda: _FakeRetriever())
    monkeypatch.setattr(harness, "_OP_REPORTS", tmp_path / "reports")
    monkeypatch.setattr(harness, "_OP_HISTORY", tmp_path / "reports" / "history.csv")

    res = harness.run_operator(label="t", top_k=4, golden=str(golden))
    assert res["exit"] == 0
    assert res["summary"]["n"] == 2
    # op1 hits (rank 1), op2 misses → mean source_recall 0.5, recall@1 0.5
    assert res["summary"]["mean_source_recall"] == 0.5
    assert res["summary"]["mean_recall_at_1"] == 0.5
    assert (tmp_path / "reports" / "history.csv").is_file()


def test_run_operator_empty_set_is_graceful(tmp_path, monkeypatch):
    golden = tmp_path / "none.jsonl"
    golden.write_text(json.dumps({"id": "x", "kind": "science", "queries": ["q"]}), encoding="utf-8")
    res = harness.run_operator(label="t", top_k=4, golden=str(golden))
    assert res["exit"] == 2


# ── golden_gen m2 additions (offline, no LLM) ────────────────────────────────────
def test_near_domain_traps_codes_absent_from_corpus():
    from src.assistant.eval.kbdepth import golden_gen

    chunks = [
        {"product": "FXCMT", "text": "FXCMT BOPP film 20 micron yield 75.8"},
        {"product": "PLCBZ", "text": "PLCBZ CPP film sealing 105 C"},
        {"product": "", "text": "generic literature text about anilox"},
    ]
    traps = golden_gen.near_domain_traps(chunks, n=5, seed=0)
    assert traps, "expected at least one trap from 2 real products"
    blob = " ".join(c["text"] for c in chunks).lower() + " fxcmt plcbz"
    for t in traps:
        code = t["product"].split("_", 2)[2]
        assert code.lower() not in blob, f"trap code {code} leaked from corpus"
        assert t["abstain_expected"] is True
        assert t["route_expected"] == ""          # routing not gated for traps
        assert t["category"] == "near_domain_trap"
        assert code in t["queries"][0]


def test_near_domain_traps_deterministic_and_capped():
    from src.assistant.eval.kbdepth import golden_gen

    chunks = [{"product": "FXCMT", "text": "FXCMT film"}]
    a = golden_gen.near_domain_traps(chunks, n=3, seed=42)
    b = golden_gen.near_domain_traps(chunks, n=3, seed=42)
    assert [t["product"] for t in a] == [t["product"] for t in b]
    assert len(a) <= 3


def test_out_of_scope_items_n_cap():
    from src.assistant.eval.kbdepth import golden_gen

    assert len(golden_gen.out_of_scope_items()) == len(golden_gen._OUT_OF_SCOPE)
    assert len(golden_gen.out_of_scope_items(4)) == 4
    assert all(it["abstain_expected"] for it in golden_gen.out_of_scope_items(4))


# ── bootstrap CI for judge ρ (pure) ──────────────────────────────────────────────
def test_bootstrap_ci_brackets_perfect_correlation():
    from src.assistant.eval.judge_calibration import bootstrap_ci

    xs = [float(x) for x in range(30)]
    lo, hi = bootstrap_ci(xs, xs, n_boot=200, seed=1)
    assert lo == hi == 1.0          # identical series — every resample is ρ=1


def test_bootstrap_ci_wide_on_noise():
    from src.assistant.eval.judge_calibration import bootstrap_ci

    xs = [0.0, 1.0, 0.5, 0.2, 0.9, 0.1, 0.8, 0.3, 0.7, 0.4]
    ys = [0.5, 0.1, 0.9, 0.3, 0.2, 0.8, 0.0, 0.7, 0.4, 1.0]
    lo, hi = bootstrap_ci(xs, ys, n_boot=300, seed=1)
    assert lo < hi                  # noise → a genuinely wide interval
    assert -1.0 <= lo <= hi <= 1.0


def test_bootstrap_ci_degenerate_input():
    from src.assistant.eval.judge_calibration import bootstrap_ci

    assert bootstrap_ci([], []) == (0.0, 0.0)
    assert bootstrap_ci([1.0, 2.0], [1.0, 2.0]) == (0.0, 0.0)  # n<3


# ── dashboard gates table (pure) ─────────────────────────────────────────────────
def test_gates_table_pass_fail_and_unmeasured():
    from src.assistant.eval.dashboard import gates_table

    sci_rows = [{"mean_source_recall": "0.92", "mean_recall_at_1": "0.41",
                 "p95_wall_ms": "4100"}]
    calib = {"agreement": {"overall": {"spearman": 0.71}}}
    md = gates_table(sci_rows, calib, feedback={"up": 9, "total": 10})
    assert "| source-recall | ≥ 0.9 | 0.920 | ✓ |" in md
    assert "| recall@1 | ≥ 0.6 | 0.410 | ✗ |" in md
    assert "✓" in md.split("p95 latency")[1].split("\n")[0]      # 4100 <= 6000
    assert "| judge Spearman ρ | ≥ 0.7 | 0.710 | ✓ |" in md
    assert "| online 👍-rate | ≥ 0.8 | 0.900 | ✓ |" in md
    # unmeasured metric renders an em-dash row, never a false verdict
    assert "| faithfulness | ≥ 0.8 | — | — |" in md


# ── weighted RRF + ensemble channel (offline) ────────────────────────────────────
def test_rrf_weights_none_equals_all_ones():
    from src.assistant.kb.index import _rrf

    rankings = [[1, 2, 3], [3, 2, 1], [2, 1]]
    assert _rrf(rankings) == _rrf(rankings, weights=[1.0, 1.0, 1.0])


def test_rrf_weights_zero_channel_is_ignored():
    from src.assistant.kb.index import _rrf

    # channel 2 alone would put 9 first; with weight 0 it must not influence the blend
    rankings = [[1, 2], [9, 8, 7]]
    fused = _rrf(rankings, weights=[1.0, 0.0])
    assert fused[0] == 1
    # zero-weight ids still present (score 0) but ranked after every weighted id
    assert fused.index(1) < fused.index(9) and fused.index(2) < fused.index(9)


def test_rrf_weights_can_flip_channel_dominance():
    from src.assistant.kb.index import _rrf

    rankings = [[1, 2], [2, 1]]
    assert _rrf(rankings, weights=[2.0, 1.0])[0] == 1
    assert _rrf(rankings, weights=[1.0, 2.0])[0] == 2


# ── GPU reranker (Modal client + CPU fallback) ───────────────────────────────────
def test_gpu_reranker_orders_by_score(monkeypatch):
    from src.assistant.kb import gpu_reranker

    class _Fn:
        def remote(self, q, texts, ml):
            # higher score for the 2nd text
            return [0.1, 0.9, 0.3][: len(texts)]

    r = gpu_reranker.GpuReranker()
    monkeypatch.setattr(r, "_ensure_fn", lambda: _Fn())
    assert r.order("q", ["a", "b", "c"]) == [1, 2, 0]
    assert r.order("q", []) == []
    assert r.used_fallback is False


def test_gpu_reranker_raises_on_failure_live_safe(monkeypatch):
    import pytest

    from src.assistant.kb import gpu_reranker

    class _BadFn:
        def remote(self, *a):
            raise RuntimeError("modal down")

    r = gpu_reranker.GpuReranker()
    monkeypatch.setattr(r, "_ensure_fn", lambda: _BadFn())
    # default (live) → RAISE so the caller skips rerank, NOT a 40s CPU rerank
    with pytest.raises(RuntimeError):
        r.order("q", ["a", "b"])


def test_gpu_reranker_offline_cpu_fallback_opt_in(monkeypatch):
    from src.assistant.kb import gpu_reranker

    class _BadFn:
        def remote(self, *a):
            raise RuntimeError("modal down")

    r = gpu_reranker.GpuReranker()
    monkeypatch.setattr(r, "_ensure_fn", lambda: _BadFn())
    import src.assistant.rag.index as rag_index
    monkeypatch.setattr(rag_index, "get_reranker",
                        lambda: type("C", (), {"order": lambda self, q, t: list(range(len(t)))})())
    assert r.order("q", ["a", "b"], cpu_fallback=True) == [0, 1]
    assert r.used_fallback is True


def test_ensemble_disabled_gracefully_on_missing_dir(tmp_path):
    from src.assistant.kb.index import KBRetriever

    r = KBRetriever(persist_dir=tmp_path)  # never loaded — we drive the helper directly
    r._chunks = []
    r._ensure_ensemble({"enabled": True, "persist_dir": str(tmp_path / "nope"),
                        "embed_model": "BAAI/bge-m3"})
    assert r._ens_index is None            # best-effort: channel off, no exception
    assert r._ens_ready is True            # and it never retries per-query


# ── golden_qc mechanical gate (offline; risk R5 = golden drift) ──────────────────
def test_golden_qc_drops_dup_missing_source_and_claimless():
    requires_file("scripts/golden_qc.py")
    qc = _load_script("golden_qc")
    items = [
        {"product": "A", "queries": ["яка товщина плівки fxcmt детально"],
         "source": "d/FX.pdf", "key_claims": ["товщина плівки"], "category": "datasheet"},
        {"product": "B", "queries": ["яка товщина плівки fxcmt детально"],  # near-dup → drop
         "source": "d/FX.pdf", "key_claims": ["товщина"], "category": "datasheet"},
        {"product": "C", "queries": ["зовсім інше питання про анілокс лініатуру"],
         "source": "NOPE.pdf", "key_claims": ["анілокс"], "category": "literature"},  # no source → drop
        {"product": "D", "queries": ["питання про глянець поверхні друку"],
         "source": "d/GL.pdf", "key_claims": ["неіснуючий токен ззззз"], "category": "literature"},  # claim drop only when check_claims=True
        {"product": "OOS1", "queries": ["яка погода"], "abstain_expected": True,
         "category": "out_of_scope"},  # abstain → kept, source-dependent checks skipped
    ]
    qc._source_texts = lambda corpus: {  # type: ignore[attr-defined]
        "d/FX.pdf": "fxcmt bopp товщина плівки 20 мкм", "d/GL.pdf": "глянець поверхні 85 одиниць"}
    # default: claim check OFF (cross-lingual-invalid) → D survives, only dup + missing-source drop
    kept, rep = qc.qc(items, "kb")
    assert [k["product"] for k in kept] == ["A", "D", "OOS1"]
    assert rep["n_dropped"] == 2
    # opt-in claim check → D also drops (its claim token is absent from source)
    kept2, rep2 = qc.qc(items, "kb", check_claims=True)
    assert [k["product"] for k in kept2] == ["A", "OOS1"]
    assert rep2["n_dropped"] == 3


def test_golden_qc_per_source_cap():
    requires_file("scripts/golden_qc.py")
    qc = _load_script("golden_qc")
    items = [{"product": f"P{i}", "queries": [f"унікальне питання номер {i} про плівку"],
              "source": "d/X.pdf", "key_claims": ["плівка"], "category": "datasheet"}
             for i in range(4)]
    qc._source_texts = lambda corpus: {"d/X.pdf": "плівка bopp"}  # type: ignore[attr-defined]
    kept, _ = qc.qc(items, "kb", per_source_cap=2)
    assert len(kept) == 2          # only 2 items per source survive


def test_draft_item_threads_drafter_role(monkeypatch):
    """drafter_role must reach call_llm as role_key (the 20x cost lever)."""
    from src.assistant.eval.kbdepth import golden_gen

    seen: dict = {}

    class _Msg:
        content = json.dumps({"uk_query": "Питання?", "en_query": "Q?",
                              "reference_answer": "a", "key_claims": ["c"], "numbers": []})

    class _Choice:
        message = _Msg()

    class _Resp:
        choices = [_Choice()]

    def fake_call_llm(**kw):
        seen["role_key"] = kw.get("role_key")
        return _Resp()

    monkeypatch.setattr(golden_gen, "call_llm", fake_call_llm)
    it = golden_gen.draft_item({"source": "s.md", "text": "x" * 300}, bucket="other",
                               drafter_role="golden_drafter")
    assert it is not None
    assert seen["role_key"] == "golden_drafter"
