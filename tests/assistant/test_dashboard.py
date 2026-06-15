"""Extra C — quality/drift dashboard rendering (pure, offline)."""
from __future__ import annotations

from src.assistant.eval import dashboard as D


def _rows():
    return [
        {"label": "baseline", "mean_source_recall": "0.711", "mean_recall_at_1": "0.368",
         "mean_mrr": "0.461", "mean_faithfulness": "0.40", "total_cost_usd": "0.50"},
        {"label": "rerank_on", "mean_source_recall": "0.840", "mean_recall_at_1": "0.560",
         "mean_mrr": "0.640", "mean_faithfulness": "0.55", "total_cost_usd": "0.62"},
    ]


def test_render_contains_sections_and_trend():
    md = D.render_markdown(_rows(), [])
    assert "Quality & Drift Dashboard" in md
    assert "trend" in md and "drift" in md
    assert "baseline" in md and "rerank_on" in md
    assert "recall@1" in md and "MRR" in md


def test_drift_direction_and_sign():
    md = D.render_markdown(_rows(), [])
    # source-recall improved 0.711 → 0.840 (higher is better) → ▲ positive ✓
    assert "▲ +0.129 ✓" in md
    # recall@1 improved 0.368 → 0.560
    assert "+0.192" in md


def test_drift_needs_two_runs():
    md = D.render_markdown(_rows()[:1], [])
    assert "need ≥2 runs" in md


def test_calibration_and_feedback_sections():
    calib = {"judge_a_model": "A", "judge_b_model": "B",
             "agreement": {"overall": {"spearman": 0.82, "n": 12}},
             "weak_criteria": ["useful"], "recommendation": "Add anchors for: useful"}
    fb = {"up": 7, "down": 2, "total": 9, "by_route": {"instructions": {"up": 5, "down": 1}}}
    md = D.render_markdown(_rows(), [], calib=calib, feedback=fb)
    assert "Spearman" in md and "0.820" in md and "useful" in md
    assert "👍 7" in md and "instructions" in md


def test_build_writes_file(tmp_path):
    import csv
    sci = tmp_path / "sci.csv"
    with sci.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["label", "mean_source_recall", "mean_mrr", "total_cost_usd"])
        w.writeheader()
        w.writerow({"label": "b", "mean_source_recall": "0.7", "mean_mrr": "0.5", "total_cost_usd": "0.1"})
    out = D.build(sci_history=sci, kb_history=tmp_path / "none.csv", out=tmp_path / "dash.md")
    assert out.is_file() and "Quality & Drift Dashboard" in out.read_text(encoding="utf-8")
