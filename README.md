# PlanDruku AI Assistant — Eval Pipeline + Demo

A production-readiness **eval pipeline** built around an embedded AI operator-assistant in a
printing / flexible-packaging **production-planning system (APS)**. It systematically checks the four
risk classes from the assignment — **PII leakage, prompt injection, hallucinations / faithfulness,
refusal patterns** — and produces a concrete **ship / not-ship verdict**.

> ### ⚖️ Verdict: **CONDITIONAL SHIP** (advisory copilot, operator-in-the-loop)
> Full gate table, numbers, and honest reasoning: **[`eval_report/REPORT.md`](eval_report/REPORT.md)**.
> The report includes an **addendum (§8)** that re-runs all four classes against the *deployed demo*
> configuration (max-quality flags + sanitized DB).

---

## The assistant being evaluated
An embedded **"?" helper** for shop-floor operators: a cheap router (Haiku) → read-only data tools /
RAG over operator-help docs / an external knowledge base (MiniLM + BGE-M3 ensemble, GPU cross-encoder
rerank on Modal T4) → answer synthesis (Sonnet) with a multi-agent critic. It is architecturally
**propose-only** — it never takes autonomous actions; the operator confirms everything.

Code: [`src/assistant/`](src/assistant/) · knowledge it is graded on: [`docs/operator_help/`](docs/operator_help/).

## The eval pipeline (the deliverable)

| Risk class | Tooling | Dataset |
|---|---|---|
| **Prompt injection** | `scripts/red_team_eval.py` (zero-LLM regex guard test) + end-to-end exfil probes | `config/red_team.jsonl` — 63 gated attacks (uk/en/ru) + FP sweep over golden |
| **PII leakage** | `src/assistant/eval/pii.py` (detectors) + `scripts/pii_eval.py` (e2e via `answer()`) | `config/pii_probes.jsonl` — 50 probes / 5 categories |
| **Hallucinations / faithfulness** | `src/assistant/eval/kbdepth/harness.py` — claim-level LLM-judge (dual-judge calibrated, weighted Cohen κ) | `config/eval_subset_prod40.jsonl` |
| **Refusal patterns** | same harness (abstention-correctness + route-accuracy) + PII benign/refusal probes | subset + probes |

- Detector unit tests: [`tests/assistant/test_pii_eval.py`](tests/assistant/test_pii_eval.py)
- Judge calibration: `scripts/run_judge_calibration.py` · gate resolution: `scripts/production_readiness_report.py` → `eval_report/gates_table.md`

## Running it
```bash
pip install -r requirements.txt
```

> **Running the test suite.** `pytest tests/` works on this curated checkout out of the box —
> tests whose system-under-test module, helper script, or golden set was pruned from the subset
> **skip** (never fail). For the *full* suite (incl. the KB loader/chunker tests) also install the
> test-only deps: `pip install -r requirements-dev.txt`.

**(a) Works out of the box** — no database, no KB index, no LLM key:
```bash
# prompt-injection guard — zero-LLM red-team:
python scripts/red_team_eval.py --golden config/kb_golden_300.jsonl --golden config/operator_golden_50.jsonl --label run1
# PII-detector unit tests (47):
python -m pytest tests/assistant/test_pii_eval.py -q
```

**(b) Needs the KB vector index + a live DB** (and `OPENROUTER_API_KEY`) — these drive the real assistant (`orchestrator.answer()`):
```bash
export OPENROUTER_API_KEY=...           # your key, never committed
python scripts/pii_eval.py --label run1
python -m src.assistant.eval.kbdepth.harness --science --answers --judge --golden config/eval_subset_prod40.jsonl --label run1
```
The KB index is built from sources (not shipped — see [`SECURITY.md`](SECURITY.md)). Full recipe + spend: [`eval_report/REPORT.md` §6](eval_report/REPORT.md).

## Live demo
A gated, deployed instance runs on **Fly.io** (max-quality retrieval config, **sanitized** data,
HTTP Basic-Auth gate + per-IP rate-limit). Deployment artifacts are in [`fly_demo/`](fly_demo/) —
including **[`demo_sanitize.sql`](fly_demo/demo_sanitize.sql)**, the data-anonymization step that
produces the demo database (names → `Демо-*`, date-cut `<2025-01-01`). URL + access credentials are
provided to reviewers on request (the demo is not open to the public).

## Security & data handling
See **[`SECURITY.md`](SECURITY.md)**. In short: **no secrets/keys** (env-only), **no customer PII**
(scrubbed; the demo runs on an anonymized DB), **no data dumps**, no production indexes.

---

*This repository is a curated subset of a larger production APS monorepo — only the assistant, its eval
pipeline, and the demo deployment artifacts, with all internal business data removed.*
