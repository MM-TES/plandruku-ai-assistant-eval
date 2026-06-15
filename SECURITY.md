# Security & data handling

This public repository is a **curated, scrubbed subset** of a private production system, prepared for
review. Sensitive material is excluded or sanitized as follows.

## No secrets / keys
- **No API keys, tokens, or passwords are committed.** The application reads them from environment
  variables only: `OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY`, the `DB_*` connection vars,
  `MODAL_TOKEN_ID`/`MODAL_TOKEN_SECRET`, `LANGSMITH_API_KEY`.
- `.env`, `config/ui_server.json`, and `~/.modal.toml` are git-ignored and not present in this repo.
- The deployed demo's secrets are injected at runtime via `flyctl secrets` and are **never baked into
  the image** (the build context excludes `.env`).

## No customer / personal data
- The production system stores **no contact PII** — there are no phone numbers, emails, or addresses in
  the schema, only customer *names*.
- All real customer **names, order IDs, and contacts** have been scrubbed from this repo — in the report
  examples, the detector test fixtures, and the probe inputs — and replaced with `Демо-клієнт N`,
  generic IDs, and `example.com` addresses.
- The **deployed demo** runs against a separate **sanitized database** (`aps_printing_demo`): customer
  names anonymized to `Демо-*` and records date-cut to `< 2025-01-01`. The anonymization is reproducible
  from [`fly_demo/demo_sanitize.sql`](fly_demo/demo_sanitize.sql).
- No database dumps, audit reports, or production data files are included.

## Public technical references (kept by design)
The knowledge-base golden datasets (`config/kb_golden_*.jsonl`) and a few config comments reference
**public material / equipment vendors** (e.g. Plastchim, BOPP-films, Erhardt-Leimer) whose **public
datasheets** form the technical corpus the assistant is graded on. These are public industry references
— not confidential supplier relationships and not customer data — and are retained so the eval datasets
remain meaningful.

## Excluded heavy / regenerable artifacts
- KB vector indexes (`models/knowledge_base_rag*`) — excluded (size + they embed the vendor corpus);
  rebuilt from sources via the build scripts.
- `reports/` (run outputs / audits that contained real data), `catboost_info/`, and all `__pycache__`.
- This repo has **no git history from the source monorepo** — it is a fresh, clean tree.

## The assistant's measured safety posture
From [`eval_report/REPORT.md`](eval_report/REPORT.md): prompt injection is fully blocked (1.0 block-rate,
0 FP, 10/10 exfil attempts blocked before any LLM spend); customer-PII exfiltration is impossible
(no contacts in the DB, 0 invented, no mass dumps); the architecture is propose-only; and the deployed
demo additionally enforces HTTP Basic-Auth gating + per-IP rate-limiting.
