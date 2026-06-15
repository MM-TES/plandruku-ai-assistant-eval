"""Patch config files INSIDE the Fly image for the demo (prod files on disk untouched).

Runs at image build (Dockerfile RUN), in the build context /app. Reads env:
  DEMO_TIER         lean|full  -> kb.ensemble.enabled = (tier=="full")
  DEMO_EXTERNAL_KB  true|false -> features.external_kb

Edits:
  config/assistant.json  — demo feature flags
  config/db.json         — database -> "${DB_NAME}" (resolved from the Fly secret at runtime)
  config/ui_server.json  — bind 0.0.0.0:8080 (network_access:true) for the Fly proxy

NOTE: db.json user/password/host/port are already "${ENV}" placeholders resolved by
src/utils/config_loader.py from os.environ; only `database` is a literal -> we point it
at the Fly DB via the DB_NAME secret. config_loader._resolve also resolves ${DB_NAME}.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

CFG = Path("config")
tier = os.environ.get("DEMO_TIER", "lean").lower()
ext_kb = os.environ.get("DEMO_EXTERNAL_KB", "false").lower() == "true"
rerank = os.environ.get("DEMO_RERANK", "false").lower() == "true"  # Modal T4 GPU rerank (needs MODAL_TOKEN_* secrets)
quality = os.environ.get("DEMO_QUALITY", "off").lower() == "max"   # TIER-FULL max-quality answer levers
if quality:
    rerank = True  # quality-max implies GPU rerank on Modal


def _load(name: str) -> dict:
    return json.loads((CFG / name).read_text(encoding="utf-8"))


def _save(name: str, obj: dict) -> None:
    (CFG / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


# --- assistant.json: demo flags (keep the demo config = prod config MINUS these) ---
a = _load("assistant.json")
a["features"]["vision"] = True         # camera/"Екран" screenshot ON — demo is Basic-Auth gated (trusted audience)
a["features"]["feedback"] = False      # no DB writes of feedback
a["features"]["history"] = False       # no query-history accumulation
a["features"]["text2sql"] = False      # never external
a["features"]["tools"] = True          # data_query = live DB brain (keep — main characteristic)
a["features"]["injection_guard"] = True  # red-team passed (keep)
a["features"]["external_kb"] = ext_kb  # default false (brand-leak gate); true ONLY after scrubbing the KB index
a["kb"]["rerank"] = rerank            # GPU rerank on Modal T4 (0 Fly RAM; live path SKIPS on failure -> safe on 2GB)
if rerank:
    a["kb"]["rerank_backend"] = "modal"   # NEVER "local" on a small VM (CPU cross-encoder ~2.3GB -> OOM)
    a["kb"]["rerank_pool"] = 120          # DEEP pool = the measured m2 source-recall uplift (0.72->0.80)
a["kb"]["ensemble"]["enabled"] = (tier == "full")  # BGE-M3 +2.27GB -> FULL tier (>=4GB VM) only

# --- unconditional GUARDS (measured-harmful if on; re-assert regardless of tier/quality) ---
a["kb"]["source_quotas_enabled"] = False                                  # any quota collapses recall@1 with ensemble
a.setdefault("agents", {}).setdefault("controller", {})["sci_full"] = False  # measured -faithfulness 0.62->0.589

# --- DEMO_QUALITY=max: extra answer-quality levers (cost/latency traded for quality) ---
if quality:
    a.setdefault("agents", {}).setdefault("answer_critic", {})["claim_check"] = True   # live grounding gate (+faithfulness)
    a.setdefault("agents", {}).setdefault("controller", {})["reasoning"] = True         # extended thinking
    a["agents"]["controller"]["reasoning_extra_body"] = {"reasoning": {"max_tokens": 1500}}  # max_tokens form (effort form is ignored)
    a["thresholds"]["answer_max_tokens"] = 1536   # fuller answers for the presentation
    a["thresholds"]["llm_timeout_s"] = 90         # room for reasoning + rerank in the serial chain
_save("assistant.json", a)

# --- db.json: point at the Fly DB (DB_NAME secret); everything else stays ${ENV} ---
d = _load("db.json")
d["database"] = "${DB_NAME}"
_save("db.json", d)

# --- ui_server.json: bind 0.0.0.0:8080 (resolve_bind_host returns 0.0.0.0 when network_access=true) ---
_save("ui_server.json", {"host": "127.0.0.1", "port": 8080, "network_access": True})

print(
    f"[patch_demo_config] tier={tier} quality={quality} external_kb={ext_kb} "
    f"ensemble={a['kb']['ensemble']['enabled']} rerank={a['kb']['rerank']} "
    f"rerank_pool={a['kb'].get('rerank_pool')} claim_check={a['agents']['answer_critic'].get('claim_check')} "
    f"reasoning={a['agents']['controller'].get('reasoning')} sci_full={a['agents']['controller'].get('sci_full')} "
    f"quotas={a['kb']['source_quotas_enabled']} -> config patched for Fly demo (8080, db=DB_NAME)"
)
