# fly_demo/ — Fly.io chat-assistant demo artifacts

Ready-to-deploy artifacts for publishing a sanitized chat-assistant demo of PlanDruku on Fly.io.
**No local Docker needed** — `flyctl deploy` builds in Fly's remote builder.

**Full handoff / step-by-step:** `R_and_D/external_access_deployment_rnd/FLY_RUNBOOK.md`
**Rationale / sizing / tiers:** `R_and_D/external_access_deployment_rnd/FLY_IO_ASSISTANT_DEMO.md`

| File | Purpose |
|---|---|
| `Dockerfile` | demo image (CPU-only torch, slim deps, bakes MiniLM, patches config in-image) |
| `fly.toml` | Fly config — TIER-LEAN (shared-cpu-1x/2GB, region fra, scale-to-zero, health `/ai-showcase`) |
| `dockerignore` | **copy to repo-root `.dockerignore`** before deploy (`Copy-Item fly_demo\dockerignore .dockerignore -Force`) |
| `requirements-fly.txt` | slim runtime pip set (torch installed separately, CPU wheel) |
| `patch_demo_config.py` | runs at build: demo feature flags + `db.json` database→`${DB_NAME}` + `ui_server.json`→0.0.0.0:8080 |

Deploy from repo root: `flyctl deploy --config fly_demo/fly.toml --remote-only`
(after building the sanitized Fly Postgres — see FLY_RUNBOOK.md F2–F5).

These artifacts patch config **inside the image only** — the prod working-tree config files are not modified.
