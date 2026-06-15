# PlanDruku chat-assistant DEMO image for Fly.io.
# NO local Docker needed — built by the Fly REMOTE builder (`flyctl deploy`).
# Build context = repo root; heavy/secret paths pruned by repo-root .dockerignore
# (copy fly_demo/dockerignore -> .dockerignore before deploy).
# TIER-LEAN default (MiniLM-only). For FULL/max-quality: build args DEMO_TIER=full DEMO_QUALITY=max
# DEMO_RERANK=true DEMO_EXTERNAL_KB=true, AND remove the models/knowledge_base_rag_bge line from
# .dockerignore so the ensemble index ships (the BGE-M3 model is baked automatically when DEMO_TIER=full).
FROM python:3.11-slim AS base

# libgomp1 = OpenMP runtime required by faiss-cpu and torch on the slim image.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 app
WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/app/.cache/huggingface

# Build args — declared here so they are in scope for BOTH the conditional BGE bake (step 3) and the patch (step 5).
ARG DEMO_TIER=lean
ARG DEMO_EXTERNAL_KB=false
ARG DEMO_RERANK=false
ARG DEMO_QUALITY=off

# 1) CPU-only torch FIRST. CRITICAL: the default torch wheel pulls ~2 GB of CUDA,
#    useless on a shared-cpu VM and blows the image (hw10 lesson: 1.75 GB -> 8.3 GB).
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 2) Slim RUNTIME deps only (NOT the full requirements.txt — no catboost/lightgbm/ocr/build deps).
COPY fly_demo/requirements-fly.txt ./requirements-fly.txt
RUN pip install --no-cache-dir -r requirements-fly.txt

# 3) Pre-bake embedders so the first query isn't a cold download. MiniLM always; BGE-M3 only for FULL.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')"
RUN if [ "$DEMO_TIER" = "full" ]; then \
      python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"; \
    fi

# 4) App code + config + the MiniLM/operator-help indexes (heavy model dirs pruned by .dockerignore).
COPY main.py ./main.py
COPY src/ ./src/
COPY templates/ ./templates/
COPY static/ ./static/
COPY config/ ./config/
COPY models/ ./models/
COPY fly_demo/patch_demo_config.py ./patch_demo_config.py

# 5) Patch config IN THE IMAGE ONLY (prod files on disk are untouched): demo feature flags + quality levers,
#    db.json database -> DB_NAME secret (resolved at runtime), ui_server.json -> 0.0.0.0:8080.
RUN DEMO_TIER=$DEMO_TIER DEMO_EXTERNAL_KB=$DEMO_EXTERNAL_KB DEMO_RERANK=$DEMO_RERANK DEMO_QUALITY=$DEMO_QUALITY \
    python patch_demo_config.py

RUN chown -R app:app /app /home/app
USER app
EXPOSE 8080
CMD ["python", "main.py", "--mode", "ui"]
