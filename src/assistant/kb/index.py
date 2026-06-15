"""FAISS HNSW + BM25 hybrid index over the knowledge-base chunks.

Build (offline, one-time) persists a FAISS index + chunk metadata to
``models/knowledge_base_rag/``. Query loads them once (singleton retriever) and
does dense (FAISS) + BM25 -> RRF, returning chunks with a relevance score used
by the orchestrator's escalation gate.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from src.assistant import config
from src.assistant.kb.corpus import KBChunk
from src.assistant.kb.embedder import KBEmbedder, get_kb_embedder
from src.utils.logger import setup_logger

_logger = setup_logger(__name__)

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PERSIST_DIR = ROOT / "models" / "knowledge_base_rag"
_FAISS_FILE = "kb.faiss"
_CHUNKS_FILE = "kb_chunks.json"
_META_FILE = "kb_meta.json"


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", (text or "").lower(), flags=re.UNICODE)


def _rrf(rankings: list[list[int]], kconst: int = 60,
         weights: list[float] | None = None) -> list[int]:
    """Reciprocal-rank fusion. m2: optional per-ranking ``weights`` (weighted RRF — a
    tuned convex blend beats plain RRF given a tuning set, arxiv 2210.11934). ``None``
    or all-1.0 weights reproduce the classic formula exactly."""
    scores: dict[int, float] = {}
    for ri, ranking in enumerate(rankings):
        w = weights[ri] if weights is not None and ri < len(weights) else 1.0
        for rank, idx in enumerate(ranking, start=1):
            scores[idx] = scores.get(idx, 0.0) + w / (kconst + rank)
    return sorted(scores, key=lambda i: scores[i], reverse=True)


_ONTOLOGY: dict | None = None


def _load_ontology() -> dict:
    """Load the retrieval ontology concepts once (best-effort, cached)."""
    global _ONTOLOGY
    if _ONTOLOGY is None:
        try:
            path = ROOT / "config" / "i18n" / "retrieval_ontology.json"
            _ONTOLOGY = json.loads(path.read_text(encoding="utf-8")).get("concepts", {})
        except Exception:  # noqa: BLE001 — ontology is an optional booster
            _ONTOLOGY = {}
    return _ONTOLOGY


def _expand_ontology(variants: list[str]) -> list[str]:
    """T1.5 (OG-RAG-lite): append one extra search variant of English/alt domain terms
    for every ontology concept whose trigger appears in the query, so BM25/dense reach
    the English-dominant corpus deterministically (no LLM). Key-free recall booster."""
    onto = _load_ontology()
    if not onto:
        return variants
    blob = " ".join(variants).lower()
    seen: set[str] = set()
    extra: list[str] = []
    for concept in onto.values():
        if any(str(t).lower() in blob for t in concept.get("triggers", [])):
            for term in concept.get("expand", []):
                low = str(term).lower()
                if low not in blob and low not in seen:
                    extra.append(str(term))
                    seen.add(low)
    if not extra:
        return variants
    # Enrich the LAST (English) variant in place rather than adding a competing variant:
    # an extra RRF ranking can DEMOTE a previously-good hit, whereas enriching only
    # broadens the lexical reach of the existing English query. (measured: net-positive)
    return variants[:-1] + [f"{variants[-1]} {' '.join(extra)}".strip()]


def _query_concepts(variants: list[str]) -> list[str]:
    """Ontology concept names whose trigger appears in the query (T1.3 topic scope)."""
    onto = _load_ontology()
    if not onto:
        return []
    blob = " ".join(variants).lower()
    return [name for name, c in onto.items()
            if any(str(t).lower() in blob for t in c.get("triggers", []))]


def _multi_query_expand(variants: list[str]) -> list[str]:
    """m2 (kb.multi_query, default OFF): ONE cheap router-model call generates up to two
    paraphrase/HyDE-style variants of the (last, usually English) query, appended as
    extra RRF rankings. Lives INSIDE retrieve_multi so the Layer-A harness measures it.
    Best-effort: any failure returns the variants unchanged."""
    try:
        from src.assistant.llm import call_llm
        from src.assistant.tracing import parse_json_object

        resp = call_llm(
            agent_name="kb_multi_query", role_key="router",
            messages=[
                {"role": "system", "content": config.prompt("kb_multi_query")},
                {"role": "user", "content": variants[-1]},
            ],
            temperature=0.7, max_tokens=200,
            response_format={"type": "json_object"},
        )
        d = parse_json_object(resp.choices[0].message.content or "{}")
        extra = [str(v).strip() for v in (d.get("variants") or []) if str(v).strip()]
        if extra:
            return variants + extra[:2]
    except Exception as exc:  # noqa: BLE001 — expansion is an optional booster
        _logger.info("multi_query expansion skipped (%s)", str(exc)[:100])
    return variants


_CODE_RE = re.compile(r"[A-Za-z][A-Za-z0-9]{2,}")


def _code_tokens(queries: list[str]) -> list[str]:
    """Latin UPPERCASE alnum tokens (len>=3) from the raw queries — likely
    product codes (``FXCMT``, ``PLCBZ``; ``B-TLL`` yields ``TLL``), lowercased to
    match the BM25 vocabulary. Generic type names (BOPP/BOPET) match here too but
    are filtered later by rarity — they hit too many chunks to be discriminative."""
    out: list[str] = []
    seen: set[str] = set()
    for q in queries:
        for tok in _CODE_RE.findall(q or ""):
            if tok.isascii() and tok.isupper() and any(ch.isalpha() for ch in tok):
                low = tok.lower()
                if low not in seen:
                    seen.add(low)
                    out.append(low)
    return out


def _product_keys(product: str) -> set[str]:
    """Normalized lookup keys for a product code: lowercased + alnum-only."""
    p = product.lower()
    return {p, re.sub(r"[^a-z0-9]", "", p)}


_HYPHEN_CODE_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)+")


def _product_candidates(variants: list[str]) -> set[str]:
    """Candidate product keys from query code tokens: each UPPERCASE token plus
    adjacent pairs joined (so «TATRAFAN SHT» → ``tatrafansht`` matches TATRAFAN_SHT,
    while a bare ``tatrafan`` does NOT pull every TATRAFAN_* datasheet). Hyphenated
    codes (``F-HSP``, ``B-THF``) are added whole + alnum-compact, since a plain split
    would drop the one-letter prefix (``F-HSP`` → ``HSP``) and miss the product key."""
    cands: set[str] = set()
    for q in variants:
        toks = [t.lower() for t in _CODE_RE.findall(q or "")
                if t.isascii() and t.isupper() and any(c.isalpha() for c in t)]
        for t in toks:
            cands.add(t)
        for a, b in zip(toks, toks[1:]):
            cands.add(a + b)
            cands.add(f"{a}_{b}")
        for h in _HYPHEN_CODE_RE.findall(q or ""):
            if h.isascii() and h.upper() == h and any(c.isalpha() for c in h):
                low = h.lower()
                cands.add(low)
                cands.add(re.sub(r"[^a-z0-9]", "", low))
    return cands


def build_index(
    chunks: list[KBChunk],
    persist_dir: Path | str = DEFAULT_PERSIST_DIR,
    *,
    embedder: KBEmbedder | None = None,
    show_progress: bool = True,
) -> int:
    """Embed chunks, build a FAISS HNSW index, persist everything. Returns count."""
    persist_dir = Path(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)
    if not chunks:
        _logger.warning("KB corpus empty — nothing indexed")
        return 0

    emb_model = embedder or get_kb_embedder()
    emb = emb_model.encode([c.text for c in chunks], show_progress=show_progress)
    dim = int(emb.shape[1])

    import faiss

    m = int(config.kb_param("faiss_m", 32))
    ef_c = int(config.kb_param("faiss_ef_construction", 200))
    ef_s = int(config.kb_param("faiss_ef_search", 64))
    index = faiss.IndexHNSWFlat(dim, m, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_c
    index.hnsw.efSearch = ef_s
    index.add(np.ascontiguousarray(emb, dtype=np.float32))
    faiss.write_index(index, str(persist_dir / _FAISS_FILE))

    (persist_dir / _CHUNKS_FILE).write_text(
        json.dumps([asdict(c) for c in chunks], ensure_ascii=False), encoding="utf-8"
    )
    (persist_dir / _META_FILE).write_text(json.dumps({
        "n_chunks": len(chunks), "dim": dim, "faiss_m": m,
        "ef_construction": ef_c, "ef_search": ef_s,
        "build_embedder_id": getattr(emb_model, "name", None),
    }, ensure_ascii=False), encoding="utf-8")
    _logger.info("KB indexed %d chunks (dim=%d) → %s", len(chunks), dim, persist_dir)
    return len(chunks)


class KBRetriever:
    """Hybrid (FAISS dense + BM25) retriever over the persisted KB index."""

    def __init__(self, persist_dir: Path | str = DEFAULT_PERSIST_DIR,
                 *, embedder: KBEmbedder | None = None) -> None:
        self.persist_dir = Path(persist_dir)
        self._embedder = embedder
        self._index: Any = None
        self._chunks: list[KBChunk] = []
        self._bm25: Any = None
        self._bm25_ready = False
        self._product_map: dict[str, list[int]] = {}   # P1.2 product key → chunk ids
        self._parent_map: dict[str, list[int]] = {}     # P1.3 parent_id → chunk ids
        self._topic_map: dict[str, list[int]] = {}      # T1.3 concept → chunk ids (keyword-tagged)
        # m2 dual-dense ensemble channel (kb.ensemble) — lazy, disabled by default:
        self._ens_index: Any = None
        self._ens_embedder: KBEmbedder | None = None
        self._ens_map: list[int] | None = None          # ensemble row → PRIMARY chunk idx
        self._ens_ready = False

    @property
    def available(self) -> bool:
        return (self.persist_dir / _FAISS_FILE).is_file() and (
            self.persist_dir / _CHUNKS_FILE).is_file()

    def _load(self) -> None:
        if self._index is not None:
            return
        import faiss

        self._index = faiss.read_index(str(self.persist_dir / _FAISS_FILE))
        raw = json.loads((self.persist_dir / _CHUNKS_FILE).read_text(encoding="utf-8"))
        self._chunks = [KBChunk(**c) for c in raw]
        # P1.2/P1.3 lookup maps (empty when an old index has no product/parent_id).
        self._product_map = {}
        self._parent_map = {}
        for idx, c in enumerate(self._chunks):
            if getattr(c, "product", None):
                for key in _product_keys(c.product):
                    self._product_map.setdefault(key, []).append(idx)
            if getattr(c, "parent_id", None):
                self._parent_map.setdefault(c.parent_id, []).append(idx)
        # T1.3: keyword-tag chunks by ontology concept (no re-index — derived at load).
        # Built only when topic_scope is enabled, so it costs nothing by default.
        ts = config.kb_param("topic_scope", {})
        if isinstance(ts, dict) and ts.get("enabled", False):
            onto = _load_ontology()
            trig = [(name, [str(t).lower() for t in c.get("triggers", [])])
                    for name, c in onto.items()]
            for idx, c in enumerate(self._chunks):
                low = (c.text or "").lower()
                for name, triggers in trig:
                    if any(t in low for t in triggers):
                        self._topic_map.setdefault(name, []).append(idx)
        if self._embedder is None:
            self._embedder = get_kb_embedder()
        # Dim guard: a config/embedder swap that does not match the built index
        # silently corrupts FAISS search — fail loud instead (no guard existed before).
        meta_path = self.persist_dir / _META_FILE
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001 — never block load on a meta parse error
                meta = {}
            idx_dim = meta.get("dim")
            if isinstance(idx_dim, int) and idx_dim != self._embedder.dim:
                raise RuntimeError(
                    f"KB index dim {idx_dim} != embedder '{self._embedder.name}' "
                    f"dim {self._embedder.dim}. Rebuild the index "
                    f"(scripts/build_knowledge_base_rag.py) or fix kb.embed_model."
                )

    def _ensure_bm25(self) -> None:
        if self._bm25_ready:
            return
        self._bm25_ready = True
        try:
            from rank_bm25 import BM25Okapi

            self._bm25 = BM25Okapi([_tokenize(c.text) for c in self._chunks])
        except Exception as exc:  # noqa: BLE001
            _logger.info("KB BM25 unavailable (%s) — dense only", exc)
            self._bm25 = None

    def _ensure_ensemble(self, ens: dict) -> None:
        """m2: lazily load the SECOND dense channel — a parallel index over the SAME
        chunks built with a different-family embedder (e.g. the sunk BGE-M3 re-embed at
        models/knowledge_base_rag_bge). Its rows are mapped to the PRIMARY chunk indices
        by chunk ``id``, so the fused ranking addresses one chunk list (dual-dense RRF
        ensemble — RND_FINDINGS Q1: two dense families fused are more robust than either
        alone). Best-effort: ANY failure disables the channel; primary path unchanged."""
        if self._ens_ready:
            return
        self._ens_ready = True
        try:
            import faiss

            pdir = Path(str(ens.get("persist_dir") or ""))
            if not pdir.is_absolute():
                pdir = ROOT / pdir
            model_name = str(ens.get("embed_model") or "").strip()
            if not model_name:
                raise RuntimeError("kb.ensemble.embed_model is required")
            idx = faiss.read_index(str(pdir / _FAISS_FILE))
            raw = json.loads((pdir / _CHUNKS_FILE).read_text(encoding="utf-8"))
            by_id = {c.id: i for i, c in enumerate(self._chunks)}
            mapping = [by_id.get(c.get("id"), -1) for c in raw]
            n_hit = sum(1 for m in mapping if m >= 0)
            if n_hit < max(1, int(len(self._chunks) * 0.5)):
                raise RuntimeError(f"chunk-id overlap too low ({n_hit}/{len(raw)})")
            emb = KBEmbedder(model_name)
            meta_path = pdir / _META_FILE
            if meta_path.is_file():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if isinstance(meta.get("dim"), int) and meta["dim"] != emb.dim:
                    raise RuntimeError(
                        f"ensemble index dim {meta['dim']} != embedder '{model_name}' dim {emb.dim}")
            self._ens_index, self._ens_map, self._ens_embedder = idx, mapping, emb
            _logger.info("KB ensemble channel loaded: %s (%d rows, %d mapped)",
                         pdir, len(raw), n_hit)
        except Exception as exc:  # noqa: BLE001 — the ensemble is an optional booster
            _logger.warning("KB ensemble channel disabled (%s)", str(exc)[:160])
            self._ens_index = None

    def _rerank_pool(self, variants: list[str], fused: list[int]) -> list[int]:
        """T1.1: reorder the top candidate pool with the cross-encoder bge-reranker-v2-m3
        (reused from rag.index) so the RIGHT literature/standard surfaces (audit #1
        Siemens-manual, #15 heat-seal-paper). Best-effort: on any model/load failure
        (serve-env torch segfault) the original fused order is kept."""
        pool_n = int(config.kb_param("rerank_pool", 30))
        pool = fused[:pool_n]
        if len(pool) < 2:
            return fused
        # Prefer the English variant for the English-dominant corpus; else the original.
        query = variants[-1] if len(variants) > 1 else variants[0]
        # m2: a DEEP pool (80) lifts source-recall but costs ~40s on CPU >> the 6s gate;
        # kb.rerank_backend="modal" serves the rerank on a T4 (~0.1-0.5s) with a reactive
        # CPU fallback. Default "local" = the original CPU cross-encoder (byte-identical).
        backend = str(config.kb_param("rerank_backend", "local"))
        try:
            texts = [self._chunks[i].text for i in pool]
            if backend == "modal":
                from src.assistant.kb.gpu_reranker import get_gpu_reranker

                order = get_gpu_reranker().order(query, texts)
            else:
                from src.assistant.rag.index import get_reranker

                order = get_reranker().order(query, texts)
        except Exception as exc:  # noqa: BLE001 — best-effort (serve-env torch may fail)
            _logger.info("KB rerank skipped (%s)", str(exc)[:120])
            return fused
        reranked = [pool[j] for j in order]
        return reranked + fused[pool_n:]

    def _apply_source_quotas(self, fused: list[int], quotas: dict, k: int) -> list[int]:
        """T0.4: guarantee each doc_type its quota of the top-ranked chunks within the
        first *k*, then keep the rest in fused order. Promotes the highest-RRF chunks of
        an under-represented lane (e.g. literature/patents crowded out by datasheets &
        site pages) into the top-k without changing relative order inside a lane."""
        reserved: list[int] = []
        rseen: set[int] = set()
        for dt, q in quotas.items():
            try:
                q = int(q)
            except (TypeError, ValueError):
                continue
            if q <= 0:
                continue
            taken = 0
            for i in fused:
                if taken >= q:
                    break
                if i in rseen:
                    continue
                if (self._chunks[i].doc_type or "") == dt:
                    reserved.append(i)
                    rseen.add(i)
                    taken += 1
        if not reserved:
            return fused
        return reserved + [i for i in fused if i not in rseen]

    def retrieve(self, query: str, top_k: int | None = None) -> list[tuple[KBChunk, float]]:
        """Return [(chunk, cosine_score)] top-k. The score gates escalation."""
        return self.retrieve_multi([query], top_k)

    def retrieve_multi(
        self, queries: list[str], top_k: int | None = None,
    ) -> list[tuple[KBChunk, float]]:
        """Cross-lingual retrieve: union dense + BM25 over several query variants.

        Each variant contributes one dense ranking (the multilingual embedder
        bridges uk/ru↔en) and one BM25 ranking (a lexical match — the English
        variant is what lets BM25 reach the English-dominant corpus). All
        rankings are fused with a single RRF; the returned cosine is the MAX
        dense similarity across variants, so the gate score reflects the
        strongest cross-lingual match. Falls back to single-query behaviour
        when only one variant is given.
        """
        if not self.available:
            return []
        self._load()
        variants = [q for q in queries if q and q.strip()]
        if not variants:
            return []
        if config.kb_param("glossary_expansion", False):
            variants = _expand_ontology(variants)  # T1.5 key-free domain expansion
        if config.kb_param("multi_query", False):
            variants = _multi_query_expand(variants)  # m2: LLM paraphrase variants (OFF)
        k = int(top_k if top_k is not None else config.kb_param("top_k", 5))
        n_dense = int(config.kb_param("dense_candidates", 40))
        n_bm25 = int(config.kb_param("bm25_candidates", 40))
        # m2: per-channel RRF weights — ONLY applied when the dual-dense ensemble is on
        # (the tuned dense=0.5 balances TWO dense channels; without BGE, MiniLM-dense must
        # stay 1.0). So toggling ensemble.enabled is a clean single-flag rollback of the
        # whole fusion change. ensemble off → classic RRF (dense=bm25=1.0), byte-identical.
        ens_cfg = config.kb_param("ensemble", {})
        ens_on = isinstance(ens_cfg, dict) and ens_cfg.get("enabled", False)
        wcfg = config.kb_param("rrf_weights", {}) if ens_on else {}
        wcfg = wcfg if isinstance(wcfg, dict) else {}
        w_dense = float(wcfg.get("dense", 1.0))
        w_bm25 = float(wcfg.get("bm25", 1.0))

        qv = self._embedder.encode(variants)
        scores, idxs = self._index.search(np.ascontiguousarray(qv, dtype=np.float32), n_dense)
        rankings: list[list[int]] = []
        weights: list[float] = []
        sim_by_id: dict[int, float] = {}
        for row_scores, row_idxs in zip(scores, idxs):
            dense_ids = [int(i) for i in row_idxs if i >= 0]
            rankings.append(dense_ids)
            weights.append(w_dense)
            for i, s in zip(row_idxs, row_scores):
                if i >= 0:
                    ii = int(i)
                    sim_by_id[ii] = max(sim_by_id.get(ii, 0.0), float(s))

        self._ensure_bm25()
        if self._bm25 is not None:
            for q in variants:
                bm = self._bm25.get_scores(_tokenize(q))
                bm_ids = [int(i) for i in np.argsort(-bm)[:n_bm25]]
                rankings.append(bm_ids)
                weights.append(w_bm25)

        # m2 dual-dense ensemble: a SECOND embedder family over the same chunks adds
        # one more ranking per variant. NB: sim_by_id (the escalation-gate cosine) stays
        # PRIMARY-only — min_score is calibrated for the primary embedder's scale; a
        # chunk surfaced only by the ensemble carries score 0.0 (gate uses the max).
        ens = config.kb_param("ensemble", {})
        if isinstance(ens, dict) and ens.get("enabled", False):
            self._ensure_ensemble(ens)
            if self._ens_index is not None and self._ens_embedder is not None:
                n_ens = int(ens.get("dense_candidates", n_dense))
                w_ens = float(wcfg.get("ensemble", ens.get("weight", 1.0)))
                qv2 = self._ens_embedder.encode(variants)
                _, idxs2 = self._ens_index.search(
                    np.ascontiguousarray(qv2, dtype=np.float32), n_ens)
                assert self._ens_map is not None
                for row_idxs in idxs2:
                    ens_ids = [self._ens_map[int(i)] for i in row_idxs
                               if i >= 0 and self._ens_map[int(i)] >= 0]
                    rankings.append(ens_ids)
                    weights.append(w_ens)

        fused = _rrf(rankings, kconst=int(config.kb_param("rrf_k", 60)), weights=weights)

        # Product-code lexical boost: pin chunks that contain a discriminative
        # query code (e.g. FXCMT → its datasheet) to the front, so the exact
        # match surfaces despite same-language dense bias burying a lone chunk.
        if config.kb_param("code_boost", True) and self._bm25 is not None:
            codes = _code_tokens(variants)
            if codes:
                max_hits = int(config.kb_param("code_boost_max_hits", 12))
                boost: list[int] = []
                bseen: set[int] = set()
                for tok in codes:
                    bm = self._bm25.get_scores([tok])
                    hit_ids = [int(i) for i in np.nonzero(bm > 0)[0]]
                    if 0 < len(hit_ids) <= max_hits:  # rare ⇒ a real product code
                        hit_ids.sort(key=lambda i: float(bm[i]), reverse=True)
                        for i in hit_ids:
                            if i not in bseen:
                                bseen.add(i)
                                boost.append(i)
                if boost:
                    fused = boost + [i for i in fused if i not in bseen]

        # P1.2 metadata scope: pin ALL chunks of a matched product (including numeric
        # spec chunks that don't carry the product token) so the whole datasheet's
        # values co-retrieve. Stronger than code_boost, which pins only token hits.
        ms = config.kb_param("metadata_scope", {})
        if isinstance(ms, dict) and ms.get("enabled", False) and self._product_map:
            prod_ids: list[int] = []
            pseen: set[int] = set()
            for key in sorted(_product_candidates(variants)):
                for cid in self._product_map.get(key, []):
                    if cid not in pseen:
                        pseen.add(cid)
                        prod_ids.append(cid)
            if prod_ids:
                if ms.get("mode") == "filter":
                    fused = prod_ids
                else:  # soft (default): pin product chunks first, keep the rest
                    fused = prod_ids + [i for i in fused if i not in pseen]

        # P1.3 parent-merge: return a qualifying datasheet WHOLE (all its chunks,
        # locator-ordered) so the LLM sees the full table, not fragments. A parent
        # qualifies when its product matched the query code OR >= merge_min_children
        # of it already surfaced in the top-k. Capped by max_parents.
        pm = config.kb_param("parent_merge", {})
        if isinstance(pm, dict) and pm.get("enabled", False) and self._parent_map:
            min_children = int(pm.get("merge_min_children", 2))
            max_parents = int(pm.get("max_parents", 2))
            expand: list[str] = []
            seen_p: set[str] = set()
            # (1) product-matched parents — robust even when NO chunk of the datasheet
            # surfaced in the candidate pool (e.g. a uk datasheet a code_boost missed).
            for key in sorted(_product_candidates(variants)):
                for cid in self._product_map.get(key, []):
                    pid0 = self._chunks[cid].parent_id
                    if pid0 and pid0 not in seen_p:
                        seen_p.add(pid0)
                        expand.append(pid0)
            # (2) parents with >= merge_min_children already in the top-k.
            counts: dict[str, int] = {}
            for i in fused[:k]:
                pid0 = self._chunks[i].parent_id
                if pid0:
                    counts[pid0] = counts.get(pid0, 0) + 1
            for i in fused:
                pid0 = self._chunks[i].parent_id
                if pid0 and pid0 not in seen_p and counts.get(pid0, 0) >= min_children:
                    seen_p.add(pid0)
                    expand.append(pid0)
            expand = expand[:max_parents]
            if expand:
                merged: list[int] = []
                mseen: set[int] = set()
                for pid0 in expand:
                    for cid in self._parent_map.get(pid0, []):
                        if cid not in mseen:
                            mseen.add(cid)
                            merged.append(cid)
                fused = merged + [i for i in fused if i not in mseen]

        # For NON-product (science) queries only (the datasheet path keeps its pins):
        # T1.3 topic promote, T1.1 rerank, THEN T0.4 quotas, THEN truncate.
        is_product = bool(_product_candidates(variants) & set(self._product_map))
        # T1.3 topic scope: promote candidates whose chunk keyword-matches a query
        # concept (anilox query → anilox-mentioning chunks first), so a topically
        # relevant doc retrieved but ranked low surfaces. 'filter' keeps only matches.
        ts = config.kb_param("topic_scope", {})
        if isinstance(ts, dict) and ts.get("enabled", False) and self._topic_map and not is_product:
            tagged: set[int] = set()
            for name in _query_concepts(variants):
                tagged.update(self._topic_map.get(name, []))
            if tagged:
                if ts.get("mode") == "filter":
                    filt = [i for i in fused if i in tagged]
                    fused = filt or fused  # never empty the result set
                else:  # soft: promote in-pool topic matches to the front
                    fused = [i for i in fused if i in tagged] + [i for i in fused if i not in tagged]
        if config.kb_param("rerank", False) and variants and not is_product:
            fused = self._rerank_pool(variants, fused)
        sq = config.kb_param("source_quotas", {})
        if (config.kb_param("source_quotas_enabled", False) and isinstance(sq, dict) and sq
                and not is_product):
            fused = self._apply_source_quotas(fused, sq, k)

        fused = fused[:k]
        return [(self._chunks[i], sim_by_id.get(i, 0.0)) for i in fused]


_SHARED_RETRIEVER: KBRetriever | None = None


def get_kb_retriever() -> KBRetriever:
    global _SHARED_RETRIEVER
    if _SHARED_RETRIEVER is None:
        # Optional override (kbdepth eval Layer B can point answer() at the eval
        # index). Inert in production — default resolves to DEFAULT_PERSIST_DIR.
        import os
        override = os.environ.get("KB_PERSIST_DIR")
        _SHARED_RETRIEVER = KBRetriever(Path(override)) if override else KBRetriever()
    return _SHARED_RETRIEVER


def warm() -> None:
    try:
        r = get_kb_retriever()
        if r.available:
            r.retrieve("розігрів", top_k=1)
            _logger.info("KB index warmed up")
    except Exception as exc:  # noqa: BLE001
        _logger.info("KB warm-up skipped (%s)", exc)


__all__ = ["build_index", "KBRetriever", "get_kb_retriever", "warm", "DEFAULT_PERSIST_DIR"]
