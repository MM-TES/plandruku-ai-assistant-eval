"""External knowledge-base RAG (escalation tier).

A SECOND, independent RAG over an external multi-format knowledge base
(``config/assistant.json: kb.path``), consulted ONLY when the in-project
operator help + live data do not cover the question (relevance-threshold gate).

Layout:
- loaders.py  — per-format text extraction (docx/pdf/xlsx/pptx/html/txt + .doc + OCR)
- corpus.py   — recursive walk → load → chunk → dedup → metadata
- embedder.py — multilingual-MiniLM (fast, 384-dim) singleton
- index.py    — FAISS HNSW + BM25 hybrid index (persisted to models/knowledge_base_rag/)
- search.py   — high-level retrieve() used by the orchestrator escalation

Build: ``python scripts/build_knowledge_base_rag.py`` (one-time, long; queries are fast).
"""
