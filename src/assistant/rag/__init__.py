"""RAG over project knowledge (docs, glossary, schema-card, per-page help).

A small static corpus indexed with a lightweight persisted numpy cosine store
(``models/assistant_rag/``). Embeddings via the configured multilingual model
(BGE-M3) loaded lazily; the embedder is injectable so retrieval math is testable
offline without a model download.
"""
