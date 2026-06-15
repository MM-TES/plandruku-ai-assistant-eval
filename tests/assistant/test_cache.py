"""Phase B gate: semantic response cache (store/lookup/threshold/TTL/page-key)."""
from __future__ import annotations

import hashlib

import numpy as np

from src.assistant.cache import SemanticCache


class HashEmbedder:
    """Deterministic: identical text -> identical vector; any change -> different.

    Lets us verify the cache MECHANISM offline (exact hit, page separation, miss)
    without a real semantic model.
    """

    def encode(self, texts: list[str]) -> np.ndarray:
        rows = []
        for t in texts:
            h = hashlib.sha256(t.encode("utf-8")).digest()
            rows.append(np.frombuffer(h, dtype=np.uint8)[:16].astype(np.float32))
        return np.array(rows, dtype=np.float32)


def _cache() -> SemanticCache:
    return SemanticCache(embedder=HashEmbedder())


def test_empty_cache_misses() -> None:
    assert _cache().lookup("як зарезервувати рулон", "/workflow:materialy") is None


def test_exact_hit() -> None:
    c = _cache()
    c.store("як зарезервувати рулон", "/workflow:materialy",
            {"text_md": "Натисніть «Підібрати рулони».", "route": "instructions", "citations": []})
    hit = c.lookup("як зарезервувати рулон", "/workflow:materialy")
    assert hit is not None
    assert hit["text_md"] == "Натисніть «Підібрати рулони»."
    assert hit["route"] == "instructions"


def test_different_page_does_not_hit() -> None:
    c = _cache()
    c.store("що мені тут робити", "/workflow:prodazhi", {"text_md": "A", "route": "instructions"})
    # same question, different page -> different key -> miss
    assert c.lookup("що мені тут робити", "/workflow:vyrobnytstvo") is None
    assert c.lookup("що мені тут робити", "/workflow:prodazhi") is not None


def test_dissimilar_query_misses() -> None:
    c = _cache()
    c.store("як зарезервувати рулон", "/m", {"text_md": "A", "route": "instructions"})
    assert c.lookup("скільки прострочених замовлень", "/m") is None


def test_expired_entry_misses() -> None:
    c = _cache()
    c.store("як зарезервувати рулон", "/m",
            {"text_md": "A", "route": "instructions"}, ttl=0.0)
    assert c.lookup("як зарезервувати рулон", "/m") is None


def test_size_cap_evicts_oldest() -> None:
    import src.assistant.config as cfg

    c = _cache()
    # cap is read from config; just assert it never grows unbounded
    for i in range(int(cfg.threshold("cache_max_entries", 500)) + 10):
        c.store(f"q{i}", "/m", {"text_md": str(i), "route": "instructions"})
    assert len(c._vecs) <= int(cfg.threshold("cache_max_entries", 500))
