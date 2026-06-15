"""GPU-served cross-encoder reranker with reactive fallback to local CPU.

Wraps the Modal T4 rerank job (``plandruku-rag-rerank``) behind the same
``order(query, texts) -> list[int]`` interface as the local
:class:`src.assistant.rag.index.Reranker`, so ``KBRetriever._rerank_pool`` can use
either backend via the ``kb.rerank_backend`` config knob.

On ANY failure — Modal credits/app/network/timeout — it logs a warning and reranks
on the local CPU CrossEncoder (slow but correct). The bottleneck the GPU solves is
LATENCY: a deep top-80 rerank is ~0.1-0.5 s on a T4 vs ~40 s on CPU, so the lever is
only shippable GPU-served; the CPU path is the safety net, not the live path.
"""
from __future__ import annotations

from src.utils.logger import setup_logger

_logger = setup_logger("kb.gpu_reranker")

_DEFAULT_APP = "plandruku-rag-rerank"
_DEFAULT_FN = "rerank_pairs"


class GpuReranker:
    """Rerank on a Modal T4; fall back to the local CPU cross-encoder on any failure."""

    def __init__(self, *, app_name: str = _DEFAULT_APP, fn_name: str = _DEFAULT_FN,
                 max_length: int = 512) -> None:
        self._app_name = app_name
        self._fn_name = fn_name
        self._max_length = max_length
        self._fn = None
        self.used_fallback = False

    def _ensure_fn(self):
        if self._fn is None:
            import modal

            self._fn = modal.Function.from_name(self._app_name, self._fn_name)
        return self._fn

    def order(self, query: str, texts: list[str], *, cpu_fallback: bool = False) -> list[int]:
        """Return indices of *texts* sorted best-first by cross-encoder relevance.

        On Modal failure the default is to RAISE (so ``KBRetriever._rerank_pool`` keeps the
        fast un-reranked ensemble order — LIVE-SAFE: a CPU rerank of an 80-deep pool is ~40s,
        which would blow the latency budget far worse than skipping the rerank). Pass
        ``cpu_fallback=True`` only OFFLINE (e.g. to complete an eval sweep) to accept the slow
        local CPU path instead of skipping."""
        if not texts:
            return []
        try:
            scores = self._ensure_fn().remote(query, list(texts), self._max_length)
            if not scores or len(scores) != len(texts):
                raise ValueError(f"reranker returned {len(scores or [])} scores for {len(texts)} texts")
            return sorted(range(len(texts)), key=lambda i: float(scores[i]), reverse=True)
        except Exception as exc:  # noqa: BLE001
            if cpu_fallback:
                _logger.warning("GPU rerank failed (%s) — OFFLINE CPU fallback (slow).", str(exc)[:160])
                self.used_fallback = True
                from src.assistant.rag.index import get_reranker

                return get_reranker().order(query, texts)
            _logger.warning("GPU rerank failed (%s) — SKIPPING rerank (live-safe).", str(exc)[:160])
            raise


_SHARED: GpuReranker | None = None


def get_gpu_reranker() -> GpuReranker:
    """Process-wide singleton (the Modal Function handle is reusable across calls)."""
    global _SHARED
    if _SHARED is None:
        _SHARED = GpuReranker()
    return _SHARED


__all__ = ["GpuReranker", "get_gpu_reranker"]
