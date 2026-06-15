"""GPU-accelerated KB embedder with reactive fallback to local CPU.

Wraps the Modal MiniLM embed job (``plandruku-kb-embed``) behind the SAME
interface as :class:`~src.assistant.kb.embedder.KBEmbedder` so it drops straight
into ``build_index(embedder=...)`` (``src/assistant/kb/index.py:160``).

On ANY failure — Modal credits exhausted, app not deployed, network, timeout,
shape mismatch — it logs a warning and re-embeds the full batch on the local CPU.
Vectors are numerically equivalent across paths (same pinned model, normalize,
fp32; see ``embed_minilm.py`` parity contract), so the FAISS index is unchanged
whichever path runs.
"""
from __future__ import annotations

import io
import tempfile
from pathlib import Path

import numpy as np

from src.assistant.kb.embedder import KBEmbedder
from src.utils.logger import setup_logger

_logger = setup_logger("kb.gpu_embedder")

_DEFAULT_APP = "plandruku-kb-embed"
_DEFAULT_FN = "embed_corpus"
_DEFAULT_VOLUME = "rag-data"
_CHUNKS_REMOTE = "chunks.parquet"
_EMBEDDINGS_REMOTE = "embeddings.npy"


class GpuFallbackEmbedder:
    """Embed on a Modal T4; fall back to local CPU on any failure.

    Implements the :class:`KBEmbedder` interface (``encode``, ``dim``, ``name``)
    so it can be injected into ``build_index``. ``used_fallback`` records whether
    the CPU path was taken (for the build summary).
    """

    def __init__(
        self,
        *,
        model_name: str | None = None,
        app_name: str = _DEFAULT_APP,
        fn_name: str = _DEFAULT_FN,
        volume_name: str = _DEFAULT_VOLUME,
    ) -> None:
        self._local = KBEmbedder(model_name)
        self._app_name = app_name
        self._fn_name = fn_name
        self._volume_name = volume_name
        self.used_fallback = False

    @property
    def name(self) -> str:
        """The resolved model id (delegated to the local embedder)."""
        return self._local.name

    @property
    def dim(self) -> int:
        """Embedding dimension (loads the local model lazily; 384 for MiniLM)."""
        return self._local.dim

    def encode(
        self, texts: list[str], *, batch_size: int = 64, show_progress: bool = False
    ) -> np.ndarray:
        """Encode on Modal T4; on any failure, re-encode the full batch locally."""
        try:
            emb = self._encode_gpu(texts, batch_size=batch_size)
            _logger.info("KB GPU embed ok: %d×%d on Modal T4", emb.shape[0], emb.shape[1])
            return emb
        except Exception as exc:  # noqa: BLE001 — reactive fallback to local CPU
            _logger.warning(
                "KB GPU embed failed (%s) — falling back to local CPU embedding.",
                str(exc)[:200],
            )
            self.used_fallback = True
            return self._local.encode(
                texts, batch_size=batch_size, show_progress=show_progress
            )

    def _encode_gpu(self, texts: list[str], *, batch_size: int) -> np.ndarray:
        """Upload texts → run the Modal job → download embeddings.npy."""
        import modal
        import pandas as pd

        with tempfile.TemporaryDirectory() as td:
            parquet_path = Path(td) / _CHUNKS_REMOTE
            pd.DataFrame({"text": list(texts)}).to_parquet(parquet_path, index=False)

            vol = modal.Volume.from_name(self._volume_name)
            with vol.batch_upload(force=True) as batch:  # overwrite prior chunks.parquet
                batch.put_file(str(parquet_path), f"/{_CHUNKS_REMOTE}")

            fn = modal.Function.from_name(self._app_name, self._fn_name)
            result = fn.remote(batch_size=batch_size)
            _logger.info("Modal embed_corpus returned: %s", result)

            # The container already committed embeddings.npy; read the latest
            # version directly (Volume.reload() is container-only, not for clients).
            raw = b"".join(vol.read_file(_EMBEDDINGS_REMOTE))

        emb = np.load(io.BytesIO(raw)).astype(np.float32, copy=False)
        if emb.shape[0] != len(texts):
            raise ValueError(
                f"GPU embeddings rows {emb.shape[0]} != chunks {len(texts)}"
            )
        return emb


__all__ = ["GpuFallbackEmbedder"]
