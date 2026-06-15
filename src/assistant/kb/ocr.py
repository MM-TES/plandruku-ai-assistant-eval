"""OCR backend (EasyOCR — Ukrainian + Russian + English). Build-time only.

EasyOCR was chosen over Tesseract (no system binary; winget/choco/admin
unavailable). It supports uk + ru (Cyrillic) together with en — English is
compatible with any EasyOCR language group, so ["uk","ru","en"] is valid and the
Cyrillic recognition network also covers Latin characters.

Quality levers (all config-driven, kb.ocr_*): render scanned PDF pages at a
higher DPI (loaders.py), magnify small text (mag_ratio), upscale tiny images,
and tune the detector threshold. The reader is a singleton (model load is slow;
do it once).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.assistant import config
from src.utils.logger import setup_logger

_logger = setup_logger(__name__)
_reader: Any = None


def _langs() -> list[str]:
    langs = config.kb_param("easyocr_langs", ["uk", "ru", "en"])
    return list(langs) if isinstance(langs, (list, tuple)) else ["uk", "ru", "en"]


def ocr_available() -> bool:
    try:
        import easyocr  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


def get_reader() -> Any:
    """Singleton EasyOCR reader. Falls back to a known-good lang set if the
    configured combination is rejected (incompatible script groups)."""
    global _reader
    if _reader is None:
        import easyocr

        requested = _langs()
        for attempt in (requested, ["uk", "ru"], ["en"]):
            try:
                _reader = easyocr.Reader(attempt, gpu=False, verbose=False)
                if attempt == requested:
                    _logger.info("EasyOCR reader loaded (langs=%s)", attempt)
                else:
                    _logger.warning("EasyOCR fell back to langs=%s (requested %s)", attempt, requested)
                break
            except Exception as exc:  # noqa: BLE001
                _logger.warning("EasyOCR langs %s failed: %s", attempt, exc)
        if _reader is None:
            raise RuntimeError("EasyOCR reader could not be initialized")
    return _reader


def _read_kwargs() -> dict[str, Any]:
    return {
        "detail": 0,
        "paragraph": True,
        "decoder": str(config.kb_param("ocr_decoder", "greedy")),  # 'beamsearch' = slower/better
        "mag_ratio": float(config.kb_param("ocr_mag_ratio", 1.5)),  # magnify small text
        "text_threshold": float(config.kb_param("ocr_text_threshold", 0.7)),
    }


def _upscale_small(arr):
    """Upscale small images so faint/small text becomes legible to the detector."""
    try:
        import numpy as np
        from PIL import Image

        h, w = arr.shape[:2]
        min_side = int(config.kb_param("ocr_min_side", 1000))
        longest = max(h, w)
        if 0 < longest < min_side:
            scale = min_side / longest
            img = Image.fromarray(arr).resize(
                (max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS
            )
            return np.asarray(img)
    except Exception:  # noqa: BLE001
        pass
    return arr


def _join(lines: list) -> str:
    return "\n".join(str(x) for x in lines if x and str(x).strip())


def ocr_array(arr) -> str:
    """OCR a numpy image array (e.g. a rendered PDF page)."""
    try:
        return _join(get_reader().readtext(_upscale_small(arr), **_read_kwargs()))
    except Exception as exc:  # noqa: BLE001
        _logger.debug("OCR(array) failed: %s", exc)
        return ""


def ocr_path(path: Path) -> str:
    """OCR an image file. Reads via PIL (Unicode-safe) then passes the array —
    cv2.imread (EasyOCR's default) fails on non-ASCII Windows paths."""
    try:
        import numpy as np
        from PIL import Image

        img = Image.open(str(path)).convert("RGB")
        return ocr_array(np.array(img))
    except Exception as exc:  # noqa: BLE001
        _logger.debug("OCR(%s) failed: %s", path.name, exc)
        return ""


__all__ = ["ocr_available", "get_reader", "ocr_array", "ocr_path"]
