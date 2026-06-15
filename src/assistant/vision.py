"""On-demand screenshot understanding (vision model).

PII note: screenshots may contain customer names. The image is processed
in-memory, never written to disk, and deliberately NOT passed through a
``@traceable`` boundary (so the raw bytes never reach LangSmith). Only the
image dimensions + a short hash are logged.
"""
from __future__ import annotations

import base64
import hashlib
from typing import Any

from src.assistant import config
from src.assistant.llm import LLMUsage, call_llm
from src.assistant.schema import PageContext
from src.utils.logger import setup_logger

_logger = setup_logger(__name__)


def _strip_data_url(b64: str) -> str:
    if b64.startswith("data:") and "," in b64:
        return b64.split(",", 1)[1]
    return b64


def _prepare(screenshot_b64: str) -> tuple[str, tuple[int, int] | None]:
    """Return a (data_url, dims). Downscales with Pillow if available."""
    raw_b64 = _strip_data_url(screenshot_b64)
    dims: tuple[int, int] | None = None
    try:
        import io

        from PIL import Image  # type: ignore

        max_px = int(config.threshold("vision_max_px", 1568))
        img = Image.open(io.BytesIO(base64.b64decode(raw_b64)))
        img = img.convert("RGB")
        if max(img.size) > max_px:
            ratio = max_px / max(img.size)
            img = img.resize((int(img.size[0] * ratio), int(img.size[1] * ratio)))
        dims = img.size
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        raw_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception as exc:  # noqa: BLE001 — Pillow optional; pass image through
        _logger.debug("vision downscale skipped: %s", exc)
    return f"data:image/png;base64,{raw_b64}", dims


def describe(
    screenshot_b64: str,
    page_context: PageContext,
    *,
    message: str = "",
    grounded: str = "",
    usage: LLMUsage | None = None,
) -> str:
    """Answer about the screen. The image is used ONLY to read context (which
    tab/filters/selection); FACTS come from *grounded* (structured system data),
    never from OCR of the pixels. Returns Ukrainian markdown.
    """
    data_url, dims = _prepare(screenshot_b64)
    sha = hashlib.sha256(_strip_data_url(screenshot_b64).encode()).hexdigest()[:12]
    _logger.info("vision request: page=%s dims=%s sha=%s", page_context.key(), dims, sha)

    if message:
        instruction = (
            f"Оператор на сторінці {page_context.key()} надіслав знімок екрана і питає: "
            f"{message}. Зрозумій КОНТЕКСТ із зображення (вкладка, фільтри, що виділено), "
            f"але всі факти бери з блоку «ДАНІ СИСТЕМИ»."
        )
    else:
        instruction = (
            f"Оператор на сторінці {page_context.key()} надіслав знімок екрана. Поясни, що це "
            f"за екран і що тут можна робити. Контекст — із зображення; факти — лише з блоку "
            f"«ДАНІ СИСТЕМИ»; точні коди/числа з картинки не зчитуй і не вигадуй."
        )
    data_block = (
        f"[ДАНІ СИСТЕМИ]\n{grounded}" if grounded
        else "[ДАНІ СИСТЕМИ] (порожньо — точних даних не надано; не вигадуй кодів/чисел, "
             "скеруй оператора до таблиці або до режиму «Запитати»)"
    )
    system_prompt = config.prompt("vision") or config.prompt("answer")
    resp = call_llm(
        agent_name="vision",
        role_key="vision",
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{instruction}\n\n{data_block}"},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
        usage=usage,
        temperature=0.2,
        max_tokens=int(config.threshold("vision_max_tokens", 1024)),
    )
    return resp.choices[0].message.content or ""


__all__ = ["describe"]
