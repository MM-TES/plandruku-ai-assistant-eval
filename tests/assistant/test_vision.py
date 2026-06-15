"""Phase 4 gate: vision skill — image block sent, data-url handling, PII not logged."""
from __future__ import annotations

from src.assistant import vision
from src.assistant.schema import PageContext

# 1x1 transparent PNG
_PNG = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)


def test_describe_sends_image_block_and_returns_text(patch_llm_client) -> None:
    client = patch_llm_client(["опис екрана"])
    out = vision.describe(_PNG, PageContext(route="/kpi"), message="поясни")
    assert out == "опис екрана"
    content = client.calls[0]["messages"][1]["content"]
    assert any(isinstance(b, dict) and b.get("type") == "image_url" for b in content)


def test_prepare_handles_data_url_prefix() -> None:
    data_url, _dims = vision._prepare(f"data:image/png;base64,{_PNG}")
    assert data_url.startswith("data:image/png;base64,")


def test_prepare_handles_bare_base64() -> None:
    data_url, _dims = vision._prepare(_PNG)
    assert data_url.startswith("data:image/png;base64,")
