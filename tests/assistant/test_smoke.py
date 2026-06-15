"""Phase 0 gate: package imports and config loads with the expected shape."""
from __future__ import annotations

from src.utils.config_loader import load_config


def test_package_imports() -> None:
    import src.assistant  # noqa: F401


def test_assistant_config_loads() -> None:
    cfg = load_config("assistant")
    assert isinstance(cfg["enabled"], bool)
    assert isinstance(cfg["features"], dict)
    # every capability flag present and boolean
    for flag in ("tools", "text2sql", "vision", "rag", "history", "feedback"):
        assert isinstance(cfg["features"][flag], bool)
    # text2sql stays off until the read-only DB role is provisioned
    assert cfg["features"]["text2sql"] is False
    assert cfg["models"]["router"]
    assert cfg["models"]["answer"]
    assert "default" in cfg["pages"]
    assert cfg["langsmith"]["project"] == "plandruku-assistant"


def test_heuristics_and_prompts_present() -> None:
    cfg = load_config("assistant")
    assert cfg["prompts"]["router"]
    assert cfg["prompts"]["answer"]
    assert cfg["prompts"]["refusal"]
    assert "instructions" in cfg["heuristics"]
