"""Phase 5 gate: assistant web endpoints (TestClient)."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.assistant.schema import AssistantResponse

pytest.importorskip("src.web.app", reason="src.web.app (UI shell) not present in this checkout")
from src.web.app import create_app


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(create_app())


def test_panel_renders(client: TestClient) -> None:
    r = client.get("/assistant/panel")
    assert r.status_code == 200
    assert 'data-testid="assistant-form"' in r.text
    assert 'data-testid="assistant-input"' in r.text
    assert 'data-testid="assistant-read-screen"' in r.text


def test_ask_returns_messages_fragment(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(
        "src.assistant.orchestrator.answer",
        lambda req, **kw: AssistantResponse(text_md="ТЕСТ-ВІДПОВІДЬ", route="instructions"),
    )
    r = client.post(
        "/assistant/ask",
        data={"message": "як це працює?", "page_context": '{"route": "/"}', "scope": "standard"},
    )
    assert r.status_code == 200
    assert "ТЕСТ-ВІДПОВІДЬ" in r.text


def test_chat_redirects_to_assistant_when_enabled(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr("src.assistant.config.is_enabled", lambda: True)
    r = client.get("/chat", follow_redirects=False)
    assert r.status_code == 302
    assert r.headers["location"] == "/assistant"


def test_bubble_present_in_base_when_enabled(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr("src.assistant.config.is_enabled", lambda: True)
    r = client.get("/assistant")
    assert r.status_code == 200
    assert 'data-testid="assistant-bubble"' in r.text
    assert "/static/assistant.js" in r.text
