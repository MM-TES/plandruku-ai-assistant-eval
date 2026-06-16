"""Streaming: orchestrator.stream_answer events + /assistant/stream endpoint."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.assistant.orchestrator import stream_answer
from src.assistant.schema import AssistantRequest, PageContext

pytest.importorskip("src.web.app", reason="src.web.app (UI shell) not present in this checkout")
from src.web.app import create_app

from tests.assistant.conftest import FakeCompletion


@pytest.fixture(autouse=True)
def _hermetic(monkeypatch) -> None:
    monkeypatch.setattr("src.assistant.grounding.context_builder.live_summary", lambda: "")
    monkeypatch.setattr("src.assistant.config.feature", lambda name: False)


def test_stream_emits_status_then_deltas_then_done(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.assistant.orchestrator._synthesize_stream",
        lambda *a, **k: iter(["Очі", "куван", "ня."]),
    )
    evs = list(stream_answer(AssistantRequest(
        message="скільки замовлень очікують?", page_context=PageContext(route="/"))))
    types = [e["type"] for e in evs]
    assert types[0] == "status"  # immediate feedback before any LLM work
    assert "done" in types
    deltas = "".join(e["text"] for e in evs if e["type"] == "delta")
    assert deltas == "Очікування."
    done = next(e for e in evs if e["type"] == "done")
    assert done["route"] == "data_query"
    assert done["text"] == "Очікування."


def test_stream_out_of_scope_is_single_delta(patch_llm_client) -> None:
    patch_llm_client([FakeCompletion('{"route": "out_of_scope", "refusal": "Вибач, не можу."}')])
    evs = list(stream_answer(AssistantRequest(message="купи мені акції")))
    done = next(e for e in evs if e["type"] == "done")
    assert done["route"] == "out_of_scope"
    assert any(e["type"] == "delta" and "Вибач" in e["text"] for e in evs)


def test_stream_endpoint_returns_sse(monkeypatch) -> None:
    client = TestClient(create_app())
    monkeypatch.setattr(
        "src.assistant.orchestrator.stream_answer",
        lambda req, **kw: iter([
            {"type": "status", "text": "Аналізую запит…"},
            {"type": "delta", "text": "привіт"},
            {"type": "done", "route": "instructions", "text": "привіт",
             "citations": [], "tool_trace": [], "usage": {}},
        ]),
    )
    r = client.post("/assistant/stream", data={"message": "як", "page_context": '{"route": "/"}'})
    assert r.status_code == 200
    assert "text/event-stream" in r.headers["content-type"]
    assert "data:" in r.text
    assert "привіт" in r.text
    assert '"idx"' in r.text  # endpoint stamps the message index onto done
