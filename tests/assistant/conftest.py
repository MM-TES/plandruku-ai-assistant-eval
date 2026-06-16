"""Shared fixtures for the assistant test-suite.

Provides an offline fake of the OpenRouter (OpenAI-SDK) client so unit tests
never hit the network, plus an autouse fixture that forces LangSmith tracing
off during tests.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest


# --- Curated-checkout guards ------------------------------------------------
# This suite ships in a *curated* subset of the production monorepo: some
# system-under-test modules, helper scripts, and golden sets are intentionally
# not included. The guards below let a test SKIP (never fail) when its
# dependency / file / dataset is absent, while running fully in the complete
# repo. They are deliberately conditional — presence of a module, a file, or a
# non-empty dataset — never an unconditional skip.
_REPO_ROOT = Path(__file__).resolve().parents[2]


def requires_import(modname: str):
    """Skip the calling test unless ``modname`` is importable in this checkout.

    Returns the imported module so a test can bind it. Thin wrapper over
    :func:`pytest.importorskip` with a curated-checkout-aware reason.
    """
    return pytest.importorskip(modname, reason=f"{modname} not present in this checkout")


def requires_file(relpath: str) -> Path:
    """Skip the calling test unless ``relpath`` exists under the repo root.

    Returns the resolved path so a test can read it.
    """
    p = _REPO_ROOT / relpath
    if not p.exists():
        pytest.skip(f"{relpath} not present in this checkout")
    return p


def skip_if_empty(golden, name: str = "golden set"):
    """Skip the calling test when ``golden`` is empty (dataset not shipped)."""
    if not golden:
        pytest.skip(f"{name} is empty in this checkout — golden not shipped")
    return golden


# --- Offline LangSmith ------------------------------------------------------
@pytest.fixture(autouse=True)
def _offline_langsmith(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force tracing off so @traceable never attempts a network call in tests."""
    monkeypatch.setenv("LANGSMITH_TRACING", "false")


# --- Fake OpenRouter / OpenAI client ---------------------------------------
class FakeMessage:
    def __init__(self, content: str | None, tool_calls: list | None = None) -> None:
        self.content = content
        self.tool_calls = tool_calls


class FakeChoice:
    def __init__(self, message: FakeMessage) -> None:
        self.message = message


class FakeUsage:
    def __init__(self, prompt_tokens: int = 10, completion_tokens: int = 20) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class FakeCompletion:
    def __init__(
        self,
        content: str | None,
        tool_calls: list | None = None,
        usage: FakeUsage | None = None,
    ) -> None:
        self.choices = [FakeChoice(FakeMessage(content, tool_calls))]
        self.usage = usage if usage is not None else FakeUsage()


class FakeCompletions:
    """Scripted ``create`` that pops queued responses; records every call."""

    def __init__(self, responses: list[Any]) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    def create(self, **kwargs: Any) -> FakeCompletion:
        self.calls.append(kwargs)
        resp: Any = self._responses.pop(0) if self._responses else FakeCompletion("{}")
        if isinstance(resp, Exception):
            raise resp
        if isinstance(resp, str):
            return FakeCompletion(resp)
        return resp


class FakeChat:
    def __init__(self, completions: FakeCompletions) -> None:
        self.completions = completions


class FakeClient:
    """Drop-in for ``openai.OpenAI`` with a scripted completions queue."""

    def __init__(self, responses: list[Any] | None = None) -> None:
        self.chat = FakeChat(FakeCompletions(responses or []))

    @property
    def calls(self) -> list[dict]:
        return self.chat.completions.calls


@pytest.fixture
def make_fake_client():
    """Factory: ``make_fake_client([FakeCompletion(...), "raw json", Exception()])``."""

    def _factory(responses: list[Any] | None = None) -> FakeClient:
        return FakeClient(responses)

    return _factory


@pytest.fixture
def patch_llm_client(monkeypatch: pytest.MonkeyPatch, make_fake_client):
    """Patch ``src.assistant.llm.get_client`` to return a scripted fake client.

    Returns the factory so a test can build the client, install it, and later
    inspect ``client.calls``.
    """

    def _install(responses: list[Any] | None = None) -> FakeClient:
        client = make_fake_client(responses)
        monkeypatch.setattr("src.assistant.llm.get_client", lambda: client)
        return client

    return _install
