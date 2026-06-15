"""Phase 4 gate: orchestrator dispatch per route (mocked LLM, offline grounding)."""
from __future__ import annotations

import pytest

from src.assistant import orchestrator
from src.assistant.orchestrator import answer
from src.assistant.schema import AssistantRequest, PageContext

from tests.assistant.conftest import FakeCompletion


@pytest.fixture(autouse=True)
def _hermetic(monkeypatch) -> None:
    # avoid hitting the DB for live counters, and make feature flags explicit
    # (config now ships enabled; individual tests opt features back on).
    monkeypatch.setattr("src.assistant.grounding.context_builder.live_summary", lambda: "")
    monkeypatch.setattr("src.assistant.config.feature", lambda name: False)


def test_instructions_route_via_heuristic(patch_llm_client) -> None:
    patch_llm_client([FakeCompletion("Щоб передати замовлення у виробництво, натисніть «Передати».")])
    res = answer(AssistantRequest(
        message="як передати у виробництво?",
        page_context=PageContext(route="/workflow", stage="vyrobnytstvo"),
    ))
    assert res.route == "instructions"
    assert "виробництво" in res.text_md
    assert res.tool_trace == []


def test_data_query_synthesizes_from_grounding(patch_llm_client) -> None:
    patch_llm_client([FakeCompletion("Зараз очікують планування кілька замовлень.")])
    res = answer(AssistantRequest(
        message="скільки замовлень очікують?",
        page_context=PageContext(route="/workflow", stage="prodazhi"),
    ))
    assert res.route == "data_query"
    assert res.tool_trace == []  # tools/text2sql off by default
    assert res.text_md


def test_out_of_scope_returns_refusal(patch_llm_client) -> None:
    patch_llm_client([FakeCompletion('{"route": "out_of_scope", "refusal": "Вибач, не можу."}')])
    res = answer(AssistantRequest(message="купи мені акції tesla"))
    assert res.route == "out_of_scope"
    assert "Вибач" in res.text_md


def test_schedule_action_delegates_to_module_08(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.assistant.orchestrator.run_schedule_command",
        lambda message, **kw: "Завантаження машин оновлено.",
    )
    res = answer(AssistantRequest(message="перебудуй розклад"))
    assert res.route == "schedule_action"
    assert res.text_md == "Завантаження машин оновлено."


def test_screen_vision_disabled_prompts_to_enable() -> None:
    res = answer(AssistantRequest(message="що тут на екрані?", screenshot_b64="Zm9v"))
    assert res.route == "screen_vision"
    assert "Прочитати екран" in res.text_md


def test_screen_vision_enabled_calls_vision(monkeypatch) -> None:
    monkeypatch.setattr("src.assistant.config.feature", lambda name: name == "vision")
    monkeypatch.setattr("src.assistant.vision.describe", lambda *a, **k: "На екрані графік завантаження машин.")
    res = answer(AssistantRequest(message="поясни графік", screenshot_b64="Zm9v"))
    assert res.route == "screen_vision"
    assert res.text_md == "На екрані графік завантаження машин."


def test_screen_vision_gathers_data_before_vision(monkeypatch, patch_llm_client) -> None:
    # Regression for the live-UI gap: the camera path must call data-tools and feed
    # verified data to vision — facts from the system, not from the JPEG. (#12345)
    import json
    from types import SimpleNamespace

    def _tc(name, args):
        return SimpleNamespace(id="c1", type="function",
                               function=SimpleNamespace(name=name, arguments=json.dumps(args)))

    patch_llm_client([
        FakeCompletion(None, tool_calls=[_tc("order_risk", {"order_id": 12345})]),
        FakeCompletion("(stop)", tool_calls=None),
    ])
    monkeypatch.setattr("src.assistant.config.feature",
                        lambda n: n in {"tools", "tool_calling", "vision"})
    monkeypatch.setattr(
        "src.assistant.data.tools.run_tool",
        lambda name, params=None, **k: {
            "tool": name, "columns": [], "error": None,
            "rows": [{"order_id": 12345, "customer_name": "Рошен", "plan_kg": 2994,
                      "total_deficit_kg": 2832, "coverage_status": "full"}],
        },
    )
    monkeypatch.setattr("src.assistant.grounding.context_builder.live_summary", lambda: "")
    captured: dict = {}

    def _fake_vision(b64, pc, *, message="", grounded="", usage=None):
        captured["grounded"] = grounded
        return "ok"

    monkeypatch.setattr("src.assistant.vision.describe", _fake_vision)

    req = AssistantRequest(
        message="який статус і дефіцит у замовлення на екрані?",
        page_context=PageContext(route="/workflow", stage="vyrobnytstvo",
                                  filters={"order_id_like": "12345"}),
        screenshot_b64="Zm9v",
    )
    res = answer(req)
    assert res.route == "screen_vision"
    assert "Рошен" in captured["grounded"]      # verified data reached vision
    assert res.tool_trace                        # ≥1 data-tool call on the camera path


def test_agentic_gather_executes_model_chosen_tools(monkeypatch, patch_llm_client) -> None:
    import json
    from types import SimpleNamespace

    from src.assistant.llm import LLMUsage

    def _tc(name, args, cid="c1"):
        return SimpleNamespace(
            id=cid, type="function",
            function=SimpleNamespace(name=name, arguments=json.dumps(args)),
        )

    # round 1: model asks to call get_deficits_top(limit=3); round 2: no tool calls → stop
    patch_llm_client([
        FakeCompletion(None, tool_calls=[_tc("get_deficits_top", {"limit": 3})]),
        FakeCompletion("(stop)", tool_calls=None),
    ])
    monkeypatch.setattr(
        "src.assistant.data.tools.run_tool",
        lambda name, params=None, **k: {
            "tool": name, "columns": [], "error": None,
            "rows": [{"sku": "2.01.51045", "total_deficit_kg": 13048.6}],
        },
    )
    req = AssistantRequest(message="топ дефіцитів?", page_context=PageContext(route="/workflow"))
    evidence, trace = orchestrator._agentic_gather(req, "GROUND", LLMUsage())
    assert "2.01.51045" in evidence            # verified data folded into evidence
    assert trace and trace[0]["tool"] == "get_deficits_top"
    assert trace[0]["args"] == {"limit": 3}     # model-chosen args executed


def test_route_tools_adds_material_and_deficit_tools() -> None:
    # a named SKU → get_material first
    assert orchestrator._route_tools("data_query", "що по 2.01.51045?")[0] == "get_material"
    # deficit data-query → get_deficits_top included (real codes, not OCR)
    assert "get_deficits_top" in orchestrator._route_tools("data_query", "топ дефіцитів?")
    # plain count question → unchanged default
    assert orchestrator._route_tools("data_query", "скільки замовлень?") == ["pending_orders_count"]


def test_tool_params_carries_sku_and_limit() -> None:
    req = AssistantRequest(message="залишок по 2.01.51045?",
                           page_context=PageContext(route="/workflow", stage="zabezpechennia"))
    p = orchestrator._tool_params(req)
    assert p["sku"] == "2.01.51045"
    assert p["limit"] == 10


def test_contextualize_rewrites_followup_with_history(patch_llm_client) -> None:
    from src.assistant.llm import LLMUsage

    client = patch_llm_client([FakeCompletion("дай повний перелік BOPP плівок виробництва ПЛАСТХІМ")])
    req = AssistantRequest(
        message="Дай повний перелік BOPP Films",
        history=[
            {"role": "user", "text": "Дай перелік плівок виробництва ПЛАСТХІМ"},
            {"role": "assistant", "text": "Plastchim виробляє BOPP, CPP, BOPE плівки…"},
        ],
    )
    out = orchestrator._contextualize_query(req, LLMUsage())
    assert "ПЛАСТХІМ" in out.message            # follow-up now carries the producer context
    assert len(client.calls) == 1               # one cheap-model rewrite call


def test_contextualize_noop_without_history() -> None:
    from src.assistant.llm import LLMUsage

    req = AssistantRequest(message="Дай перелік плівок ПЛАСТХІМ")  # no history
    out = orchestrator._contextualize_query(req, LLMUsage())
    assert out is req                            # unchanged, no LLM call


def test_contextualize_skips_screenshot_turn() -> None:
    from src.assistant.llm import LLMUsage

    req = AssistantRequest(message="що тут?", screenshot_b64="Zm9v",
                           history=[{"role": "user", "text": "попереднє"}])
    assert orchestrator._contextualize_query(req, LLMUsage()) is req


def test_kb_mode_forces_external_kb_consult(monkeypatch) -> None:
    # operator_help returns a HIGH-score hit (would normally block escalation),
    # but «База знань» mode must consult the external KB anyway and skip the veto.
    from types import SimpleNamespace

    monkeypatch.setattr("src.assistant.config.feature", lambda n: n in {"rag", "external_kb"})

    class _Retr:
        available = True

        def retrieve(self, q):
            return [(SimpleNamespace(source="operator_help", text="загальна довідка", url=None), 0.99)]

    monkeypatch.setattr("src.assistant.rag.index.get_retriever", lambda: _Retr())
    seen = {}

    def _fake_search(query, usage=None, *, force=False):
        seen["force"] = force
        return SimpleNamespace(knowledge="FXCW datasheet specs", citations=[], best_score=0.6)

    monkeypatch.setattr(orchestrator, "_search_external_kb", _fake_search)

    req = AssistantRequest(message="характеристики плівки FXCW", mode="kb")
    evidence, _citations, kb_used = orchestrator._gather_instructions(req, "GROUND")
    assert kb_used is True               # consulted despite the high operator_help score
    assert seen.get("force") is True     # relevance veto skipped in kb mode
    assert "FXCW datasheet specs" in evidence


def test_apply_mode_forces_route() -> None:
    # data mode → tools/DB path; kb mode → RAG/KB path; hybrid keeps router choice
    assert orchestrator._apply_mode("data", "instructions") == "data_query"
    assert orchestrator._apply_mode("data", "out_of_scope") == "data_query"
    assert orchestrator._apply_mode("kb", "data_query") == "instructions"
    assert orchestrator._apply_mode("kb", "analysis") == "instructions"
    assert orchestrator._apply_mode("hybrid", "data_query") == "data_query"
    # camera + commands are never forced by the mode toggle
    assert orchestrator._apply_mode("data", "screen_vision") == "screen_vision"
    assert orchestrator._apply_mode("kb", "schedule_action") == "schedule_action"


def test_data_mode_routes_instructions_question_to_tools(monkeypatch, patch_llm_client) -> None:
    # In «дані системи» mode, even an instructions-style question goes to the data
    # path (no RAG). Heuristic routes "як ..." to instructions; mode forces data_query.
    patch_llm_client([FakeCompletion("Відповідь за даними.")])
    monkeypatch.setattr("src.assistant.config.feature", lambda n: False)  # tools off → no agentic, just synth
    res = answer(AssistantRequest(message="як підтвердити матеріали?", mode="data",
                                  page_context=PageContext(route="/workflow", stage="prodazhi")))
    assert res.route == "data_query"   # forced away from instructions by data mode


def test_strip_unverified_sku_redacts_invented_code() -> None:
    ev = "tool rows ... \"sku\": \"2.01.51045\" ..."
    # a SKU present in the evidence stays
    assert "2.01.51045" in orchestrator._strip_unverified_ids("Матеріал 2.01.51045 у дефіциті.", ev)
    # a one-digit-off SKU not in evidence (classic OCR distortion) is redacted
    out = orchestrator._strip_unverified_ids("Матеріал 2.01.51845 у дефіциті.", ev)
    assert "2.01.51845" not in out
    assert "уточніть у таблиці" in out


def test_strip_unverified_ignores_dates_and_guards_family6() -> None:
    # dates (DD.MM.YYYY) must not be touched
    assert "27.07.2026" in orchestrator._strip_unverified_ids("Строк 27.07.2026.", "")
    # family-6 SKU format is also guarded when unverified
    assert "6.0809.133" not in orchestrator._strip_unverified_ids("Беремо 6.0809.133.", "")


def test_strip_unverified_guards_order_number() -> None:
    # #N present in evidence/question stays (e.g. honest "order #12345 not found")
    assert "#12345" in orchestrator._strip_unverified_ids(
        "Замовлення #12345 не знайдено.", "питання про 12345")
    # a hallucinated #N not in evidence/question is redacted
    out = orchestrator._strip_unverified_ids("Дивись замовлення #99999.", "evidence без нього")
    assert "#99999" not in out
    assert "уточніть" in out


def test_named_order_overrides_open_card() -> None:
    # Open card #15499, but the operator asks about #12345 → use #12345.
    req = AssistantRequest(
        message="а що з замовленням #12345?",
        page_context=PageContext(route="/workflow/actions", selected_order="15499"),
    )
    assert orchestrator._effective_order_id(req) == 12345
    assert orchestrator._tool_params(req)["order_id"] == 12345


def test_open_card_used_when_no_order_named() -> None:
    req = AssistantRequest(
        message="що тут робити?",
        page_context=PageContext(route="/workflow/actions", selected_order="15499"),
    )
    assert orchestrator._effective_order_id(req) == 15499


def test_cache_skipped_when_order_card_open(monkeypatch) -> None:
    # Cache key excludes the open order — an order-grounded answer must NOT be
    # cached/replayed across different orders.
    monkeypatch.setattr("src.assistant.config.feature", lambda name: name == "cache")

    class _Boom:
        def lookup(self, *a, **k):
            raise AssertionError("cache lookup must be skipped when a card is open")

        def store(self, *a, **k):
            raise AssertionError("cache store must be skipped when a card is open")

    monkeypatch.setattr(orchestrator, "get_cache", lambda: _Boom())
    with_focus = AssistantRequest(
        message="що зробити?",
        page_context=PageContext(route="/workflow/portfolio", selected_order="15500"),
    )
    assert orchestrator._cache_lookup(with_focus) is None
    orchestrator._cache_store(with_focus, "instructions", "txt", [])  # must not raise


def test_cache_used_when_no_order_open(monkeypatch) -> None:
    monkeypatch.setattr("src.assistant.config.feature", lambda name: name == "cache")
    seen = {}

    class _Rec:
        def lookup(self, msg, key):
            seen["lookup"] = (msg, key)
            return None

        def store(self, msg, key, payload):
            seen["store"] = (msg, key)

    monkeypatch.setattr(orchestrator, "get_cache", lambda: _Rec())
    req = AssistantRequest(message="як передати?", page_context=PageContext(route="/workflow"))
    orchestrator._cache_lookup(req)
    orchestrator._cache_store(req, "instructions", "txt", [])
    assert "lookup" in seen and "store" in seen
