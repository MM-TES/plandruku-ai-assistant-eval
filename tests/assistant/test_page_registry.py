"""Phase 2 gate: page-context registry + grounded-context builder."""
from __future__ import annotations

from src.assistant import config
from src.assistant.grounding import context_builder, page_registry
from src.assistant.schema import PageContext


def test_every_configured_page_has_desc_and_prompts() -> None:
    pages = config.pages()
    assert "default" in pages
    for key, page in pages.items():
        assert page.get("desc"), f"page {key} missing desc"
        assert isinstance(page.get("prompts"), list) and page["prompts"], f"page {key} missing prompts"


def test_describe_falls_back_to_default() -> None:
    d = page_registry.describe(PageContext(route="/totally-unknown"))
    assert d == config.pages()["default"]


def test_workflow_stage_resolves_specific_page() -> None:
    d = page_registry.describe(PageContext(route="/workflow", stage="zabezpechennia"))
    assert "Забезпечення" in d["desc"]


def test_context_builder_nonempty_and_includes_ids() -> None:
    ctx = context_builder.build(
        PageContext(route="/workflow", stage="materialy", visible_entity_ids=["111", "222"]),
        include_live=False,
    )
    assert "materialy" in ctx
    assert "Матеріали" in ctx
    assert "111" in ctx and "222" in ctx
