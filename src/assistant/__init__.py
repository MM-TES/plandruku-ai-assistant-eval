"""In-app AI help-bubble assistant for PlanDruku APS.

A global "?" assistant (replacing the dead-end /chat tab) that combines intent
routing, structured page-context grounding, on-demand screenshot+vision,
read-only data tools + guarded text2SQL, RAG over project docs, LangSmith
observability and user feedback.

Architecture (5 layers): widget -> orchestrator -> router -> skills -> observability.
All behaviour is config-driven via ``config/assistant.json`` (no hardcode here);
every capability is behind an independent feature flag for rollback.

See ``R_and_D/assistant_rnd/`` for the R&D report, TZ and schema-card.
"""

__all__: list[str] = []
