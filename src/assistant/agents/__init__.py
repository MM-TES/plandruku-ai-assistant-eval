"""Multi-agent answer layer (INC-5/6) — optional, config-gated stages wrapped
around the KB-answer path. Active only when ``features.multi_agent`` is true; the
orchestrator happy path is otherwise unchanged (instant rollback).

INC-5: controller (per-query stage plan) + answer_critic (completeness / no-invention
self-refine) + a spec-complete answer mode for datasheet "all characteristics" queries.
INC-6 adds query_planner + retrieval_critic.
"""
