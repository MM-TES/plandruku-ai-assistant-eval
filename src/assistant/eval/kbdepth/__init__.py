"""KB datasheet-depth evaluation (P0 / INC-0).

Autonomous, mostly-free harness that measures whether a datasheet query returns
the product's NUMERIC specifications (depth), not just prose. Layer A (numeric
context-recall) is local/deterministic/free; Layer B (answer depth) is optional
and costed. See ``harness.py``.
"""
