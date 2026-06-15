"""m2 — freeze the stratified Layer-B eval subset (LB-120) from the frozen goldens.

Layer-B (answers + judge) on the full 300-item golden costs ~$7.5-17/run — the $20
mission budget can't afford that per measurement. This script deterministically selects
a PRE-REGISTERED stratified subset (documented budget deviation from the TZ's full-300
Layer-B): KB answerable items proportional to their category mix, all-or-sampled
abstention items (off-domain + near-domain traps), and operator-help items. The output
is a normal golden-format jsonl — the harness ``--golden`` flag consumes it directly.

    python scripts/build_eval_subset.py --kb config/kb_golden_300.jsonl ^
        --operator config/operator_golden_50.jsonl --out config/eval_subset_120.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.assistant.eval.kbdepth import golden_gen  # noqa: E402
from src.utils.logger import setup_logger  # noqa: E402

_logger = setup_logger("build_eval_subset")


def _load(path: str) -> list[dict]:
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("//"):
            out.append(json.loads(line))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Freeze the LB-120 Layer-B subset (m2).")
    ap.add_argument("--kb", required=True, help="frozen KB golden (300)")
    ap.add_argument("--operator", default=None, help="frozen operator golden (50)")
    ap.add_argument("--out", default="config/eval_subset_120.jsonl")
    ap.add_argument("--n-kb", type=int, default=75)
    ap.add_argument("--n-abstain", type=int, default=25)
    ap.add_argument("--n-operator", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    rng = random.Random(args.seed)

    kb = _load(args.kb)
    answerable = [it for it in kb if not it.get("abstain_expected")]
    abstain = [it for it in kb if it.get("abstain_expected")]

    # proportional-by-category sample of answerable items
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for it in answerable:
        by_cat[str(it.get("category") or "?")].append(it)
    take: list[dict] = []
    total = len(answerable)
    for cat, items in sorted(by_cat.items()):
        items = sorted(items, key=lambda d: str(d.get("product")))
        rng.shuffle(items)
        quota = max(1, round(args.n_kb * len(items) / max(1, total)))
        take.extend(items[:quota])
    rng.shuffle(take)
    take = take[: args.n_kb]

    abstain = sorted(abstain, key=lambda d: str(d.get("product")))
    rng.shuffle(abstain)
    take_abstain = abstain[: args.n_abstain]

    take_op: list[dict] = []
    if args.operator:
        ops = _load(args.operator)
        ops = sorted(ops, key=lambda d: str(d.get("product")))
        rng.shuffle(ops)
        take_op = ops[: args.n_operator]

    subset = take + take_abstain + take_op
    out = golden_gen.emit_jsonl(subset, args.out)
    from collections import Counter

    cats = Counter(str(it.get("category")) for it in subset)
    kinds = Counter(str(it.get("kind")) for it in subset)
    print(f"\n  LB subset: {len(subset)} items → {out}")
    print(f"  kinds={dict(kinds)}")
    print(f"  categories={dict(cats)}")
    print(f"  seed={args.seed} (pre-registered; record in RESUME.md)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
