"""Run the offline dual-judge calibration on real assistant answers (Block 4b).

Generates answers for a sample of golden queries via orchestrator.answer(), then scores
each with two judge families and reports per-criterion inter-rater agreement (Spearman ρ).
Writes reports/judge_calibration/<stamp>_<label>.json.

    python scripts/run_judge_calibration.py --n 10
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.assistant import config  # noqa: E402
from src.assistant.eval import judge_calibration  # noqa: E402
from src.assistant.eval.kbdepth.golden import load_golden  # noqa: E402
from src.assistant.llm import LLMUsage  # noqa: E402
from src.utils.logger import setup_logger  # noqa: E402

_logger = setup_logger("run_judge_calibration")
_DEFAULT_GOLDEN = Path(__file__).resolve().parents[1] / "config" / "kb_golden_50.jsonl"


def _generate_examples(items: list, usage: LLMUsage) -> list[dict]:
    """Answer each golden query end-to-end (the COSTED part — cache and reuse)."""
    from src.assistant import orchestrator
    from src.assistant.schema import AssistantRequest, PageContext

    examples: list[dict] = []
    for it in items:
        try:
            resp = orchestrator.answer(AssistantRequest(message=it.queries[0],
                                                        page_context=PageContext(route="/")))
            examples.append({"id": it.product, "query": it.queries[0],
                             "answer": resp.text_md or "", "trace": resp.tool_trace,
                             "context": getattr(resp, "evidence", "") or ""})
        except Exception as exc:  # noqa: BLE001
            _logger.warning("answer failed for %s: %s", it.product, str(exc)[:120])
    return examples


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Dual-judge calibration on real answers (Block 4b).")
    ap.add_argument("--n", type=int, default=10, help="how many golden queries to sample")
    ap.add_argument("--golden", default=str(_DEFAULT_GOLDEN))
    ap.add_argument("--label", default="capstone_judge_calib")
    ap.add_argument("--seed", type=int, default=0,
                    help="m2: deterministic item-sample shuffle (different seeds → different "
                         "samples for the two-consecutive-pass stopping rule)")
    ap.add_argument("--stratified", action="store_true",
                    help="m2: build a QUALITY-SPREAD sample (answerable + abstention items) so "
                         "Spearman ρ is computable — a uniform answerable sample compresses scores "
                         "into a high band and makes rank-correlation degenerate (RND_FINDINGS Q3)")
    ap.add_argument("--abstain-frac", type=float, default=0.33,
                    help="fraction of the stratified sample that is abstention items (refusal band)")
    ap.add_argument("--reuse-answers", default=None, metavar="PATH",
                    help="m2: answers-cache JSON. Exists → judge THOSE answers (judges-only "
                         "cost, ~10x cheaper rubric iteration); missing → generate answers "
                         "once and write the cache.")
    ap.add_argument("--judge-b-model", default=None,
                    help="override models.judge_b for this run (in-process; tests a stronger "
                         "cross-family second judge without committing config)")
    args = ap.parse_args(argv)

    if not config.openrouter_api_key():
        _logger.error("OPENROUTER_API_KEY absent — calibration needs it.")
        return 2

    if args.judge_b_model:
        config._cfg()["models"]["judge_b"] = args.judge_b_model
        _logger.info("judge_b overridden → %s", args.judge_b_model)

    usage = LLMUsage()
    cache = Path(args.reuse_answers) if args.reuse_answers else None
    if cache and cache.is_file():
        examples = json.loads(cache.read_text(encoding="utf-8"))[: args.n]
        _logger.info("Reusing %d cached answers from %s (judges-only run)", len(examples), cache)
    else:
        rng = random.Random(args.seed)
        if args.stratified:
            # Quality-SPREAD sample: abstention items (refusal band) + answerable items
            # (which already span retrieval quality). A flat answerable-only sample
            # compresses judge scores → degenerate Spearman (RND_FINDINGS Q3).
            all_items = load_golden(args.golden)
            abst = [it for it in all_items if it.abstain_expected]
            answ = [it for it in all_items if it.kind in ("science", "datasheet")
                    and not it.abstain_expected]
            rng.shuffle(abst); rng.shuffle(answ)
            n_abst = min(len(abst), round(args.n * args.abstain_frac))
            items = abst[:n_abst] + answ[: args.n - n_abst]
            rng.shuffle(items)
            _logger.info("stratified sample: %d abstention + %d answerable",
                         n_abst, len(items) - n_abst)
        else:
            # Prefer non-abstain, answerable science items (a judge needs a real answer).
            items = [it for it in load_golden(args.golden)
                     if it.kind == "science" and not it.abstain_expected]
            rng.shuffle(items)
            items = items[: args.n]
        examples = _generate_examples(items, usage)
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(examples, ensure_ascii=False, indent=2), encoding="utf-8")
            _logger.info("Wrote answers cache (%d examples) → %s", len(examples), cache)

    payload = judge_calibration.run(examples, usage=usage, label=args.label)
    _logger.info("calibration: overall ρ=%s weak=%s cost=$%.4f",
                 payload["agreement"]["overall"]["spearman"], payload["weak_criteria"],
                 payload["cost_usd"])
    ov = payload["agreement"]["overall"]
    ci = ov.get("spearman_ci95")
    print(f"\n  judge calibration (n={payload['n']}):")
    print(f"  overall: weighted κ = {ov.get('weighted_kappa')}  | faithfulness κ "
          f"(correct+grounded) = {ov.get('faithfulness_kappa')}")
    print(f"  overall Spearman ρ = {ov['spearman']}"
          + (f" (95% CI {ci[0]}..{ci[1]}; degenerate under anchored ties)" if ci else ""))
    for c in ("complete", "correct", "grounded", "useful"):
        a = payload["agreement"][c]
        print(f"    {c:<9} κ={a.get('weighted_kappa')}  ρ={a['spearman']}  "
              f"MAD={a['mean_abs_diff']}  exact={a['pct_within_0.2']}")
    print(f"  weak criteria (by κ<0.4 / exact<0.6): {payload['weak_criteria'] or 'none'}")
    print(f"  recommendation: {payload['recommendation']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
