"""Build a SMALL isolated KB index for datasheet-depth eval (INC-0).

Indexes the supplier DATASHEET corpus (all datasheets, so the 6 golden products
compete with ~34 distractors that share template values — reproducing the
"buried among many" retrieval problem) into ``models/knowledge_base_rag_eval/``
WITHOUT touching the production index (``models/knowledge_base_rag/``). It reads
the SAME ``kb.*`` flags and the SAME embedder as production, so the harness measures
the real code path. Rebuild takes seconds, not the 16-min full corpus build.

    python -m src.assistant.eval.kbdepth.build_eval_index
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_ROOT))

from src.assistant import config  # noqa: E402
from src.assistant.eval.kbdepth.golden import load_golden  # noqa: E402
from src.assistant.kb import corpus  # noqa: E402
from src.assistant.kb.index import build_index  # noqa: E402
from src.utils.logger import setup_logger  # noqa: E402

_logger = setup_logger("kbdepth.build")

EVAL_PERSIST_DIR = _ROOT / "models" / "knowledge_base_rag_eval"
# Datasheet corpora to index (golden products + distractors). flexfilm nests products
# under category subdirs (pdf/Products/BOPET/F-CHC.pdf), so that glob must recurse.
_DATASHEET_GLOBS = ["sites/*/pdf/datasheet/*.pdf", "sites/*/pdf/Products/**/*.pdf"]
# Catalog-wide guides / safety sheets are not per-product datasheets — skip (huge,
# and they'd let a query "find" numbers without retrieving the right datasheet).
_SKIP_NAME = ("reference-guide", "safety-data-sheet", "web-inspection", "pr_")


def collect_paths(kb_root: Path, golden_sources: list[str]) -> list[Path]:
    """All per-product datasheet PDFs under the KB root, guaranteeing every golden source is in."""
    paths: dict[str, Path] = {}
    for pat in _DATASHEET_GLOBS:
        for p in kb_root.glob(pat):
            if p.is_file() and not any(s in p.name.lower() for s in _SKIP_NAME):
                paths[str(p.resolve())] = p
    for src in golden_sources:  # golden sources always included, even if name-skipped
        p = kb_root / src
        if p.is_file():
            paths[str(p.resolve())] = p
    return sorted(paths.values())


def build(persist_dir: Path = EVAL_PERSIST_DIR) -> dict:
    kb_root = Path(config.kb_path())
    if not kb_root.is_dir():
        _logger.error("KB path not found: %s", kb_root)
        return {"eval_chunks": 0, "files": 0}
    golden = load_golden()
    sources = [g.source for g in golden]
    paths = collect_paths(kb_root, sources)
    if not paths:
        _logger.error("No datasheet PDFs found under %s (globs=%s)", kb_root, _DATASHEET_GLOBS)
        return {"eval_chunks": 0, "files": 0}
    _logger.info("kbdepth eval index: %d datasheet files (golden=%d) from %s",
                 len(paths), len(sources), kb_root)
    chunks, stats = corpus.build_chunks(kb_root, only_paths=[str(p) for p in paths])
    n = build_index(chunks, persist_dir=persist_dir, show_progress=False)
    _logger.info("kbdepth eval index built: %d chunks → %s", n, persist_dir)
    return {"eval_chunks": n, "files": len(paths), "persist_dir": str(persist_dir),
            "by_ext": stats.get("by_ext", {})}


def main() -> int:
    out = build()
    print(out)
    return 0 if out.get("eval_chunks", 0) > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
