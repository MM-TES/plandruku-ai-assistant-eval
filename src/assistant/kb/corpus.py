"""Walk the knowledge base, extract + chunk all supported files, dedup, keep metadata."""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator

from src.assistant.kb import loaders
from src.utils.logger import setup_logger

_logger = setup_logger(__name__)

_CHUNK_SIZE = 900
_CHUNK_OVERLAP = 120
_MIN_CHUNK = 40
_SKIP_DIRS = {".git", "__pycache__", "node_modules"}


def _norm_path(p: str | Path) -> str:
    """Case-/separator-normalized absolute path for set membership (Windows-safe)."""
    return os.path.normcase(os.path.abspath(str(p)))


@dataclass
class KBChunk:
    id: str
    source: str   # path relative to the KB root (friendly citation)
    path: str     # absolute path
    locator: str  # page/slide/sheet within the file
    text: str
    # P1.2 metadata (defaults keep old kb_chunks.json loadable via KBChunk(**c)).
    product: str | None = None      # datasheet product code, e.g. "FXCMT"
    doc_type: str | None = None     # datasheet | page | literature | scanned | None
    supplier: str | None = None     # host, e.g. "plastchim.ua"
    parent_id: str | None = None    # document id (== source relpath); P1.3 merges on this
    section: str | None = None      # T1.2 detected section heading (when kb.structured_pdf)


def _looks_like_code(stem: str) -> bool:
    """A filename stem that looks like a product code (FXCMT, TATRAFAN_SHT, F-HBP)."""
    s = stem.strip()
    return (len(s) >= 3 and s.isascii() and any(c.isalpha() for c in s)
            and all(c.isalnum() or c in "_-" for c in s))


def _chunk_meta(rel: str) -> dict:
    """Derive {product, doc_type, supplier} from a KB-relative path. Conservative:
    ``product`` is set ONLY for datasheet paths whose stem looks like a code, so
    generic pages/literature never get a spurious product scope."""
    parts = rel.split("/")
    low = rel.lower()
    supplier = parts[1] if len(parts) >= 2 and parts[0] == "sites" else None
    if "/datasheet/" in low or "/products/" in low:
        doc_type = "datasheet"
    elif parts and parts[0] == "scanned":
        doc_type = "scanned"
    elif parts and parts[0] == "sites":
        doc_type = "page"
    elif "література" in low or "literature" in low:
        doc_type = "literature"
    else:
        doc_type = None
    product = None
    if doc_type == "datasheet":
        stem = Path(rel).stem
        if _looks_like_code(stem):
            product = stem.upper()
    return {"product": product, "doc_type": doc_type, "supplier": supplier}


def _splitter():
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    return RecursiveCharacterTextSplitter(
        chunk_size=_CHUNK_SIZE, chunk_overlap=_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", " ", ""],
    )


# T1.2: a heading line — numbered ("5.2 Tone Reproduction") or a short ALL-CAPS title.
# Dependency-free (no Docling/GROBID); good enough to keep a long standard/patent's
# technical sections together and label them, so a section (not the cover) is retrievable.
_HEADING_RE = re.compile(
    r"^(?:\d+(?:\.\d+){0,3}\.?\s+[A-Za-zА-Яа-яЇІЄҐ][^\n]{2,68}"
    r"|[A-ZА-ЯЇІЄҐ][A-ZА-ЯЇІЄҐ0-9 \-/]{4,58})$"
)


def _split_sections(text: str) -> list[tuple[str | None, str]]:
    """Split *text* into ``(heading, body)`` sections on detected heading lines.
    Returns ``[(None, text)]`` when no headings are found (e.g. a short datasheet),
    so the structured path is a no-op there and the datasheet eval is unchanged."""
    sections: list[tuple[str | None, str]] = []
    cur_head: str | None = None
    cur: list[str] = []
    for ln in (text or "").split("\n"):
        s = ln.strip()
        if s and len(s.split()) <= 10 and _HEADING_RE.match(s):
            if cur:
                sections.append((cur_head, "\n".join(cur)))
                cur = []
            cur_head = s
        else:
            cur.append(ln)
    if cur:
        sections.append((cur_head, "\n".join(cur)))
    return sections or [(None, text or "")]


def _structured_enabled() -> bool:
    from src.assistant import config

    v = config.kb_param("structured_pdf", {})
    return bool(v.get("enabled", False)) if isinstance(v, dict) else bool(v)


def iter_files(
    root: Path, *, include_doc: bool, ocr: bool,
    only_paths: Iterable[str | Path] | None = None,
) -> Iterator[Path]:
    """Yield supported files under *root*. When *only_paths* is given, restrict to
    exactly those files (normalized) — used by the kbdepth eval index to index a
    subset (datasheets) while keeping relpaths/metadata relative to the real KB root."""
    from src.assistant import config

    op: set[str] | None = {_norm_path(p) for p in only_paths} if only_paths is not None else None
    max_bytes = int(config.kb_param("max_file_mb", 80)) * 1024 * 1024
    # config-driven excludes for source folders whose content is indexed via a
    # converted artifact instead (e.g. "WORD" originals → WORD_MD structured Markdown).
    skip_dirs = _SKIP_DIRS | {str(d) for d in config.kb_param("build_skip_dirs", [])}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if op is not None and _norm_path(path) not in op:
            continue
        if any(part in skip_dirs for part in path.parts):
            continue
        if path.name.startswith("~$") or path.name == ".DS_Store":
            continue
        if not loaders.supported(path.suffix, include_doc=include_doc, ocr=ocr):
            continue
        # Skip oversized media-heavy files (e.g. 700 MB PPTX with embedded video):
        # extraction would hang/OOM for little text. Raise kb.max_file_mb to include.
        try:
            if path.stat().st_size > max_bytes:
                _logger.info("KB skip oversized (%d MB): %s",
                             path.stat().st_size // (1024 * 1024), path.name)
                continue
        except OSError:
            continue
        yield path


def build_chunks(
    root: Path,
    *,
    include_doc: bool = False,
    ocr: bool = False,
    ocr_langs: str = "ukr+rus",
    max_files: int | None = None,
    only_paths: Iterable[str | Path] | None = None,
    progress_cb: Callable[[int, int, str], None] | None = None,
) -> tuple[list[KBChunk], dict]:
    """Return (chunks, stats). Exact-dedup by chunk hash; never raises per file.

    *only_paths* restricts ingestion to a given subset of files (kbdepth eval index)
    while keeping *root* as the KB root so source relpaths stay production-identical.
    """
    split = _splitter()
    structured = _structured_enabled()   # T1.2: section-aware chunking + contextual prefix
    chunks: list[KBChunk] = []
    seen: set[str] = set()
    stats: dict = {"files_ok": 0, "files_empty": 0, "files_error": 0,
                   "by_ext": {}, "errors": []}

    files = list(iter_files(root, include_doc=include_doc, ocr=ocr, only_paths=only_paths))
    if max_files:
        files = files[:max_files]
    total = len(files)
    for n, path in enumerate(files, start=1):
        ext = path.suffix.lower()
        stats["by_ext"][ext] = stats["by_ext"].get(ext, 0) + 1
        try:
            segments = loaders.extract(path, include_doc=include_doc, ocr=ocr, ocr_langs=ocr_langs)
        except Exception as exc:  # noqa: BLE001 — one bad file must not stop the build
            stats["files_error"] += 1
            stats["errors"].append({"file": str(path), "error": str(exc)[:200]})
            _logger.info("KB load failed %s: %s", path.name, str(exc)[:120])
            if progress_cb:
                progress_cb(n, total, path.name)
            continue

        try:
            rel = str(path.relative_to(root)).replace("\\", "/")
        except ValueError:
            rel = path.name

        meta = _chunk_meta(rel)
        # T1.2: section-aware chunking ONLY for long prose docs (literature / standards /
        # patents = scanned), NEVER datasheets or site pages — those stay flat so the
        # datasheet eval is byte-identical (a datasheet's ALL-CAPS property labels would
        # otherwise be mis-detected as headings and fragment the spec table).
        use_structured = structured and meta["doc_type"] not in ("datasheet", "page")
        produced = 0
        for seg in segments:
            units = _split_sections(seg.text or "") if use_structured else [(None, seg.text or "")]
            for section, body in units:
                # Contextual prefix: a chunk from a long standard/patent carries its
                # section heading, so it is retrievable as "the press-fingerprint section"
                # rather than an orphaned page.
                prefix = f"[{section}]\n" if (use_structured and section) else ""
                for piece in split.split_text(body):
                    piece = piece.strip()
                    if len(piece) < _MIN_CHUNK:
                        continue
                    text_piece = f"{prefix}{piece}" if prefix else piece
                    h = hashlib.sha256(text_piece.encode("utf-8")).hexdigest()
                    if h in seen:
                        continue
                    seen.add(h)
                    chunks.append(KBChunk(
                        id=f"{len(chunks)}", source=rel, path=str(path),
                        locator=seg.locator, text=text_piece,
                        product=meta["product"], doc_type=meta["doc_type"],
                        supplier=meta["supplier"], parent_id=rel, section=section,
                    ))
                    produced += 1
        stats["files_ok" if produced else "files_empty"] += 1
        if progress_cb:
            progress_cb(n, total, path.name)

    stats["n_files"] = total
    stats["n_chunks"] = len(chunks)
    return chunks, stats


__all__ = ["KBChunk", "build_chunks", "iter_files"]
