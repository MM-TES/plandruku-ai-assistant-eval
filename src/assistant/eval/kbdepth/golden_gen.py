"""Block 1 (capstone) — generate a KB-grounded golden Q&A set for the eval harness.

Pipeline (the anti-hallucinated-golden contract):
1. Load REAL chunks from the production index's ``kb_chunks.json`` — no embedding /
   no API for sampling, just the stored ``{source, text, product, doc_type, …}``.
2. STRATIFY across ``doc_type`` (datasheet / literature / scanned-patent / site page /
   other) so the set mirrors the corpus, with a deterministic ``random.Random(seed)``.
3. For each sampled chunk (or, for datasheets, the whole parent document) the answer
   model DRAFTS one grounded item — uk+en query, a short reference answer, key claims
   and (for datasheets) the spec numbers — using ONLY that text as its source.
4. MECHANICALLY VERIFY: every drafted number must actually occur in the source text
   (``golden.numbers_in``); hallucinated numbers are dropped. The source path is pinned
   from the real chunk, never invented.
5. Add a few fixed OUT-OF-SCOPE prompts (``abstain_expected=true``) so the set also
   measures correct refusal.
6. Emit a reviewable JSONL in the existing ``GoldenItem`` schema (``// REVIEW:`` lines
   for operator spot-check; ``//`` lines are skipped by ``golden.load_golden``).

The emitted file is loadable by the existing harness:
* ``kind=="datasheet"`` items run the numeric ctx-recall path (``harness ... --golden``);
* ``kind=="science"`` items run the source-recall / routing / abstention / citation /
  faithfulness path (``harness --science --golden``).

This module is import-light (only ``golden`` + ``llm`` + ``config`` + stdlib) so the unit
test runs offline; the costed drafting happens only when ``draft_item`` is called with a
real OpenRouter key (done by ``scripts/build_kb_golden.py``).
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterable

from src.assistant import config
from src.assistant.eval.kbdepth import golden
from src.assistant.llm import LLMUsage, call_llm
from src.assistant.tracing import parse_json_object
from src.utils.logger import setup_logger

_logger = setup_logger("kbdepth.golden_gen")

_ROOT = Path(__file__).resolve().parents[4]
PROD_CHUNKS = _ROOT / "models" / "knowledge_base_rag" / "kb_chunks.json"

# doc_type → eval bucket. ``None`` doc_type (WORD_MD key-value records, misc) → "other".
_BUCKET = {
    "datasheet": "datasheet",
    "literature": "literature",
    "scanned": "scanned",   # patents / scanned standards
    "page": "page",
    None: "other",
}
# default per-bucket target (+ 4 out-of-scope). Slightly overshoots so the run reaches
# ~50 even though the datasheet bucket typically yields fewer (few real film datasheets).
DEFAULT_PLAN = {"datasheet": 8, "literature": 18, "scanned": 8, "page": 4, "other": 12}
_MIN_CHUNK_CHARS = 200          # skip thin chunks — too little to ground a question
_MIN_DATASHEET_NUMBERS = 8      # a datasheet item needs enough verified numbers for the 0.8 gate

# Fixed out-of-scope prompts: NOT sampled from the corpus (so they are reliably
# off-domain). The correct behaviour is to abstain / refuse.
_OUT_OF_SCOPE = [
    "Яка погода завтра в Києві?",
    "Порадь хороший рецепт борщу",
    "Скільки буде 234 помножити на 17?",
    "Напиши вірш про осінь",
    # m2: extended off-domain set (still clearly outside друк/пакування).
    "Як полагодити пральну машину, що не зливає воду?",
    "Який смартфон краще купити до 15 тисяч гривень?",
    "Розкажи анекдот про програмістів",
    "Як приготувати тірамісу без яєць?",
    "Хто виграв чемпіонат світу з футболу 2022 року?",
    "Порадь вправи для спини при сидячій роботі",
    "Як оформити закордонний паспорт дитині?",
    "Переклади фразу 'добрий вечір' японською",
    "Скільки калорій у тарілці вареників з картоплею?",
    "Як підключити телевізор до домашнього Wi-Fi?",
    "Напиши привітання з днем народження для колеги",
]


# ── load + stratify (free, deterministic, no API) ────────────────────────────────
def load_chunks(path: str | Path = PROD_CHUNKS) -> list[dict[str, Any]]:
    """Load the production KB chunk records (raw dicts with the KBChunk fields)."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"KB chunks not found at {p} — build the KB index first.")
    data = json.loads(p.read_text(encoding="utf-8"))
    return list(data.values()) if isinstance(data, dict) else list(data)


def _bucket_of(chunk: dict) -> str:
    return _BUCKET.get(chunk.get("doc_type"), "other")


def stratified_sample(chunks: Iterable[dict], plan: dict[str, int] | None = None,
                      seed: int = 0, oversample: float = 3.0) -> dict[str, list[dict]]:
    """Group usable chunks by bucket and return up to ``ceil(plan[b]*oversample)`` per
    bucket, shuffled deterministically. Oversampling leaves headroom for drafts that
    fail verification. Datasheet bucket is returned as one chunk PER PARENT document
    (so a whole-datasheet 'full specs' item can be drafted), others are per-chunk."""
    plan = plan or DEFAULT_PLAN
    rng = random.Random(seed)
    buckets: dict[str, list[dict]] = {b: [] for b in plan}

    # datasheets: one representative chunk per parent (source), so we can aggregate the
    # whole document's numbers downstream.
    ds_by_parent: dict[str, dict] = {}
    for c in chunks:
        if len((c.get("text") or "")) < _MIN_CHUNK_CHARS:
            continue
        b = _bucket_of(c)
        if b not in buckets:
            continue
        if b == "datasheet":
            src_low = (c.get("source") or "").lower()
            if any(t in src_low for t in ("safety", "sds", "msds")):
                continue  # safety-data-sheets carry CAS/EC numbers, not film specs — skip
            key = c.get("source") or c.get("parent_id") or c.get("id")
            ds_by_parent.setdefault(key, c)  # first chunk of the parent represents it
        else:
            buckets[b].append(c)

    buckets["datasheet"] = list(ds_by_parent.values())
    out: dict[str, list[dict]] = {}
    for b, items in buckets.items():
        rng.shuffle(items)
        take = max(1, int(plan[b] * oversample))
        out[b] = items[:take]
    return out


def parent_text(chunks: list[dict], source: str, max_chars: int = 8000) -> str:
    """Concatenate (locator-ordered) all chunk texts of one datasheet parent, so the
    drafted 'full specs' item sees the whole document's numbers, not one 900-char slice."""
    kids = [c for c in chunks if (c.get("source") == source)]
    kids.sort(key=lambda c: str(c.get("locator", "")))
    blob = "\n".join(c.get("text", "") for c in kids)
    return blob[:max_chars]


# ── grounded drafting (costed) + mechanical verification ─────────────────────────
def _draft_user_payload(source_text: str, *, datasheet: bool) -> str:
    kind_hint = (
        "Це ДАТШИТ плівки — сформулюй запит про ПОВНІ технічні характеристики, а в "
        "\"numbers\" перелічи КОЖНУ числову характеристику з тексту (вихід, товщина, "
        "OTR, міцність MD/TD, COF, haze, gloss тощо)."
        if datasheet else
        "Сформулюй КОНКРЕТНЕ фахове питання, на яке цей фрагмент дає відповідь; у "
        "\"numbers\" наведи лише ті числа, що реально є у фрагменті (можна порожній список)."
    )
    return (
        f"{kind_hint}\n\nФРАГМЕНТ БАЗИ ЗНАНЬ:\n\"\"\"{source_text}\"\"\""
    )


def _paraphrase_queries(uk: str, en: str, *, role_key: str,
                        usage: LLMUsage | None) -> tuple[str, str]:
    """m2 anti-inflation pass: QG drafts copy the chunk's wording, which INFLATES
    retrieval metrics (lexical overlap — arxiv 2109.11256). One cheap call rewrites the
    queries in natural operator phrasing away from the source wording, preserving the
    meaning and all technical terms/codes. Best-effort: any failure keeps the originals."""
    try:
        resp = call_llm(
            agent_name="golden_paraphrase", role_key=role_key,
            messages=[
                {"role": "system", "content": config.prompt("golden_paraphrase")},
                {"role": "user", "content": json.dumps({"uk_query": uk, "en_query": en},
                                                       ensure_ascii=False)},
            ],
            usage=usage, temperature=0.7, max_tokens=300,
            response_format={"type": "json_object"},
        )
        d = parse_json_object(resp.choices[0].message.content or "{}")
        new_uk = (d.get("uk_query") or "").strip()
        new_en = (d.get("en_query") or "").strip()
        if new_uk and new_en:
            return new_uk, new_en
    except Exception as exc:  # noqa: BLE001 — paraphrase is an optional de-biaser
        _logger.info("paraphrase failed (%s) — keeping verbatim queries", str(exc)[:100])
    return uk, en


def draft_item(chunk: dict, *, bucket: str, all_chunks: list[dict] | None = None,
               usage: LLMUsage | None = None, drafter_role: str = "answer",
               paraphrase: bool = False) -> dict | None:
    """Draft ONE golden item from a chunk (or its parent, for datasheets) and verify
    its numbers against the source text. Returns a GoldenItem-shaped dict, or None when
    the draft is unusable (LLM/JSON error, or a datasheet with too few verified numbers).

    The returned dict's numbers are GUARANTEED present in the source text (no
    hallucinated golden); the source path is taken verbatim from the chunk.
    ``drafter_role`` selects the drafting model (m2: ``golden_drafter`` = a ~20× cheaper
    model than the Sonnet ``answer`` role); ``paraphrase`` adds the anti-inflation pass."""
    datasheet = bucket == "datasheet"
    source = chunk.get("source") or chunk.get("parent_id") or ""
    text = parent_text(all_chunks or [], source) if (datasheet and all_chunks) else (chunk.get("text") or "")
    if not text.strip():
        return None
    try:
        resp = call_llm(
            agent_name="golden_gen", role_key=drafter_role,
            messages=[
                {"role": "system", "content": config.prompt("golden_gen")},
                {"role": "user", "content": _draft_user_payload(text, datasheet=datasheet)},
            ],
            usage=usage, temperature=0.4, max_tokens=1500 if datasheet else 700,
            response_format={"type": "json_object"},
        )
        d = parse_json_object(resp.choices[0].message.content or "{}")
    except Exception as exc:  # noqa: BLE001 — one bad draft must not stop the run
        _logger.info("golden draft failed for %s: %s", source, str(exc)[:120])
        return None

    if d.get("skip"):  # drafter judged the chunk to be bibliography/TOC/catalog — drop it
        return None
    uk = (d.get("uk_query") or "").strip()
    en = (d.get("en_query") or "").strip()
    if not uk:
        return None
    uk_verbatim, en_verbatim = uk, en
    if paraphrase:
        uk, en = _paraphrase_queries(uk, en, role_key=drafter_role, usage=usage)

    # MECHANICAL number verification: keep only numbers that actually occur in the source.
    present = golden.numbers_in(text)
    raw_numbers = d.get("numbers") or []
    verified: list[dict] = []
    for n in raw_numbers:
        val = str(n.get("value") if isinstance(n, dict) else n).strip()
        if golden.normalize_num(val) in present:
            entry = {"value": val}
            if isinstance(n, dict):
                if n.get("attr"):
                    entry["attr"] = str(n["attr"])
                if n.get("unit"):
                    entry["unit"] = str(n["unit"])
            verified.append(entry)

    is_datasheet = datasheet and len(verified) >= _MIN_DATASHEET_NUMBERS
    item: dict[str, Any] = {
        "product": (chunk.get("product") or Path(source).stem or source)[:40],
        "kind": "datasheet" if is_datasheet else "science",
        "category": bucket,
        "lang": "uk",
        "source": source,
        "source_paths": [source],
        "queries": [q for q in (uk, en) if q],
        "key_claims": [str(c) for c in (d.get("key_claims") or [])][:6],
        "reference_answer": (d.get("reference_answer") or "")[:600],
        "route_expected": "instructions",
        "abstain_expected": False,
        "min_recall": 0.8,
    }
    if verified:
        item["numbers"] = verified
    if paraphrase and (uk, en) != (uk_verbatim, en_verbatim):
        # Keep the pre-paraphrase wording for golden-artifact forensics (raw-only field —
        # load_golden stores unknown keys in .raw).
        item["queries_verbatim"] = [q for q in (uk_verbatim, en_verbatim) if q]
    # A datasheet draft that lost too many numbers is demoted to a science item (it still
    # measures source-recall / faithfulness) rather than dropped.
    return item


def out_of_scope_items(n: int | None = None) -> list[dict]:
    """Fixed off-domain prompts whose correct behaviour is to abstain."""
    pool = _OUT_OF_SCOPE if n is None else _OUT_OF_SCOPE[:max(0, n)]
    return [{
        "product": f"OOS_{i}", "kind": "science", "category": "out_of_scope", "lang": "uk",
        "source": "", "source_paths": [], "queries": [q],
        "key_claims": [], "reference_answer": "", "route_expected": "out_of_scope",
        "abstain_expected": True, "min_recall": 0.0,
    } for i, q in enumerate(pool, start=1)]


# ── m2: near-domain abstention traps (mechanical, no LLM) ─────────────────────────
_TRAP_TEMPLATES = [
    "Які технічні характеристики плівки {code}?",
    "Дай повну специфікацію матеріалу {code}",
    "Яка товщина і вихід у плівки {code}?",
]


def near_domain_traps(chunks: list[dict], n: int = 15, seed: int = 0) -> list[dict]:
    """Printing-adjacent questions about product codes that do NOT exist in the corpus —
    the correct behaviour is an honest «нема в базі знань», NOT a confident answer about
    a similar real product. Codes are built by mutating REAL product codes and verified
    ABSENT mechanically (substring scan over every chunk text + product field), so the
    trap is guaranteed unanswerable. Deterministic (seeded); zero LLM cost.

    route_expected is left EMPTY — «instructions» is a legitimate route for these;
    the gated behaviour is abstention, not routing."""
    rng = random.Random(seed)
    real_products = sorted({str(c.get("product") or "").strip() for c in chunks
                            if c.get("product")})
    # one lowercase blob of everything a retrieval could surface — absence check target
    blob = " ".join((c.get("text") or "") for c in chunks).lower()
    blob += " " + " ".join(real_products).lower()

    traps: list[dict] = []
    seen: set[str] = set()
    attempts = 0
    while len(traps) < n and attempts < 500 and real_products:
        attempts += 1
        base = rng.choice(real_products)
        # mutate: append a digit/letter or swap a character — keeps the «family look»
        mut = rng.choice([
            base + rng.choice("XQZ7"),
            base[:-1] + rng.choice("XQZ9") if len(base) > 3 else base + "X",
            base.replace("-", "-" + rng.choice("QXZ"), 1) if "-" in base else base + "-Q",
        ])
        code = mut.upper()
        if code.lower() in blob or code in seen or code.lower() == base.lower():
            continue  # still present somewhere (or duplicate) — not a trap
        seen.add(code)
        q = rng.choice(_TRAP_TEMPLATES).format(code=code)
        traps.append({
            "product": f"TRAP_{len(traps)+1}_{code}", "kind": "science",
            "category": "near_domain_trap", "lang": "uk",
            "source": "", "source_paths": [], "queries": [q],
            "key_claims": [], "reference_answer": "", "route_expected": "",
            "abstain_expected": True, "min_recall": 0.0,
        })
    return traps


def generate(path: str | Path = PROD_CHUNKS, *, n: int = 50, plan: dict[str, int] | None = None,
             seed: int = 0, usage: LLMUsage | None = None, drafter_role: str = "answer",
             paraphrase: bool = False, n_offdomain: int | None = None,
             n_traps: int = 0) -> list[dict]:
    """Full pipeline → list of GoldenItem-shaped dicts (caller writes them with emit_jsonl).
    Target ``n`` total; abstention items (off-domain + near-domain traps) fill the tail."""
    plan = plan or DEFAULT_PLAN
    chunks = load_chunks(path)
    sampled = stratified_sample(chunks, plan, seed=seed)
    items: list[dict] = []
    for bucket, target in plan.items():
        made = 0
        for c in sampled.get(bucket, []):
            if made >= target:
                break
            it = draft_item(c, bucket=bucket, all_chunks=chunks if bucket == "datasheet" else None,
                            usage=usage, drafter_role=drafter_role, paraphrase=paraphrase)
            if it is not None:
                items.append(it)
                made += 1
        _logger.info("golden_gen bucket %s: %d/%d drafted", bucket, made, target)
    # Abstention tail: off-domain + (m2) near-domain traps — KB items are trimmed to fit.
    tail = out_of_scope_items(n_offdomain) + near_domain_traps(chunks, n=n_traps, seed=seed)
    items = items[: max(0, n - len(tail))]
    items.extend(tail)
    return items


# ── m2: operator-help golden (docs/operator_help → the SEPARATE operator RAG) ──────
def generate_operator(*, n: int = 50, seed: int = 0, usage: LLMUsage | None = None,
                      drafter_role: str = "answer", paraphrase: bool = False) -> list[dict]:
    """Draft a golden set for the OPERATOR-HELP path (previously unmeasured): chunks come
    from the operator RAG corpus (``src.assistant.rag.corpus.build_chunks``), the source is
    the corpus chunk's human-friendly ``source`` title (what ``rag.index`` cites), and items
    get ``kind="operator"`` so ``harness --operator`` picks exactly these."""
    from src.assistant.rag.corpus import build_chunks

    rng = random.Random(seed)
    op_chunks = [c for c in build_chunks() if len(c.text) >= _MIN_CHUNK_CHARS]
    rng.shuffle(op_chunks)
    items: list[dict] = []
    for c in op_chunks:
        if len(items) >= n:
            break
        it = draft_item({"source": c.source, "text": c.text}, bucket="other",
                        usage=usage, drafter_role=drafter_role, paraphrase=paraphrase)
        if it is None:
            continue
        it.update({
            "kind": "operator", "category": "operator_help",
            "route_expected": "instructions",
            "source": c.source, "source_paths": [c.source],
        })
        items.append(it)
    _logger.info("golden_gen operator: %d/%d drafted from %d chunks",
                 len(items), n, len(op_chunks))
    return items


# ── emit a reviewable JSONL ───────────────────────────────────────────────────────
_HEADER = [
    "// KB-grounded golden set (Block 1 / capstone). Auto-drafted from REAL kb_chunks.json,",
    "// numbers mechanically verified against the source text, sources pinned. SPOT-REVIEW the",
    "// `// REVIEW:` lines below and fix/delete any weak item before committing. `//` lines are",
    "// skipped by golden.load_golden. Run free Layer-A: harness --science --golden <this file>.",
]


def emit_jsonl(items: list[dict], path: str | Path) -> Path:
    """Write items one-per-line with a `// REVIEW:` echo after each, for operator spot-check."""
    p = Path(path)
    lines = list(_HEADER)
    for it in items:
        lines.append(json.dumps(it, ensure_ascii=False))
        nums = ", ".join(
            f"{n.get('value')}{(' ' + n['unit']) if n.get('unit') else ''}" for n in it.get("numbers", [])
        )
        review = (f"// REVIEW: [{it.get('category')}/{it.get('kind')}] src={it.get('source') or '—'} "
                  f"| q={it.get('queries', [''])[0]!r} | numbers=[{nums}] "
                  f"| claims={len(it.get('key_claims', []))} | abstain={it.get('abstain_expected')}")
        lines.append(review)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


__all__ = [
    "load_chunks", "stratified_sample", "parent_text", "draft_item",
    "out_of_scope_items", "near_domain_traps", "generate", "generate_operator",
    "emit_jsonl", "DEFAULT_PLAN", "PROD_CHUNKS",
]
