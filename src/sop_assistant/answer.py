"""Pipeline jawaban: mask PII -> retrieval -> abstain -> glossary -> LLM -> citation guard -> status -> sitasi.

LLM hanya memberi teks dan chunk_ids. Nama dokumen, halaman, judul section, dan cuplikan
sumber diambil kode dari payload Qdrant.

Urutan penentuan status (deterministik):
1. Guard gagal setelah satu kali perbaikan -> needs_review / guard_failed.
2. LLM menyatakan not_in_sop -> not_in_sop (istilah ambigu tidak membuat pertanyaan di luar SOP jadi relevan).
3. Glossary mendeteksi ambiguitas -> needs_review / ambiguous, apa pun status dari LLM.
4. Selain itu, status dari LLM.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from sop_assistant.config import settings
from sop_assistant.engines import LLMEngine
from sop_assistant.guards import (
    Ambiguity,
    GuardResult,
    Scale,
    check_citations,
    detect_ambiguity,
    mask_pii,
    should_abstain,
    uncovered_scales,
)
from sop_assistant.prompts import SYSTEM_PROMPT, answer_schema, build_repair_prompt, build_user_prompt
from sop_assistant.retrieve import Hit, Mode, RetrievalResult, retrieve

Retriever = Callable[..., RetrievalResult]

SNIPPET_CHARS = 200
AMBIGUITY_EXTRA_K = 2  # chunk tambahan per skala saat istilah ambigu terdeteksi
ABSTAIN_ANSWER = (
    "Tidak ditemukan dalam SOP yang tersedia. Jangan mengambil keputusan berdasarkan asumsi; "
    "konfirmasikan ke koordinator layanan."
)
GUARD_FALLBACK = (
    "Jawaban tidak lolos pemeriksaan sumber. Periksa langsung sumber di bawah atau tanyakan ke koordinator layanan."
)
SCALE_GAP_NOTE = "Belum terjawab untuk {name} ({docs}); periksa SOP tersebut sebelum bertindak."


@dataclass(frozen=True)
class Source:
    chunk_id: str
    doc_id: str
    doc_title: str
    page: int
    section_no: int
    section_title: str
    snippet: str

    def label(self) -> str:
        return f"{self.doc_id} {self.doc_title}, hal. {self.page}, §{self.section_no} {self.section_title}"


@dataclass
class AnswerResult:
    question: str
    status: str
    review_reason: str | None
    answer: str
    claims: list[dict] = field(default_factory=list)
    sources: list[Source] = field(default_factory=list)
    missing: str | None = None
    escalate_to: str | None = None
    ambiguity: Ambiguity | None = None
    dense_top1: float = 0.0
    abstained: bool = False
    retrieved_ids: list[str] = field(default_factory=list)
    guard_errors: list[str] = field(default_factory=list)  # error akhir (setelah perbaikan)
    repaired: bool = False
    repair_reasons: list[str] = field(default_factory=list)  # kenapa percobaan pertama ditolak
    scale_gaps: list[str] = field(default_factory=list)  # skala ambigu yang tetap tidak terjawab
    pii_masked: int = 0
    timings_ms: dict[str, float] = field(default_factory=dict)


def _none_if_empty(value: str | None) -> str | None:
    return None if value is None or value.strip().lower() in {"", "none", "null", "-"} else value.strip()


def snippet(text: str, limit: int = SNIPPET_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + " ..."


def resolve_status(output: dict, guard: GuardResult, ambiguity: Ambiguity | None) -> tuple[str, str | None]:
    if not guard.ok:
        return "needs_review", "guard_failed"
    status = output.get("status", "needs_review")
    if status == "not_in_sop":
        return "not_in_sop", None
    if ambiguity:
        return "needs_review", "ambiguous"
    if status == "needs_review":
        return status, _none_if_empty(output.get("review_reason"))
    return "answered", None


def build_sources(claims: list[dict], context: dict[str, dict]) -> list[Source]:
    ordered = dict.fromkeys(c for claim in claims for c in claim.get("chunk_ids") or [] if c in context)
    return [
        Source(
            chunk_id=chunk_id,
            doc_id=context[chunk_id]["doc_id"],
            doc_title=context[chunk_id]["doc_title"],
            page=context[chunk_id]["page"],
            section_no=context[chunk_id]["section_no"],
            section_title=context[chunk_id]["section_title"],
            snippet=snippet(context[chunk_id]["text"]),
        )
        for chunk_id in ordered
    ]


def _with_both_scales(question: str, hits: list[Hit], ambiguity: Ambiguity, retriever: Retriever, mode: Mode) -> list[Hit]:
    """Pastikan konteks memuat chunk dari setiap skala kandidat (P dan SEV)."""
    merged = {h.chunk_id: h for h in hits}
    for scale in ambiguity.scales:
        for hit in retriever(question, k=AMBIGUITY_EXTRA_K, mode=mode, doc_ids=scale.doc_ids).hits:
            merged.setdefault(hit.chunk_id, hit)
    return list(merged.values())


def _scale_gaps(output: dict, doc_of: dict[str, str], ambiguity: Ambiguity | None) -> list[Scale]:
    if not ambiguity or output.get("status") == "not_in_sop":
        return []
    return uncovered_scales(output.get("claims") or [], doc_of, ambiguity)


def _gap_error(ambiguity: Ambiguity, scale: Scale) -> str:
    tokens = "/".join(c.token for c in ambiguity.candidates if c.scale == scale.id)
    return (
        f'Istilah "{ambiguity.term}" ambigu, tetapi tidak ada klaim yang menyebut {tokens} ({scale.name}) '
        f"dengan sitasi dari {', '.join(scale.doc_ids)}. Tambahkan klaim itu dari KONTEKS, "
        "atau nyatakan bahwa SOP tidak mengaturnya untuk skala ini."
    )


def _generate(engine: LLMEngine, user: str, schema: dict, context_text: dict[str, str]) -> tuple[dict, GuardResult]:
    try:
        output = engine.generate_json(SYSTEM_PROMPT, user, schema)
    except ValueError as exc:  # termasuk JSONDecodeError
        return {}, GuardResult(ok=False, errors=[f"Output LLM bukan JSON valid: {exc}"], valid_claims=[])
    return output, check_citations(output, context_text)


def answer_question(
    question: str,
    engine: LLMEngine,
    *,
    retriever: Retriever = retrieve,
    mode: Mode = Mode.hybrid,
    k: int | None = None,
    threshold: float | None = settings.abstain_threshold,
) -> AnswerResult:
    start = time.perf_counter()
    timings: dict[str, float] = {}
    masked, n_masked = mask_pii(question)

    t = time.perf_counter()
    retrieval = retriever(masked, k=k, mode=mode)
    timings["retrieval"] = (time.perf_counter() - t) * 1000

    if should_abstain(retrieval.dense_top1, threshold):
        timings["total"] = (time.perf_counter() - start) * 1000
        return AnswerResult(
            question=masked,
            status="not_in_sop",
            review_reason=None,
            answer=ABSTAIN_ANSWER,
            dense_top1=retrieval.dense_top1,
            abstained=True,
            retrieved_ids=[h.chunk_id for h in retrieval.hits],
            pii_masked=n_masked,
            timings_ms=timings,
        )

    ambiguity = detect_ambiguity(masked)
    hits = retrieval.hits
    if ambiguity:
        t = time.perf_counter()
        hits = _with_both_scales(masked, hits, ambiguity, retriever, mode)
        timings["retrieval"] += (time.perf_counter() - t) * 1000

    context = {h.chunk_id: h.payload for h in hits}
    context_text = {chunk_id: p["text"] for chunk_id, p in context.items()}
    doc_of = {chunk_id: p["doc_id"] for chunk_id, p in context.items()}
    schema = answer_schema(list(context))
    user = build_user_prompt(masked, hits, ambiguity)

    t = time.perf_counter()
    output, guard = _generate(engine, user, schema, context_text)
    gaps = _scale_gaps(output, doc_of, ambiguity)
    repair_reasons: list[str] = []
    if not guard.ok or gaps:
        repair_reasons = guard.errors + [_gap_error(ambiguity, scale) for scale in gaps]
        output, guard = _generate(engine, build_repair_prompt(user, output, repair_reasons), schema, context_text)
        gaps = _scale_gaps(output, doc_of, ambiguity)
    timings["llm"] = (time.perf_counter() - t) * 1000

    status, reason = resolve_status(output, guard, ambiguity)
    if guard.ok:
        claims = output.get("claims") or []
        answer_text = output.get("answer", "").strip()
        # Klaim tetap valid, tetapi satu skala tidak terjawab: katakan terus terang, jangan diam.
        notes = [SCALE_GAP_NOTE.format(name=s.name, docs=", ".join(s.doc_ids)) for s in gaps]
        answer_text = " ".join([answer_text, *notes]).strip()
    else:  # tampilkan hanya klaim yang lolos verifikasi
        claims = guard.valid_claims
        answer_text = " ".join(c["text"] for c in claims) or GUARD_FALLBACK

    timings["total"] = (time.perf_counter() - start) * 1000
    return AnswerResult(
        question=masked,
        status=status,
        review_reason=reason,
        answer=answer_text,
        claims=claims,
        sources=build_sources(claims, context),
        missing=_none_if_empty(output.get("missing")),
        escalate_to=_none_if_empty(output.get("escalate_to")),
        ambiguity=ambiguity,
        dense_top1=retrieval.dense_top1,
        retrieved_ids=list(context),
        guard_errors=guard.errors,
        repaired=bool(repair_reasons),
        repair_reasons=repair_reasons,
        scale_gaps=[s.id for s in gaps],
        pii_masked=n_masked,
        timings_ms=timings,
    )
