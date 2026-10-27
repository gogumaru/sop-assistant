"""Test guard deterministik dan pipeline jawaban dengan FakeEngine (tanpa Ollama, tanpa Qdrant)."""

from __future__ import annotations

import pytest

from sop_assistant.answer import answer_question, resolve_status
from sop_assistant.engines import FakeEngine
from sop_assistant.guards import (
    MASK,
    GuardResult,
    check_citations,
    detect_ambiguity,
    detect_escalation,
    extract_quantities,
    level_binding_errors,
    mask_pii,
    quantity_owners,
    should_abstain,
    uncovered_scales,
)
from sop_assistant.retrieve import Hit, Mode, RetrievalResult


def _payload(chunk_id: str, doc_id: str, doc_title: str, no: int, section_title: str, text: str) -> dict:
    return {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "doc_title": doc_title,
        "page": 1,
        "section_no": no,
        "section_title": section_title,
        "text": text,
    }


# Teks asli korpus (lihat data/chunks.jsonl).
PAYLOADS = {
    p["chunk_id"]: p
    for p in [
        _payload(
            "KB01-p1-s04", "KB_01", "Penanganan Tiket Layanan", 4, "Target Respons Awal",
            "P1: 15 menit; P2: 1 jam; P3: 4 jam kerja; P4: 1 hari kerja. Respons awal berarti petugas telah "
            "mengakui laporan dan menyampaikan langkah berikutnya, bukan bahwa masalah sudah selesai.",
        ),
        _payload(
            "KB01-p1-s07", "KB_01", "Penanganan Tiket Layanan", 7, "Penutupan",
            "Tiket dapat ditutup setelah penyelesaian dikonfirmasi pelanggan atau setelah dua hari kerja tanpa "
            "tanggapan sejak notifikasi selesai, kecuali tiket P1 yang memerlukan konfirmasi eksplisit.",
        ),
        _payload(
            "KB03-p1-s04", "KB_03", "Panduan Operasional Layanan", 4, "Pembaruan Berkala",
            "Tiket P1 diperbarui kepada pelanggan setidaknya setiap 30 menit selama gangguan aktif. Tiket P2 "
            "diperbarui setiap 4 jam kerja jika belum selesai. Untuk P3 dan P4, pembaruan diberikan saat ada "
            "perkembangan penting atau sesuai janji petugas.",
        ),
        _payload(
            "KB04-p1-s05", "KB_04", "Eskalasi dan Penanganan Insiden", 5, "Komunikasi Insiden",
            "Untuk SEV1, pembaruan status internal dibuat setiap 20 menit. Komunikasi pelanggan menggunakan "
            "ringkasan dampak, langkah mitigasi, dan waktu pembaruan berikutnya tanpa mengungkap detail "
            "keamanan sensitif.",
        ),
    ]
}  # fmt: skip
CONTEXT = {chunk_id: p["text"] for chunk_id, p in PAYLOADS.items()}


class FakeRetriever:
    """Mengembalikan `ranking` untuk query biasa; untuk query berfilter dokumen, semua payload dokumen itu."""

    def __init__(self, ranking: list[str], dense_top1: float = 0.7):
        self.ranking = ranking
        self.dense_top1 = dense_top1
        self.calls: list[dict] = []

    def __call__(self, query, k=None, mode=Mode.hybrid, doc_ids=None) -> RetrievalResult:
        self.calls.append({"query": query, "doc_ids": doc_ids})
        pool = self.ranking if doc_ids is None else [c for c, p in PAYLOADS.items() if p["doc_id"] in doc_ids]
        hits = [Hit(chunk_id=c, score=1.0, payload=PAYLOADS[c]) for c in pool[: k or 5]]
        return RetrievalResult(query=query, mode=mode, hits=hits, dense_top1=self.dense_top1)


def llm(status="answered", claims=(), answer="", reason="none", missing="", escalate_to="") -> dict:
    return {
        "claims": list(claims),
        "answer": answer,
        "missing": missing,
        "escalate_to": escalate_to,
        "status": status,
        "review_reason": reason,
    }


def claim(text: str, *chunk_ids: str) -> dict:
    return {"text": text, "chunk_ids": list(chunk_ids)}


# --- PII mask ---


@pytest.mark.parametrize(
    ("text", "expected_count"),
    [
        ("Kartu pelanggan 4111 1111 1111 1111 terblokir", 1),
        ("Pelanggan kasih kode OTP 482913, boleh dipakai?", 1),
        ("Nomor tiket 482913 belum diproses", 0),  # 6 digit tanpa konteks OTP/PIN bukan rahasia
    ],
)
def test_mask_pii(text, expected_count):
    masked, count = mask_pii(text)
    assert count == expected_count
    assert (MASK in masked) == bool(expected_count)


# --- Abstain ---


def test_abstain_router_inactive_until_calibrated():
    assert should_abstain(0.10, None) is False
    assert should_abstain(0.40, 0.45) is True
    assert should_abstain(0.50, 0.45) is False


# --- Glossary ---


@pytest.mark.parametrize(
    ("question", "term"),
    [
        ("Gangguan kritis harus di-update tiap berapa menit?", "gangguan kritis"),
        ("Gangguan prioritas tinggi harus dieskalasi berapa lama?", "prioritas tinggi"),
    ],
)
def test_ambiguous_terms_detected(question, term):
    ambiguity = detect_ambiguity(question)
    assert ambiguity is not None and ambiguity.term == term
    assert {s.id for s in ambiguity.scales} == {"P", "SEV"}


@pytest.mark.parametrize(
    "question",
    [
        "Tiket yang sedang diproses boleh dialihkan?",  # "sedang" = currently
        "Volume tiket tinggi minggu ini, apa yang dilaporkan?",  # "tinggi" kata umum
        "Gangguan kritis P1 harus di-update tiap berapa menit?",  # token eksplisit
        "brp lama sev 2 hrs dieskalasi",  # token eksplisit dengan spasi
    ],
)
def test_no_false_ambiguity(question):
    assert detect_ambiguity(question) is None


# --- Ekstraksi angka + satuan ---


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("P3: 4 jam kerja", {"4 jam kerja"}),
        ("P2: 1 jam", {"1 jam"}),
        ("setelah dua hari kerja", {"2 hari kerja"}),
        ("paling lambat lima hari kerja", {"5 hari kerja"}),
        ("pukul 09.00-17.00 WIB", {"pukul 9.00", "pukul 17.00"}),
        ("tiket aktif tanpa angka", set()),
    ],
)
def test_extract_quantities(text, expected):
    assert extract_quantities(text) == expected


# --- Citation guard ---


def test_guard_accepts_supported_claim():
    output = llm(claims=[claim("Respons awal P3 adalah 4 jam kerja.", "KB01-p1-s04")], answer="P3: 4 jam kerja.")
    assert check_citations(output, CONTEXT).ok


def test_guard_accepts_word_number_in_source():
    output = llm(claims=[claim("Tiket ditutup setelah 2 hari kerja tanpa tanggapan.", "KB01-p1-s07")])
    assert check_citations(output, CONTEXT).ok


def test_guard_rejects_dropped_unit_qualifier():
    # Sumber: "4 jam kerja". "4 jam" mengubah makna (jam kalender vs jam kerja).
    output = llm(claims=[claim("Respons awal P3 adalah 4 jam.", "KB01-p1-s04")])
    result = check_citations(output, CONTEXT)
    assert not result.ok and "4 jam" in result.errors[0]


def test_guard_rejects_unknown_chunk():
    output = llm(claims=[claim("Tiket P1 diperbarui setiap 30 menit.", "KB99-p1-s01")])
    assert not check_citations(output, CONTEXT).ok


def test_guard_rejects_number_in_answer_not_in_cited_chunks():
    output = llm(claims=[claim("Tiket P1 diperbarui ke pelanggan.", "KB03-p1-s04")], answer="P1 diperbarui tiap 45 menit.")
    assert not check_citations(output, CONTEXT).ok


def test_guard_rejects_answered_without_claims():
    assert not check_citations(llm(answer="Tiket boleh ditutup kapan saja."), CONTEXT).ok


def test_guard_allows_not_in_sop_without_claims():
    assert check_citations(llm(status="not_in_sop", answer="SOP tidak mengatur hal ini."), CONTEXT).ok


# --- Pasangan angka dan level ---


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        # Teks OCR KB_01 §4: titik koma terbaca titik dua, pasangan tetap benar.
        ("P1: 15 menit: P2: 1 jam: P3: 4 jam kerja", {"15 menit": [{"P1"}], "1 jam": [{"P2"}], "4 jam kerja": [{"P3"}]}),
        ("Untuk SEV1 dan SEV2, tim menyusun tinjauan paling lambat lima hari kerja.", {"5 hari kerja": [{"SEV1", "SEV2"}]}),
        ("setelah dua hari kerja tanpa tanggapan, kecuali tiket P1", {"2 hari kerja": [set()]}),  # aturan umum
    ],
)
def test_quantity_owners(source, expected):
    assert quantity_owners(source) == expected


def test_binding_catches_number_attached_to_wrong_level():
    # Kasus nyata baseline full-context q10: 30 menit milik P1, bukan SEV1.
    errors = level_binding_errors("Gangguan kritis (SEV1) harus diperbarui setiap 30 menit.", [CONTEXT["KB03-p1-s04"]])
    assert errors and "P1" in errors[0]


def test_binding_allows_correct_pairs_and_negated_premise():
    sources = [CONTEXT["KB03-p1-s04"], CONTEXT["KB04-p1-s05"]]
    assert level_binding_errors("P1: 30 menit ke pelanggan. SEV1: 20 menit untuk status internal.", sources) == []
    # Koreksi premis q13 menyebut P3 dan "4 jam kerja" (milik P2) dalam kalimat bernegasi.
    assert level_binding_errors("Tidak, tiket P3 tidak perlu diperbarui setiap 4 jam kerja.", sources) == []


def test_guard_rejects_wrong_level_binding_in_claim():
    output = llm(claims=[claim("SEV1 diperbarui setiap 30 menit.", "KB03-p1-s04")], answer="SEV1: setiap 30 menit.")
    result = check_citations(output, CONTEXT)
    assert not result.ok and result.valid_claims == []


# --- Eskalasi ---


@pytest.mark.parametrize(
    ("question", "topic"),
    [
        ("Berapa nominal kompensasi yang boleh dijanjikan petugas?", "kompensasi"),
        ("Bagaimana prosedur refund lewat e-wallet?", "kompensasi"),
        ("Siapa yang menilai dampak kebocoran data?", "kebocoran_data"),
        ("Status tiket apa saja yang dipakai?", None),
    ],
)
def test_detect_escalation(question, topic):
    rule = detect_escalation(question)
    assert (rule.id if rule else None) == topic


def test_pipeline_fills_escalation_when_llm_leaves_it_empty():
    engine = FakeEngine([llm(status="needs_review", reason="partial", answer="Nominal tidak diatur.")])
    result = answer_question(
        "Berapa nominal kompensasi yang boleh dijanjikan petugas?",
        engine,
        retriever=FakeRetriever(["KB01-p1-s04"]),
        threshold=None,
    )
    assert result.escalation_by_rule and "KB_02 §7" in result.escalate_to
    assert (result.status, result.review_reason) == ("needs_review", "partial")  # aturan tidak mengubah status


def test_pipeline_keeps_escalation_written_by_llm():
    engine = FakeEngine([llm(status="not_in_sop", answer="Tidak diatur.", escalate_to="Pejabat berwenang")])
    result = answer_question(
        "Bagaimana prosedur refund lewat e-wallet?", engine, retriever=FakeRetriever(["KB01-p1-s04"]), threshold=None
    )
    assert (result.escalate_to, result.escalation_by_rule) == ("Pejabat berwenang", False)


def test_pipeline_reports_stages():
    stages: list[str] = []
    bad = llm(claims=[claim("Respons awal P3 adalah 4 jam.", "KB01-p1-s04")], answer="4 jam.")
    good = llm(claims=[claim("Respons awal P3 adalah 4 jam kerja.", "KB01-p1-s04")], answer="4 jam kerja.")
    answer_question(
        "Respons awal P3?",
        FakeEngine([bad, good]),
        retriever=FakeRetriever(["KB01-p1-s04"]),
        threshold=None,
        on_stage=stages.append,
    )
    assert stages == ["retrieval", "llm", "repair"]


# --- Penentuan status ---

OK = GuardResult(ok=True, errors=[], valid_claims=[])
FAILED = GuardResult(ok=False, errors=["x"], valid_claims=[])
AMBIGUOUS = detect_ambiguity("gangguan kritis")


@pytest.mark.parametrize(
    ("output", "guard", "ambiguity", "expected"),
    [
        (llm(status="answered"), FAILED, None, ("needs_review", "guard_failed")),
        (llm(status="answered"), OK, AMBIGUOUS, ("needs_review", "ambiguous")),
        (llm(status="not_in_sop"), OK, AMBIGUOUS, ("not_in_sop", None)),
        (llm(status="needs_review", reason="partial"), OK, None, ("needs_review", "partial")),
        (llm(status="answered"), OK, None, ("answered", None)),
    ],
)
def test_status_precedence(output, guard, ambiguity, expected):
    assert resolve_status(output, guard, ambiguity) == expected


# --- Pipeline dengan FakeEngine ---


def test_pipeline_abstains_without_calling_llm():
    engine = FakeEngine([])
    result = answer_question(
        "Berapa tunjangan petugas piket?", engine, retriever=FakeRetriever(["KB01-p1-s04"], 0.30), threshold=0.45
    )
    assert (result.status, result.abstained) == ("not_in_sop", True)
    assert engine.calls == []


def test_pipeline_answered_sources_rendered_from_payload():
    engine = FakeEngine([llm(claims=[claim("Target respons awal P2 adalah 1 jam.", "KB01-p1-s04")], answer="1 jam.")])
    retriever = FakeRetriever(["KB01-p1-s04", "KB01-p1-s07"])
    result = answer_question("Berapa target respons awal untuk tiket P2?", engine, retriever=retriever, threshold=None)
    assert result.status == "answered"
    assert [s.label() for s in result.sources] == ["KB_01 Penanganan Tiket Layanan, hal. 1, §4 Target Respons Awal"]
    assert len(engine.calls) == 1


def test_pipeline_repairs_once_then_answers():
    bad = llm(claims=[claim("Respons awal P3 adalah 4 jam.", "KB01-p1-s04")], answer="4 jam.")
    good = llm(claims=[claim("Respons awal P3 adalah 4 jam kerja.", "KB01-p1-s04")], answer="4 jam kerja.")
    engine = FakeEngine([bad, good])
    result = answer_question("Respons awal P3?", engine, retriever=FakeRetriever(["KB01-p1-s04"]), threshold=None)
    assert (result.status, result.repaired) == ("answered", True)
    assert "DITOLAK PEMERIKSA SUMBER" in engine.calls[1]["user"]


def test_pipeline_guard_fails_twice_shows_only_verified_claims():
    response = llm(
        claims=[
            claim("Tiket P1 diperbarui ke pelanggan setiap 30 menit.", "KB03-p1-s04"),
            claim("Tiket P3 diperbarui setiap 45 menit.", "KB03-p1-s04"),
        ],
        answer="P1 tiap 30 menit, P3 tiap 45 menit.",
    )
    engine = FakeEngine([response, response])
    result = answer_question("Update tiket?", engine, retriever=FakeRetriever(["KB03-p1-s04"]), threshold=None)
    assert (result.status, result.review_reason) == ("needs_review", "guard_failed")
    assert "45 menit" not in result.answer and "30 menit" in result.answer
    assert len(result.claims) == 1


def test_pipeline_ambiguity_adds_both_scales_and_forces_review():
    engine = FakeEngine(
        [
            llm(
                claims=[
                    claim("Tiket P1 diperbarui ke pelanggan setiap 30 menit.", "KB03-p1-s04"),
                    claim("SEV1: status internal diperbarui setiap 20 menit.", "KB04-p1-s05"),
                ],
                answer="P1: 30 menit ke pelanggan. SEV1: 20 menit untuk status internal.",
            )
        ]
    )
    retriever = FakeRetriever(["KB03-p1-s04"])  # retrieval utama hanya menemukan skala P
    result = answer_question(
        "Gangguan kritis harus di-update tiap berapa menit?", engine, retriever=retriever, threshold=None
    )
    assert (result.status, result.review_reason) == ("needs_review", "ambiguous")
    assert "KB04-p1-s05" in result.retrieved_ids
    assert "CATATAN AMBIGUITAS" in engine.calls[0]["user"]


SEV_ONLY = llm(
    claims=[claim("SEV1: status internal diperbarui setiap 20 menit.", "KB04-p1-s05")],
    answer="SEV1 diperbarui setiap 20 menit untuk status internal.",
)
BOTH_SCALES = llm(
    claims=[
        claim("Tiket P1 diperbarui ke pelanggan setiap 30 menit.", "KB03-p1-s04"),
        claim("SEV1: status internal diperbarui setiap 20 menit.", "KB04-p1-s05"),
    ],
    answer="P1: 30 menit ke pelanggan. SEV1: 20 menit untuk status internal.",
)
DOC_OF = {chunk_id: p["doc_id"] for chunk_id, p in PAYLOADS.items()}


def test_uncovered_scales_detects_missing_p_scale():
    assert [s.id for s in uncovered_scales(SEV_ONLY["claims"], DOC_OF, AMBIGUOUS)] == ["P"]


def test_citing_p_document_without_p1_does_not_count_as_coverage():
    # Kasus nyata: model menyitasi section KB_03 yang memuat "gangguan kritis" tanpa menjawab P1.
    claims = SEV_ONLY["claims"] + [claim("Gangguan kritis diteruskan melalui jalur piket.", "KB03-p1-s04")]
    assert [s.id for s in uncovered_scales(claims, DOC_OF, AMBIGUOUS)] == ["P"]


def test_level_token_matches_with_space():
    claims = [claim("Untuk SEV 1, status internal tiap 20 menit.", "KB04-p1-s05")]
    assert [s.id for s in uncovered_scales(claims, DOC_OF, AMBIGUOUS)] == ["P"]  # SEV tercakup


def test_pipeline_ambiguity_answering_one_scale_triggers_repair():
    engine = FakeEngine([SEV_ONLY, BOTH_SCALES])
    result = answer_question(
        "Gangguan kritis harus di-update tiap berapa menit?",
        engine,
        retriever=FakeRetriever(["KB03-p1-s04"]),
        threshold=None,
    )
    assert result.repaired and result.scale_gaps == []
    assert "P1 (prioritas tiket)" in result.repair_reasons[0]
    assert (result.status, result.review_reason) == ("needs_review", "ambiguous")


def test_pipeline_ambiguity_gap_after_repair_is_stated_not_hidden():
    engine = FakeEngine([SEV_ONLY, SEV_ONLY])
    result = answer_question(
        "Gangguan kritis harus di-update tiap berapa menit?",
        engine,
        retriever=FakeRetriever(["KB03-p1-s04"]),
        threshold=None,
    )
    assert result.scale_gaps == ["P"]
    assert "Belum terjawab untuk prioritas tiket" in result.answer
    assert (result.status, result.review_reason) == ("needs_review", "ambiguous")


def test_pipeline_masks_pii_before_retrieval_and_llm():
    retriever = FakeRetriever(["KB01-p1-s04"])
    engine = FakeEngine([llm(status="not_in_sop", answer="SOP tidak mengatur hal ini.")])
    result = answer_question("Pelanggan kasih kode OTP 482913, boleh dipakai?", engine, retriever=retriever, threshold=None)
    assert "482913" not in retriever.calls[0]["query"]
    assert "482913" not in engine.calls[0]["user"]
    assert result.pii_masked == 1
