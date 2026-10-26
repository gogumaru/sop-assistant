"""Test ekstraksi: urutan kolom, section 1..8, judul dokumen, dan kualitas OCR."""

from __future__ import annotations

import random
import re
from pathlib import Path

import pymupdf
import pytest

from sop_assistant.config import settings
from sop_assistant.extract import (
    HEADING_RE,
    Chunk,
    Line,
    extract_corpus,
    fix_ocr_level_tokens,
    order_lines,
    page_lines,
)
from sop_assistant.metrics import char_error_rate

TRANSCRIPTS = Path(__file__).parent / "fixtures" / "transcripts"
PAGE_WIDTH = 595.3
EXPECTED_TITLES = {
    "KB_01": "Penanganan Tiket Layanan",
    "KB_02": "Pengelolaan Keluhan Pelanggan",
    "KB_03": "Panduan Operasional Layanan",
    "KB_04": "Eskalasi dan Penanganan Insiden",  # beda dari nama file: judul diambil dari halaman
}
OCR_CER_MAX = 0.02


@pytest.fixture(scope="module")
def chunks() -> list[Chunk]:
    return extract_corpus()


def section(chunks: list[Chunk], doc_id: str, no: int) -> Chunk:
    return next(c for c in chunks if c.doc_id == doc_id and c.section_no == no)


def _line(text: str, x0: float, y0: float, x1: float) -> Line:
    return Line(text, x0, y0, x1, y0 + 9, page=1)


# --- Urutan baca (tanpa PDF, koordinat meniru korpus) ---


def test_two_columns_read_left_then_right():
    lines = [
        _line("Judul", 48, 46, 277),
        _line("1. Tujuan", 48, 92, 140),
        _line("isi kiri", 48, 114, 280),
        _line("2. Kedua", 48, 300, 150),
        _line("7. Penutupan", 306, 92, 420),
        _line("isi kanan", 306, 114, 540),
    ]
    shuffled = lines[:]
    random.Random(0).shuffle(shuffled)
    assert [line.text for line in order_lines(shuffled, PAGE_WIDTH)] == [line.text for line in lines]


def test_single_column_narrow_headings_stay_with_their_body():
    # KB_02/KB_04: isi selebar halaman, heading sempit. Aturan "full-width di atas" akan gagal di sini.
    lines = [
        _line("Judul", 48, 46, 321),
        _line("1. Ruang Lingkup", 48, 92, 152),
        _line("isi satu", 48, 114, 537),
        _line("2. Verifikasi", 48, 160, 119),
        _line("isi dua", 48, 182, 518),
    ]
    shuffled = lines[:]
    random.Random(1).shuffle(shuffled)
    assert [line.text for line in order_lines(shuffled, PAGE_WIDTH)] == [line.text for line in lines]


def test_kb03_column_order_does_not_depend_on_stream_order():
    # Urutan stream KB_03 kebetulan sudah benar; acak dulu untuk membuktikan logika koordinat.
    path = next(settings.pdf_dir.glob("KB_03_*.pdf"))
    with pymupdf.open(path) as doc:
        page = doc[0]
        lines, _ = page_lines(page)
        width = page.rect.width
    random.Random(42).shuffle(lines)
    headings = [int(m.group(1)) for line in order_lines(lines, width) if (m := HEADING_RE.match(line.text))]
    assert headings == list(range(1, 9))


# --- Hasil ekstraksi korpus ---


def test_corpus_has_32_chunks_with_unique_ids(chunks):
    assert len(chunks) == 32
    ids = [c.chunk_id for c in chunks]
    assert len(set(ids)) == len(ids)
    assert all(re.fullmatch(r"KB\d{2}-p\d+-s\d{2}", chunk_id) for chunk_id in ids)


@pytest.mark.parametrize("doc_id", sorted(EXPECTED_TITLES))
def test_sections_are_exactly_1_to_8(chunks, doc_id):
    assert [c.section_no for c in chunks if c.doc_id == doc_id] == list(range(1, 9))


@pytest.mark.parametrize(("doc_id", "title"), sorted(EXPECTED_TITLES.items()))
def test_doc_title_comes_from_page(chunks, doc_id, title):
    # KB_03: nomor halaman "1" tidak boleh menempel ke judul.
    assert section(chunks, doc_id, 1).doc_title == title


def test_no_empty_chunk(chunks):
    assert all(c.text.strip() and c.section_title.strip() for c in chunks)


def test_kb01_two_columns_not_mixed(chunks):
    tujuan, penutupan = section(chunks, "KB_01", 1), section(chunks, "KB_01", 7)
    assert "dua hari kerja" in penutupan.text
    for leaked in ("Penutupan", "Catatan Khusus", "Tiket dapat ditutup"):
        assert leaked not in tujuan.text


def test_ocr_used_only_where_no_text_layer(chunks):
    ocr_docs = {c.doc_id for c in chunks if c.ocr_used}
    assert ocr_docs == {"KB_01", "KB_04"}


# --- Kualitas OCR terhadap transkripsi manual ---


@pytest.mark.parametrize("doc_id", ["KB_01", "KB_04"])
def test_ocr_character_error_rate(chunks, doc_id):
    doc_chunks = [c for c in chunks if c.doc_id == doc_id]
    hypothesis = "\n".join(
        [doc_chunks[0].doc_title] + [f"{c.section_no}. {c.section_title}\n{c.text}" for c in doc_chunks]
    )
    reference = (TRANSCRIPTS / f"{doc_id}.txt").read_text(encoding="utf-8")
    cer = char_error_rate(hypothesis, reference)
    assert cer <= OCR_CER_MAX, f"CER {doc_id} = {cer:.3%}"


# Token yang salah baca OCR-nya langsung mengubah makna jawaban.
CRITICAL_TOKENS = [
    ("KB_01", 4, "15 menit"),
    ("KB_01", 4, "4 jam kerja"),
    ("KB_01", 4, "1 hari kerja"),
    ("KB_01", 6, "Menunggu Pihak Ketiga"),
    ("KB_01", 7, "dua hari kerja"),
    ("KB_01", 7, "kecuali tiket P1"),
    ("KB_04", 2, "SEV1"),
    ("KB_04", 3, "SEV1 harus segera"),
    ("KB_04", 3, "30 menit"),
    ("KB_04", 5, "Untuk SEV1"),
    ("KB_04", 5, "20 menit"),
    ("KB_04", 7, "tidak otomatis"),
    ("KB_04", 8, "SEV1 dan SEV2"),
    ("KB_04", 8, "lima hari kerja"),
]


@pytest.mark.parametrize(
    ("ocr_text", "expected"),
    [
        ("SEVI: gangguan menyeluruh", "SEV1: gangguan menyeluruh"),
        ("kecuali tiket Pl yang", "kecuali tiket P1 yang"),
        ("PIC teknis", "PIC teknis"),  # kata lain yang diawali P/SEV tidak boleh tersentuh
        ("SEV2 dan SEV3", "SEV2 dan SEV3"),
    ],
)
def test_fix_ocr_level_tokens(ocr_text, expected):
    assert fix_ocr_level_tokens(ocr_text) == expected


@pytest.mark.parametrize(("doc_id", "no", "token"), CRITICAL_TOKENS)
def test_critical_tokens_survive_ocr(chunks, doc_id, no, token):
    assert token in section(chunks, doc_id, no).text
