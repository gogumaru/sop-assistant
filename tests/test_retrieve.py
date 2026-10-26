"""Test integrasi index + retrieval. Butuh Qdrant aktif dan model bge-m3; di-skip jika Qdrant mati.

Ini sanity check, bukan evaluasi. Recall/MRR terhadap golden set ada di eval/run_eval.py (M4).
"""

from __future__ import annotations

import pytest

from sop_assistant.extract import Chunk, extract_corpus
from sop_assistant.index import get_client, index_chunks
from sop_assistant.retrieve import Mode, retrieve

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def indexed() -> list[Chunk]:
    try:
        get_client().get_collections()
    except Exception as exc:  # koneksi ditolak, timeout, dll.
        pytest.skip(f"Qdrant tidak bisa dihubungi: {exc}")
    chunks = extract_corpus()
    index_chunks(chunks)
    return chunks


def test_reingest_is_idempotent(indexed):
    stats = index_chunks(indexed)
    assert (stats.embedded, stats.deleted, stats.total) == (0, 0, 32)


def test_payload_carries_citation_fields_but_not_embed_text(indexed):
    hit = retrieve("penutupan tiket", k=1).hits[0]
    assert {"doc_id", "doc_title", "page", "section_no", "section_title", "text"} <= hit.payload.keys()
    assert "embed_text" not in hit.payload


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("tiket boleh ditutup kalau pelanggan diam saja", "KB01-p1-s07"),
        ("sev2 eskalasi max brp menit", "KB04-p1-s03"),  # informal + token persis "sev2"
        ("jam operasional layanan reguler", "KB03-p1-s01"),
    ],
)
def test_sanity_query_hits_expected_chunk_in_top3(indexed, query, expected):
    hits = retrieve(query, k=3, mode=Mode.hybrid).hits
    assert expected in [h.chunk_id for h in hits]


def test_doc_filter_restricts_results(indexed):
    hits = retrieve("pembaruan berkala gangguan", k=5, doc_ids=["KB_04"]).hits
    assert hits and all(h.payload["doc_id"] == "KB_04" for h in hits)


def test_dense_top1_is_cosine(indexed):
    assert 0.0 < retrieve("penutupan tiket", k=1).dense_top1 <= 1.0
