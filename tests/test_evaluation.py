"""Test golden set dan metrik evaluasi (tanpa LLM, tanpa Qdrant)."""

from __future__ import annotations

import pytest

from sop_assistant.evaluation import (
    choose_threshold,
    citation_ok,
    full_context_retriever,
    load_chunk_rows,
    load_golden,
    missing_phrases,
    retrieval_metrics,
)


@pytest.fixture(scope="module")
def golden():
    return load_golden()


def test_golden_matches_brief(golden):
    assert len(golden) == 21
    assert len({item.id for item in golden}) == 21
    assert sum(not item.in_scope for item in golden) == 3


def test_golden_expected_chunks_exist(golden):
    chunk_ids = {row["chunk_id"] for row in load_chunk_rows()}
    for item in golden:
        assert set(item.expected_chunks) <= chunk_ids, item.id
        assert bool(item.expected_chunks) == item.in_scope, item.id


def test_retrieval_metrics():
    assert retrieval_metrics(["a", "b", "c"], ["b"]) == (1.0, 0.5)
    assert retrieval_metrics(["a", "b"], ["b", "z"]) == (0.5, 0.5)
    assert retrieval_metrics(["a"], ["z"]) == (0.0, 0.0)


def test_must_include_normalizes_word_numbers():
    assert missing_phrases("Paling lambat lima hari kerja untuk SEV1 dan SEV2.", ["5 hari kerja", "SEV1"]) == []
    assert missing_phrases("Tiket diperbarui setiap 30 menit.", ["20 menit"]) == ["20 menit"]


def test_citation_rule_depends_on_category():
    assert citation_ok(["KB03-p1-s04"], ["KB03-p1-s04", "KB04-p1-s05"], "factual")
    assert not citation_ok(["KB03-p1-s04"], ["KB03-p1-s04", "KB04-p1-s05"], "ambiguous")
    assert citation_ok(["KB04-p1-s07", "KB01-p1-s07"], ["KB04-p1-s07", "KB01-p1-s07"], "multi_hop")


@pytest.mark.parametrize(
    ("in_scope", "out_scope", "expected"),
    [
        ([0.60, 0.70], [0.40, 0.50], 0.55),  # ada celah: tengah celah
        ([0.60, 0.70], [0.40, 0.65], 0.50),  # tumpang tindih: tengah celah yang tersisa
        ([0.60, 0.70], [0.62, 0.65], 0.58),  # semua di atas: tepat di bawah in-scope terendah
    ],
)
def test_choose_threshold_never_abstains_in_scope(in_scope, out_scope, expected):
    threshold, _ = choose_threshold(in_scope, out_scope)
    assert threshold == pytest.approx(expected)
    assert threshold < min(in_scope)


def test_full_context_retriever_returns_whole_corpus():
    retriever = full_context_retriever(load_chunk_rows())
    assert len(retriever("apa saja").hits) == 32
    assert {h.payload["doc_id"] for h in retriever("x", doc_ids=["KB_04"]).hits} == {"KB_04"}
