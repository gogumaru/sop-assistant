"""Retrieval: hybrid (dense + sparse, fusion RRF) dan dense saja untuk ablation.

Skor RRF hanya mencerminkan peringkat dan tidak sebanding antar-query, jadi keputusan
abstain memakai cosine dense top-1 yang dihitung terpisah (`RetrievalResult.dense_top1`).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

from qdrant_client import models

from sop_assistant.config import settings
from sop_assistant.embedding import encode
from sop_assistant.index import DENSE, SPARSE, get_client


class Mode(str, Enum):
    hybrid = "hybrid"
    dense = "dense"


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    score: float
    payload: dict


@dataclass(frozen=True)
class RetrievalResult:
    query: str
    mode: Mode
    hits: list[Hit]
    dense_top1: float


def _doc_filter(doc_ids: Iterable[str] | None) -> models.Filter | None:
    if not doc_ids:
        return None
    return models.Filter(must=[models.FieldCondition(key="doc_id", match=models.MatchAny(any=list(doc_ids)))])


def _to_hits(points: list[models.ScoredPoint]) -> list[Hit]:
    return [Hit(chunk_id=p.payload["chunk_id"], score=p.score, payload=p.payload) for p in points]


def dense_search(dense: list[float], k: int, doc_ids: Iterable[str] | None = None) -> list[Hit]:
    response = get_client().query_points(
        settings.collection, query=dense, using=DENSE, query_filter=_doc_filter(doc_ids), limit=k, with_payload=True
    )
    return _to_hits(response.points)


def hybrid_search(
    dense: list[float], sparse: models.SparseVector, k: int, doc_ids: Iterable[str] | None = None
) -> list[Hit]:
    doc_filter = _doc_filter(doc_ids)
    response = get_client().query_points(
        settings.collection,
        prefetch=[
            models.Prefetch(query=dense, using=DENSE, limit=settings.prefetch_limit, filter=doc_filter),
            models.Prefetch(query=sparse, using=SPARSE, limit=settings.prefetch_limit, filter=doc_filter),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=k,
        with_payload=True,
    )
    return _to_hits(response.points)


def retrieve(
    query: str, k: int | None = None, mode: Mode = Mode.hybrid, doc_ids: Iterable[str] | None = None
) -> RetrievalResult:
    """Query di-embed sekali. dense_top1 selalu dihitung tanpa filter dokumen."""
    k = k or settings.retrieve_k
    dense_vecs, sparse_vecs = encode([query])
    dense, sparse = dense_vecs[0], sparse_vecs[0]
    top1 = dense_search(dense, 1)
    hits = hybrid_search(dense, sparse, k, doc_ids) if mode == Mode.hybrid else dense_search(dense, k, doc_ids)
    return RetrievalResult(query=query, mode=mode, hits=hits, dense_top1=top1[0].score if top1 else 0.0)


def dense_top1_score(query: str) -> float:
    return retrieve(query, k=1, mode=Mode.dense).dense_top1
