"""Index chunk ke Qdrant: named vector `dense` (1024, cosine) + sparse vector `sparse`.

Ingest ulang aman: point ID adalah UUID5 dari chunk_id (upsert, bukan duplikat),
chunk dengan content_hash sama tidak di-embed ulang, dan chunk yang sudah tidak ada
di korpus dihapus dari koleksi.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass
from functools import lru_cache

from qdrant_client import QdrantClient, models

from sop_assistant.config import settings
from sop_assistant.extract import Chunk

DENSE = "dense"
SPARSE = "sparse"
DENSE_DIM = 1024


@dataclass(frozen=True)
class IndexStats:
    embedded: int
    unchanged: int
    deleted: int
    total: int


@lru_cache(maxsize=1)
def get_client() -> QdrantClient:
    """Server Docker jika QDRANT_URL diisi; selain itu mode embedded di QDRANT_PATH."""
    if settings.qdrant_url:
        return QdrantClient(url=settings.qdrant_url)
    return QdrantClient(path=str(settings.qdrant_path))


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"sop-assistant/{chunk_id}"))


def ensure_collection(client: QdrantClient, recreate: bool = False) -> None:
    exists = client.collection_exists(settings.collection)
    if exists and recreate:
        client.delete_collection(settings.collection)
        exists = False
    if exists:
        return
    client.create_collection(
        settings.collection,
        vectors_config={DENSE: models.VectorParams(size=DENSE_DIM, distance=models.Distance.COSINE)},
        sparse_vectors_config={SPARSE: models.SparseVectorParams()},
    )
    if settings.qdrant_url:  # payload index tidak didukung mode embedded
        client.create_payload_index(settings.collection, "doc_id", models.PayloadSchemaType.KEYWORD)


def existing_hashes(client: QdrantClient) -> dict[str, str]:
    hashes: dict[str, str] = {}
    offset = None
    while True:
        points, offset = client.scroll(
            settings.collection, limit=256, offset=offset, with_payload=["content_hash"], with_vectors=False
        )
        hashes.update({str(p.id): p.payload.get("content_hash", "") for p in points})
        if offset is None:
            return hashes


def payload(chunk: Chunk) -> dict:
    data = asdict(chunk)
    data.pop("embed_text")
    return data


def index_chunks(chunks: list[Chunk], recreate: bool = False) -> IndexStats:
    from sop_assistant.embedding import encode  # impor torch hanya jika memang perlu embed

    client = get_client()
    ensure_collection(client, recreate)

    current = {point_id(c.chunk_id): c for c in chunks}
    existing = existing_hashes(client)

    stale = [pid for pid in existing if pid not in current]
    if stale:
        client.delete(settings.collection, points_selector=models.PointIdsList(points=stale))

    todo = [c for pid, c in current.items() if existing.get(pid) != c.content_hash]
    if todo:
        dense, sparse = encode([c.embed_text for c in todo])
        client.upsert(
            settings.collection,
            points=[
                models.PointStruct(id=point_id(c.chunk_id), vector={DENSE: d, SPARSE: s}, payload=payload(c))
                for c, d, s in zip(todo, dense, sparse)
            ],
        )

    total = client.count(settings.collection, exact=True).count
    return IndexStats(embedded=len(todo), unchanged=len(chunks) - len(todo), deleted=len(stale), total=total)
