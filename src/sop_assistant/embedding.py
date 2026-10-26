"""Pemuatan bge-m3 (dense 1024 + sparse lexical weights) lewat FlagEmbedding."""

from __future__ import annotations

from functools import lru_cache

from huggingface_hub import snapshot_download
from qdrant_client import models

from sop_assistant.config import settings

# Versi ONNX dan gambar README tidak dipakai FlagEmbedding (sekitar 2.2 GB).
MODEL_IGNORE_PATTERNS = ["onnx/*", "imgs/*"]


def model_path() -> str:
    """Folder lokal model; unduh sekali jika belum ada di cache HuggingFace.

    Path folder diberikan ke FlagEmbedding, bukan nama repo, karena dengan nama repo
    FlagEmbedding mengunduh sendiri seluruh isi repo termasuk file ONNX.
    """
    return snapshot_download(settings.embed_model, ignore_patterns=MODEL_IGNORE_PATTERNS)


@lru_cache(maxsize=1)
def get_embedder():
    from FlagEmbedding import BGEM3FlagModel

    return BGEM3FlagModel(model_path(), use_fp16=settings.embed_fp16, devices=settings.embed_device)


def to_sparse(weights: dict[str, float]) -> models.SparseVector:
    """lexical_weights bge-m3 ({"token_id": bobot}) ke format sparse Qdrant."""
    return models.SparseVector(indices=[int(token) for token in weights], values=[float(w) for w in weights.values()])


def encode(texts: list[str]) -> tuple[list[list[float]], list[models.SparseVector]]:
    """Satu forward pass menghasilkan vektor dense dan sparse sekaligus."""
    out = get_embedder().encode(texts, return_dense=True, return_sparse=True)
    return out["dense_vecs"].tolist(), [to_sparse(weights) for weights in out["lexical_weights"]]
