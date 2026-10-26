"""Pemuatan bge-m3 (dense 1024 + sparse lexical weights) lewat FlagEmbedding."""

from __future__ import annotations

from functools import lru_cache

from huggingface_hub import snapshot_download

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
