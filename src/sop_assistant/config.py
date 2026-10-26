"""Konfigurasi terpusat. Nilai dibaca dari .env (lihat .env.example)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

CALIBRATION_FILE = ROOT / "eval" / "calibration.json"


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes"}


def _abstain_threshold() -> float | None:
    """Env ABSTAIN_THRESHOLD menang; jika kosong pakai hasil make calibrate; jika belum ada, None."""
    if value := os.getenv("ABSTAIN_THRESHOLD", "").strip():
        return float(value)
    if CALIBRATION_FILE.exists():
        return float(json.loads(CALIBRATION_FILE.read_text())["abstain_threshold"])
    return None


@dataclass(frozen=True)
class Settings:
    # Korpus. Hanya file KB_*.pdf yang diindeks; dokumen assignment tidak pernah ikut.
    pdf_dir: Path
    pdf_glob: str
    chunks_path: Path
    # OCR
    ocr_lang: str
    ocr_dpi: int
    ocr_min_chars: int
    # Qdrant. qdrant_url kosong = mode embedded di qdrant_path.
    qdrant_url: str
    qdrant_path: Path
    collection: str
    # Embedding
    embed_model: str
    embed_device: str
    embed_fp16: bool
    # LLM
    llm_engine: str
    ollama_url: str
    ollama_model: str
    openai_base_url: str
    openai_api_key: str
    openai_model: str
    llm_timeout: float
    # Retrieval & guard
    retrieve_k: int
    prefetch_limit: int
    abstain_threshold: float | None


def load_settings() -> Settings:
    return Settings(
        pdf_dir=ROOT / "data" / "pdfs",
        pdf_glob="KB_*.pdf",
        chunks_path=ROOT / "data" / "chunks.jsonl",
        ocr_lang=os.getenv("OCR_LANG", "ind"),
        ocr_dpi=int(os.getenv("OCR_DPI", "300")),
        ocr_min_chars=int(os.getenv("OCR_MIN_CHARS", "50")),
        qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333").strip(),
        qdrant_path=ROOT / os.getenv("QDRANT_PATH", "data/qdrant_local"),
        collection=os.getenv("QDRANT_COLLECTION", "sop_chunks"),
        embed_model=os.getenv("EMBED_MODEL", "BAAI/bge-m3"),
        embed_device=os.getenv("EMBED_DEVICE", "cpu"),
        embed_fp16=_bool("EMBED_FP16", False),
        llm_engine=os.getenv("LLM_ENGINE", "ollama"),
        ollama_url=os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/"),
        ollama_model=os.getenv("OLLAMA_MODEL", "qwen2.5:7b"),
        openai_base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:1234/v1").rstrip("/"),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_model=os.getenv("OPENAI_MODEL", ""),
        llm_timeout=float(os.getenv("LLM_TIMEOUT", "120")),
        retrieve_k=int(os.getenv("RETRIEVE_K", "5")),
        prefetch_limit=int(os.getenv("PREFETCH_LIMIT", "20")),
        abstain_threshold=_abstain_threshold(),
    )


settings = load_settings()
