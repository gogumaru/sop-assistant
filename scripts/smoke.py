"""M0 smoke test: pastikan Tesseract, Qdrant, bge-m3, dan Ollama siap sebelum M1.

Tiap cek berdiri sendiri; satu gagal tidak menghentikan cek lain.
Cek embedding pertama kali bisa lama karena memuat model (sekitar 2 GB).
"""

from __future__ import annotations

import json
import shutil
import sys
import time

import httpx
from rich.console import Console

from sop_assistant.config import settings

console = Console()

# Skema mini untuk membuktikan Ollama menghormati parameter `format` (JSON schema).
SMOKE_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["ok"]},
        "angka": {"type": "integer"},
    },
    "required": ["status", "angka"],
}


def check_tesseract() -> str:
    import pytesseract

    if not shutil.which("tesseract"):
        raise RuntimeError("tesseract tidak ditemukan. Jalankan: brew install tesseract tesseract-lang")
    langs = pytesseract.get_languages(config="")
    if settings.ocr_lang not in langs:
        raise RuntimeError(f"bahasa '{settings.ocr_lang}' belum terpasang. Jalankan: brew install tesseract-lang")
    return f"versi {pytesseract.get_tesseract_version()}, bahasa '{settings.ocr_lang}' tersedia"


def check_qdrant() -> str:
    from qdrant_client import QdrantClient

    if settings.qdrant_url:
        client, mode = QdrantClient(url=settings.qdrant_url), f"server {settings.qdrant_url}"
    else:
        client, mode = QdrantClient(path=str(settings.qdrant_path)), f"embedded {settings.qdrant_path}"
    names = [c.name for c in client.get_collections().collections]
    return f"{mode}, koleksi: {names or 'belum ada'}"


def check_embedding() -> str:
    from sop_assistant.embedding import get_embedder

    start = time.perf_counter()
    out = get_embedder().encode(["Tiket P1 diperbarui setiap 30 menit."], return_dense=True, return_sparse=True)
    dim = out["dense_vecs"].shape[-1]
    n_sparse = len(out["lexical_weights"][0])
    if dim != 1024:
        raise RuntimeError(f"dimensi dense {dim}, seharusnya 1024")
    if n_sparse == 0:
        raise RuntimeError("lexical_weights kosong, vektor sparse tidak terbentuk")
    elapsed = time.perf_counter() - start
    return f"dense {dim} dim, sparse {n_sparse} token, {elapsed:.1f} dtk termasuk memuat model ({settings.embed_device})"


def check_ollama() -> str:
    payload = {
        "model": settings.ollama_model,
        "stream": False,
        "format": SMOKE_SCHEMA,
        "options": {"temperature": 0},
        "messages": [{"role": "user", "content": "Balas dalam JSON: status 'ok' dan angka 15."}],
    }
    start = time.perf_counter()
    try:
        response = httpx.post(f"{settings.ollama_url}/api/chat", json=payload, timeout=settings.llm_timeout)
        response.raise_for_status()
    except httpx.ConnectError as exc:
        raise RuntimeError(f"Ollama tidak bisa dihubungi di {settings.ollama_url}. Jalankan aplikasi Ollama.") from exc
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(f"{exc.response.text}. Coba: ollama pull {settings.ollama_model}") from exc
    data = json.loads(response.json()["message"]["content"])
    if data.get("status") != "ok":
        raise RuntimeError(f"JSON tidak sesuai skema: {data}")
    return f"{settings.ollama_model} -> {data}, {time.perf_counter() - start:.1f} dtk"


CHECKS = [
    ("Tesseract", check_tesseract),
    ("Qdrant", check_qdrant),
    ("Embedding bge-m3", check_embedding),
    ("Ollama JSON schema", check_ollama),
]


def main() -> int:
    failed = 0
    for name, check in CHECKS:
        try:
            console.print(f"[bold green]OK[/]    {name}: {check()}")
        except Exception as exc:  # smoke test: laporkan semua kegagalan, jangan berhenti di yang pertama
            failed += 1
            console.print(f"[bold red]GAGAL[/] {name}: {exc}")
    console.print(f"\n{len(CHECKS) - failed}/{len(CHECKS)} cek lolos.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
