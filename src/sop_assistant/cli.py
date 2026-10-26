"""CLI `sop`. Impor berat (torch, model) dilakukan di dalam perintah agar `sop --help` cepat."""

from __future__ import annotations

import time

import typer
from rich.console import Console
from rich.table import Table

from sop_assistant.config import settings
from sop_assistant.retrieve import Mode

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Asisten SOP on-device dengan sitasi dokumen dan halaman.",
)
console = Console()


@app.command()
def ingest(recreate: bool = typer.Option(False, "--recreate", help="Hapus dan buat ulang koleksi.")) -> None:
    """Ekstrak PDF, tulis data/chunks.jsonl, index ke Qdrant (upsert idempotent)."""
    from sop_assistant.extract import extract_corpus, write_chunks
    from sop_assistant.index import index_chunks

    start = time.perf_counter()
    chunks = extract_corpus()
    path = write_chunks(chunks)
    stats = index_chunks(chunks, recreate=recreate)
    target = settings.qdrant_url or f"embedded {settings.qdrant_path}"
    console.print(f"{len(chunks)} chunk ditulis ke {path}")
    console.print(
        f"Qdrant ({target}, koleksi {settings.collection}): {stats.embedded} di-embed, "
        f"{stats.unchanged} tidak berubah, {stats.deleted} dihapus, total {stats.total} point"
    )
    console.print(f"Selesai dalam {time.perf_counter() - start:.1f} dtk")


@app.command()
def search(
    query: str,
    k: int = typer.Option(5, help="Jumlah chunk."),
    mode: Mode = typer.Option(Mode.hybrid, help="hybrid atau dense (ablation)."),
) -> None:
    """Debug retrieval: tampilkan chunk teratas tanpa memanggil LLM."""
    from sop_assistant.retrieve import retrieve

    start = time.perf_counter()
    result = retrieve(query, k=k, mode=mode)
    elapsed = time.perf_counter() - start

    table = Table(title=f"{mode.value}: {query}")
    for column in ("#", "chunk_id", "Sumber", "Skor", "Cuplikan"):
        table.add_column(column)
    for rank, hit in enumerate(result.hits, 1):
        p = hit.payload
        source = f"{p['doc_id']} hal. {p['page']}, §{p['section_no']} {p['section_title']}"
        table.add_row(str(rank), hit.chunk_id, source, f"{hit.score:.3f}", p["text"][:60])
    console.print(table)

    threshold = settings.abstain_threshold
    threshold_note = f"threshold {threshold:.3f}" if threshold is not None else "threshold belum dikalibrasi (M4)"
    console.print(f"dense top-1 cosine: {result.dense_top1:.3f} ({threshold_note})")
    console.print(f"Latensi retrieval: {elapsed * 1000:.0f} ms (termasuk memuat model pada panggilan pertama)")


if __name__ == "__main__":
    app()
