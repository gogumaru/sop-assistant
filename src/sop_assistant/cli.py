"""CLI `sop`. Impor berat (torch, model) dilakukan di dalam perintah agar `sop --help` cepat."""

from __future__ import annotations

import json
import time
from dataclasses import asdict

import typer
from rich.console import Console
from rich.markup import escape
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


STATUS_BADGE = {
    "answered": "[bold white on green] TERJAWAB [/]",
    "needs_review": "[bold black on yellow] PERLU DITINJAU [/]",
    "not_in_sop": "[bold white on red] TIDAK ADA DI SOP [/]",
}
REASON_TEXT = {
    "ambiguous": "Istilah di pertanyaan bisa merujuk ke dua skala berbeda. Pastikan klasifikasinya sebelum bertindak.",
    "partial": "Hanya sebagian pertanyaan yang diatur SOP.",
    "conflict": "Ada aturan SOP yang bertentangan. Konfirmasikan ke koordinator.",
    "guard_failed": "Sebagian jawaban tidak lolos pemeriksaan sumber; hanya klaim terverifikasi yang ditampilkan.",
}


@app.command()
def ask(
    question: str,
    as_json: bool = typer.Option(False, "--json", help="Cetak hasil lengkap sebagai JSON."),
    mode: Mode = typer.Option(Mode.hybrid, help="hybrid atau dense (ablation)."),
) -> None:
    """Tanya SOP. Status tampil di baris pertama, sumber di bawah jawaban."""
    import httpx

    from sop_assistant.answer import answer_question
    from sop_assistant.embedding import get_embedder
    from sop_assistant.engines import get_engine

    start = time.perf_counter()
    # Muat + pemanasan (inferensi pertama torch lambat) dipisah agar latensi menjawab tidak tercampur.
    get_embedder().encode(["pemanasan"], return_dense=True, return_sparse=True)
    load_s = time.perf_counter() - start

    try:
        result = answer_question(question, get_engine(), mode=mode)
    except httpx.HTTPError as exc:
        console.print(f"[red]LLM tidak bisa dihubungi ({settings.llm_engine}): {escape(str(exc))}[/]")
        raise typer.Exit(1) from exc

    if as_json:
        print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
        return

    reason = REASON_TEXT.get(result.review_reason or "", "")
    console.print(f"{STATUS_BADGE[result.status]} {escape(reason)}")
    if result.ambiguity:
        options = " | ".join(f"{c.level} ({c.note})" for c in result.ambiguity.candidates)
        console.print(f"[yellow]Kemungkinan maksud \"{escape(result.ambiguity.term)}\": {escape(options)}[/]")
    console.print(f"\n{escape(result.answer)}")
    if result.missing:
        console.print(f"\n[bold]Tidak diatur SOP:[/] {escape(result.missing)}")
    if result.escalate_to:
        console.print(f"[bold]Eskalasi ke:[/] {escape(result.escalate_to)}")

    if result.sources:
        console.print("\n[bold]Sumber[/]")
        for source in result.sources:
            console.print(f"  {escape(source.label())}")
            console.print(f'    [dim]"{escape(source.snippet)}"[/]')

    t = result.timings_ms
    parts = [f"dense top-1 {result.dense_top1:.3f}", f"retrieval {t.get('retrieval', 0):.0f} ms"]
    if "llm" in t:
        parts.append(f"LLM {t['llm']:.0f} ms" + (" (1x perbaikan)" if result.repaired else ""))
    parts += [f"total {t.get('total', 0):.0f} ms", f"muat model {load_s:.1f} dtk"]
    if result.abstained:
        parts.append("abstain sebelum LLM")
    if result.pii_masked:
        parts.append(f"{result.pii_masked} data sensitif disamarkan")
    console.print(f"\n[dim]{' | '.join(parts)}[/]")


if __name__ == "__main__":
    app()
