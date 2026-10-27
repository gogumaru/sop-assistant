"""CLI `sop`. Impor berat (torch, model) dilakukan di dalam perintah agar `sop --help` cepat."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from enum import Enum

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


STATUS_STYLE = {"answered": "bold white on green", "needs_review": "bold black on yellow", "not_in_sop": "bold white on red"}


@app.command()
def ask(
    question: str,
    as_json: bool = typer.Option(False, "--json", help="Cetak hasil lengkap sebagai JSON."),
    mode: Mode = typer.Option(Mode.hybrid, help="hybrid atau dense (ablation)."),
) -> None:
    """Tanya SOP. Status tampil di baris pertama, sumber di bawah jawaban."""
    import httpx

    from sop_assistant.session import Assistant

    assistant = Assistant()
    try:
        result = assistant.ask(question, mode=mode)
    except httpx.HTTPError as exc:
        console.print(f"[red]LLM tidak bisa dihubungi ({settings.llm_engine}): {escape(str(exc))}[/]")
        raise typer.Exit(1) from exc

    if as_json:
        print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
        return
    render_result(result, load_s=assistant.load_s)


STAGE_TEXT = {
    "retrieval": "Mencari di SOP...",
    "llm": "Menyusun jawaban...",
    "repair": "Jawaban ditolak pemeriksa sumber, meminta perbaikan...",
}
CHAT_HELP = """\
Perintah:
  /mode hybrid|dense   ganti mode retrieval
  /json                hasil lengkap pertanyaan terakhir
  /bantuan             tampilkan bantuan ini
  /keluar              berhenti (atau Ctrl+D)"""


@app.command()
def chat(mode: Mode = typer.Option(Mode.hybrid, help="hybrid atau dense (ablation).")) -> None:
    """Mode interaktif: model dimuat sekali, lalu tanya berulang kali tanpa menunggu muat ulang."""
    import httpx

    from sop_assistant.session import Assistant

    with console.status("Memuat model..."):
        assistant = Assistant()
    console.print(f"Siap dalam {assistant.load_s:.1f} dtk. Ketik pertanyaan, /bantuan untuk perintah.\n")

    while True:
        try:
            line = console.input("[bold]sop>[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            break
        if not line:
            continue

        if line.startswith("/"):
            command, _, arg = line[1:].partition(" ")
            if command in {"keluar", "exit", "quit"}:
                break
            if command == "mode":
                try:
                    mode = Mode(arg.strip())
                    console.print(f"Mode retrieval: {mode.value}")
                except ValueError:
                    console.print("Mode harus hybrid atau dense.")
            elif command == "json":
                if assistant.history:
                    print(json.dumps(asdict(assistant.history[-1]), ensure_ascii=False, indent=2))
                else:
                    console.print("Belum ada pertanyaan.")
            else:
                console.print(CHAT_HELP)
            continue

        try:
            with console.status(STAGE_TEXT["retrieval"]) as status:
                result = assistant.ask(
                    line, mode=mode, on_stage=lambda name, info: name in STAGE_TEXT and status.update(STAGE_TEXT[name])
                )
        except httpx.HTTPError as exc:
            console.print(f"[red]LLM tidak bisa dihubungi ({settings.llm_engine}): {escape(str(exc))}[/]")
            continue
        except KeyboardInterrupt:
            console.print("Dibatalkan.")
            continue
        render_result(result)
        console.print()


@app.command()
def ui(
    port: int = typer.Option(8000, help="Port lokal."),
    open_browser: bool = typer.Option(True, "--open/--no-open", help="Buka browser setelah model siap."),
) -> None:
    """Demo web di http://127.0.0.1:<port>, memakai pipeline yang sama dengan CLI dan eval."""
    import webbrowser

    import uvicorn

    from sop_assistant.server import create_app

    url = f"http://127.0.0.1:{port}"
    console.print(f"Memuat model... UI tersedia di {url} setelah siap (Ctrl+C untuk berhenti)")
    on_ready = (lambda: webbrowser.open(url)) if open_browser else None
    uvicorn.run(create_app(on_ready=on_ready), host="127.0.0.1", port=port, log_level="warning")


def render_result(result, load_s: float | None = None) -> None:
    from sop_assistant.answer import REASON_TEXT, STATUS_LABEL

    reason = REASON_TEXT.get(result.review_reason or "", "")
    console.print(f"[{STATUS_STYLE[result.status]}] {STATUS_LABEL[result.status]} [/] {escape(reason)}")
    if result.ambiguity:
        options = " | ".join(f"{c.level} ({c.note})" for c in result.ambiguity.candidates)
        console.print(f"[yellow]Kemungkinan maksud \"{escape(result.ambiguity.term)}\": {escape(options)}[/]")
    console.print(f"\n{escape(result.answer)}")
    if result.missing:
        console.print(f"\n[bold]Tidak diatur SOP:[/] {escape(result.missing)}")
    if result.escalate_to:
        by_rule = " [dim](diisi aturan eskalasi)[/]" if result.escalation_by_rule else ""
        console.print(f"[bold]Eskalasi ke:[/] {escape(result.escalate_to)}{by_rule}")

    if result.sources:
        # Untuk not_in_sop, chunk yang disitasi bukan jawaban, hanya aturan di sekitarnya.
        console.print("\n[bold]Aturan terkait[/]" if result.status == "not_in_sop" else "\n[bold]Sumber[/]")
        for source in result.sources:
            console.print(f"  {escape(source.label())}")
            console.print(f'    [dim]"{escape(source.snippet)}"[/]')

    t = result.timings_ms
    parts = [f"dense top-1 {result.dense_top1:.3f}", f"retrieval {t.get('retrieval', 0):.0f} ms"]
    if "llm" in t:
        parts.append(f"LLM {t['llm']:.0f} ms" + (" (1x perbaikan)" if result.repaired else ""))
    parts.append(f"total {t.get('total', 0):.0f} ms")
    if load_s is not None:
        parts.append(f"muat model {load_s:.1f} dtk")
    if result.abstained:
        parts.append("abstain sebelum LLM")
    if result.pii_masked:
        parts.append(f"{result.pii_masked} data sensitif disamarkan")
    console.print(f"\n[dim]{' | '.join(parts)}[/]")


class Baseline(str, Enum):
    full_context = "full-context"


@app.command("eval")
def eval_command(
    mode: Mode = typer.Option(Mode.hybrid, help="hybrid atau dense (ablation)."),
    baseline: Baseline | None = typer.Option(None, help="full-context: seluruh korpus sebagai konteks, tanpa retrieval."),
    report_only: bool = typer.Option(False, "--report-only", help="Bangun ulang eval/report.md saja."),
) -> None:
    """Evaluasi golden set per tahap, tulis eval/results/<config>.json dan eval/report.md."""
    from sop_assistant.evaluation import CONFIG_LABEL, run, summarize, write_report

    if not report_only:
        data = run("full_context" if baseline else mode.value)
        table = Table(title=f"Ringkasan: {CONFIG_LABEL[data['config']]}")
        table.add_column("Metrik")
        table.add_column("Nilai")
        for metric, value in summarize(data).items():
            table.add_row(metric, value)
        console.print(table)
    console.print(f"Laporan: {write_report()}")


if __name__ == "__main__":
    app()
