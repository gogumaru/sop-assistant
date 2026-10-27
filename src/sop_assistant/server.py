"""Server demo web. Membungkus `Assistant` yang sama dengan CLI dan eval; tidak ada logika jawaban di sini.

Endpoint:
- GET  /                         halaman UI (src/sop_assistant/web/)
- POST /api/ask                  stream langkah sistem lalu hasil akhir (text/event-stream)
- GET  /api/page/{chunk_id}.png  halaman PDF asli dengan section yang disitasi disorot
- POST /api/feedback             suka/tidak suka ke data/feedback.jsonl (pertanyaan sudah disamarkan)
"""

from __future__ import annotations

import json
import queue
import threading
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Literal

import httpx
import pymupdf
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from sop_assistant.answer import REASON_TEXT, STATUS_LABEL, AnswerResult, clarified_question
from sop_assistant.config import ROOT, settings

WEB_DIR = Path(__file__).parent / "web"
FEEDBACK_PATH = ROOT / "data" / "feedback.jsonl"
PAGE_DPI = 144
HIGHLIGHT_RGB = (0.886, 0.518, 0.227)  # oranye #E2843A
HIGHLIGHT_PAD = 6  # point di sekeliling bbox section


class AskRequest(BaseModel):
    question: str


class FeedbackRequest(BaseModel):
    question: str
    rating: Literal["up", "down"]
    status: str
    answer: str
    cited: list[str] = []


# --- Halaman PDF dengan highlight ---


@lru_cache(maxsize=1)
def chunk_index() -> dict[str, dict]:
    lines = settings.chunks_path.read_text(encoding="utf-8").splitlines()
    return {row["chunk_id"]: row for row in map(json.loads, filter(str.strip, lines))}


def _chunk(chunk_id: str) -> dict:
    if chunk_id not in chunk_index():
        raise HTTPException(404, f"Chunk tidak dikenal: {chunk_id}")
    return chunk_index()[chunk_id]


@lru_cache(maxsize=16)
def page_height(source_file: str, page: int) -> float:
    with pymupdf.open(settings.pdf_dir / source_file) as doc:
        return doc[page - 1].rect.height


@lru_cache(maxsize=64)
def render_highlight(chunk_id: str) -> bytes:
    chunk = _chunk(chunk_id)
    with pymupdf.open(settings.pdf_dir / chunk["source_file"]) as doc:
        page = doc[chunk["page"] - 1]
        rect = pymupdf.Rect(chunk["bbox"]) + (-HIGHLIGHT_PAD, -HIGHLIGHT_PAD, HIGHLIGHT_PAD, HIGHLIGHT_PAD)
        page.draw_rect(rect, color=HIGHLIGHT_RGB, fill=HIGHLIGHT_RGB, width=1.2, fill_opacity=0.16)
        return page.get_pixmap(dpi=PAGE_DPI).tobytes("png")


def focus_ratio(chunk_id: str) -> float:
    """Posisi atas section relatif terhadap tinggi halaman, untuk menggulir viewer ke highlight."""
    chunk = _chunk(chunk_id)
    return max(0.0, (chunk["bbox"][1] - HIGHLIGHT_PAD) / page_height(chunk["source_file"], chunk["page"]))


# --- Bentuk data untuk UI ---


def result_payload(result: AnswerResult) -> dict:
    data = asdict(result)
    data["status_label"] = STATUS_LABEL[result.status]
    data["reason_text"] = REASON_TEXT.get(result.review_reason or "", "")
    data["sources_role"] = "related" if result.status == "not_in_sop" else "sources"
    for source, original in zip(data["sources"], result.sources):
        source["label"] = original.label()
        source["page_url"] = f"/api/page/{original.chunk_id}.png"
        source["focus"] = focus_ratio(original.chunk_id)
    data["clarifications"] = (
        [
            {"label": c.level, "note": c.note, "question": clarified_question(result.question, c)}
            for c in result.ambiguity.candidates
        ]
        if result.ambiguity
        else []
    )
    return data


def friendly_error(exc: Exception) -> str:
    from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

    if isinstance(exc, (ResponseHandlingException, UnexpectedResponse)):
        return "Qdrant tidak bisa dihubungi. Jalankan `make qdrant-up`, lalu coba lagi."
    if isinstance(exc, httpx.HTTPError):
        return f"LLM tidak bisa dihubungi ({settings.llm_engine}). Pastikan aplikasi Ollama berjalan."
    return f"Terjadi kesalahan: {exc}"


# --- Aplikasi ---


def create_app(assistant_factory: Callable | None = None, on_ready: Callable[[], None] | None = None) -> FastAPI:
    state: dict = {}
    ask_lock = threading.Lock()  # satu model lokal: pertanyaan diproses bergiliran

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if assistant_factory is None:
            from sop_assistant.session import Assistant

            state["assistant"] = Assistant()
        else:
            state["assistant"] = assistant_factory()
        if on_ready:
            on_ready()
        yield

    app = FastAPI(title="SOP Assistant", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")

    @app.post("/api/ask")
    def ask(request: AskRequest) -> StreamingResponse:
        events: queue.Queue = queue.Queue()

        def on_stage(name: str, info: dict) -> None:
            events.put({"type": "step", "name": name, "info": info})

        def worker() -> None:
            try:
                with ask_lock:
                    result = state["assistant"].ask(request.question, on_stage=on_stage)
                events.put({"type": "result", "result": result_payload(result)})
            except Exception as exc:  # dikirim ke UI sebagai pesan yang bisa dibaca petugas
                events.put({"type": "error", "message": friendly_error(exc)})
            finally:
                events.put(None)

        threading.Thread(target=worker, daemon=True).start()

        def stream():
            while (event := events.get()) is not None:
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.get("/api/page/{chunk_id}.png")
    def page(chunk_id: str) -> Response:
        return Response(render_highlight(chunk_id), media_type="image/png", headers={"Cache-Control": "max-age=3600"})

    @app.post("/api/feedback")
    def feedback(request: FeedbackRequest) -> dict:
        FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
        record = {"at": datetime.now().isoformat(timespec="seconds"), **request.model_dump()}
        with FEEDBACK_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return {"ok": True}

    return app
