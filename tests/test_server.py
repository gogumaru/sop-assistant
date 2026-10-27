"""Test server demo web dengan asisten palsu (tanpa model, tanpa Ollama, tanpa Qdrant)."""

from __future__ import annotations

import json

import httpx
from fastapi.testclient import TestClient

from sop_assistant import server
from sop_assistant.answer import AnswerResult, Source
from sop_assistant.guards import detect_ambiguity

QUESTION = "Gangguan kritis harus di-update tiap berapa menit?"


class FakeAssistant:
    def ask(self, question, mode=None, on_stage=None):
        on_stage("retrieval", {})
        on_stage("retrieved", {"count": 1, "top_score": 0.65})
        return AnswerResult(
            question=question,
            status="needs_review",
            review_reason="ambiguous",
            answer="P1: 30 menit ke pelanggan.",
            sources=[Source("KB03-p1-s04", "KB_03", "Panduan Operasional Layanan", 1, 4, "Pembaruan Berkala", "Tiket P1 ...")],
            ambiguity=detect_ambiguity(QUESTION),
            timings_ms={"total": 12.0},
        )


class OllamaDown:
    def ask(self, question, mode=None, on_stage=None):
        raise httpx.ConnectError("connection refused")


def events(response) -> list[dict]:
    return [json.loads(block[6:]) for block in response.text.split("\n\n") if block.startswith("data: ")]


def test_ask_streams_steps_then_result():
    with TestClient(server.create_app(assistant_factory=FakeAssistant)) as client:
        response = client.post("/api/ask", json={"question": QUESTION})
    stream = events(response)
    assert [e["type"] for e in stream] == ["step", "step", "result"]
    result = stream[-1]["result"]
    assert (result["status_label"], result["sources_role"]) == ("PERLU DITINJAU", "sources")
    source = result["sources"][0]
    assert source["page_url"] == "/api/page/KB03-p1-s04.png"
    assert 0 < source["focus"] < 1
    assert [c["question"] for c in result["clarifications"]][0].endswith("untuk P1 Kritis?")


def test_ask_reports_readable_error_when_llm_is_down():
    with TestClient(server.create_app(assistant_factory=OllamaDown)) as client:
        stream = events(client.post("/api/ask", json={"question": QUESTION}))
    assert stream[-1]["type"] == "error" and "Ollama" in stream[-1]["message"]


def test_page_is_rendered_png():
    with TestClient(server.create_app(assistant_factory=FakeAssistant)) as client:
        response = client.get("/api/page/KB01-p1-s07.png")
        missing = client.get("/api/page/KB99-p1-s01.png")
    assert response.status_code == 200 and response.headers["content-type"] == "image/png"
    assert response.content.startswith(b"\x89PNG")
    assert missing.status_code == 404


def test_feedback_is_appended_locally(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "FEEDBACK_PATH", tmp_path / "feedback.jsonl")
    with TestClient(server.create_app(assistant_factory=FakeAssistant)) as client:
        client.post("/api/feedback", json={"question": QUESTION, "rating": "down", "status": "answered", "answer": "x"})
    record = json.loads((tmp_path / "feedback.jsonl").read_text(encoding="utf-8"))
    assert record["rating"] == "down"


def test_index_is_served():
    with TestClient(server.create_app(assistant_factory=FakeAssistant)) as client:
        response = client.get("/")
    assert response.status_code == 200 and "Asisten SOP" in response.text
