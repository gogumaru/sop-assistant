"""Sesi yang memuat model sekali lalu menjawab berulang kali. Dipakai `sop ask`, `sop chat`, dan server web."""

from __future__ import annotations

import time

import httpx

from sop_assistant.answer import AnswerResult, StageCallback, answer_question
from sop_assistant.embedding import get_embedder
from sop_assistant.engines import LLMEngine, get_engine
from sop_assistant.retrieve import Mode


class Assistant:
    def __init__(self, engine: LLMEngine | None = None):
        start = time.perf_counter()
        # Inferensi pertama torch lambat; dipanaskan di sini agar latensi menjawab tidak tercampur.
        get_embedder().encode(["pemanasan"], return_dense=True, return_sparse=True)
        self.engine = engine or get_engine()
        if warm_up := getattr(self.engine, "warm_up", None):
            try:
                warm_up()
            except httpx.HTTPError:
                pass  # LLM belum siap; error yang jelas muncul saat pertanyaan pertama
        self.load_s = time.perf_counter() - start
        self.history: list[AnswerResult] = []

    def ask(self, question: str, mode: Mode = Mode.hybrid, on_stage: StageCallback | None = None) -> AnswerResult:
        result = answer_question(question, self.engine, mode=mode, on_stage=on_stage)
        self.history.append(result)
        return result
