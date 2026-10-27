"""LLM engine yang bisa ditukar. Pipeline hanya bergantung pada protocol `LLMEngine`."""

from __future__ import annotations

import json
import re
from typing import Protocol

import httpx

from sop_assistant.config import settings

# Default konteks Ollama terlalu kecil untuk baseline full-context (32 chunk + system prompt).
OLLAMA_NUM_CTX = 8192
# Default Ollama membongkar model setelah 5 menit; petugas shift bertanya dengan jeda lebih panjang.
OLLAMA_KEEP_ALIVE = "30m"


class LLMEngine(Protocol):
    name: str

    def generate_json(self, system: str, user: str, schema: dict) -> dict: ...


class OllamaEngine:
    """Default on-device. `format` = JSON schema membatasi decoding ke struktur yang valid."""

    def __init__(self, model: str | None = None, url: str | None = None, timeout: float | None = None):
        self.model = model or settings.ollama_model
        self.url = url or settings.ollama_url
        self.timeout = timeout or settings.llm_timeout
        self.name = f"ollama:{self.model}"

    def warm_up(self) -> None:
        """Muat model ke memori tanpa membuat teks, agar pertanyaan pertama tidak menanggung waktu muat."""
        response = httpx.post(
            f"{self.url}/api/generate",
            json={"model": self.model, "keep_alive": OLLAMA_KEEP_ALIVE},
            timeout=self.timeout,
        )
        response.raise_for_status()

    def generate_json(self, system: str, user: str, schema: dict) -> dict:
        payload = {
            "model": self.model,
            "stream": False,
            "keep_alive": OLLAMA_KEEP_ALIVE,
            "format": schema,
            "options": {"temperature": 0, "seed": 0, "num_ctx": OLLAMA_NUM_CTX},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        response = httpx.post(f"{self.url}/api/chat", json=payload, timeout=self.timeout)
        response.raise_for_status()
        return json.loads(response.json()["message"]["content"])


class OpenAICompatEngine:
    """Untuk reviewer tanpa Mac: LM Studio, vLLM, atau provider OpenAI-compatible. Implementasi minimal."""

    def __init__(self) -> None:
        self.base_url = settings.openai_base_url
        self.api_key = settings.openai_api_key
        self.model = settings.openai_model
        self.timeout = settings.llm_timeout
        self.name = f"openai_compat:{self.model}"

    def generate_json(self, system: str, user: str, schema: dict) -> dict:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        body = {
            "model": self.model,
            "temperature": 0,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "sop_answer", "schema": schema}},
        }
        response = httpx.post(f"{self.base_url}/chat/completions", json=body, headers=headers, timeout=self.timeout)
        if response.status_code == 400:  # provider tanpa dukungan json_schema: kirim skema lewat prompt
            body["response_format"] = {"type": "json_object"}
            body["messages"][0]["content"] = f"{system}\n\nBalas hanya JSON dengan skema:\n{json.dumps(schema)}"
            response = httpx.post(
                f"{self.base_url}/chat/completions", json=body, headers=headers, timeout=self.timeout
            )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        return json.loads(re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip()))


class FakeEngine:
    """Untuk test pipeline tanpa LLM: mengembalikan respons yang sudah disiapkan secara berurutan."""

    name = "fake"

    def __init__(self, responses: list[dict]):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def generate_json(self, system: str, user: str, schema: dict) -> dict:
        self.calls.append({"system": system, "user": user, "schema": schema})
        if not self.responses:
            raise AssertionError("FakeEngine kehabisan respons")
        return self.responses.pop(0)


def get_engine() -> LLMEngine:
    if settings.llm_engine == "ollama":
        return OllamaEngine()
    if settings.llm_engine == "openai_compat":
        return OpenAICompatEngine()
    raise ValueError(f"LLM_ENGINE tidak dikenal: {settings.llm_engine} (pilih ollama atau openai_compat)")
