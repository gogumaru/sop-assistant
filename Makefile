# Semua perintah Python dijalankan lewat `uv run` agar memakai .venv proyek (Python 3.12).
-include .env
OLLAMA_MODEL ?= qwen2.5:7b
UV := uv run
# Wheel besar (torch, transformers) sering timeout di default uv (30 dtk).
UV_HTTP_TIMEOUT ?= 120
export UV_HTTP_TIMEOUT

.PHONY: help setup models qdrant-up qdrant-down check-pdfs smoke ingest test calibrate eval demo requirements

help: ## Tampilkan daftar perintah
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "} {printf "  %-13s %s\n", $$1, $$2}'

setup: ## Buat .venv (Python 3.12), pasang dependency, buat .env
	uv sync
	@test -f .env || (cp .env.example .env && echo ".env dibuat dari .env.example")

models: ## Unduh bge-m3 dari HuggingFace dan model Ollama (sekali saja)
	$(UV) python -c "from sop_assistant.embedding import model_path; print(model_path())"
	ollama pull $(OLLAMA_MODEL)

qdrant-up: ## Jalankan server Qdrant (Docker)
	docker compose up -d qdrant

qdrant-down: ## Hentikan server Qdrant (data tetap di volume)
	docker compose down

check-pdfs: ## M0: cek PDF mana yang punya text layer dan mana yang perlu OCR
	$(UV) python scripts/check_pdfs.py

smoke: ## M0: cek Tesseract, Qdrant, bge-m3, dan Ollama siap
	$(UV) python scripts/smoke.py

ingest: ## M2: ekstrak PDF, tulis data/chunks.jsonl, index ke Qdrant
	$(UV) sop ingest

test: ## Jalankan pytest
	$(UV) pytest -q

calibrate: ## M4: hitung threshold abstain dari golden set
	$(UV) python eval/calibrate.py

eval: ## M4: evaluasi per tahap, tulis eval/report.md
	$(UV) python eval/run_eval.py

demo: ## M5: jalankan pertanyaan demo, tulis demo/demo_output.md
	$(UV) python demo/run_demo.py

requirements: ## Ekspor requirements.txt untuk pengguna pip
	uv export --no-hashes --no-dev --format requirements-txt -o requirements.txt
