# Semua perintah Python dijalankan lewat `uv run` agar memakai .venv proyek (Python 3.12).
-include .env
OLLAMA_MODEL ?= qwen2.5:7b
UV := uv run
# Wheel besar (torch, transformers) sering timeout di default uv (30 dtk).
UV_HTTP_TIMEOUT ?= 120
export UV_HTTP_TIMEOUT

.PHONY: help setup models qdrant-up qdrant-down check-pdfs smoke extract before-after ingest search ask chat test test-unit calibrate eval eval-ablation report demo requirements

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

extract: ## M1: ekstrak PDF ke data/chunks.jsonl (tanpa indexing)
	$(UV) python -m sop_assistant.extract

before-after: ## M1: tulis docs/two_column_before_after.md untuk README
	$(UV) python scripts/before_after.py

ingest: ## M2: ekstrak PDF, tulis data/chunks.jsonl, index ke Qdrant
	$(UV) sop ingest

search: ## M2: debug retrieval, contoh: make search Q="brp lama sev2 hrs dieskalasi"
	$(UV) sop search "$(Q)"

ask: ## M3: tanya SOP, contoh: make ask Q="Berapa target respons awal untuk tiket P2?"
	$(UV) sop ask "$(Q)"

chat: ## Mode interaktif: model dimuat sekali per sesi
	$(UV) sop chat

test: ## Jalankan semua test (integrasi di-skip jika Qdrant mati)
	$(UV) pytest -q

test-unit: ## Test tanpa Qdrant dan model embedding
	$(UV) pytest -q -m "not integration"

calibrate: ## M4: hitung threshold abstain dari golden set
	$(UV) python eval/calibrate.py

eval: ## M4: evaluasi per tahap (hybrid), tulis eval/report.md
	$(UV) sop eval

eval-ablation: ## Buffer: dense-only dan baseline full-context, kolom tambahan di report
	$(UV) sop eval --mode dense
	$(UV) sop eval --baseline full-context

report: ## Bangun ulang eval/report.md dari eval/results/ tanpa memanggil LLM
	$(UV) sop eval --report-only

demo: ## M5: jalankan pertanyaan demo, tulis demo/demo_output.md
	$(UV) python demo/run_demo.py

requirements: ## Ekspor requirements.txt untuk pengguna pip
	uv export --no-hashes --no-dev --format requirements-txt -o requirements.txt
