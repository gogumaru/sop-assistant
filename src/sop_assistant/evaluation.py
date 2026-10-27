"""Evaluasi per tahap: ekstraksi, retrieval, jawaban, latensi.

Tiap konfigurasi (hybrid, dense, full_context) ditulis ke eval/results/<config>.json;
eval/report.md dibangun ulang dari semua hasil yang ada sehingga kolom perbandingan
muncul otomatis begitu ablation/baseline dijalankan.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import yaml

from sop_assistant.config import CALIBRATION_FILE, ROOT, settings
from sop_assistant.guards import mask_pii, normalize_numbers
from sop_assistant.metrics import char_error_rate
from sop_assistant.retrieve import Hit, Mode, RetrievalResult, retrieve

EVAL_DIR = ROOT / "eval"
GOLDEN_PATH = EVAL_DIR / "golden.yaml"
RESULTS_DIR = EVAL_DIR / "results"
REPORT_PATH = EVAL_DIR / "report.md"
TRANSCRIPTS = ROOT / "tests" / "fixtures" / "transcripts"

CONFIGS = ("hybrid", "dense", "full_context")
CONFIG_LABEL = {"hybrid": "hybrid", "dense": "dense (ablation)", "full_context": "full-context (baseline)"}
ALL_CHUNKS_REQUIRED = {"multi_hop", "ambiguous"}
CALIBRATION_FALLBACK_MARGIN = 0.02


@dataclass(frozen=True)
class GoldenItem:
    id: str
    category: str
    question: str
    expected_status: str
    expected_reason: str | None = None
    expected_chunks: list[str] = field(default_factory=list)
    must_include: list[str] = field(default_factory=list)
    must_not_include: list[str] = field(default_factory=list)

    @property
    def in_scope(self) -> bool:
        return self.expected_status != "not_in_sop"


def load_golden(path: Path = GOLDEN_PATH) -> list[GoldenItem]:
    return [GoldenItem(**item) for item in yaml.safe_load(path.read_text(encoding="utf-8"))]


def load_chunk_rows() -> list[dict]:
    lines = settings.chunks_path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


# --- Metrik murni (dites di tests/test_evaluation.py) ---


def retrieval_metrics(ranked: list[str], expected: list[str]) -> tuple[float, float]:
    """(recall, reciprocal rank) dari daftar chunk_id berperingkat terhadap expected_chunks."""
    recall = sum(chunk_id in ranked for chunk_id in expected) / len(expected)
    rr = next((1 / rank for rank, chunk_id in enumerate(ranked, 1) if chunk_id in expected), 0.0)
    return recall, rr


def _normalized(text: str) -> str:
    return " ".join(normalize_numbers(text).split())


def missing_phrases(answer: str, phrases: list[str]) -> list[str]:
    return [p for p in phrases if _normalized(p) not in _normalized(answer)]


def citation_ok(cited: list[str], expected: list[str], category: str) -> bool:
    found = [chunk_id for chunk_id in expected if chunk_id in cited]
    return len(found) == len(expected) if category in ALL_CHUNKS_REQUIRED else bool(found)


def choose_threshold(in_scope: list[float], out_of_scope: list[float]) -> tuple[float, str]:
    """Threshold abstain. Prioritas: tidak ada pertanyaan in-scope yang ikut abstain."""
    lowest_in = min(in_scope)
    below = [s for s in out_of_scope if s < lowest_in]
    if len(below) == len(out_of_scope):
        return (lowest_in + max(out_of_scope)) / 2, "tengah celah antara in-scope terendah dan out-of-scope tertinggi"
    if below:
        return (
            (lowest_in + max(below)) / 2,
            f"skor tumpang tindih: {len(out_of_scope) - len(below)} out-of-scope di atas in-scope terendah "
            "dan diserahkan ke LLM; threshold di tengah celah yang tersisa",
        )
    return (
        lowest_in - CALIBRATION_FALLBACK_MARGIN,
        "semua out-of-scope di atas in-scope terendah; threshold tepat di bawah in-scope terendah, "
        "penolakan diserahkan ke LLM",
    )


def percentile(values: list[float], q: float) -> float | None:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)] if ordered else None


# --- Menjalankan evaluasi ---


def full_context_retriever(payloads: list[dict]):
    """Baseline tanpa retrieval: seluruh korpus menjadi konteks."""

    def retriever(query: str, k: int | None = None, mode: Mode = Mode.hybrid, doc_ids=None) -> RetrievalResult:
        pool = [p for p in payloads if not doc_ids or p["doc_id"] in doc_ids]
        hits = [Hit(chunk_id=p["chunk_id"], score=1.0, payload=p) for p in pool]
        return RetrievalResult(query=query, mode=mode, hits=hits, dense_top1=1.0)

    return retriever


@dataclass(frozen=True)
class Verdict:
    status_ok: bool
    citation_ok: bool | None  # None = tidak berlaku (out-of-scope)
    missing: list[str]
    forbidden: list[str]
    notes: list[str]

    @property
    def passed(self) -> bool:
        return self.status_ok and self.citation_ok is not False and not self.missing and not self.forbidden


def judge(item: GoldenItem, result) -> Verdict:
    """Bandingkan AnswerResult dengan harapan golden. Dipakai eval dan demo."""
    cited = [s.chunk_id for s in result.sources]
    status_ok = result.status == item.expected_status and (
        item.expected_reason is None or result.review_reason == item.expected_reason
    )
    cite_ok = citation_ok(cited, item.expected_chunks, item.category) if item.expected_chunks else None
    missing = missing_phrases(result.answer, item.must_include)
    forbidden = [p for p in item.must_not_include if not missing_phrases(result.answer, [p])]

    notes = []
    if not status_ok:
        notes.append("status beda")
    if cite_ok is False:
        notes.append(f"sitasi kurang (harus: {', '.join(item.expected_chunks)})")
    if missing:
        notes.append(f"tidak memuat: {', '.join(missing)}")
    if forbidden:
        notes.append(f"memuat terlarang: {', '.join(forbidden)}")
    if result.abstained:
        notes.append("abstain sebelum LLM")
    if result.repaired:
        notes.append("1x perbaikan")
    if result.scale_gaps:
        notes.append(f"skala tidak terjawab: {', '.join(result.scale_gaps)}")
    return Verdict(status_ok=status_ok, citation_ok=cite_ok, missing=missing, forbidden=forbidden, notes=notes)


def evaluate_item(item: GoldenItem, engine, config: str, retriever, mode: Mode, threshold: float | None) -> dict:
    from sop_assistant.answer import answer_question

    recall = rr = None
    ranked: list[str] = []
    if config != "full_context" and item.in_scope:
        masked, _ = mask_pii(item.question)
        ranked = [h.chunk_id for h in retrieve(masked, k=settings.retrieve_k, mode=mode).hits]
        recall, rr = retrieval_metrics(ranked, item.expected_chunks)

    result = answer_question(item.question, engine, retriever=retriever, mode=mode, threshold=threshold)
    verdict = judge(item, result)
    context_recall = (
        sum(c in result.retrieved_ids for c in item.expected_chunks) / len(item.expected_chunks)
        if item.expected_chunks
        else None
    )

    return {
        "id": item.id,
        "category": item.category,
        "question": item.question,
        "in_scope": item.in_scope,
        "expected_status": item.expected_status,
        "expected_reason": item.expected_reason,
        "expected_chunks": item.expected_chunks,
        "status": result.status,
        "review_reason": result.review_reason,
        "answer": result.answer,
        "missing": result.missing,
        "escalate_to": result.escalate_to,
        "cited": [s.chunk_id for s in result.sources],
        "sources": [s.label() for s in result.sources],
        "retrieved_top_k": ranked,
        "retrieval_recall": recall,
        "retrieval_rr": rr,
        "context_recall": context_recall,
        "status_ok": verdict.status_ok,
        "citation_ok": verdict.citation_ok,
        "must_include_missing": verdict.missing,
        "must_not_include_found": verdict.forbidden,
        "passed": verdict.passed,
        "abstained": result.abstained,
        "repaired": result.repaired,
        "repair_reasons": result.repair_reasons,
        "guard_errors": result.guard_errors,
        "guard_failed": result.review_reason == "guard_failed",
        "escalation_by_rule": result.escalation_by_rule,
        "scale_gaps": result.scale_gaps,
        "dense_top1": result.dense_top1,
        "latency_ms": result.timings_ms,
        "notes": verdict.notes,
    }


def run(config: str) -> dict:
    from sop_assistant.embedding import get_embedder
    from sop_assistant.engines import get_engine

    if config not in CONFIGS:
        raise ValueError(f"Konfigurasi tidak dikenal: {config}")
    items = load_golden()

    start = time.perf_counter()
    get_embedder().encode(["pemanasan"], return_dense=True, return_sparse=True)
    load_s = time.perf_counter() - start

    engine = get_engine()
    if config == "full_context":
        retriever, mode, threshold = full_context_retriever(load_chunk_rows()), Mode.hybrid, None
    else:
        retriever, mode, threshold = retrieve, Mode(config), settings.abstain_threshold

    records = []
    for item in items:
        print(f"[{config}] {item.id}: {item.question}", flush=True)
        records.append(evaluate_item(item, engine, config, retriever, mode, threshold))

    data = {
        "config": config,
        "engine": engine.name,
        "threshold": threshold,
        "model_load_s": load_s,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "records": records,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{config}.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data


# --- Ringkasan dan laporan ---


def _frac(count: int, total: int) -> str:
    return f"{count}/{total} ({count / total:.0%})" if total else "-"


def _mean(values: list[float | None]) -> str:
    present = [v for v in values if v is not None]
    return f"{sum(present) / len(present):.2f}" if present else "-"


def summarize(data: dict) -> dict[str, str]:
    records = data["records"]
    in_scope = [r for r in records if r["in_scope"]]
    out_scope = [r for r in records if not r["in_scope"]]
    cited = [r for r in records if r["citation_ok"] is not None]
    with_includes = [r for r in records if r["expected_status"] != "not_in_sop"]
    llm_called = [r for r in records if not r["abstained"]]
    totals = [r["latency_ms"]["total"] for r in records]
    p50, p95 = percentile(totals, 0.5), percentile(totals, 0.95)
    return {
        "Lolos keseluruhan": _frac(sum(r["passed"] for r in records), len(records)),
        "Akurasi status": _frac(sum(r["status_ok"] for r in records), len(records)),
        "Ketepatan sitasi (in-scope)": _frac(sum(bool(r["citation_ok"]) for r in cited), len(cited)),
        "must_include terpenuhi": _frac(sum(not r["must_include_missing"] for r in with_includes), len(with_includes)),
        "Recall@5 retrieval (in-scope)": _mean([r["retrieval_recall"] for r in in_scope]),
        "MRR retrieval (in-scope)": _mean([r["retrieval_rr"] for r in in_scope]),
        "Recall konteks akhir (setelah glossary)": _mean([r["context_recall"] for r in in_scope]),
        "Guard lolos percobaan pertama": _frac(sum(not r["repaired"] for r in llm_called), len(llm_called)),
        "Guard gagal setelah perbaikan": str(sum(r["guard_failed"] for r in records)),
        "Out-of-scope ditolak threshold": _frac(sum(r["abstained"] for r in out_scope), len(out_scope)),
        "In-scope salah abstain": str(sum(r["abstained"] for r in in_scope)),
        "Latensi total p50 / p95": f"{p50 / 1000:.1f} / {p95 / 1000:.1f} dtk" if totals else "-",
        "Muat model (sekali)": f"{data['model_load_s']:.1f} dtk",
        "Engine": data["engine"],
    }


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _status(status: str, reason: str | None) -> str:
    return f"{status}/{reason}" if reason else status


def extraction_section() -> list[str]:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_extract.py", "-q", "-p", "no:cacheprovider"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    summary = next((line for line in reversed(proc.stdout.splitlines()) if line.strip()), "tidak ada output")
    rows = load_chunk_rows()
    lines = [f"- pytest `tests/test_extract.py`: {summary}", f"- Jumlah chunk: {len(rows)}"]
    for doc_id in ("KB_01", "KB_04"):
        doc = [r for r in rows if r["doc_id"] == doc_id]
        hypothesis = "\n".join(
            [doc[0]["doc_title"]] + [f"{r['section_no']}. {r['section_title']}\n{r['text']}" for r in doc]
        )
        reference = (TRANSCRIPTS / f"{doc_id}.txt").read_text(encoding="utf-8")
        lines.append(f"- CER OCR {doc_id} terhadap transkripsi manual: {char_error_rate(hypothesis, reference):.2%}")
    return lines


def calibration_section() -> list[str]:
    if not CALIBRATION_FILE.exists():
        return ["Belum dikalibrasi (jalankan `make calibrate`). Router abstain nonaktif."]
    cal = json.loads(CALIBRATION_FILE.read_text(encoding="utf-8"))
    return [
        f"- Threshold cosine dense top-1: **{cal['abstain_threshold']:.3f}**",
        f"- Metode: {cal['method']}",
        f"- In-scope terendah: {cal['lowest_in_scope']:.3f}; out-of-scope tertinggi: {cal['highest_out_of_scope']:.3f}",
        f"- {cal['warning']}",
    ]


def write_report() -> Path:
    results = {
        c: json.loads((RESULTS_DIR / f"{c}.json").read_text(encoding="utf-8"))
        for c in CONFIGS
        if (RESULTS_DIR / f"{c}.json").exists()
    }
    if not results:
        raise SystemExit("Belum ada hasil di eval/results/. Jalankan `sop eval` dulu.")
    primary = results.get("hybrid") or next(iter(results.values()))
    summaries = {c: summarize(d) for c, d in results.items()}

    lines = [
        "# Laporan Evaluasi",
        "",
        f"Di-generate oleh `sop eval` pada {datetime.now().isoformat(timespec='seconds')}. "
        f"Golden set: {len(primary['records'])} pertanyaan (`eval/golden.yaml`). "
        "Hasil mentah per konfigurasi ada di `eval/results/`.",
        "",
        "## Ringkasan metrik",
        "",
        "| Metrik | " + " | ".join(CONFIG_LABEL[c] for c in results) + " |",
        "|---|" + "---|" * len(results),
    ]
    for metric in summaries[next(iter(results))]:
        lines.append(f"| {metric} | " + " | ".join(_cell(summaries[c][metric]) for c in results) + " |")

    lines += ["", "## Ekstraksi", "", *extraction_section(), "", "## Kalibrasi abstain", "", *calibration_section()]

    lines += [
        "",
        f"## Per pertanyaan ({CONFIG_LABEL[primary['config']]})",
        "",
        "| ID | Kategori | Pertanyaan | Expected | Aktual | Sitasi | Lolos | Catatan |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in primary["records"]:
        lines.append(
            f"| {r['id']} | {r['category']} | {_cell(r['question'])} "
            f"| {_status(r['expected_status'], r['expected_reason'])} | {_status(r['status'], r['review_reason'])} "
            f"| {', '.join(r['cited']) or '-'} | {'ya' if r['passed'] else 'tidak'} | {_cell('; '.join(r['notes']))} |"
        )

    if len(results) > 1:
        lines += ["", "## Perbandingan per pertanyaan", "", "| ID | " + " | ".join(CONFIG_LABEL[c] for c in results) + " |"]
        lines.append("|---|" + "---|" * len(results))
        by_config = {c: {r["id"]: r for r in d["records"]} for c, d in results.items()}
        for r in primary["records"]:
            cells = []
            for c in results:
                other = by_config[c][r["id"]]
                cells.append("lolos" if other["passed"] else f"gagal ({_status(other['status'], other['review_reason'])})")
            lines.append(f"| {r['id']} | " + " | ".join(cells) + " |")

    lines += ["", f"## Jawaban lengkap ({CONFIG_LABEL[primary['config']]})", ""]
    for r in primary["records"]:
        lines += [f"### {r['id']}: {r['question']}", "", f"Status: `{_status(r['status'], r['review_reason'])}`", ""]
        lines += [f"> {r['answer']}", ""]
        if r["missing"]:
            lines += [f"Tidak diatur SOP: {r['missing']}", ""]
        if r["escalate_to"]:
            lines += [f"Eskalasi ke: {r['escalate_to']}", ""]
        lines += [f"- {source}" for source in r["sources"]] or ["- (tanpa sumber)"]
        lines.append("")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    return REPORT_PATH
