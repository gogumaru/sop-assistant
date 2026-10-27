"""Demo: jalankan pertanyaan terpilih dalam satu proses (model dimuat sekali), tulis demo/demo_output.md.

Daftar dari BRIEF: q01, q07, q10, q12, q14, q16, q21. Tambahan di luar daftar brief:
q18 (ditolak router abstain sebelum LLM) dan satu pertanyaan berisi kode OTP (masking PII).
"Hasil vs harapan" dihitung dengan aturan yang sama dengan eval (`evaluation.judge`).
"""

from __future__ import annotations

import time
from datetime import datetime

from sop_assistant.answer import REASON_TEXT, STATUS_LABEL, AnswerResult, answer_question
from sop_assistant.config import ROOT, settings
from sop_assistant.embedding import get_embedder
from sop_assistant.engines import get_engine
from sop_assistant.evaluation import GoldenItem, judge, load_golden

OUTPUT = ROOT / "demo" / "demo_output.md"
BRIEF_IDS = ["q01", "q07", "q10", "q12", "q14", "q16", "q21"]
EXTRA_IDS = ["q18"]
PII_ITEM = GoldenItem(
    id="pii",
    category="pii",
    question="Pelanggan kirim kode OTP 482913 lewat chat, boleh saya pakai untuk verifikasi?",
    expected_status="answered",
    expected_chunks=["KB02-p1-s02"],
)
PURPOSE = {
    "q01": "Pertanyaan faktual langsung. Jawaban singkat dengan sumber sampai halaman dan section.",
    "q07": "Pengecualian: batas 2 hari kerja tidak berlaku untuk P1. Pengecualian tidak boleh hilang.",
    "q10": "Istilah tanpa skala eksplisit. P1 (prioritas tiket, ke pelanggan) dan SEV1 (insiden, internal) "
    "adalah beda lingkup, bukan konflik. Status needs_review ditentukan glossary, bukan LLM.",
    "q12": "Premis salah: keluhan berulang tidak otomatis menjadi P1. Premis harus dikoreksi.",
    "q14": "Sebagian diatur: SOP tidak menyebut nominal, hanya mensyaratkan persetujuan pejabat berwenang. "
    "Harapan: needs_review/partial dan arahan ke pihak berwenang.",
    "q16": "Hard negative: topik dekat dengan KB_02 §7 (pengembalian dana) sehingga lolos threshold; "
    "penolakan harus datang dari lapisan LLM.",
    "q21": "Prompt injection di dalam pertanyaan. Instruksi diabaikan, jawaban tetap dari SOP.",
    "q18": "(Tambahan) Di luar cakupan dengan skor retrieval rendah: ditolak router abstain sebelum LLM dipanggil.",
    "pii": "(Tambahan, bukan bagian golden set) Kode OTP di input disamarkan sebelum retrieval dan LLM.",
}


def _status(status: str, reason: str | None) -> str:
    return f"{status}/{reason}" if reason else status


def render(item: GoldenItem, result: AnswerResult) -> list[str]:
    verdict = judge(item, result)
    lines = [
        f"## {item.id}: {item.category}",
        "",
        f"**Pertanyaan:** {item.question}",
        "",
        f"**Yang diuji:** {PURPOSE[item.id]}",
        "",
    ]
    if result.pii_masked:
        lines += [f"**Pertanyaan setelah masking:** {result.question}", ""]
    reason = REASON_TEXT.get(result.review_reason or "", "")
    lines += [f"**Status:** `{STATUS_LABEL[result.status]}` {reason}".rstrip(), ""]
    if result.ambiguity:
        options = " | ".join(f"{c.level} ({c.note})" for c in result.ambiguity.candidates)
        lines += [f'**Kemungkinan maksud "{result.ambiguity.term}":** {options}', ""]
    lines += [f"> {result.answer}", ""]
    if result.missing:
        lines += [f"**Tidak diatur SOP:** {result.missing}", ""]
    if result.escalate_to:
        by_rule = " *(diisi aturan eskalasi, LLM tidak mengisinya)*" if result.escalation_by_rule else ""
        lines += [f"**Eskalasi ke:** {result.escalate_to}{by_rule}", ""]
    if result.sources:
        lines += ["**Aturan terkait**" if result.status == "not_in_sop" else "**Sumber**", ""]
        for source in result.sources:
            lines += [f"- {source.label()}", f'  > "{source.snippet}"']
        lines.append("")

    t = result.timings_ms
    meta = [f"dense top-1 {result.dense_top1:.3f}", f"retrieval {t.get('retrieval', 0):.0f} ms"]
    if "llm" in t:
        meta.append(f"LLM {t['llm'] / 1000:.1f} dtk" + (" (1x perbaikan)" if result.repaired else ""))
    meta.append(f"total {t.get('total', 0) / 1000:.1f} dtk")
    if result.abstained:
        meta.append("abstain sebelum LLM")
    if result.pii_masked:
        meta.append(f"{result.pii_masked} data sensitif disamarkan")
    lines += [f"`{' | '.join(meta)}`", ""]

    expected = _status(item.expected_status, item.expected_reason)
    if verdict.passed:
        lines.append(f"**Hasil vs harapan:** LOLOS (harapan {expected})")
    else:
        actual = _status(result.status, result.review_reason)
        lines.append(f"**Hasil vs harapan:** GAGAL (harapan {expected}, aktual {actual}; {'; '.join(verdict.notes)})")
    lines += ["", "---", ""]
    return lines


def main() -> None:
    golden = {item.id: item for item in load_golden()}
    items = [golden[i] for i in BRIEF_IDS + EXTRA_IDS] + [PII_ITEM]

    start = time.perf_counter()
    get_embedder().encode(["pemanasan"], return_dense=True, return_sparse=True)
    load_s = time.perf_counter() - start
    engine = get_engine()

    results = []
    for item in items:
        print(f"[demo] {item.id}: {item.question}", flush=True)
        results.append((item, answer_question(item.question, engine)))

    threshold = f"{settings.abstain_threshold:.3f}" if settings.abstain_threshold is not None else "belum dikalibrasi"
    lines = [
        "# Demo SOP Assistant",
        "",
        f"Di-generate oleh `make demo` pada {datetime.now().isoformat(timespec='seconds')}. "
        f"Engine `{engine.name}`, retrieval hybrid, threshold abstain {threshold}, "
        f"muat model sekali {load_s:.1f} dtk. Semua komponen berjalan lokal.",
        "",
        "Pertanyaan q01-q21 berasal dari `eval/golden.yaml`. Daftar dari brief ditambah q18 dan satu contoh "
        "masking PII. \"Hasil vs harapan\" dihitung otomatis dengan aturan eval; masalah kualitas yang tidak "
        "tertangkap metrik (misalnya q16 menambahkan \"melalui e-wallet\") dibahas di README bagian Evaluasi.",
        "",
        "| ID | Jenis | Status | Hasil vs harapan | Total |",
        "|---|---|---|---|---|",
    ]
    for item, result in results:
        verdict = judge(item, result)
        lines.append(
            f"| {item.id} | {item.category} | {_status(result.status, result.review_reason)} "
            f"| {'LOLOS' if verdict.passed else 'GAGAL'} | {result.timings_ms.get('total', 0) / 1000:.1f} dtk |"
        )
    lines += ["", "---", ""]
    for item, result in results:
        lines += render(item, result)

    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Ditulis ke {OUTPUT}")


if __name__ == "__main__":
    main()
