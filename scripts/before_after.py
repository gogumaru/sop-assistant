"""Bahan README: urutan baca sebelum vs sesudah penanganan kolom.

Menulis docs/two_column_before_after.md dari PDF asli, bukan contoh rekaan.
"""

from __future__ import annotations

import pymupdf

from sop_assistant.config import ROOT, settings
from sop_assistant.extract import Line, extract_pdf, page_lines

OUTPUT = ROOT / "docs" / "two_column_before_after.md"
ROW_TOLERANCE_PT = 4.0
PREVIEW_ROWS = 10
SNIPPET_CHARS = 70


def naive_rows(lines: list[Line]) -> list[str]:
    """Baca per baris visual dari kiri ke kanan, seperti OCR/ekstraksi tanpa deteksi kolom."""
    rows: list[list[Line]] = []
    for line in sorted(lines, key=lambda line: line.y0):
        if rows and abs(line.y0 - rows[-1][0].y0) <= ROW_TOLERANCE_PT:
            rows[-1].append(line)
        else:
            rows.append([line])
    return [" ".join(line.text for line in sorted(row, key=lambda line: line.x0)) for row in rows]


def after_summary(doc_prefix: str) -> str:
    path = next(settings.pdf_dir.glob(f"{doc_prefix}_*.pdf"))
    chunks = extract_pdf(path)
    rows = [f"doc_title: {chunks[0].doc_title}"]
    rows += [f"{c.section_no}. {c.section_title}: {c.text[:SNIPPET_CHARS]}..." for c in chunks]
    return "\n".join(rows)


def code(text: str) -> str:
    return f"```text\n{text}\n```"


def main() -> None:
    kb01 = next(settings.pdf_dir.glob("KB_01_*.pdf"))
    kb03 = next(settings.pdf_dir.glob("KB_03_*.pdf"))
    with pymupdf.open(kb01) as doc:
        raw_kb01 = doc[0].get_text()
        ocr_kb01, _ = page_lines(doc[0])
    with pymupdf.open(kb03) as doc:
        raw_kb03 = doc[0].get_text()

    parts = [
        "# Penanganan dua kolom: sebelum vs sesudah",
        "Di-generate oleh `make before-after` dari PDF asli.",
        "## KB_01: tanpa text layer, dua kolom (kiri 1-6, kanan 7-8)",
        f"**1. `page.get_text()` mentah** ({len(raw_kb01.strip())} karakter):",
        code(raw_kb01.strip() or "(kosong: halaman hanya berisi gambar)"),
        f"**2. OCR dibaca per baris visual, kiri ke kanan** ({PREVIEW_ROWS} baris pertama). "
        "Section 1 dan 7 tercampur dalam satu baris:",
        code("\n".join(naive_rows(ocr_kb01)[:PREVIEW_ROWS])),
        "**3. Sesudah urutan kolom dan potong per section:**",
        code(after_summary("KB_01")),
        "## KB_03: ada text layer, dua kolom (kiri 1-6, kanan 7-8)",
        "**1. `page.get_text()` mentah** (6 baris pertama). Nomor halaman `1` menempel tepat setelah judul:",
        code("\n".join(raw_kb03.strip().splitlines()[:6])),
        "**2. Sesudah:**",
        code(after_summary("KB_03")),
    ]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n\n".join(parts) + "\n", encoding="utf-8")
    print(f"Ditulis ke {OUTPUT}")


if __name__ == "__main__":
    main()
