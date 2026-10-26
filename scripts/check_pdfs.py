"""M0: cek text layer tiap PDF SOP.

Halaman dengan teks lebih pendek dari OCR_MIN_CHARS dianggap tidak punya text layer
dan akan diproses lewat OCR oleh extract.py. Catatan: teks yang bisa di-select di
Preview/Safari belum tentu text layer; macOS Live Text meng-OCR gambar saat dibuka.
"""

from __future__ import annotations

import sys

import pymupdf
from rich.console import Console
from rich.table import Table

from sop_assistant.config import settings


def main() -> int:
    pdfs = sorted(settings.pdf_dir.glob(settings.pdf_glob))
    if not pdfs:
        print(f"Tidak ada file {settings.pdf_glob} di {settings.pdf_dir}")
        return 1

    table = Table(title="Cek text layer PDF SOP")
    for column in ("File", "Hal.", "Karakter", "Blok teks", "Gambar", "DPI gambar (perkiraan)", "Status"):
        table.add_column(column)

    for path in pdfs:
        with pymupdf.open(path) as doc:
            for page in doc:
                chars = len(page.get_text().strip())
                text_blocks = sum(1 for block in page.get_text("blocks") if block[6] == 0)
                images = page.get_images(full=True)
                # Perkiraan resolusi efektif: lebar piksel gambar dibagi lebar halaman dalam inci.
                dpi = f"{images[0][2] / (page.rect.width / 72):.0f}" if images else "-"
                status = "text layer" if chars >= settings.ocr_min_chars else "[bold red]PERLU OCR[/]"
                table.add_row(
                    path.name, str(page.number + 1), str(chars), str(text_blocks), str(len(images)), dpi, status
                )

    Console().print(table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
