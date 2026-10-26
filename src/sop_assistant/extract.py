"""Ekstraksi PDF SOP menjadi satu chunk per section bernomor.

Alur per halaman:
1. Ambil baris beserta koordinat. Halaman tanpa text layer (teks < OCR_MIN_CHARS)
   dirender lalu di-OCR dengan Tesseract; koordinat piksel dikonversi ke point PDF
   sehingga text layer dan OCR memakai satu representasi `Line`.
2. Susun urutan baca dengan `order_lines` (pita horizontal + kolom).
3. Buang nomor halaman dan noise, potong per heading `N. Judul`, gabungkan baris
   yang terpotong hard-wrap menjadi satu paragraf.

Jalankan `python -m sop_assistant.extract` untuk menulis data/chunks.jsonl.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pymupdf
import pytesseract
from PIL import Image

from sop_assistant.config import settings

# Heading: "7. Penutupan". Judul wajib diawali huruf supaya "09.00-17.00" tidak terbaca heading.
HEADING_RE = re.compile(r"^\s*(\d{1,2})\.\s*([A-Za-z].*)$")
HEADING_MAX_CHARS = 60
DOC_ID_RE = re.compile(r"^KB_(\d+)")
# Nomor halaman hanya dibuang jika berada di margin atas/bawah, bukan angka di tengah isi.
PAGE_NUMBER_MARGIN = 0.08
# Tesseract membaca "SEV1" sebagai "SEVI". Token level punya format pasti (SEV1-4, P1-4),
# jadi I/l/| setelah SEV atau P dikoreksi ke 1. Hanya untuk teks OCR.
OCR_LEVEL_ONE_RE = re.compile(r"\b(SEV|P)[Il|]\b")


@dataclass(frozen=True)
class Line:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    page: int


@dataclass
class Section:
    no: int
    title: str
    heading: Line
    body: list[Line] = field(default_factory=list)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    source_file: str
    page: int
    section_no: int
    section_title: str
    text: str
    embed_text: str
    bbox: list[float]
    ocr_used: bool
    content_hash: str


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()


def fix_ocr_level_tokens(text: str) -> str:
    return OCR_LEVEL_ONE_RE.sub(r"\g<1>1", text)


def native_lines(page: pymupdf.Page) -> list[Line]:
    lines = []
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:  # 1 = blok gambar
            continue
        for line in block["lines"]:
            text = _clean("".join(span["text"] for span in line["spans"]))
            if text:
                lines.append(Line(text, *line["bbox"], page=page.number + 1))
    return lines


def ocr_lines(page: pymupdf.Page, dpi: int, lang: str) -> list[Line]:
    """OCR satu halaman; kata dikelompokkan per baris Tesseract (block, par, line)."""
    pixmap = page.get_pixmap(dpi=dpi)
    image = Image.open(io.BytesIO(pixmap.tobytes("png")))
    data = pytesseract.image_to_data(image, lang=lang, config="--psm 3", output_type=pytesseract.Output.DICT)

    groups: dict[tuple[int, int, int], list[int]] = {}
    for i, word in enumerate(data["text"]):
        if data["level"][i] == 5 and word.strip():  # level 5 = kata
            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            groups.setdefault(key, []).append(i)

    scale = 72 / dpi
    lines = []
    for idx in groups.values():
        idx.sort(key=lambda i: data["left"][i])
        text = fix_ocr_level_tokens(_clean(" ".join(data["text"][i] for i in idx)))
        lines.append(
            Line(
                text,
                min(data["left"][i] for i in idx) * scale,
                min(data["top"][i] for i in idx) * scale,
                max(data["left"][i] + data["width"][i] for i in idx) * scale,
                max(data["top"][i] + data["height"][i] for i in idx) * scale,
                page=page.number + 1,
            )
        )
    return lines


def page_lines(page: pymupdf.Page) -> tuple[list[Line], bool]:
    """Baris mentah (belum diurutkan) dan apakah OCR dipakai."""
    if len(page.get_text().strip()) >= settings.ocr_min_chars:
        return native_lines(page), False
    return ocr_lines(page, settings.ocr_dpi, settings.ocr_lang), True


def order_lines(lines: list[Line], page_width: float) -> list[Line]:
    """Urutan baca berbasis koordinat, tidak bergantung pada urutan stream PDF.

    Baris yang melintasi garis tengah halaman (x0 < tengah < x1) adalah teks selebar
    halaman dan menjadi pemisah pita. Di dalam tiap pita, kolom kiri dibaca dari atas
    ke bawah, lalu kolom kanan. Aturan "blok > 60% lebar = full-width, taruh di atas"
    sengaja tidak dipakai: di dokumen satu kolom (KB_02, KB_04) isi section lebar
    tetapi heading-nya sempit, sehingga semua isi akan terbaca sebelum semua heading.
    """
    mid = page_width / 2
    ordered: list[Line] = []
    band: list[Line] = []

    def flush() -> None:
        left = [line for line in band if line.x0 < mid]
        right = [line for line in band if line.x0 >= mid]
        for column in (left, right):
            ordered.extend(sorted(column, key=lambda line: (line.y0, line.x0)))
        band.clear()

    for line in sorted(lines, key=lambda line: (line.y0, line.x0)):
        if line.x0 < mid < line.x1:
            flush()
            ordered.append(line)
        else:
            band.append(line)
    flush()
    return ordered


def is_noise(line: Line, page_height: float) -> bool:
    if not any(ch.isalnum() for ch in line.text):  # garis, bullet, artefak OCR
        return True
    in_margin = line.y0 > page_height * (1 - PAGE_NUMBER_MARGIN) or line.y1 < page_height * PAGE_NUMBER_MARGIN
    return line.text.isdigit() and in_margin


def join_lines(texts: list[str]) -> str:
    """Gabungkan baris hard-wrap. Tanda hubung di akhir baris dipertahankan (masing-masing)."""
    out = ""
    for text in texts:
        if not out:
            out = text
        elif out.endswith("-") and text[:1].islower():
            out += text
        else:
            out += " " + text
    return out


def split_sections(lines: list[Line]) -> tuple[list[Line], list[Section]]:
    """Pisahkan preamble (judul dokumen) dan section.

    Heading hanya diterima jika nomornya tepat nomor sebelumnya + 1. Ini menahan noise
    OCR dan baris isi yang kebetulan diawali angka.
    """
    preamble: list[Line] = []
    sections: list[Section] = []
    for line in lines:
        match = HEADING_RE.match(line.text)
        is_heading = (
            match is not None
            and int(match.group(1)) == len(sections) + 1
            and len(line.text) <= HEADING_MAX_CHARS
            and not line.text.endswith(".")
        )
        if is_heading:
            sections.append(Section(no=len(sections) + 1, title=match.group(2).strip(), heading=line))
        elif sections:
            sections[-1].body.append(line)
        else:
            preamble.append(line)
    return preamble, sections


def extract_pdf(path: Path) -> list[Chunk]:
    match = DOC_ID_RE.match(path.name)
    if not match:
        raise ValueError(f"Nama file tidak berpola KB_XX: {path.name}")
    doc_id, short_id = f"KB_{match.group(1)}", f"KB{match.group(1)}"

    lines: list[Line] = []
    ocr_pages: set[int] = set()
    with pymupdf.open(path) as doc:
        for page in doc:
            raw, ocr_used = page_lines(page)
            if ocr_used:
                ocr_pages.add(page.number + 1)
            ordered = order_lines(raw, page.rect.width)
            lines.extend(line for line in ordered if not is_noise(line, page.rect.height))

    preamble, sections = split_sections(lines)
    if not preamble or not sections:
        raise ValueError(f"{path.name}: judul dokumen atau section tidak ditemukan")
    doc_title = join_lines([line.text for line in preamble])

    chunks = []
    for section in sections:
        text = join_lines([line.text for line in section.body])
        embed_text = f"[{doc_title} > {section.no}. {section.title}] {text}"
        members = [section.heading, *section.body]
        page = section.heading.page
        chunks.append(
            Chunk(
                chunk_id=f"{short_id}-p{page}-s{section.no:02d}",
                doc_id=doc_id,
                doc_title=doc_title,
                source_file=path.name,
                page=page,
                section_no=section.no,
                section_title=section.title,
                text=text,
                embed_text=embed_text,
                bbox=[
                    round(min(line.x0 for line in members), 1),
                    round(min(line.y0 for line in members), 1),
                    round(max(line.x1 for line in members), 1),
                    round(max(line.y1 for line in members), 1),
                ],
                ocr_used=page in ocr_pages,
                content_hash=hashlib.sha256(embed_text.encode()).hexdigest(),
            )
        )
    return chunks


def extract_corpus(pdf_dir: Path | None = None) -> list[Chunk]:
    pdf_dir = pdf_dir or settings.pdf_dir
    return [chunk for path in sorted(pdf_dir.glob(settings.pdf_glob)) for chunk in extract_pdf(path)]


def write_chunks(chunks: list[Chunk], path: Path | None = None) -> Path:
    path = path or settings.chunks_path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
    return path


def main() -> None:
    chunks = extract_corpus()
    path = write_chunks(chunks)
    for doc_id in sorted({c.doc_id for c in chunks}):
        doc_chunks = [c for c in chunks if c.doc_id == doc_id]
        sections = [c.section_no for c in doc_chunks]
        ocr = "OCR" if any(c.ocr_used for c in doc_chunks) else "text layer"
        print(f"{doc_id} | {doc_chunks[0].doc_title} | {ocr} | section {sections}")
    print(f"{len(chunks)} chunk ditulis ke {path}")


if __name__ == "__main__":
    main()
