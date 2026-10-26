"""Guard deterministik: masking PII, abstain, glossary ambiguitas, citation guard.

Tidak ada LLM di modul ini. Keputusan yang menentukan keamanan jawaban dibuat oleh
aturan yang bisa diuji, bukan oleh model kecil.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from sop_assistant.config import ROOT

# --- PII mask (pada input, sebelum retrieval dan LLM) ---

MASK = "[DISAMARKAN]"
CARD_RE = re.compile(r"(?<!\d)\d(?:[ -]?\d){12,18}(?!\d)")  # 13-19 digit, boleh berspasi/strip
OTP_CONTEXT_RE = re.compile(r"\b(otp|pin|kode verifikasi)\b", re.IGNORECASE)
SIX_DIGIT_RE = re.compile(r"(?<!\d)\d{6}(?!\d)")


def mask_pii(text: str) -> tuple[str, int]:
    """Samarkan nomor kartu, dan angka 6 digit jika ada konteks OTP/PIN. Kembalikan (teks, jumlah)."""
    masked, n_card = CARD_RE.subn(MASK, text)
    n_otp = 0
    if OTP_CONTEXT_RE.search(masked):
        masked, n_otp = SIX_DIGIT_RE.subn(MASK, masked)
    return masked, n_card + n_otp


# --- Abstain router ---


def should_abstain(dense_top1: float, threshold: float | None) -> bool:
    """Threshold None (belum dikalibrasi) = router nonaktif; keputusan diserahkan ke LLM + guard."""
    return threshold is not None and dense_top1 < threshold


# --- Glossary ambiguitas ---

GLOSSARY_PATH = ROOT / "glossary.yaml"


@dataclass(frozen=True)
class Candidate:
    scale: str
    token: str  # mis. "P1" atau "SEV1"
    level: str
    note: str

    def token_pattern(self) -> re.Pattern[str]:
        """Token SEV1 cocok dengan "SEV1", "sev1", dan "SEV 1"."""
        prefix = self.token.rstrip("0123456789")
        return re.compile(rf"\b{re.escape(prefix)}\s?{self.token[len(prefix):]}\b", re.IGNORECASE)


@dataclass(frozen=True)
class Scale:
    id: str
    name: str
    doc_ids: tuple[str, ...]


@dataclass(frozen=True)
class Ambiguity:
    term: str
    candidates: tuple[Candidate, ...]
    scales: tuple[Scale, ...]


@dataclass(frozen=True)
class Glossary:
    explicit_level: re.Pattern[str]
    entries: tuple[tuple[tuple[str, ...], tuple[Candidate, ...]], ...]
    scales: dict[str, Scale]


@lru_cache(maxsize=4)
def load_glossary(path: Path = GLOSSARY_PATH) -> Glossary:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    scales = {sid: Scale(sid, s["name"], tuple(s["doc_ids"])) for sid, s in data["scales"].items()}
    entries = tuple(
        (tuple(term.lower() for term in entry["terms"]), tuple(Candidate(**c) for c in entry["candidates"]))
        for entry in data["ambiguous_terms"]
    )
    return Glossary(re.compile(data["explicit_level_pattern"], re.IGNORECASE), entries, scales)


def detect_ambiguity(question: str, glossary: Glossary | None = None) -> Ambiguity | None:
    glossary = glossary or load_glossary()
    text = question.lower()
    if glossary.explicit_level.search(text):
        return None
    for terms, candidates in glossary.entries:
        for term in sorted(terms, key=len, reverse=True):
            if re.search(rf"\b{re.escape(term)}\b", text):
                scales = tuple(dict.fromkeys(glossary.scales[c.scale] for c in candidates))
                return Ambiguity(term=term, candidates=candidates, scales=scales)
    return None


def uncovered_scales(claims: list[dict], doc_of: dict[str, str], ambiguity: Ambiguity) -> list[Scale]:
    """Skala kandidat yang tidak punya klaim yang menyebut token levelnya DAN menyitasi dokumennya.

    Prompt meminta jawaban per skala, tetapi model kecil sering hanya menjawab satu.
    Cek per dokumen saja terlalu longgar: model memenuhinya dengan menyitasi section yang
    kebetulan memuat "gangguan kritis" (KB_03 §1) tanpa menjawab P1.
    """

    def covered(scale: Scale) -> bool:
        patterns = [c.token_pattern() for c in ambiguity.candidates if c.scale == scale.id]
        for claim in claims:
            cites_scale = any(doc_of.get(c) in scale.doc_ids for c in claim.get("chunk_ids") or [])
            if cites_scale and any(p.search(claim.get("text", "")) for p in patterns):
                return True
        return False

    return [scale for scale in ambiguity.scales if not covered(scale)]


# --- Citation guard (setelah LLM) ---

WORD_NUMBERS = {
    "satu": "1", "dua": "2", "tiga": "3", "empat": "4", "lima": "5",
    "enam": "6", "tujuh": "7", "delapan": "8", "sembilan": "9", "sepuluh": "10",
}  # fmt: skip
WORD_NUMBER_RE = re.compile(r"\b(" + "|".join(WORD_NUMBERS) + r")\b")
# "jam kerja" sebelum "jam", "hari kerja" sebelum "hari": "4 jam" tidak boleh lolos sebagai "4 jam kerja".
QUANTITY_RE = re.compile(r"\b(\d+)\s*(menit|jam kerja|jam|hari kerja|hari|minggu|bulan)\b")
CLOCK_RE = re.compile(r"\b(\d{1,2})[.:](\d{2})\b")


def normalize_numbers(text: str) -> str:
    return WORD_NUMBER_RE.sub(lambda m: WORD_NUMBERS[m.group(1)], text.lower())


def extract_quantities(text: str) -> set[str]:
    """Angka + satuan dan jam, dinormalisasi: "dua hari kerja" -> "2 hari kerja", "09.00" -> "pukul 9.00"."""
    normalized = normalize_numbers(text)
    found = {f"{int(n)} {unit}" for n, unit in QUANTITY_RE.findall(normalized)}
    found |= {f"pukul {int(h)}.{m}" for h, m in CLOCK_RE.findall(normalized)}
    return found


@dataclass(frozen=True)
class GuardResult:
    ok: bool
    errors: list[str]
    valid_claims: list[dict]


def check_citations(output: dict, context: dict[str, str]) -> GuardResult:
    """Cek output LLM terhadap teks chunk di konteks (chunk_id -> teks).

    - Semua chunk_ids yang disitasi harus ada di konteks.
    - Status answered wajib punya klaim.
    - Setiap angka+satuan di klaim harus ada di chunk yang disitasi klaim itu.
    - Setiap angka+satuan di `answer` harus ada di salah satu chunk yang disitasi.
    Keterbatasan: hanya mengecek angka ADA di sumber, bukan bahwa angka itu milik level
    yang benar (misalnya "4 jam kerja" untuk P2, bukan P3).
    """
    errors: list[str] = []
    valid: list[dict] = []
    claims = output.get("claims") or []
    if output.get("status") == "answered" and not claims:
        errors.append("Status answered tetapi tidak ada klaim bersitasi.")

    for i, claim in enumerate(claims, 1):
        ids = claim.get("chunk_ids") or []
        known = [c for c in ids if c in context]
        claim_errors = []
        if not ids:
            claim_errors.append(f"Klaim {i} tidak punya chunk_ids.")
        if unknown := [c for c in ids if c not in context]:
            claim_errors.append(f"Klaim {i} menyitasi chunk yang tidak ada di konteks: {unknown}.")
        source_quantities = set().union(*(extract_quantities(context[c]) for c in known))
        if missing := extract_quantities(claim.get("text", "")) - source_quantities:
            claim_errors.append(f"Klaim {i}: {sorted(missing)} tidak tertulis di chunk yang disitasi {known}.")
        errors.extend(claim_errors)
        if not claim_errors:
            valid.append(claim)

    cited = {c for claim in claims for c in claim.get("chunk_ids") or [] if c in context}
    cited_quantities = set().union(*(extract_quantities(context[c]) for c in cited))
    if stray := extract_quantities(output.get("answer", "")) - cited_quantities:
        errors.append(f"Jawaban memuat {sorted(stray)} yang tidak tertulis di chunk mana pun yang disitasi.")

    return GuardResult(ok=not errors, errors=errors, valid_claims=valid)
