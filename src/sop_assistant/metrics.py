"""Metrik evaluasi yang dipakai test dan eval/run_eval.py."""

from __future__ import annotations

import re


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, 1):
        current = [i]
        for j, char_b in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (char_a != char_b)))
        previous = current
    return previous[-1]


def char_error_rate(hypothesis: str, reference: str) -> float:
    """CER = jarak edit / panjang referensi, setelah spasi dinormalkan."""
    hyp = re.sub(r"\s+", " ", hypothesis).strip()
    ref = re.sub(r"\s+", " ", reference).strip()
    return levenshtein(hyp, ref) / max(len(ref), 1)
