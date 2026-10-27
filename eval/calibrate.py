"""Pilih ABSTAIN_THRESHOLD dari cosine dense top-1 pertanyaan golden, tulis ke eval/calibration.json.

Skor RRF tidak dipakai karena hanya mencerminkan peringkat. Peringatan: threshold dari
21 pertanyaan (18 in-scope, 3 out-of-scope) berisiko overfit.
"""

from __future__ import annotations

import json

from rich.console import Console
from rich.table import Table

from sop_assistant.config import CALIBRATION_FILE
from sop_assistant.evaluation import choose_threshold, load_golden
from sop_assistant.guards import mask_pii
from sop_assistant.retrieve import dense_top1_score

WARNING = (
    "Threshold dipilih dari golden set kecil yang juga dipakai untuk evaluasi; "
    "berisiko overfit dan perlu dikalibrasi ulang dengan pertanyaan nyata."
)


def main() -> None:
    items = load_golden()
    scores = {item.id: dense_top1_score(mask_pii(item.question)[0]) for item in items}
    in_scope = [scores[i.id] for i in items if i.in_scope]
    out_scope = [scores[i.id] for i in items if not i.in_scope]
    threshold, method = choose_threshold(in_scope, out_scope)

    table = Table(title="Cosine dense top-1 per pertanyaan")
    for column in ("ID", "Kategori", "Lingkup", "Skor", "Abstain?"):
        table.add_column(column)
    for item in sorted(items, key=lambda i: scores[i.id], reverse=True):
        scope = "in-scope" if item.in_scope else "[red]out-of-scope[/]"
        table.add_row(item.id, item.category, scope, f"{scores[item.id]:.3f}", "ya" if scores[item.id] < threshold else "")
    console = Console()
    console.print(table)
    console.print(f"Threshold: [bold]{threshold:.3f}[/] ({method})")
    console.print(f"[dim]{WARNING}[/]")

    CALIBRATION_FILE.write_text(
        json.dumps(
            {
                "abstain_threshold": round(threshold, 4),
                "method": method,
                "lowest_in_scope": round(min(in_scope), 4),
                "highest_out_of_scope": round(max(out_scope), 4),
                "n_in_scope": len(in_scope),
                "n_out_of_scope": len(out_scope),
                "scores": {k: round(v, 4) for k, v in scores.items()},
                "warning": WARNING,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    console.print(f"Ditulis ke {CALIBRATION_FILE}")


if __name__ == "__main__":
    main()
