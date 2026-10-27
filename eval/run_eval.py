"""Jalankan evaluasi, sama dengan `sop eval`.

Contoh:
    python eval/run_eval.py                      # hybrid
    python eval/run_eval.py --mode dense         # ablation
    python eval/run_eval.py --baseline full-context
    python eval/run_eval.py --report-only        # bangun ulang report.md dari eval/results/
"""

from __future__ import annotations

import argparse

from sop_assistant.evaluation import run, summarize, write_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["hybrid", "dense"], default="hybrid")
    parser.add_argument("--baseline", choices=["full-context"])
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()

    if not args.report_only:
        data = run("full_context" if args.baseline else args.mode)
        for metric, value in summarize(data).items():
            print(f"{metric}: {value}")
    print(f"Laporan: {write_report()}")


if __name__ == "__main__":
    main()
