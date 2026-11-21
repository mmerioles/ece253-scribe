"""Group the per-image OCR results of `main.py blur` by blur condition and method.

Reads every `blur/outputs/<photo>/comparison.csv` written by `main.py` (one
row per line: reference, prediction), pools the lines of all photos that share
a condition (clean / blur_motion_mild / blur_motion_severe / blur_defocus_moderate)
and a method suffix (none / `__ctrl` resized control / `__l0` restored), and
prints corpus-level CER / WER (total edits over total reference characters,
same convention as main.py's summary) as a Markdown table ready for the slides.

    uv run blur/summarize_by_condition.py            # table on stdout
    uv run blur/summarize_by_condition.py --csv blur/outputs/by_condition.csv
"""
from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

from jiwer import cer, wer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from main import normalize  # noqa: E402  (same normalization as the shared evaluator)

NAME = re.compile(r"^(?P<writer>[a-z]+)_(?P<page>p\d+)_(?P<cond>[a-z_]+?)(?:__(?P<method>[a-z0-9]+))?$")
COND_ORDER = ["clean", "blur_motion_mild", "blur_motion_severe", "blur_defocus_moderate"]
METHOD_ORDER = ["none", "ctrl", "l0"]
METHOD_LABEL = {"none": "original photo", "ctrl": "resized control", "l0": "L0 blind deblur"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outputs", type=Path, default=Path(__file__).resolve().parent / "outputs")
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--strict", action="store_true", help="use raw text instead of normalized")
    args = ap.parse_args()

    groups: dict[tuple[str, str], tuple[list[str], list[str], set[str]]] = defaultdict(lambda: ([], [], set()))
    for comp in sorted(args.outputs.glob("*/comparison.csv")):
        m = NAME.match(comp.parent.name)
        if not m:
            continue
        cond, method = m.group("cond"), m.group("method") or "none"
        refs, preds, pages = groups[(cond, method)]
        with comp.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                r, p = row["reference"], row["prediction"]
                if not args.strict:
                    r, p = normalize(r), normalize(p)
                refs.append(r)
                preds.append(p)
        pages.add(m.group("page"))

    rows = []
    for cond in COND_ORDER + sorted({c for c, _ in groups} - set(COND_ORDER)):
        for method in METHOD_ORDER:
            if (cond, method) not in groups:
                continue
            refs, preds, pages = groups[(cond, method)]
            rows.append(dict(condition=cond, method=METHOD_LABEL.get(method, method), pages=len(pages),
                             lines=len(refs), cer=cer(refs, preds), wer=wer(refs, preds),
                             exact_lines=sum(a == b for a, b in zip(refs, preds))))
    if not rows:
        print(f"no comparison.csv under {args.outputs}; run `uv run main.py blur` first")
        return 1

    print("| condition | method | pages | lines | CER | WER | exact lines |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for r in rows:
        print(f"| {r['condition']} | {r['method']} | {r['pages']} | {r['lines']} | "
              f"{r['cer']:.1%} | {r['wer']:.1%} | {r['exact_lines']}/{r['lines']} |")
    if args.csv:
        with args.csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
