# Pencil / Sharpie handoff

## Goal and current status

Extract front-side pencil from mirrored Sharpie bleed-through. Keep the architecture simple and use the proposal's Rowley-Brooke CVPR 2013 method as the reference.

The current prototype removes much of the Sharpie, but **erases pencil at overlaps and is not a successful pencil-only recovery**. The latest request is to improve filtering or reconnect missing strokes. That follow-up has not been implemented yet; only a source-image crop was inspected after the first experiment.

## Run

From the repository root, with Python 3.11 and `uv` installed:

```sh
uv sync --locked
uv run bleed/remove_bleed.py bleed/samples/pencil-sheet08/front.jpg bleed/samples/pencil-sheet08/back.jpg
uv run bleed/evaluate_pair.py
uv run python -m unittest discover -s bleed -p test_remove_bleed.py
```

The OCR command downloads the pinned TrOCR weights on its first run. It uses CPU inference. Generated files go to `bleed/outputs/pencil-sheet08/` and are gitignored; rerunning overwrites them. The source JPEGs are preserved. No absolute local paths are needed.

## Files to start with

- `remove_bleed.py`: registration, detrending, four-class joint histogram, component refinement, background filling and foreground protection; one module.
- `evaluate_pair.py`: compares input, protection off/on, and strong removal using the existing root `main.py` recognizer and identical line crops.
- `samples/pencil-sheet08/`: original pair, intended transcription, line boxes and provenance. `front.jpg` is pencil; `back.jpg` is Sharpie. Only the back is mirrored during processing.
- [METHOD.md](METHOD.md): algorithm details, paper link, and explicit implementation deviations. This is a modified prototype: ICM replaces QPBO and Telea replaces exemplar synthesis. Do not call it an exact paper reproduction.
- [Saved comparison](results/pencil-sheet08/comparison.png): input / conservative / strong removal, left to right.
- [Filtered preview](results/pencil-sheet08/filtered-preview.png), [metrics](results/pencil-sheet08.json), and per-line CSVs in `results/pencil-sheet08/`: committed review artifacts; full-resolution outputs can be regenerated.

## Latest measurements

One development sheet, ten lines; lower error is better:

| Variant | Character error | Word error |
| --- | ---: | ---: |
| Input | 93.75% | 126.67% |
| Protection off | 76.56% | 97.78% |
| Conservative filter | 73.44% | 100.00% |
| Strong removal | 80.86% | 97.78% |

No variant recognizes any entire line correctly. Errors can exceed 100% because insertions count. Four tests cover mirroring, rejecting blank registration, keeping dark interior ink valid, and preserving foreground/invalid pixels.

## Next experiment

1. Start from the original front/back pair, not the already erased result. Try estimating and subtracting the bleed contribution within overlaps instead of replacing the entire region with paper. Keep this as a clearly labeled adaptation alongside the current baseline.
2. Test only small, local stroke-gap repair as a separate ablation. Large missing sections cannot be verified from these photos; do not redraw letters from the known transcription and claim recovery.
3. Inspect overlap crops and rerun the fixed OCR evaluation. Preserve the baseline comparison and report failures as well as gains.

There is no clean pre-bleed pencil image or pixel foreground mask, so PSNR/SSIM and foreground F1 are unavailable. The Sharpie photo and `inputs/sheet08/perfect.jpg` are **not** clean pencil ground truth. Keep this pair with sheet08 in any dataset split. The ten existing clean sheets and shared OCR baseline should remain intact; the duplicate baseline folder was already removed.

Do not push without the user's instruction. No restoration or gap-repair success should be inferred from the current passing safeguard tests.
