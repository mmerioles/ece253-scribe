# SCRIBE

From the repo root, with [uv](https://docs.astral.sh/uv/getting-started/installation/) installed:

```powershell
uv run main.py bleed
uv run main.py blur
uv run main.py shadow
```

Each command evaluates input images in that artifact's `inputs/` folder using
one shared TrOCR model. First run downloads dependencies and model weights.

- Put individual work in `bleed/`, `blur/`, or `shadow/`.
- Blur/shadow: each input needs `photo.jpg`, `photo.txt` (one reference line per crop, no list numbers),
  and `photo.boxes.json` (`[left, top, right, bottom]` fractions from 0 to 1).
- Bleed uses `inputs/sheetNN/front-NN.jpg`, matching boxes, and shared `truth.txt`; reverse `back.jpg` images are excluded. See [bleed/README.md](bleed/README.md).
- Add `--prepare-only` to inspect crops without OCR.
- Results: `<artifact>/outputs/<photo-name>/`; combined scores: `outputs/summary.json`
  inside that artifact. Use unique photo names. Reruns overwrite matching results.

CER/WER are character/word error rates; lower is better. Normalized scores ignore
case and punctuation. Set scores count edits across all lines, not averages of image rates.

Pencil/Sharpie bleed removal: see [bleed/README.md](bleed/README.md).
