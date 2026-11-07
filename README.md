# SCRIBE: minimal TrOCR baseline

Read a photo of handwritten notes, one line at a time, and compare the prediction
with a reference transcription. No web UI, training, restoration, or automatic
line detection. Your supplied photo, ten reference sentences, and manually chosen
line boxes are included in `inputs/`.

## Run (PowerShell)

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if needed, then:

```powershell
cd C:\Users\matth\Desktop\repos\ece253-scribe
uv run main.py
```

uv installs Python/dependencies automatically. The first OCR run downloads the
public model from Hugging Face (roughly 1.3 GB of weights); allow time and disk
space. Inference runs locally on CPU; the photo is not uploaded. No API key is
needed. Subsequent runs reuse the downloaded dependencies and model.

To inspect the line crops without loading the model:

```powershell
uv run main.py --prepare-only
```

## Inputs

Put a photo and two matching files in `inputs/`:

```text
inputs/
  baseline.png         # supplied photo
  baseline.txt         # one reference sentence per line, no list numbers
  baseline.boxes.json  # one crop box per line, in reading order
```

Run another photo with `uv run main.py inputs/my-photo.jpg`. It needs
`my-photo.txt` and `my-photo.boxes.json` alongside it.

Each crop box is `[left, top, right, bottom]`, expressed as fractions of the image
width/height after correcting EXIF orientation. Example: `[0.15, 0.06, 0.90, 0.13]`
means start 15% across and 6% down, end 90% across and 13% down. The supplied boxes
are for **this photo**, exclude the list numbers, and include the full sentence.
New framing requires new boxes. Inspect `outputs/<photo-name>/line-*.png` before
trusting scores. Reuse boxes only for geometrically aligned versions of a page.

## Results

`outputs/baseline/` contains the line crops, `predictions.txt`, a per-line
`comparison.csv`, and overall `metrics.json`. Running the same photo name replaces
its results. Give restored versions different names to keep separate results.

- **CER**: character edits divided by reference characters (lower is better).
- **WER**: word edits divided by reference words (lower is better).
- **Exact lines**: how many normalized sentences match completely.
- **Normalized** scores ignore case/punctuation and normalize whitespace.
- **Strict** scores retain case/punctuation in the supplied reference.

The reference is your typed list, not OCR output. The photo has some punctuation
that differs from that list (for example, a period on line 2), so normalized scores
are useful for this first check. For a strictly transcribed benchmark, manually
correct the reference to the actual handwriting before evaluation. Never change
it just to match the model. Errors include insertions, so CER/WER can exceed 100%:
we do not label `100 - error rate` as accuracy. Overall rates aggregate edits over
all reference lines rather than averaging per-line percentages.

This is a **controlled, manually cropped baseline**, not full-page OCR. It does
not remove blur, shadows, grid lines, or bleed-through. TrOCR's standard processor
resizes/normalizes the RGB crops. Keep these crops, model, decoding, and scoring
rules fixed when comparing restoration methods.

Model: [microsoft/trocr-base-handwritten](https://huggingface.co/microsoft/trocr-base-handwritten).
The model is intended for single handwritten text lines. `uv.lock` pins Python
dependencies; `REVISION` in `main.py` pins the model and processor to the same
checkpoint. `metrics.json` records that revision.
