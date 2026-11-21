<<<<<<< HEAD
# Bleed dataset and handoff

Extract normal left-to-right foreground pencil/pen text. The reverse-side Sharpie is noise, not ground truth. The current filter still loses strokes at overlaps; this import does not improve restoration.

## Data

`inputs/sheet01/` through `sheet10/` follow the handwritten page numbers in Data.zip. Each contains:
=======
# Bleed-through removal

**Problem:** Sharpie on the back of a page bleeds through to the front. We remove it using a photo of each side, keeping everything else as photographed, including front writing that passes under the bleed.

**Approach:** We built on Rowley-Brooke et al., CVPR 2013 ([paper](https://www.cv-foundation.org/openaccess/content_cvpr_2013/papers/Rowley-Brooke_A_Non-parametric_Framework_2013_CVPR_paper.pdf)) and added one step: **stroke protection**.

![Pipeline: paper vs ours](results/diagrams/pipeline-paper-vs-ours.png)

## How it works

1. **Align** the back photo to the front (mirror, then match features).
2. **Find the bleed.** Label each pixel as paper, front ink, bleed, or both, as in the paper.
3. **Protect front strokes** (our addition). The paper erases any front ink that is fainter than the bleed, so pencil or pen under Sharpie disappears. We detect those strokes and keep them, removing only the bleed's darkening.
4. **Fill in** the remaining bleed with paper texture, continuing the printed grid lines underneath.

Pixels with no bleed are left exactly as they were.
>>>>>>> 695b8894e8f46b52b69238b062d6af35619a575b

- `back.jpg`: reverse-side noise reference; mirror only during registration.
- `front-01.jpg`, etc.: foreground with bleed, photographed several times.
- `truth.txt`: ten foreground sentences shared by all captures, without list numbers.
- `front-01.boxes.json`, etc.: ten crop rectangles in EXIF-oriented image coordinates, normalized to 0–1.
- `metadata.json`: original filenames, hashes, roles and review notes.

<<<<<<< HEAD
There are 41 original JPEGs: 10 backs and 31 fronts (sheet06 has four fronts; others have three). Original bytes and EXIF are preserved; the loaders and previews apply EXIF orientation. Use an EXIF-aware viewer. Capture order is not a severity ranking.

Start your review at [REVIEW.md](REVIEW.md): it links all photo/box previews and transcriptions. Labels are provisional wherever bleed obscures letters. Check sheet03 line4 spelling and sheet06 line6, which repeats the fan sentence instead of the reverse-side battery sentence. Sheet05 has bold foreground and scribbled reverse noise. After checking a sheet, edit its truth file and set `user_verified` in its metadata to true. REVIEW.md is an import-time snapshot; truth.txt is authoritative.

`dataset.json` records the import inventory. Old inputs, pilot samples and scores are preserved under `archive/2026-11-16/`; legacy sheets05/07 correspond to new sheets07/05. Archive data is excluded from evaluation. Keep every capture of a physical sheet in the same train/test split, including its legacy captures.

No clean foreground-only photos or pixel masks were supplied: truth.txt is an OCR reference, not a clean image target.

## Run from repository root

```sh
uv sync --locked
uv run main.py bleed --prepare-only
uv run main.py bleed
uv run bleed/run_dataset.py --ocr
uv run bleed/remove_bleed.py bleed/inputs/sheet08/front-01.jpg bleed/inputs/sheet08/back.jpg
uv run bleed/evaluate_pair.py --sample bleed/inputs/sheet08 --capture front-01
uv run python -m unittest discover -s bleed -p "test_*.py"
```

The first command after setup creates 310 crops without loading OCR. `main.py bleed` evaluates only fronts using shared truth; backs are excluded. OCR downloads pinned TrOCR weights on its first run. Outputs are gitignored under `bleed/outputs/sheetNN/front-NN/`; reruns overwrite matching files. Old generated output folders may remain locally and are not new-dataset measurements.

## Implementation

`remove_bleed.py` contains the restoration; `evaluate_pair.py` reuses root `main.py` to compare input and filtered variants with identical crops. [METHOD.md](METHOD.md) describes the Rowley-Brooke reference and deviations: ICM replaces QPBO and Telea replaces exemplar synthesis. This is a modified prototype, not an exact reproduction. Restoration never uses the truth text to redraw letters.

Next: finish manual label/box review, then improve overlap preservation and evaluate on sheet-level held-out data. Archived pilot scores do not measure this new dataset.

`run_dataset.py` applies the existing filter to all front/back pairs, then optionally scores all four variants with one OCR model load. It records failures explicitly in `outputs/dataset-run.json`; rejected registrations receive no filtered result. Matching retries at 1600 pixels if native-resolution features fail, retaining the same support and coverage checks. Outputs remain full resolution.

Open `outputs/review.html` after the batch to browse every successful comparison and the failure list. Columns are input / conservative / strong removal. The `--ocr` summary scores only successfully registered captures, so always report the failure count alongside CER/WER. It is a development-set comparison with provisional labels, not a held-out accuracy claim.

Latest completed batch: [November 19 results](results/2026-11-19/README.md), with the failure list, metrics and a saved example.
=======
From the repo root:

```sh
uv run bleed/recover_overlap.py bleed/samples/pencil-sheet08/front.jpg bleed/samples/pencil-sheet08/back.jpg
uv run bleed/eval_protection.py    # stroke-protection test (~15 min)
uv run bleed/synthetic_inks.py --sheets sheet01 sheet03   # other ink colours (~25 min)
uv run bleed/evaluate_pair.py      # OCR score
uv run python -m unittest discover -s bleed -p 'test_*.py'
```

Outputs go to `bleed/outputs/`. Saved results are in `bleed/results/`.

| Output | What it is |
| --- | --- |
| `minimal.png` | **Our result** (default) |
| `rowley_brooke.png` | The paper's method, for comparison |
| `minimal_protection_off.png` | Ours without stroke protection |
| `recovered_clean.png` | Ink redrawn on white, for OCR only |

## Results (sheet 08)

We pasted real pencil strokes under real Sharpie bleed, then checked how much pencil survived.

| Method | Pencil kept under bleed | Bleed left | OCR character error |
| --- | ---: | ---: | ---: |
| Raw photo | – | – | 93.8% |
| Rowley-Brooke 2013 | 7.6% | 9.9% | 55.5% |
| **Ours** | **59.4%** | **11.6%** | **40.6%** |

- **Stroke protection keeps about 8× more pencil** and cuts character errors by about a quarter.
- Pencil outside the bleed is never changed.
- **The cost:** slightly more bleed is left behind, and word error is still a bit higher than the paper's, because recovered words are mostly but not exactly right.

**Other inks:** on a synthetic test with pencil, blue pen, red pen and black marker, the paper erases about half of pen and pencil under the bleed. Ours keeps it.

## Limits

- Tested on one real page, and settings were tuned on that page.
- Bleed texture inside solid Sharpie can't be predicted from the back photo; this causes most of our errors.
- Front inks that also bleed through (marker, felt-tip) aren't handled.
- Much lighter pencil on another page might not be protected.
- The DE-GAN (deep learning) branch is not implemented yet.
>>>>>>> 695b8894e8f46b52b69238b062d6af35619a575b
