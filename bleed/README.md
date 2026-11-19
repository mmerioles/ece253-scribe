# Bleed-through removal: Rowley-Brooke base + foreground-stroke protection

Goal: remove reverse-side bleed (Sharpie) from a photo of the front, using a photo of the back. Keep everything else exactly as photographed, including front strokes that pass under the bleed, for any front ink.

- **Base:** Rowley-Brooke, Pitié & Kokaram, CVPR 2013 (proposal ref. [13]), checked line by line against the paper; see [METHOD.md](METHOD.md) §1.
- **Our contribution:** stage 3, *protect foreground strokes*. The paper has no such step: its strokes survive only if the classifier calls them front ink, which fails for ink fainter than the bleed.
- **Diagrams** (build with `tectonic`):
  - `results/diagrams/pipeline-comparison.{pdf,svg,png}`: the proposal's Fig. 4 for slides, with the classical branch (Rowley-Brooke, modified) above DE-GAN (front only). Source: `diagrams/comparison.tex`.
  - `results/diagrams/pipeline-paper-vs-ours.*`: the paper's stages next to ours. Source: `diagrams/paper-vs-ours.tex`.
- **DE-GAN** is not implemented yet.

## Run

From the repository root (Python 3.11, `uv`):

```sh
uv run bleed/recover_overlap.py bleed/samples/pencil-sheet08/front.jpg bleed/samples/pencil-sheet08/back.jpg  # outputs incl. rowley_brooke.png, minimal*.png
uv run bleed/eval_protection.py      # stroke-protection ablation on the real pair (~15 min)
uv run bleed/synthetic_inks.py --sheets sheet01 sheet03   # known-truth multi-ink test (~25 min)
uv run bleed/evaluate_pair.py        # TrOCR on the 10 annotated lines
uv run python -m unittest discover -s bleed -p 'test_*.py'
```

Outputs go to `bleed/outputs/` (gitignored); summaries are kept in `bleed/results/`.

| File | What it is |
| --- | --- |
| `rowley_brooke.png` | paper §3.3: blfg replaced with the exemplar plate, blended. Same registration, labels and plate as ours. |
| `minimal_protection_off.png` | ours without protection: every interference pixel → plate |
| `minimal_protection_on.png` | the paper's rule ("foreground left unchanged") applied to our stroke mask |
| `minimal.png` | **default**: confident strokes keep their ink with the bleed share divided out; everything else touched → paper |
| `recovered_clean.png` | separate OCR aid (ink redrawn on white, grid removed); not a minimal edit |

## Effect of stroke protection on sheet 08 (`eval_protection.py`)

Real pencil patches, taken from clean parts of the sheet, are multiplied into the front photo over real Sharpie bleed. The full pipeline is rerun, and the pencil kept is measured from the output images. This is semi-synthetic on the real photo: the paste multiplies density, which is the model our decomposition inverts, with no black-floor clipping. Truth masks come from line boxes and the back photo, not from our model. There are 24 pastes; the held-out half is in brackets.

| Output | Pencil kept under bleed (recall) | Pencil density kept | Bleed left visible | Clean pencil altered | OCR CER / WER |
| --- | ---: | ---: | ---: | ---: | ---: |
| Raw photo | – | – | – | – | 93.8% / 126.7% |
| Rowley-Brooke 2013 (faithful) | 7.6% (7.5%) | 11% | 9.9% | 0.3% | 55.5% / 73.3% |
| Ours, protection off | 6.2% (5.7%) | 8% | 11.3% | 0.0% | 54.3% / 77.8% |
| Ours, protection on (paper rule) | 59.4% (53.7%) | 71% | 11.6% | 0.0% | 46.5% / 86.7% |
| **Ours (default)** | **59.4% (53.7%)** | **67%** | **11.6%** | **0.0%** | **40.6% / 80.0%** |

How to read it:
- Rowley-Brooke → off changes only identification; off → default changes only protection.
- **Protection keeps 8× more of the pencil under the bleed (7.6% → 59.4%) and lowers character error from 55.5% to 40.6%, which is 38 fewer wrong characters out of 256.** Clean pencil is untouched.
- **The costs:**
  - leftover bleed is slightly higher, 11.6% vs 9.9% of bleed-only pixels;
  - word error is still worse, 80.0% vs 73.3%: words come back with most letters right but not exactly right.
- **The stroke mask (`strict_ink`):** a protected pixel needs bleed-free density `d_g − a·m_b > τ = 0.3` and must connect to front ink visible outside the bleed. The earlier mask (`p > 0`) kept bleed mottle inside opaque Sharpie cores as "ink". That kept 73% of the pencil but OCR read the blotches as extra characters (75.4% CER).
- **How τ was chosen** (`sweep_protection.py`, `results/pencil-sheet08-overlap/protection-sweep.json`, figure `results/diagrams/protection-tradeoff.{png,svg,pdf}`, source `diagrams/tradeoff.tex`):
  - every τ from 0.25 to 0.4 beats Rowley-Brooke overall (36.7–41.8% CER);
  - τ = 0.3 is the strictest setting whose held-out pasted-pencil recall stays above the 50% floor we fixed beforehand.
  - Rule we fixed beforehand: 10+ characters better than Rowley-Brooke on both even and odd lines. τ = 0.3 meets it on even lines (31 characters) but not on odd lines (7). τ was chosen after seeing both halves, so neither half is a clean hold-out. Treat OCR as one-sheet evidence.
- Colour per ink from visible strokes, median smoothing and a darkness cap were also tried; none helped once the mask was strict (sweep JSON).
- On and default protect the same strokes; on leaves the bleed on them (see the synthetic test).

Crops: `results/pencil-sheet08-overlap/ocr-example-line6.jpg` (real line 6 with TrOCR readings: Rowley-Brooke "Philippines special print", ours "I misplaced a decimal points"), `pasted-ladder.jpg` (pasted pencil) and `real-ladder.jpg` (real line 5).

## Other front inks, known truth (`synthetic_inks.py`)

Measured with the earlier, permissive stroke mask (τ = 0, no anchoring); not rerun with the current default. Clean sheet 01 recoloured as four inks, combined with sheet 08's measured bleed and real back photo (one sheet; the run was stopped after sheet 01). Cell = stroke density in overlaps / truth (1.00 perfect), RGB colour error in brackets.

| Front ink | Rowley-Brooke | Ours off | Ours on | **Ours divide** | Bleed left (RB / divide) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pencil | 0.60 (41) | 0.59 (46) | 1.48 (33) | **1.00 (11)** | 7% / 2% |
| Blue pen | 0.51 (44) | 0.49 (51) | 1.40 (29) | **1.00 (8)** | 11% / 5% |
| Red pen | 0.55 (44) | 0.54 (50) | 1.43 (30) | **1.00 (8)** | 10% / 4% |
| Black marker | 0.96 (15) | 0.85 (44) | 1.19 (18) | **1.00 (6)** | 21% / 15% |

- Rowley-Brooke keeps dark marker (classified as front ink) but erases ~half of pen and pencil under the bleed; off behaves the same.
- On keeps strokes with the bleed still on them (1.2–1.5× too dark).
- Untouched pixels were bit-identical in every case.
- Caveat: the composite multiplies densities, the model divide inverts, so divide's 1.00 is partly by construction. An earlier rerun clipped at the camera's black floor gave 0.94–0.97.

## Limits

- One real pair. Thresholds were tuned on it (tune/test split for the bleed model; rendering constants by inspection).
- Bleed texture inside opaque Sharpie can't be predicted from the back photo. Falsely kept ink there is the main error source.
- Front inks that themselves bleed into the back photo (felt-tip, marker) need a two-sided decomposition (not done).
- τ is an absolute density, set above this sheet's median pencil (0.22). Lighter pencil on another sheet could lose protection; an image-relative threshold is the next step.
- Photos larger than 6000 px are downscaled with a warning.
- The pair belongs with sheet08 in any dataset split. The Sharpie photo and `inputs/sheet08/perfect.jpg` are not clean pencil ground truth.
