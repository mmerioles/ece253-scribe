# November 19 batch results

Attempted all 31 fronts. 28 processed; all three sheet05 captures failed registration against the scribbled reverse. No filtered result is claimed for failures.

| Variant | Normalized CER | Normalized WER | Exact lines / 280 |
| --- | ---: | ---: | ---: |
| input | 98.71% | 122.40% | 0 |
| without_protection | 61.02% | 85.24% | 4 |
| filtered | 60.88% | 85.78% | 3 |
| strong_removal | 67.21% | 93.73% | 0 |

These are development results using provisional transcriptions, not held-out scores. All variants use identical crops on the same 28 captures. Repeated shots are correlated.

Visual inspection of all 28 comparisons shows reduced Sharpie but missing pencil strokes and residual mirrored ink. Strong removal is a destructive diagnostic, not the recommended output. No letter reconstruction or truth-guided drawing was used.

[Example: input / conservative / strong](example.png) | [Full metrics and failures](summary.json) | [Local full batch gallery](../../outputs/review.html)

Regenerate from the repository root: `uv run bleed/run_dataset.py --ocr`. The gallery and full-resolution images live in ignored `bleed/outputs/`. Six safeguard tests pass.

Implementation changes: one batch runner reuses the existing restoration and OCR; registration retries at 1600 pixels with unchanged acceptance checks, then applies the transform at native resolution. No restoration/classification parameters were changed.
