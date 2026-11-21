# Improved method: full dataset

Run from the repository root:

```sh
uv run bleed/run_dataset.py --method improved --ocr
```

This applies recover_overlap.py defaults, selected on the original sheet08 pilot, to every inputs/sheetNN/front-NN.jpg with its matching back.jpg. No new dataset tuning is performed. Original JPEGs and truth files are unchanged.

Outputs: bleed/outputs/improved/sheetNN/front-NN/. Open bleed/outputs/improved/review.html to compare input, Rowley-Brooke and our minimal output. The four scored variants are input, rowley_brooke, minimal_protection_off and minimal. The baseline shares our registration, labels and paper plate; it is a matched restoration-stage comparison, not a completely separate original author implementation.

The batch reuses the improved method and exports only the relevant images. Source hashes, input hashes, registration diagnostics, parameters and failures are in dataset-run.json. The pinned TrOCR model is loaded once. All variants use the same EXIF-oriented line boxes and shared truth.txt. Failed registrations are listed and excluded from paired OCR scores. Report that count with the metrics.

Use --method legacy to run the old prototype separately. November 19 results used that older pipeline and do not measure the improved method. Full images and the gallery are generated/ignored; compact published summaries belong in results/.

The dataset has repeated captures of ten physical sheets; these are not independent samples. The original sheet08 was used in method development, so results including it are development results. Labels await user verification, and no clean foreground-only target exists for the new photos.

For faster restoration on a multicore machine, add `--workers 3`. Add `--resume` to reuse completed restorations whose source-code and input hashes match; OCR is recomputed.

Latest completed experiment: [November 20 improved results](results/2026-11-20-improved/README.md). Attempted 31 captures, processed 28. CER: raw 98.7%, matched baseline 53.6%, improved 26.8%. The three sheet05 captures failed registration.
