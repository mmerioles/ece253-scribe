# Paired pencil / Sharpie experiment

Run from the repository root:

```powershell
uv run bleed/remove_bleed.py bleed/inputs/sheet08/front-01.jpg bleed/inputs/sheet08/back.jpg
uv run bleed/evaluate_pair.py
uv run python -m unittest discover -s bleed -p test_remove_bleed.py
```

The first command writes to `bleed/outputs/sheet08/front-01/`. `filtered.png` is the conservative result. `without_protection.png` disables only the added thin-stroke protection; it still retains the two foreground classes. `strong_removal.png` is a deliberately destructive diagnostic that replaces all reverse-ink regions, including overlaps. It is **not** the paper baseline and can erase pencil. The comparison image is input / conservative / strong, left to right. All outputs are actual processing results, with no transcription or generated lettering used in restoration.

## Reference and scope

[Rowley-Brooke, Pitie and Kokaram, CVPR 2013](https://www.cv-foundation.org/openaccess/content_cvpr_2013/papers/Rowley-Brooke_A_Non-parametric_Framework_2013_CVPR_paper.pdf), proposal reference [13]. This is a **modified prototype, not an exact reproduction or a validated paper baseline**. No DE-GAN model is implemented here.

The retained structure is local histogram-mode detrending (Section 3.1), four joint intensity classes with Mahalanobis costs and image-neighbour connections (3.2.1), connected-component boundary rules (3.2.2), and background-only restoration with edge blending (3.3). Both-side foreground regions are retained.

Explicit implementation differences:

- SIFT/RANSAC homography registration is added; the paper assumes registered inputs. EXIF orientation is honored and only the reverse side is mirrored. `--already-mirrored` skips this flip for pre-mirrored data, but still estimates registration.
- Border/table extremes are excluded from local modes. Otsu-based seed selection and covariance regularization supply unspecified initialization details; they are not ground-truth labels.
- Sequential ICM replaces QPBO. Empirical neighbour frequencies are normalized to degree four to prevent rare JPEG intensity pairs overwhelming paper pixels. The paper's alpha formula is clipped at zero. Table 1's asymmetric entry is symmetrized. These changes mean this is not optimization of the paper's original energy; convergence is recorded and capped at 20 sweeps.
- OpenCV Telea inpainting replaces Criminisi exemplar synthesis. A separate paper plate excludes every non-background class and the top 10% of background gradients. This does not reproduce original paper texture, especially graph lines.
- Added protection retains thin dark continuations connected to front-only strokes. Its constants are development heuristics, not validation-selected parameters. Very dark bleed can still look like overlap.

The whole restoration is one module; the evaluator reuses `main.py` and the existing fixed TrOCR model. No restoration input includes labels, line boxes, text, or a clean reference. The OCR evaluator alone uses those annotations.

## Historical pilot data and interpretation

The following describes the November 16 pilot, now archived under `archive/2026-11-16/`. The active November 19 dataset and commands are documented in [README.md](README.md); these historical scores are not new-dataset measurements.

`front.jpg` is the pencil side (`20261116_220102.jpg`); `back.jpg` is the Sharpie side (`20261116_220112.jpg`). Originals are preserved byte-for-byte. These are associated with sheet08 and must stay in the same dataset split. They live outside the clean-baseline `inputs/` so the reverse image is not accidentally counted as an independent OCR sample.

The reference text is the intended sheet08 sentence list, checked against visible pencil wording. Where Sharpie obscures a letter, it is an intended transcription, not proof of an independently visible pencil stroke. List numbers are excluded. The same manually annotated fractional line boxes are used on every output. No line segmentation or text-based reconstruction is performed.

This pair is a **development case** used while implementing the method, not a held-out test. No clean pencil-before-bleed photograph or pixel foreground mask was supplied. Therefore no PSNR, SSIM or foreground F1 is reported. OCR errors may exceed 100% because insertions count as errors. A lower OCR error alone does not prove stroke preservation.

The expected limitation is overlap: removing all reverse ink also removes pencil passing through that ink. Conservative protection leaves interference; the stronger diagnostic makes missing strokes visible. Graph lines are retained because they are physical front-side marks, not necessarily reverse interference. A clean pre-bleed pencil capture and manually marked overlap crops are needed to measure true pencil preservation.

`run.json` records input hashes, transform, registration support, image size, classifier settings and runtime. Registration residuals are inlier fit errors, not an independent measurement of alignment everywhere. `ocr-summary.json` and per-line CSVs record recognition. `bleed/archive/2026-11-16/results/pencil-sheet08.json` preserves the final experiment summary outside ignored outputs.

## Multi-capture registration

For the November 19 batch, failed native-resolution SIFT matching retries at a 1600-pixel maximum side. The same inlier count, ratio, coverage and residual checks apply at the feature resolution. The homography is converted back to native coordinates before warping, and run.json identifies the residual coordinate system. This wider-context fallback is an implementation adaptation, not a new restoration method. Rejected pairs are reported, never forced through.
