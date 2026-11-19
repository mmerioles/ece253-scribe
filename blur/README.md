# Blur (Justin)

Each handwritten page is photographed in four conditions and scored with the
shared `main.py` TrOCR baseline. Restored versions are added later as extra
inputs (same `.txt`/`.boxes.json`), so "with vs. without restoration" is just
two rows of `blur/outputs/summary.json`.

## Inputs (`blur/inputs/`)

| file | how it was shot |
|---|---|
| `justin_pNN_clean.jpg` | same spot, steady, no artifact |
| `justin_pNN_blur_motion_mild.jpg` | shutter 1/30 s, slight hand movement during exposure |
| `justin_pNN_blur_motion_severe.jpg` | shutter 1/8 s, same movement |
| `justin_pNN_blur_defocus_moderate.jpg` | same distance, focus locked on an object at a different depth |
| `justin_pNN_<condition>__l0.jpg` | restored by the classical module (L0 blind kernel on a high-ink patch + hyper-Laplacian deconvolution), 2800 px long side |
| `justin_pNN_<condition>__ctrl.jpg` | the same photo only resized to 2800 px — resolution-matched "no restoration" control |
| `justin_pNN_<condition>__nafnet.jpg` | learned method, to be added |

Pages: p01–p04 pen/squared, p05–p07 pen/blank, p08 pencil/squared, p09–p10
pencil/blank. All pages carry the same 10 sentences; the `.txt` is
`bleed/inputs/baseline.txt` (no list numbers — the boxes exclude them) **split
the way it was physically written**: on p01 sentence 5 wraps onto a second
line ("The shadow passed every" / "test") so its `.txt` has 11 lines, p02 has
12. Boxes are per physical line, and `main.py` needs box count == txt lines.

## Boxes: generate, don't draw

```
uv run blur/make_boxes.py blur/inputs/justin_p06_clean.jpg blur/inputs/justin_p06_blur_*.jpg --lines 10
```

`--lines` is the number of *physical* lines on that page (11 for p01, 12 for p02).

The first file is the sharp reference for that page. The script finds the
page's four corners in each photo (white page on the dark desk), takes the
line layout once from the reference, and projects it into every photo — so it
also works on photos far too blurred for feature matching. Needs numpy +
Pillow only. Check `blur/outputs/boxes_preview/*.jpg` (red = line boxes,
green = detected page) before running OCR; fix a bad one by hand if needed.

Then:

```
uv run main.py blur --prepare-only          # look at the crops
uv run main.py blur                         # OCR + CER/WER -> blur/outputs/
uv run blur/summarize_by_condition.py       # CER/WER per condition x method (Markdown table)
```

The restored `__l0` / `__ctrl` inputs were produced with
`scripts/restore_blur_photos.py` from the `scribe_blur` package (numpy/scipy/
OpenCV; not part of this uv environment). `blur/restore_log.csv` records the
kernel-estimation patch, kernel support and sharpness (LAPV) before/after for
each photo. Known failure: faint pencil + defocus (p09/p10 defocus) — the
two-tone prior finds no dark strokes, the kernel degenerates and noise is
amplified; those rows are kept in the evaluation on purpose.
