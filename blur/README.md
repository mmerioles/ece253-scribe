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
| `justin_pNN_<condition>__<method>.jpg` | restored output of `<method>` (e.g. `__l0`, `__nafnet`), added later |

Pages: p01–p04 pen/squared, p05–p07 pen/blank, p08 pencil/squared, p09–p10
pencil/blank. All pages carry the same 10 lines, so every `.txt` is a copy of
`bleed/inputs/baseline.txt` (no list numbers — the boxes exclude them).

## Boxes: generate, don't draw

```
uv run blur/make_boxes.py blur/inputs/justin_p06_clean.jpg blur/inputs/justin_p06_blur_*.jpg
```

The first file is the sharp reference for that page. The script finds the
page's four corners in each photo (white page on the dark desk), takes the
line layout once from the reference, and projects it into every photo — so it
also works on photos far too blurred for feature matching. Needs numpy +
Pillow only. Check `blur/outputs/boxes_preview/*.jpg` (red = line boxes,
green = detected page) before running OCR; fix a bad one by hand if needed.

Then:

```
uv run main.py blur --prepare-only   # look at the crops
uv run main.py blur                  # OCR + CER/WER -> blur/outputs/
```
