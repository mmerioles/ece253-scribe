# TrOCR baseline

## Run

```powershell
cd C:\Users\matth\Desktop\repos\ece253-scribe
uv run main.py
```

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/). First run downloads dependencies and the model. Your sample photo is included.

## New images

Put these matching files in `inputs/`:
- `photo.jpg` — page image
- `photo.txt` — one reference sentence per line, without list numbers
- `photo.boxes.json` — one `[left, top, right, bottom]` crop per line, using fractions from 0 to 1

Copy the sample files as a starting point. Adjust crop boxes for each new photo; exclude list numbers.

```powershell
uv run main.py inputs/photo.jpg --prepare-only  # inspect crops first
uv run main.py inputs/photo.jpg
```

## Results

Find crops, predictions, `comparison.csv`, and `metrics.json` in `outputs/<photo-name>/`.

CER = character error rate; WER = word error rate. Lower is better. Normalized scores ignore case and punctuation; strict scores retain them. Reusing a photo name overwrites its results.
