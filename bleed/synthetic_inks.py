"""Known-truth stress test for the minimal bleed filter across front-ink types.

Clean fronts (bleed/inputs/sheetNN/perfect.jpg, Sharpie on graph paper, no reverse
writing) are recoloured as black marker, blue pen, red pen and pencil. The REAL
bleed layer measured on the pencil-sheet08 pair (per-channel optical density,
with sheet08's own grid removed) is multiplied onto each, and sheet08's real back
photo is used as the reverse. Because the clean front is known, we can measure:

* untouched:  pixels with no predicted bleed must be bit-identical to the input;
* strokes:    front ink in overlaps, retained density and colour error vs truth;
* bleed:      bleed-only pixels left visibly darker than the truth (> 0.07 OD);
* PSNR:       over all pixels the filter touched.

Limits: the bleed is real but the front is composited (multiplicative density),
and sheet08's back is reused for every front.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))
import recover_overlap as ro
import remove_bleed as rb

ROOT = Path(__file__).resolve().parents[1]
INKS = {  # per-channel density multipliers applied to the clean (gray) ink density
    'black_marker': (1.0, 1.0, 1.0),
    'blue_pen': (.75, .55, .12),
    'red_pen': (.10, .70, .60),
    'pencil': (.35, .35, .37),
}


def real_bleed_layer(state, result):
    """Per-channel bleed density from sheet08: observed where no front ink, model elsewhere."""
    D, _ = ro.density_rgb(state['front'])
    minimal = ro.minimal_restore(state, result, protection='on', paths=False, ink_tau=0., ink_anchor=False)  # earlier, permissive mask: sheet08's own pencil stays out of the bleed layer
    touched, ink = minimal['touched'], minimal['ink']
    grid = np.stack([ro.grid_through(D[..., c], touched) for c in range(3)], -1)
    observed = np.clip(D - grid, 0, None)
    model = result['bleed'][..., None] * np.array(minimal['bleed_ratio'], np.float32)
    B = np.where(ink[..., None], model, observed)
    return np.where(touched[..., None], B, 0).astype(np.float32)


def clean_front(sheet, shape):
    img = np.array(ImageOps.exif_transpose(Image.open(ROOT/f'bleed/inputs/{sheet}/perfect.jpg')).convert('RGB'))
    if img.shape[0] < img.shape[1]:
        img = np.rot90(img).copy()
    return cv2.resize(img, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)


def evaluate(truth, synth, out, F, B, touched):
    gray = lambda im: cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).astype(np.float32)
    Dt, Do = -np.log(np.maximum(gray(truth), 1)/255), -np.log(np.maximum(gray(out), 1)/255)
    Fg, Bg = F.mean(-1), B.mean(-1)
    overlap = (Fg > .15) & (Bg > .05)
    bleed_only = (Bg > .1) & (Fg < .03)
    err = np.abs(out.astype(np.float32) - truth)
    mse = np.mean((out[touched].astype(np.float32) - truth[touched])**2)
    return dict(
        untouched_identical=bool((out[~touched] == synth[~touched]).all()),
        stroke_pixels=int(overlap.sum()),
        stroke_mae_rgb=float(err[overlap].mean()),
        stroke_density_ratio=float(np.median(Do[overlap] / np.maximum(Dt[overlap], 1e-3))),   # 1 = truth
        bleed_left_visible=float(((Do - Dt) > .07)[bleed_only].mean()),
        psnr_touched=float(10*np.log10(255**2/max(mse, 1e-6))),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sheets', nargs='+', default=['sheet01', 'sheet03', 'sheet05'])
    parser.add_argument('--output', type=Path, default=ROOT/'bleed/outputs/synthetic-inks')
    parser.add_argument('--inks', nargs='+', default=list(INKS))
    parser.add_argument('--saturate', action='store_true',
                        help='clip the composite at the camera black floor measured in sheet08 Sharpie cores')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sample = ROOT/'bleed/samples/pencil-sheet08'
    _, state08, result08 = ro.run(sample/'front.jpg', sample/'back.jpg', args.output/'_sheet08', save=False)
    B = real_bleed_layer(state08, result08)
    back = state08['back_aligned']
    core = (state08['d_back'] > 1.35) & state08['valid']
    floor = np.percentile(state08['front'][core], 5, axis=0).astype(np.uint8)   # darkest the camera records
    print('black floor RGB', floor.tolist())
    rows = []
    for sheet in args.sheets:
        base = clean_front(sheet, B.shape[:2])
        Dclean, paper = ro.density_rgb(base)
        g = np.clip(Dclean.mean(-1), 0, None)
        for ink in args.inks:
            mult = INKS[ink]
            F = g[..., None] * np.array(mult, np.float32)
            truth = np.uint8(np.clip(paper*np.exp(-F), 0, 255))
            synth = np.uint8(np.clip(paper*np.exp(-(F + B)), 0, 255))
            if args.saturate:   # real overlaps of two dark inks bottom out at the sensor's black floor
                synth = np.maximum(synth, floor)
            valid = state08['valid']
            st = ro.prepare(synth, back, valid, ro.density(synth), state08['d_back'])
            res = ro.restore_overlap(st, {})
            row = dict(sheet=sheet, ink=ink)
            row['input'] = evaluate(truth, synth, synth, F, B, np.ones(valid.shape, bool) & valid)
            # Faithful Rowley-Brooke et al. 2013 baseline: replace blfg with the exemplar plate, blend.
            rb_out, _ = rb.rowley_brooke_restore(st['front'], st['labels'], valid, plate=st['plate'])
            rb_touched = cv2.dilate(((st['labels'] == rb.BACK) & valid).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
            row['rowley_brooke'] = evaluate(truth, synth, rb_out, F, B, rb_touched)
            if sheet == args.sheets[0]:
                Image.fromarray(rb_out).save(args.output/f'{sheet}-{ink}-rowley_brooke.png')
            strokes = ro.front_strokes(st, res)
            for mode in ('off', 'on', 'divide'):
                m = ro.minimal_restore(st, res, protection=mode, strokes=strokes)
                row[mode] = evaluate(truth, synth, m['output'], F, B, m['touched'])
                if sheet == args.sheets[0]:
                    Image.fromarray(m['output']).save(args.output/f'{sheet}-{ink}-{mode}.png')
            if sheet == args.sheets[0]:
                Image.fromarray(synth).save(args.output/f'{sheet}-{ink}-input.png')
                Image.fromarray(truth).save(args.output/f'{sheet}-{ink}-truth.png')
            print(json.dumps({k: (v if isinstance(v, str) else {kk: round(vv, 3) if isinstance(vv, float) else vv
                                                               for kk, vv in v.items()}) for k, v in row.items()}), flush=True)
            rows.append(row)
    summary = {}
    for ink in args.inks:
        for mode in ('input', 'rowley_brooke', 'off', 'on', 'divide'):
            sel = [r[mode] for r in rows if r['ink'] == ink]
            summary.setdefault(ink, {})[mode] = {k: float(np.mean([s[k] for s in sel])) for k in sel[0]}
    (args.output/'summary.json').write_text(json.dumps(dict(rows=rows, summary=summary), indent=2))
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
