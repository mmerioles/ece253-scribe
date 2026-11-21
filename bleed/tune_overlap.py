"""Tune and score recover_overlap.py against two pixel-level truths.

No clean pencil photograph exists for this pair, so we use two proxies:

* Bleed-only truth (real data). The reverse side's list numbers lie to the right
  of every manually boxed pencil line, so reverse-ink pixels at
  x in [box_right + margin, 0.86 W] contain no pencil. The truth there is paper:
  any recovered "pencil" is leaked bleed.
* Pencil-over-bleed truth (semi-synthetic). Real pencil patches from bleed-free
  parts of the boxes are added, in optical density, onto real bleed in other
  lines. Recovered pencil = p(with paste) - p(without paste) is compared with the
  pasted density. Paste locations are split into tune and test halves.

Line boxes are evaluation annotations only; restoration never receives them.
Parameters are chosen by coordinate search on the tune half (even lines for
bleed-only truth, even-seed pastes); the test half is reported separately.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import recover_overlap as ro

ROOT = Path(__file__).resolve().parents[1]
STROKE = 0.15          # optical density above which a pixel counts as pencil ink
VISIBLE = 0.07         # residue darker than ~7% below paper counts as leaked bleed


def bleed_only_mask(state, boxes, m_back):
    h, w = m_back.shape
    mask = np.zeros((h, w), np.int8) - 1          # -1 none, else line index
    dark = cv2.cvtColor(state['front'], cv2.COLOR_RGB2GRAY) < 45   # punch holes / page edge
    for i, (l, t, r, b) in enumerate(boxes):
        y0, y1 = int(t*h) - 20, int(b*h) + 20
        x0, x1 = int(r*w) + 60, int(.86*w)
        sel = np.zeros((h, w), bool); sel[y0:y1, x0:x1] = True
        mask[sel & (m_back > .5) & ~dark & state['valid']] = i
    return mask


def clean_pencil_mask(state, boxes):
    """Visible front-only pencil: ink in a boxed line, >=5 px from any reverse ink."""
    h, w = state['d_front'].shape
    reverse = ro.back_map(state['d_back'], INITIAL) > .05
    reverse = cv2.dilate(reverse.astype(np.uint8), np.ones((11, 11), np.uint8)) > 0
    mask = np.full((h, w), -1, np.int8)
    for i, (l, t, r, b) in enumerate(boxes):
        sel = np.zeros((h, w), bool); sel[int(t*h):int(b*h), int(l*w):int(r*w)] = True
        mask[sel & (state['d_front'] > STROKE) & ~reverse & state['valid']] = i
    return mask


def make_pastes(state, boxes, region, m_back, seed_count=24):
    """Pencil patches (clean source) -> bleed-heavy targets on other lines."""
    h, w = region.shape
    d = state['d_front']
    near = cv2.dilate(region.astype(np.uint8), np.ones((31, 31), np.uint8)) > 0
    sources = []
    for i, (l, t, r, b) in enumerate(boxes):
        y0, y1 = int(t*h), int(b*h); ph = y1 - y0
        for x in range(int(l*w), int(r*w) - 160, 40):
            if near[y0:y1, x:x+160].any():
                continue
            patch = np.clip(d[y0:y1, x:x+160], 0, None)
            # remove grid lines / paper texture: keep ink above a local floor
            patch = np.where(patch > .08, patch, 0)
            if (patch > STROKE).mean() > .04:
                sources.append((i, patch))
    rng = np.random.default_rng(0)
    pastes = []
    for k in range(seed_count):
        src_line, patch = sources[rng.integers(len(sources))]
        ph, pw = patch.shape
        for _ in range(200):
            i = int(rng.integers(len(boxes)))
            if i == src_line:
                continue
            l, t, r, b = boxes[i]
            y = int(t*h) + int(rng.integers(-5, 6)); x = int(rng.integers(int(l*w), int(r*w) - pw))
            if m_back[y:y+ph, x:x+pw].mean() > .2 and y + ph < h:
                pastes.append(dict(seed=k, y=y, x=x, patch=patch, line=i))
                break
    return pastes, len(sources)


def score(state, params, pastes, bleed_mask, split):
    """split: 0 = tune (even), 1 = test (odd), None = all."""
    base = ro.restore_overlap(state, params)
    p0 = base['pencil']
    sel = bleed_mask >= 0 if split is None else (bleed_mask >= 0) & (bleed_mask % 2 == split)
    leak_px = float((p0[sel] > VISIBLE).mean())
    leak_density = float(p0[sel].sum() / max(state['d_front'][sel].clip(0).sum(), 1e-6))
    clean = state['clean_pencil'] >= 0 if split is None else state['clean_pencil'] % 2 == split
    clean &= state['clean_pencil'] >= 0
    gray_in = cv2.cvtColor(state['front'], cv2.COLOR_RGB2GRAY).astype(np.int16)
    gray_out = cv2.cvtColor(base['output'], cv2.COLOR_RGB2GRAY).astype(np.int16)
    altered = float((np.abs(gray_out - gray_in)[clean] > 10).mean())
    use = [p for p in pastes if split is None or p['seed'] % 2 == split]
    d = state['d_front'].copy()
    for p in use:
        ph, pw = p['patch'].shape
        d[p['y']:p['y']+ph, p['x']:p['x']+pw] += p['patch']
    p1 = ro.restore_overlap(state, params, d_front=d)['pencil']
    tp = fp = fn = 0; kept = total = err = n = 0.
    for p in use:
        ph, pw = p['patch'].shape
        sl = (slice(p['y'], p['y']+ph), slice(p['x'], p['x']+pw))
        rec = (p1[sl] - p0[sl]).clip(0)
        truth = p['patch']; inside = state['bleed_footprint'][sl]   # fixed, param-independent
        t_ink, r_ink = (truth > STROKE) & inside, (rec > STROKE) & inside
        tp += (t_ink & r_ink).sum(); fp += (~t_ink & r_ink).sum(); fn += (t_ink & ~r_ink).sum()
        kept += np.minimum(rec, truth)[t_ink].sum(); total += truth[t_ink].sum()
        err += np.abs(rec - truth)[t_ink].sum(); n += t_ink.sum()
    precision, recall = tp/max(tp+fp, 1), tp/max(tp+fn, 1)
    f1 = 2*precision*recall/max(precision+recall, 1e-9)
    out = dict(pencil_f1=f1, pencil_precision=precision, pencil_recall=recall,
                pencil_density_retained=kept/max(total, 1e-9), pencil_density_mae=err/max(n, 1),
                bleed_leak_pixels=leak_px, bleed_leak_density=leak_density,
                front_only_pencil_altered=altered, objective=f1 - leak_px - altered)
    return {**{k: float(v) for k, v in out.items()}, 'pastes': len(use), 'bleed_pixels': int(sel.sum())}


# Hand-set starting point for the search (the first prototype's values).
INITIAL = dict(back_floor=0.25, back_span=1.0, back_sigma=1.5, gain_sigma=8.0, lam0=0.03, lam_ink=0.10,
               lam_edge=0.15, iterations=6, region_dilate=3, hyst_low=0.0, min_area=0, penalty='l1')

SEARCH = dict(back_floor=[.15, .25, .35], back_span=[1.0, 1.8, 2.5, 3.5, 5.0], back_sigma=[0, 1, 1.5, 2.5, 3.5], gain_sigma=[1.5, 2, 3, 5, 8],
              lam0=[0, .03, .06, .1, .15], lam_ink=[0, .1, .2, .35, .5, .75, 1.0], lam_edge=[0, .15, .3, .6],
              region_dilate=[3, 5, 8, 12, 16, 24], hyst_low=[0, .06, .1, .15, .2], min_area=[0, 80, 150, 250, 400, 700], penalty=['l1', 'l0'])


def coordinate_search(state, pastes, bleed_mask, rounds=3):
    params = dict(INITIAL)
    best = score(state, params, pastes, bleed_mask, 0)['objective']
    trace = [dict(params=dict(params), objective=best)]
    for _ in range(rounds):
        improved = False
        for key, values in SEARCH.items():
            for value in values:
                if value == params[key]:
                    continue
                trial = {**params, key: value}
                objective = score(state, trial, pastes, bleed_mask, 0)['objective']
                if objective > best + 1e-4:
                    best, params, improved = objective, trial, True
                    trace.append(dict(params=dict(params), objective=best))
                    print(f'  {key}={value}: tune objective {best:.4f}', flush=True)
        if not improved:
            break
    return params, trace


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sample', type=Path, default=ROOT/'bleed/samples/pencil-sheet08')
    parser.add_argument('--output', type=Path, default=ROOT/'bleed/outputs/pencil-sheet08')
    args = parser.parse_args()
    boxes = json.loads((args.sample/'front.boxes.json').read_text(encoding='utf-8-sig'))
    report, state, result = ro.run(args.sample/'front.jpg', args.sample/'back.jpg', args.output, save=False)
    m_back = ro.back_map(state['d_back'], INITIAL)
    bleed_mask = bleed_only_mask(state, boxes, m_back)
    state['clean_pencil'] = clean_pencil_mask(state, boxes)
    state['bleed_footprint'] = m_back > .1
    pastes, n_sources = make_pastes(state, boxes, result['region'], m_back)
    print(f'{n_sources} clean pencil windows, {len(pastes)} pastes, {int((bleed_mask>=0).sum())} bleed-only pixels')
    print('initial', json.dumps(score(state, INITIAL, pastes, bleed_mask, 0)))
    params, trace = coordinate_search(state, pastes, bleed_mask)
    summary = dict(initial=dict(params=INITIAL, tune=score(state, INITIAL, pastes, bleed_mask, 0),
                                test=score(state, INITIAL, pastes, bleed_mask, 1)),
                   tuned=dict(params=params, tune=score(state, params, pastes, bleed_mask, 0),
                              test=score(state, params, pastes, bleed_mask, 1)),
                   trace=trace, pastes=len(pastes), clean_source_windows=n_sources)
    (args.output/'tuned_params.json').write_text(json.dumps(params, indent=2))
    (args.output/'tuning.json').write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps({k: summary[k] for k in ('initial', 'tuned')}, indent=2, default=float))
    final = ro.restore_overlap(state, params)
    from PIL import Image
    Image.fromarray(final['output']).save(args.output/'recovered.png')
    Image.fromarray(np.uint8(np.clip(255*np.exp(-final['pencil']), 0, 255))).save(args.output/'pencil_density.png')


if __name__ == '__main__':
    main()
