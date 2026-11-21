"""Image-based ablation of stroke protection on the real pencil-sheet08 pair.

Ladder, all on the same dense registration, Eq. 1 labels and Sec. 3.3.1 plate:
  rowley_brooke  paper Sec. 3.3: replace blfg pixels with the exemplar plate, blend;
  off            our identification (labels + decomposition), every touched pixel -> plate;
  on             + the paper's own rule applied to our mask: detected strokes left unchanged;
  divide         + our protection: strokes keep their ink, bleed share divided out (default).

Truths (none come from our decomposition; they use line boxes and the aligned back photo):
  * pasted pencil: real clean pencil patches (from bleed-free parts of the boxed lines) are
    multiplied into the FRONT PHOTO over real bleed in other lines; the whole pipeline is
    re-run on the pasted photo, and recovered pencil = D(out_pasted) - D(out_plain), measured
    from the OUTPUT IMAGES, on truth ink pixels lying over reverse ink (back density > .1).
  * bleed-only zones right of every pencil line: output darker than the input's continued
    grid + 0.07 OD counts as bleed left visible.
  * front-only pencil (>=5 px from any reverse ink): fraction changed by > 10 grey levels.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import recover_overlap as ro
import remove_bleed as rb
import tune_overlap as tu

ROOT = Path(__file__).resolve().parents[1]
MODES = ('rowley_brooke', 'off', 'on', 'divide')


def outputs(st, res):
    rb_out, _ = rb.rowley_brooke_restore(st['front'], st['labels'], st['valid'], plate=st['plate'])
    strokes = ro.front_strokes(st, res)
    out = dict(rowley_brooke=rb_out)
    for m in ('off', 'on', 'divide'):
        out[m] = ro.minimal_restore(st, res, protection=m, strokes=strokes)['output']
    return out


def gray_density(img):
    return ro.density(img) if img.ndim == 3 else None


def main():
    sample = ROOT/'bleed/samples/pencil-sheet08'
    out_dir = ROOT/'bleed/outputs/pencil-sheet08'
    boxes = json.loads((sample/'front.boxes.json').read_text(encoding='utf-8-sig'))
    _, st, res = ro.run(sample/'front.jpg', sample/'back.jpg', out_dir, save=False)
    m_back = ro.back_map(st['d_back'], tu.INITIAL)
    footprint = m_back > .1
    bleed_mask = tu.bleed_only_mask(st, boxes, m_back)
    clean = tu.clean_pencil_mask(st, boxes)
    pastes, n_src = tu.make_pastes(st, boxes, res['region'], m_back)
    print(f'{len(pastes)} pastes from {n_src} windows; {int((bleed_mask>=0).sum())} bleed-only px', flush=True)

    plain = outputs(st, res)
    paste_d = np.zeros(st['d_front'].shape, np.float32)
    for p in pastes:
        ph, pw = p['patch'].shape
        sl = (slice(p['y'], p['y']+ph), slice(p['x'], p['x']+pw))
        paste_d[sl] = np.maximum(paste_d[sl], p['patch'])
    front_p = np.uint8(np.clip(np.round(st['front'].astype(np.float32)*np.exp(-paste_d)[..., None]), 0, 255))
    st_p = ro.prepare(front_p, st['back_aligned'], st['valid'], ro.density(front_p), st['d_back'])
    res_p = ro.restore_overlap(st_p, {})
    pasted = outputs(st_p, res_p)

    D_in, _ = ro.density_rgb(st['front'])
    grid = ro.grid_through(D_in.mean(-1), footprint)   # input's printed grid continued under the reverse ink
    gin = cv2.cvtColor(st['front'], cv2.COLOR_RGB2GRAY).astype(np.int16)
    report = {}
    for m in MODES:
        D0, D1 = ro.density(plain[m]), ro.density(pasted[m])
        row = {}
        for split, name in ((0, 'tune'), (1, 'test'), (None, 'all')):
            use = [p for p in pastes if split is None or p['seed'] % 2 == split]
            kept = total = tp = fn = fp = 0.
            for p in use:
                ph, pw = p['patch'].shape
                sl = (slice(p['y'], p['y']+ph), slice(p['x'], p['x']+pw))
                rec = (D1[sl] - D0[sl]).clip(0); truth = p['patch']
                t_ink = (truth > tu.STROKE) & footprint[sl]
                r_ink = (rec > tu.STROKE) & footprint[sl]
                tp += (t_ink & r_ink).sum(); fn += (t_ink & ~r_ink).sum(); fp += (~t_ink & r_ink).sum()
                kept += np.minimum(rec, truth)[t_ink].sum(); total += truth[t_ink].sum()
            sel = bleed_mask >= 0 if split is None else (bleed_mask >= 0) & (bleed_mask % 2 == split)
            csel = clean >= 0 if split is None else (clean >= 0) & (clean % 2 == split)
            gout = cv2.cvtColor(plain[m], cv2.COLOR_RGB2GRAY).astype(np.int16)
            prec, rec_ = tp/max(tp+fp, 1), tp/max(tp+fn, 1)
            row[name] = dict(pencil_recall=float(rec_), pencil_precision=float(prec),
                             pencil_f1=float(2*prec*rec_/max(prec+rec_, 1e-9)),
                             pencil_density_retained=float(kept/max(total, 1e-9)),
                             bleed_left_visible=float(((D0 - grid) > tu.VISIBLE)[sel].mean()),
                             front_only_altered=float((np.abs(gout - gin)[csel] > 10).mean()))
        report[m] = row
        print(m, json.dumps({k: round(v, 3) for k, v in row['all'].items()}), flush=True)
    res_dir = ROOT/'bleed/results/pencil-sheet08-overlap'
    (res_dir/'protection-ablation.json').write_text(json.dumps(dict(pastes=len(pastes), modes=report), indent=2))
    # Shared-state Rowley-Brooke output replaces the old SIFT-only one used for OCR.
    Image.fromarray(plain['rowley_brooke']).save(out_dir/'rowley_brooke.png')
    for m in MODES:   # pasted-pencil outputs, for visual checks
        Image.fromarray(pasted[m]).save(out_dir/f'pasted_{m}.png')
    Image.fromarray(front_p).save(out_dir/'pasted_input.png')


if __name__ == '__main__':
    main()
