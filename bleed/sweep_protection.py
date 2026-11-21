"""Sweep how strictly stage 3 decides what is a front stroke, on sheet08.

For each setting: TrOCR CER (all / even lines / odd lines) on the real photo, and on the
pasted-pencil photo the pencil recall, bleed left visible and clean pencil altered
(same image-based measures as eval_protection.py).

Acceptance rule (fixed before the sweep): a setting "wins" only if its CER is at least
10 characters (~3.9 points) below the faithful Rowley-Brooke output on BOTH even and odd
lines, while keeping pasted-pencil recall >= 50%. Choose on even lines; report odd.
"""
import json
import pickle
import sys
from itertools import product
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import recover_overlap as ro
import remove_bleed as rb
import tune_overlap as tu
from main import MODEL, REVISION, cer, normalize

SAMPLE = ROOT/'bleed/samples/pencil-sheet08'
CACHE = ROOT/'bleed/outputs/pencil-sheet08/sweep-cache.pkl'


def states():
    if CACHE.exists():
        return pickle.loads(CACHE.read_bytes())
    boxes = json.loads((SAMPLE/'front.boxes.json').read_text(encoding='utf-8-sig'))
    _, st, res = ro.run(SAMPLE/'front.jpg', SAMPLE/'back.jpg', ROOT/'bleed/outputs/pencil-sheet08', save=False)
    m_back = ro.back_map(st['d_back'], tu.INITIAL)
    pastes, _ = tu.make_pastes(st, boxes, res['region'], m_back)
    paste_d = np.zeros(st['d_front'].shape, np.float32)
    for p in pastes:
        ph, pw = p['patch'].shape
        sl = (slice(p['y'], p['y']+ph), slice(p['x'], p['x']+pw))
        paste_d[sl] = np.maximum(paste_d[sl], p['patch'])
    front_p = np.uint8(np.clip(np.round(st['front'].astype(np.float32)*np.exp(-paste_d)[..., None]), 0, 255))
    st_p = ro.prepare(front_p, st['back_aligned'], st['valid'], ro.density(front_p), st['d_back'])
    res_p = ro.restore_overlap(st_p, {})
    data = dict(st=st, res=res, st_p=st_p, res_p=res_p, pastes=pastes, boxes=boxes,
                bleed_mask=tu.bleed_only_mask(st, boxes, m_back), clean=tu.clean_pencil_mask(st, boxes),
                footprint=m_back > .1,
                strokes=ro.front_strokes(st, res), strokes_p=ro.front_strokes(st_p, res_p))
    CACHE.write_bytes(pickle.dumps(data, protocol=4))
    return data


class OCR:
    def __init__(self):
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        torch.set_num_threads(4)
        self.torch = torch
        self.proc = TrOCRProcessor.from_pretrained(MODEL, revision=REVISION, use_fast=False)
        self.model = VisionEncoderDecoderModel.from_pretrained(MODEL, revision=REVISION, use_safetensors=True).eval()
        self.refs = [normalize(r) for r in (SAMPLE/'front.txt').read_text(encoding='utf-8-sig').splitlines()]

    def __call__(self, image, boxes):
        page = Image.fromarray(image); w, h = page.size
        preds = []
        with self.torch.inference_mode():
            for l, t, r, b in boxes:
                crop = page.crop((round(l*w), round(t*h), round(r*w), round(b*h)))
                ids = self.model.generate(self.proc(images=crop, return_tensors='pt').pixel_values,
                                          max_new_tokens=128, num_beams=1, do_sample=False)
                preds.append(normalize(self.proc.batch_decode(ids, skip_special_tokens=True)[0].strip()))
        sub = lambda idx: float(cer([self.refs[i] for i in idx], [preds[i] for i in idx]))
        n = len(self.refs)
        return dict(cer=sub(range(n)), cer_even=sub(range(1, n, 2)), cer_odd=sub(range(0, n, 2)),
                    chars_even=sum(len(self.refs[i]) for i in range(1, n, 2)),
                    chars_odd=sum(len(self.refs[i]) for i in range(0, n, 2)), preds=preds)


def pixel_metrics(data, plain, pasted):
    st = data['st']
    D0, D1 = ro.density(plain), ro.density(pasted)
    D_in, _ = ro.density_rgb(st['front'])
    grid = ro.grid_through(D_in.mean(-1), data['footprint'])
    tp = fn = kept = total = 0.
    for p in data['pastes']:
        ph, pw = p['patch'].shape
        sl = (slice(p['y'], p['y']+ph), slice(p['x'], p['x']+pw))
        rec = (D1[sl] - D0[sl]).clip(0); truth = p['patch']
        t_ink = (truth > tu.STROKE) & data['footprint'][sl]
        tp += (t_ink & (rec > tu.STROKE)).sum(); fn += (t_ink & ~(rec > tu.STROKE)).sum()
        kept += np.minimum(rec, truth)[t_ink].sum(); total += truth[t_ink].sum()
    gin = cv2.cvtColor(st['front'], cv2.COLOR_RGB2GRAY).astype(np.int16)
    gout = cv2.cvtColor(plain, cv2.COLOR_RGB2GRAY).astype(np.int16)
    return dict(pencil_recall=float(tp/max(tp+fn, 1)), pencil_density_retained=float(kept/max(total, 1e-9)),
                bleed_left_visible=float(((D0 - grid) > tu.VISIBLE)[data['bleed_mask'] >= 0].mean()),
                front_only_altered=float((np.abs(gout - gin)[data['clean'] >= 0] > 10).mean()))


def main():
    data = states()
    st, res, st_p, res_p = data['st'], data['res'], data['st_p'], data['res_p']
    ocr = OCR()
    rows = []
    rb_plain, _ = rb.rowley_brooke_restore(st['front'], st['labels'], st['valid'], plate=st['plate'])
    rb_past, _ = rb.rowley_brooke_restore(st_p['front'], st_p['labels'], st_p['valid'], plate=st_p['plate'])
    ref = dict(name='rowley_brooke', **ocr(rb_plain, data['boxes']), **pixel_metrics(data, rb_plain, rb_past))
    rows.append(ref); print(json.dumps({k: v for k, v in ref.items() if k != 'preds'}), flush=True)
    grid = [dict(protection='off')] + [
        dict(protection='divide', stroke_colour=c, ink_tau=t, ink_min_area=a, ink_anchor=an)
        for c, t, (a, an) in product(('divide', 'visible'), (0., .1, .2, .3),
                                     ((0, False), (40, False), (120, False), (0, True), (40, True)))]
    for kw in grid:
        plain = ro.minimal_restore(st, res, strokes=data['strokes'], **kw)['output']
        past = ro.minimal_restore(st_p, res_p, strokes=data['strokes_p'], **kw)['output']
        row = dict(name=json.dumps(kw), **kw, **ocr(plain, data['boxes']), **pixel_metrics(data, plain, past))
        rows.append(row)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items() if k != 'preds'}),
              flush=True)
    out = ROOT/'bleed/results/pencil-sheet08-overlap/protection-sweep.json'
    out.write_text(json.dumps(rows, indent=1))


if __name__ == '__main__':
    main()
