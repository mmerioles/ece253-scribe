"""Run the existing paired experiment on every foreground capture."""
import argparse
import contextlib
import json
import html
import hashlib
import cv2
from PIL import Image
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'bleed'))
from bleed.remove_bleed import run
from main import MODEL, REVISION, evaluate, normalize, read_inputs


def process_capture(job):
    front, output, method = job
    key = f'{front.parent.name}/{front.stem}'
    target = output/key
    target.mkdir(parents=True, exist_ok=True)
    print(f'Processing {key}', flush=True)
    try:
        refs, boxes = read_inputs(front)
        with (target/'processing.log').open('w') as log, contextlib.redirect_stdout(log):
            cv2.setRNGSeed(0)
            if method == 'improved':
                from recover_overlap import run as recover
                result, state, recovered = recover(front, front.parent/'back.jpg', target, save='minimal')
                del state, recovered
                thumbs = []
                for name in ['input','rowley_brooke','minimal']:
                    with Image.open(target/f'{name}.png') as im:
                        thumbs.append(im.resize((600,800)))
                preview = Image.new('RGB',(1800,800))
                for i,im in enumerate(thumbs): preview.paste(im,(600*i,0))
                preview.save(target/'comparison.png')
            else:
                result = run(front, front.parent/'back.jpg', target)
            result['front_sha256'] = hashlib.sha256(front.read_bytes()).hexdigest()
            result['back_sha256'] = hashlib.sha256((front.parent/'back.jpg').read_bytes()).hexdigest()
        return {'capture': key, **result}, None
    except Exception as error:
        return None, {'capture': key, 'error': str(error)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ocr', action='store_true', help='Score all four variants (loads OCR once)')
    parser.add_argument('--method', choices=['legacy', 'improved'], default='improved')
    parser.add_argument('--workers', type=int, default=1, choices=range(1,5))
    parser.add_argument('--resume', action='store_true', help='Reuse completed restoration only when source and input hashes match')
    args = parser.parse_args()
    variants = ['input', 'rowley_brooke', 'minimal_protection_off', 'minimal'] if args.method == 'improved' else ['input', 'without_protection', 'filtered', 'strong_removal']
    output = ROOT/'bleed/outputs'
    if args.method == 'improved':
        output = output/'improved'
    output.mkdir(parents=True, exist_ok=True)
    report = {'method': args.method, 'captures': [], 'failures': [], 'scores': {},
              'source_sha256': {name: hashlib.sha256((ROOT/'bleed'/name).read_bytes()).hexdigest() for name in ['recover_overlap.py','remove_bleed.py']}}
    jobs = []
    fronts = sorted((ROOT/'bleed/inputs').glob('sheet*/front-*.jpg'))
    pending = []
    previous = {}
    manifest = output/'dataset-run.json'
    if args.resume and manifest.exists():
        saved = json.loads(manifest.read_text(encoding='utf-8'))
        if saved.get('source_sha256') == report['source_sha256'] and saved.get('method') == args.method:
            previous = {item['capture']: item for item in saved['captures']}
    for front in fronts:
        key = f'{front.parent.name}/{front.stem}'
        cached = previous.get(key)
        valid_cache = cached and all((output/key/f'{v}.png').exists() for v in variants)
        if valid_cache:
            valid_cache = cached['front_sha256'] == hashlib.sha256(front.read_bytes()).hexdigest() and cached['back_sha256'] == hashlib.sha256((front.parent/'back.jpg').read_bytes()).hexdigest()
        if valid_cache:
            report['captures'].append(cached)
            print(f'Reusing completed {key}', flush=True)
        else:
            pending.append((front, output, args.method))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result, error in pool.map(process_capture, pending):
            if error:
                report['failures'].append(error)
                print(f"FAILED {error['capture']}: {error['error']}", flush=True)
            else:
                report['captures'].append(result)
                print(f"Done {result['capture']}: {result['elapsed_seconds']:.1f}s", flush=True)
            manifest.write_text(json.dumps(report, indent=2), encoding='utf-8')
    report['captures'].sort(key=lambda item: item['capture'])
    for item in report['captures']:
        front = ROOT/'bleed/inputs'/f"{item['capture']}.jpg"
        refs, boxes = read_inputs(front)
        jobs.append((output/item['capture'], refs, boxes))
    final_image = 'minimal' if args.method == 'improved' else 'filtered'
    caption = 'Input / Rowley-Brooke baseline on shared registration / improved method' if args.method == 'improved' else 'Input / conservative / strong removal'
    cards = ['<!doctype html><meta charset="utf-8"><title>Bleed batch review</title><style>body{font:16px system-ui;margin:24px;background:#eee}img{width:100%;max-width:1800px}section{margin:30px 0}</style><h1>Bleed batch review</h1><p>See comparison order below. Missing strokes remain; these are not clean ground truth.</p>']
    cards.append(f'<p>{caption}</p>')
    for item in report['captures']:
        key = item['capture']
        cards.append(f'<section><h2>{html.escape(key)}</h2><a href="{key}/{final_image}.png">Full-resolution filtered image</a><br><img loading="lazy" src="{key}/comparison.png"></section>')
    for item in report['failures']:
        cards.append(f"<p>{html.escape(item['capture'])}: FAILED - {html.escape(item['error'])}</p>")
    (output/'review.html').write_text('\n'.join(cards), encoding='utf-8')
    if args.ocr and jobs:
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        from jiwer import cer, wer
        torch.set_num_threads(4)
        processor = TrOCRProcessor.from_pretrained(MODEL, revision=REVISION, use_fast=False)
        model = VisionEncoderDecoderModel.from_pretrained(MODEL, revision=REVISION, use_safetensors=True).eval()
        totals = {name: ([], []) for name in variants}
        for target, refs, boxes in jobs:
            summary = {}
            for name, (all_refs, all_preds) in totals.items():
                directory = target/'ocr'/name
                print(f'OCR {target.parent.name}/{target.name}/{name}', flush=True)
                with (target/f'ocr-{name}.log').open('w') as log, contextlib.redirect_stdout(log):
                    _, preds = evaluate(target/f'{name}.png', refs, boxes, processor, model, directory)
                all_refs.extend(refs); all_preds.extend(preds)
                summary[name] = json.loads((directory/'metrics.json').read_text())
            print(f'Completed OCR {target.parent.name}/{target.name}', flush=True)
            (target/'ocr-summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        for name, (refs, preds) in totals.items():
            a, b = list(map(normalize, refs)), list(map(normalize, preds))
            report['scores'][name] = {'line_count': len(a), 'normalized_cer': cer(a,b), 'normalized_wer': wer(a,b), 'exact_lines': sum(x==y for x,y in zip(a,b))}
        report['model'] = MODEL
        report['revision'] = REVISION
    (output/'dataset-run.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'completed': len(jobs), 'failures': report['failures'], 'scores': report['scores']}, indent=2))


if __name__ == '__main__':
    main()
