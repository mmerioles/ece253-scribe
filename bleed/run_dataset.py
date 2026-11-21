"""Run the existing paired experiment on every foreground capture."""
import argparse
import contextlib
import json
import html
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bleed.remove_bleed import run
from main import MODEL, REVISION, evaluate, normalize, read_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ocr', action='store_true', help='Score all four variants (loads OCR once)')
    args = parser.parse_args()
    output = ROOT/'bleed/outputs'
    output.mkdir(exist_ok=True)
    report = {'captures': [], 'failures': [], 'scores': {}}
    jobs = []
    for front in sorted((ROOT/'bleed/inputs').glob('sheet*/front-*.jpg')):
        key = f'{front.parent.name}/{front.stem}'
        target = output/key
        target.mkdir(parents=True, exist_ok=True)
        print(f'Processing {key}', flush=True)
        try:
            refs, boxes = read_inputs(front)
            with (target/'processing.log').open('w') as log, contextlib.redirect_stdout(log):
                result = run(front, front.parent/'back.jpg', target)
            report['captures'].append({'capture': key, **result})
            jobs.append((target, refs, boxes))
            print(f"Done {key}: {result['elapsed_seconds']:.1f}s", flush=True)
        except Exception as error:
            report['failures'].append({'capture': key, 'error': str(error)})
            print(f'FAILED {key}: {error}', flush=True)
        (output/'dataset-run.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    cards = ['<!doctype html><meta charset="utf-8"><title>Bleed batch review</title><style>body{font:16px system-ui;margin:24px;background:#eee}img{width:100%;max-width:1800px}section{margin:30px 0}</style><h1>Bleed batch review</h1><p>Left: input. Middle: conservative filter. Right: destructive strong removal. Missing strokes remain; these are not clean ground truth.</p>']
    for item in report['captures']:
        key = item['capture']
        cards.append(f'<section><h2>{html.escape(key)}</h2><a href="{key}/filtered.png">Full-resolution filtered image</a><br><img loading="lazy" src="{key}/comparison.png"></section>')
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
        totals = {name: ([], []) for name in ['input', 'without_protection', 'filtered', 'strong_removal']}
        for target, refs, boxes in jobs:
            summary = {}
            for name, (all_refs, all_preds) in totals.items():
                directory = target/'ocr'/name
                print(f'OCR {target.parent.name}/{target.name}/{name}', flush=True)
                with (target/f'ocr-{name}.log').open('w') as log, contextlib.redirect_stdout(log):
                    _, preds = evaluate(target/f'{name}.png', refs, boxes, processor, model, directory)
                all_refs.extend(refs); all_preds.extend(preds)
                summary[name] = json.loads((directory/'metrics.json').read_text())
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
