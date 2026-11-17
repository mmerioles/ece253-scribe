"""Score the same manually annotated pencil lines before/after restoration."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from main import MODEL, REVISION, evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample', type=Path, default=ROOT/'bleed/samples/pencil-sheet08')
    parser.add_argument('--output', type=Path, default=ROOT/'bleed/outputs/pencil-sheet08')
    args = parser.parse_args()
    import torch
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    torch.set_num_threads(4)
    processor = TrOCRProcessor.from_pretrained(MODEL,revision=REVISION,use_fast=False)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL,revision=REVISION,use_safetensors=True).eval()
    refs = (args.sample/'front.txt').read_text(encoding='utf-8-sig').splitlines()
    boxes = json.loads((args.sample/'front.boxes.json').read_text(encoding='utf-8-sig'))
    summary = {}
    for name in ['input','without_protection','filtered','strong_removal']:
        directory = args.output/'ocr'/name
        evaluate(args.output/f'{name}.png',refs,boxes,processor,model,directory)
        summary[name] = json.loads((directory/'metrics.json').read_text())
    (args.output/'ocr-summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')


if __name__ == '__main__':
    main()
