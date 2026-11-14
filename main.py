"""Crop a handwritten page, recognize each line, and score against reference text."""

import argparse
import csv
import json
import unicodedata
from pathlib import Path

from PIL import Image, ImageOps
from jiwer import cer, wer

MODEL = "microsoft/trocr-base-handwritten"
REVISION = "eaacaf452b06415df8f10bb6fad3a4c11e609406"


def normalize(text):
    """Ignore case/punctuation; retain letters, numbers, and word boundaries."""
    text = unicodedata.normalize("NFKC", text).casefold()
    text = "".join(c for c in text if not unicodedata.category(c).startswith("P"))
    return " ".join(text.split())


def read_inputs(image_path):
    references = image_path.with_suffix(".txt").read_text(encoding="utf-8-sig").splitlines()
    boxes = json.loads(image_path.with_suffix(".boxes.json").read_text(encoding="utf-8-sig"))
    if not references or len(references) != len(boxes) or any(not normalize(r) for r in references):
        raise ValueError("Provide one nonempty reference line per crop box.")
    for box in boxes:
        if (not isinstance(box, list) or len(box) != 4
                or not all(isinstance(v, (int, float)) for v in box)
                or not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1)):
            raise ValueError("Boxes must be [left, top, right, bottom], between 0 and 1.")
    return references, boxes


def evaluate(image_path, references, boxes, processor=None, model=None, output=None):
    if output is None:
        output = image_path.parent.parent / "outputs" / image_path.stem
    output.mkdir(parents=True, exist_ok=True)
    page = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    width, height = page.size
    crops = []
    for i, (left, top, right, bottom) in enumerate(boxes, 1):
        crop = page.crop((round(left * width), round(top * height),
                          round(right * width), round(bottom * height)))
        crop.save(output / f"line-{i:02}.png")
        crops.append(crop)
    print(f"Saved {len(crops)} line crops in {output}", flush=True)
    if model is None:
        return

    import torch

    predictions = []
    with torch.inference_mode():
        for i, crop in enumerate(crops, 1):
            pixels = processor(images=crop, return_tensors="pt").pixel_values
            ids = model.generate(pixels, max_new_tokens=128, num_beams=1, do_sample=False)
            prediction = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
            predictions.append(prediction)
            print(f"{i:02}. {prediction}", flush=True)

    normalized_refs = [normalize(r) for r in references]
    normalized_preds = [normalize(p) for p in predictions]
    scores = {
        "model": MODEL,
        "model_revision": getattr(model.config, "_commit_hash", None),
        "image": str(image_path),
        "line_count": len(references),
        "strict_cer": cer(references, predictions),
        "strict_wer": wer(references, predictions),
        "normalized_cer": cer(normalized_refs, normalized_preds),
        "normalized_wer": wer(normalized_refs, normalized_preds),
        "normalized_exact_lines": sum(a == b for a, b in zip(normalized_refs, normalized_preds)),
        "normalization": "NFKC, lowercase, remove Unicode punctuation, collapse whitespace",
        "decoding": {"num_beams": 1, "do_sample": False, "max_new_tokens": 128},
    }
    (output / "predictions.txt").write_text("\n".join(predictions) + "\n", encoding="utf-8")
    (output / "metrics.json").write_text(json.dumps(scores, indent=2), encoding="utf-8")
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["line", "reference", "prediction", "normalized_cer", "normalized_wer"])
        for i, (ref, pred) in enumerate(zip(references, predictions), 1):
            writer.writerow([i, ref, pred, cer(normalize(ref), normalize(pred)),
                             wer(normalize(ref), normalize(pred))])
    print(f"\nNormalized CER: {scores['normalized_cer']:.2%} | WER: {scores['normalized_wer']:.2%}")
    print(f"Strict CER: {scores['strict_cer']:.2%} | WER: {scores['strict_wer']:.2%}")
    print(f"Exact lines (normalized): {scores['normalized_exact_lines']}/{len(references)}")
    print(f"Saved predictions, comparison.csv, and metrics.json in {output}")
    return references, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", choices=["bleed", "blur", "shadow"])
    parser.add_argument("--prepare-only", action="store_true", help="Save crops without loading OCR")
    args = parser.parse_args()
    folder = Path(__file__).resolve().parent / args.artifact
    images = sorted(p for p in (folder / "inputs").rglob("*")
                    if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"})
    if not images:
        print(f"No images in {folder / 'inputs'}. Add a photo, matching .txt, and .boxes.json.")
        return
    if len({p.relative_to(folder / "inputs").with_suffix("").as_posix().casefold() for p in images}) != len(images):
        parser.error("Use unique image names: photo.png and photo.jpg would share outputs.")
    jobs = []
    for image_path in images:
        try:
            references, boxes = read_inputs(image_path)
        except (OSError, ValueError, TypeError) as error:
            parser.error(f"{image_path.name}: {error}")
        jobs.append((image_path, references, boxes))

    processor = model = None
    if not args.prepare_only:
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        torch.set_num_threads(min(4, torch.get_num_threads()))
        print(f"Loading {MODEL} once for {len(images)} image(s)...", flush=True)
        processor = TrOCRProcessor.from_pretrained(MODEL, revision=REVISION, use_fast=False)
        model = VisionEncoderDecoderModel.from_pretrained(
            MODEL, revision=REVISION, use_safetensors=True).eval()

    all_refs, all_preds = [], []
    for image_path, references, boxes in jobs:
        print(f"\n--- {image_path.relative_to(folder / 'inputs')} ---", flush=True)
        output = folder / "outputs" / image_path.relative_to(folder / "inputs").with_suffix("")
        result = evaluate(image_path, references, boxes, processor, model, output)
        if result:
            all_refs.extend(result[0])
            all_preds.extend(result[1])
    if args.prepare_only:
        return
    refs, preds = list(map(normalize, all_refs)), list(map(normalize, all_preds))
    summary = {
        "artifact": args.artifact,
        "model": MODEL,
        "model_revision": REVISION,
        "images": [p.relative_to(folder / "inputs").as_posix() for p in images],
        "image_count": len(images),
        "line_count": len(all_refs),
        "strict_cer": cer(all_refs, all_preds),
        "strict_wer": wer(all_refs, all_preds),
        "normalized_cer": cer(refs, preds),
        "normalized_wer": wer(refs, preds),
        "normalized_exact_lines": sum(a == b for a, b in zip(refs, preds)),
    }
    (folder / "outputs" / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSET TOTAL: {len(images)} image(s), {len(all_refs)} lines")
    print(f"Normalized CER: {summary['normalized_cer']:.2%} | WER: {summary['normalized_wer']:.2%}")


if __name__ == "__main__":
    main()
