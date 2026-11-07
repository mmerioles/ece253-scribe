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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", nargs="?", type=Path, default=Path("inputs/baseline.png"))
    parser.add_argument("--prepare-only", action="store_true", help="Save line crops without loading OCR")
    args = parser.parse_args()
    image_path = args.image
    references = image_path.with_suffix(".txt").read_text(encoding="utf-8").splitlines()
    boxes = json.loads(image_path.with_suffix(".boxes.json").read_text(encoding="utf-8"))
    if not references or len(references) != len(boxes) or any(not normalize(r) for r in references):
        parser.error("Provide one nonempty reference line per crop box.")
    for box in boxes:
        if (len(box) != 4 or not all(isinstance(v, (int, float)) for v in box)
                or not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1)):
            parser.error("Each box must be [left, top, right, bottom], between 0 and 1.")

    output = Path("outputs") / image_path.stem
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
    if args.prepare_only:
        return

    import torch
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    # Keep CPU inference usable without oversubscribing a desktop's threads.
    torch.set_num_threads(min(4, torch.get_num_threads()))
    print(f"Loading {MODEL} on CPU (first run downloads the model)...", flush=True)
    processor = TrOCRProcessor.from_pretrained(MODEL, revision=REVISION, use_fast=False)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL, revision=REVISION, use_safetensors=True).eval()
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


if __name__ == "__main__":
    main()
