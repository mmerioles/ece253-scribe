#!/usr/bin/env python
"""Restore real blurred phone photos with the classical blur module and write
them next to the originals as extra OCR inputs (``<stem>__l0.jpg`` + copies of
the ``.txt`` / ``.boxes.json``), so the team's ``main.py blur`` scores
"no preprocessing" and "L0 blind deblur" on identical line crops.

Pipeline per photo (grayscale, [0,1]):
  1. downscale to ``--work`` px long side for kernel estimation
  2. pick the highest-ink patch (our modification: estimate k on a patch, not the
     mostly-blank page); blind L0 kernel estimation (Pan et al. 2014) with
     ``pencil_adapt``; kernel support ``--ksize`` at work resolution
  3. upsample the kernel to the output resolution and run the non-blind
     hyper-Laplacian deconvolution on the full page at ``--out_long`` px
  4. also write ``<stem>__ctrl.jpg``: the same photo merely resized to
     ``--out_long`` (the fair "without restoration" control at equal resolution)

Writes a CSV with the LAPV sharpness before/after and the estimated kernel's
support size. These are sanity numbers, not results; OCR CER from main.py is
the result.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scribe_blur import detect, l0_deblur, nonblind, patch, psf  # noqa: E402


def page_interior(g: np.ndarray, inset: float = 0.08):
    """(y0, x0, y1, x1) of the page interior: bright page on a dark desk,
    corners from the team's make_boxes.page_corners, inset by ``inset`` of the
    page size so the page edge (huge gradient, no text) can never be picked
    as the kernel-estimation patch. Falls back to the whole image."""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "mmerioles" / "ece253-scribe" / "blur"))
        from make_boxes import page_corners  # type: ignore
        c = page_corners(g * 255.0)
        x0, y0 = c.min(axis=0); x1, y1 = c.max(axis=0)
        dx, dy = inset * (x1 - x0), inset * (y1 - y0)
        return int(y0 + dy), int(x0 + dx), int(y1 - dy), int(x1 - dx)
    except Exception:
        h, w = g.shape
        return 0, 0, h, w


def load_gray(path: Path) -> np.ndarray:
    """EXIF-rotated grayscale float image in [0,1] (same orientation main.py uses)."""
    from PIL import Image, ImageOps
    im = ImageOps.exif_transpose(Image.open(path)).convert("L")
    return np.asarray(im, dtype=np.float32) / 255.0


def resize_long(img: np.ndarray, long_side: int) -> np.ndarray:
    h, w = img.shape
    s = long_side / max(h, w)
    if s >= 1:
        return img
    return cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)


def upsample_kernel(k: np.ndarray, factor: float) -> np.ndarray:
    n = int(round(k.shape[0] * factor)) | 1
    ku = cv2.resize(k.astype(np.float32), (n, n), interpolation=cv2.INTER_CUBIC)
    ku = np.clip(ku, 0, None)
    return psf.center_kernel(ku / ku.sum())


def restore_one(path: Path, args, rng) -> dict:
    t0 = time.time()
    g_full = load_gray(path)
    g_out = resize_long(g_full, args.out_long)
    g_work = resize_long(g_full, args.work)
    lapv_in = detect.variance_of_laplacian_normalized(g_out)

    iy0, ix0, iy1, ix1 = page_interior(g_work)
    inner = g_work[iy0:iy1, ix0:ix1]
    by, bx, bh, bw = patch.select_high_ink_patch(inner, args.patch, n=1)[0]
    boxes = [(by + iy0, bx + ix0, bh, bw)]
    k_work = l0_deblur.estimate_kernel(g_work, args.ksize, patch=boxes[0], pencil_adapt=True,
                                       levels=args.levels, iters=args.iters, beta_g_max=1e3)
    factor = max(g_out.shape) / max(g_work.shape)
    k_out = upsample_kernel(k_work, factor) if factor > 1.01 else k_work
    x_hat = nonblind.hyper_laplacian_deconv(g_out, k_out, lam=args.lam)
    x_hat = np.clip(x_hat, 0, 1)
    lapv_out = detect.variance_of_laplacian_normalized(x_hat)

    stem = path.stem
    out_l0 = path.with_name(f"{stem}__l0.jpg")
    out_ctrl = path.with_name(f"{stem}__ctrl.jpg")
    cv2.imwrite(str(out_l0), (x_hat * 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(str(out_ctrl), (g_out * 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 95])
    for suf in (".txt", ".boxes.json"):
        src = path.with_suffix(suf)
        if src.exists():
            shutil.copy(src, out_l0.with_suffix(suf))
            shutil.copy(src, out_ctrl.with_suffix(suf))
    # kernel image for the slide / report
    kimg = cv2.resize((k_out / k_out.max() * 255).astype(np.uint8), (128, 128), interpolation=cv2.INTER_NEAREST)
    kdir = args.kernel_dir
    kdir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(kdir / f"{stem}_kernel.png"), kimg)
    nz = k_out > 0.02 * k_out.max()
    ys, xs = np.nonzero(nz)
    support = int(max(ys.max() - ys.min(), xs.max() - xs.min()) + 1) if ys.size else 0
    row = dict(photo=path.name, out_size=f"{g_out.shape[1]}x{g_out.shape[0]}", patch=boxes[0],
               lapv_in=round(lapv_in, 1), lapv_l0=round(lapv_out, 1), lapv_gain=round(lapv_out / max(lapv_in, 1e-6), 2),
               kernel_support_px=support, seconds=round(time.time() - t0, 1))
    print(" ".join(f"{k}={v}" for k, v in row.items()), flush=True)
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("photos", nargs="+", type=Path)
    ap.add_argument("--work", type=int, default=1400, help="long side for kernel estimation")
    ap.add_argument("--out_long", type=int, default=2800, help="long side of the restored output (and control)")
    ap.add_argument("--ksize", type=int, default=25, help="kernel support at work resolution (odd)")
    ap.add_argument("--patch", type=int, default=256)
    ap.add_argument("--levels", type=int, default=4)
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--lam", type=float, default=4e-3)
    ap.add_argument("--kernel_dir", type=Path, default=Path("outputs/kernels"))
    ap.add_argument("--csv", type=Path, default=Path("outputs/restore_log.csv"))
    args = ap.parse_args(argv)
    rng = np.random.default_rng(0)
    rows = []
    for p in args.photos:
        try:
            rows.append(restore_one(p, args, rng))
        except Exception as e:  # keep going; report at the end
            print(f"FAILED {p.name}: {type(e).__name__}: {e}", flush=True)
            rows.append(dict(photo=p.name, error=f"{type(e).__name__}: {e}"))
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with args.csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
