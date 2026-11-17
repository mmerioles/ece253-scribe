"""Generate <photo>.boxes.json line boxes for photos of the same handwritten page.

Why: main.py needs one [left, top, right, bottom] box (fractions of the image)
per text line for every input photo, and a page photographed four times (clean,
two motion-blur levels, defocus) is framed slightly differently each time.
Drawing 40 box files by hand is error-prone, so this script

  1. finds the paper's four corners in each photo (bright page on a dark desk),
  2. takes the line layout ONCE from the sharp reference photo (row ink profile
     on the rectified page; the leading list numbers are excluded, as in
     bleed/inputs/baseline.boxes.json),
  3. maps that layout through each photo's page homography and writes the
     axis-aligned box in that photo's own fractional coordinates.

Because only the page outline is used (no feature matching), it works on photos
far too blurred for ORB/SIFT. It needs numpy + Pillow only (both already in
the project environment).

Usage (from the repo root):

    uv run blur/make_boxes.py blur/inputs/p06_clean.jpg \
        blur/inputs/p06_motion_mild.jpg blur/inputs/p06_motion_severe.jpg \
        blur/inputs/p06_defocus.jpg --lines 10

The first photo is the sharp reference. Previews with the boxes drawn are saved
to blur/outputs/boxes_preview/<photo>.jpg -- always look at them before running
OCR. Options: --keep-numbers to include the "1." list numbers in the boxes,
--pad to change the margin around each line (fraction of page height).

Assumptions: dark background around the page (black desk), the whole page in
frame, lines written left to right with clear gaps between lines. If corner
detection fails for a photo the script says so and skips it.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

PAGE_W, PAGE_H = 850, 1100  # working resolution of the rectified page (letter aspect)


# ----------------------------------------------------------------------------
# geometry helpers (numpy only)
# ----------------------------------------------------------------------------
def homography(src, dst):
    """Direct linear transform: 3x3 H with dst ~ H @ src for four point pairs."""
    A = []
    for (x, y), (u, v) in zip(src, dst):
        A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, _, vt = np.linalg.svd(np.asarray(A, dtype=np.float64))
    H = vt[-1].reshape(3, 3)
    return H / H[2, 2]


def apply_h(H, pts):
    pts = np.asarray(pts, dtype=np.float64)
    p = np.c_[pts, np.ones(len(pts))] @ H.T
    return p[:, :2] / p[:, 2:3]


def warp_to_page(gray, H_page_to_photo):
    """Rectify the page into a PAGE_W x PAGE_H array by inverse mapping (nearest)."""
    ys, xs = np.mgrid[0:PAGE_H, 0:PAGE_W]
    pts = np.c_[xs.ravel(), ys.ravel()]
    src = apply_h(H_page_to_photo, pts)
    sx = np.clip(np.rint(src[:, 0]).astype(int), 0, gray.shape[1] - 1)
    sy = np.clip(np.rint(src[:, 1]).astype(int), 0, gray.shape[0] - 1)
    return gray[sy, sx].reshape(PAGE_H, PAGE_W)


# ----------------------------------------------------------------------------
# page corner detection
# ----------------------------------------------------------------------------
def load_gray(path, long_side=1200):
    img = ImageOps.exif_transpose(Image.open(path))  # same orientation main.py uses
    W, H = img.size
    s = long_side / max(W, H)
    small = img.convert("L").resize((round(W * s), round(H * s)), Image.BILINEAR)
    return np.asarray(small, dtype=np.float32), (W, H), s


def largest_component(mask):
    """Flood-fill based largest 4-connected component (numpy, no scipy)."""
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    best, best_n, cur = 0, 0, 0
    for y0, x0 in zip(*np.nonzero(mask)):
        if labels[y0, x0]:
            continue
        cur += 1
        stack = [(y0, x0)]
        labels[y0, x0] = cur
        n = 0
        while stack:
            y, x = stack.pop()
            n += 1
            for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not labels[yy, xx]:
                    labels[yy, xx] = cur
                    stack.append((yy, xx))
        if n > best_n:
            best, best_n = cur, n
    return labels == best


def page_corners(gray):
    """Corners (TL, TR, BR, BL) of the brightest large blob, in small-image pixels."""
    # bright page vs dark desk: threshold halfway between the dark and bright modes
    lo, hi = np.percentile(gray, 5), np.percentile(gray, 95)
    thr = lo + 0.65 * (hi - lo)
    mask = gray > thr
    # work at a coarser grid for the flood fill, then refine corners at full small-res
    step = 4
    coarse = largest_component(mask[::step, ::step])
    comp = np.kron(coarse, np.ones((step, step), dtype=bool))[: mask.shape[0], : mask.shape[1]] & mask
    if comp.sum() < 0.2 * mask.size:
        raise ValueError("page blob too small -- is the whole page in frame on a dark background?")
    # Fit each page edge as a straight line through the blob's outline, with
    # outlier rejection, so reflections or shadows stuck to the page (which
    # would drag an extreme-point corner away) are ignored.
    ys, xs = np.nonzero(comp)
    y_lo, y_hi = np.percentile(ys, 10), np.percentile(ys, 90)
    x_lo, x_hi = np.percentile(xs, 10), np.percentile(xs, 90)
    rows = np.arange(int(y_lo), int(y_hi))
    left = np.array([np.argmax(comp[y]) for y in rows], dtype=float)                 # x = f(y)
    right = np.array([comp.shape[1] - 1 - np.argmax(comp[y, ::-1]) for y in rows], dtype=float)
    cols = np.arange(int(x_lo), int(x_hi))
    top = np.array([np.argmax(comp[:, x]) for x in cols], dtype=float)               # y = f(x)
    bottom = np.array([comp.shape[0] - 1 - np.argmax(comp[::-1, x]) for x in cols], dtype=float)

    def robust_fit(t, v):
        """v = a*t + b with two rounds of residual trimming (MAD-based)."""
        a, b = np.polyfit(t, v, 1)
        for _ in range(2):
            r = v - (a * t + b)
            mad = np.median(np.abs(r - np.median(r))) + 1e-6
            keep = np.abs(r - np.median(r)) < 3 * 1.4826 * mad
            if keep.sum() < 10:
                break
            a, b = np.polyfit(t[keep], v[keep], 1)
        return a, b

    aL, bL = robust_fit(rows, left)      # x = aL*y + bL
    aR, bR = robust_fit(rows, right)
    aT, bT = robust_fit(cols, top)       # y = aT*x + bT
    aB, bB = robust_fit(cols, bottom)

    def meet(ax, bx, ay, by):
        # x = ax*y + bx  and  y = ay*x + by
        x = (ax * by + bx) / (1 - ax * ay)
        return [x, ay * x + by]

    tl, tr = meet(aL, bL, aT, bT), meet(aR, bR, aT, bT)
    br, bl = meet(aR, bR, aB, bB), meet(aL, bL, aB, bB)
    return np.array([tl, tr, br, bl], dtype=np.float64)


# ----------------------------------------------------------------------------
# line layout from the reference page
# ----------------------------------------------------------------------------
def runs(binary):
    """(start, end) index pairs of True runs in a 1-D boolean array."""
    idx = np.flatnonzero(np.diff(np.r_[False, binary, False]))
    return list(zip(idx[::2], idx[1::2]))


def line_layout(page, n_lines, keep_numbers, pad):
    """Return n_lines boxes [x0, y0, x1, y1] in rectified-page pixels."""
    inner = page[int(0.03 * PAGE_H): int(0.97 * PAGE_H), int(0.03 * PAGE_W): int(0.97 * PAGE_W)]
    paper = np.median(inner)
    ink = inner < paper - 0.35 * (paper - inner.min())
    # a row that is mostly "ink" is the desk showing past a slightly-off page
    # edge (or a ruled line), never handwriting -- drop it
    ink[ink.mean(axis=1) > 0.5] = False
    # likewise columns (left/right edge of the desk)
    ink[:, ink.mean(axis=0) > 0.5] = False
    row = ink.mean(axis=1)
    k = 7
    row = np.convolve(row, np.ones(k) / k, mode="same")
    bands = [(a, b) for a, b in runs(row > 0.004) if b - a >= 8]

    def n_word_clusters(a, b):
        cols = ink[a:b].mean(axis=0) > 0.02
        return sum(1 for c0, c1 in runs(cols) if c1 - c0 >= 2)

    # a text line has several word clusters; a page-edge reflection, a stray
    # mark or the small page-id tag has one or two
    bands = [(a, b) for a, b in bands if n_word_clusters(a, b) >= 3]
    # a thin band glued to the top/bottom of the analysed area is the page edge
    # (desk/shadow showing past a slightly-off corner fit), not a text line
    bands = [(a, b) for a, b in bands
             if not ((a == 0 or b == ink.shape[0]) and b - a < 0.05 * PAGE_H)]
    # merge bands separated by tiny gaps (dots on i, descenders), keep the n largest gaps
    while len(bands) > n_lines:
        gaps = [bands[i + 1][0] - bands[i][1] for i in range(len(bands) - 1)]
        j = int(np.argmin(gaps))
        bands[j:j + 2] = [(bands[j][0], bands[j + 1][1])]
    # lines whose descenders touch the next line form one tall band: split the
    # tallest band at the weakest row inside it until we have n_lines
    while len(bands) < n_lines:
        j = int(np.argmax([b - a for a, b in bands]))
        a, b = bands[j]
        margin = max(8, int(0.3 * (b - a) / 2))
        if b - a < 3 * margin:
            break
        cut = a + margin + int(np.argmin(row[a + margin: b - margin]))
        bands[j:j + 1] = [(a, cut), (cut, b)]
    if len(bands) != n_lines:
        raise ValueError(f"found {len(bands)} text bands on the reference page, expected {n_lines}; "
                         "check the reference photo or pass --lines")
    bands.sort()
    oy, ox = int(0.03 * PAGE_H), int(0.03 * PAGE_W)
    boxes = []
    for a, b in bands:
        cols = ink[a:b].mean(axis=0) > 0.02
        cruns = [(c0, c1) for c0, c1 in runs(cols) if c1 - c0 >= 2]
        if not cruns:
            raise ValueError("empty text band")
        x0 = cruns[0][0]
        if not keep_numbers and len(cruns) > 1:
            # drop the leading "N." cluster: first gap wider than 1.5 % of the page
            for (c0, c1), (d0, _d1) in zip(cruns, cruns[1:]):
                if d0 - c1 > 0.015 * PAGE_W:
                    x0 = d0
                    break
        x1 = cruns[-1][1]
        m = pad * PAGE_H
        boxes.append([ox + x0 - m, oy + a - m, ox + x1 + m, oy + b + m])
    return boxes


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("photos", nargs="+", type=Path, help="sharp reference photo first, then the others")
    ap.add_argument("--lines", type=int, default=10)
    ap.add_argument("--keep-numbers", action="store_true")
    ap.add_argument("--pad", type=float, default=0.008, help="margin around each line, fraction of page height")
    ap.add_argument("--preview-dir", type=Path, default=None,
                    help="default: <artifact>/outputs/boxes_preview next to the inputs folder")
    args = ap.parse_args()

    ref = args.photos[0]
    preview_dir = args.preview_dir or ref.resolve().parent.parent / "outputs" / "boxes_preview"
    preview_dir.mkdir(parents=True, exist_ok=True)
    page_rect = np.array([[0, 0], [PAGE_W, 0], [PAGE_W, PAGE_H], [0, PAGE_H]], dtype=np.float64)

    # 1. line layout from the reference
    gray, _size, _s = load_gray(ref)
    corners = page_corners(gray)
    H_ref = homography(page_rect, corners)
    page = warp_to_page(gray, H_ref)
    layout = line_layout(page, args.lines, args.keep_numbers, args.pad)
    print(f"reference {ref.name}: {len(layout)} lines")

    # 2. map to every photo (reference included)
    for photo in args.photos:
        try:
            gray, (W, H), s = load_gray(photo)
            c = page_corners(gray)
        except ValueError as e:
            print(f"  {photo.name}: SKIPPED ({e})")
            continue
        H_p = homography(page_rect, c)
        boxes = []
        for x0, y0, x1, y1 in layout:
            quad = apply_h(H_p, [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]) / s
            l, t = quad.min(axis=0)
            r, b = quad.max(axis=0)
            boxes.append([round(max(l, 0) / W, 4), round(max(t, 0) / H, 4),
                          round(min(r, W) / W, 4), round(min(b, H) / H, 4)])
        out = photo.with_suffix(".boxes.json")
        out.write_text(json.dumps(boxes, indent=2) + "\n", encoding="utf-8")
        # preview
        img = ImageOps.exif_transpose(Image.open(photo)).convert("RGB")
        d = ImageDraw.Draw(img)
        d.polygon([tuple(p / s) for p in c], outline=(0, 255, 0), width=max(2, W // 600))
        for i, (l, t, r, b) in enumerate(boxes, 1):
            d.rectangle([l * W, t * H, r * W, b * H], outline=(255, 0, 0), width=max(2, W // 600))
            d.text((l * W + 4, t * H + 2), str(i), fill=(255, 0, 0))
        img.thumbnail((1600, 1600))
        img.save(preview_dir / f"{photo.stem}.jpg", quality=85)
        print(f"  {photo.name}: wrote {out.name}, preview {preview_dir / (photo.stem + '.jpg')}")


if __name__ == "__main__":
    main()
