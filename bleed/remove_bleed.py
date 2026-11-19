"""Paired bleed removal following Rowley-Brooke et al. (CVPR 2013).

One pipeline: register -> detrend -> joint histogram MRF -> components -> restore.
See bleed/README.md for an overview and limits.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage, sparse

BG, FRONT, BACK, BOTH = range(4)


def read_image(path, max_side):
    image = np.array(ImageOps.exif_transpose(Image.open(path)).convert('RGB'))
    scale = min(1, max_side / max(image.shape[:2]))
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def register(front, back, already_mirrored=False):
    """Map reverse to front coordinates; reject weak or collapsed registrations."""
    if not already_mirrored:
        back = back[:, ::-1].copy()
    sift = cv2.SIFT_create(nfeatures=12000)
    kf, df = sift.detectAndCompute(cv2.cvtColor(front, cv2.COLOR_RGB2GRAY), None)
    kb, db = sift.detectAndCompute(cv2.cvtColor(back, cv2.COLOR_RGB2GRAY), None)
    if df is None or db is None:
        raise ValueError('Insufficient features for registration.')
    good = [m for pair in cv2.BFMatcher().knnMatch(db, df, k=2)
            if len(pair) == 2 for m, n in [pair] if m.distance < .7 * n.distance]
    if len(good) < 12:
        raise ValueError('Too few distinctive front/back matches.')
    src = np.float32([kb[m.queryIdx].pt for m in good])
    dst = np.float32([kf[m.trainIdx].pt for m in good])
    H, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 3)
    if H is None or inliers.sum() < 20 or inliers.mean() < .25:
        raise ValueError('Unreliable registration; do not use reverse-side guidance.')
    selected = inliers.ravel().astype(bool)
    error = np.linalg.norm(cv2.perspectiveTransform(src[:, None], H)[:, 0] - dst, axis=1)
    h, w = front.shape[:2]
    span = np.ptp(dst[selected], axis=0) / [w, h]
    if min(span) < .4 or np.median(error[selected]) > 2:
        raise ValueError('Registration lacks page coverage or accuracy.')
    aligned = cv2.warpPerspective(back, H, (w, h), borderValue=(255, 255, 255))
    valid = cv2.warpPerspective(np.ones(back.shape[:2], np.uint8), H, (w, h), flags=cv2.INTER_NEAREST) > 0
    return aligned, valid, dict(homography=H.tolist(), matches=len(good), inliers=int(selected.sum()),
                               median_error_pixels=float(np.median(error[selected])), coverage=span.tolist())


def detrend(image, block=200, overlap=50):
    """Section 3.1: align overlapping local histogram modes by additive offsets."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    ys = np.unique(np.r_[np.arange(0, max(1, h-block), block-overlap), max(0, h-block)])
    xs = np.unique(np.r_[np.arange(0, max(1, w-block), block-overlap), max(0, w-block)])
    def mode(patch):
        pixels = patch[(patch > 65)&(patch < 235)]
        if not len(pixels): pixels = gray[(gray > 65)&(gray < 235)]
        return np.argmax(ndimage.gaussian_filter1d(np.bincount(pixels,minlength=256).astype(float),1))
    modes = np.array([[mode(gray[y:y+block, x:x+block])
        for x in xs] for y in ys], np.float32)
    yy, xx = np.indices(gray.shape, dtype=np.float32)
    mx = np.interp(xx[0], xs + block/2, np.arange(len(xs)))
    my = np.interp(yy[:, 0], ys + block/2, np.arange(len(ys)))
    background = cv2.remap(modes, np.tile(mx.astype(np.float32), (h, 1)),
                           np.tile(my.astype(np.float32)[:, None], (1, w)), cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)
    # Retain a common intensity offset; clipping paper to zero collapses the MRF.
    return np.uint8(np.clip(255-(gray.astype(float)-background+modes.mean()),0,255)), background


TABLE1 = np.array([[.66, .00065, .0069, .00013],      # Rowley-Brooke et al. 2013, Table 1:
                   [.0065, .13, .0001, .0022],         # likelihood of co-occurrence of labels
                   [.0069, .0001, .13, .0021],         # rows l_x, columns l_y, order
                   [.00013, .0022, .0021, .046]])      # bgbg, fgbl, blfg, fgfg


def classify(recto, verso, valid, solver='expansion', neighbourhood='set'):
    """Joint-histogram labelling, Rowley-Brooke et al. (CVPR 2013) Sec. 3.2.1.

    E(l) = sum_{x in X} [ U_x(l_x) + alpha * sum_{y in N_x} V(l_x, l_y) ]     (Eq. 1)
      X      non-zero joint-histogram entries (recto, verso intensity pairs)
      U_x    Mahalanobis distance from x to the centre of the label cluster
      N_x    the SET of histogram points whose pixels are 4-neighbours of a pixel
             with intensity pair x in the recto-verso image                  (Eq. 2)
      V      -ln Table 1; summing over x and y in N_x counts each undirected
             edge in both directions, so an edge costs V(a,b) + V(b,a)
      alpha  = I * 5.8845e-7 - 0.0024522, I = number of distinct pairs       (Eq. 3)
    Solved by graph-cut alpha-expansion, cycling through the four labels until
    convergence; this equals the paper's QPBO wherever the move is submodular
    (all moves here except one Table-1 triple off by 0.01, which is truncated).
    Cluster centres/covariances are initialised from Otsu seeds: the paper says
    'distance from these maxima' without specifying how peaks are found.
    solver='icm', neighbourhood='normalized' reproduce the earlier prototype.
    """
    r, v = recto[valid], verso[valid]
    tv = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    low_back = r[v < np.median(v)+5]
    tr = cv2.threshold(low_back, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    # A front-dark seed within reverse ink must exceed typical bleed darkness.
    overlap_threshold = max(tr, float(np.percentile(r[v > tv], 85)))
    seeds = np.zeros(r.shape, np.uint8)
    seeds[(r > tr) & (v <= tv)] = FRONT
    seeds[v > tv] = BACK
    seeds[(v > tv) & (r > overlap_threshold)] = BOTH
    pairs = np.stack([r, v], axis=1).astype(float)
    ids = recto.astype(np.int32)*256 + verso
    occupied, inverse = np.unique(ids[valid], return_inverse=True)
    points = np.stack([occupied//256, occupied%256], axis=1)
    unary = []
    centers = []
    for label in range(4):
        sample = pairs[seeds == label]
        if len(sample) < 10:
            raise ValueError('Four joint classes not supported by this pair.')
        center = sample.mean(axis=0)
        cov = np.cov(sample.T) + np.eye(2)*4
        delta = points-center
        unary.append(np.sqrt(np.maximum(0, np.einsum('ij,jk,ik->i', delta, np.linalg.inv(cov), delta))))
        centers.append(center.tolist())
    unary = np.stack(unary, axis=1)
    lookup = np.full(65536, -1, np.int32); lookup[occupied] = np.arange(len(occupied))
    nodes = lookup[ids]
    edge_a, edge_b = [], []
    for a, b, ok in [(nodes[:-1], nodes[1:], valid[:-1]&valid[1:]),
                     (nodes[:, :-1], nodes[:, 1:], valid[:, :-1]&valid[:, 1:])]:
        edge_a.append(a[ok]); edge_b.append(b[ok])
    a, b = np.concatenate(edge_a), np.concatenate(edge_b)
    adjacency = sparse.coo_matrix((np.ones(len(a)*2), (np.r_[a,b], np.r_[b,a])),
                                  shape=(len(occupied),len(occupied))).tocsr()
    adjacency.setdiag(0); adjacency.eliminate_zeros()
    if neighbourhood == 'set':
        adjacency.data[:] = 1                                     # N_x is a set (Eq. 2)
    else:   # earlier prototype: neighbour frequencies normalised to degree four
        adjacency = sparse.diags(4/np.maximum(1, np.asarray(adjacency.sum(axis=1)).ravel())) @ adjacency
    adjacency = adjacency.tocsr()
    pair_cost = -np.log(TABLE1) - np.log(TABLE1.T)                # V(a,b) + V(b,a) per undirected edge
    alpha = len(occupied)*5.8845e-7 - .0024522                     # Eq. 3
    alpha_used = max(0., alpha)
    labels = unary.argmin(axis=1)
    if solver == 'expansion':
        labels, sweeps = alpha_expansion(unary, sparse.triu(adjacency, 1).tocoo(), pair_cost, alpha_used, labels)
    else:
        potential = -np.log((TABLE1 + TABLE1.T)/2)
        sweeps = []
        for _ in range(20):
            changed = 0
            for i in range(len(labels)):
                neighbors = adjacency.indices[adjacency.indptr[i]:adjacency.indptr[i+1]]
                weights = adjacency.data[adjacency.indptr[i]:adjacency.indptr[i+1]]
                new = (unary[i] + alpha_used*(potential[:, labels[neighbors]]*weights).sum(axis=1)).argmin()
                changed += new != labels[i]; labels[i] = new
            sweeps.append(int(changed))
            if not changed:
                break
    result = np.zeros(recto.shape, np.uint8); result[valid] = labels[inverse]
    return result, dict(front_seed_threshold=tr, back_seed_threshold=tv, overlap_seed_threshold=overlap_threshold,
                        centers=centers, histogram_bins=len(occupied), alpha=alpha, solver=solver,
                        neighbourhood=neighbourhood, edges=int(adjacency.nnz//2), sweeps=sweeps)


def mrf_energy(unary, edges, pair_cost, alpha, labels):
    return float(unary[np.arange(len(labels)), labels].sum() +
                 alpha*pair_cost[labels[edges.row], labels[edges.col]].sum())


def alpha_expansion(unary, edges, pair_cost, alpha, labels, max_sweeps=10, scale=1e4):
    """Boykov-Veksler-Zabih expansion moves, each solved exactly by min-cut
    (scipy maximum_flow). Non-submodular pair terms are truncated (Rother 2005)."""
    from scipy.sparse.csgraph import breadth_first_order, maximum_flow
    n = len(labels); i, j = edges.row, edges.col
    S, T = n, n + 1
    history = [mrf_energy(unary, edges, pair_cost, alpha, labels)]
    for _ in range(max_sweeps):
        improved = False
        for lab in range(unary.shape[1]):
            li, lj = labels[i], labels[j]
            A = alpha*pair_cost[li, lj]; B = alpha*pair_cost[li, lab]
            C = alpha*pair_cost[lab, lj]; D = alpha*np.full(len(i), pair_cost[lab, lab])
            A = np.minimum(A, B + C - D)                          # truncation keeps the move submodular
            # E(xi,xj) = A + (C-A) xi + (D-C) xj + (B+C-A-D)(1-xi) xj ; x = 1 means "switch to lab"
            cost1 = unary[:, lab].copy(); cost0 = unary[np.arange(n), labels].copy()
            np.add.at(cost1, i, C - A); np.add.at(cost1, j, D - C)
            fixed = labels == lab
            cost1[fixed] = cost0[fixed] = 0
            base = np.minimum(cost0, cost1); cost0 -= base; cost1 -= base
            w = B + C - A - D
            to_int = lambda z: np.round(z*scale).astype(np.int64)
            rows = np.r_[np.full(n, S), np.arange(n), i]
            cols = np.r_[np.arange(n), np.full(n, T), j]
            caps = np.r_[to_int(cost1), to_int(cost0), to_int(w)]
            keep = caps > 0
            graph = sparse.csr_matrix((caps[keep].astype(np.int32), (rows[keep], cols[keep])), shape=(n+2, n+2))
            flow = maximum_flow(graph, S, T).flow
            residual = (graph - flow).tocsr(); residual.data[residual.data < 0] = 0; residual.eliminate_zeros()
            reach = breadth_first_order(residual, S, directed=True, return_predecessors=False)
            source_side = np.zeros(n+2, bool); source_side[reach] = True
            switch = ~source_side[:n] & ~fixed                     # sink side = x = 1
            if switch.any():
                trial = labels.copy(); trial[switch] = lab
                e = mrf_energy(unary, edges, pair_cost, alpha, trial)
                if e < history[-1] - 1e-9:
                    labels, improved = trial, True
                    history.append(e)
        if not improved:
            break
    return labels, history


def refine(labels, valid):
    """Apply the four connected-component boundary rules (Sec. 3.2.2)."""
    labels = labels.copy()
    structure = ndimage.generate_binary_structure(2, 1)
    cc, n = ndimage.label((labels != BG)&valid, structure)
    sizes = np.bincount(cc.ravel())[1:]
    small = .1 * np.mean(sizes[sizes >= 4]) if np.any(sizes >= 4) else 1
    for _ in range(10):
        changed = False
        previous = labels.copy()
        for label in range(4):
            cc, _ = ndimage.label((labels == label)&valid, structure)
            sizes = np.bincount(cc.ravel())
            for index, sl in enumerate(ndimage.find_objects(cc), 1):
                if sl is None or (label != BOTH and sizes[index] >= small):
                    continue
                sl = tuple(slice(max(0,s.start-1), min(labels.shape[k],s.stop+1)) for k,s in enumerate(sl))
                component = cc[sl] == index
                edge = ndimage.binary_dilation(component, structure)&~component&valid[sl]
                counts = np.bincount(labels[sl][edge], minlength=4); present = counts > 0
                target = label
                if label == BG and counts.sum(): target = counts.argmax()
                elif label == BOTH:
                    if not (present[FRONT] and present[BACK]):
                        target = FRONT if present[FRONT] else BACK if present[BACK] else BG
                elif label in (FRONT,BACK):
                    opposite = BACK if label == FRONT else FRONT
                    if present[BOTH] and not present[BG]: target = BOTH
                    elif present[opposite] and not present[BOTH]: target = opposite
                if target != label:
                    labels[sl][component] = target; changed = True
        if not changed or np.array_equal(previous, labels): break
    return labels


def background_plate(front,labels,valid):
    """Use background-only sources, excluding strong gradients (Sec. 3.3.1).

    Telea replaces the paper's exemplar-based synthesis, but source text must
    still be excluded: inpainting only the removal mask copies adjacent ink.
    """
    gray=cv2.cvtColor(front,cv2.COLOR_RGB2GRAY).astype(np.float32)
    gradient=cv2.magnitude(cv2.Sobel(gray,cv2.CV_32F,1,0),cv2.Sobel(gray,cv2.CV_32F,0,1))
    source=(labels==BG)&valid
    if not source.any(): raise ValueError('No clean background available.')
    holes=((labels!=BG)|(gradient>np.percentile(gradient[source],90)))&valid
    holes=cv2.dilate(holes.astype(np.uint8),np.ones((5,5),np.uint8))
    holes[~valid]=0
    return cv2.inpaint(front,holes*255,5,cv2.INPAINT_TELEA)


def criminisi_inpaint(image, hole, source, patch=9, search=40):
    """Exemplar-based inpainting, Criminisi, Perez & Toyama (TIP 2004), as used by
    Rowley-Brooke et al. Sec. 3.3.1. Fill order by priority P(p) = C(p) D(p), with
    confidence C and data term D = |grad I_perp . n| / 255; each target patch is
    filled from the source patch (fully inside `source`) with the least SSD over
    its known pixels. The best-match search is limited to a (2*search+1)^2 window
    around the target (a standard speed-up; the original searches globally).
    """
    import heapq
    h, w = hole.shape
    k = patch//2
    pad = search + k + 1
    img = np.pad(image.astype(np.float32), ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
    unknown = np.pad(hole.astype(bool), pad, constant_values=False)
    src = np.pad(source & ~hole, pad, constant_values=False)
    centers = cv2.erode(src.astype(np.uint8), np.ones((patch, patch), np.uint8)) > 0   # whole patch is source
    conf = (~unknown).astype(np.float32)
    gray = img.mean(-1)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)/8; gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)/8
    gx[unknown] = 0; gy[unknown] = 0

    def priority(y, x):
        win = unknown[y-k:y+k+1, x-k:x+k+1]
        c = conf[y-k:y+k+1, x-k:x+k+1][~win].sum()/patch**2
        m = unknown[y-1:y+2, x-1:x+2].astype(np.float32)
        nx, ny = m[:, 2].sum() - m[:, 0].sum(), m[2].sum() - m[0].sum()
        norm = np.hypot(nx, ny) + 1e-6
        kw = ~unknown[y-1:y+2, x-1:x+2]
        ix = gx[y-1:y+2, x-1:x+2][kw]; iy = gy[y-1:y+2, x-1:x+2][kw]
        if len(ix):
            a = np.argmax(ix**2 + iy**2); px, py = -iy[a], ix[a]           # isophote = rotated gradient
            d = abs(px*nx + py*ny)/norm/255 + 1e-3
        else:
            d = 1e-3
        return c*d, c

    def front_pixels(mask_region):
        fr = unknown & cv2.dilate((~unknown).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        return np.argwhere(fr & mask_region)

    heap = []
    for y, x in front_pixels(np.ones_like(unknown)):
        pr, _ = priority(y, x); heapq.heappush(heap, (-pr, int(y), int(x)))
    filled = 0
    while heap:
        _, y, x = heapq.heappop(heap)
        if not unknown[y, x]:
            continue
        pr, c = priority(y, x)
        if heap and -heap[0][0] > pr + 1e-9:                         # stale entry: re-queue
            heapq.heappush(heap, (-pr, y, x)); continue
        tgt = img[y-k:y+k+1, x-k:x+k+1]
        known = (~unknown[y-k:y+k+1, x-k:x+k+1]).astype(np.float32)
        y0, x0 = y - search - k, x - search - k
        win = img[y0:y+search+k+1, x0:x+search+k+1]
        cand = centers[y-search:y+search+1, x-search:x+search+1]
        if not cand.any():
            # no exemplar nearby: widen to any source patch on the same rows (rare)
            fill = np.median(img[centers], axis=0) if centers.any() else img[y, x]
            m = unknown[y-k:y+k+1, x-k:x+k+1]
            img[y-k:y+k+1, x-k:x+k+1][m] = fill
        else:
            ssd = cv2.matchTemplate(win, tgt, cv2.TM_SQDIFF, mask=np.repeat(known[..., None], 3, -1))
            ssd[~cand] = np.inf
            by, bx = np.unravel_index(np.argmin(ssd), ssd.shape)
            sy, sx = y0 + by + k, x0 + bx + k
            m = unknown[y-k:y+k+1, x-k:x+k+1].copy()
            img[y-k:y+k+1, x-k:x+k+1][m] = img[sy-k:sy+k+1, sx-k:sx+k+1][m]
            gx[y-k:y+k+1, x-k:x+k+1][m] = gx[sy-k:sy+k+1, sx-k:sx+k+1][m]
            gy[y-k:y+k+1, x-k:x+k+1][m] = gy[sy-k:sy+k+1, sx-k:sx+k+1][m]
        conf[y-k:y+k+1, x-k:x+k+1][m] = c
        unknown[y-k:y+k+1, x-k:x+k+1][m] = False
        filled += int(m.sum())
        r0 = (slice(y-k-1, y+k+2), slice(x-k-1, x+k+2))
        for yy, xx in np.argwhere(unknown[r0] & cv2.dilate((~unknown[r0]).astype(np.uint8),
                                                          np.ones((3, 3), np.uint8)).astype(bool)):
            yy += y-k-1; xx += x-k-1
            p2, _ = priority(yy, xx); heapq.heappush(heap, (-p2, int(yy), int(xx)))
    return np.uint8(np.clip(np.round(img[pad:pad+h, pad:pad+w]), 0, 255))


def exemplar_plate(front, labels, valid, region=None, patch=9, search=40):
    """Sec. 3.3.1 clean background: source = bgbg minus its top 10% gradients;
    every other label is a hole. `region` limits filling to where the plate is used."""
    gray = cv2.cvtColor(front, cv2.COLOR_RGB2GRAY).astype(np.float32)
    grad = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
    bg = (labels == BG) & valid
    if not bg.any():
        raise ValueError('No clean background available.')
    source = bg & (grad <= np.percentile(grad[bg], 90))
    hole = ~source & valid
    if region is not None:
        hole &= region
    return criminisi_inpaint(front, hole, source, patch, search)


def blend_restore(image, plate, replace, width=2):
    """Sec. 3.3.2: weighted sum of original and plate across the inner and outer
    edges of the replaced components (linear ramp over 2*width pixels)."""
    inside = ndimage.distance_transform_edt(replace)
    outside = ndimage.distance_transform_edt(~replace)
    wgt = np.clip((inside - outside + width) / (2*width), 0, 1).astype(np.float32)[..., None]
    return np.uint8(np.clip(np.round(image*(1 - wgt) + plate*wgt), 0, 255)), wgt[..., 0]


def rowley_brooke_restore(front, labels, valid, plate=None):
    """Sec. 3.3: replace recto bleed-through (blfg; the paper's text says fgbl,
    which by its own label definitions would erase recto text) with the clean
    background plate, blended at component edges. fgbl and fgfg stay unchanged."""
    replace = (labels == BACK) & valid
    if plate is None:
        plate = exemplar_plate(front, labels, valid, region=cv2.dilate(replace.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0)
    out, _ = blend_restore(front.astype(np.float32), plate.astype(np.float32), replace)
    out[~valid] = front[~valid]
    return out, plate


def page_mask(front):
    """Fill the page outline: dark interior ink remains valid processing data."""
    paper=(cv2.cvtColor(front,cv2.COLOR_RGB2GRAY)>65).astype(np.uint8)
    contours,_=cv2.findContours(paper,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not contours: raise ValueError('No page found.')
    page=np.zeros_like(paper)
    cv2.drawContours(page,[max(contours,key=cv2.contourArea)],-1,1,-1)
    return cv2.erode(page,np.ones((15,15),np.uint8))>0


def restore(front, labels, recto, verso, valid, protect=True, plate=None):
    """Restore reverse-only regions; never synthesize text in overlap regions."""
    mask = (labels == BACK)&valid
    # Expand a little to cover registration uncertainty at stroke boundaries.
    mask = cv2.dilate(mask.astype(np.uint8), np.ones((3,3),np.uint8)) > 0
    uncertain = (labels == BOTH) | (labels == FRONT)
    protected = np.zeros(mask.shape, bool)
    if protect:
        # Thin dark continuations into reverse ink are ambiguous, not safe to erase.
        thin = cv2.morphologyEx(recto, cv2.MORPH_TOPHAT, np.ones((5,5),np.uint8)) > 14
        seeds = (labels == FRONT)&(verso < np.median(verso[valid])+10)&thin
        connected = ndimage.binary_propagation(seeds, mask=thin, structure=np.ones((3,3)))
        protected = uncertain | connected
        mask &= ~protected
    else:
        # Paper's BOTH and FRONT labels are still foreground, even in the ablation.
        mask &= ~uncertain
    mask &= valid
    # Practical substitution for exemplar synthesis: local Telea paper fill.
    if plate is None: plate=background_plate(front,labels,valid)
    alpha = cv2.GaussianBlur(mask.astype(np.float32), (3,3), .5)
    alpha[protected | ~valid] = 0
    output = np.uint8(np.clip(front*(1-alpha[...,None])+plate*alpha[...,None],0,255))
    return output, mask, protected


def run(front_path, back_path, output, max_side=4000, already_mirrored=False):
    start = time.perf_counter(); cv2.setRNGSeed(0); cv2.setNumThreads(4)
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    front, back = read_image(front_path,max_side), read_image(back_path,max_side)
    back, valid, registration = register(front,back,already_mirrored)
    r, _ = detrend(front); v, _ = detrend(back)
    valid &= page_mask(front)
    labels, classification = classify(r,v,valid)
    initial = labels.copy(); labels = refine(labels,valid)
    plate=background_plate(front,labels,valid)
    cleaned, mask, protected = restore(front,labels,r,v,valid,plate=plate)
    # Faithful Rowley-Brooke et al. 2013 restoration (Sec. 3.3): exemplar plate, replace blfg, blend.
    faithful, exemplar = rowley_brooke_restore(front, labels, valid)
    unprotected, _, _ = restore(front,labels,r,v,valid,False,plate)
    # Diagnostic upper-removal comparison, NOT the paper's overlap-preserving method.
    strong_mask = (v > classification['back_seed_threshold']) & valid
    radius = max(1, round(max_side/1000))
    strong_mask = cv2.dilate(strong_mask.astype(np.uint8), np.ones((2*radius+1,2*radius+1),np.uint8))
    strong_mask[~valid] = 0
    strong = np.where(strong_mask[...,None]>0,plate,front)
    palette = np.array([[245,245,245],[40,150,60],[230,70,60],[140,70,190]],np.uint8)
    for name, array in [('rowley_brooke_coarse_alignment',faithful),('background_exemplar',exemplar),
                        ('input',front),('aligned_back',back),('background',plate),('filtered',cleaned),('without_protection',unprotected),
                        ('strong_removal',strong),('strong_removal_mask',strong_mask*255),
                        ('labels_initial',palette[initial]),('labels',palette[labels]),
                        ('removal_mask',mask.astype(np.uint8)*255),('protected_mask',protected.astype(np.uint8)*255)]:
        Image.fromarray(array).save(output/f'{name}.png')
    thumb_size = (600,round(front.shape[0]*600/front.shape[1]))
    thumbs = [np.array(Image.fromarray(a).resize(thumb_size)) for a in (front,cleaned,strong)]
    Image.fromarray(np.concatenate(thumbs,axis=1)).save(output/'comparison.png')
    report = dict(method='Rowley-Brooke structure; ICM/Telea approximations (see README.md)',
        front_sha256=hashlib.sha256(Path(front_path).read_bytes()).hexdigest(),
        back_sha256=hashlib.sha256(Path(back_path).read_bytes()).hexdigest(),
        max_side=max_side, mirrored_back=not already_mirrored, working_size=list(front.shape[1::-1]),
        registration=registration, classification=classification,
        removed_fraction=float(mask[valid].mean()), elapsed_seconds=time.perf_counter()-start,
        ground_truth='No clean pencil image or pixel mask supplied; no PSNR/SSIM/F1 claimed.')
    (output/'run.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('front',type=Path); parser.add_argument('back',type=Path)
    parser.add_argument('--output',type=Path,default=Path('bleed/outputs/pencil-sheet08'))
    parser.add_argument('--max-side',type=int,default=4000)
    parser.add_argument('--already-mirrored',action='store_true')
    args = parser.parse_args()
    if args.max_side < 400: parser.error('--max-side must be at least 400')
    run(args.front,args.back,args.output,args.max_side,args.already_mirrored)
