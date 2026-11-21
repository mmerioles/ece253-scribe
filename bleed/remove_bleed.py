"""Paired bleed removal following Rowley-Brooke et al. (CVPR 2013).

One pipeline: register -> detrend -> joint histogram MRF -> components -> restore.
See bleed/METHOD.md for reproduction differences and limits on pencil recovery.
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


def _register(front, back, already_mirrored=False):
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


def register(front, back, already_mirrored=False):
    """Try native matching, then wider-context features at 1600 pixels."""
    try:
        return _register(front, back, already_mirrored)
    except ValueError:
        if max(front.shape[:2]) <= 1600:
            raise
    def reduced(image):
        h, w = image.shape[:2]
        scale = min(1, 1600 / max(h, w))
        small = cv2.resize(image, (round(w*scale), round(h*scale)), interpolation=cv2.INTER_AREA)
        return small, np.diag([small.shape[1]/w, small.shape[0]/h, 1.0])
    # Mirror before resizing so pixel-center rounding cannot change mirror geometry.
    source = back if already_mirrored else back[:, ::-1].copy()
    f, sf = reduced(front)
    b, sb = reduced(source)
    _, _, report = _register(f, b, True)
    H = np.linalg.inv(sf) @ np.array(report['homography']) @ sb
    h, w = front.shape[:2]
    aligned = cv2.warpPerspective(source, H, (w, h), borderValue=(255,255,255))
    valid = cv2.warpPerspective(np.ones(source.shape[:2], np.uint8), H, (w,h), flags=cv2.INTER_NEAREST) > 0
    report.update(homography=H.tolist(), feature_max_side=1600,
                  error_coordinate_system='reduced feature image',
                  median_error_full_resolution_pixels=report['median_error_pixels']/min(sf[0,0],sf[1,1]))
    return aligned, valid, report


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


def classify(recto, verso, valid):
    """Joint histogram Mahalanobis unaries + image-neighbour MRF (Sec. 3.2.1).

    Deterministic ICM is used instead of QPBO. Histogram seeds are initialized
    by Otsu splits; the paper does not specify a peak-initialization algorithm.
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
    # Photograph adaptation: empirical neighbour frequencies avoid rare intensity
    # pairs overwhelming the background mode after JPEG quantization.
    adjacency = sparse.diags(4/np.maximum(1,np.asarray(adjacency.sum(axis=1)).ravel())) @ adjacency
    adjacency = adjacency.tocsr()
    probability = np.array([[.66,.00065,.0069,.00013], [.0065,.13,.0001,.0022],
                            [.0069,.0001,.13,.0021], [.00013,.0022,.0021,.046]])
    # Table 1's asymmetric entry is symmetrized for an undirected graph.
    potential = -np.log((probability+probability.T)/2)
    alpha = max(0, len(occupied)*5.8845e-7-.0024522)
    labels = unary.argmin(axis=1)
    changes = []
    for _ in range(20):
        changed = 0
        for i in range(len(labels)):
            neighbors = adjacency.indices[adjacency.indptr[i]:adjacency.indptr[i+1]]
            weights = adjacency.data[adjacency.indptr[i]:adjacency.indptr[i+1]]
            costs = unary[i] + alpha*(potential[:, labels[neighbors]]*weights).sum(axis=1)
            new = costs.argmin()
            changed += new != labels[i]; labels[i] = new
        changes.append(int(changed))
        if not changed:
            break
    result = np.zeros(recto.shape, np.uint8); result[valid] = labels[inverse]
    return result, dict(front_seed_threshold=tr, back_seed_threshold=tv, overlap_seed_threshold=overlap_threshold,
                        centers=centers, histogram_bins=len(occupied), alpha=alpha, icm_changes=changes)


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
    unprotected, _, _ = restore(front,labels,r,v,valid,False,plate)
    # Diagnostic upper-removal comparison, NOT the paper's overlap-preserving method.
    strong_mask = (v > classification['back_seed_threshold']) & valid
    radius = max(1, round(max_side/1000))
    strong_mask = cv2.dilate(strong_mask.astype(np.uint8), np.ones((2*radius+1,2*radius+1),np.uint8))
    strong_mask[~valid] = 0
    strong = np.where(strong_mask[...,None]>0,plate,front)
    palette = np.array([[245,245,245],[40,150,60],[230,70,60],[140,70,190]],np.uint8)
    for name, array in [('input',front),('aligned_back',back),('background',plate),('filtered',cleaned),('without_protection',unprotected),
                        ('strong_removal',strong),('strong_removal_mask',strong_mask*255),
                        ('labels_initial',palette[initial]),('labels',palette[labels]),
                        ('removal_mask',mask.astype(np.uint8)*255),('protected_mask',protected.astype(np.uint8)*255)]:
        Image.fromarray(array).save(output/f'{name}.png')
    thumb_size = (600,round(front.shape[0]*600/front.shape[1]))
    thumbs = [np.array(Image.fromarray(a).resize(thumb_size)) for a in (front,cleaned,strong)]
    Image.fromarray(np.concatenate(thumbs,axis=1)).save(output/'comparison.png')
    report = dict(method='Rowley-Brooke structure; ICM/Telea approximations (see METHOD.md)',
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
    parser.add_argument('--output',type=Path,default=None)
    parser.add_argument('--max-side',type=int,default=4000)
    parser.add_argument('--already-mirrored',action='store_true')
    args = parser.parse_args()
    if args.max_side < 400: parser.error('--max-side must be at least 400')
    if args.output is None:
        args.output = Path(__file__).resolve().parent/'outputs'/args.front.parent.name/args.front.stem
    run(args.front,args.back,args.output,args.max_side,args.already_mirrored)
