"""Overlap-preserving bleed removal: Rowley-Brooke pipeline + bleed/pencil decomposition.

Adaptation of the proposal's classical branch (Sec. 4.3, Fig. 4). The paper's
stages are kept: mirror + align, classify interference from the joint front/back
intensities (Sec. 3.2.1), connected-component refinement (3.2.2), and restoration
from background paper (3.3). Two changes target pencil retention:

1. Dense registration refinement. A tile-wise normalized cross-correlation field
   on the ink maps corrects residual paper warp that the global homography leaves
   (several pixels, comparable to stroke width).
2. Overlap decomposition instead of whole-region replacement. Inside the
   interference region, front optical density is modelled as

       d_front(x) = a(x) * m_back(x) + p(x),   p >= 0,

   where m_back is the registered reverse-ink map, a(x) is a smooth bleed gain
   (ink penetration varies stroke by stroke), and p is front-side pencil. We solve

       min_{a, p>=0}  sum (d_front - a m_back - p)^2 + sum lam(x) p + smooth(a)

   by alternating minimization: a is a Gaussian-weighted least-squares fit given
   p, and p is the proximal step of the sparsity penalty given a (non-negative
   soft threshold for L1, hard threshold for L0; L0 keeps pencil density unbiased). lam(x) increases with the
   reverse-ink density and its edge strength, where model noise is larger.
   Restored overlap pixels are paper * exp(-p), so pencil seen through or around
   the bleed is kept at its observed density, not redrawn.

Pixels outside the interference region (front-only pencil, background) are
copied from the input unchanged. No learned model, transcription or line boxes
are used during restoration.
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

from remove_bleed import (BACK, BOTH, classify, detrend, exemplar_plate, page_mask,
                          read_image, refine, register)

# Selected by bleed/tune_overlap.py (coordinate search on the tune half; see
# bleed/results/pencil-sheet08-overlap/tuning.json). Hand-set starting values are
# tune_overlap.INITIAL.
DEFAULTS = dict(
    back_floor=0.25,     # reverse density below this is paper/grid, not ink
    back_span=3.5,       # density range mapped to m_back; large span keeps Sharpie-core variation unclipped
    back_sigma=2.5,      # paper diffusion blur applied to the reverse map
    gain_sigma=1.5,      # spatial scale of the bleed gain a(x), pixels
    lam0=0.15,           # base sparsity weight on pencil density
    lam_ink=0.5,        # extra weight proportional to m_back
    lam_edge=0.0,        # extra weight at reverse-ink edges (registration slack)
    iterations=6,
    region_dilate=5,     # pixels around interference that are re-estimated
    hyst_low=0.0,        # weak-residual threshold for stroke continuity (0 = off)
    min_area=700,        # drop isolated confident components smaller than this
    texture_k=0,         # >0: non-local patch prediction of bleed texture (Deck 13 NLM), k neighbours
    texture_h=0.3,       # NLM bandwidth on back-density patch distance
    texture_gain=1.0,    # shrink toward the smooth model (<1)
    lam_var=0.0,         # adaptive threshold: + lam_var * local residual std (Deck 7 adaptive filter)
    var_sigma=12.0,      # window for the local residual std
    scales=(),           # extra diffusion blur scales for the bleed model (spatially varying PSF)
    ridge=0.01,          # ridge on the multi-scale gain fit
    penalty='l1',        # sparsity penalty on pencil: 'l1' (soft) or 'l0' (hard, unbiased)
    single_resample=False,# compose registration warps, resample once (bicubic); equal within noise
)

# Recognition rendering (render_clean). Set by inspection on this sheet, not by the search.
CLEAN = dict(
    gap_core=1.35,       # reverse density above this = opaque core; pencil there is unknown, not absent
    gap_radius=7,        # Navier-Stokes inpainting radius for the pencil layer in cores
    grid_length=61,      # straight-line opening length that only printed grid survives
    speck_density=0.12, speck_area=30,
    ink_gain=3.0,        # rendered ink = exp(-ink_gain * density) on white
    fill='paths',        # gap filling in opaque cores: 'paths' (evidence-guided), 'diffusion' (NS), 'none'
    path_evidence=0.10,  # min mean residual density along an accepted path (bleed noise median ~0)
    path_tortuosity=1.8, # max path length / chord
    path_max_gap=90,     # max chord length in pixels
    path_radius=0,       # tube half-width; 0 = from the stroke width at the ports
    path_mode='observed',# 'observed': admit only measured residue on the path; 'interp': also fill end densities
    path_floor=0.04,     # observed mode: min residue density admitted on the path
)


def paper_level(gray):
    """Local paper brightness: closing removes ink, median removes grid lines."""
    small = cv2.resize(gray, None, fx=.25, fy=.25, interpolation=cv2.INTER_AREA)
    small = cv2.morphologyEx(small, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    small = cv2.medianBlur(small, 21).astype(np.float32)
    return cv2.resize(cv2.GaussianBlur(small, (0, 0), 5), gray.shape[::-1])


def density(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return np.log(paper_level(gray) / np.maximum(gray.astype(np.float32), 1)).astype(np.float32)


def refine_registration(d_front, d_back, valid, tile=192, step=96, radius=14):
    """Dense residual shift field from tile NCC on ink maps; robust and smooth."""
    h, w = d_front.shape
    f = cv2.GaussianBlur(np.clip((d_front-.3)/.5, 0, 1).astype(np.float32), (0, 0), 1.5)
    b = cv2.GaussianBlur(np.clip((d_back-.6)/.6, 0, 1).astype(np.float32), (0, 0), 1.5)
    ys, xs = np.arange(0, h-tile, step), np.arange(0, w-tile, step)
    dx = np.full((len(ys), len(xs)), np.nan, np.float32); dy = dx.copy()

    def subpixel(a, c, e):
        d = a - 2*c + e
        return 0. if d >= 0 else .5*(a-e)/d

    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            t = f[y:y+tile, x:x+tile]
            if (t.mean() < .02 or y < radius or x < radius or y+tile+radius > h
                    or x+tile+radius > w or not valid[y:y+tile, x:x+tile].all()):
                continue
            r = cv2.matchTemplate(b[y-radius:y+tile+radius, x-radius:x+tile+radius], t, cv2.TM_CCOEFF_NORMED)
            _, score, _, (lx, ly) = cv2.minMaxLoc(r)
            if score < .5 or lx in (0, 2*radius) or ly in (0, 2*radius):
                continue
            dx[i, j] = lx - radius + subpixel(r[ly, lx-1], r[ly, lx], r[ly, lx+1])
            dy[i, j] = ly - radius + subpixel(r[ly-1, lx], r[ly, lx], r[ly+1, lx])
    measured = int(np.isfinite(dx).sum())
    if measured < 8:
        return np.zeros((h, w), np.float32), np.zeros((h, w), np.float32), dict(tiles=measured)

    def clean(a):
        local = ndimage.generic_filter(a, lambda v: np.nanmedian(v) if np.isfinite(v).any() else np.nan,
                                       size=5, mode='nearest')
        a = np.where(np.abs(a - local) > 2, np.nan, a)
        idx = ndimage.distance_transform_edt(~np.isfinite(a), return_distances=False, return_indices=True)
        return ndimage.gaussian_filter(a[tuple(idx)], 1).astype(np.float32)

    dx, dy = clean(dx), clean(dy)
    mx = np.interp(np.arange(w), xs + tile/2, np.arange(len(xs))).astype(np.float32)
    my = np.interp(np.arange(h), ys + tile/2, np.arange(len(ys))).astype(np.float32)
    MX, MY = np.meshgrid(mx, my)
    DX = cv2.remap(dx, MX, MY, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    DY = cv2.remap(dy, MX, MY, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return DX, DY, dict(tiles=measured, max_shift_pixels=float(np.hypot(dx, dy).max()),
                        median_shift_pixels=float(np.median(np.hypot(dx, dy))))


def warp(image, DX, DY, border):
    h, w = DX.shape
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(image, gx+DX, gy+DY, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=border)


def back_map(d_back, params):
    m = np.clip((d_back - params['back_floor']) / params['back_span'], 0, 1).astype(np.float32)
    return cv2.GaussianBlur(m, (0, 0), params['back_sigma']) if params['back_sigma'] > 0 else m


def regressors(m_back, scales):
    """Reverse ink seen through paper at several diffusion scales (spatially varying PSF)."""
    return [m_back] + [cv2.GaussianBlur(m_back, (0, 0), float(sc)) for sc in scales]


def fit_gains(X, target, weight, sigma, ridge):
    """Gaussian-windowed least squares: target ~ sum_k a_k(x) X_k, gains smooth at sigma."""
    K = len(X)
    if K == 1:
        num = cv2.GaussianBlur(weight*X[0]*target, (0, 0), sigma)
        den = cv2.GaussianBlur(weight*X[0]*X[0], (0, 0), sigma) + 1e-4
        return [np.clip(num/den, 0, 2)]
    f = 4                                                   # gains are smooth: solve on a coarse grid
    small = lambda z: cv2.resize(cv2.GaussianBlur(z, (0, 0), sigma), None, fx=1/f, fy=1/f,
                                 interpolation=cv2.INTER_AREA)
    A = np.empty(small(X[0]).shape + (K, K), np.float32); B = np.empty(A.shape[:-1], np.float32)
    for i in range(K):
        B[..., i] = small(weight*X[i]*target)
        for j in range(i, K):
            A[..., i, j] = A[..., j, i] = small(weight*X[i]*X[j])
    scale = np.trace(A, axis1=-2, axis2=-1)[..., None, None]/K + 1e-4
    G = np.linalg.solve(A + ridge*scale*np.eye(K, dtype=np.float32), B[..., None])[..., 0]
    h, w = target.shape
    return [cv2.resize(np.ascontiguousarray(G[..., k]), (w, h), interpolation=cv2.INTER_LINEAR) for k in range(K)]


def decompose(d_front, m_back, region, params):
    """Alternating minimization for bleed gains a_k(x) and non-negative pencil p(x).

    Bleed model: sum_k a_k(x) (G_k * m_back). With scales=() this is a(x) m_back.
    """
    rows, cols = np.nonzero(region.any(1))[0], np.nonzero(region.any(0))[0]
    p = np.zeros_like(d_front); bleed = np.zeros_like(d_front)
    if not len(rows):
        return p, bleed, []
    pad = int(4*max([params['gain_sigma'], *params['scales']]))
    sl = (slice(max(rows[0]-pad, 0), rows[-1]+pad+1), slice(max(cols[0]-pad, 0), cols[-1]+pad+1))
    d, m, reg = d_front[sl], m_back[sl], region[sl]
    edge = cv2.magnitude(cv2.Sobel(m, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(m, cv2.CV_32F, 0, 1, ksize=3)) / 4
    lam = params['lam0'] + params['lam_ink']*m + params['lam_edge']*np.clip(edge, 0, 1)
    lam_base = lam
    X = regressors(m, params['scales'])
    weight = reg.astype(np.float32)
    q = np.zeros_like(d)
    history = []
    for _ in range(params['iterations']):
        gains = fit_gains(X, d - q, weight, params['gain_sigma'], params['ridge'])
        b = np.clip(sum(g*x for g, x in zip(gains, X)), 0, None)
        residual = d - b
        if params['lam_var'] > 0:
            # Adaptive local noise reduction (ECE 253 Deck 7, p.11): scale the threshold
            # by the local residual std, measured on pixels not currently explained as ink.
            w = (reg & (q <= 0) & (m > .05)).astype(np.float32)
            s2 = cv2.GaussianBlur(w*np.clip(residual, -.6, .6)**2, (0, 0), params['var_sigma']) / \
                (cv2.GaussianBlur(w, (0, 0), params['var_sigma']) + 1e-3)
            lam = lam_base + params['lam_var']*np.sqrt(s2)*np.clip(m*4, 0, 1)
        if params['penalty'] == 'l0':
            # prox of lam^2/4 * [p != 0]: hard threshold, keeps observed density unbiased
            q = np.where(residual > lam/2, residual, 0) * reg
        else:
            q = np.maximum(residual - lam/2, 0) * reg     # prox of lam*p with p >= 0 (L1)
        sparsity = (lam**2/4*(q > 0)) if params['penalty'] == 'l0' else lam*q
        history.append(float(np.mean(((d - b - q)**2)[reg]) + np.mean(sparsity[reg])))
    if params['texture_k'] > 0:
        b = b + predict_texture(d - b, q, m, reg, params, sl)
        residual = d - b
        q = np.maximum(residual - lam/2, 0) * reg if params['penalty'] != 'l0' else np.where(residual > lam/2, residual, 0)*reg
    p[sl] = q; bleed[sl] = b
    return p, bleed, history


TEXTURE_EXCLUDE = None   # evaluation hook: pixels never used as texture training data


def predict_texture(residual, q, m, reg, params, sl, half=3, stride=2, train_max=250000, seed=0):
    """Non-local 'targeted database' prediction of bleed texture (ECE 253 Deck 13, NLM).

    The blotchy front-side bleed is not a smooth function of the back density, so we
    predict it non-locally: each interference pixel is described by its back-density
    patch (which pencil cannot contaminate), matched against confident bleed-only
    pixels of the same image, and assigned the NLM-weighted mean of their residuals.
    """
    from scipy.spatial import cKDTree
    offs = [(dy, dx) for dy in range(-half, half+1, stride) for dx in range(-half, half+1, stride)]
    pad = np.pad(m, half, mode='edge')
    h, w = m.shape
    feats = np.stack([pad[half+dy:half+dy+h, half+dx:half+dx+w] for dy, dx in offs], -1)
    query = reg & (m > .05)
    train = query & (q <= 0) & (m > .2)
    if TEXTURE_EXCLUDE is not None:
        train &= ~TEXTURE_EXCLUDE[sl]
    ty, tx = np.nonzero(train)
    if len(ty) < 1000:
        return np.zeros_like(residual)
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(ty), min(train_max, len(ty)), replace=False)
    ty, tx = ty[pick], tx[pick]
    tree = cKDTree(feats[ty, tx])
    qy, qx = np.nonzero(query)
    k = int(params['texture_k'])
    dist, idx = tree.query(feats[qy, qx], k=k+1, workers=-1)
    targets = np.clip(residual[ty, tx], -.6, .6)[idx]
    selfhit = (ty[idx] == qy[:, None]) & (tx[idx] == qx[:, None])    # leave-one-out
    hh = params['texture_h']
    wts = np.exp(-dist**2/hh**2) * ~selfhit
    pred = (wts*targets).sum(1) / np.maximum(wts.sum(1), 1e-6)
    out = np.zeros_like(residual); out[qy, qx] = pred*params['texture_gain']
    return out


def stroke_continuity(p, residual, d_front, region, params):
    """Handwriting prior: pencil is continuous, bleed residue is speckle.

    Hysteresis: weak residual (> hyst_low) is kept only when 8-connected to
    confident pencil or to front ink entering the region from outside. Confident
    components that are small and isolated from outside ink are dropped. This
    only selects among observed residual values; nothing is drawn.
    """
    if params['hyst_low'] <= 0 and params['min_area'] <= 0:
        return p
    eight = np.ones((3, 3), bool)
    outside_ink = (~region) & (d_front > .15)
    entering = ndimage.binary_dilation(outside_ink, eight, iterations=2) & region
    strong = p > .02
    out = p.copy()
    if params['min_area'] > 0:
        cc, n = ndimage.label(strong, eight)
        if n:
            area = np.bincount(cc.ravel(), minlength=n+1)
            touches = np.bincount(cc[entering & strong].ravel(), minlength=n+1) > 0
            drop = (area < params['min_area']) & ~touches; drop[0] = False
            out[drop[cc]] = 0
            strong = out > .02
    if params['hyst_low'] > 0:
        weak = (residual > params['hyst_low']) & region
        cc, n = ndimage.label(weak | strong, eight)
        seeded = np.zeros(n+1, bool)
        seeded[np.unique(cc[strong | (entering & weak)])] = True; seeded[0] = False
        keep = seeded[cc] & weak & ~strong
        out[keep] = residual[keep]
    return out


def run(front_path, back_path, output, max_side=6000, already_mirrored=False, params=None, save=True):
    params = {**DEFAULTS, **(params or {})}
    start = time.perf_counter(); cv2.setNumThreads(4)
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    front, back = read_image(front_path, max_side), read_image(back_path, max_side)
    native = Image.open(front_path).size
    if max(native) > max_side:
        print(f'warning: front {native} downscaled to max side {max_side}; '
              'untouched pixels match the resized image, not the original file')
    back, valid, registration = register(front, back, already_mirrored)
    valid &= page_mask(front)
    d_front, d_back = density(front), density(back)
    DX, DY, dense = refine_registration(d_front, d_back, valid)
    if params.get('single_resample', True):
        # ECE 253 Deck 2 p.28-32: compose homography + shift field and resample the
        # original (mirrored) back once, bicubic, instead of two bilinear warps.
        raw = read_image(back_path, max_side)
        raw = raw if already_mirrored else raw[:, ::-1].copy()
        h, w = DX.shape
        gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        pts = np.stack([gx + DX, gy + DY], -1).reshape(-1, 1, 2)
        Hinv = np.linalg.inv(np.array(registration['homography']))
        src = cv2.perspectiveTransform(pts, Hinv).reshape(h, w, 2)
        back = cv2.remap(raw, src[..., 0], src[..., 1], cv2.INTER_CUBIC,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        inside = cv2.remap(np.ones(raw.shape[:2], np.uint8), src[..., 0], src[..., 1], cv2.INTER_NEAREST,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0
        valid &= inside
    else:
        back = warp(back, DX, DY, (255, 255, 255))
        valid &= warp(valid.astype(np.uint8), DX, DY, 0) > 0
    d_back = density(back)
    state = prepare(front, back, valid, d_front, d_back)
    result = restore_overlap(state, params)
    report = dict(method='Rowley-Brooke stages + dense registration + overlap decomposition (recover_overlap.py)',
                  params=params, registration=registration, dense_registration=dense,
                  classification=state['classification'],
                  region_fraction=float(result['region'][valid].mean()),
                  objective_history=result['history'], elapsed_seconds=time.perf_counter()-start)
    strokes = front_strokes(state, result)
    minimal = minimal_restore(state, result, params, strokes=strokes)
    ablations = {f'minimal_protection_{m}': minimal_restore(state, result, params, protection=m, strokes=strokes)['output']
                 for m in ('off', 'on')}
    report['minimal'] = dict(touched_fraction=float(minimal['touched'][valid].mean()),
                             front_ink_in_bleed_fraction=float(minimal['ink'][valid].mean()),
                             bleed_rgb_ratio=minimal['bleed_ratio'])
    if save == 'minimal':
        # Batch export: identical restoration, without unrelated cleanup experiments.
        from remove_bleed import rowley_brooke_restore
        rb_out, _ = rowley_brooke_restore(front, state['labels'], valid, plate=state['plate'])
        images = {'input': front, 'rowley_brooke': rb_out, 'minimal': minimal['output'],
                  'minimal_protection_off': ablations['minimal_protection_off'],
                  'aligned_back_dense': back}
        for name, array in images.items():
            Image.fromarray(array).save(output/f'{name}.png', compress_level=1)
        report['elapsed_seconds'] = time.perf_counter()-start
        (output/'recover_run.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        return report, state, result
    if save:
        Image.fromarray(minimal['output']).save(output/'minimal.png')
        # Rowley-Brooke 2013 Sec. 3.3 baseline on the SAME registration, labels and plate as ours,
        # so the only differences downstream are identification extent and stroke protection.
        import remove_bleed
        rb_out, _ = remove_bleed.rowley_brooke_restore(front, state['labels'], valid, plate=state['plate'])
        Image.fromarray(rb_out).save(output/'rowley_brooke.png')
        for name, array in ablations.items():
            Image.fromarray(array).save(output/f'{name}.png')
        Image.fromarray(minimal['touched'].astype(np.uint8)*255).save(output/'minimal_touched_mask.png')
        palette = np.array([[245, 245, 245], [40, 150, 60], [230, 70, 60], [140, 70, 190]], np.uint8)
        clean = render_clean(d_front, result, d_back, params)
        clean_nogap = render_clean(d_front, result, d_back, params, gaps=False)
        clean_diffusion = render_clean(d_front, result, d_back, params, clean=dict(fill='diffusion'))
        baseline = output/'filtered.png'
        if baseline.exists():   # same cleanup on the previous filter, for a fair comparison
            fake = dict(region=np.zeros(valid.shape, bool), pencil=np.zeros(valid.shape, np.float32))
            Image.fromarray(render_clean(density(np.array(Image.open(baseline).convert('RGB'))), fake, d_back,
                                         params, gaps=False)).save(output/'filtered_clean.png')
        for name, array in [('input', front), ('aligned_back_dense', back), ('recovered', result['output']),
                            ('recovered_clean', clean), ('recovered_clean_nogap', clean_nogap),
                            ('recovered_clean_diffusion', clean_diffusion),
                            ('labels_dense', palette[state['labels']]),
                            ('pencil_density', np.uint8(np.clip(255*np.exp(-result['pencil']), 0, 255))),
                            ('bleed_model', np.uint8(np.clip(255*np.exp(-result['bleed']), 0, 255))),
                            ('region_mask', result['region'].astype(np.uint8)*255)]:
            Image.fromarray(array).save(output/f'{name}.png')
        (output/'recover_run.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report, state, result


def prepare(front, back, valid, d_front, d_back):
    """Paper stages 3.2.1-3.2.2 on the densely registered pair; reused across params."""
    r, _ = detrend(front); v, _ = detrend(back)
    labels, classification = classify(r, v, valid)
    labels = refine(labels, valid)
    # Sec. 3.3.1 clean background plate (Criminisi exemplar inpainting), filled only
    # where restoration can use it: around any reverse ink or bleed-type label.
    near = ((labels == BACK) | (labels == BOTH) | (back_map(d_back, DEFAULTS) > .1)) & valid
    near = cv2.dilate(near.astype(np.uint8), np.ones((21, 21), np.uint8)) > 0
    plate = exemplar_plate(front, labels, valid, region=near)
    return dict(front=front, back_aligned=back, valid=valid, d_front=d_front, d_back=d_back, labels=labels,
                plate=plate, classification=classification)


def restore_overlap(state, params, d_front=None):
    """Sec. 3.3 restoration, keeping the decomposed pencil density in overlaps."""
    params = {**DEFAULTS, **params}
    front, valid, plate = state['front'], state['valid'], state['plate']
    d_front = state['d_front'] if d_front is None else d_front
    m_back = back_map(state['d_back'], params)
    interference = ((state['labels'] == BACK) | (state['labels'] == BOTH) | (m_back > .15)) & valid
    k = 2*int(params['region_dilate']) + 1
    region = cv2.dilate(interference.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
    region &= valid
    p, bleed, history = decompose(d_front, m_back, region, params)
    p = stroke_continuity(p, d_front - bleed, d_front, region, params)
    alpha = cv2.GaussianBlur(region.astype(np.float32), (0, 0), 1.0)
    alpha[~valid] = 0
    pencil_rgb = plate.astype(np.float32) * np.exp(-p)[..., None]
    out = front.astype(np.float32)*(1-alpha[..., None]) + pencil_rgb*alpha[..., None]
    return dict(output=np.uint8(np.clip(out, 0, 255)), pencil=p, bleed=bleed,
                region=region, history=history)


def density_rgb(image):
    """Per-channel optical density relative to each channel's local paper level."""
    paper = np.stack([paper_level(np.ascontiguousarray(image[..., c])) for c in range(3)], -1)
    return np.log(paper / np.maximum(image.astype(np.float32), 1)).astype(np.float32), paper


def grid_through(D, hidden, length=241, cap=0.35):
    """Printed-grid density continued under `hidden` pixels: a long-line opening that
    ignores hidden pixels, so grid lines resume through removed bleed."""
    work = np.where(hidden, 10., np.clip(D, 0, None)).astype(np.float32)
    best = np.zeros_like(work)
    for k in [np.ones((1, length), np.uint8), np.ones((length, 1), np.uint8)]:
        best = np.maximum(best, cv2.morphologyEx(work, cv2.MORPH_OPEN, k))
    best[best > 1] = 0                                   # windows fully hidden: no evidence
    return np.minimum(best, cap)


def nearby_paper_texture(front, paper, clean_paper, holes, offsets=(24, 36, 48, 64, 96)):
    """Proposal Sec. 4.3: restore 'using nearby paper texture'. Each hole pixel takes
    the paper texture (photo / local paper level) of the nearest clean-paper pixel
    at a fixed offset, so grain and shading noise match the surroundings."""
    ratio = front.astype(np.float32) / np.maximum(paper, 1)
    texture = np.ones_like(ratio)
    todo = holes.copy()
    h, w = holes.shape
    for d in offsets:
        for dy, dx in [(0, d), (0, -d), (d, 0), (-d, 0), (d, d), (-d, -d), (d, -d), (-d, d)]:
            if not todo.any():
                return texture
            ys, xs = np.nonzero(todo)
            sy, sx = ys + dy, xs + dx
            ok = (sy >= 0) & (sy < h) & (sx >= 0) & (sx < w)
            ys, xs, sy, sx = ys[ok], xs[ok], sy[ok], sx[ok]
            good = clean_paper[sy, sx]
            texture[ys[good], xs[good]] = ratio[sy[good], sx[good]]
            todo[ys[good], xs[good]] = False
    return texture


def front_strokes(state, result, clean=None, paths=True, touch=0.02):
    """Interference pixels (predicted bleed > touch) and the front strokes found inside them."""
    clean = {**CLEAN, **(clean or {})}
    gray = state['d_front']
    touched = result['region'] & state['valid'] & (result['bleed'] > touch)
    ink = touched & (result['pencil'] > 0)
    if paths:
        base, _ = pencil_layer(gray, result, state['d_back'], {**clean, 'fill': 'none'})
        filled, _ = pencil_layer(gray, result, state['d_back'], {**clean, 'fill': 'paths', 'path_mode': 'observed'})
        ink |= touched & (filled > base + 1e-6)
    return touched, ink


def strict_ink(state, result, touched, ink, tau=0., min_area=0, anchor=False):
    """Keep only protected components that look like front strokes: divided density > tau,
    and (area >= min_area, or 8-connected to front ink visible outside the bleed when anchor)."""
    d_div = np.clip(state['d_front'] - result['bleed'], 0, None)
    ink = ink & (d_div > tau)
    if not (min_area or anchor):
        return ink
    eight = np.ones((3, 3), bool)
    visible = (state['d_front'] > .15) & ~touched & state['valid']
    cc, n = ndimage.label(ink | visible, eight)
    keep = np.zeros(n + 1, bool)
    if min_area:
        keep |= np.bincount(cc[ink], minlength=n + 1) >= min_area
    if anchor:
        keep[np.unique(cc[visible])] = True
    keep[0] = False
    return ink & keep[cc]


def visible_chroma(D, gray, touched, valid, sigma=60):
    """Per-channel density ratio of front ink seen OUTSIDE the bleed, averaged locally:
    protected strokes take the colour of the same writing where it is unobstructed."""
    clean = (~touched & valid & (gray > .2)).astype(np.float32)
    den = cv2.GaussianBlur(gray*clean, (0, 0), sigma)
    glob = np.array([np.median(D[..., c][clean > 0] / gray[clean > 0]) for c in range(3)], np.float32) \
        if clean.sum() > 100 else np.ones(3, np.float32)
    out = np.empty(D.shape, np.float32)
    for c in range(3):
        num = cv2.GaussianBlur(D[..., c]*clean, (0, 0), sigma)
        out[..., c] = np.where(den > 1e-3, num/np.maximum(den, 1e-6), glob[c])
    return out


def minimal_restore(state, result, params=None, clean=None, protection='divide', paths=True, touch=0.02, strokes=None,
                    fill='exemplar', ink_tau=0.3, ink_min_area=0, ink_anchor=True, stroke_colour='divide',
                    ink_cap=None, ink_smooth=0):
    """Default output: the photograph, edited only where reverse ink is predicted.

    Follows the proposal's classical branch (Sec. 4.3, Fig. 4): interference is
    identified from the joint front/back intensities (classification + component
    refinement + decomposition), and interference pixels are restored from nearby
    paper texture, with printed grid lines continued. Pixels with predicted bleed
    density <= `touch` are copied bit-for-bit.

    protection (the proposal's 'Protect foreground strokes' block, tested off/on):
      'off'    without the modification: every interference pixel is restored to
               paper, overlaps included (erases front strokes in overlaps);
      'on'     the modification as written in the proposal: front strokes detected
               inside the interference (decomposed p > 0, plus residue admitted by
               the minimal-path step) are left unchanged (keeps the bleed on them);
      'divide' the modification refined after testing (default): those stroke
               pixels keep their own ink while the bleed share is divided out per
               channel, in * exp(k_c * bleed), capped at paper.
    Stroke mask (`strict_ink`): a protected pixel needs bleed-free density d_g - a*m_b > ink_tau
    (0.3, chosen by the sweep in sweep_protection.py) and must belong to a component connected to
    front ink visible outside the bleed (ink_anchor). Without these, bleed mottle inside opaque
    Sharpie cores is kept as 'ink' and OCR reads it as extra characters.
    """
    params = {**DEFAULTS, **(params or {})}
    clean = {**CLEAN, **(clean or {})}
    front, valid = state['front'], state['valid']
    D, paper = density_rgb(front)
    gray = state['d_front']
    bleed = result['bleed']
    touched, ink = strokes if strokes is not None else front_strokes(state, result, clean, paths, touch)
    ink = strict_ink(state, result, touched, ink, ink_tau, ink_min_area, ink_anchor)
    if protection == 'off':
        ink[:] = False
    ref = touched & ~ink & (bleed > .3) & (gray > .2)
    ratio = np.array([np.median(D[..., c][ref] / gray[ref]) for c in range(3)], np.float32) if ref.sum() > 100 \
        else np.ones(3, np.float32)
    holes = touched & ~ink
    grid = np.stack([grid_through(D[..., c], touched) for c in range(3)], -1)
    out = front.astype(np.float32).copy()
    if fill == 'exemplar':   # paper Sec. 3.3.1 plate (texture already from nearby paper), grid continued
        out[holes] = (state['plate'].astype(np.float32) * np.exp(-grid))[holes]
    else:                    # earlier simplified fill: local paper level x texture at a fixed offset
        clean_paper = valid & ~result['region'] & (gray < .06)
        texture = nearby_paper_texture(front, paper, clean_paper, holes)
        out[holes] = (paper * np.exp(-grid) * texture)[holes]
    if protection == 'divide' and stroke_colour == 'visible':
        # grey density with the bleed divided out, coloured like the same ink outside the bleed
        d_div = np.clip(gray - bleed, 0, None)
        if ink_smooth:   # median inside the stroke mask: removes the bleed's mottle, keeps stroke edges
            dm = cv2.medianBlur(np.uint8(np.clip(np.where(ink, d_div, 0)*100, 0, 255)), ink_smooth).astype(np.float32)/100
            d_div = np.where(ink, np.maximum(dm, 0), d_div)
        if ink_cap is not None:   # no darker than the front ink seen outside the bleed (quantile ink_cap)
            vis = (gray > .15) & ~touched & valid
            if vis.sum() > 100:
                d_div = np.minimum(d_div, np.quantile(gray[vis], ink_cap))
        chroma = visible_chroma(D, gray, touched, valid)
        out[ink] = (paper * np.exp(-d_div[..., None] * chroma))[ink]
    elif protection == 'divide':
        divided = np.minimum(front.astype(np.float32) * np.exp(bleed[..., None] * ratio), paper)
        out[ink] = divided[ink]
    return dict(output=np.uint8(np.clip(np.round(out), 0, 255)), touched=touched, ink=ink,
                bleed_ratio=ratio.tolist())


def continue_gaps(p, core, radius):
    """Treat opaque-core pencil as missing data: NS inpainting continues strokes that
    enter and leave a core. Observed residual is kept where it is larger."""
    if not core.any():
        return p
    p8 = np.uint8(np.clip(p*255, 0, 255))
    filled = cv2.inpaint(p8, core.astype(np.uint8)*255, radius, cv2.INPAINT_NS).astype(np.float32)/255
    return np.where(core, np.maximum(p, filled), p)


def stroke_ports(D, core, min_stroke=30):
    """Places where visible pencil (outside the core) touches an opaque core."""
    eight = np.ones((3, 3), bool)
    ink = (D > .08) & ~core
    cc, n = ndimage.label(ink, eight)
    big = np.bincount(cc.ravel(), minlength=n+1) >= min_stroke; big[0] = False
    ring = ndimage.binary_dilation(core, eight, iterations=2) & ~core
    pc, m = ndimage.label(big[cc] & ring, eight)
    ports = []
    for k, sl in enumerate(ndimage.find_objects(pc), 1):
        ys, xs = np.nonzero(pc[sl] == k)
        if len(ys) < 3:
            continue
        ys, xs = ys + sl[0].start, xs + sl[1].start
        y0, y1, x0, x1 = max(ys.min()-8, 0), ys.max()+9, max(xs.min()-8, 0), xs.max()+9
        near = D[y0:y1, x0:x1][ink[y0:y1, x0:x1]]
        ports.append(dict(y=ys, x=xs, cy=float(ys.mean()), cx=float(xs.mean()),
                          density=float(np.median(near)) if len(near) else .2,
                          width=float(np.clip(len(ys)/2, 2, 7))))
    return ports


def complete_strokes(D, residual, core, clean):
    """Evidence-guided minimal paths through opaque cores (geodesic completion).

    Stroke ends on the core boundary are paired; for each pair the cheapest
    8-connected path through the core is found with Dijkstra, where cost is low
    where the decomposition still sees positive pencil residue. A path is kept
    only if its mean evidence beats `path_evidence` and it is not tortuous.
    In 'observed' mode the path only admits measured residue (>= path_floor)
    inside a thin tube, so no value is synthesized. In 'interp' mode, pixels
    weaker than the interpolated end-point density are filled with it. Returns
    the completed layer and the mask of synthesized (not observed) pixels.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra
    out = D.copy(); filled = np.zeros(D.shape, bool)
    evidence = np.clip(cv2.GaussianBlur(residual.astype(np.float32), (0, 0), 1.5), 0, .6)
    ports = stroke_ports(D, core)
    if len(ports) < 2:
        return out, filled
    zone_lab, _ = ndimage.label(ndimage.binary_dilation(core, np.ones((3, 3), bool), iterations=3))
    groups = {}
    for i, pt in enumerate(ports):
        z = zone_lab[int(round(pt['cy'])), int(round(pt['cx']))]
        if z: groups.setdefault(z, []).append(i)
    candidates = []
    for z, idx in groups.items():
        if len(idx) < 2:
            continue
        allp = np.concatenate([ports[i]['y'] for i in idx]); allx = np.concatenate([ports[i]['x'] for i in idx])
        sl = (slice(max(allp.min()-10, 0), allp.max()+11), slice(max(allx.min()-10, 0), allx.max()+11))
        zc = core[sl]; ev = evidence[sl]; ink = (D[sl] > .08) & ~zc
        h, w = zc.shape
        cost = np.where(zc, 1/(.05 + ev), np.where(ink, 2., 200.)).astype(np.float64)
        node = np.arange(h*w).reshape(h, w); a, b, wt = [], [], []
        for dy, dx in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            ya, yb = slice(0, h-dy), slice(dy, h)
            xa, xb = (slice(0, w-dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w+dx))
            a.append(node[ya, xa].ravel()); b.append(node[yb, xb].ravel())
            wt.append((np.hypot(dy, dx)*(cost[ya, xa]+cost[yb, xb])/2).ravel())
        graph = coo_matrix((np.concatenate(wt), (np.concatenate(a), np.concatenate(b))), shape=(h*w, h*w)).tocsr()
        local = lambda i: (np.clip(np.round(ports[i]['cy']).astype(int)-sl[0].start, 0, h-1),
                           np.clip(np.round(ports[i]['cx']).astype(int)-sl[1].start, 0, w-1))
        src = [node[local(i)] for i in idx]
        dist, pred = dijkstra(graph, directed=False, indices=src, return_predecessors=True)
        for u in range(len(idx)):
            for v in range(u+1, len(idx)):
                i, j = idx[u], idx[v]
                chord = np.hypot(ports[i]['cy']-ports[j]['cy'], ports[i]['cx']-ports[j]['cx'])
                if chord > clean['path_max_gap'] or chord < 4 or not np.isfinite(dist[u, src[v]]):
                    continue
                path = [src[v]]
                while path[-1] != src[u] and pred[u, path[-1]] >= 0:
                    path.append(pred[u, path[-1]])
                py, px = np.divmod(np.array(path), w)
                inside = zc[py, px]
                if inside.sum() < 2:
                    continue
                steps = np.hypot(np.diff(py), np.diff(px)).sum()
                mean_ev = float(ev[py, px][inside].mean())
                if steps/chord > clean['path_tortuosity'] or mean_ev < clean['path_evidence']:
                    continue
                candidates.append((mean_ev, i, j, py+sl[0].start, px+sl[1].start))
    used = set()
    for mean_ev, i, j, py, px in sorted(candidates, key=lambda c: -c[0]):
        if i in used or j in used:
            continue
        used.update((i, j))
        n = len(py); t = np.linspace(0, 1, n)
        dens = (1-t)*ports[j]['density'] + t*ports[i]['density']   # path runs from port j to port i
        width = (ports[i]['width'] + ports[j]['width'])/2
        line = np.zeros(D.shape, np.float32)
        line[py, px] = dens
        radius = int(clean['path_radius']) or max(1, int(round(width/2)))
        tube = cv2.dilate(line, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1,)*2)) * core
        observed = np.clip(residual, 0, None)
        if clean['path_mode'] == 'observed':
            sel = (tube > 0) & (observed >= clean['path_floor'])
            newv = observed                                  # measured values only; nothing synthesized
        else:
            keep_obs = observed >= tube
            newv = np.where(keep_obs, observed, tube)
            sel = tube > 0
            filled |= sel & ~keep_obs
        out[sel] = np.maximum(out[sel], newv[sel])
    return out, filled


def remove_grid(D, length):
    """Top-hat against long horizontal/vertical openings: printed grid survives a
    straight `length`-pixel line, handwriting does not."""
    h = cv2.morphologyEx(D, cv2.MORPH_OPEN, np.ones((1, length), np.uint8))
    v = cv2.morphologyEx(D, cv2.MORPH_OPEN, np.ones((length, 1), np.uint8))
    return np.clip(D - np.maximum(h, v), 0, None)


def despeckle(D, density_threshold, area):
    cc, n = ndimage.label(D > density_threshold, np.ones((3, 3)))
    keep = np.bincount(cc.ravel(), minlength=n+1) >= area; keep[0] = False
    return D * ndimage.binary_dilation(keep[cc], iterations=2)


def pencil_layer(d_front, result, d_back, clean):
    """Grid-free pencil density with the chosen gap filling; also the filled mask."""
    p = result['pencil']
    core = (d_back > clean['gap_core']) & result['region']
    filled = np.zeros(p.shape, bool)
    if clean['fill'] == 'diffusion':
        q = continue_gaps(p, core, clean['gap_radius']); filled = core & (q > p); p = q
    D = remove_grid(np.where(result['region'], p, np.clip(d_front, 0, None)).astype(np.float32), clean['grid_length'])
    if clean['fill'] == 'paths':
        D, filled = complete_strokes(D, d_front - result['bleed'], core, clean)
    return D, filled


def render_clean(d_front, result, d_back, params, clean=None, gaps=True):
    """Pencil-only page for recognition: front ink outside the interference region,
    decomposed pencil inside, grid and specks removed, drawn as ink on white."""
    clean = {**CLEAN, **(clean or {})}
    if not gaps:
        clean['fill'] = 'none'
    D, _ = pencil_layer(d_front, result, d_back, clean)
    D = despeckle(D, clean['speck_density'], clean['speck_area'])
    gray = np.uint8(np.clip(255*np.exp(-clean['ink_gain']*D), 0, 255))
    return np.dstack([gray]*3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('front', type=Path); parser.add_argument('back', type=Path)
    parser.add_argument('--output', type=Path, default=Path('bleed/outputs/pencil-sheet08'))
    parser.add_argument('--params', type=Path, help='JSON file of tuned parameters')
    parser.add_argument('--max-side', type=int, default=6000,
                        help='larger images are downscaled (outputs are then not bit-identical to the original)')
    parser.add_argument('--already-mirrored', action='store_true')
    args = parser.parse_args()
    tuned = json.loads(args.params.read_text()) if args.params else None
    report, _, _ = run(args.front, args.back, args.output, args.max_side, args.already_mirrored, tuned)
    print(json.dumps({k: report[k] for k in ('dense_registration', 'region_fraction', 'elapsed_seconds')}, indent=2))
