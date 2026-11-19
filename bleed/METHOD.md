# Method: Rowley-Brooke et al. (CVPR 2013) base + foreground-stroke protection

Reference: [Rowley-Brooke, Pitié and Kokaram, "A Non-parametric Framework for Document Bleed-Through Removal", CVPR 2013](https://www.cv-foundation.org/openaccess/content_cvpr_2013/papers/Rowley-Brooke_A_Non-parametric_Framework_2013_CVPR_paper.pdf) (proposal ref. [13]). Diagrams: `results/diagrams/pipeline-paper-vs-ours.*` (paper vs ours, stage by stage; `diagrams/paper-vs-ours.tex`) and `results/diagrams/pipeline-comparison.*` (proposal Fig. 4: classical vs DE-GAN; `diagrams/comparison.tex`).

## 1. The base, checked against the paper text

`remove_bleed.py` implements the paper. Each item below was checked against the PDF.

| Paper | Implementation | Status |
| --- | --- | --- |
| 3.1 registration "assumed" | SIFT/RANSAC homography after mirroring, plus dense tile-NCC refinement (`recover_overlap.refine_registration`) | **addition** (the paper's data is pre-registered) |
| 3.1 detrending: align local histogram peaks, 200×200 blocks, overlap 50 | `detrend(block=200, overlap=50)` on both sides; border/table extremes excluded from block modes | matches (+ exclusion) |
| 3.2.1 Eq. 1, `E(l) = Σ_x [U_x(l_x) + α Σ_{y∈N_x} V(l_x,l_y)]` over the joint recto/verso histogram | `classify`: one node per occupied (r,v) bin, binary set adjacency `N_x` (Eq. 2) | matches |
| U = Mahalanobis distance to each label's cluster | per-label mean/covariance (regularised), Otsu initialisation | matches (initialisation unspecified in the paper) |
| V = −ln Table 1 (asymmetric) | each undirected edge costs `V(a,b)+V(b,a)` = the paper's sum over both directed neighbour pairs | matches |
| Eq. 3, `α = I·5.8845e−7 − 0.0024522`, I = number of distinct intensities in the recto-verso image | `I = len(occupied)` (distinct (r,v) pairs); α = 0.0088 on sheet 08 | matches |
| QPBO, cycling through the four labels | graph-cut α-expansion cycling through the four labels; one non-submodular label triple is truncated (by 0.01) | **solver substitute**; verified equal to brute force on small graphs (`test_remove_bleed.py`) |
| 3.2.2 connected-component rules: small bgbg → majority neighbour; fgfg needs both fgbl and blfg on its edge; small fgbl/blfg → fgfg if edge has fgfg but no bgbg, → the opposite label if edge has it but no fgfg | `refine`, iterated to convergence (cap 10); used in both pipelines; checked rule by rule | matches |
| 3.3 replace the verso's foreground in the recto (blfg) | `rowley_brooke_restore`: blfg → plate | matches (the text says "fgbl for the recto", read as a typo; fgbl is front ink) |
| 3.3.1 Criminisi exemplar inpainting; source = bgbg minus top 10% gradients | `criminisi_inpaint`, `exemplar_plate` | matches, except the patch search is limited to a 40 px window for speed |
| 3.3.2 blending at inner/outer edges | `blend_restore`, 2 px ramp | matches (ramp width not given in the paper) |

An earlier prototype deviated from the paper, and those deviations are now removed:
- ICM solver;
- neighbour counts normalised to degree 4;
- averaged Table 1 probabilities;
- α clipped at zero;
- Telea inpainting.

These remain available only as `classify(..., solver='icm', neighbourhood='normalized')`, so the old outputs can be reproduced.

**Fair baseline.** `rowley_brooke.png` is computed from the same registered pair, labels and plate as ours (`eval_protection.py`, `recover_overlap.run`). The first difference between the two rows of the diagram is therefore at stage 2.

## 2. Our changes

1. **Identification (stage 2).** The Eq. 1 labels are kept. In addition, front density is decomposed as `d_g = a(x)·m_b + p` with `p ≥ 0`, solved by alternating minimisation: windowed least squares for `a`, an L1 prox for `p`. Here `m_b` is the aligned back density, left unclipped inside Sharpie cores. The interference region is the blfg/fgfg labels plus `m_b > .15`. Pixels with predicted bleed ≤ 0.02 OD are copied bit-for-bit.
2. **Stroke protection (stage 3) — our contribution.** The paper has no explicit stroke-protection step. Its strokes survive only if the classifier labels them fgbl/fgfg, which fails when front ink is fainter than the bleed: pencil or pen under Sharpie gets labelled blfg and inpainted away. We detect front ink inside the interference (`p > 0`, plus measured residue admitted by minimal paths). We keep only confident strokes (`strict_ink`): bleed-free density `d_g − a·m_b > τ = 0.3` and connected to front ink visible outside the bleed. τ was chosen by `sweep_protection.py`; see README. The stroke set S is handled in one of three ways:
   - `off`: replace it with the plate, like the paper;
   - `on`: the paper's own rule, "foreground pixels are left unchanged", applied to our mask, so the bleed stays on the stroke;
   - `divide` (default): `f̂ = g·exp(k_c·a·m_b)`, which removes only the bleed's share per channel and is capped at paper.
3. **Restoration (stage 4).** Non-stroke interference gets the paper's exemplar plate, with printed grid lines continued under it (`grid_through`, a 241 px line opening): `f̂ = r_b·e^{−G}`.

Bleed-model parameters were set by coordinate search on the tune half of the pixel truths (`tune_overlap.py`). Pixel results are reported on the held-out half as well.

## 3. Evaluating the contribution

The ladder isolates each change. All rows share the registration, labels and plate.

| Step | What changes |
| --- | --- |
| Rowley-Brooke → ours off | identification only (labels → labels + decomposition) |
| ours off → ours divide | **stroke protection only** |
| ours on vs divide | how protected pixels are treated (paper's "leave unchanged" vs divide out bleed) |

Three measurements, all computed from output images:
- **Real photo, semi-synthetic, `eval_protection.py`.** Real clean pencil patches are multiplied into the front photo over real bleed (the model our decomposition inverts; no black-floor clipping). The whole pipeline is rerun, and recovered pencil = output density with paste − without. Bleed-only zones right of each line should become paper. Front-only pencil should be unchanged. Truth masks come from line boxes and the back photo, not from our decomposition. Results: `results/pencil-sheet08-overlap/protection-ablation.json`.
- **Synthetic inks, `synthetic_inks.py`.** Clean sheet 01 is recoloured as four inks and combined with sheet 08's measured bleed. The composite is multiplicative, the model `divide` inverts, so divide's ≈1.0 density ratio is partly by construction. A `--saturate` rerun clips at the camera's black floor.
- **OCR, `evaluate_pair.py`.** TrOCR on 10 lines, with a 256-character reference: one character is 0.4 points of CER. This is secondary, single-sheet evidence.

Numbers and interpretation: [README.md](README.md).

## 4. Course methods tried without gain

Scored on the held-out pixel objective before the label fix (model 0.647 then; 0.639 with the faithful labels, not re-scored for these):
- adaptive local-variance threshold (Deck 7): 0.580–0.645;
- single bicubic resampling of composed warps (Deck 2): 0.642;
- non-local patch prediction of bleed texture (Deck 13): 0.637–0.646.

They are kept as off-by-default switches (`lam_var`, `single_resample`, `texture_k`). Bleed texture inside opaque Sharpie can't be predicted from the back photo; that is a limit of this capture.

## Data notes

`samples/pencil-sheet08/front.jpg` is the pencil side and `back.jpg` the Sharpie side; originals are preserved. The pair is a development case and belongs with sheet08 in any split. The OCR reference is the intended sentence list; where Sharpie hides a letter it is an intended transcription, not proof of a visible stroke. No restoration step receives line boxes, text or a clean reference.
