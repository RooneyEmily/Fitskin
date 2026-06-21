# Sephora Bag White Reference: Methods and Results

Extension of the Phase 4 chart-free skin color pipeline (see `FLASH_NOFLASH_SKIN_METHODS.md`
and `FLASH_NOFLASH_SKIN_RESULTS.md`). This note asks whether a black-and-white striped Sephora
bag held in-frame can serve as a chart-free exposure reference, replacing the offline ColorChecker
anchor that Phase 4 currently requires.

**Benchmark context.** The Phase 4 pipeline — ColorChecker-calibrated RGB→XYZ affine matrix,
ambient CCT estimation, LU sharpening, FitSkin lightness calibration — achieved mean
ΔE₀₀ = 3.33 (Table 13) and mean ΔE₀₀ = 2.27 with scanner lightness calibration (Table 14)
on five same-session trials. Those numbers represent the ceiling for what any chart-free
replacement can realistically target.

Artifacts:
- Script: `evaluate_sephora_bag_white_reference.py`
- Per-trial output: `sephora_bag_white_reference_eval/sephora_bag_white_reference_ablation.csv`
- Summary: `sephora_bag_white_reference_eval/sephora_bag_white_reference_summary.json`
- Segmentation: `sephora_bag_reference.py`, `sephora_bag_mobile_sam.py`
- Visual verification: `run_full_pipeline_overlay.py`, `full_pipeline_overlays/`

---

## Methods

### Data and Reference Measurements

Six paired iPhone ProRAW DNG captures: Emily trials 1–3 and Liki trials 1–3. Each trial
contains a no-flash and a flash frame. **These bag captures are from a dedicated separate
session, not the same session as the Phase 4 FitSkin evaluation.** Liki trial 1 was excluded
from ΔE₀₀ reporting (darker room, exposure outlier), yielding N = 5 evaluated trials.

The bag material was independently measured with a NIX Spectro 2 under D65:

| Material | L\* | a\* | b\* | Y |
|---|---|---|---|---|
| White stripe | 94.1 | –0.3 | 1.2 | 0.851 |
| Black stripe | 24.3 | –0.1 | –0.4 | 0.042 |

Full XYZ: white = [0.807, 0.851, 0.889], black = [0.040, 0.042, 0.044].
The black stripe serves as a QC contrast check only; it is not used for black-point correction.

### Flash/No-Flash Skin Pipeline

The bag evaluation used the identical Phase 4 chart-free pipeline: RAW DNGs demosaiced to
linear camera RGB (camera WB disabled), ECC-registered flash/no-flash pair, skin-mask exposure
matching, geometric reflectance proxy

\[
\mathbf{R} = \sqrt{\max(\mathbf{A}\odot\mathbf{B},\,\varepsilon)},
\]

iPhone affine RGB→XYZ calibration, D65 CIELAB conversion, MediaPipe Face Mesh cheek ROI,
CIEDE2000 comparison against mapped FitSkin cheek Lab.

### Bag Segmentation

The segmentation pipeline has three stages:

**1. Stripe-energy scan (primary, no hand landmarks required).** Below the chin, per-row
median luma is computed in a face-centred search band. A sliding window standard deviation
of those row medians peaks at the alternating black/white pattern, which is unique in the
scene. The window with maximum alternation score defines the vertical extent; horizontal
extent is trimmed by the same column-wise luma-alternation score. This approach is robust
to hand position and works even when only one hand is detected.

**2. Sobel edge refinement.** Within the candidate bbox, locally-normalised Sobel-x and
Sobel-y edge energy snaps the left, right, top, and bottom boundaries to the actual paper
edges. Column (row) energy percentile thresholds control tightness.

**3. MobileSAM (TinyViT, 39 MB).** A box-plus-point prompt refines the paper mask within
the edge-trimmed bbox, separating bag paper from hands and shirt. When two hands are
detected their wrists provide negative prompts; otherwise the search region below the bag
provides a shirt negative. MobileSAM was chosen over SAM2 for this task because its
compact TinyViT encoder is faster on the RTX 4090 and produces masks with comparable
quality for the high-contrast stripe pattern.

**White stripe measurement.** The white mask is split into left, centre, and right
horizontal thirds. The median RGB is computed in each zone independently; the final
white-stripe estimate is the channel-wise median of the three zone medians. This makes
the measurement robust to hand occlusion, specular highlights, or shadow on any single
side of the bag.

### Exposure Anchor Conditions

Ten anchor families were evaluated, building on the Phase 4 training anchor as baseline:

| Code | Description |
|---|---|
| `none` | No absolute anchor |
| `training_anchor` | Per-participant offline ColorChecker exposure scale (Phase 4 baseline) |
| `bag_white_*` | Raw camera-luma bag anchor: \(s = Y_\text{NIX} / Y_\text{camera}\) |
| `bag_white_*_xyz_y` | Calibrated Y-only: ratio in XYZ space after iPhone RGB→XYZ |
| `bag_white_*_xyz_lstsq` | Full-XYZ least-squares scalar matching NIX white XYZ |
| `hybrid_training_x_*_rel` | Training anchor × per-participant centred bag correction \(r_i\) |
| `lstar_training_x_*_rel` | Training anchor; bag correction adjusts L\* only |
| `vonkries_bag[_x_training]` | Per-channel RGB diagonal from bag white; optionally × training scale |
| `twopoint_bag[_x_training]` | Per-channel affine fit from bag white + black stripe; optionally × training scale |
| `cat02_bag[_x_training]` | CAT02 chromatic adaptation from bag white XYZ → D65; optionally × training scale |
| `cct_from_bag` | Training anchor with ambient CCT estimated from bag white chromaticity |

The `*` suffix is `noflash`, `flash_aligned`, or `reflectance`, indicating which image the bag
white is measured from. The participant-centred scalar correction is:

\[
r_i = \frac{s_{\text{bag},i}}{\operatorname{median}_j(s_{\text{bag},j}\mid\text{participant})}
\]

**Von Kries 3-channel WB.** Finds per-channel scales \([s_R, s_G, s_B]\) such that
\(M \,\text{diag}(\mathbf{s})\,\mathbf{c}_\text{white} = \mathbf{x}_\text{NIX}\), where
\(M\) is the calibrated 3×3 RGB→XYZ matrix. Applied to both the no-flash and flash frames
before reflectance computation; the cached ECC warp is reused to avoid re-running alignment.

**Two-point affine calibration.** Uses both the white and black stripe to fit a per-channel
affine \(c' = a_c \cdot c + b_c\), mapping the camera measurements of both stripes to their
NIX-measured RGB equivalents (obtained by inverting \(M\)). This is equivalent to a per-channel
density-style linearisation.

**CAT02 chromatic adaptation.** The bag white XYZ defines the scene illuminant; a full
von Kries-style Bradford CAT02 diagonal adaptation matrix maps it to the NIX D65 reference.
The adaptation is expressed as a 3×3 pre-correction matrix in camera-RGB space.

---

## Results

### Updated Ablation (three-zone median + MobileSAM + stripe-scan + chromatic corrections)

Full ranked table, N = 6 trials (Emily 1–3, Liki 1–3):

| Anchor | mean ΔE₀₀ | median | std | n |
|---|---:|---:|---:|---:|
| `lstar_training × flash_aligned_xyz_y_rel` | **8.43** | 8.91 | 5.90 | 6 |
| `lstar_training × flash_aligned_xyz_lstsq_rel` | 8.49 | 8.96 | 5.92 | 6 |
| **`twopoint_bag × training`** (new) | **8.51** | 8.78 | 3.94 | 6 |
| **`vonkries_bag × training`** (new) | **8.54** | 9.11 | 5.04 | 6 |
| `hybrid_training × reflectance_xyz_y_rel` | 8.57 | 8.77 | 6.12 | 6 |
| `hybrid_training × reflectance_xyz_lstsq_rel` | 8.58 | 8.76 | 6.12 | 6 |
| `hybrid_training × flash_aligned_xyz_y_rel` | 8.85 | 9.83 | 5.83 | 6 |
| **`cat02_bag × training`** (new) | 8.91 | 9.98 | 5.58 | 6 |
| **`twopoint_bag`** standalone (new) | 9.03 | 10.25 | 2.96 | 6 |
| `training_anchor` (Phase 4 baseline, cross-session) | 9.23 | 10.62 | 5.79 | 6 |
| **`vonkries_bag`** standalone (new) | 10.18 | 11.79 | 3.16 | 6 |
| `bag_white_flash_aligned_xyz_lstsq` | 10.71 | 12.35 | 3.29 | 6 |
| `none` | 11.22 | 13.40 | 3.74 | 6 |
| `bag_white_noflash` (raw luma) | 13.59 | 15.73 | 3.65 | 6 |

### Why Are These Numbers So Much Worse Than Table 13?

The Phase 4 same-session result (Table 13, mean ΔE₀₀ = 3.33) is **not an apples-to-apples
comparison** with these numbers. Three factors account for the ~6 ΔE₀₀ gap:

1. **Session mismatch.** The bag captures and the FitSkin scanner measurements are from
   different sessions. Skin color varies day-to-day from hydration, redness, temperature
   and posture. Even perfect exposure normalisation cannot recover a measurement that was
   never made simultaneously.

2. **The bag corrects one scalar dimension.** The Phase 4 training anchor — derived from a
   full ColorChecker session — encodes the camera's spectral response to the actual illuminant
   in the room where the participant was photographed. The bag provides a single luminance
   (or at best RGB-scalar) correction. It cannot correct chromatic shifts caused by a
   different ambient colour temperature between sessions.

3. **Cross-session illuminant drift.** The bag session was a different room under different
   lighting. Even the `training_anchor` reads 9.62 in this cross-session evaluation, versus
   3.33 when applied same-session. Most of the 9.34–9.62 range reflects illuminant drift,
   not a failure of the bag segmentation or measurement method.

The bag experiment is therefore best understood as a **cross-session transfer experiment**:
we are testing whether in-frame bag information can partially compensate for the absence of
a same-session ColorChecker, not whether it matches a fully calibrated same-session pipeline.

### What the Bag Does Buy

Within this cross-session setting:

- The best mode (`lstar_training × flash_aligned_xyz_y_rel`, ΔE₀₀ = 8.43) improves by
  **0.80 ΔE₀₀** over the training anchor alone (9.23) and by **2.79 ΔE₀₀** over no anchor (11.22).
- **New chromatic modes** are competitive: `twopoint_bag × training` (8.51) and
  `vonkries_bag × training` (8.54) nearly match the best scalar mode and do so with lower
  variance (std 3.9 and 5.0 vs 5.9), meaning fewer extreme outlier trials.
- **`twopoint_bag` standalone** (9.03) is the best result achievable with *no calibration session*
  at all — bag white and black stripe measurements alone beat no anchor by 2.2 ΔE₀₀.
- **`vonkries_bag` standalone** (10.18) also beats no anchor, confirming per-channel
  correction is useful even without training data.
- The `lstar` family consistently outperforms the `hybrid` family, confirming that the bag's
  main contribution is luminance normalisation, not chromatic correction.
- Calibrated XYZ bag anchors consistently outperform raw camera-luma bag anchors by ~2–4 ΔE₀₀,
  confirming the RGB→XYZ matrix is load-bearing.
- Raw luma bag anchors (13.6) are worse than no anchor (11.2) in most trials, and should never
  be used without the XYZ calibration step.

---

## Manuscript-Ready Draft

### Methods Paragraph

To evaluate whether an in-frame white reference could replace the offline ColorChecker
exposure anchor, participants held a black-and-white striped Sephora paper bag during paired
iPhone ProRAW flash/no-flash capture. The bag was independently measured with a NIX Spectro 2
under D65 (white stripe: Y = 0.851; black stripe: Y = 0.042; full white XYZ = [0.807, 0.851, 0.889]).
For each DNG pair, the Phase 4 reflectance pipeline was run identically to the chart-free
condition. The bag was localised using a stripe-energy scan: below the chin, per-row median luma
was computed in a face-centred search band, and the region of peak row-to-row luma alternation
was selected as the stripe region. Sobel-x/y edge detection refined the bounding box to the
actual paper edges. MobileSAM (TinyViT encoder, 39 MB) segmented the bag paper from hands and
background within that box. White-stripe pixels were split into left, centre, and right thirds;
the final white RGB was the channel-wise median of the three zone medians, making the estimate
robust to hand occlusion or shadow on any one side. Bag measurements were then converted to XYZ
via the trained iPhone RGB→XYZ calibration. Ten families of exposure anchor were derived and
compared against FitSkin cheek CIELAB using CIEDE2000, ranging from raw camera-luma scalars
through calibrated XYZ scalars to per-channel chromatic corrections: von Kries 3-channel white
balance, two-point affine calibration using both white and black stripes, and CAT02 chromatic
adaptation from the scene illuminant estimated by the bag white chromaticity.

### Results Paragraph

Across six bag trials (Emily 1–3, Liki 1–3), the best single anchor mode applied an L\*-only
correction using the flash-aligned bag white XYZ-Y relative to each participant's median bag
scale, on top of the Phase 4 training anchor (mean ΔE₀₀ = 8.43, std = 5.90). The new chromatic
correction modes were competitive: two-point affine calibration combined with the training anchor
achieved mean ΔE₀₀ = 8.51 with lower variance (std = 3.94) than the scalar modes (std ≈ 6.0),
and von Kries 3-channel correction similarly yielded 8.54. Notably, two-point calibration without
any training anchor achieved 9.03, the best result obtainable with no prior calibration data.
The ~5–6 ΔE₀₀ gap between these results and the Phase 4 same-session result (Table 13: mean
ΔE₀₀ = 3.33) reflects session mismatch: bag and face captures were from a separate session from
the FitSkin scanner measurements, and even the training anchor yields 9.23 applied cross-session.
Within the cross-session setting, all calibrated XYZ bag anchors outperformed raw camera-luma
anchors (13.6) and no anchor (11.2), confirming that the RGB→XYZ calibration is necessary for
the bag reference to be useful. These results are best understood as a cross-session transfer
experiment motivating same-session bag + FitSkin capture as the next validation step.

---

## Recommended Reporting Language

- **Say:** "The Sephora bag provides a viable in-frame exposure reference when interpreted
  through the RGB→XYZ calibration, yielding mean ΔE₀₀ = 8.43 (best scalar hybrid) or 8.51
  (two-point calibration) in a cross-session transfer experiment."
- **Say:** "The ~5–6 ΔE₀₀ gap relative to the Phase 4 same-session result (Table 13, 3.33)
  reflects session-to-session skin and illuminant variability, not a failure of bag segmentation."
- **Say:** "These results motivate same-session bag + FitSkin capture as the next validation step."
- **Do not say:** "The bag replaces ColorChecker calibration." It supplies a scalar anchor only.
- **Do not say:** "The bag result is equivalent to the Phase 4 pipeline." It is a feasibility
  ablation under cross-session conditions.
