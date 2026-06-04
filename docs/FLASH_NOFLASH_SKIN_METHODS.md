# Flash / no-flash skin color — methods text (paper-ready)

Use this document for LaTeX / manuscript wording. Citations are split so **Lu & Drew (2006)** applies only to illuminant estimation, not the primary reflectance metric.

---

## Overview (revised)

Skin color was estimated from paired images captured with and without the device flash. Each pair was converted to linear RGB, registered (ECC), and exposure-matched. The **primary chart-free skin metric** is a per-channel geometric-mean reflectance proxy on the aligned pair (see below; **not** from Lu & Drew). **Ambient illuminant estimation** and optional white-balance comparison paths follow Lu & Drew (CIC 2006). Facial skin was isolated with the same MediaPipe face-mesh tessellation mask used in the Canon RAW pipeline. Mean CIELAB (D65) was compared to FitSkin cheek measurements.

---

## Registration and exposure matching

Let **A** denote no-flash linear RGB on the reference grid. The flash frame was warped onto **A** with enhanced correlation coefficient (ECC) alignment, then **exposure-scaled** so median luma matched **A** on reliable pixels. Denote the result **B**: the flash image after **both** geometric registration and exposure matching to no-flash. All flash/no-flash formulas below use this **B**, not the raw flash capture.

---

## Primary metric: reflectance proxy (no Lu & Drew citation)

Following a practical flash/no-flash skin-color recipe used in this project (internal notes; not Lu & Drew 2006), per-pixel reflectance in linear RGB was:

\[
\mathbf{R} = \sqrt{\max(\mathbf{A} \odot \mathbf{B}, \varepsilon)}
\]

where **A** is aligned no-flash linear RGB, **B** is the **ECC-aligned, exposure-matched** flash linear RGB on the same pixel grid, and \(\odot\) is elementwise multiplication. This geometric mean attenuates capture-to-capture illumination differences while retaining skin-color information. **Lu & Drew (2006) does not define this reflectance step**; it is independent of the ambient CCT estimate below.

Cheek statistics were computed on **R** in linear RGB → CIE XYZ (trained ``camera_rgb_to_xyz`` when available) → CIELAB (D65). Optional ``--exposure-anchor-from-training`` multiplies **R** by the per-participant white-patch scale from offline ColorChecker training (chart-free at inference).

**Improved inference flags (Phase 3+):**

- ``--cheek-roi``: reflectance Lab mean inside MediaPipe cheek hull ∩ mesh (matches FitSkin cheek sampling better than full-face mesh mean).
- ``--exposure-scale-skin-mask``: after ECC, flash exposure scale from median luma on skin mask (not global scene).
- ``--reflectance-pre-wb booth``: divide **A** and **B** by booth Planck RGB (``--known-ambient-cct-k``) before ``sqrt(A⊙B)``.
- Retrain matrix with ``train_flash_noflash_checker_calibration.py``; default is unweighted stacked lstsq. Optional ``--huber-matrix`` (per-trial Huber, median 3×3) is experimental.
- ``--reflectance-pre-wb booth`` is **deprecated** (RGB WB before sqrt breaks D65 matrix). Use ``--reflectance-cat booth`` (Bradford CAT on reflectance XYZ after the matrix).
- ``--reflectance-fusion`` ``geometric|log|arithmetic``; ``--raw-u01-percentile-skin`` scales u01 from skin-mask percentile (pilot: large ΔE₀₀ drop — validate on new captures).
- Training: ``--issa-skin-rows issa_median_caucasian,issa_median_african``, ``--huber-matrix-stacked`` (experimental; separate calibration dir recommended).

**Factorial ablation (median ΔE₀₀ vs FitSkin cheek):** see ``flash_noflash_ablation_decomposition.json`` — cheek ROI alone 4.66 (−0.59 vs 5.26 mesh); skin exposure alone with mesh ROI 5.30 (negligible); both 4.50 (−0.17 vs cheek-only).

**Tier-3 offline calibration (median ΔE₀₀, same cheek+skin inference):** see ``flash_noflash_tier3_ablation.json``. **Affine** ``[R,G,B,1]→XYZ`` lstsq (**3.50**, −1.0 vs 4.50) is the only reliable win; ISSA-only rows (+0.09), weighted lstsq, and Huber stacked **hurt**. Production bundle: ``calibration/tier3_affine/`` (auto-selected by ``run_flash_noflash_skin_lab_raw.sh`` when ``camera_rgb_to_xyz_affine.npy`` exists). Re-run ablation: ``python3 run_tier3_calibration_ablation.py``.

---

## Secondary: ambient illuminant (Lu & Drew, 2006)

Lu & Drew (CIC 2006) estimate ambient illumination from flash/no-flash pairs via **log-difference geometric-mean chromaticity**, not via \(\sqrt{\mathbf{A}\odot\mathbf{B}}\).

Pure-flash (flash-only) light:

\[
\mathbf{F} = \max(\mathbf{B} - \mathbf{A}, 0)
\]

Per-pixel log-difference chromaticity \(\boldsymbol{\chi}\) is computed from **A** and **F**, compared to a synthetic Planckian locus at an estimated flash CCT, and the ambient correlated color temperature is the nearest Planckian match to the scene median \(\boldsymbol{\chi}\).

**Deviations from the paper as implemented:** no spectral-sharpening matrix \(M\) unless a calibrated \(3\times3\) matrix is supplied (`--lu-sharpening-matrix`); flash CCT is auto-estimated from neutral pure-flash pixels (fallback 5500 K only if auto fails).

**Lu-style white balance (comparison only):** no-flash linear RGB divided by Planckian RGB at the **estimated** ambient CCT. This path is **not** the primary skin metric. When booth illuminant metadata are available, an additional comparison WB uses known ambient CCT/Duv (`--known-ambient-cct-k`, `--known-ambient-duv`).

The ambient CCT estimate **does not feed** the reflectance equation above.

---

## Tertiary: SCR-AWB (Zhou et al., 2025)

Single-frame portrait AWB on **no-flash linear RAW** only (no flash term): median skin RGB on the same face mask, monochromator-measured sensor sensitivity \(S_j(\lambda)\), and an ISSA cheek reflectance prior \(r(\lambda)\) (ethnic median; P1 caucasian, P2 african by default). Illuminant SPD is a non-negative combination of three Planckian basis spectra; solve \(\mathbf{M}\boldsymbol{\alpha}\approx\mathbf{rgb}_{\mathrm{skin}}\) (Zhou et al., Eq. 6 spirit), then diagonal WB. **Comparison arm only** — not the primary \(\sqrt{\mathbf{A}\odot\mathbf{B}}\) metric. Enable with `--scr-awb` (requires `--iphone-calibration`).

---

## Facial skin segmentation

Unchanged from prior draft: MediaPipe Face Mesh on aligned no-flash preview → `build_skin_mask_from_mesh` (tessellation, IOD-scaled eye/lip/ brow exclusion, morphological cleanup). The same binary mask is applied to **R** and to comparison display paths.

---

## Software note (implementation detail — omit from main Methods)

- **8-bit previews** (no-flash, aligned flash, Lu WB): built for visualization and face-mesh detection; optional comparison Lab uses `skimage.color.rgb2lab(..., illuminant='D65')` on display sRGB BGR.
- **Reflectance statistics** use the **linear** map **R**, not preview encoding.
- Reflectance previews apply 99th-percentile scaling before 8-bit sRGB export (`reflectance_preview.png`); this scaling is not used for reported Lab means.

---

## Suggested one-paragraph replacement (reflectance + Lu split)

> After ECC alignment and exposure matching, we formed aligned no-flash linear RGB **A** and exposure-matched aligned flash linear RGB **B**. The primary skin-color estimate was the per-channel geometric mean \(\mathbf{R}=\sqrt{\mathbf{A}\odot\mathbf{B}}\) (practical flash/no-flash recipe; not Lu & Drew 2006). Separately, following Lu and Drew (CIC 2006), pure-flash \(\mathbf{F}=\max(\mathbf{B}-\mathbf{A},0)\) and log-difference chromaticity yielded an ambient CCT used only for optional white-balance comparison images; spectral sharpening was omitted unless a calibrated sharpening matrix was available. Cheek mean CIELAB (D65) was taken from **R** under a MediaPipe tessellation skin mask and compared to FitSkin scanner cheek Lab via ΔE₀₀.
