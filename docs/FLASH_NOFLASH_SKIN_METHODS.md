# Phase 4: iPhone flash / no-flash skin color — methodology (paper-ready)

Updated for the production stack (affine calibration, cheek ROI, skin-mask exposure, dual reporting vs FitSkin cheek scanner). Use section **“LaTeX Methods draft”** for manuscript paste-in. **Results:** `FLASH_NOFLASH_SKIN_RESULTS.md`. Code: [github.com/RooneyEmily/Fitskin](https://github.com/RooneyEmily/Fitskin).

---

## 1. Purpose and scope

We estimate facial skin color from paired **iPhone ProRAW (DNG)** captures with and without the device flash, and compare the result to **FitSkin** multi-point cheek measurements (CIELAB, D65). The **primary** endpoint is chart-free agreement: no in-scene Macbeth ColorChecker at inference and no FitSkin-derived fit on evaluation trials. A **secondary reporting** row applies per-participant lightness gains estimated from the same trials; that row is explicitly **not** independent validation.

**Cohort for locked results:** $N = 5$ trials (Participant 1 trials 1–3; Participant 2 trials 2–3). Participant 2 trial 1 is excluded (alignment / quality). Agreement is summarized as **ΔE₀₀** (CIEDE2000) between pipeline cheek Lab and FitSkin cheek Lab.

---

## 2. Acquisition

- **Device:** iPhone 17 Pro (ProRAW DNG).
- **Pairs per trial:** one no-flash and one flash image in the same booth session.
- **Booth illuminant (metadata):** $T \approx 6546\,\mathrm{K}$, $D_{uv} = 0.0017$ (fixed at inference for optional comparison paths; not applied inside the primary reflectance equation).
- **Reference:** FitSkin scan export mapped to trial IDs via a session-mapping CSV (cheek region Lab under D65).

RAW files are read as **linear camera RGB** (`rawpy`, camera white balance off, half-resolution demosaic for speed). A 99.5th-percentile per-channel scale maps linear RGB to $[0,1]$ for alignment and masking; reported reflectance statistics use the **linear** map before display encoding.

---

## 3. Offline calibration (training sessions only)

Calibration uses **separate trials** where a 24-patch Macbeth ColorChecker was visible in the no-flash frame. Inference on evaluation trials does **not** detect or use an in-scene chart.

### 3.1 Colorimetric matrix

Stacked least squares on all training patches (neutral and skin rows down-weighted relative to white/gray anchors) fits camera RGB to CIE XYZ (D65). The **production** model is an **affine** map on homogeneous coordinates:

\[
\begin{bmatrix} X \\ Y \\ Z \end{bmatrix}
=
\mathbf{M}_{\mathrm{aff}}
\begin{bmatrix} R \\ G \\ B \\ 1 \end{bmatrix},
\qquad
\mathbf{M}_{\mathrm{aff}} \in \mathbb{R}^{3 \times 4}.
\]

A plain $3 \times 3$ linear map (no offset) remains available for ablation (`calibration/tier3_lstsq_only/`). On the evaluation cohort, affine calibration reduced median ΔE₀₀ by **1.0** versus the Tier-2 lstsq matrix (4.50 → **3.50**). Weighted lstsq, Huber stacked fits, and ISSA-only training rows **increased** error and are not used in production.

### 3.2 Exposure anchor (chart-free at inference)

Per-participant scale factors are the **median** white-patch linear RGB scale across training checker trials (P1: 0.961; P2: 1.393). At inference, the reflectance map $\mathbf{R}$ is multiplied by the participant anchor before XYZ→Lab. This transfers absolute lightness from offline training without a visible chart in the evaluation frame.

### 3.3 Bundled assets

Production bundle: `calibration/tier3_affine/` (`camera_rgb_to_xyz_affine.npy`, `exposure_anchor_by_participant.json`, monochromator flash SPD and sensor sensitivity for optional SCR-AWB). Training script: `train_flash_noflash_checker_calibration.py --matrix-affine`.

---

## 4. Inference pipeline (chart-free production stack)

All steps below are applied on every evaluation trial unless noted.

### 4.1 Registration and flash exposure

Let **A** be no-flash linear RGB on the reference grid. The flash frame is warped onto **A** with **enhanced correlation coefficient (ECC)** alignment. Flash exposure is scaled so the **median luma on the facial skin mask** matches no-flash (not global scene scaling). Denote the aligned, exposure-matched flash linear RGB **B**. All reflectance formulas use **B**, not the raw flash capture.

### 4.2 Reflectance proxy (primary metric)

Per-pixel reflectance in linear RGB:

\[
\mathbf{R} = \sqrt{\max(\mathbf{A} \odot \mathbf{B}, \varepsilon)},
\]

with elementwise product $\odot$. This geometric mean is a **practical flash/no-flash skin recipe** used in this project; it is **not** the reflectance model of Lu & Drew (2006). Lu & Drew are cited only for **ambient illuminant estimation** on a comparison branch (Section 5).

### 4.3 Linear RGB → cheek CIELAB

1. $\mathbf{R}' = s_{\mathrm{anchor}}\,\mathbf{R}$ (participant exposure anchor from Section 3.2).
2. $\mathbf{R}' \rightarrow \mathrm{XYZ} \rightarrow L^*a^*b^*$ via $\mathbf{M}_{\mathrm{aff}}$ and D65 reference white.
3. **Cheek ROI:** mean Lab inside the intersection of (i) a MediaPipe Face Mesh tessellation skin mask and (ii) a convex **cheek hull** mask (`flash_noflash_face_roi.py`), with symmetric $L^*$, $a^*$, $b^*$ trims (5% each tail) and minimum skin chroma $C_{ab}^* \ge 2$. This spatial definition matches FitSkin cheek sampling better than a full-face mesh mean (median ΔE₀₀ 5.26 → 4.66 with lstsq matrix; skin-mask exposure adds 4.66 → 4.50 when cheek ROI is active).

Optional **reporting-only** step (Section 6): multiply $L^*$ by a per-participant gain $s_{\mathrm{FS}}$ fit to match median pipeline vs FitSkin $L^*$ on the **same** evaluation trials.

### 4.4 Face segmentation

MediaPipe Face Mesh on an 8-bit sRGB preview of aligned no-flash drives `build_skin_mask_from_mesh` (tessellation, inter-ocular-distance–scaled eye/lip/brow exclusion, morphological cleanup). The same mask family defines skin for ECC exposure scaling and for cheek/interior statistics.

---

## 5. Secondary and comparison paths (not primary results)

### 5.1 Lu & Drew (2006) ambient illuminant

Pure-flash $\mathbf{F} = \max(\mathbf{B} - \mathbf{A}, 0)$ and log-difference chromaticity yield an estimated ambient CCT for **optional** white-balance comparison images on no-flash previews. Flash CCT is auto-estimated from neutral pure-flash pixels. This estimate **does not** enter $\sqrt{\mathbf{A} \odot \mathbf{B}}$. Median cheek ΔE₀₀ on the evaluation cohort remains poor for no-flash-only and Lu-WB paths (e.g. Lu-WB median ≈ 23.8; no-flash median ≈ 25.7) relative to reflectance (3.50).

### 5.2 SCR-AWB (Zhou et al., 2025 spirit)

Single-frame AWB on no-flash linear RAW using monochromator sensor sensitivity, three-term Planckian illuminant basis, and an ISSA cheek reflectance prior (participant-specific ethnic median by default). **Comparison only**; requires `--scr-awb` and the calibration bundle.

### 5.3 Deprecated / rejected variants

- **Booth RGB pre-WB** before $\sqrt{\mathbf{A}\odot\mathbf{B}}$ breaks consistency with the D65-trained matrix; do not use.
- **Bradford CAT** on reflectance XYZ after the matrix: minimal gain in pilot; optional.
- **Weighted / Huber matrix training:** unstable (median ΔE₀₀ $\gg 10$).

---

## 6. Evaluation and dual reporting

### 6.1 Reference and metric

- **Reference:** FitSkin exported cheek $L^*a^*b^*$ (D65).
- **Metric:** ΔE₀₀ (CIEDE2000) between pipeline cheek mean and FitSkin cheek mean.
- **Offsets in tables:** pipeline minus FitSkin ($\Delta L^*$, $\Delta a^*$, $\Delta b^*$).

### 6.2 Primary result (chart-free)

Full stack: affine $\mathbf{M}_{\mathrm{aff}}$, cheek ROI, skin-mask ECC exposure, exposure anchor, booth metadata for comparisons only, **no** `--fitskin-lightness-calibration`.

| Statistic | Value |
|-----------|-------|
| Median ΔE₀₀ | **3.50** |
| Mean ΔE₀₀ | 3.33 |
| Median $\Delta L^*$ | −3.5 |
| Median $\Delta a^*$ | −0.4 |
| Median $\Delta b^*$ | −2.6 |

LaTeX table: `figures/flash_noflash_phase4/tables/phase4_chartfree.tex` (`tab:phase4_chartfree`).

### 6.3 Reporting calibration (same trials)

Two degrees of freedom: one multiplicative $L^*$ gain per participant ($s_{\mathrm{FS}}$: P1 = 1.053, P2 = 1.063), estimated by matching median pipeline vs FitSkin $L^*$ on the evaluation cohort. **Not held-out validation**—only for scanner-aligned reporting alongside the chart-free row.

| Statistic | Value |
|-----------|-------|
| Median ΔE₀₀ | **2.47** |
| Mean ΔE₀₀ | 2.27 |
| Median $\Delta L^*$ | −2.3 |

LaTeX table: `figures/flash_noflash_phase4/tables/phase4_fitskin.tex` (`tab:phase4_fitskin`).

The earlier pilot median 3.25 (lstsq matrix + FitSkin lightness, without cheek ROI / skin exposure / affine) is **superseded**; reporting 2.47 reflects both a better chart-free stack and smaller lightness gains.

### 6.4 How to phrase in the Results

- **Primary:** “Chart-free flash/no-flash reflectance with offline affine calibration achieved median ΔE₀₀ = 3.50 vs FitSkin cheek ($N=5$ trials).”
- **Reporting:** “Applying per-participant FitSkin lightness alignment on the same trials reduced median ΔE₀₀ to 2.47; this is reporting calibration, not independent validation.”
- **Gap:** Chart-free (3.50) remains ~1.0 ΔE₀₀ above reporting (2.47) on the same trials, consistent with uncorrected $L^*$ bias in the primary stack.

---

## 7. Ablation summary (development; optional in supplement)

| Stage | Median ΔE₀₀ | Independent? |
|-------|-------------|--------------|
| Mesh ROI, global flash exposure, lstsq $3\times3$ | 5.26 | Yes |
| + Cheek ROI | 4.66 | Yes |
| + Skin-mask flash exposure | 4.50 | Yes |
| + Affine $\mathbf{M}_{\mathrm{aff}}$ (chart-free) | **3.50** | Yes |
| + FitSkin $L^*$ gains (same trials) | **2.47** | No |

Factorial detail: `flash_noflash_ablation_decomposition.json`. Tier-3 training detail: `flash_noflash_tier3_ablation.json`. Paper bundle: `flash_noflash_dual_reporting_stack.json`.

---

## 8. Reproducibility

```bash
export RAW_DATA_ROOT="/path/to/RAW Dataset"
OUT_DIR=./flash_noflash_tier3_affine_nocal ./run_flash_noflash_skin_lab_raw.sh
OUT_DIR=./flash_noflash_tier3_affine_fitskin ./run_flash_noflash_skin_lab_raw.sh --fitskin-lightness-calibration
python3 bundle_flash_noflash_phase4.py
```

Frozen outputs: `figures/flash_noflash_phase4/` (CSVs, summaries, LaTeX, `MANIFEST-sha256.txt`).

---

## LaTeX Methods draft

### Subsection: Flash / no-flash iPhone skin color

Paired iPhone ProRAW (DNG) images were acquired with and without the built-in flash in a booth ($T \approx 6546\,\mathrm{K}$, $D_{uv} = 0.0017$). Images were demosaiced to linear camera RGB without in-camera white balance. The flash frame was registered to the no-flash frame by enhanced correlation coefficient alignment; flash intensity was scaled so median luminance on a MediaPipe face-mesh skin mask matched the no-flash frame. Per-pixel reflectance in linear RGB was $\mathbf{R} = \sqrt{\mathbf{A} \odot \mathbf{B}}$, where $\mathbf{A}$ is aligned no-flash RGB and $\mathbf{B}$ is aligned, exposure-matched flash RGB. This geometric mean is a practical flash/no-flash recipe and is not the reflectance model of Lu and Drew (2006).

A $3 \times 4$ affine map from $[R,G,B,1]^\top$ to CIE XYZ (D65) was fit offline from Macbeth ColorChecker patches in separate training captures (24 patches per training trial, unweighted stacked least squares with anchor down-weighting). Evaluation trials did not use an in-scene chart. Reflectance was scaled by a per-participant exposure anchor (median white-patch scale from training) before conversion to CIELAB. Cheek color was the trimmed mean of $L^*a^*b^*$ inside the intersection of the mesh skin mask and a cheek convex hull, with 5\% symmetric tails removed on each channel and a minimum chroma threshold ($C_{ab}^* \ge 2$).

Agreement with FitSkin was quantified as CIEDE2000 ($\Delta E_{00}$) between pipeline cheek Lab and FitSkin cheek Lab (D65). The primary analysis used the chart-free stack only ($N=5$ trials; one trial excluded for quality). A secondary reporting analysis multiplied $L^*$ by one gain per participant ($s_{\mathrm{FS}}$) estimated from the same trials to align lightness with the scanner; we report this as scanner-aligned calibration, not held-out validation. Lu and Drew (2006) log-difference chromaticity and SCR-AWB-style single-frame AWB were implemented as comparison paths only and were not used for the primary reflectance statistic.

### One-paragraph short form

After ECC alignment and skin-mask exposure matching, we formed $\mathbf{R} = \sqrt{\mathbf{A} \odot \mathbf{B}}$ on linear iPhone RAW, mapped $\mathbf{R}$ to CIELAB (D65) with an offline affine ColorChecker calibration and per-participant exposure anchor, and averaged inside a cheek ROI on a MediaPipe skin mask. Primary agreement with FitSkin cheek Lab was median $\Delta E_{00} = 3.50$ ($N=5$, chart-free). Per-participant FitSkin lightness alignment on the same trials yielded median $\Delta E_{00} = 2.47$ (reporting only).

---

## Citation hygiene

| Claim | Cite |
|-------|------|
| $\sqrt{\mathbf{A}\odot\mathbf{B}}$ reflectance | Internal / practical recipe — **not** Lu & Drew 2006 |
| Ambient CCT comparison | Lu & Drew, CIC 2006 |
| SCR-AWB comparison | Zhou et al., 2025 (as implemented) |
| ΔE₀₀ | CIEDE2000 standard |
| FitSkin reference | FitSkin device / export documentation |

Do **not** describe reporting 2.47 as validation of generalization. Do **not** cite legacy median 3.25 or rejected Tier-3 trainers in main results.
