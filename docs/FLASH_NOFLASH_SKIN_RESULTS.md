# Phase 4: iPhone flash / no-flash skin color — results (paper-ready)

Companion to `FLASH_NOFLASH_SKIN_METHODS.md`. Frozen numbers from `figures/flash_noflash_phase4/`. Code: [github.com/RooneyEmily/Fitskin](https://github.com/RooneyEmily/Fitskin).

---

## 1. Overview

We evaluated chart-free flash/no-flash skin color on **five** iPhone DNG trial pairs (Participant 1, trials 1–3; Participant 2, trials 2–3) against **FitSkin cheek** CIELAB (D65). Participant 2 trial 1 was excluded due to alignment quality. The **primary** endpoint is CIEDE2000 (ΔE₀₀) under the production stack: affine offline calibration, geometric reflectance $\mathbf{R}=\sqrt{\mathbf{A}\odot\mathbf{B}}$, cheek ROI, skin-mask flash exposure scaling, and per-participant exposure anchor from training—**without** in-scene ColorChecker or FitSkin-derived fits on these trials.

A **secondary reporting** analysis applies one multiplicative $L^*$ gain per participant ($s_{\mathrm{FS}}$) estimated from the **same** five trials; we report it only as scanner-aligned calibration, not as held-out validation.

---

## 2. Primary result: chart-free agreement

Table `tab:phase4_chartfree` lists per-trial pipeline and FitSkin cheek coordinates, per-channel offsets (pipeline − FitSkin), and ΔE₀₀.

### 2.1 Summary statistics

| Statistic | ΔE₀₀ | $\Delta L^*$ | $\Delta a^*$ | $\Delta b^*$ |
|-----------|------|--------------|--------------|--------------|
| **Median** | **3.50** | −3.5 | −0.4 | −2.6 |
| Mean | 3.33 | −3.4 | −0.7 | −2.4 |
| Range (trials) | 2.75 – 3.80 | −4.0 – −2.7 | −1.5 – −0.2 | −2.8 – −2.0 |

Pipeline cheek $L^*$ was **lower** than FitSkin on every trial (median $\Delta L^* = -3.5$). Chroma offsets were modest except Participant 2 trial 2 ($\Delta a^* = -1.5$). The pipeline underestimated $b^*$ consistently (median $\Delta b^* = -2.6$), indicating residual yellow–blue disagreement after affine calibration.

### 2.2 Per-trial ΔE₀₀

| Trial | ΔE₀₀ | Dominant offset |
|-------|------|-----------------|
| P1 T1 | 3.50 | $\Delta L^*=-4.0$, $\Delta b^*=-2.1$ |
| P1 T2 | 2.75 | $\Delta L^*=-3.0$ (best trial) |
| P1 T3 | 3.53 | $\Delta L^*=-3.9$, $\Delta b^*=-2.6$ |
| P2 T2 | 3.09 | $\Delta a^*=-1.5$, $\Delta b^*=-2.7$ |
| P2 T3 | 3.80 | $\Delta L^*=-3.5$, $\Delta b^*=-2.8$ (worst trial) |

Participant 1 cluster near $L^* \approx 65$ (pipeline) vs $69$ (FitSkin); Participant 2 near $49$–$50$ vs $52$–$53$. Affine calibration and cheek ROI brought agreement into a **mid-single-digit ΔE₀₀** band; no trial exceeded 4.0 under the chart-free stack.

### 2.3 Interpretation (primary)

Chart-free flash/no-flash reflectance with offline affine ColorChecker training achieved **median ΔE₀₀ = 3.50** vs FitSkin cheek ($N=5$). Errors are **lightness-dominated**; chroma agreement is materially better than early pipeline variants (full-face mesh, global flash exposure: median 5.26). The remaining gap is consistent with (i) different sensing geometries (iPhone fusion vs spatially resolved scanner cheek), (ii) exposure-anchor transfer without an in-trial white reference, and (iii) uncorrected per-participant $L^*$ scale.

---

## 3. Reporting result: FitSkin lightness alignment (same trials)

Table `tab:phase4_fitskin` applies per-participant gains $s_{\mathrm{FS}}$ (P1 = 1.053, P2 = 1.063) fit to align median pipeline vs FitSkin $L^*$ on these trials.

### 3.1 Summary statistics

| Statistic | ΔE₀₀ | $\Delta L^*$ | $\Delta a^*$ | $\Delta b^*$ |
|-----------|------|--------------|--------------|--------------|
| **Median** | **2.47** | −2.3 | −0.2 | −2.2 |
| Mean | 2.27 | −2.2 | −0.4 | −2.1 |
| Range (trials) | 1.80 – 2.60 | −2.7 – −1.5 | −1.2 – −0.1 | −2.3 – −1.8 |

### 3.2 Per-trial ΔE₀₀

| Trial | ΔE₀₀ | $s_{\mathrm{FS}}$ |
|-------|------|-------------------|
| P1 T1 | 2.47 | 1.053 |
| P1 T2 | 1.80 | 1.053 |
| P1 T3 | 2.55 | 1.053 |
| P2 T2 | 1.94 | 1.063 |
| P2 T3 | 2.60 | 1.063 |

Lightness calibration reduced median $\Delta L^*$ from −3.5 to −2.3 and median ΔE₀₀ from **3.50 to 2.47** (−1.03). Gains are **much smaller** than an earlier lstsq-only pilot (P1 ≈ 1.09, P2 ≈ 1.22) because the affine + cheek + skin stack already partially closes the $L^*$ gap.

### 3.3 Interpretation (reporting)

Same-trial FitSkin lightness alignment yields **median ΔE₀₀ = 2.47**. This row must **not** be described as independent validation: two parameters were estimated from the evaluation cohort’s own pipeline–scanner $L^*$ offsets. It is appropriate for **scanner-aligned reporting** alongside the chart-free primary metric. Residual error after calibration remains chroma-heavy ($\Delta b^* \approx -2.2$ median); further improvement would require chroma or exposure modeling, not $L^*$ scaling alone.

### 3.4 Chart-free vs reporting on the same trials

| Stack | Median ΔE₀₀ | Δ vs chart-free |
|-------|-------------|-----------------|
| Chart-free (primary) | 3.50 | — |
| + FitSkin $L^*$ gains | 2.47 | −1.03 |

The ~1.0 ΔE₀₀ separation between primary and reporting rows is expected: reporting absorbs systematic $L^*$ bias that the chart-free stack leaves uncorrected on evaluation trials.

---

## 4. Development ablations (context for primary stack)

Median ΔE₀₀ vs FitSkin cheek on the same $N=5$ cohort (trial exclusions held constant):

| Stage | Median ΔE₀₀ | Notes |
|-------|-------------|--------|
| Mesh ROI, global flash exposure, lstsq $3\times3$ | 5.26 | Spatial mismatch with FitSkin cheek |
| + Cheek ROI only | 4.66 | −0.59 |
| + Skin-mask flash exposure | 4.50 | −0.17 vs cheek-only |
| + Affine $[R,G,B,1]\!\to\!$XYZ (chart-free) | **3.50** | −1.0 vs 4.50; **primary** |
| + FitSkin $L^*$ gains (same trials) | **2.47** | Reporting only |

**Factorial:** Skin-mask exposure alone did not improve the cohort median when Lab was still averaged over the full mesh (5.30 vs 5.26 baseline). Gain appeared when cheek ROI was active (4.66 → 4.50).

**Tier-3 training:** Only affine matrix training helped; weighted lstsq and Huber stacked training produced medians >10 ΔE₀₀ and were rejected.

**Superseded pilot:** An earlier run (lstsq matrix + FitSkin lightness, before cheek ROI / skin exposure / affine) reported median 3.25; that figure should **not** be cited as the current primary or reporting result.

---

## 5. Comparison paths (not primary)

On the chart-free evaluation cohort, alternative pipelines remained far from FitSkin cheek agreement (medians from the chart-free run summary):

| Path | Median ΔE₀₀ |
|------|-------------|
| **Reflectance (primary)** | **3.50** |
| Lu booth WB (known 6546 K) | 17.2 |
| Lu estimated ambient WB | 23.8 |
| No-flash linear only | 25.7 |
| Flash aligned only | 29.2 |

These confirm that single-frame or WB-only processing does not substitute for flash/no-flash reflectance fusion under affine calibration.

---

## 6. LaTeX Results draft

### Subsection: Agreement with FitSkin cheek color

We compared cheek-region CIELAB from the chart-free flash/no-flash pipeline to FitSkin cheek measurements on five iPhone trial pairs (Table~\ref{tab:phase4_chartfree}). The primary stack—offline affine ColorChecker calibration, geometric reflectance fusion, cheek ROI, skin-mask flash exposure scaling, and training-derived exposure anchors, with no in-scene chart and no FitSkin fit on evaluation trials—achieved a **median $\Delta E_{00} = 3.50$** (mean 3.33; range 2.75–3.80). Pipeline $L^*$ was below FitSkin on all trials (median $\Delta L^* = -3.5$); median chroma offsets were $\Delta a^* = -0.4$ and $\Delta b^* = -2.6$. The best single trial was P1 trial 2 ($\Delta E_{00} = 2.75$); the largest error was P2 trial 3 ($\Delta E_{00} = 3.80$), where $\Delta a^* = -1.0$ and $\Delta b^* = -2.8$ contributed alongside lightness.

For scanner-aligned reporting, we applied one multiplicative $L^*$ gain per participant ($s_{\mathrm{FS}} = 1.053$ and $1.063$) estimated from the same five trials (Table~\ref{tab:phase4_fitskin}). This reduced median $\Delta E_{00}$ to **2.47** (mean 2.27; range 1.80–2.60) and median $\Delta L^*$ to $-2.3$. We treat this as **calibration for comparison to the scanner**, not held-out validation. After lightness alignment, residual disagreement was still driven primarily by $b^*$ (median $\Delta b^* = -2.2$).

Relative to earlier pipeline variants on the same cohort, cheek ROI and skin-mask exposure reduced median $\Delta E_{00}$ from 5.26 to 4.50 before affine calibration; the affine map further reduced the median to 3.50. Single-frame no-flash and Lu-style white-balance comparison paths yielded medians above 17 $\Delta E_{00}$ and were not competitive with reflectance fusion.

### Short paragraph (Results)

Chart-free flash/no-flash skin color agreed with FitSkin cheek measurements with median $\Delta E_{00} = 3.50$ ($N=5$ trials). Errors were dominated by pipeline lightness below the scanner (median $\Delta L^* = -3.5$). Per-participant FitSkin lightness alignment on the same trials lowered median $\Delta E_{00}$ to 2.47 for reporting purposes only. Development ablations showed that cheek ROI, skin-mask flash scaling, and affine offline calibration were each necessary to reach this agreement band; white-balance-only baselines remained above 17 $\Delta E_{00}$.

---

## 7. Suggested figure / table callouts

| Artifact | Caption hook |
|----------|----------------|
| `tab:phase4_chartfree` | Primary agreement; chart-free stack |
| `tab:phase4_fitskin` | Same-trial lightness calibration; not validation |
| Optional supplement | Ablation staircase 5.26 → 4.50 → 3.50 → 2.47 |

---

## 8. What not to say in Results

- Do **not** call median 2.47 “validation accuracy” or “generalization.”
- Do **not** cite median 3.25 from the legacy pilot as the current reporting result.
- Do **not** report weighted/Huber matrix or pre-WB booth ablations as failed main-line results without labeling them as rejected development runs.
- Prefer **median** ΔE₀₀ for the primary summary (mean 3.33 is pulled by cross-participant $L^*$ scale); report both if space allows.

---

## 9. Reproducibility

Per-trial CSVs: `figures/flash_noflash_phase4/flash_noflash_chartfree_per_trial.csv`, `flash_noflash_fitskin_reporting_per_trial.csv`. Summaries: `flash_noflash_chartfree_summary.json`, `flash_noflash_fitskin_reporting_summary.json`. Checksums: `MANIFEST-sha256.txt`.
