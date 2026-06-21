# Sephora bag as in-scene white reference — feasibility note

**Data locations**

| Asset | Path |
|-------|------|
| Flash/no-flash pairs (Emily + Liki, 3 trials each) | `/media/mabl-main/Data/Bag image pairs/flash_noflash images/sephorabag_target/` |
| Duplicate / partial mirror | `/media/mabl-main/Data-Karthik/CameraColorProject/CameraColorProject/Sephora bag images/` |
| NIX Spectro 2 export (bag material) | `sephorabag_target/Sephorabag_measurement.csv` |
| CR250 spectra + XYZ summary | `.../Sephora bag measurement/CR250_Measurements/CR250_all_measurements.xlsx` |
| ColorChecker captures (same collection) | `.../colorchecker target/*.DNG` |

**Instruments:** NIX labels say *Nix Spectro 2*, M2, D65. Karthik folder also has CR250 `.mat` / Excel (9 spots: 3 bright, 3 mid, 3 dark by Y).

---

## 1. What you have

### NIX (bag material, not in-scene)

| Patch | $L^*$ | $a^*$ | $b^*$ | $Y$ (rel.) | Notes |
|-------|-------|-------|-------|------------|--------|
| Sephorawhite1–3 (mean) | **93.6** | −0.42 | 2.58 | 0.844 | Warm off-white paper |
| Sephorablack1–3 (mean) | **24.4** | 0.30 | 1.22 | 0.042 | Dark gray, not $L^*\approx 0$ |

White repeats are tight ($L^* \approx 93.6$–94.4). Black repeats are tight ($L^* \approx 24$–24.5). Good **material** reference.

### CR250 (Summary sheet)

| Index | $Y$ (instrument) | Interpretation |
|-------|------------------|----------------|
| 1–3 | ~497–501 | White stripe |
| 4–6 | ~397 | Likely second white area / angle |
| 7–9 | ~17 | Black stripe |

Use **ratios** (white/black $\approx 29.5$) for sanity checks; absolute XYZ units differ from NIX export scale.

### Images

- **6 paired trials:** Emily T1–T3, Liki T1–T3 (`*_Flash.DNG` / `*_NoFlash.DNG`).
- Bag is **in frame** on chest: horizontal black/white stripes, small format (~paper bag).
- Preview (Emily T1 no-flash): stripes are readable; **specular glare** on white stripes under flash.

---

## 2. Fit to current pipeline

Today, absolute lightness uses an **offline** exposure anchor from Macbeth **white patch** in training (`exposure_anchor_by_participant.json`, P1 ≈ 0.961, P2 ≈ 1.393). That transfers scale **without** a chart in the evaluation frame.

A Sephora **white stripe** could instead supply a **per-trial, in-scene** scale:

\[
s_{\mathrm{bag}} = \frac{Y_{\mathrm{NIX,white}}}{Y_{\mathrm{camera,white\_stripe}}}
\]

applied to reflectance linear RGB (same slot as `--exposure-anchor-from-training`), where $Y_{\mathrm{NIX,white}} = 0.844$ from the NIX export (D65).

**Black stripe** could support a two-point lightness model (offset + gain), but NIX black is only $L^*\approx 24$, so black-point correction is **limited**.

---

## 3. Pilot sampling (Emily T1, manual bag ROI)

On one readable pair (half-res align, u01 RAW, center bag ROI, bright/dark quantiles within ROI):

| Stripe | Camera $Y$ (u01) | Scale to NIX white $Y$ |
|--------|------------------|-------------------------|
| White (no-flash) | 0.319 | **2.65** |
| Black (no-flash) | 0.030 | — |

Affine $L^*$ on white stripe **before** NIX scaling was far below 93.6; after NIX-Y scaling, $L^*$ moves toward scanner white (exact value depends on ROI mask).

**Compared to MCC training anchor (P1 ≈ 0.961):** this pilot implies a **much larger** in-scene scale (~2.7×). That may mean:

1. Bag white in the photo is **not** the same as contact NIX white (distance, angle, vignetting, stripe vs. spot measurement).
2. ROI still mixes **skin, shirt, glare**, or gray stripes with white.
3. Scene u01 normalization is **global**, not bag-local (bag occupies small area).

→ **Feasible for exposure, but segmentation and geometry must be tightened before trusting numbers.**

---

## 4. Feasibility verdict

| Use case | Feasible? | Comment |
|----------|-----------|---------|
| **Per-trial exposure / $L^*$ anchor** (replace offline MCC anchor) | **Yes, with work** | Best match to “chart-free white in frame.” Needs reliable white-stripe mask; validate on all 6 trials vs FitSkin ΔE₀₀. |
| **D65 neutral white for chroma** | **Weak** | NIX white is warm ($b^*\approx 2.6$). OK for **luminance** scaling only. |
| **Black-point correction** | **Limited** | Black stripe $L^*\approx 24$ (textile), not true black. |
| **Replace 24-patch matrix calibration** | **No** | Only two colors; insufficient for full RGB→XYZ. Keep `tier3_affine` MCC training. |
| **Auto-detect like ColorChecker** | **No (today)** | No MCC geometry; need stripe/bag detector or fixed ROI + heuristics. |
| **Flash vs no-flash consistency** | **Caution** | Glare on white stripes under flash; prefer **$\sqrt{A\odot B}$ reflectance** on stripe pixels or **no-flash-only** for scale. |

**Overall:** Using the Sephora bag as an **in-scene luminance reference** is **reasonable and worth a Phase 4b experiment**. It is **not** a drop-in spectral substitute for Macbeth or for FitSkin cheek chroma.

---

## 5. Recommended experiment plan

### Step A — Lock spectral references

1. Parse `Sephorabag_measurement.csv` → `sephora_bag_nix_reference.json` (white/black Lab + XYZ).
2. Map CR250 indices 1–3 / 7–9 to white/black; cross-check $Y$ ratio vs NIX.

### Step B — Stripe segmentation

1. Detect bag bbox (manual ROI per trial, or color + aspect ratio + stripe frequency).
2. Within bbox: label **white** vs **black** pixels (k-means on $Y$ + low chroma, or horizontal stripe prior).
3. Exclude **specular** pixels: drop top 1% $Y$ within white mask on flash frames.
4. Export overlays for QC (`bag_stripe_mask.png`).

### Step C — Compare three exposure strategies (same 6 trials)

| Condition | Scale source |
|-----------|----------------|
| Baseline | `--exposure-anchor-from-training` (MCC, current) |
| Bag per-trial | $s_{\mathrm{bag}}$ from white stripe vs NIX |
| MCC in-scene | `colorchecker target/` DNGs if same booth/session |

**Metric:** ΔE₀₀ vs FitSkin cheek (if scans exist for Emily/Liki) or vs NIX spot on cheek if available.

### Step D — Paper language

- If bag scale **reduces** median $|\Delta L^*|$ without FitSkin lightness fit: “in-scene luminance reference.”
- If not: report as **negative result** (material white ≠ imaged white under booth flash).

---

## 6. Risks to call out in the paper

1. **Contact vs imaged:** NIX measures flat patch; camera sees curved bag, shadows, holder’s hands.
2. **Glare:** Flash pairs show hot spots on white stripes (see `sephora_bag_preview_emily_t1_noflash.png`).
3. **Not CIE white:** $L^*\approx 94$, $b^*\approx 2.6$ — scaling fixes mean level, not hue.
4. **Small area:** Few pixels vs face; global RAW u01 may dominate.
5. **Participant coverage:** Only Emily + Liki bag trials so far; not P1/P2 Phase 4 cohort unless same people.

---

## 7. Next code (repo)

Suggested module: `sephora_bag_reference.py`

- `load_nix_reference(csv) -> white/black Lab/XYZ`
- `segment_bag_stripes(bgr_or_linear) -> white_mask, black_mask`
- `exposure_scale_from_white_stripe(linear_rgb, mask, nix_y) -> float`
- Hook in `flash_no_flash_skin_lab.py` as `--exposure-anchor sephora-bag` (per-trial, overrides training anchor)

Reproduction after implementation:

```bash
export RAW_DATA_ROOT="/media/mabl-main/Data/Bag image pairs/flash_noflash images/sephorabag_target"
python3 flash_no_flash_skin_lab.py \
  --data-root "$RAW_DATA_ROOT" \
  --iphone-calibration ./calibration/tier3_affine \
  --cheek-roi --exposure-scale-skin-mask \
  --exposure-anchor sephora-bag \
  --nix-bag-reference ./sephora_bag_nix_reference.json \
  --out-dir ./flash_noflash_sephora_bag_anchor
```

---

## 8. Bottom line

| Question | Answer |
|----------|--------|
| Can the bag replace MCC **matrix** calibration? | **No** |
| Can white stripes replace MCC **per-trial exposure** anchor? | **Plausible — test on 6 pairs** |
| Is NIX data sufficient? | **Yes** for material white/black Lab |
| Biggest engineering gap? | **Robust stripe segmentation + glare rejection** |

**Recommendation:** Proceed as a **targeted ablation** (exposure anchor only), not a replacement for the Tier-3 affine + FitSkin evaluation stack, until ΔE₀₀ or $\Delta L^*$ improves on held-out skin references.

---

## 9. Phase 4b ablation update

Script:

```bash
python3 evaluate_sephora_bag_white_reference.py --sam2
```

Outputs:

- `sephora_bag_white_reference_eval/sephora_bag_white_reference_ablation.csv`
- `sephora_bag_white_reference_eval/sephora_bag_white_reference_summary.json`

The initial raw-luma bag anchors were not helpful, but calibrated XYZ-space bag anchors are better. The best mode is `bag_white_flash_aligned_xyz_lstsq`, which chooses the scalar exposure scale that best matches the full NIX white XYZ after the camera RGB→XYZ transform.

| Anchor mode | n | Mean ΔE₀₀ | Median ΔE₀₀ | Max ΔE₀₀ |
|-------------|---:|----------:|------------:|---------:|
| `bag_white_flash_aligned_xyz_lstsq` | 5 | **8.85** | **10.67** | **11.97** |
| `bag_white_flash_aligned_xyz_y` | 5 | 8.97 | 10.81 | 12.10 |
| `bag_white_noflash_xyz_lstsq` | 5 | 9.10 | 10.87 | 12.56 |
| `training_anchor` | 5 | 9.62 | 14.00 | 15.54 |
| `none` | 5 | 10.74 | 13.20 | 14.73 |
| raw bag luma anchors | 5 | 11.13-11.61 | 13.07-13.59 | 14.34-15.23 |

Because these bag captures are different sessions, the hybrid tests use only session-relative bag variation after centering by each participant's median bag scale. Those hybrid and L*-only modes mostly land near the existing `training_anchor`, so the strongest signal is not "bag as small correction to MCC"; it is "bag as calibrated in-frame XYZ exposure reference."

Current recommendation: keep the iPhone RGB→XYZ affine calibration, use SAM2 + hands for bag segmentation, and prefer `bag_white_flash_aligned_xyz_lstsq` for the bag-only chart-free exposure ablation. Report the session-mapping caveat clearly because the FitSkin comparison here is useful but not a perfectly matched same-session validation.
