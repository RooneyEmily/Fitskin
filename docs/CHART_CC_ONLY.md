# ColorChecker skin color (original method)

**One photo. ColorChecker in frame. No flash/no-flash.**

This is `run_chart_cc.py` — the classical target-based pipeline (Wu et al.; Choi skin-weighted patches). It does **not** use:

- Flash / no-flash pairs or √(A⊙B) reflectance  
- Offline trained camera matrices  
- ISSA or demographic priors (unless you opt in)  
- FitSkin at capture time (FitSkin is only for evaluation in our manifests)

---

## Capture

1. Hold a **Macbeth ColorChecker Classic 24** in the same lighting as the face.  
2. Take **one** photo (no-flash frame is enough).  
3. Face + chart both visible.

ProRAW/DNG preferred; bundled JPEG cohort works for testing.

---

## Pipeline (in scene)

```
Image
  → detect 24-patch ColorChecker
  → gray white balance from chart (white or neutral column)
  → fit camera RGB → XYZ from 24 patch means vs canonical MCC D65
  → apply matrix to full frame
  → mean L*a*b* on face skin mask (cheek or mesh ROI)
  → output = skin tone estimate (CIELAB, D65)
```

No second exposure. No flash subtraction.

---

## Commands

### Bundled JPEG cohort (6 trials, in repo)

```bash
python3 run_chart_cc.py
# Outputs: chart_cc_output/comparison.csv
# Median ΔE₀₀ vs FitSkin cheek ≈ 4.9
```

Default: **cheek ROI**, **3×3** matrix, plain weighted least squares.

### Pansor iPhone DNG (local data)

```bash
python3 scripts/build_pansor_manifest.py --data-root "/path/to/Pansor Images"

python3 run_chart_cc.py \
  --input-mode dng \
  --manifest data/pansor/manifest_pansor_fitskin.csv \
  --cc-only \
  --no-include-flash \
  --out-dir chart_cc_output/pansor_dng
```

Use **`--no-include-flash`** so only the single no-flash frame is processed (not the flash mate).

### Zero-prior fixed recipe (same settings every person)

```bash
python3 run_chart_cc.py \
  --input-mode dng \
  --manifest data/pansor/manifest_pansor_fitskin.csv \
  --cc-only \
  --chart-only \
  --no-include-flash \
  --out-dir chart_cc_output/pansor_dng_chart_only
```

Mesh ROI + affine 3×4; median ΔE₀₀ ≈ **2.2** on Pansor vs FitSkin (cross-session).

### Best accuracy on mixed skin tones (uses preview L* — optional)

```bash
python3 run_chart_cc.py \
  --input-mode dng \
  --manifest data/pansor/manifest_pansor_fitskin.csv \
  --cc-only \
  --skin-tone auto \
  --no-include-flash \
  --out-dir chart_cc_output/pansor_dng_auto_tone
```

Median ΔE₀₀ ≈ **1.4** on Pansor. This is still chart-only correction; the only extra step is picking cheek vs mesh ROI from uncorrected preview L*.

---

## Results summary (Pansor DNG, N=6, vs FitSkin cheek)

| Mode | Flash/no-flash? | Median ΔE₀₀ |
|------|-----------------|-------------|
| **Chart CC (this doc)** | No | **1.4–4.6** depending on flags |
| Chart CC `--chart-only` | No | **2.2** |
| Chart CC default cheek | No | **4.6** |
| Flash/no-flash (`run_pipeline4.py`) | **Yes** | ~3.5–11.5 (different method) |

---

## Outputs

Under `--out-dir`:

| File | Content |
|------|---------|
| `comparison.csv` | Per-trial L*a*b*, ΔE₀₀ vs FitSkin (if in manifest) |
| `summary.json` | Mean/median ΔE₀₀ |
| `Lab_chart_cc_vs_fitskin_cheek.png` | Scatter plot |
| `skin_mask_overlays/noflash/` | Full face mesh tessellation (green tint) |
| `skin_mask_overlays/cheek_vs_mesh/` | **Green = cheek ROI, yellow = mesh-only**, white outline = cheek hull |
| `skin_lab_histograms/` | L*, a*, b* histogram panels (unless `--no-histograms`) |

Regenerate cheek overlays only:

```bash
python3 scripts/visualize_cheek_segmentation.py
```

---

## Cross-dataset results (unified FitSkin reference)

May-20 **median cheek** Lab per participant (same reference for all datasets):

| Images | `--chart-only` median ΔE₀₀ | Default cheek median ΔE₀₀ |
|--------|---------------------------|---------------------------|
| Pansor ProRAW DNG (Jun-16) | **2.2** | 4.6 |
| Bundled JPEG (May-20 booth) | 4.9 | 4.9 |

Chart-only wins on ProRAW; JPEG cohort is limited by small in-frame chart (~2% of frame) and display-referred chroma bias.

---

## Flags (chart method only)

| Flag | Effect |
|------|--------|
| `--no-include-flash` | Single frame only (recommended) |
| `--chart-only` | Mesh + affine; no `--skin-tone` routing |
| `--skin-tone auto` | Light→cheek+3×3, dark→mesh+affine |
| `--roi cheek\|mesh` | Skin sampling region |
| `--affine` | 3×4 fit instead of 3×3 |
| `--huber` | Robust patch fit (usually worse on this cohort) |

---

## Code entry points

| File | Role |
|------|------|
| `run_chart_cc.py` | CLI (`--chart-only`, `--legacy`, `--skin-tone auto`) |
| `scripts/chart_cc_fitskin_lib.py` | Detect CC → WB → matrix → skin Lab |
| `scripts/visualize_cheek_segmentation.py` | Cheek vs mesh ROI debug PNGs |
| `skin_tone_policy.py` | Optional `--skin-tone auto` only (not used with `--chart-only`) |

For flash/no-flash chart-free capture, see `run_pipeline4.py` / `flash_no_flash_skin_lab.py` — **different pipeline**.
