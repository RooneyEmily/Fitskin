# Flash / no-flash iPhone skin color (Phase 4)

**Public repo:** [github.com/RooneyEmily/Fitskin](https://github.com/RooneyEmily/Fitskin) — sync with `scripts/sync_to_fitskin_github.sh`.

Chart-free flash/no-flash geometric reflectance on iPhone DNG pairs, compared to **FitSkin cheek** scanner Lab (D65). Offline **ColorChecker** training produces the affine camera matrix; **inference does not use an in-scene chart**.

## Quick start

```bash
cd /path/to/emonet_feature_extraction

# 1) Train Tier-3 affine calibration (once; needs RAW pairs with visible MCC)
python3 train_flash_noflash_checker_calibration.py \
  --data-root "$RAW_DATA_ROOT" \
  --out-dir ./calibration/tier3_affine \
  --matrix-affine

# 2) Chart-free evaluation (primary result)
export RAW_DATA_ROOT="/path/to/RAW Dataset"   # Participant */Trial */*.DNG
OUT_DIR=./flash_noflash_tier3_affine_nocal ./run_flash_noflash_skin_lab_raw.sh

# 3) Optional reporting (FitSkin L* gains on same trials — not held-out)
OUT_DIR=./flash_noflash_tier3_affine_fitskin ./run_flash_noflash_skin_lab_raw.sh \
  --fitskin-lightness-calibration

# 4) Pin tables/CSVs for GitHub
python3 bundle_flash_noflash_phase4.py
```

`run_flash_noflash_skin_lab_raw.sh` auto-selects `calibration/tier3_affine` when `camera_rgb_to_xyz_affine.npy` exists.

## Results (locked bundle)

See **`figures/flash_noflash_phase4/`** (LaTeX tables, per-trial CSVs, `MANIFEST-sha256.txt`).

| Reporting mode | Median ΔE₀₀ vs FitSkin cheek | Notes |
|----------------|------------------------------|--------|
| **Chart-free** | **3.50** | Primary agreement; no FitSkin fit on eval trials |
| FitSkin lightness (same trials) | **2.47** | P1 gain 1.053, P2 gain 1.063; not independent validation |

Evolution and paper wording: `flash_noflash_dual_reporting_stack.json`.

## Key scripts

| Script | Role |
|--------|------|
| `flash_no_flash_skin_lab.py` | Main pipeline (DNG, align, reflectance, Lab, FitSkin compare) |
| `flash_noflash_face_roi.py` | Cheek hull mask |
| `flash_noflash_spectral.py` | Planck / ISSA helpers |
| `train_flash_noflash_checker_calibration.py` | Offline MCC matrix + exposure anchors |
| `run_flash_noflash_skin_lab_raw.sh` | Production wrapper |
| `run_tier3_calibration_ablation.py` | Train/eval calibration variants |
| `bundle_flash_noflash_phase4.py` | Refresh `figures/flash_noflash_phase4/` |
| `plot_flash_noflash_skin_vs_mst.py` | Optional MST comparison figures |

## Calibration bundle

**Production:** `calibration/tier3_affine/`

- `camera_rgb_to_xyz_affine.npy` — affine `[R,G,B,1]→XYZ` (use at inference)
- `exposure_anchor_by_participant.json` — white-patch scales from training
- `iphone_calibration_bundle.json` — metadata (matrix JSON duplicate for readers)

**Comparison only:** `calibration/tier3_lstsq_only/` (3×3 lstsq, no affine offset).

## Environment

- Python 3.10+
- `rawpy`, `opencv-python`, `numpy`, `scikit-image`, `mediapipe` (face mesh)
- FitSkin CSV mapping: set `--fitskin-scan-csv` / `--fitskin-mapping-csv` or disable with `--no-fitskin`
- Raw data layout: `Participant N/Trial M/IMG_*_NoFlash.DNG` and `*_Flash.DNG`

## Methods and results (paper text)

- `docs/FLASH_NOFLASH_SKIN_METHODS.md` — methodology
- `docs/FLASH_NOFLASH_SKIN_RESULTS.md` — results (ΔE₀₀ tables, interpretation, LaTeX draft)

## Git hygiene

Experimental run directories (`flash_noflash_*`, `flash_tier*`, …) are **gitignored**. Only `figures/flash_noflash_phase4/` and `calibration/tier3_affine/` (small `.npy`/JSON) are intended for the public repo.

## What not to ship in main results

- Legacy `flash_noflash_dng_output` (median 3.25; superseded)
- Tier-3 weighted lstsq / Huber stacked / `--reflectance-pre-wb booth` ablations (hurt or unstable)
