# Pansor production: Sephora bag CAT02

Validated on the **Pansor iPhone cohort** (2026-06-16): 2 participants × bag + ColorChecker × 3 trials, vs **May 20 FitSkin** cheek (cross-session).

## Locked production method

| Trial type | Chromatic correction |
|------------|---------------------|
| Sephora Bag (`condition_code=BAG`) | **`cat02_bag`** — bag white stripe → CAT02 matrix on linear RGB before reflectance |
| Color Checker (`CC`) | **`none`** — in-scene CC correction hurt ΔE on this dark-room data |

Reflectance path (all trials): geometric `sqrt(noflash × flash_aligned)` with tier-3 affine calibration, cheek ROI, skin-mask exposure scaling, and **`--raw-camera-wb`** for app-exported ProRAW.

## Pansor bag ablation (pinned)

From `data/pansor/pansor_ablation_summary.json` (6 bag trials):

| Mode | Mean ΔE₀₀ vs FitSkin |
|------|----------------------|
| `none` | 6.50 |
| **`cat02_bag`** | **5.02** |

CC trials (6): stay at `none` (~5.39 mean ΔE).

## Run production pipeline

```bash
# 1) Build manifest (DNG paths are local)
export PANSOR_DATA_ROOT="/path/to/Pansor Images"
python3 scripts/build_pansor_manifest.py --data-root "$PANSOR_DATA_ROOT"

# 2) Production output (reflectance only; bag CAT02 on BAG trials only)
python3 flash_no_flash_skin_lab.py \
  --manifest data/pansor/manifest_pansor_fitskin.csv \
  --input-mode dng \
  --iphone-calibration ./calibration/tier3_affine \
  --cheek-roi --exposure-scale-skin-mask \
  --raw-camera-wb \
  --bag-cat02 auto \
  --production \
  --out-dir ./results/pansor_production
```

NIX bag reference: `calibration/sephora_bag_nix_reference.json`.

## Key modules

| File | Role |
|------|------|
| `flash_no_flash_skin_lab.py` | Main pipeline; `--bag-cat02 auto`, `--production` |
| `bag_chromatic_correction.py` | CAT02 matrix from bag white stripe |
| `sephora_bag_reference.py` | Bag stripe segmentation (hands + optional MobileSAM) |
| `evaluate_pansor_ablation.py` | Re-run streamlined ablation (`none` + `cat02_bag`) |

## Implementation notes

- **`--bag-cat02 auto`** uses manifest `condition_code` / `condition` to skip CC trials (`skipped_not_bag_trial`).
- Stripe-only false positives on ColorChecker are rejected unless the trial is explicitly `BAG`.
- MobileSAM (`mobile_sam.pt`, not in git) is optional; set `MOBILE_SAM=0` to disable.
