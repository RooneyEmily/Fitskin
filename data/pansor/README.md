# Pansor iPhone cohort (2026-06-16)

Two participants (Emily / Liki), three trials each on **Sephora Bag** and **ColorChecker**.
FitSkin ground truth is **May 20 median cheek** (cross-session).

## Build manifest (local paths)

DNG exports are not in git. Point at your download folder and build the CSV:

```bash
export PANSOR_DATA_ROOT="/path/to/Pansor Images"
python3 scripts/build_pansor_manifest.py \
  --data-root "$PANSOR_DATA_ROOT" \
  --fitskin-mapping data/pansor/pansor_fitskin_mapping.csv \
  --out data/pansor/manifest_pansor_fitskin.csv
```

## Production evaluation

```bash
python3 flash_no_flash_skin_lab.py \
  --manifest data/pansor/manifest_pansor_fitskin.csv \
  --input-mode dng \
  --iphone-calibration ./calibration/tier3_affine \
  --cheek-roi --exposure-scale-skin-mask \
  --raw-camera-wb \
  --bag-cat02 auto \
  --production \
  --out-dir ./results/pansor_production

python3 evaluate_pansor_ablation.py \
  --manifest data/pansor/manifest_pansor_fitskin.csv \
  --out-dir ./results/pansor_ablation
```

Pinned ablation summary (bag `cat02_bag` vs `none`): `pansor_ablation_summary.json`.
