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

## Chart CC on ColorChecker DNGs (linear RAW)

Uses camera-linear RGB + in-scene MCC (better than JPEG preview path):

```bash
python3 run_chart_cc.py \
  --input-mode dng \
  --manifest data/pansor/manifest_pansor_fitskin.csv \
  --cc-only \
  --out-dir chart_cc_output/pansor_dng
```

**Best ΔE₀₀ on this cohort (Jun 2026 tweaks):** mesh ROI + affine fit  
(median **2.23** vs **4.64** cheek/3×3). P2 trials drop to **~0.9–1.7** ΔE₀₀.

```bash
python3 run_chart_cc.py \
  --input-mode dng \
  --manifest data/pansor/manifest_pansor_fitskin.csv \
  --cc-only \
  --roi mesh \
  --affine \
  --no-overlays \
  --out-dir chart_cc_output/pansor_dng_mesh_affine
```

Run all tweak benchmarks: `python3 scripts/benchmark_pansor_tweaks.py`  
→ `results/pansor_tweaks/summary.csv`

## Production evaluation

```bash
python3 flash_no_flash_skin_lab.py \
  --manifest data/pansor/manifest_pansor_fitskin.csv \
  --input-mode dng \
  --iphone-calibration ./calibration/tier3_affine \
  --exposure-scale-skin-mask \
  --raw-camera-wb \
  --bag-cat02 auto \
  --production \
  --out-dir ./results/pansor_production

python3 evaluate_pansor_ablation.py \
  --manifest data/pansor/manifest_pansor_fitskin.csv \
  --out-dir ./results/pansor_ablation
```

Pinned ablation summary (bag `cat02_bag` vs `none`): `pansor_ablation_summary.json`.
