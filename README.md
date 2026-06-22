# Fitskin — iPhone flash/no-flash skin color vs FitSkin scanner

Chart-free **flash / no-flash** geometric reflectance on iPhone DNG pairs, compared to **FitSkin cheek** scanner CIELAB (D65). Offline ColorChecker training produces an affine camera matrix; evaluation trials do not use an in-scene chart.

**Paper bundle (frozen tables):** [`figures/flash_noflash_phase4/`](figures/flash_noflash_phase4/)

| Mode | Median ΔE₀₀ vs FitSkin cheek | Independent validation? |
|------|------------------------------|-------------------------|
| Chart-free (primary) | **3.50** | Yes |
| + FitSkin lightness on same trials | **2.47** | No (reporting only) |

Cohort: $N=5$ trials (P1 T1–T3, P2 T2–T3; P2 T1 excluded).

## Pansor bag CAT02 (production extension)

Validated on the **Pansor iPhone cohort** (2026-06-16). On Sephora Bag trials, in-scene **`cat02_bag`** chromatic correction improves mean ΔE₀₀ vs FitSkin from **6.50 → 5.02** (6 bag trials). ColorChecker trials stay uncorrected.

See [`docs/PANSOR_BAG_CAT02.md`](docs/PANSOR_BAG_CAT02.md) and pinned summary [`data/pansor/pansor_ablation_summary.json`](data/pansor/pansor_ablation_summary.json).

```bash
export PANSOR_DATA_ROOT="/path/to/Pansor Images"
python3 scripts/build_pansor_manifest.py --data-root "$PANSOR_DATA_ROOT"

python3 flash_no_flash_skin_lab.py \
  --manifest data/pansor/manifest_pansor_fitskin.csv \
  --input-mode dng \
  --iphone-calibration ./calibration/tier3_affine \
  --cheek-roi --exposure-scale-skin-mask \
  --raw-camera-wb --bag-cat02 auto --production \
  --out-dir ./results/pansor_production
```

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Reproduce

```bash
export RAW_DATA_ROOT="/path/to/RAW Dataset"   # Participant */Trial */*.DNG

# Chart-free evaluation (primary)
OUT_DIR=./flash_noflash_tier3_affine_nocal ./run_flash_noflash_skin_lab_raw.sh

# Optional reporting (same-trial FitSkin L* gains — not held-out)
OUT_DIR=./flash_noflash_tier3_affine_fitskin ./run_flash_noflash_skin_lab_raw.sh \
  --fitskin-lightness-calibration

# Refresh committed tables / CSVs
python3 bundle_flash_noflash_phase4.py
```

Production calibration: [`calibration/tier3_affine/`](calibration/tier3_affine/).

Train calibration from checker sessions:

```bash
python3 train_flash_noflash_checker_calibration.py \
  --data-root "$RAW_DATA_ROOT" \
  --out-dir ./calibration/tier3_affine \
  --matrix-affine
```

## FitSkin reference data

Provide scan export CSV and trial mapping (not shipped in this repo):

```bash
python3 flash_no_flash_skin_lab.py \
  --data-root "$RAW_DATA_ROOT" \
  --fitskin-scan-csv /path/to/scan-sessions.csv \
  --fitskin-mapping-csv /path/to/noflash_pairs_fitskin_mapping.csv \
  ...
```

Use `--no-fitskin` for pipeline-only Lab output.

## Layout

| Path | Role |
|------|------|
| `flash_no_flash_skin_lab.py` | Main pipeline |
| `bag_chromatic_correction.py` | Sephora bag CAT02 (Pansor production) |
| `run_flash_noflash_skin_lab_raw.sh` | Production wrapper |
| `evaluate_pansor_ablation.py` | Pansor streamlined ablation |
| `bundle_flash_noflash_phase4.py` | Pin `figures/flash_noflash_phase4/` |
| `docs/FLASH_NOFLASH_SKIN_METHODS.md` | Paper methods text |
| `docs/PANSOR_BAG_CAT02.md` | Pansor bag CAT02 production notes |
| `figures/flash_noflash_phase4/tables/*.tex` | LaTeX tables |
| `flash_noflash_dual_reporting_stack.json` | Evolution + wording |

## Methods

See [`docs/FLASH_NOFLASH_SKIN_METHODS.md`](docs/FLASH_NOFLASH_SKIN_METHODS.md). Primary metric: $\mathbf{R}=\sqrt{\mathbf{A}\odot\mathbf{B}}$ on aligned linear RAW (not Lu & Drew 2006 reflectance).

## Citation

If you use this code, cite the associated manuscript and pin the git tag or commit SHA used for reported ΔE₀₀ values (`figures/flash_noflash_phase4/MANIFEST-sha256.txt`).

## Related repos

- [RooneyEmily/color-emotion-physio](https://github.com/RooneyEmily/color-emotion-physio) — broader physio / circumplex work
