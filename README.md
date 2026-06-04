# Fitskin — iPhone flash/no-flash skin color vs FitSkin scanner

Chart-free **flash / no-flash** geometric reflectance on iPhone DNG pairs, compared to **FitSkin cheek** scanner CIELAB (D65). Offline ColorChecker training produces an affine camera matrix; evaluation trials do not use an in-scene chart.

**Paper bundle (frozen tables):** [`figures/flash_noflash_phase4/`](figures/flash_noflash_phase4/)

| Mode | Median ΔE₀₀ vs FitSkin cheek | Independent validation? |
|------|------------------------------|-------------------------|
| Chart-free (primary) | **3.50** | Yes |
| + FitSkin lightness on same trials | **2.47** | No (reporting only) |

Cohort: $N=5$ trials (P1 T1–T3, P2 T2–T3; P2 T1 excluded).

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
| `run_flash_noflash_skin_lab_raw.sh` | Production wrapper |
| `bundle_flash_noflash_phase4.py` | Pin `figures/flash_noflash_phase4/` |
| `docs/FLASH_NOFLASH_SKIN_METHODS.md` | Paper methods text |
| `figures/flash_noflash_phase4/tables/*.tex` | LaTeX tables |
| `flash_noflash_dual_reporting_stack.json` | Evolution + wording |

## Methods

See [`docs/FLASH_NOFLASH_SKIN_METHODS.md`](docs/FLASH_NOFLASH_SKIN_METHODS.md). Primary metric: $\mathbf{R}=\sqrt{\mathbf{A}\odot\mathbf{B}}$ on aligned linear RAW (not Lu & Drew 2006 reflectance).

## Citation

If you use this code, cite the associated manuscript and pin the git tag or commit SHA used for reported ΔE₀₀ values (`figures/flash_noflash_phase4/MANIFEST-sha256.txt`).

## Related repos

- [RooneyEmily/color-emotion-physio](https://github.com/RooneyEmily/color-emotion-physio) — broader physio / circumplex work
- [RooneyEmily/emonet_feature_extraction](https://github.com/RooneyEmily/emonet_feature_extraction) — EmoNet integration mirror
