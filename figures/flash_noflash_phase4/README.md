# Phase 4 flash / no-flash skin color — frozen paper bundle

Pinned outputs for **chart-free** (primary) and **FitSkin lightness reporting** (same trials, not independent validation) iPhone DNG flash/no-flash skin Lab vs FitSkin cheek scanner.

Regenerate after re-running the pipeline:

```bash
# Chart-free (primary agreement metric, median ΔE₀₀ ≈ 3.50)
OUT_DIR=./flash_noflash_tier3_affine_nocal ./run_flash_noflash_skin_lab_raw.sh

# Reporting calibration (median ΔE₀₀ ≈ 2.47; not held-out)
OUT_DIR=./flash_noflash_tier3_affine_fitskin ./run_flash_noflash_skin_lab_raw.sh \
  --fitskin-lightness-calibration

python3 bundle_flash_noflash_phase4.py
```

## Files

| File | Role |
|------|------|
| `flash_noflash_chartfree_per_trial.csv` | Slim per-trial Lab + ΔE₀₀ (no absolute paths) |
| `flash_noflash_fitskin_reporting_per_trial.csv` | Same, with FitSkin lightness gain column |
| `flash_noflash_chartfree_summary.json` | Run metadata + aggregate ΔE₀₀ |
| `flash_noflash_fitskin_reporting_summary.json` | Reporting run metadata |
| `flash_noflash_dual_reporting_stack.json` | Evolution table + paper wording |
| `flash_noflash_ablation_decomposition.json` | Tier-2 factorial (cheek ROI, skin exposure) |
| `flash_noflash_tier3_ablation.json` | Offline calibration ablation (affine vs lstsq, etc.) |
| `tables/phase4_chartfree.tex` | LaTeX for Table `tab:phase4_chartfree` |
| `tables/phase4_fitskin.tex` | LaTeX for Table `tab:phase4_fitskin` |
| `MANIFEST-sha256.txt` | Integrity checksums |

## Production stack (inference)

- Calibration: `calibration/tier3_affine/` (affine `[R,G,B,1]→XYZ`, exposure anchors)
- Script: `flash_no_flash_skin_lab.py` via `run_flash_noflash_skin_lab_raw.sh`
- Flags: `--cheek-roi`, `--exposure-scale-skin-mask`, `--exposure-anchor-from-training`, `--exclude-trials P2_T1`
- Booth metadata: `--known-ambient-cct-k 6546`, `--known-ambient-duv 0.0017`
- Optional reporting only: `--fitskin-lightness-calibration`

Methods text: `docs/FLASH_NOFLASH_SKIN_METHODS.md`. Overview: `README_FLASH_NOFLASH.md`.

## Cohort

$N = 5$ trials: P1 T1–T3, P2 T2–T3 (P2 T1 excluded). Reference: FitSkin cheek Lab (D65), metric `reflectance_cheek_de00`.

## Primary vs reporting

| Stack | Median ΔE₀₀ | Independent validation? |
|-------|-------------|-------------------------|
| Chart-free affine + cheek + skin exposure | **3.50** | Yes (no FitSkin fit on these trials) |
| Same + per-participant FitSkin $L^*$ gain | **2.47** | No (gains fit on same trials) |

Do **not** cite legacy pilot `flash_noflash_dng_output` (median 3.25) or failed Tier-3 variants (weighted/Huber/pre-WB) in main results.
