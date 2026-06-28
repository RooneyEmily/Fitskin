# Original ColorChecker pipeline (no flash/no-flash)

The **original** in-scene ColorChecker skin color code lives in **`mabl-flash-illumination`**, not in the flash/no-flash reflectance path.

## Canonical old code

| File | Role |
|------|------|
| [`mabl-flash-illumination/scripts/run_fitskin_chart_cc_pipeline.py`](../../mabl-flash-illumination/scripts/run_fitskin_chart_cc_pipeline.py) | CLI entry point |
| [`mabl-flash-illumination/scripts/chart_cc_fitskin_lib.py`](../../mabl-flash-illumination/scripts/chart_cc_fitskin_lib.py) | Detect CC → WB → 3×3 → cheek Lab |
| [`mabl-flash-illumination/docs/FITSKIN_CHART_CC_PIPELINE.md`](../../mabl-flash-illumination/docs/FITSKIN_CHART_CC_PIPELINE.md) | Full documentation |

**Not** flash/no-flash: see `run_pipeline4.py` / `flash_no_flash_skin_lab.py` for √(A⊙B) reflectance.

### Run the original (mabl-flash-illumination)

```bash
cd ../mabl-flash-illumination

# One no-flash frame per trial (recommended)
python3 scripts/run_fitskin_chart_cc_pipeline.py --no-include-flash

# Default also processes flash frames from the same manifest pairs
python3 scripts/run_fitskin_chart_cc_pipeline.py
```

Manifest: `data/manifest_noflash_pairs_fitskin.csv` (absolute paths to Downloads JPEG pairs).  
Output: `results/fitskin_chart_cc/`

### Original algorithm (single photo)

1. Detect Macbeth ColorChecker Classic 24  
2. White-patch gray balance  
3. **Huber-weighted** 3×3 linear RGB → canonical MCC D65 XYZ  
4. Apply to frame → Bradford CAT to D65  
5. Mean Lab on **cheek ROI** (default)  
6. Compare to FitSkin cheek Lab (evaluation only)

No flash pair. No offline matrix. No ISSA.

---

## Fitskin copy (`run_chart_cc.py`)

Fitskin vendors the same library under `scripts/chart_cc_fitskin_lib.py` plus bundled JPEGs in `data/chart_cc_jpeg/`.

To match **old defaults** inside Fitskin:

```bash
cd Fitskin
python3 run_chart_cc.py --legacy --no-include-flash
```

`--legacy` → cheek ROI + Huber fit (same as original).  
Later additions (`--input-mode dng`, `--skin-tone auto`, `--chart-only`) are optional; ignore them for the original method.

### Bundled JPEG quickstart (original-style)

```bash
python3 run_chart_cc.py --legacy --no-include-flash
# → chart_cc_output/comparison.csv
```

---

## What changed in the Fitskin fork

| Setting | Original (`run_fitskin_chart_cc_pipeline.py`) | Fitskin default today |
|---------|-----------------------------------------------|------------------------|
| Patch fit | Huber (`--no-huber` to disable) | Plain lstsq (`--huber` or `--legacy`) |
| ROI | Cheek | Cheek (`--legacy`) or mesh (`--chart-only`) |
| Input | JPEG paths in manifest | JPEG bundled + optional DNG (Pansor) |
| Flash frames | Optional (`--no-include-flash`) | Same |
| Skin-tone routing | None | `--skin-tone auto` (new) |

---

## Related (debug only, not primary)

- `mabl-flash-illumination/scripts/run_chart_pipeline_vs_fitskin.py` — PR-250 spectrometer reference path (booth lighting; **not** the FitSkin JPEG validation pipeline).
