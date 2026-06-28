# Chart CC vs Pipeline 4 — why ΔE numbers differ

## Short answer

**Chart CC should beat chart-free on the same images.** It does.

| Method | Data | Median ΔE₀₀ vs FitSkin |
|--------|------|------------------------|
| Chart CC (`run_chart_cc.py`) | JPEG + ColorChecker in scene | **~4.9** |
| Chart CC + **`--skin-tone auto`** | Pansor ProRAW DNG ($N=6$ CC trials) | **~1.4** |
| Chart CC fixed cheek ROI (no auto) | Same Pansor DNG | **~4.6** |
| Flash/no-flash (`run_pipeline4.py`) on **same JPEGs** | Same 6 trials | **~11.5** |
| Flash/no-flash (`run_pipeline4.py`) on **booth DNG** | Chart-free booth RAW | **~3.5** |

Pipeline 4 on booth DNG is **not** comparable to chart CC on JPEG — different captures, processing (RAW vs JPEG), and lighting.

## Pansor DNG results (ColorChecker in scene, Jun 2026)

Ground truth: FitSkin May 20 median cheek Lab (cross-session). Six trials: P1_CC_T1–T3 (Emily), P2_CC_T1–T3 (Likitha, Indian).

| Trial | Fixed cheek + 3×3 ΔE₀₀ | **`--skin-tone auto`** ΔE₀₀ | Auto tier | Auto ROI |
|-------|------------------------|-------------------------------|-----------|----------|
| P1_CC_T1 | 5.04 | 5.04 | light | cheek |
| P1_CC_T2 | 2.25 | 2.25 | light | cheek |
| P1_CC_T3 | 1.09 | 1.09 | light | cheek |
| P2_CC_T1 | 4.39 | **0.96** | dark | mesh + affine |
| P2_CC_T2 | 5.95 | **1.73** | dark | mesh + affine |
| P2_CC_T3 | 4.89 | **0.93** | dark | mesh + affine |
| **Median** | **4.64** | **1.41** | | |

**Skin-tone auto** probes preview cheek $L^*$ (threshold 42): lighter skin → cheek ROI + $3\times3$; darker skin → face-mesh ROI + $3\times4$ affine. This fixes Likitha's $L^*$ overshoot (cheek hull read ~57 vs FitSkin ~52).

### Reproduce (Pansor DNG)

```bash
# 1. Build manifest (paths are local; not committed)
python3 scripts/build_pansor_manifest.py \
  --data-root "/path/to/Pansor Images"

# 2. Best chart CC (skin-tone auto)
python3 run_chart_cc.py \
  --input-mode dng \
  --manifest data/pansor/manifest_pansor_fitskin.csv \
  --cc-only \
  --skin-tone auto \
  --out-dir chart_cc_output/pansor_dng_auto_tone

# 3. Baseline (fixed cheek, no auto) — for comparison
python3 run_chart_cc.py \
  --input-mode dng \
  --manifest data/pansor/manifest_pansor_fitskin.csv \
  --cc-only \
  --out-dir chart_cc_output/pansor_dng
```

Outputs: `comparison.csv`, cheek Lab scatter plot, skin mask overlays under `chart_cc_output/`.

## What chart CC does

1. Detect ColorChecker in the JPEG  
2. White-balance from chart white patch  
3. Fit 3×3 (linear sRGB → canonical MCC D65 XYZ) from **24 patch means**  
4. Apply matrix to full frame → cheek ROI Lab (D65)  
5. Compare to FitSkin cheek Lab  

Default fit is **plain weighted least squares** (not Huber) — best on this cohort.

## Remaining gap vs FitSkin (especially a*)

Documented in `mabl-flash-illumination/docs/CHROMA_FITSKIN_INVESTIGATION.md`:

- iPhone JPEG is display-referred; skin a*/b* are often higher than FitSkin probe  
- Cheek geometry vs scanner probe  
- P2_T1: FitSkin scan May 8 vs photos May 24 (session mismatch)  

Chart patch fit quality: mean patch ΔE_ab ≈ 9–10 on the 24 patches (canonical MCC reference).

## Flags

```bash
python3 run_chart_cc.py              # default: bundled JPEG, plain lstsq, overlays on
python3 run_chart_cc.py --input-mode dng --manifest data/pansor/manifest_pansor_fitskin.csv --cc-only
python3 run_chart_cc.py --skin-tone auto   # recommended for mixed light/dark skin (DNG or JPEG)
python3 run_chart_cc.py --huber        # previous Huber IRWS (~5.2 median on JPEG)
python3 run_chart_cc.py --affine     # force 3×4 affine (auto sets this for dark tier)
python3 run_chart_cc.py --no-overlays
```

See `skin_tone_policy.py` for tier routing logic (preview $L^* \ge 42$ → light/cheek, else dark/mesh+affine).
