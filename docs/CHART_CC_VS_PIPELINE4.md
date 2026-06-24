# Chart CC vs Pipeline 4 — why ΔE numbers differ

## Short answer

**Chart CC should beat chart-free on the same images.** It does.

| Method | Data | Median ΔE₀₀ vs FitSkin |
|--------|------|------------------------|
| Chart CC (`run_chart_cc.py`) | JPEG + ColorChecker in scene | **~4.9** |
| Flash/no-flash (`run_pipeline4.py`) on **same JPEGs** | Same 6 trials | **~11.5** |
| Flash/no-flash (`run_pipeline4.py`) on **booth DNG** | Chart-free booth RAW | **~3.5** |

Pipeline 4 on booth DNG is **not** comparable to chart CC on JPEG — different captures, processing (RAW vs JPEG), and lighting.

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
python3 run_chart_cc.py              # default: plain lstsq, overlays on
python3 run_chart_cc.py --huber        # previous Huber IRWS (~5.2 median)
python3 run_chart_cc.py --affine     # 3×4 affine fit (experimental)
python3 run_chart_cc.py --no-overlays
```
