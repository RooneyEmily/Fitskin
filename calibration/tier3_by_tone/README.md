# Skin-tone offline calibration bundles

Chart-free inference (`flash_no_flash_skin_lab.py`). When a ColorChecker is in frame, use chart CC with `--skin-tone auto`.

| Subdir | Training manifest | ISSA prior |
|--------|-------------------|------------|
| `dark/` | `data/training/manifest_tone_dark.csv` (Pansor ProRAW CC; booth/JPEG excluded — they bias L* high on ProRAW) | issa_median_south_asian (Liki, Indian) |
| `light/` | `data/training/manifest_tone_light.csv` (Pansor + booth RAW + chart_cc JPEG) | issa_median_caucasian |

```bash
python3 scripts/train_skin_tone_bundles.py

python3 flash_no_flash_skin_lab.py \
  --iphone-calibration-tone-root calibration/tier3_by_tone \
  --skin-tone auto \
  ...
```
