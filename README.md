# Fitskin — iPhone flash/no-flash skin color vs FitSkin scanner

Chart-free **flash / no-flash** geometric reflectance on iPhone DNG pairs, compared to **FitSkin cheek** scanner CIELAB (D65).

## Reproduce (clone → install → run)

Everything needed is in this repo: code, calibration, FitSkin reference Lab, and booth RAW DNGs.

```bash
git clone https://github.com/RooneyEmily/Fitskin.git
cd Fitskin
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
python3 -m pip install --upgrade pip
pip install -r requirements.txt

# Full 5-trial cohort (bundled RAW under data/phase4_booth_raw/)
python3 run_pipeline4.py

# Single trial smoke test (target ΔE₀₀ ≈ 2.75)
python3 run_pipeline4.py --trial P1_T2 \
  data/phase4_booth_raw/Participant\ 1/Trial\ 2/IMG_0787_NoFlash.DNG \
  data/phase4_booth_raw/Participant\ 1/Trial\ 2/IMG_0786_Flash.DNG
```

**`run_pipeline4.py`** runs the production stack (affine calibration, cheek ROI, skin-mask exposure, exposure anchor) and prints pipeline vs FitSkin Lab plus ΔE₀₀ per trial.

| Target | Median ΔE₀₀ |
|--------|-------------|
| 5-trial Phase 4 cohort (excludes P2_T1) | **≈ 3.50** |
| Single trial P1_T2 (typical) | **≈ 2.75** |

FitSkin reference values: `data/phase4_fitskin_reference.csv` (same-session scanner cheek Lab from the paper).

Booth RAW DNGs: `data/phase4_booth_raw/` (~500 MB, 12 files). You can also point at your own copy:

```bash
python3 run_pipeline4.py /path/to/RAW/Dataset
```

**App-exported ProRAW** (not booth RAW): add `--app-proraw` (enables embedded camera white balance). Do **not** use `--app-proraw` for booth RAW.

Output: `pipeline4_output/flash_noflash_skin_lab.csv`

---

## Chart-only skin color (zero prior)

**One photo + ColorChecker in frame.** No flash pair, no offline matrix, no demographics.

Full documentation: **[`docs/CHART_CC_ONLY.md`](docs/CHART_CC_ONLY.md)**

```bash
# Bundled JPEG cohort (6 trials, reproducible out of the box)
python3 run_chart_cc.py --no-include-flash

# Zero-prior fixed recipe: mesh ROI + affine 3×4 (same for every person)
python3 run_chart_cc.py --chart-only --no-include-flash

# iPhone ProRAW / DNG (build manifest locally — see data/pansor/README.md)
python3 scripts/build_pansor_manifest.py --data-root "/path/to/Pansor Images"
python3 run_chart_cc.py --input-mode dng \
  --manifest data/pansor/manifest_pansor_fitskin.csv \
  --cc-only --chart-only --no-include-flash \
  --out-dir chart_cc_output/pansor_dng_chart_only
```

| Mode | Data | Median ΔE₀₀ vs FitSkin* |
|------|------|-------------------------|
| `--chart-only` | Pansor ProRAW DNG | **≈ 2.2** |
| Default (cheek + 3×3) | Bundled JPEG | **≈ 4.9** |
| `--legacy` | Original Huber + cheek | See [`docs/CHART_CC_ORIGINAL.md`](docs/CHART_CC_ORIGINAL.md) |

\*FitSkin May-20 median cheek reference per participant. Cross-session captures (Pansor) report higher absolute ΔE than same-session booth.

**Cheek segmentation debug:**

```bash
python3 scripts/visualize_cheek_segmentation.py
```

Outputs under `chart_cc_output/`:

| Path | Content |
|------|---------|
| `comparison.csv` | Per-trial Lab + ΔE vs FitSkin |
| `skin_mask_overlays/noflash/` | Full face mesh (green) |
| `skin_mask_overlays/cheek_vs_mesh/` | **Green = cheek ROI, yellow = mesh-only** |
| `Lab_chart_cc_vs_fitskin_cheek.png` | Scatter plot |

Skip overlays: `--no-overlays`

---

## Chart CC pipeline (ColorChecker in scene)

Uses the **chart in the photo** for white balance + RGB→XYZ correction (not flash/no-flash reflectance).

```bash
python3 run_chart_cc.py
```

Bundled JPEGs: `data/chart_cc_jpeg/` (~87 MB). Expected median **ΔE₀₀ ≈ 4.9** vs FitSkin cheek (6 trials).

On the **same JPEG images**, chart CC (~5) beats chart-free flash/no-flash (~11.5 ΔE₀₀).
Pipeline 4 on **booth DNG** (no chart in frame) reaches ~3.5 — different capture, not comparable.

See also [`docs/CHART_CC_VS_PIPELINE4.md`](docs/CHART_CC_VS_PIPELINE4.md) for method comparison tables.

**Paper bundle (frozen tables):** [`figures/flash_noflash_phase4/`](figures/flash_noflash_phase4/)

| Mode | Median ΔE₀₀ vs FitSkin cheek | Independent validation? |
|------|------------------------------|-------------------------|
| Chart-free (primary) | **3.50** | Yes |
| + FitSkin lightness on same trials | **2.47** | No (reporting only) |

Cohort: $N=5$ trials (P1 T1–T3, P2 T2–T3; P2 T1 excluded).

## Pansor bag CAT02 (separate cohort)

Cross-session validation on June 2026 iPhone captures — **higher absolute ΔE** than Phase 4 above. See [`docs/PANSOR_BAG_CAT02.md`](docs/PANSOR_BAG_CAT02.md). Not run by `run_pipeline4.py`.

## Install

**macOS:** Apple’s built-in Python 3.9 often breaks `pip install`. Use **[`docs/INSTALL_MAC.md`](docs/INSTALL_MAC.md)** or run `bash scripts/setup_mac.sh` after installing Python 3.12 from [python.org](https://www.python.org/downloads/).

**Linux / Windows:**

```bash
python3 -m venv .venv && source .venv/bin/activate
python3 -m pip install --upgrade pip
pip install -r requirements.txt
```

**Python:** 3.10–3.12 recommended. Avoid macOS Command Line Tools Python 3.9.

**If `pip install` hangs on “Building wheel for opencv-contrib-python”:** upgrade pip, `git pull`, reinstall — `requirements.txt` pins OpenCV below 4.13 so pre-built wheels are used.

**First install downloads ~250 MB** (OpenCV, MediaPipe, JAX). Expect a few minutes on good Wi‑Fi.

## Other entry points (optional)

| Script | When to use |
|--------|-------------|
| `run_pipeline4.py` | **Start here** — reproduce ΔE vs FitSkin |
| `pipeline4_walkthrough.ipynb` | Step-by-step teaching notebook (not the production runner) |
| `flash_no_flash_skin_lab.py` | Low-level CLI if you need extra flags |

Production calibration: [`calibration/tier3_affine/`](calibration/tier3_affine/).

## Methods

See [`docs/FLASH_NOFLASH_SKIN_METHODS.md`](docs/FLASH_NOFLASH_SKIN_METHODS.md). Primary metric: $\mathbf{R}=\sqrt{\mathbf{A}\odot\mathbf{B}}$ on aligned linear RAW.

## Citation

Pin the git commit SHA used for reported ΔE₀₀ values (`figures/flash_noflash_phase4/MANIFEST-sha256.txt`).
