# Fitskin — iPhone flash/no-flash skin color vs FitSkin scanner

Chart-free **flash / no-flash** geometric reflectance on iPhone DNG pairs, compared to **FitSkin cheek** scanner CIELAB (D65).

## Reproduce (clone → install → run)

Everything needed is in this repo: code, calibration, FitSkin reference Lab, and booth RAW DNGs.

```bash
git clone https://github.com/RooneyEmily/Fitskin.git
cd Fitskin
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
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

**Paper bundle (frozen tables):** [`figures/flash_noflash_phase4/`](figures/flash_noflash_phase4/)

| Mode | Median ΔE₀₀ vs FitSkin cheek | Independent validation? |
|------|------------------------------|-------------------------|
| Chart-free (primary) | **3.50** | Yes |
| + FitSkin lightness on same trials | **2.47** | No (reporting only) |

Cohort: $N=5$ trials (P1 T1–T3, P2 T2–T3; P2 T1 excluded).

## Pansor bag CAT02 (separate cohort)

Cross-session validation on June 2026 iPhone captures — **higher absolute ΔE** than Phase 4 above. See [`docs/PANSOR_BAG_CAT02.md`](docs/PANSOR_BAG_CAT02.md). Not run by `run_pipeline4.py`.

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

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
