# Fitskin — iPhone flash/no-flash skin color vs FitSkin scanner

Chart-free **flash / no-flash** geometric reflectance on iPhone DNG pairs, compared to **FitSkin cheek** scanner CIELAB (D65).

## Reproduce (one command)

```bash
git clone https://github.com/RooneyEmily/Fitskin.git
cd Fitskin
pip install -r requirements.txt

# Full booth dataset (Participant */Trial */*NoFlash*.DNG + *Flash*.DNG)
python3 run_pipeline4.py /path/to/RAW/Dataset

# Or a single trial (two DNG files) — e.g. best Phase 4 trial P1_T2 (target ΔE₀₀ ≈ 2.75)
python3 run_pipeline4.py --trial P1_T2 /path/to/NoFlash.DNG /path/to/Flash.DNG
```

**`run_pipeline4.py`** is the only script you need. It runs the production stack (affine calibration, cheek ROI, skin-mask exposure, exposure anchor) and prints pipeline vs FitSkin Lab plus ΔE₀₀ per trial.

| Target | Median ΔE₀₀ |
|--------|-------------|
| 5-trial Phase 4 cohort (excludes P2_T1) | **≈ 3.50** |
| Single trial P1_T2 (typical) | **≈ 2.75** |

FitSkin reference values are bundled in `data/phase4_fitskin_reference.csv` (same-session scanner cheek Lab from the paper).

**App-exported ProRAW** (not booth RAW): add `--app-proraw` (enables embedded camera white balance).

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
