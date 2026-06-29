# Quickstart — chart-only ColorChecker on a new machine

**One photo + ColorChecker. No flash pair, no offline training, no demographics.**

Bundled test images (~87 MB) and FitSkin reference are **in the repo** — nothing else to download except Python packages on first run.

**Needs:** `git`, Python **3.10–3.12**, ~400 MB disk, internet for first `pip install`.

---

## Copy-paste (pick one)

### A — One line (curl, empty machine)

```bash
curl -fsSL https://raw.githubusercontent.com/RooneyEmily/Fitskin/main/install_chart_cc.sh | bash
```

### B — Clone + install script (recommended)

```bash
git clone https://github.com/RooneyEmily/Fitskin.git && \
  cd Fitskin && \
  bash install_chart_cc.sh
```

### C — Already cloned

```bash
cd Fitskin
bash install_chart_cc.sh
```

All three: clone (if needed) → create `.venv` → `pip install` → run `--chart-only` on 6 JPEG trials → write `chart_cc_output/`.

**Expected:** median ΔE₀₀ ≈ **4.9** on bundled JPEGs (evaluation vs FitSkin cheek in manifest).

---

## What you get

| Output | Description |
|--------|-------------|
| `chart_cc_output/comparison.csv` | Per-trial L\*a\*b\* and ΔE₀₀ |
| `chart_cc_output/summary.json` | Mean/median ΔE₀₀ |
| `chart_cc_output/skin_mask_overlays/cheek_vs_mesh/` | Green = cheek ROI, yellow = mesh |
| `chart_cc_output/Lab_chart_cc_vs_fitskin_cheek.png` | Scatter plot |

---

## macOS

Apple’s built-in Python 3.9 **will not work**. Install Python 3.12 first:

https://www.python.org/downloads/release/python-31210/ → macOS installer `.pkg` → reopen Terminal → run option **B** above.

Details: [`INSTALL_MAC.md`](INSTALL_MAC.md)

---

## Your own images

Build a manifest CSV (`subject_id, participant, trial, path_noflash, path_flash, fitskin_cheek_L, fitskin_cheek_a, fitskin_cheek_b` — FitSkin columns optional):

```bash
source .venv/bin/activate
python3 run_chart_cc.py \
  --manifest /path/to/manifest.csv \
  --chart-only --no-include-flash \
  --out-dir chart_cc_output/my_run
```

iPhone ProRAW DNG: [`data/pansor/README.md`](../data/pansor/README.md)

Full method: [`CHART_CC_ONLY.md`](CHART_CC_ONLY.md)
