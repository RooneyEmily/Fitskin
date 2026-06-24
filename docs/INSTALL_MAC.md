# macOS install (Emily / Justinas laptops)

Apple’s **Command Line Tools Python 3.9** often fails with:

- `Building wheel for opencv-contrib-python` (20–40 min compile)
- `TypeError: encode() argument 'encoding' must be str, not None` during `pip install`

**Fix: use Python 3.12 from python.org, not system `python3`.**

## One-time: install Python 3.12

python.org no longer ships macOS installers for 3.12.11+ (source only). Use **3.12.10** (last Mac installer):

1. Open https://www.python.org/downloads/release/python-31210/
2. Under **Files**, download **macOS 64-bit universal2 installer** (`.pkg`)
3. Run it (default options are fine)
4. Quit and reopen Terminal

**Alternative (Homebrew):** `brew install python@3.12` then use `python3.12` or `/opt/homebrew/bin/python3.12`.

## Clone and run (use a simple path — no spaces)

```bash
cd ~
rm -rf Fitskin
git clone https://github.com/RooneyEmily/Fitskin.git
cd Fitskin
bash scripts/setup_mac.sh
source .venv/bin/activate
python3 run_pipeline4.py
```

Expected output: **median ΔE₀₀ ≈ 3.50** on 5 trials.

## If `setup_mac.sh` says “No Python 3.10+ found”

After installing from python.org, try explicitly:

```bash
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m venv .venv
source .venv/bin/activate
export PIP_NO_COMPILE=1
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 run_pipeline4.py
```

## Do not use

- `python3` from **Xcode Command Line Tools** alone (usually 3.9)
- Folder paths with spaces for first-time setup (use `~/Fitskin` not `Fitskin Visualizations`)
