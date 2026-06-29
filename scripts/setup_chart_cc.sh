#!/usr/bin/env bash
# Install Python deps and run chart-only ColorChecker on bundled JPEG cohort.
# Called by install_chart_cc.sh — do not require a prior manual pip install.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MANIFEST="$ROOT/data/manifest_chart_cc_fitskin.csv"
JPEG_DIR="$ROOT/data/chart_cc_jpeg"

if [[ ! -f "$MANIFEST" ]]; then
  echo "Missing $MANIFEST — incomplete clone?" >&2
  exit 1
fi
if [[ ! -d "$JPEG_DIR/Participant_1" ]]; then
  echo "Missing bundled JPEGs under $JPEG_DIR — incomplete clone?" >&2
  exit 1
fi

pick_python() {
  local candidates=(
    python3.12 python3.11 python3.10 python3
    /Library/Frameworks/Python.framework/Versions/3.12/bin/python3
    /Library/Frameworks/Python.framework/Versions/3.11/bin/python3
    /opt/homebrew/bin/python3.12
    /opt/homebrew/bin/python3.11
    /usr/local/bin/python3.12
    /usr/bin/python3
  )
  local py ver major minor
  for py in "${candidates[@]}"; do
    if command -v "$py" >/dev/null 2>&1; then
      ver="$("$py" -c 'import sys; print(sys.version_info[0], sys.version_info[1])')"
      read -r major minor <<< "$ver"
      if [[ "$major" -eq 3 && "$minor" -ge 10 ]]; then
        echo "$py"
        return 0
      fi
    fi
  done
  return 1
}

if ! PY="$(pick_python)"; then
  cat <<'EOF'
No Python 3.10+ found.

macOS:  https://www.python.org/downloads/release/python-31210/  (install .pkg, reopen Terminal)
Linux:  sudo apt install python3.12 python3.12-venv    # Debian/Ubuntu
        sudo dnf install python3.11                      # Fedora
EOF
  exit 1
fi

VER="$("$PY" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')"
echo "==> Python $VER ($PY)"

if [[ ! -d .venv ]]; then
  echo "==> Creating virtualenv .venv"
  if ! "$PY" -m venv .venv 2>/dev/null; then
    cat <<'EOF'
venv failed. On Debian/Ubuntu install:
  sudo apt install python3.12-venv
EOF
    exit 1
  fi
fi
# shellcheck disable=SC1091
source .venv/bin/activate

export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1
export PIP_NO_COMPILE=1

echo "==> Installing dependencies (first run downloads ~250 MB, a few minutes) ..."
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo ""
echo "==> Running chart-only ColorChecker (6 bundled JPEG trials, zero prior) ..."
echo ""

python run_chart_cc.py --chart-only --no-include-flash --no-histograms

echo ""
echo "============================================================"
echo "SUCCESS"
echo "============================================================"
echo "  Results CSV:  $ROOT/chart_cc_output/comparison.csv"
echo "  Summary:      $ROOT/chart_cc_output/summary.json"
echo "  Cheek masks:  $ROOT/chart_cc_output/skin_mask_overlays/cheek_vs_mesh/"
echo ""
echo "Run again anytime:"
echo "  cd $ROOT"
echo "  source .venv/bin/activate"
echo "  python3 run_chart_cc.py --chart-only --no-include-flash"
echo ""
echo "Docs: docs/CHART_CC_ONLY.md  |  docs/QUICKSTART_CHART_CC.md"
