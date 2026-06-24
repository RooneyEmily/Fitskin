#!/usr/bin/env bash
# Reliable macOS setup — avoids Apple Command Line Tools Python 3.9 pip crashes.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

pick_python() {
  local candidates=(
    python3.12 python3.11 python3.10
    /Library/Frameworks/Python.framework/Versions/3.12/bin/python3
    /Library/Frameworks/Python.framework/Versions/3.11/bin/python3
    /opt/homebrew/bin/python3.12
    /opt/homebrew/bin/python3.11
    /usr/local/bin/python3.12
    /usr/local/bin/python3.11
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

Install Python 3.12 (one-time):
  1. Open https://www.python.org/downloads/
  2. Download macOS installer for Python 3.12
  3. Run the installer (use default options)
  4. Re-run:  bash scripts/setup_mac.sh

Or with Homebrew:  brew install python@3.12
EOF
  exit 1
fi

VER="$("$PY" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')"
echo "Using $PY (Python $VER)"

if "$PY" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)'; then
  :
else
  echo "Python 3.10+ required. Apple CLT python3.9 breaks pip on macOS."
  exit 1
fi

rm -rf .venv
"$PY" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1
export PIP_NO_COMPILE=1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo ""
echo "Setup OK. Run the pipeline:"
echo "  cd \"$ROOT\""
echo "  source .venv/bin/activate"
echo "  python3 run_pipeline4.py"
