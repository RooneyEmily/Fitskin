#!/usr/bin/env bash
# Chart-only ColorChecker: clone (if needed) → install deps → run bundled smoke test.
#
# Paste on a new Linux/macOS machine (needs git + Python 3.10+):
#
#   curl -fsSL https://raw.githubusercontent.com/RooneyEmily/Fitskin/main/install_chart_cc.sh | bash
#
# Or after git clone:
#
#   git clone https://github.com/RooneyEmily/Fitskin.git && cd Fitskin && bash install_chart_cc.sh
#
set -euo pipefail

REPO_DIR="${FITSKIN_REPO:-Fitskin}"
CLONE_URL="${FITSKIN_CLONE_URL:-https://github.com/RooneyEmily/Fitskin.git}"

if [[ -f run_chart_cc.py ]]; then
  ROOT="$(pwd)"
elif [[ -f "$REPO_DIR/run_chart_cc.py" ]]; then
  ROOT="$(cd "$REPO_DIR" && pwd)"
else
  if ! command -v git >/dev/null 2>&1; then
    echo "git is required. Install git, then re-run." >&2
    exit 1
  fi
  echo "Cloning Fitskin into ./${REPO_DIR} ..."
  git clone --depth 1 "$CLONE_URL" "$REPO_DIR"
  ROOT="$(cd "$REPO_DIR" && pwd)"
fi

cd "$ROOT"
exec bash scripts/setup_chart_cc.sh
