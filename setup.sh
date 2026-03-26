#!/usr/bin/env bash
set -euo pipefail

# Resolve repo root (directory containing this script)
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

echo "Repo: $REPO_DIR"

# Check python3 availability
if ! command -v python3 >/dev/null 2>&1; then
  echo "Error: python3 is not installed or not on PATH."
  echo "macOS: install via 'brew install python' or from python.org."
  exit 1
fi

# Create venv if missing
if [ ! -d ".venv" ]; then
  echo "Creating virtual environment at .venv ..."
  python3 -m venv .venv
fi

# Activate venv
# shellcheck disable=SC1091
source .venv/bin/activate
echo "Using Python: $(python -V)"

# Upgrade pip tooling
python -m pip install --upgrade pip setuptools wheel

# Install dependencies
if [ -f "requirements.txt" ]; then
  echo "Installing Python dependencies from requirements.txt ..."
  python -m pip install -r requirements.txt
else
  echo "Warning: requirements.txt not found. Skipping dependency install."
fi

# Register Jupyter kernel for this venv
if python -c "import ipykernel" >/dev/null 2>&1; then
  python -m ipykernel install --user --name "grwm-venv" --display-name "Python (grwm-venv)" || true
else
  echo "ipykernel not installed; skipping kernel registration."
fi

# Data setup (run download only if core CSVs are missing)
NEED_DATA=0
for f in data/players.csv data/clubs.csv data/appearances.csv; do
  if [ ! -f "$f" ]; then
    NEED_DATA=1
    break
  fi
done

if [ "$NEED_DATA" -eq 1 ]; then
  echo "Fetching data via src/modules/data_download.py ..."
  python -c "from src.modules.data_download import download; download()" || {
    echo "Data download failed. You may need Kaggle credentials or to download data manually."
  }
else
  echo "Data files detected. Skipping download."
fi

echo
echo "Setup complete."
echo "To activate this environment in your current shell:"
echo "  source \"$REPO_DIR/.venv/bin/activate\""

