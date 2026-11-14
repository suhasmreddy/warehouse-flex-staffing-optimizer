#!/usr/bin/env bash
# ==============================================================================
# scripts/reproduce.sh
# Full end-to-end reproduction from a fresh repository checkout on UCI Dataset 502
# Usage:
#   python3 -m venv .venv && source .venv/bin/activate
#   pip install -r requirements.txt
#   bash scripts/reproduce.sh
# ==============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

echo "=== [Step 1/4] Running automated test suite ==="
if command -v pytest >/dev/null 2>&1; then
    pytest -q
else
    python3 -m unittest discover -s tests -p "test_*.py" -v
fi

echo "=== [Step 2/4] Downloading UCI Online Retail II (Dataset ID 502) ==="
python3 -m src.cli download --config configs/base.yml

echo "=== [Step 3/4] Running DuckDB SQL + Chronological Ridge + Pyomo/HiGHS MILP + Replay ==="
python3 -m src.cli run \
    --config configs/base.yml \
    --output-dir results/uci \
    --dataset-mode uci_online_retail_ii

echo "=== [Step 4/4] Generating empirical UCI summary artifacts ==="
python3 scripts/render_uci_report.py

echo "=== Reproduction complete! Inspect results/uci/ and results/uci/figures/ ==="
