#!/usr/bin/env bash
set -euo pipefail

# Run from the study2 folder.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

mkdir -p results plots

echo "[1/3] Building maturity columns in panel data..."
python scripts/maturity_columns.py

echo "[2/3] Computing maturity descriptives..."
python scripts/maturity_repos_descriptives.py \
  --output-table results/maturity_repos_descriptives.csv

echo "[3/3] Rendering DiD notebook..."
Rscript -e "rmarkdown::render('notebooks/DiffinDiff.Rmd', output_format = 'html_document')"

echo "Replication run complete. Outputs are in results/ and plots/."
