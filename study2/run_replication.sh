#!/usr/bin/env bash
set -euo pipefail

# Run from the study2 folder.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

mkdir -p results plots

# The shipped data/panel_event_monthly_final509.csv already carries the
# maturity/matching flag columns, so no panel-preparation step is needed.

echo "[1/2] Computing maturity descriptives..."
python scripts/maturity_repos_descriptives.py \
  --output-table results/maturity_repos_descriptives.csv

echo "[2/2] Rendering DiD notebook..."
Rscript -e "rmarkdown::render('notebooks/DiffinDiff.Rmd', output_format = 'html_document')"

echo "Replication run complete. Outputs are in results/ and plots/."
