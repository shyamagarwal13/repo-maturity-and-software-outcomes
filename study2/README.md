# Study 2 Replication Package

This folder contains the replication materials for Study 2 (maturity-stratified DiD analysis).

## Contents

- `data/`: input datasets used by scripts and notebook
- `data/panel_event_monthly.csv`: the analysis panel with final RAMP maturity labels (509 of 518 treated repositories accessible at re-collection, plus matched controls); this is the panel the paper's Study 2 results are computed from
- `data/maturity_levels.csv`: final RAMP level (1-4) per treated repository; input to `scripts/maturity_columns.py`
- `scripts/maturity_columns.py`: adds maturity/matching flags to the panel
- `scripts/maturity_repos_descriptives.py`: maturity-level descriptive statistics
- `notebooks/DiffinDiff.Rmd`: main DiD/event-study analysis notebook
- `results/`: generated CSV outputs (created if missing)
- `plots/`: generated figures (created if missing)

## Environment

### Python

- Python 3.9+ recommended
- Install dependencies:

```bash
pip install -r requirements.txt
```

### R

Required R packages are listed in `r-packages.txt`.

Install them in R:

```r
install.packages(scan("r-packages.txt", what = "character"))
```

## Reproduction Steps

Run from `study2/`:

1) Build the analysis panel, or use the shipped one. The panel can be rebuilt from Agarwal et al.'s original panel (`data/panel_event_monthly_original.csv`) plus the maturity labels (`data/maturity_levels.csv`) and matching (`data/matching.csv`); this reproduces the shipped `data/panel_event_monthly.csv` byte-for-byte, so you may equivalently skip this step and use the shipped file directly:

```bash
python scripts/maturity_columns.py
```

2) (Optional) Compute descriptive maturity statistics:

```bash
python scripts/maturity_repos_descriptives.py --output-table results/maturity_repos_descriptives.csv
```

3) Render the analysis notebook (reads `data/panel_event_monthly.csv`):

```bash
Rscript -e "rmarkdown::render('notebooks/DiffinDiff.Rmd', output_format = 'html_document')"
```

## Expected Outputs

The notebook writes the following key files in `results/`:

- `maturity_repos_descriptives.csv` (written by `maturity_repos_descriptives.py`)
- `static_effects_base_settings.csv`
- `dynamic_effects_base_settings.csv`
- `static_effects_full_subset.csv`
- `dynamic_effects_full_subset.csv`
- `static_effects_agent_subset.csv`
- `dynamic_effects_agent_subset.csv`
- `static_effects_ide_subset.csv`
- `dynamic_effects_ide_subset.csv`

It also writes multiple PDF figures to `plots/`.

## Maturity Labels and Primary Specification

Treated repositories carry final RAMP maturity labels: 236 at Level 1, 210 at Level 2, 23 at Level 3, and 40 at Level 4 (509 total; 9 of the original 518 treated repositories were inaccessible at re-collection). The paper's primary specification estimates effects within the agent-first stratum (`*_agent_subset` outputs); pooled (`*_full_subset`) and IDE-first (`*_ide_subset`) estimates are reported as robustness checks.

## Notes

- The notebook uses relative paths (`../data`, `../results`, `../plots`). The provided `run_replication.sh` renders it from `study2/`.
- `results/` and `plots/` are auto-created by the notebook if absent.
