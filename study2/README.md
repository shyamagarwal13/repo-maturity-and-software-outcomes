# Study 2 Replication Package

This folder contains the replication materials for Study 2 (maturity-stratified DiD analysis).

## Contents

- `data/`: input datasets used by scripts and notebook
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

1) Prepare maturity columns in panel data:

```bash
python scripts/maturity_columns.py
```

2) (Optional) Compute descriptive maturity statistics:

```bash
python scripts/maturity_repos_descriptives.py --output-table results/maturity_repos_descriptives.csv
```

3) Render the analysis notebook:

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

## Notes

- The notebook uses relative paths (`../data`, `../results`, `../plots`). The provided `run_replication.sh` renders it from `study2/`.
- `results/` and `plots/` are auto-created by the notebook if absent.
