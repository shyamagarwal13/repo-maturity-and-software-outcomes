# Package Manifest

## Top Level

- `README.md`: package overview for anonymous review
- `LICENSE_NOTE.md`: licensing and redistribution note for this local review build
- `study1/`: merged materials for Study 1
- `study2/`: replication materials for Study 2
- `human-validation/`: human-label validation of the classifier
- `docs/`: package manifest and sanitization record

## Study 1

- `study1/README.md`: summary of the Study 1 workflow
- `study1/REPRODUCE_STUDY1.md`: setup and reproduction notes
- `study1/CHANGES_SINCE_SUBMISSION.md`: summary of the analyzer refresh relative to the initial submission
- `study1/collector/`: collection code, tests, configs, and artifact definitions
- `study1/analyzer/`: analysis code, notebooks, tests, configs, and artifact definitions
  - `analyzer/src/maturity_scorer.py`: multi-signal AI-adoption maturity scoring engine
  - `analyzer/src/artifact_filtering.py`: boilerplate / documentation pre-filter shared by the filtration notebook and the scorer
  - `analyzer/notebooks/research/`: research notebooks `RQ1`–`RQ3` (structure, robustness, dynamics)

## Study 2

- `study2/data/`: analysis panel (buildable from the original panel via `scripts/maturity_columns.py`, or used as shipped), maturity labels, matching
- `study2/scripts/`, `study2/notebooks/`, `study2/run_replication.sh`: end-to-end reproduction of the paper's Study 2 results
- `study2/results/`, `study2/plots/`: shipped outputs (regenerated exactly by the notebook)

## Human validation

- `human-validation/`: self-contained human-annotation validation of the classifier (labeled data, evaluation notebook, per-category metrics, confusion matrices)

## Data notes

- Study 1's corporate repository inputs and derived per-repository data are restricted by research agreements and are not redistributed; the Study 1 directories contain code, configuration, and notebooks only

## Docs

- `docs/SANITIZATION_LOG.md`: summary of removed and sanitized content
- `docs/PACKAGE_MANIFEST.md`: this manifest
