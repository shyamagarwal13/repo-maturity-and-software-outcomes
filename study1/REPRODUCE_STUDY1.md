# Reproducing Study 1

## Prerequisites

- Python 3.10 or newer
- Git available on the command line
- Enough disk space for repository clones, embeddings, and derived outputs
- Optional API credentials if you want to collect private repositories or run LLM-assisted notebook sections

## Minimal Setup

1. Create and activate a virtual environment.
2. Install collector dependencies:
   `pip install -r study1/collector/requirements.txt`
3. Install analyzer dependencies:
   `pip install -r study1/analyzer/requirements.txt`
4. If needed, copy `study1/collector/config.example.yaml` to `study1/collector/config.yaml` and set local values such as author anonymization secret and output paths.
5. If needed, copy `study1/analyzer/.env.example` to `study1/analyzer/.env`.

## High-Level Execution Sequence

1. Run the collector from `study1/collector/`.
   Example:
   `python scripts/artifacts_collection.py <target-repository-or-group-url>`
2. Confirm that a collection bundle was written under `study1/collector/output/`.
3. Run the analyzer from `study1/analyzer/`.
   Typical workflow:
   - launch Jupyter and execute the notebooks in order
   - or invoke the analysis modules directly from Python scripts
4. Review generated intermediate tables, cluster assignments, maturity outputs, and optional report artifacts.

## Notes on Missing Inputs

This package does not include the original repository cohort, private repository access tokens, or all intermediate outputs used during the full study workflow.

As a result, a reviewer can inspect the code, run the pipeline on their own accessible repositories, and reproduce the computation pattern, but may not be able to recreate every study-specific result from the submission using public materials alone.

Some notebook sections optionally call external LLM APIs for interpretation or narrative generation. Those sections are not required to inspect the core pipeline and can be skipped if credentials are unavailable.

## Expected Outputs

Collector outputs typically include:
- file-level artifact metadata
- embedding vectors and embedding metadata
- temporal artifact history tables
- aggregated repository metrics
- a manifest and snapshot of artifact definitions

Analyzer outputs typically include:
- filtered metadata and embeddings
- clustering assignments and category labels
- maturity scoring tables
- optional repository-level or collection-level reports
