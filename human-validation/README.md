# Human-Label Validation of the AIME Machine Labelers (Notebook 15)

Self-contained package for reviewing and reproducing the validation of the two
machine file-labelers — the embedding (AIME) labeler and the Haiku LLM labeler —
against human judgment, at the file level and at the repository-maturity level.

## Start here

1. **`notebooks/15_human_label_evaluation_conclusions.md`** — the hand-written
   conclusions: study design, statistical methods, findings F1–F5, limitations.
2. **`notebooks/15. human_label_evaluation.html`** — the frozen executed run
   (immutable results; open in any browser, no setup needed).
3. **`data/validation/human_labeling/paper/`** — every statistic as
   `summary.json` plus five LaTeX (booktabs) tables ready for the paper.

## Headline results

- Inter-annotator agreement: Krippendorff's α = 0.572 (moderate — the task is
  genuinely hard); pairwise Cohen's κ 0.50–0.81.
- File level (n = 175 reference items): embedding 81.7% vs Haiku 81.1%
  accuracy, κ ≈ 0.74 both; statistically indistinguishable (McNemar p = 1.0).
- Repository level (35 repos): embedding reproduces the human-derived maturity
  level in 34/35 (97%, quadratic weighted κ = 0.83); Haiku in 27/35 (77%,
  κ_w = 0.74). Rates are stratum-weighted, not population rates.

## Reproducing the run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cd notebooks
jupyter nbconvert --to notebook --execute --inplace "15. human_label_evaluation.ipynb"
```

Runs in ~1 minute. The notebook is deterministic (seeded); it re-derives every
CSV, figure, and `paper/` export in place from the inputs below. The shipped
`.ipynb` already contains the executed outputs.

## Contents

```
notebooks/
  15. human_label_evaluation.ipynb        the analysis (computation only)
  15. human_label_evaluation.html         frozen executed run
  15_human_label_evaluation_conclusions.md  hand-written conclusions
src/
  maturity_scorer.py                      AIME scorer (verbatim from the repo;
                                          notebook uses aggregate_repo_maturity)
  artifact_filtering.py                   verbatim dependency
  artifact_config_loader.py, embedding_generator.py
                                          import-only stubs for collector-repo
                                          modules (never called by notebook 15)
data/
  msrc_file_predictions.parquet           embedding-labeler file predictions
  msrc_file_predictions_llm.parquet       Haiku overlay file predictions
  validation/human_labeling/
    manifest.csv                          195 sampled items + machine labels +
                                          rater assignment (notebook 14, seed 14)
    sampled_repos.csv, repo_pool.csv      the 35-repo draw and the 509-repo pool
    instructions.md                       annotator instructions (11-label protocol)
    returned/                             normalized canonical annotator workbooks
                                          (raw originals withheld for anonymity)
    human_labels_reference.csv            per-item reference labels + provenance
    human_file_eval.csv, human_repo_eval.csv  evaluation outputs
    paper/                                summary.json + LaTeX tables
  validation/figures/                     confusion matrices, repo-level chart
```

## Provenance & scope notes

- Sample: November 2025 snapshot only; strict-whitelisted artifacts; 35 repos
  drawn level-balanced from the 268-repo eligible pool — repo-level rates are
  therefore stratum-weighted (see the conclusions file, Limitations).
- Raters were blinded to machine labels and worked independently; normalized
  copies of their returned workbooks are in `returned/`.
- The full pipeline (collection, embeddings, scoring) lives in the
  ai-artifacts-analyzer and ai-artifacts-collector repos; this package contains
  only what notebook 15 needs.

