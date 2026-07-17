# Changes Since Initial Submission

This note summarizes how the Study 1 replication materials were refreshed to match the
revised analysis. It describes **code and methodology changes only**; the corresponding
results and their interpretation are reported in the paper.

No repository cohort, private inputs, or derived data are included in this package before or
after the refresh — the data directories remain empty by design.

## Maturity scoring engine (`analyzer/src/maturity_scorer.py`)

The scorer was revised to be more conservative and to attribute levels more precisely:

- **Strict artifact admission.** A file contributes level evidence only through the
  whitelisted tool definitions in `Artifacts/*.json` (plus exact-basename recovery of
  canonical tool files nested deeper in a repository, e.g. a nested `CLAUDE.md` / `AGENTS.md`).
  The earlier broad catch-all over generic markdown is no longer used.
- **Boilerplate and documentation pre-filter.** Project-boilerplate files (README, LICENSE,
  contributing/security/issue templates, …) and files living under `doc/`, `docs/`, or
  `documentation/` segments are dropped before classification. This logic is centralized in
  the new module `analyzer/src/artifact_filtering.py`, which is shared with the filtration
  notebook so both stages use one definition.
- **Strict L1 cap.** A repository is promoted above L1 only when at least one file is
  attributed to a tool through `Artifacts/*.json`. Semantic (content / path) evidence alone
  no longer promotes a repository.
- **New absorber category.** A `general-documentation` category was added so ordinary
  human-facing technical writing is classified as documentation rather than mistaken for an
  AI-tool artifact.
- **Tool-dialect bridge.** Category names used by individual tool definitions are mapped onto
  the canonical template categories so tool detection produces level evidence consistently
  across tools.
- **Tunable configuration.** A `ScoringConfig` dataclass exposes the semantic score/margin
  thresholds and the filters above as explicit switches, which the robustness notebook uses
  for sensitivity analysis.

## New and updated modules

- Added `analyzer/src/artifact_filtering.py` (imported by the scorer and the filtration
  notebook).
- Updated `analyzer/src/report_generator.py` to track the new scorer diagnostics.
- Updated the scorer and report-generator unit tests and added
  `analyzer/tests/test_artifact_filtering.py`. Tests are fixture-based and require no cohort
  data.

## Notebooks

- Refreshed the pipeline notebooks (`2`, `4`, `5`, `6`, `7`, and the predefined-category
  variant) to the revised code paths.
- Added the research notebooks under `analyzer/notebooks/research/`:
  `RQ1_maturity_structure`, `RQ2_robustness`, and `RQ3_dynamics`.
- All notebook outputs and runtime metadata are cleared; see `docs/SANITIZATION_LOG.md`.

## Scope note

Some upstream intermediate tables consumed by the research notebooks derive from the
restricted cohort and are not bundled. The research notebooks are therefore auditable
(every computation and statistical test is visible) but not end-to-end runnable from public
materials alone; see `REPRODUCE_STUDY1.md`.
