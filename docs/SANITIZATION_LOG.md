# Sanitization Log

This package was assembled by copying the two public Study 1 codebases into a single anonymous directory structure and then sanitizing the result for double-blind review.

Removed:
- git metadata directories
- reviewer-irrelevant local agent configuration files and helper docs
- upstream README files with attribution-heavy project descriptions
- development notes and exploratory materials not needed for the packaged Study 1 workflow
- notebook outputs and widget state
- artifact reference markdown files that were not required by the executable pipeline

Renamed or reorganized:
- the two source repositories were placed under `study1/collector/` and `study1/analyzer/`
- reviewer-facing documentation was rewritten at the package, study, and component levels

Sanitized:
- institution and author references in retained documentation and templates
- hardcoded local filesystem paths in notebooks
- default analyzer path configuration so it points to the packaged sibling collector
- example secret material in configuration templates

Intentionally left unchanged for functionality:
- core source code implementing collection, embedding, temporal analysis, clustering, scoring, and reporting
- artifact definition JSON files in `Artifacts/`
- generic provider URLs, API examples, and test fixtures that are operational or explanatory rather than identifying
- optional external API hooks used by some notebook-driven report generation steps

## Refresh update

The Study 1 analyzer was refreshed to match the revised analysis (see
`study1/CHANGES_SINCE_SUBMISSION.md`). The same sanitization discipline was applied to all
updated and newly added files:

- notebook outputs, execution counts, and widget/runtime metadata were cleared from every
  updated pipeline notebook and every newly added research notebook (`RQ1`–`RQ3`)
- hardcoded local filesystem paths, working-directory names, and run-name stamps introduced
  by the refreshed notebooks were replaced with package-relative placeholders and the
  standard `example-run` organization placeholder
- no repository cohort, private inputs, embeddings, scores, or other derived data were added;
  data directories remain empty by design
