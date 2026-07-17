# Study 1 Overview

Study 1 is organized as a two-stage pipeline.

`collector/` scans one or more git repositories for AI-related artifact files, extracts text, computes embeddings, and writes structured output bundles with temporal metadata.

`analyzer/` consumes those bundles to perform filtering, clustering, maturity scoring, and report generation. The analyzer depends on a few shared modules that remain housed in the sibling `collector/` directory.

Maturity scoring (`analyzer/src/maturity_scorer.py`) assigns each repository a four-level
adoption maturity label from three signals — tool detection against the `Artifacts/*.json`
definitions, path-semantic intent, and content-semantic classification. Two supporting
behaviors keep the label conservative: a boilerplate / documentation pre-filter
(`analyzer/src/artifact_filtering.py`) removes project files that are not AI-tool artifacts,
and the level is only promoted above L1 when at least one file is attributed to a tool via the
`Artifacts/*.json` definitions. The research notebooks under `analyzer/notebooks/research/`
(`RQ1`–`RQ3`) build the structure, robustness, and dynamics analyses on top of these scores.

The intended sequence is:
1. run the collector on the target repository set
2. review the generated bundle in `collector/output/`
3. run the analyzer modules or notebooks on the collected outputs
4. run the `research/` notebooks (`RQ1`–`RQ3`) on the analyzer outputs

See `REPRODUCE_STUDY1.md` for setup and execution notes, and
`CHANGES_SINCE_SUBMISSION.md` for what changed relative to the initial submission.
