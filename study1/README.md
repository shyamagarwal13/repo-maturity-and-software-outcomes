# Study 1 Overview

Study 1 is organized as a two-stage pipeline.

`collector/` scans one or more git repositories for AI-related artifact files, extracts text, computes embeddings, and writes structured output bundles with temporal metadata.

`analyzer/` consumes those bundles to perform filtering, clustering, maturity scoring, and report generation. The analyzer depends on a few shared modules that remain housed in the sibling `collector/` directory.

The intended sequence is:
1. run the collector on the target repository set
2. review the generated bundle in `collector/output/`
3. run the analyzer modules or notebooks on the collected outputs

See `REPRODUCE_STUDY1.md` for setup and execution notes.
