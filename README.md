# Anonymous Replication Package for ASE 2026 Submission

This directory contains an anonymous replication package prepared for double-blind review.

Study 1 materials are included in `study1/` as two coordinated components:
- `collector/` discovers AI-related repository artifacts, extracts text, computes embeddings, and records temporal metadata.
- `analyzer/` filters collected outputs, performs clustering and maturity analysis, and generates reports.

Study 2 materials are included in this submission build under `study2/`.

Directory overview:
- `study1/`: code and documentation for the public portion of Study 1
- `study2/`: replication materials for Study 2
- `data/public/`: location for shareable public inputs
- `data/anonymized/`: location for sanitized or derived outputs
- `data/placeholders/`: notes for omitted restricted artifacts
- `docs/`: package manifest and sanitization record

Some non-public intermediate artifacts, restricted repository inputs, tokens, and environment-specific outputs are intentionally not redistributed in this package.

See `study1/REPRODUCE_STUDY1.md` for the Study 1 workflow and `docs/` for package notes.
