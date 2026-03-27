# Study 1 Analyzer

This component performs downstream analysis for Study 1 using outputs produced by the sibling collector.

Main responsibilities:
- filter collected embeddings and metadata
- cluster artifacts and assign semantic categories
- compute maturity scores
- summarize temporal health patterns
- generate repository-level or collection-level reports

Core code lives in `src/`. Analysis notebooks are provided in `notebooks/`.

The analyzer expects the packaged sibling path `../collector` unless `COLLECTOR_REPO_PATH` is overridden in `.env`. See `../REPRODUCE_STUDY1.md` for execution notes and known limitations.
