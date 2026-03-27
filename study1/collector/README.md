# Study 1 Collector

This component performs artifact discovery and data collection for Study 1.

Main responsibilities:
- load artifact definitions from `Artifacts/*.json`
- clone or update target repositories
- discover matching AI-related files
- extract text content
- compute embeddings
- derive temporal metadata and anonymized author statistics
- write self-contained output bundles

Primary entry point:
- `scripts/artifacts_collection.py`

Configuration template:
- `config.example.yaml`

The collector can operate on a single repository, a repository group, or a file containing repository URLs. See `../REPRODUCE_STUDY1.md` for the minimal execution sequence used in this package.
