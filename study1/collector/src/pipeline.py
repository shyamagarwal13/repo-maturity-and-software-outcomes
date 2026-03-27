"""Pipeline module for artifact collection and embedding generation.

Extracts the shared pipeline logic from the notebook so both
``notebooks/1. embedding_artifacts_collection.ipynb`` and
``scripts/artifacts_collection.py`` import from a single source of truth.

All functions are side-effect-free (no print statements). Callers handle
logging and progress display.
"""

import gc
import json
import logging
import pickle
import shutil
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from sentence_transformers import SentenceTransformer

from src.artifact_config_loader import load_json_configs, load_shared_config
from src.embedding_generator import (
    add_embeddings_to_artifacts,
    load_embedding_model,
)
from src.file_discovery import discover_artifacts
from src.git_operations import (
    checkout_branch,
    clone_repository,
    find_commit_at_date,
    pull_latest,
    reset_to_commit,
)
from src.temporal_analyzer import (
    analyze_artifact_history,
    anonymize_author,
    collect_repo_static_metrics,
    obfuscate_author,
)
from src.text_extractor import get_file_size, read_text_file


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """Configuration for a single pipeline run."""

    repo_url: str
    branch: str
    repo_name: str

    # Paths
    clone_base_dir: Path
    artifacts_dir: Path
    output_dir: Path
    repo_path: Optional[Path] = None  # Set after clone/prepare

    # Embedding
    embedding_model: str = "nomic-ai/nomic-embed-text-v1.5"
    embedding_cache_dir: Optional[str] = None
    embedding_batch_size: int = 32
    embedding_memory_budget_gib: int = 2

    # Git timeouts (seconds)
    git_timeout: int = 300       # clone / pull
    git_log_timeout: int = 60    # rev-list / shortlog / ls-files

    # Date window — controls both the repository checkout state and temporal
    # analysis range.  When end_date is in the past the repository is reset to
    # the last commit on or before that date so that artifact discovery and
    # git history analysis are consistent.
    start_date: str = "2020-01-01"
    end_date: str = field(default_factory=lambda: date.today().isoformat())
    author_strategy: str = "obfuscate"  # "obfuscate" or "anonymize"
    author_salt: str = ""               # used by obfuscate
    author_secret: str = ""             # used by anonymize
    author_org: str = ""                # used by anonymize
    author_hash_length: int = 12        # used by anonymize
    author_prefix: str = "user-"        # used by anonymize


@dataclass
class PipelineResult:
    """Results produced by a single pipeline run."""

    artifacts: List[Dict[str, Any]] = field(default_factory=list)
    metadata_df: Optional[pd.DataFrame] = None
    embeddings_data: Optional[Dict[str, Any]] = None  # file_ids, embeddings, model, dim
    timeseries_df: Optional[pd.DataFrame] = None
    commits_df: Optional[pd.DataFrame] = None
    repo_metrics: Optional[Dict[str, Any]] = None
    n_with_embedding: int = 0
    n_without_embedding: int = 0


# ---------------------------------------------------------------------------
# YAML configuration helpers
# ---------------------------------------------------------------------------

_logger = logging.getLogger(__name__)

_KNOWN_YAML_KEYS = {
    "embedding": {"batch_size", "memory_budget_gib"},
    "temporal": {"start_date", "end_date"},
    "author": {"strategy", "secret", "hash_length", "prefix"},
    "git": {"timeout", "log_timeout"},
    "paths": {"output_dir", "clone_dir"},
}


def load_config(path: Optional[Path] = None) -> dict:
    """Load pipeline settings from a YAML file.

    Args:
        path: Path to the YAML config file. When *None* or the file does not
            exist, an empty dict is returned so that all defaults apply.

    Returns:
        Parsed YAML as a nested dict.
    """
    if path is None or not Path(path).is_file():
        return {}

    with open(path) as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        return {}

    # Normalise sections with all keys commented out (YAML parses as None)
    for section in list(cfg):
        if cfg[section] is None:
            cfg[section] = {}

    # Warn on unknown top-level or nested keys
    for section, value in cfg.items():
        if section not in _KNOWN_YAML_KEYS:
            _logger.warning("Unknown config section: %s", section)
        elif isinstance(value, dict):
            for key in value:
                if key not in _KNOWN_YAML_KEYS[section]:
                    _logger.warning(
                        "Unknown config key: %s.%s", section, key
                    )

    return cfg


def config_to_pipeline_config(yaml_cfg: dict, **overrides) -> PipelineConfig:
    """Build a :class:`PipelineConfig` from YAML values and per-repo overrides.

    *yaml_cfg* supplies study-wide defaults (batch size, date range, author
    settings). *overrides* supply per-repo values (``repo_url``, ``branch``,
    ``repo_name``, paths, secrets) and always win over YAML values.
    """
    embedding = yaml_cfg.get("embedding", {})
    temporal = yaml_cfg.get("temporal", {})
    author = yaml_cfg.get("author", {})

    kwargs: Dict[str, Any] = {}

    # Embedding
    if "batch_size" in embedding:
        kwargs["embedding_batch_size"] = embedding["batch_size"]
    if "memory_budget_gib" in embedding:
        kwargs["embedding_memory_budget_gib"] = embedding["memory_budget_gib"]

    # Git
    git = yaml_cfg.get("git", {})
    if "timeout" in git:
        kwargs["git_timeout"] = git["timeout"]
    if "log_timeout" in git:
        kwargs["git_log_timeout"] = git["log_timeout"]

    # Temporal
    if "start_date" in temporal:
        kwargs["start_date"] = temporal["start_date"]
    if "end_date" in temporal and temporal["end_date"] is not None:
        kwargs["end_date"] = temporal["end_date"]

    # Author
    if "strategy" in author:
        kwargs["author_strategy"] = author["strategy"]
    if "secret" in author:
        kwargs["author_secret"] = author["secret"]
    if "hash_length" in author:
        kwargs["author_hash_length"] = author["hash_length"]
    if "prefix" in author:
        kwargs["author_prefix"] = author["prefix"]

    # Paths
    paths = yaml_cfg.get("paths", {})
    if "output_dir" in paths:
        kwargs["output_dir"] = Path(paths["output_dir"])
    if "clone_dir" in paths:
        kwargs["clone_base_dir"] = Path(paths["clone_dir"])

    # Overrides always win
    kwargs.update(overrides)

    return PipelineConfig(**kwargs)


# ---------------------------------------------------------------------------
# Pipeline functions
# ---------------------------------------------------------------------------

def load_tool_configs(
    artifacts_dir: Path,
) -> Tuple[Dict[str, Any], Any]:
    """Load tool configurations and shared config from the Artifacts directory.

    Returns:
        (tool_configs, shared_config) tuple.
    """
    tool_configs = load_json_configs(str(artifacts_dir))
    shared_config = load_shared_config(str(artifacts_dir))
    return tool_configs, shared_config


def clone_and_prepare_repo(
    config: PipelineConfig,
    token: Optional[str] = None,
) -> Path:
    """Clone (if needed), checkout, and pull latest for the repository.

    Updates ``config.repo_path`` in place and returns the validated path.

    Raises:
        FileNotFoundError: If the repo path does not exist after preparation.
    """
    repo_path = config.repo_path or (config.clone_base_dir / config.repo_name)

    if not repo_path.exists():
        clone_repository(
            config.repo_url,
            str(config.clone_base_dir),
            branch=config.branch,
            token=token,
            timeout=config.git_timeout,
        )

    # Detect actual branch (clone may have defaulted to a different name)
    try:
        import subprocess as _sp
        actual_branch = _sp.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(repo_path), capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        actual_branch = config.branch

    try:
        checkout_branch(str(repo_path), config.branch)
    except Exception:
        # Requested branch doesn't exist — use the repo's default branch
        config.branch = actual_branch

    pull_latest(str(repo_path), config.branch, timeout=config.git_timeout)

    if not repo_path.exists():
        raise FileNotFoundError(f"Repository path does not exist: {repo_path}")

    # If end_date is in the past, rewind the branch to match that date
    if config.end_date < date.today().isoformat():
        sha = find_commit_at_date(str(repo_path), config.branch, config.end_date)
        if sha is None:
            raise ValueError(
                f"No commits found on '{config.branch}' before {config.end_date}"
            )
        reset_to_commit(str(repo_path), sha)

    config.repo_path = repo_path
    return repo_path


def collect_repo_metrics_data(
    repo_path: Path,
    config: PipelineConfig,
) -> Dict[str, Any]:
    """Collect static repository metrics and enrich with config metadata.

    Does **not** write to disk — use :func:`export_results` for that.
    """
    metrics = collect_repo_static_metrics(str(repo_path), timeout=config.git_log_timeout)
    metrics["repo_name"] = config.repo_name
    metrics["repo_url"] = config.repo_url
    metrics["branch"] = config.branch
    return metrics


def discover_and_extract(
    repo_path: Path,
    tool_configs: Dict[str, Any],
    shared_config: Any,
) -> List[Dict[str, Any]]:
    """Discover artifacts (Phase 3) and extract text content (Phase 4).

    Returns the artifact list with ``text_content`` and ``file_size`` populated.
    """
    artifacts = discover_artifacts(repo_path, tool_configs, shared_config)

    for artifact in artifacts:
        if "absolute_path" in artifact:
            file_path = artifact["absolute_path"]
        else:
            file_path = str(repo_path / artifact["file_path"])

        try:
            artifact["file_size"] = get_file_size(file_path)
            result = read_text_file(file_path)
            if result["success"]:
                artifact["text_content"] = result["content"]
            else:
                artifact["text_content"] = None
        except Exception:
            artifact["text_content"] = None
            artifact["file_size"] = 0

    return artifacts


def load_model(config: PipelineConfig) -> SentenceTransformer:
    """Load the embedding model specified in *config*."""
    return load_embedding_model(
        config.embedding_model,
        cache_dir=config.embedding_cache_dir,
    )


def generate_embeddings(
    artifacts: List[Dict[str, Any]],
    config: PipelineConfig,
    model: Optional[SentenceTransformer] = None,
) -> Tuple[List[Dict[str, Any]], int]:
    """Generate embeddings for artifacts.

    Args:
        artifacts: Artifact list (mutated in place).
        config: Pipeline configuration.
        model: Pre-loaded model for reuse (batch mode). If ``None``, the model
            is loaded internally and unloaded after embedding.

    Returns:
        (artifacts, embedding_dim) tuple.
    """
    owns_model = model is None
    if owns_model:
        model = load_model(config)

    embedding_dim = model.get_sentence_embedding_dimension()

    artifacts = add_embeddings_to_artifacts(
        artifacts,
        model,
        config.embedding_model,
        batch_size=config.embedding_batch_size,
        memory_budget_gib=config.embedding_memory_budget_gib,
    )

    if owns_model:
        del model
        gc.collect()

    return artifacts, embedding_dim


def build_metadata(
    artifacts: List[Dict[str, Any]],
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Assign file IDs and build metadata DataFrame.

    Returns:
        (metadata_df, artifacts) — artifacts are updated with ``file_id``.
    """
    rows: List[Dict[str, Any]] = []
    for i, artifact in enumerate(artifacts):
        file_id = f"file_{i:03d}"
        artifact["file_id"] = file_id
        rows.append(
            {
                "file_id": file_id,
                "repo_name": artifact.get(
                    "artifact_name", artifact["file_path"].split("/")[-1]
                ),
                "tool_name": artifact["tool_name"],
                "artifact_path": artifact["file_path"],
                "artifact_name": artifact.get("artifact_name", artifact["file_path"]),
                "is_standard": artifact.get("is_standard", False),
                "discovery_step": artifact.get("discovery_step", "unknown"),
                "discovery_method": artifact.get("discovery_method", "unknown"),
                "file_size": artifact.get("file_size", 0),
                "has_embedding": artifact.get("embedding") is not None,
            }
        )

    metadata_df = pd.DataFrame(rows)
    return metadata_df, artifacts


def run_temporal_analysis(
    repo_path: Path,
    artifacts: List[Dict[str, Any]],
    config: PipelineConfig,
) -> Dict[str, Any]:
    """Run temporal (git history) analysis on the discovered artifacts.

    Returns:
        Dict with ``artifact_timeseries`` and ``commit_aggregated`` keys.
    """
    # Build the author hashing callable based on strategy
    if config.author_strategy == "anonymize":
        hash_fn = lambda identifier: anonymize_author(
            identifier,
            config.author_org,
            config.author_secret,
            config.author_hash_length,
            config.author_prefix,
        )
    else:
        hash_fn = lambda identifier: obfuscate_author(identifier, config.author_salt)

    temporal_artifacts = [
        {
            "path": a["file_path"],
            "type": a.get("artifact_type", "unknown"),
        }
        for a in artifacts
    ]
    return analyze_artifact_history(
        repo_path=str(repo_path),
        artifacts=temporal_artifacts,
        start_date=config.start_date,
        end_date=config.end_date,
        hash_fn=hash_fn,
    )


def export_results(
    artifacts: List[Dict[str, Any]],
    metadata_df: pd.DataFrame,
    temporal_result: Dict[str, Any],
    repo_metrics: Optional[Dict[str, Any]],
    config: PipelineConfig,
) -> Dict[str, Path]:
    """Export all pipeline outputs to disk.

    Returns:
        Dict mapping output type name to its file path.
    """
    # Bundle Artifacts/ config and manifest at the output root level
    bundle_artifacts_config(config.artifacts_dir, config.output_dir)
    write_manifest(config.output_dir, config)

    output_dir = config.output_dir / config.repo_name
    output_dir.mkdir(parents=True, exist_ok=True)
    repo_name = config.repo_name
    exported: Dict[str, Path] = {}

    # --- File metadata (only rows with embeddings for consistency) ----------
    metadata_with_emb = metadata_df[metadata_df["has_embedding"] == True].copy()  # noqa: E712
    meta_path = output_dir / f"{repo_name}_file_artifacts.csv"
    metadata_with_emb.to_csv(meta_path, index=False)
    exported["file_artifacts"] = meta_path

    # --- Embeddings pickle --------------------------------------------------
    emb_pairs = [
        (a["file_id"], a["embedding"])
        for a in artifacts
        if a.get("embedding") is not None
    ]
    if emb_pairs:
        file_ids = [p[0] for p in emb_pairs]
        embeddings_array = np.stack([p[1] for p in emb_pairs])
    else:
        file_ids = []
        embeddings_array = np.array([]).reshape(0, 768)

    embeddings_data = {
        "file_ids": file_ids,
        "embeddings": embeddings_array,
        "model": config.embedding_model,
        "dimension": embeddings_array.shape[1] if embeddings_array.size else 768,
    }
    pkl_path = output_dir / f"{repo_name}_embeddings.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(embeddings_data, f)
    exported["embeddings"] = pkl_path

    # --- Embedding metadata -------------------------------------------------
    emb_meta_rows = [
        {
            "file_id": a.get("file_id", ""),
            "has_embedding": True,
            "embedding_model": a.get("embedding_model", ""),
            "embedding_dim": a.get("embedding_dim", ""),
        }
        for a in artifacts
        if a.get("embedding") is not None
    ]
    emb_meta_path = output_dir / f"{repo_name}_embedding_metadata.csv"
    pd.DataFrame(emb_meta_rows).to_csv(emb_meta_path, index=False)
    exported["embedding_metadata"] = emb_meta_path

    # --- Temporal data ------------------------------------------------------
    timeseries = temporal_result.get("artifact_timeseries", [])
    commits = temporal_result.get("commit_aggregated", [])

    timeseries_df = pd.DataFrame(timeseries)
    commits_df = pd.DataFrame(commits)

    if not timeseries_df.empty:
        ts_path = output_dir / f"{repo_name}_artifact_timeseries.csv"
        timeseries_df.to_csv(ts_path, index=False)
        exported["artifact_timeseries"] = ts_path

    if not commits_df.empty:
        commits_path = output_dir / f"{repo_name}_commit_aggregated.csv"
        commits_df.to_csv(commits_path, index=False)
        exported["commit_aggregated"] = commits_path

    # --- Repo metrics JSON --------------------------------------------------
    if repo_metrics is not None:
        metrics_path = output_dir / f"{repo_name}_repo_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(repo_metrics, f, indent=2)
        exported["repo_metrics"] = metrics_path

    return exported


def bundle_artifacts_config(artifacts_dir: Path, output_dir: Path) -> Path:
    """Copy the Artifacts configuration directory into the output root.

    Idempotent: safe to call multiple times (overwrites existing copy).

    Returns:
        Path to the copied Artifacts directory inside *output_dir*.
    """
    dest = output_dir / "Artifacts"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(artifacts_dir, dest, dirs_exist_ok=True)
    return dest


def write_manifest(output_dir: Path, config: PipelineConfig) -> Path:
    """Write a manifest.json describing the collection run.

    Returns:
        Path to the written manifest file.
    """
    manifest = {
        "schema_version": "2.0",
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "embedding_model": config.embedding_model,
        "embedding_dim": 768,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest_path


def check_output_complete(output_dir: Path, repo_name: str) -> bool:
    """Check whether pipeline output already exists for *repo_name*.

    Returns ``True`` when the output directory contains the expected 4 CSVs
    and 1 PKL file.
    """
    repo_output = output_dir / repo_name
    if not repo_output.exists():
        return False
    csv_count = len(list(repo_output.glob("*.csv")))
    pkl_count = len(list(repo_output.glob("*.pkl")))
    return csv_count >= 4 and pkl_count >= 1


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_pipeline(
    config: PipelineConfig,
    token: Optional[str] = None,
    model: Optional[SentenceTransformer] = None,
) -> PipelineResult:
    """Run the full collection pipeline for a single repository.

    Args:
        config: Pipeline configuration.
        token: Optional git auth token.
        model: Optional pre-loaded embedding model. When provided, the model
            is reused and **not** unloaded (batch mode). When ``None``, the
            model is loaded and unloaded internally.

    Returns:
        :class:`PipelineResult` with all outputs populated.
    """
    # Phase 1 — tool configs
    print("  [phase 1/8] Loading tool configs...")
    tool_configs, shared_config = load_tool_configs(config.artifacts_dir)

    # Phase 2 — clone / checkout / pull
    print("  [phase 2/8] Cloning / checkout / pull...")
    repo_path = clone_and_prepare_repo(config, token=token)

    # Repo metrics
    repo_metrics = collect_repo_metrics_data(repo_path, config)

    # Phases 3+4 — discover & extract
    print("  [phase 3/8] Discovering & extracting artifacts...")
    artifacts = discover_and_extract(repo_path, tool_configs, shared_config)
    print(f"  [phase 3/8] Found {len(artifacts)} artifacts")

    # Phases 5+6 — embeddings
    print("  [phase 5/8] Generating embeddings...")
    artifacts, embedding_dim = generate_embeddings(artifacts, config, model=model)

    n_with = sum(1 for a in artifacts if a.get("embedding") is not None)
    n_without = len(artifacts) - n_with

    # Phase 7 — metadata
    print("  [phase 7/8] Building metadata...")
    metadata_df, artifacts = build_metadata(artifacts)

    # Phase 8 — temporal analysis
    print(f"  [phase 8/8] Temporal analysis ({len(artifacts)} artifacts)...")
    temporal_result = run_temporal_analysis(repo_path, artifacts, config)

    timeseries_df = pd.DataFrame(temporal_result.get("artifact_timeseries", []))
    commits_df = pd.DataFrame(temporal_result.get("commit_aggregated", []))

    # Build embeddings_data dict for the result
    emb_pairs = [
        (a["file_id"], a["embedding"])
        for a in artifacts
        if a.get("embedding") is not None
    ]
    if emb_pairs:
        file_ids = [p[0] for p in emb_pairs]
        embeddings_array = np.stack([p[1] for p in emb_pairs])
    else:
        file_ids = []
        embeddings_array = np.array([]).reshape(0, embedding_dim)

    embeddings_data = {
        "file_ids": file_ids,
        "embeddings": embeddings_array,
        "model": config.embedding_model,
        "dimension": embedding_dim,
    }

    # Export
    print("  [export] Writing results to disk...")
    export_results(artifacts, metadata_df, temporal_result, repo_metrics, config)

    return PipelineResult(
        artifacts=artifacts,
        metadata_df=metadata_df,
        embeddings_data=embeddings_data,
        timeseries_df=timeseries_df,
        commits_df=commits_df,
        repo_metrics=repo_metrics,
        n_with_embedding=n_with,
        n_without_embedding=n_without,
    )
