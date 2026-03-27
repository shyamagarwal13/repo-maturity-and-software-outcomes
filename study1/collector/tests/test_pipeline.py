"""Tests for the pipeline module.

All heavy dependencies (model loading, git cloning, file I/O) are mocked
so that tests run fast and without network or GPU access.
"""

import json
import pickle
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from src.pipeline import (
    PipelineConfig,
    PipelineResult,
    build_metadata,
    bundle_artifacts_config,
    check_output_complete,
    clone_and_prepare_repo,
    collect_repo_metrics_data,
    config_to_pipeline_config,
    discover_and_extract,
    export_results,
    generate_embeddings,
    load_config,
    load_model,
    load_tool_configs,
    run_pipeline,
    run_temporal_analysis,
    write_manifest,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config(tmp_path):
    """Minimal PipelineConfig pointing at tmp directories."""
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    return PipelineConfig(
        repo_url="https://github.com/test/repo.git",
        branch="main",
        repo_name="repo",
        clone_base_dir=tmp_path / "clones",
        artifacts_dir=artifacts_dir,
        output_dir=tmp_path / "output",
        repo_path=tmp_path / "clones" / "repo",
        embedding_model="nomic-ai/nomic-embed-text-v1.5",
        embedding_batch_size=16,
        start_date="2020-01-01",
        end_date=date.today().isoformat(),
        author_strategy="obfuscate",
        author_salt="test-salt",
        author_secret="",
        author_org="",
    )


@pytest.fixture
def sample_artifacts():
    """Artifacts after discover + extract (with text_content)."""
    return [
        {
            "file_path": "CLAUDE.md",
            "absolute_path": "/repo/CLAUDE.md",
            "tool_name": "claude-code",
            "artifact_name": "CLAUDE.md",
            "is_standard": True,
            "discovery_step": "tool_standard",
            "discovery_method": "exact_path",
            "file_size": 1234,
            "text_content": "# Claude instructions",
        },
        {
            "file_path": ".cursorrules",
            "absolute_path": "/repo/.cursorrules",
            "tool_name": "cursor",
            "artifact_name": ".cursorrules",
            "is_standard": True,
            "discovery_step": "tool_standard",
            "discovery_method": "exact_path",
            "file_size": 567,
            "text_content": "cursor rules here",
        },
        {
            "file_path": "empty.md",
            "absolute_path": "/repo/empty.md",
            "tool_name": "unknown",
            "artifact_name": "empty.md",
            "is_standard": False,
            "discovery_step": "non_standard_root",
            "discovery_method": "fallback",
            "file_size": 0,
            "text_content": None,
        },
    ]


@pytest.fixture
def embedded_artifacts(sample_artifacts):
    """Artifacts after embedding generation."""
    for a in sample_artifacts:
        if a["text_content"]:
            a["embedding"] = np.random.randn(768).astype(np.float32)
            a["embedding_model"] = "nomic-ai/nomic-embed-text-v1.5"
            a["embedding_dim"] = 768
        else:
            a["embedding"] = None
            a["embedding_model"] = "nomic-ai/nomic-embed-text-v1.5"
            a["embedding_dim"] = 768
    return sample_artifacts


# ---------------------------------------------------------------------------
# PipelineConfig / PipelineResult dataclass tests
# ---------------------------------------------------------------------------

class TestPipelineConfig:
    def test_defaults(self, tmp_path):
        cfg = PipelineConfig(
            repo_url="https://example.com/repo.git",
            branch="main",
            repo_name="repo",
            clone_base_dir=tmp_path,
            artifacts_dir=tmp_path,
            output_dir=tmp_path,
        )
        assert cfg.embedding_model == "nomic-ai/nomic-embed-text-v1.5"
        assert cfg.embedding_batch_size == 32
        assert cfg.start_date == "2020-01-01"
        assert cfg.repo_path is None

    def test_custom_values(self, config):
        assert config.embedding_batch_size == 16
        assert config.author_salt == "test-salt"


class TestPipelineResult:
    def test_defaults(self):
        r = PipelineResult()
        assert r.artifacts == []
        assert r.metadata_df is None
        assert r.n_with_embedding == 0


# ---------------------------------------------------------------------------
# load_config / config_to_pipeline_config
# ---------------------------------------------------------------------------

class TestLoadConfig:
    def test_valid_file(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            "embedding:\n  batch_size: 64\n"
            "temporal:\n  start_date: '2022-06-01'\n"
            "author:\n  strategy: anonymize\n  hash_length: 8\n  prefix: 'dev-'\n"
        )

        result = load_config(cfg_file)

        assert result["embedding"]["batch_size"] == 64
        assert result["temporal"]["start_date"] == "2022-06-01"
        assert result["author"]["strategy"] == "anonymize"
        assert result["author"]["hash_length"] == 8
        assert result["author"]["prefix"] == "dev-"

    def test_missing_file_returns_empty(self, tmp_path):
        result = load_config(tmp_path / "nonexistent.yaml")
        assert result == {}

    def test_none_path_returns_empty(self):
        result = load_config(None)
        assert result == {}

    def test_partial_config(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("embedding:\n  batch_size: 16\n")

        result = load_config(cfg_file)

        assert result["embedding"]["batch_size"] == 16
        assert "temporal" not in result
        assert "author" not in result

    def test_empty_file_returns_empty(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("")

        result = load_config(cfg_file)
        assert result == {}

    def test_warns_on_unknown_section(self, tmp_path, caplog):
        import logging
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("unknown_section:\n  key: value\n")

        with caplog.at_level(logging.WARNING, logger="src.pipeline"):
            load_config(cfg_file)

        assert "Unknown config section: unknown_section" in caplog.text

    def test_warns_on_unknown_key(self, tmp_path, caplog):
        import logging
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("embedding:\n  bad_key: 42\n")

        with caplog.at_level(logging.WARNING, logger="src.pipeline"):
            load_config(cfg_file)

        assert "Unknown config key: embedding.bad_key" in caplog.text


class TestConfigToPipelineConfig:
    def test_yaml_values_applied(self, tmp_path):
        yaml_cfg = {
            "embedding": {"batch_size": 64},
            "temporal": {"start_date": "2023-01-01"},
            "author": {
                "strategy": "anonymize",
                "secret": "my-study-secret",
                "hash_length": 8,
                "prefix": "dev-",
            },
        }

        cfg = config_to_pipeline_config(
            yaml_cfg,
            repo_url="https://github.com/test/repo.git",
            branch="main",
            repo_name="repo",
            clone_base_dir=tmp_path / "clones",
            artifacts_dir=tmp_path / "artifacts",
            output_dir=tmp_path / "output",
        )

        assert cfg.embedding_batch_size == 64
        assert cfg.start_date == "2023-01-01"
        assert cfg.author_strategy == "anonymize"
        assert cfg.author_secret == "my-study-secret"
        assert cfg.author_hash_length == 8
        assert cfg.author_prefix == "dev-"

    def test_overrides_win(self, tmp_path):
        yaml_cfg = {
            "embedding": {"batch_size": 64},
            "temporal": {"start_date": "2023-01-01"},
        }

        cfg = config_to_pipeline_config(
            yaml_cfg,
            repo_url="https://github.com/test/repo.git",
            branch="main",
            repo_name="repo",
            clone_base_dir=tmp_path / "clones",
            artifacts_dir=tmp_path / "artifacts",
            output_dir=tmp_path / "output",
            embedding_batch_size=128,
            start_date="2024-06-01",
        )

        assert cfg.embedding_batch_size == 128
        assert cfg.start_date == "2024-06-01"

    def test_empty_yaml_uses_defaults(self, tmp_path):
        cfg = config_to_pipeline_config(
            {},
            repo_url="https://github.com/test/repo.git",
            branch="main",
            repo_name="repo",
            clone_base_dir=tmp_path / "clones",
            artifacts_dir=tmp_path / "artifacts",
            output_dir=tmp_path / "output",
        )

        assert cfg.embedding_batch_size == 32
        assert cfg.start_date == "2020-01-01"
        assert cfg.author_strategy == "obfuscate"
        assert cfg.author_hash_length == 12
        assert cfg.author_prefix == "user-"

    def test_paths_applied(self, tmp_path):
        yaml_cfg = {
            "paths": {
                "output_dir": "out/data",
                "clone_dir": "/abs/clones",
            },
        }

        cfg = config_to_pipeline_config(
            yaml_cfg,
            repo_url="https://github.com/test/repo.git",
            branch="main",
            repo_name="repo",
            artifacts_dir=tmp_path / "artifacts",
        )

        assert cfg.output_dir == Path("out/data")
        assert cfg.clone_base_dir == Path("/abs/clones")

    def test_paths_overridden_by_kwargs(self, tmp_path):
        yaml_cfg = {
            "paths": {
                "output_dir": "yaml/output",
                "clone_dir": "yaml/clones",
            },
        }

        cfg = config_to_pipeline_config(
            yaml_cfg,
            repo_url="https://github.com/test/repo.git",
            branch="main",
            repo_name="repo",
            clone_base_dir=tmp_path / "override_clones",
            artifacts_dir=tmp_path / "override_artifacts",
            output_dir=tmp_path / "override_output",
        )

        # Overrides always win over YAML paths
        assert cfg.output_dir == tmp_path / "override_output"
        assert cfg.clone_base_dir == tmp_path / "override_clones"
        assert cfg.artifacts_dir == tmp_path / "override_artifacts"

    def test_paths_placeholder_passed_as_is(self, tmp_path):
        """Placeholders like {organisation} are not resolved in pipeline.py."""
        yaml_cfg = {
            "paths": {
                "output_dir": "output/{organisation}",
            },
        }

        cfg = config_to_pipeline_config(
            yaml_cfg,
            repo_url="https://github.com/test/repo.git",
            branch="main",
            repo_name="repo",
            clone_base_dir=tmp_path,
            artifacts_dir=tmp_path,
        )

        # The literal placeholder string should be preserved
        assert "{organisation}" in str(cfg.output_dir)

    def test_git_settings_applied(self, tmp_path):
        yaml_cfg = {
            "git": {
                "timeout": 600,
                "log_timeout": 120,
            },
        }

        cfg = config_to_pipeline_config(
            yaml_cfg,
            repo_url="https://github.com/test/repo.git",
            branch="main",
            repo_name="repo",
            clone_base_dir=tmp_path / "clones",
            artifacts_dir=tmp_path / "artifacts",
            output_dir=tmp_path / "output",
        )

        assert cfg.git_timeout == 600
        assert cfg.git_log_timeout == 120

    def test_embedding_memory_budget_applied(self, tmp_path):
        yaml_cfg = {
            "embedding": {"memory_budget_gib": 4},
        }

        cfg = config_to_pipeline_config(
            yaml_cfg,
            repo_url="https://github.com/test/repo.git",
            branch="main",
            repo_name="repo",
            clone_base_dir=tmp_path / "clones",
            artifacts_dir=tmp_path / "artifacts",
            output_dir=tmp_path / "output",
        )

        assert cfg.embedding_memory_budget_gib == 4

    def test_git_and_embedding_defaults(self, tmp_path):
        cfg = config_to_pipeline_config(
            {},
            repo_url="https://github.com/test/repo.git",
            branch="main",
            repo_name="repo",
            clone_base_dir=tmp_path / "clones",
            artifacts_dir=tmp_path / "artifacts",
            output_dir=tmp_path / "output",
        )

        assert cfg.git_timeout == 300
        assert cfg.git_log_timeout == 60
        assert cfg.embedding_memory_budget_gib == 2

    def test_end_date_null_uses_default(self, tmp_path):
        yaml_cfg = {"temporal": {"end_date": None}}

        cfg = config_to_pipeline_config(
            yaml_cfg,
            repo_url="https://github.com/test/repo.git",
            branch="main",
            repo_name="repo",
            clone_base_dir=tmp_path / "clones",
            artifacts_dir=tmp_path / "artifacts",
            output_dir=tmp_path / "output",
        )

        # end_date should be today (the dataclass default), not None
        from datetime import date
        assert cfg.end_date == date.today().isoformat()


# ---------------------------------------------------------------------------
# load_tool_configs
# ---------------------------------------------------------------------------

class TestLoadToolConfigs:
    @patch("src.pipeline.load_shared_config")
    @patch("src.pipeline.load_json_configs")
    def test_returns_tuple(self, mock_load, mock_shared, tmp_path):
        mock_load.return_value = {"cursor": Mock()}
        mock_shared.return_value = Mock()

        tools, shared = load_tool_configs(tmp_path / "Artifacts")

        mock_load.assert_called_once_with(str(tmp_path / "Artifacts"))
        mock_shared.assert_called_once_with(str(tmp_path / "Artifacts"))
        assert "cursor" in tools
        assert shared is not None


# ---------------------------------------------------------------------------
# clone_and_prepare_repo
# ---------------------------------------------------------------------------

class TestCloneAndPrepareRepo:
    @patch("src.pipeline.pull_latest")
    @patch("src.pipeline.checkout_branch")
    @patch("src.pipeline.clone_repository")
    def test_clones_when_missing(self, mock_clone, mock_checkout, mock_pull, config):
        # repo_path does not exist → should clone
        assert not config.repo_path.exists()

        # Simulate clone creating the directory as a side effect
        def create_dir(*args, **kwargs):
            config.repo_path.mkdir(parents=True, exist_ok=True)
            return str(config.repo_path)

        mock_clone.side_effect = create_dir

        result = clone_and_prepare_repo(config, token="tok123")

        mock_clone.assert_called_once_with(
            config.repo_url,
            str(config.clone_base_dir),
            branch="main",
            token="tok123",
            timeout=config.git_timeout,
        )
        mock_checkout.assert_called_once_with(str(config.repo_path), "main")
        mock_pull.assert_called_once_with(str(config.repo_path), "main", timeout=config.git_timeout)
        assert result == config.repo_path

    @patch("src.pipeline.pull_latest")
    @patch("src.pipeline.checkout_branch")
    @patch("src.pipeline.clone_repository")
    def test_skips_clone_when_exists(self, mock_clone, mock_checkout, mock_pull, config):
        config.repo_path.mkdir(parents=True)

        clone_and_prepare_repo(config)

        mock_clone.assert_not_called()
        mock_checkout.assert_called_once()
        mock_pull.assert_called_once()

    @patch("src.pipeline.pull_latest")
    @patch("src.pipeline.checkout_branch")
    @patch("src.pipeline.clone_repository")
    def test_raises_if_path_missing_after_clone(self, mock_clone, mock_checkout, mock_pull, config):
        # Path still doesn't exist after clone (simulating a failure)
        with pytest.raises(FileNotFoundError):
            clone_and_prepare_repo(config)

    @patch("src.pipeline.pull_latest")
    @patch("src.pipeline.checkout_branch")
    @patch("src.pipeline.clone_repository")
    def test_sets_repo_path_from_default(self, mock_clone, mock_checkout, mock_pull, config):
        config.repo_path = None
        expected = config.clone_base_dir / config.repo_name
        expected.mkdir(parents=True)

        result = clone_and_prepare_repo(config)

        assert result == expected
        assert config.repo_path == expected

    @patch("src.pipeline.reset_to_commit")
    @patch("src.pipeline.find_commit_at_date", return_value="abc123")
    @patch("src.pipeline.pull_latest")
    @patch("src.pipeline.checkout_branch")
    @patch("src.pipeline.clone_repository")
    def test_resets_to_end_date_when_in_past(
        self, mock_clone, mock_checkout, mock_pull, mock_find, mock_reset, config
    ):
        config.repo_path.mkdir(parents=True)
        config.end_date = "2025-03-31"

        clone_and_prepare_repo(config)

        mock_find.assert_called_once_with(
            str(config.repo_path), config.branch, "2025-03-31"
        )
        mock_reset.assert_called_once_with(str(config.repo_path), "abc123")

    @patch("src.pipeline.reset_to_commit")
    @patch("src.pipeline.find_commit_at_date", return_value=None)
    @patch("src.pipeline.pull_latest")
    @patch("src.pipeline.checkout_branch")
    @patch("src.pipeline.clone_repository")
    def test_raises_when_no_commit_before_end_date(
        self, mock_clone, mock_checkout, mock_pull, mock_find, mock_reset, config
    ):
        config.repo_path.mkdir(parents=True)
        config.end_date = "2019-01-01"

        with pytest.raises(ValueError, match="No commits found"):
            clone_and_prepare_repo(config)

        mock_reset.assert_not_called()

    @patch("src.pipeline.reset_to_commit")
    @patch("src.pipeline.find_commit_at_date")
    @patch("src.pipeline.pull_latest")
    @patch("src.pipeline.checkout_branch")
    @patch("src.pipeline.clone_repository")
    def test_skips_reset_when_end_date_is_today(
        self, mock_clone, mock_checkout, mock_pull, mock_find, mock_reset, config
    ):
        config.repo_path.mkdir(parents=True)
        from datetime import date
        config.end_date = date.today().isoformat()

        clone_and_prepare_repo(config)

        mock_find.assert_not_called()
        mock_reset.assert_not_called()


# ---------------------------------------------------------------------------
# collect_repo_metrics_data
# ---------------------------------------------------------------------------

class TestCollectRepoMetrics:
    @patch("src.pipeline.collect_repo_static_metrics")
    def test_enriches_with_config(self, mock_metrics, config):
        mock_metrics.return_value = {"total_commits": 100, "total_files": 50}

        result = collect_repo_metrics_data(config.repo_path, config)

        assert result["total_commits"] == 100
        assert result["repo_name"] == "repo"
        assert result["repo_url"] == config.repo_url
        assert result["branch"] == "main"


# ---------------------------------------------------------------------------
# discover_and_extract
# ---------------------------------------------------------------------------

class TestDiscoverAndExtract:
    @patch("src.pipeline.read_text_file")
    @patch("src.pipeline.get_file_size")
    @patch("src.pipeline.discover_artifacts")
    def test_populates_text_and_size(self, mock_discover, mock_size, mock_read, tmp_path):
        mock_discover.return_value = [
            {"file_path": "a.md", "absolute_path": "/repo/a.md"},
            {"file_path": "b.md", "absolute_path": "/repo/b.md"},
        ]
        mock_size.return_value = 42
        mock_read.return_value = {"success": True, "content": "hello"}

        result = discover_and_extract(tmp_path, {}, None)

        assert len(result) == 2
        assert result[0]["text_content"] == "hello"
        assert result[0]["file_size"] == 42

    @patch("src.pipeline.read_text_file")
    @patch("src.pipeline.get_file_size")
    @patch("src.pipeline.discover_artifacts")
    def test_handles_read_failure(self, mock_discover, mock_size, mock_read, tmp_path):
        mock_discover.return_value = [
            {"file_path": "bad.md", "absolute_path": "/repo/bad.md"},
        ]
        mock_size.return_value = 10
        mock_read.return_value = {"success": False, "error": "decode error"}

        result = discover_and_extract(tmp_path, {}, None)

        assert result[0]["text_content"] is None

    @patch("src.pipeline.read_text_file")
    @patch("src.pipeline.get_file_size")
    @patch("src.pipeline.discover_artifacts")
    def test_handles_exception(self, mock_discover, mock_size, mock_read, tmp_path):
        mock_discover.return_value = [
            {"file_path": "err.md", "absolute_path": "/repo/err.md"},
        ]
        mock_size.side_effect = OSError("nope")

        result = discover_and_extract(tmp_path, {}, None)

        assert result[0]["text_content"] is None
        assert result[0]["file_size"] == 0

    @patch("src.pipeline.read_text_file")
    @patch("src.pipeline.get_file_size")
    @patch("src.pipeline.discover_artifacts")
    def test_uses_absolute_path_when_present(self, mock_discover, mock_size, mock_read, tmp_path):
        mock_discover.return_value = [
            {"file_path": "sub/a.md", "absolute_path": "/custom/sub/a.md"},
        ]
        mock_size.return_value = 5
        mock_read.return_value = {"success": True, "content": "ok"}

        discover_and_extract(tmp_path, {}, None)

        mock_size.assert_called_with("/custom/sub/a.md")
        mock_read.assert_called_with("/custom/sub/a.md")

    @patch("src.pipeline.read_text_file")
    @patch("src.pipeline.get_file_size")
    @patch("src.pipeline.discover_artifacts")
    def test_falls_back_to_repo_path(self, mock_discover, mock_size, mock_read, tmp_path):
        mock_discover.return_value = [
            {"file_path": "a.md"},  # no absolute_path
        ]
        mock_size.return_value = 5
        mock_read.return_value = {"success": True, "content": "ok"}

        discover_and_extract(tmp_path, {}, None)

        expected = str(tmp_path / "a.md")
        mock_size.assert_called_with(expected)
        mock_read.assert_called_with(expected)


# ---------------------------------------------------------------------------
# load_model
# ---------------------------------------------------------------------------

class TestLoadModel:
    @patch("src.pipeline.load_embedding_model")
    def test_passes_config_values(self, mock_load, config):
        mock_load.return_value = Mock()

        load_model(config)

        mock_load.assert_called_once_with(
            "nomic-ai/nomic-embed-text-v1.5",
            cache_dir=None,
        )

    @patch("src.pipeline.load_embedding_model")
    def test_passes_custom_cache(self, mock_load, config):
        config.embedding_cache_dir = "/my/cache"
        mock_load.return_value = Mock()

        load_model(config)

        mock_load.assert_called_once_with(
            "nomic-ai/nomic-embed-text-v1.5",
            cache_dir="/my/cache",
        )


# ---------------------------------------------------------------------------
# generate_embeddings
# ---------------------------------------------------------------------------

class TestGenerateEmbeddings:
    @patch("src.pipeline.add_embeddings_to_artifacts")
    def test_reuses_provided_model(self, mock_add, config, sample_artifacts):
        model = Mock()
        model.get_sentence_embedding_dimension.return_value = 768
        mock_add.return_value = sample_artifacts

        artifacts, dim = generate_embeddings(sample_artifacts, config, model=model)

        mock_add.assert_called_once_with(
            sample_artifacts,
            model,
            config.embedding_model,
            batch_size=16,
            memory_budget_gib=config.embedding_memory_budget_gib,
        )
        assert dim == 768

    @patch("src.pipeline.load_model")
    @patch("src.pipeline.add_embeddings_to_artifacts")
    def test_loads_model_when_none(self, mock_add, mock_load, config, sample_artifacts):
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_load.return_value = mock_model
        mock_add.return_value = sample_artifacts

        artifacts, dim = generate_embeddings(sample_artifacts, config, model=None)

        mock_load.assert_called_once_with(config)
        assert dim == 768

    @patch("src.pipeline.add_embeddings_to_artifacts")
    def test_does_not_unload_provided_model(self, mock_add, config, sample_artifacts):
        model = Mock()
        model.get_sentence_embedding_dimension.return_value = 768
        mock_add.return_value = sample_artifacts

        generate_embeddings(sample_artifacts, config, model=model)

        # Model should still be usable (not deleted)
        model.get_sentence_embedding_dimension()  # no error


# ---------------------------------------------------------------------------
# build_metadata
# ---------------------------------------------------------------------------

class TestBuildMetadata:
    def test_assigns_file_ids(self, sample_artifacts):
        df, arts = build_metadata(sample_artifacts)

        assert arts[0]["file_id"] == "file_000"
        assert arts[1]["file_id"] == "file_001"
        assert arts[2]["file_id"] == "file_002"

    def test_metadata_columns(self, sample_artifacts):
        df, _ = build_metadata(sample_artifacts)

        expected_cols = {
            "file_id", "repo_name", "tool_name", "artifact_path",
            "artifact_name", "is_standard", "discovery_step",
            "discovery_method", "file_size", "has_embedding",
        }
        assert set(df.columns) == expected_cols

    def test_has_embedding_flag(self, embedded_artifacts):
        df, _ = build_metadata(embedded_artifacts)

        assert df.loc[0, "has_embedding"] == True
        assert df.loc[1, "has_embedding"] == True
        assert df.loc[2, "has_embedding"] == False

    def test_row_count(self, sample_artifacts):
        df, _ = build_metadata(sample_artifacts)
        assert len(df) == 3

    def test_empty_artifacts(self):
        df, arts = build_metadata([])
        assert len(df) == 0
        assert arts == []


# ---------------------------------------------------------------------------
# run_temporal_analysis
# ---------------------------------------------------------------------------

class TestRunTemporalAnalysis:
    @patch("src.pipeline.analyze_artifact_history")
    def test_delegates_correctly(self, mock_analyze, config, sample_artifacts):
        mock_analyze.return_value = {
            "artifact_timeseries": [{"action": "created"}],
            "commit_aggregated": [{"commit_sha": "abc"}],
        }

        result = run_temporal_analysis(config.repo_path, sample_artifacts, config)

        mock_analyze.assert_called_once()
        call_kwargs = mock_analyze.call_args[1]
        assert call_kwargs["repo_path"] == str(config.repo_path)
        assert call_kwargs["start_date"] == "2020-01-01"
        assert call_kwargs["end_date"] == config.end_date
        assert callable(call_kwargs["hash_fn"])
        assert len(call_kwargs["artifacts"]) == 3
        assert result["artifact_timeseries"][0]["action"] == "created"

    @patch("src.pipeline.analyze_artifact_history")
    def test_obfuscate_strategy_uses_obfuscate_author(self, mock_analyze, config, sample_artifacts):
        """With obfuscate strategy, hash_fn should use obfuscate_author."""
        mock_analyze.return_value = {"artifact_timeseries": [], "commit_aggregated": []}
        config.author_strategy = "obfuscate"
        config.author_salt = "my-salt"

        run_temporal_analysis(config.repo_path, sample_artifacts, config)

        hash_fn = mock_analyze.call_args[1]["hash_fn"]
        from src.temporal_analyzer import obfuscate_author
        assert hash_fn("test@example.com") == obfuscate_author("test@example.com", "my-salt")

    @patch("src.pipeline.analyze_artifact_history")
    def test_anonymize_strategy_uses_anonymize_author(self, mock_analyze, config, sample_artifacts):
        """With anonymize strategy, hash_fn should use anonymize_author."""
        mock_analyze.return_value = {"artifact_timeseries": [], "commit_aggregated": []}
        config.author_strategy = "anonymize"
        config.author_secret = "s3cret"
        config.author_org = "myorg"
        config.author_hash_length = 12
        config.author_prefix = "user-"

        run_temporal_analysis(config.repo_path, sample_artifacts, config)

        hash_fn = mock_analyze.call_args[1]["hash_fn"]
        from src.temporal_analyzer import anonymize_author
        assert hash_fn("test@example.com") == anonymize_author(
            "test@example.com", "myorg", "s3cret", 12, "user-"
        )


# ---------------------------------------------------------------------------
# export_results
# ---------------------------------------------------------------------------

class TestExportResults:
    def test_creates_output_files(self, config, embedded_artifacts):
        metadata_df, _ = build_metadata(embedded_artifacts)
        temporal = {
            "artifact_timeseries": [{"action": "created", "path": "a.md"}],
            "commit_aggregated": [{"sha": "abc", "files": 2}],
        }
        metrics = {"total_commits": 10, "repo_name": "repo"}

        exported = export_results(
            embedded_artifacts, metadata_df, temporal, metrics, config
        )

        assert "file_artifacts" in exported
        assert "embeddings" in exported
        assert "embedding_metadata" in exported
        assert "artifact_timeseries" in exported
        assert "commit_aggregated" in exported
        assert "repo_metrics" in exported

        # Verify files exist on disk
        for path in exported.values():
            assert path.exists(), f"Missing: {path}"

    def test_metadata_csv_filters_no_embedding(self, config, embedded_artifacts):
        metadata_df, _ = build_metadata(embedded_artifacts)
        temporal = {"artifact_timeseries": [], "commit_aggregated": []}

        exported = export_results(
            embedded_artifacts, metadata_df, temporal, None, config
        )

        df = pd.read_csv(exported["file_artifacts"])
        # The third artifact (empty.md) has no embedding and should be filtered
        assert len(df) == 2
        assert all(df["has_embedding"])

    def test_embeddings_pickle_shape(self, config, embedded_artifacts):
        metadata_df, _ = build_metadata(embedded_artifacts)
        temporal = {"artifact_timeseries": [], "commit_aggregated": []}

        exported = export_results(
            embedded_artifacts, metadata_df, temporal, None, config
        )

        with open(exported["embeddings"], "rb") as f:
            data = pickle.load(f)

        assert data["embeddings"].shape == (2, 768)
        assert len(data["file_ids"]) == 2
        assert data["model"] == "nomic-ai/nomic-embed-text-v1.5"

    def test_repo_metrics_json(self, config, embedded_artifacts):
        metadata_df, _ = build_metadata(embedded_artifacts)
        temporal = {"artifact_timeseries": [], "commit_aggregated": []}
        metrics = {"total_commits": 42}

        exported = export_results(
            embedded_artifacts, metadata_df, temporal, metrics, config
        )

        with open(exported["repo_metrics"]) as f:
            data = json.load(f)
        assert data["total_commits"] == 42

    def test_no_repo_metrics(self, config, embedded_artifacts):
        metadata_df, _ = build_metadata(embedded_artifacts)
        temporal = {"artifact_timeseries": [], "commit_aggregated": []}

        exported = export_results(
            embedded_artifacts, metadata_df, temporal, None, config
        )

        assert "repo_metrics" not in exported

    def test_empty_temporal(self, config, embedded_artifacts):
        metadata_df, _ = build_metadata(embedded_artifacts)
        temporal = {"artifact_timeseries": [], "commit_aggregated": []}

        exported = export_results(
            embedded_artifacts, metadata_df, temporal, None, config
        )

        assert "artifact_timeseries" not in exported
        assert "commit_aggregated" not in exported

    def test_output_dir_created(self, config, embedded_artifacts):
        metadata_df, _ = build_metadata(embedded_artifacts)
        temporal = {"artifact_timeseries": [], "commit_aggregated": []}

        # Output dir doesn't exist yet
        assert not (config.output_dir / config.repo_name).exists()

        export_results(embedded_artifacts, metadata_df, temporal, None, config)

        assert (config.output_dir / config.repo_name).exists()

    def test_bundles_artifacts_config(self, config, embedded_artifacts):
        # Create a JSON file in the artifacts dir
        (config.artifacts_dir / "cursor-files.json").write_text('{"tool": "cursor"}')

        metadata_df, _ = build_metadata(embedded_artifacts)
        temporal = {"artifact_timeseries": [], "commit_aggregated": []}

        export_results(embedded_artifacts, metadata_df, temporal, None, config)

        assert (config.output_dir / "Artifacts").is_dir()
        assert (config.output_dir / "Artifacts" / "cursor-files.json").exists()

    def test_writes_manifest(self, config, embedded_artifacts):
        metadata_df, _ = build_metadata(embedded_artifacts)
        temporal = {"artifact_timeseries": [], "commit_aggregated": []}

        export_results(embedded_artifacts, metadata_df, temporal, None, config)

        manifest_path = config.output_dir / "manifest.json"
        assert manifest_path.exists()
        with open(manifest_path) as f:
            data = json.load(f)
        assert data["schema_version"] == "2.0"
        assert data["embedding_model"] == config.embedding_model


# ---------------------------------------------------------------------------
# bundle_artifacts_config
# ---------------------------------------------------------------------------

class TestBundleArtifactsConfig:
    def test_copies_artifacts_dir(self, tmp_path):
        src_dir = tmp_path / "Artifacts"
        src_dir.mkdir()
        (src_dir / "cursor-files.json").write_text('{"tool": "cursor"}')
        (src_dir / "claude-files.json").write_text('{"tool": "claude"}')

        out_dir = tmp_path / "output"
        out_dir.mkdir()

        bundle_artifacts_config(src_dir, out_dir)

        dest = out_dir / "Artifacts"
        assert dest.is_dir()
        assert (dest / "cursor-files.json").exists()
        assert (dest / "claude-files.json").exists()

    def test_idempotent(self, tmp_path):
        src_dir = tmp_path / "Artifacts"
        src_dir.mkdir()
        (src_dir / "tool.json").write_text('{"v": 1}')

        out_dir = tmp_path / "output"
        out_dir.mkdir()

        bundle_artifacts_config(src_dir, out_dir)
        bundle_artifacts_config(src_dir, out_dir)  # second call should not error

        assert (out_dir / "Artifacts" / "tool.json").exists()

    def test_preserves_json_files(self, tmp_path):
        src_dir = tmp_path / "Artifacts"
        src_dir.mkdir()
        content = '{"tool": "cursor", "patterns": ["*.md"]}'
        (src_dir / "cursor-files.json").write_text(content)

        out_dir = tmp_path / "output"
        out_dir.mkdir()

        bundle_artifacts_config(src_dir, out_dir)

        copied = (out_dir / "Artifacts" / "cursor-files.json").read_text()
        assert json.loads(copied) == json.loads(content)


# ---------------------------------------------------------------------------
# write_manifest
# ---------------------------------------------------------------------------

class TestWriteManifest:
    def test_writes_manifest_json(self, tmp_path, config):
        config.output_dir = tmp_path / "output"

        write_manifest(config.output_dir, config)

        manifest_path = config.output_dir / "manifest.json"
        assert manifest_path.exists()
        with open(manifest_path) as f:
            data = json.load(f)
        assert "schema_version" in data
        assert "collected_at" in data
        assert "embedding_model" in data
        assert "embedding_dim" in data

    def test_schema_version(self, tmp_path, config):
        config.output_dir = tmp_path / "output"

        write_manifest(config.output_dir, config)

        with open(config.output_dir / "manifest.json") as f:
            data = json.load(f)
        assert data["schema_version"] == "2.0"

    def test_collected_at_is_iso(self, tmp_path, config):
        from datetime import datetime as dt

        config.output_dir = tmp_path / "output"

        write_manifest(config.output_dir, config)

        with open(config.output_dir / "manifest.json") as f:
            data = json.load(f)
        # Should parse without error — valid ISO 8601
        parsed = dt.fromisoformat(data["collected_at"])
        assert parsed is not None

    def test_embedding_model_from_config(self, tmp_path, config):
        config.output_dir = tmp_path / "output"
        config.embedding_model = "custom-model/v2"

        write_manifest(config.output_dir, config)

        with open(config.output_dir / "manifest.json") as f:
            data = json.load(f)
        assert data["embedding_model"] == "custom-model/v2"


# ---------------------------------------------------------------------------
# check_output_complete
# ---------------------------------------------------------------------------

class TestCheckOutputComplete:
    def test_returns_true_when_complete(self, tmp_path):
        out = tmp_path / "output" / "repo"
        out.mkdir(parents=True)
        for name in ["a.csv", "b.csv", "c.csv", "d.csv"]:
            (out / name).write_text("data")
        (out / "e.pkl").write_bytes(b"\x80")

        assert check_output_complete(tmp_path / "output", "repo") is True

    def test_returns_false_when_missing_csv(self, tmp_path):
        out = tmp_path / "output" / "repo"
        out.mkdir(parents=True)
        for name in ["a.csv", "b.csv"]:
            (out / name).write_text("data")
        (out / "e.pkl").write_bytes(b"\x80")

        assert check_output_complete(tmp_path / "output", "repo") is False

    def test_returns_false_when_missing_pkl(self, tmp_path):
        out = tmp_path / "output" / "repo"
        out.mkdir(parents=True)
        for name in ["a.csv", "b.csv", "c.csv", "d.csv"]:
            (out / name).write_text("data")

        assert check_output_complete(tmp_path / "output", "repo") is False

    def test_returns_false_when_dir_missing(self, tmp_path):
        assert check_output_complete(tmp_path / "output", "repo") is False


# ---------------------------------------------------------------------------
# run_pipeline (integration)
# ---------------------------------------------------------------------------

class TestRunPipeline:
    @patch("src.pipeline.run_temporal_analysis")
    @patch("src.pipeline.add_embeddings_to_artifacts")
    @patch("src.pipeline.discover_and_extract")
    @patch("src.pipeline.collect_repo_metrics_data")
    @patch("src.pipeline.clone_and_prepare_repo")
    @patch("src.pipeline.load_tool_configs")
    def test_full_pipeline(
        self,
        mock_configs,
        mock_clone,
        mock_metrics,
        mock_discover,
        mock_embed,
        mock_temporal,
        config,
        sample_artifacts,
    ):
        # Setup mocks
        mock_configs.return_value = ({}, Mock())
        mock_clone.return_value = config.repo_path
        config.repo_path.mkdir(parents=True)
        mock_metrics.return_value = {"total_commits": 5}

        # Add text content so embeddings work
        for a in sample_artifacts:
            if a["text_content"]:
                a["embedding"] = np.random.randn(768).astype(np.float32)
                a["embedding_model"] = config.embedding_model
                a["embedding_dim"] = 768
            else:
                a["embedding"] = None
                a["embedding_model"] = config.embedding_model
                a["embedding_dim"] = 768

        mock_discover.return_value = sample_artifacts
        mock_embed.return_value = sample_artifacts

        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 768

        mock_temporal.return_value = {
            "artifact_timeseries": [],
            "commit_aggregated": [],
        }

        result = run_pipeline(config, token="tok", model=mock_model)

        assert isinstance(result, PipelineResult)
        assert result.n_with_embedding == 2
        assert result.n_without_embedding == 1
        assert result.repo_metrics["total_commits"] == 5
        assert result.metadata_df is not None
        assert len(result.artifacts) == 3

    @patch("src.pipeline.run_temporal_analysis")
    @patch("src.pipeline.add_embeddings_to_artifacts")
    @patch("src.pipeline.discover_and_extract")
    @patch("src.pipeline.collect_repo_metrics_data")
    @patch("src.pipeline.clone_and_prepare_repo")
    @patch("src.pipeline.load_tool_configs")
    def test_pipeline_exports_files(
        self,
        mock_configs,
        mock_clone,
        mock_metrics,
        mock_discover,
        mock_embed,
        mock_temporal,
        config,
        sample_artifacts,
    ):
        mock_configs.return_value = ({}, Mock())
        mock_clone.return_value = config.repo_path
        config.repo_path.mkdir(parents=True)
        mock_metrics.return_value = {"total_commits": 1}

        for a in sample_artifacts:
            if a["text_content"]:
                a["embedding"] = np.random.randn(768).astype(np.float32)
                a["embedding_model"] = config.embedding_model
                a["embedding_dim"] = 768
            else:
                a["embedding"] = None
                a["embedding_model"] = config.embedding_model
                a["embedding_dim"] = 768

        mock_discover.return_value = sample_artifacts
        mock_embed.return_value = sample_artifacts

        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 768

        mock_temporal.return_value = {
            "artifact_timeseries": [],
            "commit_aggregated": [],
        }

        run_pipeline(config, model=mock_model)

        # Verify output files created
        out = config.output_dir / config.repo_name
        assert out.exists()
        assert (out / f"{config.repo_name}_file_artifacts.csv").exists()
        assert (out / f"{config.repo_name}_embeddings.pkl").exists()
        assert (out / f"{config.repo_name}_embedding_metadata.csv").exists()
