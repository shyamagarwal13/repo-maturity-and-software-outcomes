"""Tests for temporal_health module."""

import os

import pandas as pd
import pytest

from src.temporal_health import (
    GROUNDING_CATEGORIES,
    AGENTIC_CATEGORIES,
    CODE_STYLE_CATEGORIES,
    TemporalHealth,
    classify_artifact_lifecycle,
    analyze_temporal_health,
    load_timeseries,
    _build_health_verdicts,
    _category_tier,
)


# ============================================================================
# Helpers
# ============================================================================

def _make_commits(dates):
    """Create a DataFrame of commits from a list of date strings."""
    return pd.DataFrame({
        "commit_date": pd.to_datetime(dates, utc=True),
    })


def _make_timeseries_csv(tmp_path, rows):
    """Write a timeseries CSV and return the path."""
    df = pd.DataFrame(rows)
    csv_path = tmp_path / "test_artifact_timeseries.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def _make_classifications(entries):
    """Create a file_classifications DataFrame."""
    return pd.DataFrame(entries)


# ============================================================================
# Tests: classify_artifact_lifecycle
# ============================================================================

class TestClassifyArtifactLifecycle:

    def test_single_commit_is_set_and_forget(self):
        commits = _make_commits(["2024-06-01"])
        horizon = pd.Timestamp("2024-12-01", tz="UTC")
        assert classify_artifact_lifecycle(commits, horizon) == "set-and-forget"

    def test_empty_commits_is_set_and_forget(self):
        commits = pd.DataFrame({"commit_date": pd.Series([], dtype="datetime64[ns, UTC]")})
        horizon = pd.Timestamp("2024-12-01", tz="UTC")
        assert classify_artifact_lifecycle(commits, horizon) == "set-and-forget"

    def test_commits_within_window_is_burst(self):
        # All commits within 30 days
        commits = _make_commits(["2024-06-01", "2024-06-10", "2024-06-20"])
        horizon = pd.Timestamp("2024-12-01", tz="UTC")
        assert classify_artifact_lifecycle(commits, horizon) == "burst"

    def test_spread_commits_is_steady(self):
        # Commits spread over 6+ months, last one recent
        commits = _make_commits([
            "2024-01-15", "2024-04-20", "2024-07-10", "2024-11-05",
        ])
        horizon = pd.Timestamp("2024-12-01", tz="UTC")
        assert classify_artifact_lifecycle(commits, horizon) == "steady"

    def test_old_commits_before_horizon_is_abandoned(self):
        # Last commit more than 6 months before horizon
        commits = _make_commits(["2023-06-01", "2023-08-15", "2024-01-10"])
        horizon = pd.Timestamp("2024-12-01", tz="UTC")
        assert classify_artifact_lifecycle(commits, horizon) == "abandoned"

    def test_custom_thresholds(self):
        # With a tight burst window (7 days), 20-day span is not burst
        # Last commit (June 21) is only ~2 months before horizon (Sep 1) — not abandoned with 3mo threshold
        commits = _make_commits(["2024-06-01", "2024-06-21"])
        horizon = pd.Timestamp("2024-09-01", tz="UTC")
        result = classify_artifact_lifecycle(
            commits, horizon, burst_window_days=7, abandonment_months=3,
        )
        assert result == "steady"

        # Same commits with a far horizon — now June 21 is > 3 months before Jan 2025
        horizon_far = pd.Timestamp("2025-01-01", tz="UTC")
        result2 = classify_artifact_lifecycle(
            commits, horizon_far, burst_window_days=7, abandonment_months=3,
        )
        assert result2 == "abandoned"


# ============================================================================
# Tests: analyze_temporal_health
# ============================================================================

class TestAnalyzeTemporalHealth:

    def test_no_csv_returns_empty(self, tmp_path):
        fc = _make_classifications([
            {"artifact_path": "CLAUDE.md", "assigned_category": "rules"},
        ])
        result = analyze_temporal_health(str(tmp_path), fc)
        assert result.has_timeseries is False
        assert result.artifact_lifecycles.empty
        assert result.health_verdicts == []

    def test_end_to_end_with_csv(self, tmp_path):
        # Write a fake timeseries CSV
        rows = [
            {"commit_sha": "abc1", "commit_date": "2024-01-15T10:00:00Z",
             "artifact_path": "CLAUDE.md", "artifact_type": "instructions",
             "action": "created", "author_hash": "auth1"},
            {"commit_sha": "abc2", "commit_date": "2024-04-20T10:00:00Z",
             "artifact_path": "CLAUDE.md", "artifact_type": "instructions",
             "action": "modified", "author_hash": "auth1"},
            {"commit_sha": "abc3", "commit_date": "2024-08-10T10:00:00Z",
             "artifact_path": "CLAUDE.md", "artifact_type": "instructions",
             "action": "modified", "author_hash": "auth2"},
            {"commit_sha": "def1", "commit_date": "2024-03-01T10:00:00Z",
             "artifact_path": ".cursorrules", "artifact_type": "ide_configs",
             "action": "created", "author_hash": "auth1"},
        ]
        _make_timeseries_csv(tmp_path, rows)

        fc = _make_classifications([
            {"artifact_path": "CLAUDE.md", "assigned_category": "rules"},
            {"artifact_path": ".cursorrules", "assigned_category": "configuration"},
        ])

        result = analyze_temporal_health(str(tmp_path), fc)
        assert result.has_timeseries is True
        assert len(result.artifact_lifecycles) == 2
        assert len(result.health_verdicts) > 0
        assert "rules" in result.author_diversity
        assert result.horizon_date is not None

        # CLAUDE.md should be "steady" (spread over months)
        claude_row = result.artifact_lifecycles[
            result.artifact_lifecycles["artifact_path"] == "CLAUDE.md"
        ].iloc[0]
        assert claude_row["lifecycle"] == "steady"

        # .cursorrules should be "set-and-forget" (1 commit)
        cursor_row = result.artifact_lifecycles[
            result.artifact_lifecycles["artifact_path"] == ".cursorrules"
        ].iloc[0]
        assert cursor_row["lifecycle"] == "set-and-forget"

    def test_empty_classifications(self, tmp_path):
        rows = [
            {"commit_sha": "abc1", "commit_date": "2024-01-15T10:00:00Z",
             "artifact_path": "CLAUDE.md", "artifact_type": "instructions",
             "action": "created", "author_hash": "auth1"},
        ]
        _make_timeseries_csv(tmp_path, rows)

        fc = pd.DataFrame()
        result = analyze_temporal_health(str(tmp_path), fc)
        assert result.has_timeseries is False

    def test_no_matching_paths(self, tmp_path):
        rows = [
            {"commit_sha": "abc1", "commit_date": "2024-01-15T10:00:00Z",
             "artifact_path": "CLAUDE.md", "artifact_type": "instructions",
             "action": "created", "author_hash": "auth1"},
        ]
        _make_timeseries_csv(tmp_path, rows)

        fc = _make_classifications([
            {"artifact_path": "OTHER.md", "assigned_category": "rules"},
        ])
        result = analyze_temporal_health(str(tmp_path), fc)
        assert result.has_timeseries is False


# ============================================================================
# Tests: health verdicts
# ============================================================================

class TestHealthVerdicts:

    def _make_summary(self, category, steady=0, burst=0, set_and_forget=0, abandoned=0):
        return pd.DataFrame([{
            "category": category,
            "total_artifacts": steady + burst + set_and_forget + abandoned,
            "total_commits": 10,
            "unique_authors": 2,
            "steady": steady,
            "burst": burst,
            "set-and-forget": set_and_forget,
            "abandoned": abandoned,
        }])

    def test_grounding_set_and_forget_is_concern(self):
        summary = self._make_summary("rules", set_and_forget=5)
        verdicts = _build_health_verdicts(summary)
        assert len(verdicts) == 1
        assert verdicts[0]["verdict"] == "concern"
        assert verdicts[0]["dominant_lifecycle"] == "set-and-forget"

    def test_grounding_steady_is_healthy(self):
        summary = self._make_summary("rules", steady=5)
        verdicts = _build_health_verdicts(summary)
        assert verdicts[0]["verdict"] == "healthy"

    def test_grounding_abandoned_is_concern(self):
        summary = self._make_summary("architecture", abandoned=3)
        verdicts = _build_health_verdicts(summary)
        assert verdicts[0]["verdict"] == "concern"

    def test_grounding_burst_is_warning(self):
        summary = self._make_summary("configuration", burst=4)
        verdicts = _build_health_verdicts(summary)
        assert verdicts[0]["verdict"] == "warning"

    def test_code_style_set_and_forget_is_healthy(self):
        summary = self._make_summary("code-style", set_and_forget=5)
        verdicts = _build_health_verdicts(summary)
        assert verdicts[0]["verdict"] == "healthy"
        assert verdicts[0]["tier"] == "code-style"

    def test_code_style_abandoned_is_healthy(self):
        summary = self._make_summary("code-style", abandoned=3)
        verdicts = _build_health_verdicts(summary)
        assert verdicts[0]["verdict"] == "healthy"

    def test_session_logs_burst_is_healthy(self):
        summary = self._make_summary("session-logs", burst=10)
        verdicts = _build_health_verdicts(summary)
        assert verdicts[0]["verdict"] == "healthy"

    def test_agentic_abandoned_is_warning(self):
        summary = self._make_summary("agents", abandoned=5)
        verdicts = _build_health_verdicts(summary)
        assert verdicts[0]["verdict"] == "warning"

    def test_agentic_set_and_forget_is_healthy(self):
        summary = self._make_summary("skills", set_and_forget=3)
        verdicts = _build_health_verdicts(summary)
        assert verdicts[0]["verdict"] == "healthy"

    def test_flows_abandoned_is_concern(self):
        summary = self._make_summary("flows", abandoned=2)
        verdicts = _build_health_verdicts(summary)
        assert verdicts[0]["verdict"] == "concern"


# ============================================================================
# Tests: author diversity
# ============================================================================

class TestAuthorDiversity:

    def test_single_author(self, tmp_path):
        rows = [
            {"commit_sha": "a1", "commit_date": "2024-01-15T10:00:00Z",
             "artifact_path": "CLAUDE.md", "artifact_type": "instructions",
             "action": "created", "author_hash": "auth1"},
            {"commit_sha": "a2", "commit_date": "2024-03-15T10:00:00Z",
             "artifact_path": "CLAUDE.md", "artifact_type": "instructions",
             "action": "modified", "author_hash": "auth1"},
        ]
        _make_timeseries_csv(tmp_path, rows)

        fc = _make_classifications([
            {"artifact_path": "CLAUDE.md", "assigned_category": "rules"},
        ])

        result = analyze_temporal_health(str(tmp_path), fc)
        assert result.author_diversity["rules"] == 1

    def test_multiple_authors(self, tmp_path):
        rows = [
            {"commit_sha": "a1", "commit_date": "2024-01-15T10:00:00Z",
             "artifact_path": "CLAUDE.md", "artifact_type": "instructions",
             "action": "created", "author_hash": "auth1"},
            {"commit_sha": "a2", "commit_date": "2024-03-15T10:00:00Z",
             "artifact_path": "CLAUDE.md", "artifact_type": "instructions",
             "action": "modified", "author_hash": "auth2"},
            {"commit_sha": "a3", "commit_date": "2024-06-15T10:00:00Z",
             "artifact_path": "CLAUDE.md", "artifact_type": "instructions",
             "action": "modified", "author_hash": "auth3"},
        ]
        _make_timeseries_csv(tmp_path, rows)

        fc = _make_classifications([
            {"artifact_path": "CLAUDE.md", "assigned_category": "rules"},
        ])

        result = analyze_temporal_health(str(tmp_path), fc)
        assert result.author_diversity["rules"] == 3


# ============================================================================
# Tests: no-history and unknown verdicts
# ============================================================================

class TestNoHistory:

    def test_artifacts_without_commits_get_no_history(self, tmp_path):
        """Artifacts in file_classifications but NOT in timeseries get lifecycle='no-history'."""
        rows = [
            {"commit_sha": "a1", "commit_date": "2024-01-15T10:00:00Z",
             "artifact_path": "CLAUDE.md", "artifact_type": "instructions",
             "action": "created", "author_hash": "auth1"},
            {"commit_sha": "a2", "commit_date": "2024-06-15T10:00:00Z",
             "artifact_path": "CLAUDE.md", "artifact_type": "instructions",
             "action": "modified", "author_hash": "auth1"},
        ]
        _make_timeseries_csv(tmp_path, rows)

        fc = _make_classifications([
            {"artifact_path": "CLAUDE.md", "assigned_category": "rules"},
            {"artifact_path": "other.md", "assigned_category": "architecture"},
            {"artifact_path": "style.json", "assigned_category": "code-style"},
        ])

        result = analyze_temporal_health(str(tmp_path), fc)
        assert result.has_timeseries is True
        # All 3 artifacts should be in lifecycles
        assert len(result.artifact_lifecycles) == 3

        # CLAUDE.md has history
        claude_row = result.artifact_lifecycles[
            result.artifact_lifecycles["artifact_path"] == "CLAUDE.md"
        ].iloc[0]
        assert claude_row["lifecycle"] != "no-history"

        # other.md and style.json have no history
        for path in ["other.md", "style.json"]:
            row = result.artifact_lifecycles[
                result.artifact_lifecycles["artifact_path"] == path
            ].iloc[0]
            assert row["lifecycle"] == "no-history"
            assert row["commit_count"] == 0

    def test_category_summary_includes_no_history_column(self, tmp_path):
        rows = [
            {"commit_sha": "a1", "commit_date": "2024-01-15T10:00:00Z",
             "artifact_path": "CLAUDE.md", "artifact_type": "instructions",
             "action": "created", "author_hash": "auth1"},
        ]
        _make_timeseries_csv(tmp_path, rows)

        fc = _make_classifications([
            {"artifact_path": "CLAUDE.md", "assigned_category": "rules"},
            {"artifact_path": "other.md", "assigned_category": "rules"},
        ])

        result = analyze_temporal_health(str(tmp_path), fc)
        assert "no-history" in result.category_summaries.columns
        rules_row = result.category_summaries[
            result.category_summaries["category"] == "rules"
        ].iloc[0]
        assert rules_row["no-history"] == 1  # other.md

    def test_unknown_verdict_when_no_history_only(self):
        """Category with only no-history artifacts gets verdict='unknown'."""
        summary = pd.DataFrame([{
            "category": "rules",
            "total_artifacts": 5,
            "total_commits": 0,
            "unique_authors": 0,
            "steady": 0,
            "burst": 0,
            "set-and-forget": 0,
            "abandoned": 0,
        }])
        verdicts = _build_health_verdicts(summary)
        assert len(verdicts) == 1
        assert verdicts[0]["verdict"] == "unknown"
        assert verdicts[0]["dominant_lifecycle"] == "no-history"


# ============================================================================
# Tests: load_timeseries
# ============================================================================

class TestLoadTimeseries:

    def test_finds_csv_in_subdirectory(self, tmp_path):
        subdir = tmp_path / "nested"
        subdir.mkdir()
        rows = [
            {"commit_sha": "a1", "commit_date": "2024-01-15T10:00:00Z",
             "artifact_path": "test.md", "artifact_type": "instructions",
             "action": "created", "author_hash": "auth1"},
        ]
        df = pd.DataFrame(rows)
        df.to_csv(subdir / "repo_artifact_timeseries.csv", index=False)

        result = load_timeseries(str(tmp_path))
        assert result is not None
        assert len(result) == 1

    def test_returns_none_when_no_csv(self, tmp_path):
        result = load_timeseries(str(tmp_path))
        assert result is None


# ============================================================================
# Tests: _category_tier
# ============================================================================

class TestCategoryTier:

    def test_grounding(self):
        for cat in GROUNDING_CATEGORIES:
            assert _category_tier(cat) == "grounding"

    def test_agentic(self):
        for cat in AGENTIC_CATEGORIES:
            assert _category_tier(cat) == "agentic"

    def test_code_style(self):
        for cat in CODE_STYLE_CATEGORIES:
            assert _category_tier(cat) == "code-style"

    def test_flows(self):
        assert _category_tier("flows") == "flows"

    def test_session_logs(self):
        assert _category_tier("session-logs") == "session-logs"

    def test_unknown_defaults_to_grounding(self):
        assert _category_tier("unknown-category") == "grounding"
