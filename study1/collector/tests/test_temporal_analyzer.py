"""
Tests for temporal analyzer module (Phase 7).

This module tests git log analysis functionality for tracking AI artifact evolution.
"""

import pytest
from datetime import datetime
import os
import tempfile
import subprocess
from pathlib import Path

from src.temporal_analyzer import (
    anonymize_author,
    obfuscate_author,
    parse_iso_date,
    format_iso_date,
    determine_file_action,
    get_file_history,
    build_artifact_timeseries,
    build_commit_aggregated,
    analyze_artifact_history,
    collect_repo_static_metrics,
)


def test_obfuscate_author():
    """Obfuscate author email with salt"""
    email = "test@example.com"
    salt = "my-salt-123"

    hash1 = obfuscate_author(email, salt)

    assert len(hash1) == 16
    assert isinstance(hash1, str)

    # Same input should produce same hash
    hash2 = obfuscate_author(email, salt)
    assert hash1 == hash2

    # Different salt should produce different hash
    hash3 = obfuscate_author(email, "different-salt")
    assert hash1 != hash3


def test_obfuscate_author_different_emails():
    """Different emails produce different hashes"""
    salt = "same-salt"

    hash1 = obfuscate_author("user1@example.com", salt)
    hash2 = obfuscate_author("user2@example.com", salt)

    assert hash1 != hash2


def test_parse_iso_date():
    """Parse ISO 8601 date string"""
    date_str = "2023-06-15T10:30:00Z"

    dt = parse_iso_date(date_str)

    assert dt.year == 2023
    assert dt.month == 6
    assert dt.day == 15


def test_parse_iso_date_variations():
    """Parse various ISO 8601 formats"""
    # With timezone
    dt1 = parse_iso_date("2023-06-15T10:30:00+00:00")
    assert dt1.year == 2023

    # Date only
    dt2 = parse_iso_date("2023-06-15")
    assert dt2.year == 2023
    assert dt2.month == 6
    assert dt2.day == 15


def test_format_iso_date():
    """Format datetime to ISO string"""
    dt = datetime(2023, 6, 15, 10, 30, 0)

    iso_str = format_iso_date(dt)

    assert "2023-06-15" in iso_str
    assert "10:30:00" in iso_str


def test_determine_file_action_created():
    """Determine if file was created (A status)"""
    # This test requires a real git repo
    # For now, we'll test the logic with mocked git commands
    pass


def test_determine_file_action_modified():
    """Determine if file was modified (M status)"""
    pass


def test_determine_file_action_deleted():
    """Determine if file was deleted (D status)"""
    pass


def test_get_file_history():
    """Get git history for a single file"""
    # This requires a real git repository
    # Will implement with test fixtures
    pass


def test_build_artifact_timeseries():
    """Build artifact timeseries from git history"""
    # This requires a real git repository with test data
    pass


def test_build_commit_aggregated():
    """Build aggregated commit statistics"""
    # This requires a real git repository
    pass


def test_analyze_artifact_history():
    """Full temporal analysis integration test"""
    # Test with sample repository
    pass


def test_author_privacy():
    """Ensure author emails are hashed, not stored plain"""
    email = "test@example.com"
    salt = "salt"
    author_hash = obfuscate_author(email, salt)

    # Hash should not contain @ or domain
    assert "@" not in author_hash
    assert "example.com" not in author_hash
    assert "test" not in author_hash


def test_obfuscate_author_empty_email():
    """Handle empty email gracefully"""
    hash_result = obfuscate_author("", "salt")
    assert len(hash_result) == 16
    assert isinstance(hash_result, str)


# =============================================================================
# Tests for anonymize_author
# =============================================================================

def test_anonymize_author_deterministic():
    """Same inputs always produce the same hash."""
    h1 = anonymize_author("user@example.com", "org", "secret")
    h2 = anonymize_author("user@example.com", "org", "secret")
    assert h1 == h2


def test_anonymize_author_prefix_and_length():
    """Output has the expected prefix and hex length."""
    result = anonymize_author("user@example.com", "org", "secret", hash_length=12, prefix="user-")
    assert result.startswith("user-")
    # 5 chars for "user-" + 12 hex chars
    assert len(result) == 5 + 12


def test_anonymize_author_normalizes_case_and_whitespace():
    """Leading/trailing whitespace and case differences produce the same hash."""
    h1 = anonymize_author("user@example.com", "org", "secret")
    h2 = anonymize_author("  User@Example.COM  ", "org", "secret")
    assert h1 == h2


def test_anonymize_author_empty_input():
    """Empty or whitespace-only input returns empty string."""
    assert anonymize_author("", "org", "secret") == ""
    assert anonymize_author("   ", "org", "secret") == ""


def test_anonymize_author_different_org_produces_different_hash():
    """Different org scopes produce different hashes for the same identifier."""
    h1 = anonymize_author("user@example.com", "org-a", "secret")
    h2 = anonymize_author("user@example.com", "org-b", "secret")
    assert h1 != h2


def test_anonymize_author_different_secret_produces_different_hash():
    """Different secrets produce different hashes."""
    h1 = anonymize_author("user@example.com", "org", "secret-1")
    h2 = anonymize_author("user@example.com", "org", "secret-2")
    assert h1 != h2


def test_anonymize_author_custom_prefix():
    """Custom prefix is applied."""
    result = anonymize_author("user@example.com", "org", "secret", prefix="dev-")
    assert result.startswith("dev-")


def test_anonymize_author_custom_hash_length():
    """Custom hash_length controls hex portion length."""
    result = anonymize_author("user@example.com", "org", "secret", hash_length=8, prefix="u-")
    assert len(result) == 2 + 8  # "u-" + 8 hex chars


def test_parse_iso_date_invalid():
    """Handle invalid date format"""
    with pytest.raises(ValueError):
        parse_iso_date("not-a-date")


def test_format_iso_date_with_timezone():
    """Format datetime with timezone info"""
    dt = datetime(2023, 6, 15, 10, 30, 0)
    iso_str = format_iso_date(dt)

    # Should be in ISO format
    assert isinstance(iso_str, str)
    assert len(iso_str) > 0


# =============================================================================
# Fixtures for collect_repo_static_metrics tests
# =============================================================================

@pytest.fixture
def temp_git_repo(tmp_path):
    """Create a temporary git repo with known state for testing."""
    repo = tmp_path / "test_repo"
    repo.mkdir()

    def _run(args, **kwargs):
        return subprocess.run(
            args, cwd=str(repo), capture_output=True, text=True, check=True,
            **kwargs
        )

    # Init repo
    _run(["git", "init"])
    _run(["git", "config", "user.email", "alice@example.com"])
    _run(["git", "config", "user.name", "Alice"])

    # First commit: 2 files
    (repo / "file1.txt").write_text("line1\nline2\nline3\n")
    (repo / "file2.py").write_text("print('hello')\n")
    _run(["git", "add", "."])
    _run(["git", "commit", "-m", "Initial commit"])

    # Tag v1.0
    _run(["git", "tag", "v1.0"])

    # Second commit by different author
    _run(["git", "config", "user.email", "bob@example.com"])
    _run(["git", "config", "user.name", "Bob"])
    (repo / "file3.md").write_text("# README\n\nHello world\n")
    _run(["git", "add", "."])
    _run(["git", "commit", "-m", "Add readme"])

    # Tag v2.0
    _run(["git", "tag", "v2.0"])

    # Third commit by Alice again
    _run(["git", "config", "user.email", "alice@example.com"])
    _run(["git", "config", "user.name", "Alice"])
    (repo / "file1.txt").write_text("line1\nline2\nline3\nline4\n")
    _run(["git", "add", "."])
    _run(["git", "commit", "-m", "Update file1"])

    return str(repo)


# =============================================================================
# Tests for collect_repo_static_metrics
# =============================================================================

class TestCollectRepoStaticMetrics:
    """Tests for collect_repo_static_metrics function."""

    def test_returns_all_expected_keys(self, temp_git_repo):
        """Result contains all expected metric keys."""
        metrics = collect_repo_static_metrics(temp_git_repo)

        expected_keys = [
            "total_commits",
            "total_authors",
            "total_files",
            "total_lines",
            "first_commit_date",
            "last_commit_date",
            "total_tags",
            "total_remote_branches",
            "commits_last_year",
        ]
        for key in expected_keys:
            assert key in metrics, f"Missing key: {key}"

    def test_total_commits(self, temp_git_repo):
        """Counts all commits correctly."""
        metrics = collect_repo_static_metrics(temp_git_repo)
        assert metrics["total_commits"] == 3

    def test_total_authors(self, temp_git_repo):
        """Counts unique authors correctly."""
        metrics = collect_repo_static_metrics(temp_git_repo)
        assert metrics["total_authors"] == 2

    def test_total_files(self, temp_git_repo):
        """Counts tracked files correctly."""
        metrics = collect_repo_static_metrics(temp_git_repo)
        assert metrics["total_files"] == 3  # file1.txt, file2.py, file3.md

    def test_total_lines(self, temp_git_repo):
        """Counts total lines across all tracked files."""
        metrics = collect_repo_static_metrics(temp_git_repo)
        # file1.txt: 4 lines, file2.py: 1 line, file3.md: 3 lines = 8
        assert metrics["total_lines"] == 8

    def test_first_commit_date(self, temp_git_repo):
        """First commit date is a valid ISO 8601 string."""
        metrics = collect_repo_static_metrics(temp_git_repo)
        assert metrics["first_commit_date"] is not None
        # Should be parseable
        parse_iso_date(metrics["first_commit_date"])

    def test_last_commit_date(self, temp_git_repo):
        """Last commit date is a valid ISO 8601 string."""
        metrics = collect_repo_static_metrics(temp_git_repo)
        assert metrics["last_commit_date"] is not None
        parse_iso_date(metrics["last_commit_date"])

    def test_first_before_last(self, temp_git_repo):
        """First commit date is before or equal to last commit date."""
        metrics = collect_repo_static_metrics(temp_git_repo)
        first = parse_iso_date(metrics["first_commit_date"])
        last = parse_iso_date(metrics["last_commit_date"])
        assert first <= last

    def test_total_tags(self, temp_git_repo):
        """Counts tags correctly."""
        metrics = collect_repo_static_metrics(temp_git_repo)
        assert metrics["total_tags"] == 2  # v1.0, v2.0

    def test_total_remote_branches_local_only(self, temp_git_repo):
        """Local-only repo has 0 remote branches."""
        metrics = collect_repo_static_metrics(temp_git_repo)
        assert metrics["total_remote_branches"] == 0

    def test_invalid_repo_path(self, tmp_path):
        """Non-repo path returns zero/None for all metrics."""
        not_a_repo = str(tmp_path / "nonexistent")
        os.makedirs(not_a_repo)
        metrics = collect_repo_static_metrics(not_a_repo)

        assert metrics["total_commits"] == 0
        assert metrics["total_authors"] == 0
        assert metrics["total_files"] == 0
        assert metrics["total_lines"] == 0
        assert metrics["first_commit_date"] is None
        assert metrics["last_commit_date"] is None
        assert metrics["total_tags"] == 0
        assert metrics["total_remote_branches"] == 0

    def test_empty_repo(self, tmp_path):
        """Repo with no commits returns zeros."""
        repo = tmp_path / "empty_repo"
        repo.mkdir()
        subprocess.run(
            ["git", "init"], cwd=str(repo),
            capture_output=True, check=True
        )
        metrics = collect_repo_static_metrics(str(repo))

        assert metrics["total_commits"] == 0
        assert metrics["total_authors"] == 0
        assert metrics["total_files"] == 0
        assert metrics["total_lines"] == 0
        assert metrics["first_commit_date"] is None
        assert metrics["last_commit_date"] is None

    def test_commits_last_year(self, temp_git_repo):
        """Commits in the last year includes recent commits."""
        metrics = collect_repo_static_metrics(temp_git_repo)
        # All 3 commits were just created, so they're within the last year
        assert metrics["commits_last_year"] == 3

    def test_integer_types(self, temp_git_repo):
        """Numeric metrics are integers, not strings."""
        metrics = collect_repo_static_metrics(temp_git_repo)
        for key in ["total_commits", "total_authors", "total_files",
                     "total_lines", "total_tags", "total_remote_branches",
                     "commits_last_year"]:
            assert isinstance(metrics[key], int), f"{key} should be int, got {type(metrics[key])}"

    def test_languages_present(self, temp_git_repo):
        """Result contains a 'languages' key."""
        metrics = collect_repo_static_metrics(temp_git_repo)
        assert "languages" in metrics

    def test_languages_is_dict(self, temp_git_repo):
        """Languages value is a dict."""
        metrics = collect_repo_static_metrics(temp_git_repo)
        assert isinstance(metrics["languages"], dict)

    def test_languages_counts(self, temp_git_repo):
        """Correct language counts for the fixture repo (.py and .md files)."""
        metrics = collect_repo_static_metrics(temp_git_repo)
        langs = metrics["languages"]
        # file2.py → Python, file3.md → Markdown, file1.txt → not mapped
        assert langs.get("Python") == 1
        assert langs.get("Markdown") == 1
        assert "txt" not in langs and "Text" not in langs
