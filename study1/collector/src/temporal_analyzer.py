"""
Temporal analyzer module for tracking AI artifact evolution in git repositories.

This module provides functions to analyze git history and track when AI artifacts
were introduced and how they evolved over time.
"""

import subprocess
import hashlib
import hmac
from datetime import datetime
from typing import Callable, List, Dict, Optional
import re
from pathlib import Path


_EXT_TO_LANGUAGE = {
    ".py": "Python",
    ".js": "JavaScript",
    ".ts": "TypeScript",
    ".tsx": "TypeScript",
    ".jsx": "JavaScript",
    ".java": "Java",
    ".go": "Go",
    ".rs": "Rust",
    ".rb": "Ruby",
    ".c": "C",
    ".h": "C",
    ".cpp": "C++",
    ".cxx": "C++",
    ".cc": "C++",
    ".hpp": "C++",
    ".cs": "C#",
    ".swift": "Swift",
    ".kt": "Kotlin",
    ".kts": "Kotlin",
    ".md": "Markdown",
    ".mdc": "Markdown",
    ".json": "JSON",
    ".yaml": "YAML",
    ".yml": "YAML",
    ".html": "HTML",
    ".htm": "HTML",
    ".css": "CSS",
    ".scss": "CSS",
    ".sh": "Shell",
    ".bash": "Shell",
    ".zsh": "Shell",
    ".sql": "SQL",
    ".r": "R",
    ".R": "R",
    ".m": "Objective-C",
    ".lua": "Lua",
    ".dart": "Dart",
    ".ex": "Elixir",
    ".exs": "Elixir",
    ".hs": "Haskell",
    ".scala": "Scala",
    ".php": "PHP",
    ".pl": "Perl",
    ".pm": "Perl",
    ".xml": "XML",
    ".toml": "TOML",
    ".vue": "Vue",
    ".svelte": "Svelte",
}


def obfuscate_author(email: str, salt: str) -> str:
    """
    Hash author email for privacy protection.

    Args:
        email: Author email address
        salt: Salt string for hashing

    Returns:
        First 16 characters of SHA256 hash

    Example:
        >>> obfuscate_author("user@example.com", "my-salt")
        'a1b2c3d4e5f6g7h8'
    """
    hash_input = email + salt
    hash_output = hashlib.sha256(hash_input.encode()).hexdigest()
    return hash_output[:16]


def anonymize_author(
    identifier: str,
    org: str,
    secret: str,
    hash_length: int = 12,
    prefix: str = "user-",
) -> str:
    """
    HMAC-based author anonymization compatible with the reference PHP implementation.

    Produces a deterministic, non-reversible pseudonym from an author identifier
    (email or name) using HMAC-SHA256 keyed by *secret* and scoped to *org*.

    Args:
        identifier: Author email or name
        org: Organisation name used to scope the hash
        secret: HMAC secret key
        hash_length: Number of hex characters to keep (default 12)
        prefix: String prepended to the hash (default "user-")

    Returns:
        Anonymized string, e.g. ``"user-a1b2c3d4e5f6"``; empty string for
        blank identifiers.

    Example:
        >>> anonymize_author("user@example.com", "myorg", "s3cret")
        'user-...'
    """
    if not identifier.strip():
        return ""
    message = f"{org}:{identifier.strip().lower()}"
    hash_output = hmac.new(
        secret.encode(), message.encode(), hashlib.sha256
    ).hexdigest()
    return f"{prefix}{hash_output[:hash_length]}"


def parse_iso_date(date_str: str) -> datetime:
    """
    Parse ISO 8601 date string to datetime object.

    Args:
        date_str: ISO 8601 formatted date string

    Returns:
        datetime object

    Raises:
        ValueError: If date string format is invalid

    Example:
        >>> parse_iso_date("2023-06-15T10:30:00Z")
        datetime.datetime(2023, 6, 15, 10, 30)
    """
    # Try common ISO formats
    formats = [
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%d",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue

    # If none of the formats work, raise error
    raise ValueError(f"Unable to parse date string: {date_str}")


def format_iso_date(dt: datetime) -> str:
    """
    Format datetime object to ISO 8601 string.

    Args:
        dt: datetime object

    Returns:
        ISO 8601 formatted string

    Example:
        >>> dt = datetime(2023, 6, 15, 10, 30, 0)
        >>> format_iso_date(dt)
        '2023-06-15T10:30:00Z'
    """
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _fetch_all_commits_with_status(
    repo_path: str,
    start_date: str,
    end_date: str,
) -> List[Dict]:
    """
    Fetch all commits in a date range with per-file name-status info in a single git call.

    Returns a list of dicts, each with keys:
        commit_sha, author_name, author_email, commit_date, files

    where ``files`` is a list of (status, path) tuples.
    ``status`` is a single letter: A (added), M (modified), D (deleted), etc.
    Renames (R*) are mapped to the *new* path with status "M".
    """
    cmd = [
        "git", "log",
        "--name-status",
        "--first-parent",
        "--pretty=format:COMMIT_START%H|%an|%ae|%ad",
        "--date=iso",
        f"--after={start_date}",
        f"--before={end_date}",
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError:
        return []

    commits: List[Dict] = []
    current: Optional[Dict] = None

    for line in result.stdout.split("\n"):
        if line.startswith("COMMIT_START"):
            if current is not None:
                commits.append(current)
            header = line[len("COMMIT_START"):]
            parts = header.split("|")
            if len(parts) >= 4:
                current = {
                    "commit_sha": parts[0],
                    "author_name": parts[1],
                    "author_email": parts[2],
                    "commit_date": parts[3],
                    "files": [],
                }
            else:
                current = None
        elif line.strip() and current is not None:
            # name-status line: tab-separated  STATUS\tPATH  (or STATUS\tOLD\tNEW for renames)
            cols = line.split("\t")
            if len(cols) >= 2:
                status = cols[0].strip()
                if status.startswith("R"):
                    # Rename: use new path, treat as modified
                    path = cols[-1]
                    current["files"].append(("M", path))
                else:
                    path = cols[1]
                    current["files"].append((status[0], path))

    if current is not None:
        commits.append(current)

    return commits


def _build_artifact_lookup(
    artifacts: List[Dict],
) -> tuple:
    """
    Build O(1) lookup structures from the artifacts list.

    Returns:
        (artifact_paths: Set[str], path_to_type: Dict[str, str])
    """
    artifact_paths: set = set()
    path_to_type: Dict[str, str] = {}

    for artifact in artifacts:
        file_path = artifact.get("path", artifact.get("artifact_path", ""))
        artifact_type = artifact.get("type", artifact.get("artifact_type", "unknown"))
        artifact_paths.add(file_path)
        path_to_type[file_path] = artifact_type

    return artifact_paths, path_to_type


def determine_file_action(repo_path: str, commit_sha: str, file_path: str) -> str:
    """
    Determine if file was created, modified, or deleted in a commit.

    Args:
        repo_path: Path to git repository
        commit_sha: Commit hash
        file_path: Relative path to file

    Returns:
        One of: "created", "modified", "deleted"

    Example:
        >>> determine_file_action("/path/to/repo", "a1b2c3d", ".cursorrules")
        'created'
    """
    cmd = [
        "git", "diff-tree", "--no-commit-id", "--name-status", "-r",
        commit_sha, "--", file_path
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True
        )

        output = result.stdout.strip()
        if not output:
            return "modified"  # Default if no status found

        # Parse status (first character)
        status = output.split()[0]

        if status == "A":
            return "created"
        elif status == "M":
            return "modified"
        elif status == "D":
            return "deleted"
        else:
            return "modified"

    except subprocess.CalledProcessError:
        return "modified"


def get_file_history(
    repo_path: str,
    file_path: str,
    start_date: str,
    end_date: str
) -> List[Dict]:
    """
    Get git history for a single file.

    Args:
        repo_path: Path to git repository
        file_path: Relative path to file
        start_date: Start date (ISO 8601)
        end_date: End date (ISO 8601)

    Returns:
        List of commit information dictionaries

    Example:
        >>> get_file_history("/repo", ".cursorrules", "2023-01-01", "2023-12-31")
        [{'commit_sha': 'a1b2c3d', 'author_name': 'John', ...}]
    """
    cmd = [
        "git", "log",
        "--follow",
        "--pretty=format:%H|%an|%ae|%ad|%s",
        "--date=iso",
        f"--after={start_date}",
        f"--before={end_date}",
        "--",
        file_path
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True
        )

        history = []
        for line in result.stdout.strip().split('\n'):
            if not line:
                continue

            parts = line.split('|')
            if len(parts) >= 5:
                history.append({
                    "commit_sha": parts[0][:7],  # Abbreviated
                    "author_name": parts[1],
                    "author_email": parts[2],
                    "commit_date": parts[3],
                    "message": parts[4] if len(parts) > 4 else ""
                })

        return history

    except subprocess.CalledProcessError:
        return []


def build_artifact_timeseries(
    repo_path: str,
    artifacts: List[Dict],
    start_date: str,
    end_date: str,
    hash_fn: Callable[[str], str],
) -> List[Dict]:
    """
    Build artifact timeseries table from git history.

    Args:
        repo_path: Path to git repository
        artifacts: List of artifact dictionaries with 'path' and 'type' keys
        start_date: Start date (ISO 8601)
        end_date: End date (ISO 8601)
        hash_fn: Callable that hashes an author identifier string

    Returns:
        List of timeseries records

    Example:
        >>> artifacts = [{'path': '.cursorrules', 'type': 'ide_configs'}]
        >>> build_artifact_timeseries("/repo", artifacts, "2023-01-01", "2023-12-31", lambda x: x[:16])
        [{'commit_sha': 'a1b2c3d', 'commit_date': '2023-06-15T10:30:00Z', ...}]
    """
    if not artifacts:
        return []

    # Single git subprocess call for all commits in the date range
    print("  Artifact history: fetching commits...", end="", flush=True)
    all_commits = _fetch_all_commits_with_status(repo_path, start_date, end_date)

    # Build O(1) lookup from artifact list
    artifact_paths, path_to_type = _build_artifact_lookup(artifacts)

    status_map = {"A": "created", "M": "modified", "D": "deleted"}

    timeseries = []
    matches = 0

    print(
        f"\r  Artifact history: processing {len(all_commits)} commits "
        f"for {len(artifact_paths)} artifacts...",
        end="",
        flush=True,
    )

    for commit in all_commits:
        author_hash = hash_fn(commit["author_email"])
        author_name_hash = hash_fn(commit["author_name"])
        short_sha = commit["commit_sha"][:7]

        # Parse and format date once per commit
        try:
            commit_dt = parse_iso_date(commit["commit_date"])
            commit_date_iso = format_iso_date(commit_dt)
        except ValueError:
            commit_date_iso = commit["commit_date"]

        for status, path in commit["files"]:
            if path in artifact_paths:
                matches += 1
                timeseries.append({
                    "commit_sha": short_sha,
                    "commit_date": commit_date_iso,
                    "artifact_path": path,
                    "artifact_type": path_to_type[path],
                    "action": status_map.get(status, "modified"),
                    "author_hash": author_hash,
                    "author_name_hash": author_name_hash,
                })

    print(f" {matches} matches")
    return timeseries


def build_commit_aggregated(
    repo_path: str,
    start_date: str,
    end_date: str,
    hash_fn: Callable[[str], str],
) -> List[Dict]:
    """
    Build aggregated commit statistics for repository.

    Args:
        repo_path: Path to git repository
        start_date: Start date (ISO 8601)
        end_date: End date (ISO 8601)
        hash_fn: Callable that hashes an author identifier string

    Returns:
        List of commit statistics records

    Example:
        >>> build_commit_aggregated("/repo", "2023-01-01", "2023-12-31", lambda x: x[:16])
        [{'commit_date': '2023-06-15T10:30:00Z', 'files_modified': 3, ...}]
    """
    cmd = [
        "git", "log",
        "--pretty=format:%H|%an|%ae|%ad",
        "--numstat",
        "--date=iso",
        f"--after={start_date}",
        f"--before={end_date}"
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True
        )

        commits = []
        current_commit = None

        for line in result.stdout.split('\n'):
            if not line.strip():
                continue

            # Check if this is a commit line (contains |)
            if '|' in line:
                # Save previous commit if exists
                if current_commit:
                    commits.append(current_commit)

                # Start new commit
                parts = line.split('|')
                if len(parts) >= 4:
                    author_hash = hash_fn(parts[2])
                    author_name_hash = hash_fn(parts[1])

                    # Parse date
                    try:
                        commit_dt = parse_iso_date(parts[3])
                        commit_date_iso = format_iso_date(commit_dt)
                    except ValueError:
                        commit_date_iso = parts[3]

                    current_commit = {
                        "commit_date": commit_date_iso,
                        "commit_sha": parts[0][:7],
                        "author_hash": author_hash,
                        "author_name_hash": author_name_hash,
                        "files_modified": 0,
                        "files_added": 0,
                        "files_deleted": 0,
                        "total_additions": 0,
                        "total_deletions": 0
                    }
            else:
                # This is a numstat line
                if current_commit:
                    parts = line.split()
                    if len(parts) >= 3:
                        additions = parts[0]
                        deletions = parts[1]

                        # Handle binary files (marked with -)
                        if additions != '-':
                            current_commit["total_additions"] += int(additions)
                        if deletions != '-':
                            current_commit["total_deletions"] += int(deletions)

                        # Count file modifications
                        if additions == '0' and deletions != '0':
                            current_commit["files_deleted"] += 1
                        elif additions != '0' and deletions == '0':
                            current_commit["files_added"] += 1
                        else:
                            current_commit["files_modified"] += 1

        # Add last commit
        if current_commit:
            commits.append(current_commit)

        return commits

    except subprocess.CalledProcessError:
        return []


def collect_repo_static_metrics(repo_path: str, timeout: int = 60) -> Dict[str, Optional[str]]:
    """
    Collect static repository-level metrics using git commands.

    Args:
        repo_path: Path to git repository
        timeout: Timeout in seconds for individual git commands

    Returns:
        Dictionary with:
        - total_commits: int
        - total_authors: int
        - total_files: int
        - total_lines: int
        - first_commit_date: ISO 8601 string or None
        - last_commit_date: ISO 8601 string or None
        - total_tags: int
        - total_remote_branches: int
        - commits_last_year: int
    """
    metrics: Dict[str, Optional[str]] = {}

    def _run_git(args: List[str]) -> str:
        """Run a git command and return stripped stdout, or empty string on failure."""
        try:
            result = subprocess.run(
                ["git"] + args,
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=True,
                timeout=timeout
            )
            return result.stdout.strip()
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return ""

    # Total commits
    out = _run_git(["rev-list", "--count", "HEAD"])
    metrics["total_commits"] = int(out) if out.isdigit() else 0

    # Total authors (unique emails)
    out = _run_git(["shortlog", "-sne", "HEAD"])
    if out:
        metrics["total_authors"] = len(out.splitlines())
    else:
        metrics["total_authors"] = 0

    # Total tracked files
    out = _run_git(["ls-files"])
    if out:
        file_list = out.splitlines()
        metrics["total_files"] = len(file_list)
    else:
        file_list = []
        metrics["total_files"] = 0

    # Language breakdown by file extension
    lang_counts: Dict[str, int] = {}
    for f in file_list:
        ext = Path(f).suffix.lower()
        # Special case: .R must stay uppercase for the lookup
        if ext == ".r":
            ext = Path(f).suffix
        lang = _EXT_TO_LANGUAGE.get(ext)
        if lang:
            lang_counts[lang] = lang_counts.get(lang, 0) + 1
    metrics["languages"] = dict(sorted(lang_counts.items(), key=lambda x: x[1], reverse=True))

    # Total lines across all tracked files (binary files are skipped by wc)
    try:
        ls_result = subprocess.run(
            ["git", "ls-files"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True,
            timeout=timeout
        )
        if ls_result.stdout.strip():
            wc_result = subprocess.run(
                ["xargs", "wc", "-l"],
                cwd=repo_path,
                input=ls_result.stdout,
                capture_output=True,
                text=True,
                timeout=timeout * 2
            )
            # Last line of wc -l output is "  12345 total"
            lines = wc_result.stdout.strip().splitlines()
            if lines:
                last_line = lines[-1].strip()
                # Single file: no "total" line, just "<count> <filename>"
                if "total" in last_line:
                    parts = last_line.split()
                    metrics["total_lines"] = int(parts[0]) if parts[0].isdigit() else 0
                elif len(lines) == 1:
                    parts = last_line.split()
                    metrics["total_lines"] = int(parts[0]) if parts[0].isdigit() else 0
                else:
                    metrics["total_lines"] = 0
            else:
                metrics["total_lines"] = 0
        else:
            metrics["total_lines"] = 0
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        metrics["total_lines"] = 0

    # First commit date
    out = _run_git(["log", "--reverse", "--format=%aI", "--max-count=1"])
    metrics["first_commit_date"] = out if out else None

    # Last commit date
    out = _run_git(["log", "--format=%aI", "--max-count=1"])
    metrics["last_commit_date"] = out if out else None

    # Total tags
    out = _run_git(["tag"])
    if out:
        metrics["total_tags"] = len(out.splitlines())
    else:
        metrics["total_tags"] = 0

    # Total remote branches
    out = _run_git(["branch", "-r"])
    if out:
        # Filter out HEAD pointer line (e.g. "origin/HEAD -> origin/main")
        branches = [l.strip() for l in out.splitlines() if "->" not in l]
        metrics["total_remote_branches"] = len(branches)
    else:
        metrics["total_remote_branches"] = 0

    # Commits in the last year (relative to last commit date, not wall clock)
    if metrics.get("last_commit_date"):
        out = _run_git(["rev-list", "--count", "--since=1.year.ago", "HEAD"])
        metrics["commits_last_year"] = int(out) if out.isdigit() else 0
    else:
        metrics["commits_last_year"] = 0

    return metrics


def analyze_artifact_history(
    repo_path: str,
    artifacts: List[Dict],
    start_date: str,
    end_date: str,
    hash_fn: Callable[[str], str],
) -> Dict:
    """
    Analyze git history for all artifact files.

    This is the main entry point for temporal analysis.

    Args:
        repo_path: Path to git repository
        artifacts: List of artifact dictionaries
        start_date: Start date (ISO 8601)
        end_date: End date (ISO 8601)
        hash_fn: Callable that hashes an author identifier string

    Returns:
        Dictionary with 'artifact_timeseries' and 'commit_aggregated' keys

    Example:
        >>> artifacts = [{'path': '.cursorrules', 'type': 'ide_configs'}]
        >>> result = analyze_artifact_history("/repo", artifacts, "2023-01-01", "2023-12-31", lambda x: x[:16])
        >>> 'artifact_timeseries' in result
        True
    """
    artifact_timeseries = build_artifact_timeseries(
        repo_path, artifacts, start_date, end_date, hash_fn
    )

    commit_aggregated = build_commit_aggregated(
        repo_path, start_date, end_date, hash_fn
    )

    return {
        "artifact_timeseries": artifact_timeseries,
        "commit_aggregated": commit_aggregated
    }
