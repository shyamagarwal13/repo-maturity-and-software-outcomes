"""
File-Level Data Collection Module

This module collects complete metadata and term frequencies for each individual artifact file.
It builds file-level metadata tables and term frequency matrices for analysis.
"""

from typing import List, Dict, Set
import os


def generate_file_id(index: int) -> str:
    """
    Generate unique file ID.

    Args:
        index: File index number

    Returns:
        String file ID in format 'file_XXX' (e.g., file_001, file_002)
    """
    return f"file_{index:03d}"


def extract_repo_name(repo_path: str) -> str:
    """
    Extract repository name from path.

    Args:
        repo_path: Path to repository (local path or git URL)

    Returns:
        Repository name string

    Examples:
        /path/to/repos/my-repo -> my-repo
        https://github.com/user/repo.git -> user/repo
    """
    # Handle GitHub URLs
    if repo_path.startswith("http"):
        # Extract user/repo from URL like https://github.com/user/repo.git
        parts = repo_path.rstrip(".git").split("/")
        if len(parts) >= 2:
            return f"{parts[-2]}/{parts[-1]}"
        return parts[-1]

    # Handle local paths
    return os.path.basename(repo_path.rstrip("/"))


def get_artifact_name(file_path: str) -> str:
    """
    Extract standardized artifact name.

    Args:
        file_path: Relative file path

    Returns:
        Artifact name string

    Examples:
        .cursorrules -> .cursorrules
        .claude/commands/test.md -> commands/test.md
        .cursor/rules/python.mdc -> rules/python.mdc
    """
    # If the file is in the root (no directory separators)
    if "/" not in file_path:
        return file_path

    # For files in subdirectories, remove the tool-specific prefix
    # e.g., .claude/commands/test.md -> commands/test.md
    parts = file_path.split("/", 1)
    if len(parts) > 1:
        return parts[1]

    return file_path


def _create_metadata_record(artifact: Dict, index: int) -> Dict:
    """
    Create a single metadata record for an artifact.

    Args:
        artifact: Artifact dictionary with metadata
        index: File index for generating file_id

    Returns:
        Metadata dictionary with all required fields
    """
    absolute_path = artifact.get("absolute_path", "")
    repo_name = extract_repo_name(absolute_path) if absolute_path else "unknown"
    file_path = artifact.get("file_path", "")

    return {
        "file_id": generate_file_id(index),
        "repo_name": repo_name,
        "tool_name": artifact.get("tool_name", "unknown"),
        "artifact_path": file_path,
        "artifact_name": get_artifact_name(file_path),
        "is_standard": artifact.get("is_standard", False),
        "word_count": artifact.get("word_count", 0),
        "unique_terms": artifact.get("unique_terms", 0),
        "file_size": artifact.get("file_size", 0),
        "is_binary": artifact.get("is_binary", False)
    }


def build_file_metadata(artifacts: List[Dict]) -> List[Dict]:
    """
    Build file-level metadata table.

    Args:
        artifacts: List of artifact dictionaries with metadata

    Returns:
        List of metadata dictionaries with fields:
            - file_id: Unique identifier (e.g., "file_001")
            - repo_name: Repository identifier
            - tool_name: Tool/IDE name
            - artifact_path: Relative file path
            - artifact_name: Standardized artifact name
            - is_standard: Boolean from JSON config
            - word_count: Total words in file
            - unique_terms: Unique terms in file
            - file_size: File size in bytes
            - is_binary: Boolean
    """
    return [_create_metadata_record(artifact, index)
            for index, artifact in enumerate(artifacts)]


def build_file_tf_matrix(artifacts: List[Dict], vocabulary: Set[str]) -> Dict:
    """
    Build term frequency matrix for all files.

    Args:
        artifacts: List of artifact dictionaries with word_frequencies
        vocabulary: Set of all unique terms across all files

    Returns:
        Dictionary with:
            - file_ids: List of file IDs
            - vocabulary: Sorted list of terms
            - matrix: List of lists with term frequencies (rows=files, cols=terms)
    """
    # Sort vocabulary for consistent column ordering
    sorted_vocab = sorted(vocabulary)

    # Build file IDs
    file_ids = [generate_file_id(i) for i in range(len(artifacts))]

    # Build term frequency matrix
    matrix = []
    for artifact in artifacts:
        word_frequencies = artifact.get("word_frequencies", {})

        # Create row with frequency for each term in vocabulary
        # Use 0 for terms not present in this file
        row = [word_frequencies.get(term, 0) for term in sorted_vocab]
        matrix.append(row)

    return {
        "file_ids": file_ids,
        "vocabulary": sorted_vocab,
        "matrix": matrix
    }
