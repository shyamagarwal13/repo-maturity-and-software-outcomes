"""
File Discovery Module

Discovers AI artifact files in cloned repositories using patterns from JSON configurations.
Supports three discovery methods: exact_path, glob, and regex.

This module implements Phase 3 of the AI Artifact Data Collection pipeline.
It scans repositories for artifact files using various discovery methods defined
in the tool registry.

Example:
    >>> from artifact_config_loader import load_json_configs
    >>> from file_discovery import discover_artifacts
    >>>
    >>> tool_registry = load_json_configs("Artifacts/")
    >>> artifacts = discover_artifacts("/path/to/repo", tool_registry)
    >>> for artifact in artifacts:
    ...     print(f"{artifact['tool_name']}: {artifact['file_path']}")
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Any, Set, Optional, Union


# Excluded root files - common documentation that's not AI artifact related
EXCLUDED_ROOT_FILES = {
    "CHANGELOG.md",
    "LICENSE.md",
    "CONTRIBUTING.md",
    "CODE_OF_CONDUCT.md",
    "SECURITY.md",
    "HISTORY.md",
}

# Excluded directories for non-standard file discovery
# These are typically generated, vendor, or build directories
EXCLUDED_DIRECTORIES = {
    ".git",
    "node_modules",
    "vendor",
    "dist",
    "build",
    "__pycache__",
    ".venv",
    "venv",
    ".tox",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "target",        # Rust
    "bin",
    "obj",           # .NET
    "packages",      # NuGet
}


class DiscoveryContext:
    """
    Tracks discovered files to prevent duplicates across discovery steps.

    This class maintains a set of absolute paths that have been discovered
    during the artifact discovery process. It allows checking if a file
    has already been discovered and marking new files as discovered.

    Discovery step values:
        - "tool_standard" - Step 1: Tool-specific standard patterns
        - "shared_in_tool_folder" - Step 2a: Shared patterns in tool folders
        - "shared_in_root" - Step 2b: Shared patterns in repository root
        - "non_standard_root" - Step 3a: Non-standard files in root
        - "non_standard_other" - Step 3b: Non-standard files in other locations

    Example:
        >>> context = DiscoveryContext()
        >>> context.is_discovered("/path/to/file.txt")
        False
        >>> context.mark_discovered("/path/to/file.txt")
        >>> context.is_discovered("/path/to/file.txt")
        True
        >>> context.discovered_count()
        1
    """

    def __init__(self):
        """Initialize an empty discovery context."""
        self._discovered: Set[str] = set()

    def is_discovered(self, absolute_path: str) -> bool:
        """
        Check if a file has already been discovered.

        Args:
            absolute_path: Full filesystem path to check

        Returns:
            True if the file was already discovered, False otherwise
        """
        return absolute_path in self._discovered

    def mark_discovered(self, absolute_path: str) -> None:
        """
        Mark a file as discovered.

        Args:
            absolute_path: Full filesystem path to mark as discovered
        """
        self._discovered.add(absolute_path)

    def discovered_count(self) -> int:
        """
        Get the number of discovered files.

        Returns:
            Count of unique files discovered so far
        """
        return len(self._discovered)


def discover_exact_path(repo_path: str, pattern: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Find files matching exact path.

    This discovery method checks if a specific file exists at an exact path
    relative to the repository root. This is the most efficient method when
    you know the exact location of an artifact file.

    Args:
        repo_path: Absolute path to the repository root directory
        pattern: Pattern dictionary containing:
            - exact_path (str): Relative path to the file from repo root
            - is_standard (bool): Whether this is a standard artifact pattern
            - discovery_method (str): Should be "exact_path"

    Returns:
        List containing a single file info dictionary if found, empty list otherwise.
        Each dictionary contains:
            - file_path (str): Relative path from repo root
            - absolute_path (str): Full filesystem path
            - is_standard (bool): Whether this is a standard artifact
            - discovery_method (str): "exact_path"

    Example:
        >>> pattern = {
        ...     "exact_path": ".cursorrules",
        ...     "is_standard": True,
        ...     "discovery_method": "exact_path"
        ... }
        >>> results = discover_exact_path("/path/to/repo", pattern)
        >>> if results:
        ...     print(f"Found: {results[0]['file_path']}")
    """
    exact_path = pattern.get("exact_path")
    is_standard = pattern.get("is_standard", True)

    # Build absolute path
    absolute_path = Path(repo_path) / exact_path

    # Check if file exists
    if absolute_path.exists():
        return [{
            "file_path": exact_path,
            "absolute_path": str(absolute_path),
            "is_standard": is_standard,
            "discovery_method": "exact_path"
        }]

    return []


def discover_glob(repo_path: str, pattern: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Find files matching glob pattern.

    This discovery method uses glob patterns to find files matching wildcards.
    Supports both simple patterns (*.yml) and recursive patterns (**/*.md).

    Args:
        repo_path: Absolute path to the repository root directory
        pattern: Pattern dictionary containing:
            - glob_pattern (str): Glob pattern to match (e.g., ".cursor/**/*.mdc")
            - is_standard (bool): Whether this is a standard artifact pattern
            - discovery_method (str): Should be "glob"
            - recursive (bool, optional): Whether pattern uses recursive wildcards

    Returns:
        List of file info dictionaries for all matching files.
        Each dictionary contains:
            - file_path (str): Relative path from repo root
            - absolute_path (str): Full filesystem path
            - is_standard (bool): Whether this is a standard artifact
            - discovery_method (str): "glob"

    Example:
        >>> pattern = {
        ...     "glob_pattern": ".cursor/**/*.mdc",
        ...     "is_standard": True,
        ...     "discovery_method": "glob",
        ...     "recursive": True
        ... }
        >>> results = discover_glob("/path/to/repo", pattern)
        >>> for result in results:
        ...     print(result['file_path'])
    """
    glob_pattern = pattern.get("glob_pattern")
    is_standard = pattern.get("is_standard", True)

    # Use pathlib for glob matching
    repo = Path(repo_path)
    results = []

    # Use glob to find matching files
    for file_path in repo.glob(glob_pattern):
        if file_path.is_file():
            # Get relative path from repo
            relative_path = file_path.relative_to(repo)
            results.append({
                "file_path": str(relative_path),
                "absolute_path": str(file_path),
                "is_standard": is_standard,
                "discovery_method": "glob"
            })

    return results


def discover_regex(repo_path: str, pattern: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Find files matching regex pattern.

    This discovery method walks the entire repository tree and matches filenames
    against a regular expression pattern. This is the most flexible but also
    the slowest discovery method.

    Args:
        repo_path: Absolute path to the repository root directory
        pattern: Pattern dictionary containing:
            - regex_pattern (str): Regular expression to match filenames
            - is_standard (bool): Whether this is a standard artifact pattern
            - discovery_method (str): Should be "regex"

    Returns:
        List of file info dictionaries for all matching files.
        Each dictionary contains:
            - file_path (str): Relative path from repo root
            - absolute_path (str): Full filesystem path
            - is_standard (bool): Whether this is a standard artifact
            - discovery_method (str): "regex"

    Note:
        - The .git directory is automatically skipped during traversal
        - The regex is matched against the filename only, not the full path

    Example:
        >>> pattern = {
        ...     "regex_pattern": r"^\\.aider.*",
        ...     "is_standard": False,
        ...     "discovery_method": "regex"
        ... }
        >>> results = discover_regex("/path/to/repo", pattern)
        >>> for result in results:
        ...     print(result['file_path'])
    """
    regex_pattern = pattern.get("regex_pattern")
    is_standard = pattern.get("is_standard", False)

    # Compile regex
    regex = re.compile(regex_pattern)

    repo = Path(repo_path)
    results = []

    # Walk through repository files
    for root, dirs, files in os.walk(repo):
        # Skip .git directory
        if '.git' in dirs:
            dirs.remove('.git')

        for filename in files:
            # Check if filename matches regex
            if regex.match(filename):
                file_path = Path(root) / filename
                relative_path = file_path.relative_to(repo)
                results.append({
                    "file_path": str(relative_path),
                    "absolute_path": str(file_path),
                    "is_standard": is_standard,
                    "discovery_method": "regex"
                })

    return results


def find_non_standard_files(repo_path: str, tool_name: str, tool_dir: str) -> List[Dict[str, Any]]:
    """
    Find non-standard files in tool directories.

    This function discovers all files within a tool's directory that may not
    be captured by standard patterns. Useful for finding custom configurations,
    user-defined commands, or other non-standard artifacts.

    Args:
        repo_path: Absolute path to the repository root directory
        tool_name: Name of the tool (e.g., "claude-code", "cursor")
        tool_dir: Tool directory path relative to repo (e.g., ".claude", ".cursor")

    Returns:
        List of file info dictionaries for all files in the tool directory.
        Each dictionary contains:
            - file_path (str): Relative path from repo root
            - absolute_path (str): Full filesystem path
            - tool_name (str): Name of the tool
            - artifact_name (str): Filename only
            - is_standard (bool): Always False for non-standard files
            - discovery_method (str): "directory_walk"

    Note:
        - Returns empty list if tool directory doesn't exist
        - The .git directory is automatically skipped during traversal

    Example:
        >>> files = find_non_standard_files(
        ...     "/path/to/repo",
        ...     "claude-code",
        ...     ".claude"
        ... )
        >>> for file in files:
        ...     print(f"Non-standard: {file['file_path']}")
    """
    tool_path = Path(repo_path) / tool_dir

    # Check if tool directory exists
    if not tool_path.exists():
        return []

    results = []

    # Walk through all files in tool directory
    for root, dirs, files in os.walk(tool_path):
        # Skip .git directory
        if '.git' in dirs:
            dirs.remove('.git')

        for filename in files:
            file_path = Path(root) / filename
            relative_path = file_path.relative_to(repo_path)
            results.append({
                "file_path": str(relative_path),
                "absolute_path": str(file_path),
                "tool_name": tool_name,
                "artifact_name": filename,
                "is_standard": False,
                "discovery_method": "directory_walk"
            })

    return results


def deduplicate_artifacts(artifacts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove duplicate files found by multiple patterns.

    When the same file is discovered by multiple patterns (e.g., both a standard
    pattern and a non-standard directory walk), this function ensures each file
    appears only once in the results. When duplicates exist, it prefers the
    version marked as is_standard=True.

    Args:
        artifacts: List of artifact dictionaries, potentially containing duplicates

    Returns:
        List of unique artifacts with duplicates removed.
        When duplicates exist, the standard version is preferred.

    Note:
        Duplicates are identified by matching absolute_path values.

    Example:
        >>> artifacts = [
        ...     {"absolute_path": "/repo/.cursorrules", "is_standard": True},
        ...     {"absolute_path": "/repo/.cursorrules", "is_standard": False},
        ...     {"absolute_path": "/repo/.aider.conf.yml", "is_standard": True}
        ... ]
        >>> unique = deduplicate_artifacts(artifacts)
        >>> len(unique)
        2
        >>> unique[0]["is_standard"]  # .cursorrules entry
        True
    """
    # Create dict keyed by absolute_path
    unique_artifacts = {}

    for artifact in artifacts:
        abs_path = artifact["absolute_path"]

        if abs_path not in unique_artifacts:
            # First time seeing this path
            unique_artifacts[abs_path] = artifact
        else:
            # Duplicate found - prefer standard version
            existing = unique_artifacts[abs_path]
            if artifact.get("is_standard", False) and not existing.get("is_standard", False):
                unique_artifacts[abs_path] = artifact

    return list(unique_artifacts.values())


def discover_shared_in_config_folders(
    repo_path: Path,
    tool_configs: Dict[str, Any],
    shared_config: Any,
    context: DiscoveryContext
) -> List[Dict[str, Any]]:
    """
    Step 2a: Discover shared artifacts in tool config folders.

    Searches for shared patterns (AGENTS.md, mcp.json, etc.) inside each
    tool's config_folders. Found artifacts are attributed to the tool that
    owns the folder.

    Args:
        repo_path: Path object pointing to the repository root directory
        tool_configs: Dictionary of ToolConfig objects keyed by tool name
        shared_config: ToolConfig containing shared artifact patterns
        context: DiscoveryContext for tracking discovered files

    Returns:
        List of discovered artifact dictionaries with:
            - file_path (str): Relative path from repo root
            - absolute_path (str): Full filesystem path
            - tool_name (str): Name of the tool that owns the config folder
            - artifact_name (str): Filename only
            - is_standard (bool): Whether this is a standard artifact
            - discovery_method (str): Method used ("exact_path", "glob")
            - discovery_step (str): Always "shared_in_tool_folder"

    Example:
        >>> context = DiscoveryContext()
        >>> artifacts = discover_shared_in_config_folders(
        ...     Path("/repo"), tool_configs, shared_config, context
        ... )
        >>> for a in artifacts:
        ...     print(f"{a['tool_name']}: {a['file_path']}")
    """
    # Ensure repo_path is a Path object
    if isinstance(repo_path, str):
        repo_path = Path(repo_path)

    artifacts = []

    for tool_name, tool_config in tool_configs.items():
        for config_folder in tool_config.config_folders:
            # Remove trailing slash if present for consistency
            config_folder = config_folder.rstrip('/')
            folder_path = repo_path / config_folder

            if not folder_path.exists():
                continue

            # Search for each shared pattern in this config folder
            for pattern in shared_config.artifact_patterns:
                discovery_method = pattern.discovery_method.value

                if discovery_method == "exact_path":
                    # Check for exact file in config folder
                    exact_path = pattern.exact_path
                    file_path = folder_path / exact_path
                    if file_path.exists() and file_path.is_file():
                        abs_path = str(file_path.resolve())
                        if not context.is_discovered(abs_path):
                            context.mark_discovered(abs_path)
                            relative_path = file_path.relative_to(repo_path)
                            artifacts.append({
                                "file_path": str(relative_path),
                                "absolute_path": abs_path,
                                "tool_name": tool_name,
                                "artifact_name": file_path.name,
                                "is_standard": pattern.is_standard,
                                "discovery_method": "exact_path",
                                "discovery_step": "shared_in_tool_folder"
                            })

                elif discovery_method == "glob":
                    # Apply glob pattern within config folder
                    glob_pattern = pattern.glob_pattern
                    # For mcp patterns like **/*mcp*.json, search within folder
                    for file_path in folder_path.glob(glob_pattern):
                        if file_path.is_file():
                            abs_path = str(file_path.resolve())
                            if not context.is_discovered(abs_path):
                                context.mark_discovered(abs_path)
                                relative_path = file_path.relative_to(repo_path)
                                artifacts.append({
                                    "file_path": str(relative_path),
                                    "absolute_path": abs_path,
                                    "tool_name": tool_name,
                                    "artifact_name": file_path.name,
                                    "is_standard": pattern.is_standard,
                                    "discovery_method": "glob",
                                    "discovery_step": "shared_in_tool_folder"
                                })

    return artifacts


def discover_shared_in_root(
    repo_path: Path,
    shared_config: Any,
    context: DiscoveryContext
) -> List[Dict[str, Any]]:
    """
    Step 2b: Discover shared artifacts in project root.

    Searches for shared patterns (AGENTS.md, .mcp.json, etc.) in the repository
    root only (not subdirectories). Found artifacts are tagged with tool_name="shared".

    Args:
        repo_path: Path object pointing to the repository root directory
        shared_config: ToolConfig containing shared artifact patterns
        context: DiscoveryContext for tracking discovered files

    Returns:
        List of discovered artifact dictionaries with:
            - file_path (str): Relative path from repo root
            - absolute_path (str): Full filesystem path
            - tool_name (str): Always "shared"
            - artifact_name (str): Filename only
            - is_standard (bool): Whether this is a standard artifact
            - discovery_method (str): Method used ("exact_path", "glob")
            - discovery_step (str): Always "shared_in_root"

    Note:
        - Only searches in the repository root, not subdirectories
        - Skips generic fallback patterns (like "**/*.md")
        - Uses DiscoveryContext to prevent duplicates from earlier steps

    Example:
        >>> context = DiscoveryContext()
        >>> artifacts = discover_shared_in_root(
        ...     Path("/repo"), shared_config, context
        ... )
        >>> for a in artifacts:
        ...     print(f"{a['tool_name']}: {a['file_path']}")
    """
    # Ensure repo_path is a Path object
    if isinstance(repo_path, str):
        repo_path = Path(repo_path)

    artifacts = []

    for pattern in shared_config.artifact_patterns:
        discovery_method = pattern.discovery_method.value

        # Skip generic fallback patterns (like **/*.md that matches all markdown)
        # But allow patterns like **/*mcp*.json which target specific files
        if discovery_method == "glob":
            glob_pattern = pattern.glob_pattern
            # Skip if it's a generic catch-all pattern (e.g., **/*.md)
            if "**" in glob_pattern and glob_pattern.count("*") == 3:
                # Pattern like **/*.md - skip as it's a fallback
                continue

        if discovery_method == "exact_path":
            # Check for exact file in root
            exact_path = pattern.exact_path
            file_path = repo_path / exact_path
            if file_path.exists() and file_path.is_file():
                abs_path = str(file_path.resolve())
                if not context.is_discovered(abs_path):
                    context.mark_discovered(abs_path)
                    artifacts.append({
                        "file_path": exact_path,
                        "absolute_path": abs_path,
                        "tool_name": "shared",
                        "artifact_name": file_path.name,
                        "is_standard": pattern.is_standard,
                        "discovery_method": "exact_path",
                        "discovery_step": "shared_in_root"
                    })

        elif discovery_method == "glob":
            # Apply glob pattern in root only (non-recursive)
            glob_pattern = pattern.glob_pattern
            # Remove ** for recursive patterns, only match in root
            root_pattern = glob_pattern.replace("**/", "").replace("/**", "")
            for file_path in repo_path.glob(root_pattern):
                # Only include files directly in root (no subdirectories)
                if file_path.is_file() and file_path.parent == repo_path:
                    abs_path = str(file_path.resolve())
                    if not context.is_discovered(abs_path):
                        context.mark_discovered(abs_path)
                        relative_path = file_path.relative_to(repo_path)
                        artifacts.append({
                            "file_path": str(relative_path),
                            "absolute_path": abs_path,
                            "tool_name": "shared",
                            "artifact_name": file_path.name,
                            "is_standard": pattern.is_standard,
                            "discovery_method": "glob",
                            "discovery_step": "shared_in_root"
                        })

    return artifacts


def discover_non_standard_root(
    repo_path: Path,
    context: DiscoveryContext
) -> List[Dict[str, Any]]:
    """
    Step 3a: Discover non-standard *.md and *.mdc files in project root.

    Finds markdown files in the project root that weren't discovered in earlier
    steps and aren't in the exclusion list. These might be custom instruction
    files or other relevant documentation.

    Args:
        repo_path: Path object pointing to the repository root directory
        context: DiscoveryContext for tracking discovered files

    Returns:
        List of discovered artifact dictionaries with:
            - file_path (str): Filename (relative path from root)
            - absolute_path (str): Full filesystem path
            - tool_name (str): Always "unknown"
            - artifact_name (str): Filename
            - is_standard (bool): Always False
            - artifact_category (str): "unknown"
            - discovery_method (str): "fallback"
            - discovery_step (str): "non_standard_root"

    Note:
        - Only searches in the repository root, not subdirectories
        - Files in EXCLUDED_ROOT_FILES are skipped
        - Files already in context are skipped (prevents duplicates)

    Example:
        >>> context = DiscoveryContext()
        >>> artifacts = discover_non_standard_root(Path("/repo"), context)
        >>> for a in artifacts:
        ...     print(f"{a['artifact_name']} ({a['tool_name']})")
    """
    # Ensure repo_path is a Path object
    if isinstance(repo_path, str):
        repo_path = Path(repo_path)

    artifacts = []

    for ext in ("*.md", "*.mdc"):
        for md_file in repo_path.glob(ext):
            # Skip if already discovered
            abs_path = str(md_file.resolve())
            if context.is_discovered(abs_path):
                continue

            # Skip excluded files
            if md_file.name in EXCLUDED_ROOT_FILES:
                continue

            # Add artifact with tool_name="unknown"
            artifact = {
                "file_path": md_file.name,
                "absolute_path": abs_path,
                "tool_name": "unknown",
                "artifact_name": md_file.name,
                "is_standard": False,
                "artifact_category": "unknown",
                "discovery_method": "fallback",
                "discovery_step": "non_standard_root",
            }
            artifacts.append(artifact)
            context.mark_discovered(abs_path)

    return artifacts


def discover_non_standard_other(
    repo_path: Path,
    tool_configs: Dict[str, Any],
    context: DiscoveryContext
) -> List[Dict[str, Any]]:
    """
    Step 3b: Discover artifact files in folders not owned by tools.

    Walks the repository, skipping tool config folders and excluded
    directories, to find markdown files that haven't been discovered
    in earlier steps.

    Args:
        repo_path: Path object pointing to the repository root directory
        tool_configs: Dictionary of ToolConfig objects keyed by tool name
        context: DiscoveryContext for tracking discovered files

    Returns:
        List of discovered artifact dictionaries with:
            - file_path (str): Relative path from repo root
            - absolute_path (str): Full filesystem path
            - tool_name (str): Always "unknown"
            - artifact_name (str): Filename
            - is_standard (bool): Always False
            - artifact_category (str): "unknown"
            - discovery_method (str): "fallback"
            - discovery_step (str): "non_standard_other"

    Note:
        - Excluded directories (node_modules, .git, etc.) are skipped
        - Tool config folders are skipped (handled by earlier steps)
        - Root level files are skipped (handled by step 3a)
        - Files already in context are skipped (prevents duplicates)

    Example:
        >>> context = DiscoveryContext()
        >>> artifacts = discover_non_standard_other(
        ...     Path("/repo"), tool_configs, context
        ... )
        >>> for a in artifacts:
        ...     print(f"{a['file_path']} ({a['tool_name']})")
    """
    # Ensure repo_path is a Path object
    if isinstance(repo_path, str):
        repo_path = Path(repo_path)

    artifacts = []

    # Collect all tool config folders (normalized without trailing slash)
    tool_folders = set()
    # Map config folder names to tool names for attribution
    folder_to_tool = {}
    for tool_name, tool_config in tool_configs.items():
        for folder in tool_config.config_folders:
            normalized = folder.rstrip("/")
            tool_folders.add(normalized)
            folder_to_tool[normalized] = tool_name

    for root, dirs, files in os.walk(repo_path):
        rel_root = Path(root).relative_to(repo_path)

        # Skip excluded directories (modify dirs in-place to prevent descent)
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRECTORIES]

        # Skip tool config folders
        rel_root_str = str(rel_root)
        if rel_root_str in tool_folders or any(
            rel_root_str.startswith(tf) for tf in tool_folders
        ):
            continue

        # Skip root (handled in step 3a)
        if rel_root == Path("."):
            continue

        for file in files:
            file_path = Path(root) / file
            rel_path_str = str(file_path.relative_to(repo_path))

            # Check if this file is inside a nested tool config folder
            matched_tool = "unknown"
            for folder, tool in folder_to_tool.items():
                if f"/{folder}/" in f"/{rel_path_str}" or rel_path_str.startswith(f"{folder}/"):
                    matched_tool = tool
                    break

            # Filter by extension: .md/.mdc everywhere, .json only in tool folders
            is_in_tool_folder = matched_tool != "unknown"
            if is_in_tool_folder:
                if not file.endswith((".md", ".mdc", ".json")):
                    continue
            else:
                if not file.endswith((".md", ".mdc")):
                    continue

            abs_path = str(file_path.absolute())

            if context.is_discovered(abs_path):
                continue

            artifact = {
                "file_path": rel_path_str,
                "absolute_path": abs_path,
                "tool_name": matched_tool,
                "artifact_name": file,
                "is_standard": False,
                "artifact_category": "unknown",
                "discovery_method": "fallback",
                "discovery_step": "non_standard_other",
            }
            artifacts.append(artifact)
            context.mark_discovered(abs_path)

    return artifacts


def discover_artifacts(
    repo_path: Union[str, Path],
    tool_configs: Dict[str, Any],
    shared_config: Optional[Any] = None
) -> List[Dict[str, Any]]:
    """
    Discover all artifacts in a repository using hierarchical discovery.

    This is the main entry point for artifact discovery. It orchestrates all
    5 discovery steps in order and returns a unified list of discovered artifacts.

    Discovery Steps:
        1. Tool-specific standard artifacts (attributed to tool)
        2a. Shared artifacts in tool config folders (attributed to tool)
        2b. Shared artifacts in project root (attributed to "shared")
        3a. Non-standard *.md files in root (attributed to "unknown")
        3b. Non-standard *.md files in other folders (attributed to "unknown")

    Args:
        repo_path: Path to the repository root directory (string or Path)
        tool_configs: Dictionary of ToolConfig objects keyed by tool name.
            Typically loaded using artifact_config_loader.load_json_configs()
        shared_config: Optional ToolConfig for shared artifact patterns.
            If not provided, steps 2a and 2b are skipped (backward compatible).

    Returns:
        List of discovered artifact dictionaries with:
            - file_path (str): Relative path from repo root
            - absolute_path (str): Full filesystem path
            - tool_name (str): Name of tool, "shared", or "unknown"
            - artifact_name (str): Filename only
            - is_standard (bool): Whether this is a standard artifact
            - discovery_method (str): Method used
            - discovery_step (str): Which step discovered it

    Note:
        - Results are returned in discovery step order
        - No duplicate entries (DiscoveryContext prevents re-discovery)
        - Empty repositories return an empty list (not an error)

    Example:
        >>> from artifact_config_loader import load_json_configs, load_shared_config
        >>> from file_discovery import discover_artifacts
        >>>
        >>> tool_configs = load_json_configs("Artifacts/")
        >>> shared_config = load_shared_config("Artifacts/")
        >>> artifacts = discover_artifacts("/path/to/repo", tool_configs, shared_config)
        >>>
        >>> for artifact in artifacts:
        ...     print(f"{artifact['tool_name']}: {artifact['file_path']}")
    """
    # Ensure repo_path is a Path object
    if isinstance(repo_path, str):
        repo_path = Path(repo_path)

    context = DiscoveryContext()
    all_artifacts = []

    # Step 1: Tool-specific standard artifacts
    for tool_name, tool_config in tool_configs.items():
        for pattern_obj in tool_config.artifact_patterns:
            # Convert pattern object to dict for compatibility
            pattern = {
                "discovery_method": pattern_obj.discovery_method.value,
                "is_standard": pattern_obj.is_standard
            }

            # Add pattern-specific fields
            if hasattr(pattern_obj, "exact_path") and pattern_obj.exact_path:
                pattern["exact_path"] = pattern_obj.exact_path
            if hasattr(pattern_obj, "glob_pattern") and pattern_obj.glob_pattern:
                pattern["glob_pattern"] = pattern_obj.glob_pattern
            if hasattr(pattern_obj, "regex_pattern") and pattern_obj.regex_pattern:
                pattern["regex_pattern"] = pattern_obj.regex_pattern

            # Get discovery_method
            discovery_method = pattern["discovery_method"]

            # Call appropriate discovery function
            found_files = []
            if discovery_method == "exact_path":
                found_files = discover_exact_path(str(repo_path), pattern)
            elif discovery_method == "glob":
                found_files = discover_glob(str(repo_path), pattern)
            elif discovery_method == "regex":
                found_files = discover_regex(str(repo_path), pattern)

            # Add tool metadata and mark as discovered
            for file_info in found_files:
                abs_path = file_info["absolute_path"]
                if not context.is_discovered(abs_path):
                    context.mark_discovered(abs_path)
                    file_info["tool_name"] = tool_name
                    file_info["artifact_name"] = Path(file_info["file_path"]).name
                    file_info["discovery_step"] = "tool_standard"
                    all_artifacts.append(file_info)

    # Steps 2a, 2b require shared_config
    if shared_config:
        # Step 2a: Shared in tool config folders
        all_artifacts.extend(
            discover_shared_in_config_folders(
                repo_path, tool_configs, shared_config, context
            )
        )

        # Step 2b: Shared in root
        all_artifacts.extend(
            discover_shared_in_root(repo_path, shared_config, context)
        )

    # Step 3a: Non-standard root
    all_artifacts.extend(
        discover_non_standard_root(repo_path, context)
    )

    # Step 3b: Non-standard other folders
    all_artifacts.extend(
        discover_non_standard_other(repo_path, tool_configs, context)
    )

    return all_artifacts
