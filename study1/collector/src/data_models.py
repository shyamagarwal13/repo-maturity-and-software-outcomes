"""
Data models for AI Artifact Data Collection system.

These models represent the structure of JSON configuration files
and internal data structures used throughout the pipeline.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Literal
from enum import Enum


# ============================================================================
# JSON Configuration Models (from *-files.json)
# ============================================================================

class DiscoveryMethod(str, Enum):
    """Methods for discovering artifact files."""
    EXACT_PATH = "exact_path"
    GLOB = "glob"
    REGEX = "regex"


class ArtifactStatus(str, Enum):
    """Status of an artifact pattern."""
    STABLE = "stable"
    DEPRECATED = "deprecated"
    EXPERIMENTAL = "experimental"


@dataclass
class ArtifactPattern:
    """
    Represents a single artifact pattern from JSON configuration.

    Example from cursor-files.json:
    {
      "pattern": ".cursorrules",
      "type": "file",
      "discovery_method": "exact_path",
      "is_standard": true,
      ...
    }
    """
    pattern: str
    type: Literal["file", "directory"]
    description: str
    file_type: str
    status: ArtifactStatus
    is_standard: bool
    artifact_category: str
    scope: str
    discovery_method: DiscoveryMethod
    recursive: bool = False

    # Discovery method specific fields
    exact_path: Optional[str] = None
    glob_pattern: Optional[str] = None
    regex_pattern: Optional[str] = None
    path_prefix: Optional[str] = None

    # Optional metadata
    notes: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> 'ArtifactPattern':
        """Create ArtifactPattern from dictionary (loaded from JSON)."""
        return cls(
            pattern=data["pattern"],
            type=data["type"],
            description=data["description"],
            file_type=data["file_type"],
            status=ArtifactStatus(data["status"]),
            is_standard=data["is_standard"],
            artifact_category=data["artifact_category"],
            scope=data["scope"],
            discovery_method=DiscoveryMethod(data["discovery_method"]),
            recursive=data.get("recursive", False),
            exact_path=data.get("exact_path"),
            glob_pattern=data.get("glob_pattern"),
            regex_pattern=data.get("regex_pattern"),
            path_prefix=data.get("path_prefix"),
            notes=data.get("notes", [])
        )


@dataclass
class ToolConfig:
    """
    Represents a complete tool configuration from *-files.json.

    Example:
    {
      "tool_name": "cursor",
      "config_folders": [".cursor/"],
      "root_files": [".cursorrules"],
      "artifact_patterns": [...]
    }
    """
    tool_name: str
    artifact_patterns: List[ArtifactPattern]
    config_folders: List[str] = field(default_factory=list)
    root_files: List[str] = field(default_factory=list)
    index: int = -1  # Set during registry building

    @classmethod
    def from_dict(cls, data: dict) -> 'ToolConfig':
        """Create ToolConfig from dictionary (loaded from JSON)."""
        return cls(
            tool_name=data["tool_name"],
            artifact_patterns=[
                ArtifactPattern.from_dict(p)
                for p in data["artifact_patterns"]
            ],
            config_folders=data.get("config_folders", []),
            root_files=data.get("root_files", [])
        )


# ============================================================================
# Internal Data Structures
# ============================================================================

@dataclass
class DiscoveredArtifact:
    """
    Represents a discovered artifact file with metadata.

    Used from Phase 3 onwards.
    """
    file_path: str
    absolute_path: str
    tool_name: str
    artifact_name: str
    is_standard: bool
    discovery_method: DiscoveryMethod

    # Added in Phase 4
    text_content: Optional[str] = None
    is_binary: bool = False
    encoding: Optional[str] = None
    file_size: int = 0

    # Added in Phase 5
    word_frequencies: Dict[str, int] = field(default_factory=dict)
    word_count: int = 0
    unique_terms: int = 0

    # Added in Phase 6
    file_id: Optional[str] = None
    repo_name: Optional[str] = None


@dataclass
class FileMetadata:
    """
    File-level metadata for export (Phase 6).
    """
    file_id: str
    repo_name: str
    tool_name: str
    artifact_path: str
    artifact_name: str
    is_standard: bool
    word_count: int
    unique_terms: int
    file_size: int = 0
    is_binary: bool = False


@dataclass
class ToolMetadata:
    """
    Tool-level aggregated metadata (Phase 7).
    """
    repo_name: str
    tool_name: str
    file_count: int
    total_word_count: int
    unique_terms: int
    file_paths: List[str]
    aggregated_frequencies: Dict[str, int] = field(default_factory=dict)


@dataclass
class RepositoryFeatures:
    """
    Repository-level features with N-dimensional vector (Phase 8).
    """
    repo_name: str
    tools_vector: List[int]  # N-dimensional binary vector
    tool_names: List[str]    # Ordered list matching vector positions
    total_artifact_files: int

    def to_dict(self) -> dict:
        """Convert to dictionary for CSV export."""
        return {
            "repo_name": self.repo_name,
            "tools_vector": self.tools_vector,
            "tool_names": self.tool_names,
            "total_artifact_files": self.total_artifact_files
        }


@dataclass
class ArtifactTimelineEvent:
    """
    Single event in artifact timeline (Phase 9).
    """
    commit_sha: str
    commit_date: str  # ISO 8601
    artifact_path: str
    artifact_type: str
    action: Literal["created", "modified", "deleted"]
    author_hash: str
    author_name_hash: str = ""


@dataclass
class CommitAggregated:
    """
    Aggregated commit statistics (Phase 9).
    """
    commit_date: str  # ISO 8601
    commit_sha: str
    author_hash: str
    author_name_hash: str = ""
    files_modified: int = 0
    files_added: int = 0
    files_deleted: int = 0
    total_additions: int = 0
    total_deletions: int = 0


@dataclass
class TFMatrix:
    """
    Term Frequency matrix for any level (file/tool/repo).
    """
    row_ids: List[str]  # file_ids, tool_names, or repo_names
    vocabulary: List[str]  # Ordered list of terms
    matrix: List[List[int]]  # Rows × vocabulary

    def __post_init__(self):
        """Validate matrix dimensions."""
        assert len(self.matrix) == len(self.row_ids)
        if len(self.matrix) > 0:
            assert all(len(row) == len(self.vocabulary) for row in self.matrix)


@dataclass
class ToolRegistry:
    """
    Complete tool registry built from all JSON files (Phase 1).
    """
    tools: Dict[str, ToolConfig]
    tool_names_ordered: List[str]  # Ordered by index
    tool_count: int

    def get_tool_index(self, tool_name: str) -> int:
        """Get vector index for a tool."""
        if tool_name not in self.tools:
            raise ValueError(f"Tool '{tool_name}' not found in registry")
        return self.tools[tool_name].index

    def get_tool_config(self, tool_name: str) -> ToolConfig:
        """Get configuration for a tool."""
        if tool_name not in self.tools:
            raise ValueError(f"Tool '{tool_name}' not found in registry")
        return self.tools[tool_name]


# ============================================================================
# Type Aliases for Common Structures
# ============================================================================

WordFrequencies = Dict[str, int]
Vocabulary = set[str]
ToolsVector = List[int]


# ============================================================================
# Validation Functions
# ============================================================================

def validate_tool_config(data: dict) -> bool:
    """
    Validate tool configuration dictionary before creating ToolConfig.

    Args:
        data: Dictionary loaded from JSON

    Returns:
        True if valid

    Raises:
        ValueError: If validation fails
    """
    # Check required fields
    if "tool_name" not in data:
        raise ValueError("Missing required field: tool_name")

    if "artifact_patterns" not in data:
        raise ValueError("Missing required field: artifact_patterns")

    if not isinstance(data["artifact_patterns"], list):
        raise ValueError("artifact_patterns must be a list")

    # Validate each pattern
    for i, pattern in enumerate(data["artifact_patterns"]):
        if "discovery_method" not in pattern:
            raise ValueError(f"Pattern {i}: Missing discovery_method")

        if pattern["discovery_method"] not in ["exact_path", "glob", "regex"]:
            raise ValueError(
                f"Pattern {i}: Invalid discovery_method: {pattern['discovery_method']}"
            )

        if "is_standard" not in pattern:
            raise ValueError(f"Pattern {i}: Missing is_standard field")

        if not isinstance(pattern["is_standard"], bool):
            raise ValueError(f"Pattern {i}: is_standard must be boolean")

    return True


def validate_artifact_pattern(data: dict) -> bool:
    """
    Validate artifact pattern dictionary.

    Args:
        data: Pattern dictionary from JSON

    Returns:
        True if valid

    Raises:
        ValueError: If validation fails
    """
    required_fields = [
        "pattern", "type", "description", "file_type", "status",
        "is_standard", "artifact_category", "scope", "discovery_method"
    ]

    for field in required_fields:
        if field not in data:
            raise ValueError(f"Missing required field: {field}")

    # Validate discovery_method specific fields
    method = data["discovery_method"]

    if method == "exact_path" and "exact_path" not in data:
        raise ValueError("exact_path discovery_method requires exact_path field")

    if method == "glob" and "glob_pattern" not in data:
        raise ValueError("glob discovery_method requires glob_pattern field")

    if method == "regex" and "regex_pattern" not in data:
        raise ValueError("regex discovery_method requires regex_pattern field")

    return True
