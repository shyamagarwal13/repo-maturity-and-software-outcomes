"""
Phase 01: JSON Configuration Loader

This module loads and parses all *-files.json configuration files from the
Artifacts directory to build a dynamic tool registry with N-dimensional vector support.
"""

import json
import glob
from pathlib import Path
from typing import Dict, List, Optional

from src.data_models import (
    ToolConfig,
    ToolRegistry,
    ArtifactPattern,
    validate_tool_config,
    DiscoveryMethod
)


def load_shared_config(artifacts_dir: str) -> Optional[ToolConfig]:
    """
    Load shared-artifacts.json separately from tool configs.

    Args:
        artifacts_dir: Path to directory containing shared-artifacts.json

    Returns:
        ToolConfig for shared artifacts, or None if file doesn't exist
    """
    artifacts_path = Path(artifacts_dir)
    shared_path = artifacts_path / "shared-artifacts.json"

    if shared_path.exists():
        with open(shared_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return ToolConfig.from_dict(data)

    return None


def load_json_configs(artifacts_dir: str) -> Dict[str, ToolConfig]:
    """
    Load all JSON configuration files from artifacts directory.

    Args:
        artifacts_dir: Path to directory containing *-files.json files

    Returns:
        Dictionary mapping tool names to ToolConfig objects with index assigned
        Example: {
            "cursor": ToolConfig(tool_name="cursor", artifact_patterns=[...], index=0),
            "aider": ToolConfig(tool_name="aider", artifact_patterns=[...], index=1),
        }

    Raises:
        FileNotFoundError: If artifacts_dir doesn't exist
        json.JSONDecodeError: If JSON files are malformed
        ValueError: If tool configuration is invalid
    """
    # Ensure directory exists
    artifacts_path = Path(artifacts_dir)
    if not artifacts_path.exists():
        raise FileNotFoundError(f"Directory not found: {artifacts_dir}")

    # Find all *-files.json files
    json_pattern = str(artifacts_path / "*-files.json")
    json_files = sorted(glob.glob(json_pattern))

    if not json_files:
        raise FileNotFoundError(f"No *-files.json files found in {artifacts_dir}")

    # Load and parse each JSON file
    tools = {}
    tool_names = []

    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Validate configuration
        validate_tool_config(data)

        # Create ToolConfig object
        tool_config = ToolConfig.from_dict(data)
        tool_name = tool_config.tool_name

        # Store tool
        tools[tool_name] = tool_config
        tool_names.append(tool_name)

    # Sort tool names alphabetically for consistency
    tool_names.sort()

    # Assign indices based on sorted order
    for idx, tool_name in enumerate(tool_names):
        tools[tool_name].index = idx

    return tools


def get_tool_names(tool_registry: Dict[str, ToolConfig]) -> List[str]:
    """
    Extract ordered list of tool names from registry.

    Args:
        tool_registry: Dictionary of ToolConfig objects (from load_json_configs)

    Returns:
        Ordered list of tool names, sorted alphabetically
        Example: ["aider", "claude-code", "cursor", ...]
    """
    # Extract tool names and sort by index
    tools_with_index = [(tool.index, name) for name, tool in tool_registry.items()]
    tools_with_index.sort()  # Sort by index (first element of tuple)

    return [name for _, name in tools_with_index]


def get_tool_count(tool_registry: Dict[str, ToolConfig]) -> int:
    """
    Get N (dimensionality of tool vector).

    Args:
        tool_registry: Dictionary of ToolConfig objects

    Returns:
        Integer count of tools (N)
    """
    return len(tool_registry)


def build_tool_registry(artifacts_dir: str) -> ToolRegistry:
    """
    Build complete tool registry from JSON files.

    This is a convenience function that combines loading and metadata extraction.

    Args:
        artifacts_dir: Path to directory containing *-files.json files

    Returns:
        ToolRegistry object with all metadata
    """
    tools = load_json_configs(artifacts_dir)
    tool_names = get_tool_names(tools)
    tool_count = get_tool_count(tools)

    return ToolRegistry(
        tools=tools,
        tool_names_ordered=tool_names,
        tool_count=tool_count
    )
