"""
Tests for Artifact Configuration Loader

This test suite follows TDD methodology:
- Tests are written BEFORE implementation
- All tests should FAIL initially (RED phase)
- Then implementation makes them pass (GREEN phase)
"""

import pytest
import json
import glob
from pathlib import Path

# Import functions we'll implement
from src.artifact_config_loader import (
    load_json_configs,
    get_tool_names,
    get_tool_count,
)
from src.data_models import validate_tool_config


def test_load_json_configs():
    """Load all JSON files from Artifacts directory"""
    tools = load_json_configs("Artifacts/")
    assert isinstance(tools, dict)
    assert len(tools) > 0
    assert "cursor" in tools
    assert "claude-code" in tools
    assert "aider" in tools


def test_tool_count_matches_json_files():
    """Tool count equals number of JSON files"""
    tools = load_json_configs("Artifacts/")
    N = get_tool_count(tools)
    # Count actual JSON files
    json_files = glob.glob("Artifacts/*-files.json")
    assert N == len(json_files)


def test_tool_names_ordered():
    """Tool names are in consistent order"""
    tools = load_json_configs("Artifacts/")
    names = get_tool_names(tools)
    assert isinstance(names, list)
    assert len(names) > 0
    # Second call should return same order
    names2 = get_tool_names(tools)
    assert names == names2


def test_tool_has_patterns():
    """Each tool has artifact_patterns array (may be empty)"""
    tools = load_json_configs("Artifacts/")
    for tool_name, tool_data in tools.items():
        assert hasattr(tool_data, "artifact_patterns")
        assert isinstance(tool_data.artifact_patterns, list)
        # Some tools may have empty patterns (e.g., amazon-q-cli)
        # This is valid - just ensure the field exists and is a list


def test_patterns_have_discovery_method():
    """Each pattern has discovery_method field"""
    tools = load_json_configs("Artifacts/")
    for tool_name, tool_data in tools.items():
        for pattern in tool_data.artifact_patterns:
            assert hasattr(pattern, "discovery_method")
            assert pattern.discovery_method.value in ["exact_path", "glob", "regex"]


def test_patterns_have_is_standard():
    """Each pattern has is_standard boolean"""
    tools = load_json_configs("Artifacts/")
    for tool_name, tool_data in tools.items():
        for pattern in tool_data.artifact_patterns:
            assert hasattr(pattern, "is_standard")
            assert isinstance(pattern.is_standard, bool)


def test_tool_index_mapping():
    """Each tool has unique index for vector position"""
    tools = load_json_configs("Artifacts/")
    indices = [tool.index for tool in tools.values()]
    # All indices should be unique
    assert len(indices) == len(set(indices))
    # Indices should be 0 to N-1
    assert min(indices) == 0
    assert max(indices) == len(tools) - 1


def test_invalid_json_raises_error():
    """Invalid JSON file raises appropriate error"""
    # Test with fixtures/invalid_json/
    # First, let's check if directory exists, if not, skip test
    test_dir = "tests/fixtures/invalid_json/"
    if not Path(test_dir).exists():
        pytest.skip("Test fixture directory not created yet")

    with pytest.raises((json.JSONDecodeError, FileNotFoundError)):
        load_json_configs(test_dir)


def test_missing_required_field():
    """Missing required field in config fails validation"""
    invalid_config = {
        # Missing tool_name
        "artifact_patterns": []
    }
    with pytest.raises(ValueError):
        validate_tool_config(invalid_config)


def test_validate_tool_config():
    """Valid config passes validation"""
    valid_config = {
        "tool_name": "test-tool",
        "artifact_patterns": [
            {
                "pattern": ".test",
                "discovery_method": "exact_path",
                "is_standard": True
            }
        ]
    }
    assert validate_tool_config(valid_config) == True


def test_tool_registry_structure():
    """Tool registry has correct structure with ToolConfig objects"""
    tools = load_json_configs("Artifacts/")

    # Should return dict of ToolConfig objects
    for tool_name, tool_config in tools.items():
        assert tool_config.tool_name == tool_name
        assert hasattr(tool_config, "index")
        assert hasattr(tool_config, "artifact_patterns")


def test_consistent_tool_order():
    """Tool names are sorted alphabetically for consistency"""
    tools = load_json_configs("Artifacts/")
    names = get_tool_names(tools)

    # Should be sorted
    assert names == sorted(names)


def test_artifact_patterns_are_dataclass_objects():
    """Artifact patterns are ArtifactPattern dataclass instances"""
    from src.data_models import ArtifactPattern

    tools = load_json_configs("Artifacts/")
    for tool_name, tool_data in tools.items():
        for pattern in tool_data.artifact_patterns:
            assert isinstance(pattern, ArtifactPattern)


# ============================================================================
# Phase 10.1: config_folders and root_files tests
# ============================================================================

def test_tool_config_has_config_folders():
    """Verify cursor has .cursor/ in config_folders"""
    tools = load_json_configs("Artifacts/")
    cursor_config = tools["cursor"]
    assert hasattr(cursor_config, "config_folders")
    assert ".cursor/" in cursor_config.config_folders


def test_tool_config_has_root_files():
    """Verify claude-code has CLAUDE.md in root_files"""
    tools = load_json_configs("Artifacts/")
    claude_config = tools["claude-code"]
    assert hasattr(claude_config, "root_files")
    assert "CLAUDE.md" in claude_config.root_files


def test_tool_config_empty_config_folders():
    """Verify tools without config_folders have empty list"""
    tools = load_json_configs("Artifacts/")
    # All tools should have config_folders as a list (possibly empty)
    for tool_name, tool_config in tools.items():
        assert hasattr(tool_config, "config_folders")
        assert isinstance(tool_config.config_folders, list)


def test_all_tools_have_config_folders_field():
    """Verify config_folders field exists on all tool configs"""
    tools = load_json_configs("Artifacts/")
    for tool_name, tool_config in tools.items():
        assert hasattr(tool_config, "config_folders"), f"{tool_name} missing config_folders"
        assert hasattr(tool_config, "root_files"), f"{tool_name} missing root_files"


# ============================================================================
# Phase 10.2: Separate shared artifacts loading tests
# ============================================================================

def test_load_json_configs_excludes_shared():
    """Load shared-artifacts.json separately - 'shared' not in tool registry"""
    tools = load_json_configs("Artifacts/")
    assert "shared" not in tools


def test_load_shared_config_returns_config():
    """load_shared_config returns ToolConfig with patterns"""
    from src.artifact_config_loader import load_shared_config
    shared_config = load_shared_config("Artifacts/")
    assert shared_config is not None
    assert shared_config.tool_name == "shared"
    assert len(shared_config.artifact_patterns) > 0


def test_load_shared_config_has_agents_pattern():
    """Shared config contains AGENTS.md pattern"""
    from src.artifact_config_loader import load_shared_config
    shared_config = load_shared_config("Artifacts/")
    pattern_names = [p.pattern for p in shared_config.artifact_patterns]
    assert "AGENTS.md" in pattern_names


def test_tool_count_excludes_shared():
    """Tool count is N-1 (shared excluded from count)"""
    tools = load_json_configs("Artifacts/")
    N = get_tool_count(tools)
    # Count actual *-files.json files (excluding shared-artifacts.json)
    json_files = glob.glob("Artifacts/*-files.json")
    assert N == len(json_files)
    # Verify shared-artifacts.json exists but is not counted
    assert Path("Artifacts/shared-artifacts.json").exists()
