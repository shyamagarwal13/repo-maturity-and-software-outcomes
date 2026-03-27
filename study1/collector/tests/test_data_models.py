import pytest
from src.data_models import (
    ArtifactPattern, ToolConfig, ToolRegistry, RepositoryFeatures,
    DiscoveryMethod, ArtifactStatus,
    validate_tool_config, validate_artifact_pattern
)


def test_artifact_pattern_from_dict():
    """Create ArtifactPattern from dictionary"""
    data = {
        "pattern": ".cursorrules",
        "type": "file",
        "description": "Cursor rules",
        "file_type": "text",
        "status": "stable",
        "is_standard": True,
        "artifact_category": "rules",
        "scope": "project",
        "discovery_method": "exact_path",
        "exact_path": ".cursorrules",
        "recursive": False
    }

    pattern = ArtifactPattern.from_dict(data)

    assert pattern.pattern == ".cursorrules"
    assert pattern.discovery_method == DiscoveryMethod.EXACT_PATH
    assert pattern.is_standard == True
    assert pattern.exact_path == ".cursorrules"


def test_tool_config_from_dict():
    """Create ToolConfig from dictionary"""
    data = {
        "tool_name": "cursor",
        "artifact_patterns": [
            {
                "pattern": ".cursorrules",
                "type": "file",
                "description": "Rules",
                "file_type": "text",
                "status": "stable",
                "is_standard": True,
                "artifact_category": "rules",
                "scope": "project",
                "discovery_method": "exact_path",
                "exact_path": ".cursorrules",
                "recursive": False
            }
        ]
    }

    config = ToolConfig.from_dict(data)

    assert config.tool_name == "cursor"
    assert len(config.artifact_patterns) == 1
    assert config.artifact_patterns[0].pattern == ".cursorrules"


def test_validate_tool_config_valid():
    """Valid tool config passes validation"""
    data = {
        "tool_name": "cursor",
        "artifact_patterns": [
            {
                "discovery_method": "exact_path",
                "is_standard": True
            }
        ]
    }

    assert validate_tool_config(data) == True


def test_validate_tool_config_missing_tool_name():
    """Missing tool_name raises error"""
    data = {
        "artifact_patterns": []
    }

    with pytest.raises(ValueError, match="tool_name"):
        validate_tool_config(data)


def test_validate_tool_config_invalid_discovery_method():
    """Invalid discovery_method raises error"""
    data = {
        "tool_name": "test",
        "artifact_patterns": [
            {
                "discovery_method": "invalid_method",
                "is_standard": True
            }
        ]
    }

    with pytest.raises(ValueError, match="discovery_method"):
        validate_tool_config(data)


def test_tool_registry():
    """Test ToolRegistry functionality"""
    tool1 = ToolConfig(tool_name="cursor", artifact_patterns=[], index=0)
    tool2 = ToolConfig(tool_name="aider", artifact_patterns=[], index=1)

    registry = ToolRegistry(
        tools={"cursor": tool1, "aider": tool2},
        tool_names_ordered=["cursor", "aider"],
        tool_count=2
    )

    assert registry.get_tool_index("cursor") == 0
    assert registry.get_tool_index("aider") == 1
    assert registry.get_tool_config("cursor") == tool1


def test_repository_features_to_dict():
    """RepositoryFeatures converts to dict correctly"""
    features = RepositoryFeatures(
        repo_name="test-repo",
        tools_vector=[1, 0, 1],
        tool_names=["cursor", "aider", "claude-code"],
        total_artifact_files=10
    )

    data = features.to_dict()

    assert data["repo_name"] == "test-repo"
    assert data["tools_vector"] == [1, 0, 1]
    assert len(data["tool_names"]) == 3
