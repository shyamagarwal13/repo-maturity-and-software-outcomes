"""
Tests for File Discovery

This test suite follows TDD methodology:
- Tests are written BEFORE implementation
- All tests should FAIL initially (RED phase)
- Then implementation makes them pass (GREEN phase)
"""

import pytest
from pathlib import Path
import os

# Import functions we'll implement
from src.file_discovery import (
    discover_artifacts,
    discover_exact_path,
    discover_glob,
    discover_regex,
    find_non_standard_files,
    deduplicate_artifacts
)


@pytest.fixture
def sample_repo(tmp_path):
    """Create a sample repository for testing"""
    repo = tmp_path / "sample_repo"
    repo.mkdir()
    return repo


@pytest.fixture
def tool_registry():
    """Load the actual tool registry for testing"""
    from src.artifact_config_loader import load_json_configs

    # Load actual tool configurations from Artifacts directory
    return load_json_configs("Artifacts/")


def test_discover_exact_path(sample_repo):
    """Find file by exact path"""
    # Create test file
    test_file = sample_repo / ".cursorrules"
    test_file.write_text("test content")

    pattern = {
        "exact_path": ".cursorrules",
        "is_standard": True,
        "discovery_method": "exact_path"
    }

    results = discover_exact_path(str(sample_repo), pattern)
    assert len(results) == 1
    assert results[0]["file_path"] == ".cursorrules"


def test_discover_exact_path_not_found(sample_repo):
    """Exact path returns empty if not found"""
    pattern = {
        "exact_path": ".nonexistent",
        "is_standard": True,
        "discovery_method": "exact_path"
    }

    results = discover_exact_path(str(sample_repo), pattern)
    assert len(results) == 0


def test_discover_glob_simple(sample_repo):
    """Find files with simple glob pattern"""
    # Create test files
    (sample_repo / ".aider").mkdir()
    (sample_repo / ".aider" / "config.yml").write_text("test")

    pattern = {
        "glob_pattern": ".aider/*.yml",
        "is_standard": True,
        "discovery_method": "glob"
    }

    results = discover_glob(str(sample_repo), pattern)
    assert len(results) >= 1
    assert any(".aider/config.yml" in r["file_path"] for r in results)


def test_discover_glob_recursive(sample_repo):
    """Find files with recursive glob pattern"""
    # Create nested structure
    (sample_repo / ".cursor").mkdir()
    (sample_repo / ".cursor" / "rules").mkdir()
    (sample_repo / ".cursor" / "rules" / "test.mdc").write_text("test")
    (sample_repo / ".cursor" / "rules" / "nested").mkdir()
    (sample_repo / ".cursor" / "rules" / "nested" / "deep.mdc").write_text("test")

    pattern = {
        "glob_pattern": ".cursor/rules/**/*.mdc",
        "recursive": True,
        "is_standard": True,
        "discovery_method": "glob"
    }

    results = discover_glob(str(sample_repo), pattern)
    assert len(results) == 2


def test_discover_regex(sample_repo):
    """Find files matching regex pattern"""
    # Create test files
    (sample_repo / ".aider.conf.yml").write_text("test")
    (sample_repo / ".aider.tags").write_text("test")
    (sample_repo / "regular.txt").write_text("test")

    pattern = {
        "regex_pattern": r"^\.aider.*",
        "is_standard": False,
        "discovery_method": "regex"
    }

    results = discover_regex(str(sample_repo), pattern)
    assert len(results) >= 2
    assert all(r["file_path"].startswith(".aider") for r in results)


def test_discover_artifacts_all_methods(sample_repo, tool_registry):
    """Discover artifacts using all discovery methods"""
    # Create test files for multiple tools
    (sample_repo / ".cursorrules").write_text("cursor rules")
    (sample_repo / ".aider.conf.yml").write_text("aider config")
    (sample_repo / ".claude").mkdir()
    (sample_repo / ".claude" / "settings.json").write_text("{}")

    artifacts = discover_artifacts(str(sample_repo), tool_registry)

    assert len(artifacts) > 0
    # Check we found files from different tools
    tools_found = set(a["tool_name"] for a in artifacts)
    assert len(tools_found) > 0


def test_artifact_metadata(sample_repo, tool_registry):
    """Artifacts have complete metadata"""
    (sample_repo / ".cursorrules").write_text("test")

    artifacts = discover_artifacts(str(sample_repo), tool_registry)

    for artifact in artifacts:
        assert "file_path" in artifact
        assert "absolute_path" in artifact
        assert "tool_name" in artifact
        assert "artifact_name" in artifact
        assert "is_standard" in artifact
        assert "discovery_method" in artifact


def test_deduplicate_artifacts():
    """Remove duplicate artifacts, prefer standard"""
    artifacts = [
        {
            "absolute_path": "/repo/.cursorrules",
            "file_path": ".cursorrules",
            "is_standard": True,
            "tool_name": "cursor"
        },
        {
            "absolute_path": "/repo/.cursorrules",
            "file_path": ".cursorrules",
            "is_standard": False,
            "tool_name": "cursor"
        }
    ]

    unique = deduplicate_artifacts(artifacts)
    assert len(unique) == 1
    assert unique[0]["is_standard"] == True


def test_find_non_standard_files(sample_repo):
    """Find non-standard files in tool directory"""
    # Create .claude directory with various files
    claude_dir = sample_repo / ".claude"
    claude_dir.mkdir()
    (claude_dir / "custom_config.json").write_text("{}")
    (claude_dir / "commands").mkdir()
    (claude_dir / "commands" / "my_command.md").write_text("test")

    files = find_non_standard_files(str(sample_repo), "claude-code", ".claude")

    assert len(files) >= 2
    assert all(not f["is_standard"] for f in files)


def test_empty_repository(tmp_path, tool_registry):
    """Handle empty repository gracefully"""
    empty_repo = tmp_path / "empty"
    empty_repo.mkdir()

    artifacts = discover_artifacts(str(empty_repo), tool_registry)
    assert len(artifacts) == 0


def test_binary_files_included(sample_repo):
    """Binary files are discovered (filtering happens in Phase 4)"""
    # Create a binary file
    binary_file = sample_repo / ".cursor" / "cache.bin"
    binary_file.parent.mkdir(exist_ok=True)
    binary_file.write_bytes(b'\x00\x01\x02\x03')

    pattern = {
        "glob_pattern": ".cursor/*",
        "discovery_method": "glob",
        "is_standard": False
    }

    results = discover_glob(str(sample_repo), pattern)
    # Binary files should be discovered
    assert any("cache.bin" in r["file_path"] for r in results)


def test_symlinks_handling(sample_repo):
    """Handle symbolic links appropriately"""
    # Create a symlink
    target = sample_repo / "target.txt"
    target.write_text("target")
    link = sample_repo / ".cursor_link"
    link.symlink_to(target)

    pattern = {
        "exact_path": ".cursor_link",
        "is_standard": True,
        "discovery_method": "exact_path"
    }

    # Should discover symlink
    results = discover_exact_path(str(sample_repo), pattern)
    assert len(results) == 1


# ============================================================================
# Phase 10.3: Discovery step tracking tests
# ============================================================================

def test_discovery_context_tracks_files():
    """DiscoveryContext tracks files - basic add/check functionality"""
    from src.file_discovery import DiscoveryContext

    context = DiscoveryContext()
    assert context.discovered_count() == 0

    context.mark_discovered("/path/to/file.txt")
    assert context.discovered_count() == 1
    assert context.is_discovered("/path/to/file.txt")


def test_discovery_context_prevents_duplicates():
    """DiscoveryContext prevents duplicates - same path returns True"""
    from src.file_discovery import DiscoveryContext

    context = DiscoveryContext()
    context.mark_discovered("/path/to/file.txt")

    # Same path should be detected as already discovered
    assert context.is_discovered("/path/to/file.txt")
    # Different path should not be discovered
    assert not context.is_discovered("/path/to/other.txt")


def test_artifacts_have_discovery_step(sample_repo, tool_registry):
    """All artifacts have discovery_step field"""
    # Create test files
    (sample_repo / ".cursorrules").write_text("cursor rules")
    (sample_repo / ".aider.conf.yml").write_text("aider config")

    artifacts = discover_artifacts(str(sample_repo), tool_registry)

    assert len(artifacts) > 0
    for artifact in artifacts:
        assert "discovery_step" in artifact, f"Missing discovery_step in {artifact}"


def test_discovery_step_is_tool_standard(sample_repo, tool_registry):
    """Existing pattern discoveries use 'tool_standard' step"""
    # Create test file
    (sample_repo / ".cursorrules").write_text("cursor rules")

    artifacts = discover_artifacts(str(sample_repo), tool_registry)

    assert len(artifacts) > 0
    for artifact in artifacts:
        assert artifact["discovery_step"] == "tool_standard"


# ============================================================================
# Phase 10.4: Shared artifacts in tool config folders tests
# ============================================================================

@pytest.fixture
def shared_config():
    """Load the shared artifacts config for testing"""
    from src.artifact_config_loader import load_shared_config
    return load_shared_config("Artifacts/")


def test_shared_in_tool_folder_agents_md(sample_repo, tool_registry, shared_config):
    """AGENTS.md in .cursor/ tagged as cursor"""
    from src.file_discovery import discover_shared_in_config_folders, DiscoveryContext

    # Create .cursor folder with AGENTS.md
    cursor_dir = sample_repo / ".cursor"
    cursor_dir.mkdir()
    (cursor_dir / "AGENTS.md").write_text("# Cursor Agent Instructions")

    context = DiscoveryContext()
    artifacts = discover_shared_in_config_folders(
        sample_repo, tool_registry, shared_config, context
    )

    assert len(artifacts) >= 1
    agents_artifact = next((a for a in artifacts if "AGENTS.md" in a["file_path"]), None)
    assert agents_artifact is not None
    assert agents_artifact["tool_name"] == "cursor"
    assert agents_artifact["discovery_step"] == "shared_in_tool_folder"


def test_shared_in_tool_folder_mcp_json(sample_repo, tool_registry, shared_config):
    """mcp.json in .claude/ tagged as claude-code"""
    from src.file_discovery import discover_shared_in_config_folders, DiscoveryContext

    # Create .claude folder with mcp.json
    claude_dir = sample_repo / ".claude"
    claude_dir.mkdir()
    (claude_dir / "mcp.json").write_text('{"mcpServers": {}}')

    context = DiscoveryContext()
    artifacts = discover_shared_in_config_folders(
        sample_repo, tool_registry, shared_config, context
    )

    assert len(artifacts) >= 1
    mcp_artifact = next((a for a in artifacts if "mcp.json" in a["file_path"]), None)
    assert mcp_artifact is not None
    assert mcp_artifact["tool_name"] == "claude-code"
    assert mcp_artifact["discovery_step"] == "shared_in_tool_folder"


def test_shared_in_tool_folder_not_rediscovered(sample_repo, tool_registry, shared_config):
    """File already discovered in step 1 not duplicated"""
    from src.file_discovery import discover_shared_in_config_folders, DiscoveryContext

    # Create .claude folder with settings.json (already discovered by step 1)
    claude_dir = sample_repo / ".claude"
    claude_dir.mkdir()
    settings_file = claude_dir / "settings.json"
    settings_file.write_text('{}')

    # Also add AGENTS.md (new file)
    (claude_dir / "AGENTS.md").write_text("# Agent")

    # Simulate step 1 already discovered settings.json
    context = DiscoveryContext()
    context.mark_discovered(str(settings_file.resolve()))

    artifacts = discover_shared_in_config_folders(
        sample_repo, tool_registry, shared_config, context
    )

    # AGENTS.md should be found, but settings.json should NOT be re-discovered
    # (settings.json is not a shared pattern anyway, but the context should prevent duplicates)
    assert all(a["absolute_path"] != str(settings_file.resolve()) for a in artifacts)


def test_shared_in_tool_folder_multiple_tools(sample_repo, tool_registry, shared_config):
    """Each tool folder is searched"""
    from src.file_discovery import discover_shared_in_config_folders, DiscoveryContext

    # Create multiple tool folders with AGENTS.md
    cursor_dir = sample_repo / ".cursor"
    cursor_dir.mkdir()
    (cursor_dir / "AGENTS.md").write_text("# Cursor")

    claude_dir = sample_repo / ".claude"
    claude_dir.mkdir()
    (claude_dir / "AGENTS.md").write_text("# Claude")

    context = DiscoveryContext()
    artifacts = discover_shared_in_config_folders(
        sample_repo, tool_registry, shared_config, context
    )

    # Should find AGENTS.md in both folders
    tool_names = [a["tool_name"] for a in artifacts if "AGENTS.md" in a["file_path"]]
    assert "cursor" in tool_names
    assert "claude-code" in tool_names


def test_shared_in_tool_folder_nonexistent_folder(sample_repo, tool_registry, shared_config):
    """Missing folders skipped gracefully"""
    from src.file_discovery import discover_shared_in_config_folders, DiscoveryContext

    # Don't create any tool folders - they should be skipped gracefully
    context = DiscoveryContext()
    artifacts = discover_shared_in_config_folders(
        sample_repo, tool_registry, shared_config, context
    )

    # Should return empty list, no errors
    assert artifacts == []


# ============================================================================
# Phase 10.5: Shared artifacts in project root tests
# ============================================================================

def test_shared_in_root_agents_md(sample_repo, shared_config):
    """AGENTS.md in root tagged as 'shared'"""
    from src.file_discovery import discover_shared_in_root, DiscoveryContext

    # Create AGENTS.md in root
    (sample_repo / "AGENTS.md").write_text("# Agent Instructions")

    context = DiscoveryContext()
    artifacts = discover_shared_in_root(sample_repo, shared_config, context)

    assert len(artifacts) >= 1
    agents_artifact = next((a for a in artifacts if a["artifact_name"] == "AGENTS.md"), None)
    assert agents_artifact is not None
    assert agents_artifact["tool_name"] == "shared"
    assert agents_artifact["discovery_step"] == "shared_in_root"


def test_shared_in_root_mcp_json(sample_repo, shared_config):
    """.mcp.json in root tagged as 'shared'"""
    from src.file_discovery import discover_shared_in_root, DiscoveryContext

    # Create .mcp.json in root
    (sample_repo / ".mcp.json").write_text('{"mcpServers": {}}')

    context = DiscoveryContext()
    artifacts = discover_shared_in_root(sample_repo, shared_config, context)

    assert len(artifacts) >= 1
    mcp_artifact = next((a for a in artifacts if "mcp" in a["artifact_name"].lower()), None)
    assert mcp_artifact is not None
    assert mcp_artifact["tool_name"] == "shared"


def test_shared_in_root_not_rediscovered(sample_repo, tool_registry, shared_config):
    """File from step 2a not duplicated"""
    from src.file_discovery import discover_shared_in_config_folders, discover_shared_in_root, DiscoveryContext

    # Create AGENTS.md in .cursor (step 2a) AND in root
    cursor_dir = sample_repo / ".cursor"
    cursor_dir.mkdir()
    cursor_agents = cursor_dir / "AGENTS.md"
    cursor_agents.write_text("# Cursor Agent")

    root_agents = sample_repo / "AGENTS.md"
    root_agents.write_text("# Root Agent")

    # Run step 2a first
    context = DiscoveryContext()
    step2a_artifacts = discover_shared_in_config_folders(
        sample_repo, tool_registry, shared_config, context
    )

    # Now run step 2b with same context
    step2b_artifacts = discover_shared_in_root(sample_repo, shared_config, context)

    # Step 2b should only find root AGENTS.md (not cursor's)
    # Root AGENTS.md should be tagged as "shared"
    assert len(step2b_artifacts) >= 1
    root_artifact = next((a for a in step2b_artifacts if a["artifact_name"] == "AGENTS.md"), None)
    assert root_artifact is not None
    assert root_artifact["tool_name"] == "shared"

    # Cursor AGENTS.md should have been found in step 2a
    cursor_artifact = next((a for a in step2a_artifacts if "cursor" in a["file_path"]), None)
    assert cursor_artifact is not None
    assert cursor_artifact["tool_name"] == "cursor"


def test_shared_in_root_discovery_step(sample_repo, shared_config):
    """Correct discovery_step value"""
    from src.file_discovery import discover_shared_in_root, DiscoveryContext

    # Create files in root
    (sample_repo / "AGENTS.md").write_text("# Agent")
    (sample_repo / "AGENTS.override.md").write_text("# Override")

    context = DiscoveryContext()
    artifacts = discover_shared_in_root(sample_repo, shared_config, context)

    # All artifacts should have discovery_step = "shared_in_root"
    for artifact in artifacts:
        assert artifact["discovery_step"] == "shared_in_root"


def test_shared_in_root_only_root(sample_repo, shared_config):
    """Subdirectory files not discovered here"""
    from src.file_discovery import discover_shared_in_root, DiscoveryContext

    # Create AGENTS.md in subdirectory (should NOT be discovered)
    src_dir = sample_repo / "src"
    src_dir.mkdir()
    (src_dir / "AGENTS.md").write_text("# Src Agent - should not be found")

    # No files in root
    context = DiscoveryContext()
    artifacts = discover_shared_in_root(sample_repo, shared_config, context)

    # Should find nothing (file is in subdirectory, not root)
    assert len(artifacts) == 0


# ============================================================================
# Phase 10.6: Non-standard markdown files in root tests
# ============================================================================

def test_non_standard_root_custom_md(sample_repo):
    """Custom INSTRUCTIONS.md discovered"""
    from src.file_discovery import discover_non_standard_root, DiscoveryContext

    # Create custom markdown file in root
    (sample_repo / "INSTRUCTIONS.md").write_text("# Custom instructions")

    context = DiscoveryContext()
    artifacts = discover_non_standard_root(sample_repo, context)

    assert len(artifacts) >= 1
    instructions = next((a for a in artifacts if a["artifact_name"] == "INSTRUCTIONS.md"), None)
    assert instructions is not None
    assert instructions["tool_name"] == "unknown"
    assert instructions["is_standard"] == False


def test_non_standard_root_excludes_readme(sample_repo):
    """README.md is NOT excluded (not in exclusion list)"""
    from src.file_discovery import discover_non_standard_root, DiscoveryContext

    # Create README.md in root
    (sample_repo / "README.md").write_text("# Project README")

    context = DiscoveryContext()
    artifacts = discover_non_standard_root(sample_repo, context)

    # README.md should be discovered (it's NOT in the exclusion list)
    readme = next((a for a in artifacts if a["artifact_name"] == "README.md"), None)
    assert readme is not None


def test_non_standard_root_excludes_changelog(sample_repo):
    """CHANGELOG.md is excluded"""
    from src.file_discovery import discover_non_standard_root, DiscoveryContext, EXCLUDED_ROOT_FILES

    # Verify CHANGELOG.md is in exclusion list
    assert "CHANGELOG.md" in EXCLUDED_ROOT_FILES

    # Create CHANGELOG.md in root
    (sample_repo / "CHANGELOG.md").write_text("# Changelog")

    context = DiscoveryContext()
    artifacts = discover_non_standard_root(sample_repo, context)

    # CHANGELOG.md should NOT be discovered
    changelog = next((a for a in artifacts if a["artifact_name"] == "CHANGELOG.md"), None)
    assert changelog is None


def test_non_standard_root_not_rediscovered(sample_repo, tool_registry):
    """CLAUDE.md not duplicated (already discovered in step 1)"""
    from src.file_discovery import discover_artifacts, discover_non_standard_root, DiscoveryContext

    # Create CLAUDE.md in root (will be discovered by step 1)
    (sample_repo / "CLAUDE.md").write_text("# Claude instructions")

    # Run step 1 discovery
    step1_artifacts = discover_artifacts(str(sample_repo), tool_registry)

    # Build context from step 1
    context = DiscoveryContext()
    for artifact in step1_artifacts:
        context.mark_discovered(artifact["absolute_path"])

    # Now run step 3a
    step3a_artifacts = discover_non_standard_root(sample_repo, context)

    # CLAUDE.md should NOT be in step 3a results (already discovered)
    claude = next((a for a in step3a_artifacts if a["artifact_name"] == "CLAUDE.md"), None)
    assert claude is None


def test_non_standard_root_discovery_step(sample_repo):
    """Correct discovery_step value"""
    from src.file_discovery import discover_non_standard_root, DiscoveryContext

    # Create files in root
    (sample_repo / "CUSTOM.md").write_text("# Custom")
    (sample_repo / "CODING_STYLE.md").write_text("# Style")

    context = DiscoveryContext()
    artifacts = discover_non_standard_root(sample_repo, context)

    # All artifacts should have discovery_step = "non_standard_root"
    for artifact in artifacts:
        assert artifact["discovery_step"] == "non_standard_root"
        assert artifact["discovery_method"] == "fallback"


# ============================================================================
# Phase 10.7: Non-standard markdown files in other folders tests
# ============================================================================

def test_non_standard_other_docs_folder(sample_repo, tool_registry):
    """docs/GUIDE.md discovered with tool_name='unknown'"""
    from src.file_discovery import discover_non_standard_other, DiscoveryContext

    # Create markdown file in docs folder
    docs_dir = sample_repo / "docs"
    docs_dir.mkdir()
    (docs_dir / "GUIDE.md").write_text("# Guide")

    context = DiscoveryContext()
    artifacts = discover_non_standard_other(sample_repo, tool_registry, context)

    assert len(artifacts) >= 1
    guide = next((a for a in artifacts if "GUIDE.md" in a["file_path"]), None)
    assert guide is not None
    assert guide["tool_name"] == "unknown"
    assert guide["is_standard"] == False
    assert guide["discovery_step"] == "non_standard_other"


def test_non_standard_other_skips_tool_folders(sample_repo, tool_registry):
    """.cursor/custom.md NOT discovered here (tool folders handled elsewhere)"""
    from src.file_discovery import discover_non_standard_other, DiscoveryContext

    # Create markdown in tool folder (should be skipped)
    cursor_dir = sample_repo / ".cursor"
    cursor_dir.mkdir()
    (cursor_dir / "custom.md").write_text("# Custom cursor file")

    context = DiscoveryContext()
    artifacts = discover_non_standard_other(sample_repo, tool_registry, context)

    # Should NOT find custom.md (it's in a tool folder)
    cursor_file = next((a for a in artifacts if ".cursor" in a["file_path"]), None)
    assert cursor_file is None


def test_non_standard_other_skips_excluded_dirs(sample_repo, tool_registry):
    """node_modules/*.md excluded"""
    from src.file_discovery import discover_non_standard_other, DiscoveryContext, EXCLUDED_DIRECTORIES

    # Verify node_modules is in excluded directories
    assert "node_modules" in EXCLUDED_DIRECTORIES

    # Create markdown in node_modules
    nm_dir = sample_repo / "node_modules" / "some-pkg"
    nm_dir.mkdir(parents=True)
    (nm_dir / "README.md").write_text("# Package README")

    context = DiscoveryContext()
    artifacts = discover_non_standard_other(sample_repo, tool_registry, context)

    # Should NOT find README.md in node_modules
    nm_file = next((a for a in artifacts if "node_modules" in a["file_path"]), None)
    assert nm_file is None


def test_non_standard_other_skips_git(sample_repo, tool_registry):
    """.git/*.md excluded"""
    from src.file_discovery import discover_non_standard_other, DiscoveryContext, EXCLUDED_DIRECTORIES

    # Verify .git is in excluded directories
    assert ".git" in EXCLUDED_DIRECTORIES

    # Create markdown in .git/hooks
    git_dir = sample_repo / ".git" / "hooks"
    git_dir.mkdir(parents=True)
    (git_dir / "README.md").write_text("# Git hooks")

    context = DiscoveryContext()
    artifacts = discover_non_standard_other(sample_repo, tool_registry, context)

    # Should NOT find README.md in .git
    git_file = next((a for a in artifacts if ".git" in a["file_path"]), None)
    assert git_file is None


def test_non_standard_other_skips_root(sample_repo, tool_registry):
    """Root files NOT discovered here (handled by step 3a)"""
    from src.file_discovery import discover_non_standard_other, DiscoveryContext

    # Create markdown in root (should be skipped - handled by step 3a)
    (sample_repo / "README.md").write_text("# Project README")

    context = DiscoveryContext()
    artifacts = discover_non_standard_other(sample_repo, tool_registry, context)

    # Should NOT find root README.md (handled in step 3a)
    root_readme = next((a for a in artifacts if a["file_path"] == "README.md"), None)
    assert root_readme is None


def test_non_standard_other_nested_folders(sample_repo, tool_registry):
    """src/utils/NOTES.md discovered"""
    from src.file_discovery import discover_non_standard_other, DiscoveryContext

    # Create nested markdown file
    utils_dir = sample_repo / "src" / "utils"
    utils_dir.mkdir(parents=True)
    (utils_dir / "NOTES.md").write_text("# Development notes")

    context = DiscoveryContext()
    artifacts = discover_non_standard_other(sample_repo, tool_registry, context)

    assert len(artifacts) >= 1
    notes = next((a for a in artifacts if "NOTES.md" in a["file_path"]), None)
    assert notes is not None
    assert notes["file_path"] == "src/utils/NOTES.md"
    assert notes["tool_name"] == "unknown"
    assert notes["discovery_step"] == "non_standard_other"


def test_non_standard_other_nested_tool_folder_attributed(sample_repo, tool_registry):
    """v2/.claude/commands/foo.md attributed to claude-code, not unknown"""
    from src.file_discovery import discover_non_standard_other, DiscoveryContext

    # Create a .claude folder nested inside a subdirectory
    nested_claude = sample_repo / "v2" / ".claude" / "commands"
    nested_claude.mkdir(parents=True)
    (nested_claude / "deploy.md").write_text("# Deploy command")

    # Also create a nested .cursor folder
    nested_cursor = sample_repo / "v2" / ".cursor"
    nested_cursor.mkdir(parents=True)
    (nested_cursor / "rules.md").write_text("# Cursor rules")

    # And a truly unknown file for contrast
    docs_dir = sample_repo / "v2" / "docs"
    docs_dir.mkdir(parents=True)
    (docs_dir / "GUIDE.md").write_text("# Guide")

    context = DiscoveryContext()
    artifacts = discover_non_standard_other(sample_repo, tool_registry, context)

    # .claude file should be attributed to claude-code
    claude_file = next((a for a in artifacts if "deploy.md" in a["file_path"]), None)
    assert claude_file is not None
    assert claude_file["tool_name"] == "claude-code"
    assert claude_file["discovery_step"] == "non_standard_other"

    # .cursor file should be attributed to cursor
    cursor_file = next((a for a in artifacts if "rules.md" in a["file_path"]), None)
    assert cursor_file is not None
    assert cursor_file["tool_name"] == "cursor"

    # docs/GUIDE.md should remain unknown
    guide_file = next((a for a in artifacts if "GUIDE.md" in a["file_path"]), None)
    assert guide_file is not None
    assert guide_file["tool_name"] == "unknown"


def test_non_standard_other_discovers_mdc_files(sample_repo, tool_registry):
    """*.mdc files are discovered in general folders"""
    from src.file_discovery import discover_non_standard_other, DiscoveryContext

    docs_dir = sample_repo / "docs"
    docs_dir.mkdir(parents=True)
    (docs_dir / "rules.mdc").write_text("# Some MDC rules")

    context = DiscoveryContext()
    artifacts = discover_non_standard_other(sample_repo, tool_registry, context)

    mdc_file = next((a for a in artifacts if "rules.mdc" in a["file_path"]), None)
    assert mdc_file is not None
    assert mdc_file["tool_name"] == "unknown"
    assert mdc_file["discovery_step"] == "non_standard_other"


def test_non_standard_other_discovers_json_in_nested_tool_folder(sample_repo, tool_registry):
    """*.json files discovered inside nested tool folders, not outside"""
    from src.file_discovery import discover_non_standard_other, DiscoveryContext

    # JSON inside nested .cursor folder -> should be found and attributed
    nested_cursor = sample_repo / "v2" / ".cursor"
    nested_cursor.mkdir(parents=True)
    (nested_cursor / "settings.json").write_text('{"key": "value"}')

    # JSON inside nested .claude folder -> should be found and attributed
    nested_claude = sample_repo / "v2" / ".claude"
    nested_claude.mkdir(parents=True)
    (nested_claude / "settings.json").write_text('{"key": "value"}')

    # JSON outside tool folders -> should NOT be found
    src_dir = sample_repo / "src"
    src_dir.mkdir(parents=True)
    (src_dir / "config.json").write_text('{"key": "value"}')

    context = DiscoveryContext()
    artifacts = discover_non_standard_other(sample_repo, tool_registry, context)

    # .cursor/settings.json attributed to cursor
    cursor_json = next((a for a in artifacts if "v2/.cursor/settings.json" in a["file_path"]), None)
    assert cursor_json is not None
    assert cursor_json["tool_name"] == "cursor"

    # .claude/settings.json attributed to claude-code
    claude_json = next((a for a in artifacts if "v2/.claude/settings.json" in a["file_path"]), None)
    assert claude_json is not None
    assert claude_json["tool_name"] == "claude-code"

    # src/config.json should NOT be discovered
    src_json = next((a for a in artifacts if "src/config.json" in a["file_path"]), None)
    assert src_json is None


def test_non_standard_other_discovers_mdc_in_nested_tool_folder(sample_repo, tool_registry):
    """*.mdc files inside nested tool folders attributed to correct tool"""
    from src.file_discovery import discover_non_standard_other, DiscoveryContext

    nested_cursor = sample_repo / "v2" / ".cursor" / "rules"
    nested_cursor.mkdir(parents=True)
    (nested_cursor / "typescript.mdc").write_text("# TS rules for cursor")

    context = DiscoveryContext()
    artifacts = discover_non_standard_other(sample_repo, tool_registry, context)

    mdc_file = next((a for a in artifacts if "typescript.mdc" in a["file_path"]), None)
    assert mdc_file is not None
    assert mdc_file["tool_name"] == "cursor"
    assert mdc_file["discovery_step"] == "non_standard_other"


def test_non_standard_root_discovers_mdc(sample_repo):
    """*.mdc files discovered in project root"""
    from src.file_discovery import discover_non_standard_root, DiscoveryContext

    (sample_repo / "rules.mdc").write_text("# Root MDC rules")

    context = DiscoveryContext()
    artifacts = discover_non_standard_root(sample_repo, context)

    mdc_file = next((a for a in artifacts if "rules.mdc" in a["file_path"]), None)
    assert mdc_file is not None
    assert mdc_file["tool_name"] == "unknown"
    assert mdc_file["discovery_step"] == "non_standard_root"


# ============================================================================
# Phase 10.8: Integration tests for discover_artifacts orchestration
# ============================================================================

@pytest.fixture
def comprehensive_repo(tmp_path):
    """
    Create comprehensive temp repo with all artifact types:
    - Tool standard artifacts (step 1)
    - Shared artifacts in tool folders (step 2a)
    - Shared artifacts in root (step 2b)
    - Non-standard root files (step 3a)
    - Non-standard other files (step 3b)
    - Excluded directories (should not appear)
    """
    repo = tmp_path / "comprehensive_repo"
    repo.mkdir()

    # Step 1: Tool-specific standard artifacts
    (repo / "CLAUDE.md").write_text("# Claude instructions")
    (repo / ".cursorrules").write_text("cursor rules content")
    cursor_rules = repo / ".cursor" / "rules"
    cursor_rules.mkdir(parents=True)
    (cursor_rules / "python.mdc").write_text("# Python rules")

    # Step 2a: Shared artifacts in tool config folders
    (repo / ".cursor" / "AGENTS.md").write_text("# Cursor agents")
    claude_dir = repo / ".claude"
    claude_dir.mkdir()
    (claude_dir / "mcp.json").write_text('{"mcpServers": {}}')

    # Step 2b: Shared artifacts in root
    (repo / "AGENTS.md").write_text("# Root agents")
    (repo / ".mcp.json").write_text('{"servers": {}}')

    # Step 3a: Non-standard root files
    (repo / "INSTRUCTIONS.md").write_text("# Custom instructions")
    (repo / "README.md").write_text("# Project README")

    # Step 3b: Non-standard other files
    docs_dir = repo / "docs"
    docs_dir.mkdir()
    (docs_dir / "AI_GUIDE.md").write_text("# AI Guide")

    # Excluded directories (should NOT appear in results)
    nm_dir = repo / "node_modules" / "pkg"
    nm_dir.mkdir(parents=True)
    (nm_dir / "README.md").write_text("# Package README")

    return repo


def test_discover_artifacts_all_steps(comprehensive_repo, tool_registry, shared_config):
    """Integration test: All 5 discovery steps execute and return results"""
    from src.file_discovery import discover_artifacts

    artifacts = discover_artifacts(comprehensive_repo, tool_registry, shared_config)

    # Should find artifacts from all steps
    assert len(artifacts) >= 8  # At minimum: tool standard + shared + non-standard

    # Verify we have results from each discovery step
    steps_found = set(a["discovery_step"] for a in artifacts)
    assert "tool_standard" in steps_found
    assert "shared_in_tool_folder" in steps_found
    assert "shared_in_root" in steps_found
    assert "non_standard_root" in steps_found
    assert "non_standard_other" in steps_found


def test_discover_artifacts_correct_attribution(comprehensive_repo, tool_registry, shared_config):
    """Each artifact has correct tool_name based on discovery step"""
    from src.file_discovery import discover_artifacts

    artifacts = discover_artifacts(comprehensive_repo, tool_registry, shared_config)

    # Build lookup by file path
    by_path = {a["file_path"]: a for a in artifacts}

    # Step 1: Tool-specific - attributed to tool
    assert by_path["CLAUDE.md"]["tool_name"] == "claude-code"
    assert by_path["CLAUDE.md"]["discovery_step"] == "tool_standard"

    assert by_path[".cursorrules"]["tool_name"] == "cursor"
    assert by_path[".cursorrules"]["discovery_step"] == "tool_standard"

    # Step 2a: Shared in tool folder - attributed to tool
    cursor_agents = by_path.get(".cursor/AGENTS.md")
    assert cursor_agents is not None
    assert cursor_agents["tool_name"] == "cursor"
    assert cursor_agents["discovery_step"] == "shared_in_tool_folder"

    # Step 2b: Shared in root - attributed to "shared"
    root_agents = by_path.get("AGENTS.md")
    assert root_agents is not None
    assert root_agents["tool_name"] == "shared"
    assert root_agents["discovery_step"] == "shared_in_root"

    # Step 3a: Non-standard root - attributed to "unknown"
    instructions = by_path.get("INSTRUCTIONS.md")
    assert instructions is not None
    assert instructions["tool_name"] == "unknown"
    assert instructions["discovery_step"] == "non_standard_root"

    # Step 3b: Non-standard other - attributed to "unknown"
    ai_guide = by_path.get("docs/AI_GUIDE.md")
    assert ai_guide is not None
    assert ai_guide["tool_name"] == "unknown"
    assert ai_guide["discovery_step"] == "non_standard_other"


def test_discover_artifacts_no_duplicates(comprehensive_repo, tool_registry, shared_config):
    """Same file should not appear twice in results"""
    from src.file_discovery import discover_artifacts

    artifacts = discover_artifacts(comprehensive_repo, tool_registry, shared_config)

    # Check for duplicate absolute paths
    absolute_paths = [a["absolute_path"] for a in artifacts]
    assert len(absolute_paths) == len(set(absolute_paths)), "Duplicate absolute paths found"

    # Check for duplicate file paths
    file_paths = [a["file_path"] for a in artifacts]
    assert len(file_paths) == len(set(file_paths)), "Duplicate file paths found"


def test_discover_artifacts_backward_compatible(comprehensive_repo, tool_registry):
    """Works without shared_config (steps 2a, 2b skipped)"""
    from src.file_discovery import discover_artifacts

    # Call without shared_config (old API)
    artifacts = discover_artifacts(comprehensive_repo, tool_registry)

    # Should still find tool standard and non-standard artifacts
    steps_found = set(a["discovery_step"] for a in artifacts)
    assert "tool_standard" in steps_found
    assert "non_standard_root" in steps_found
    assert "non_standard_other" in steps_found

    # Should NOT have shared artifacts (steps 2a, 2b skipped)
    assert "shared_in_tool_folder" not in steps_found
    assert "shared_in_root" not in steps_found


def test_discover_artifacts_step_order(comprehensive_repo, tool_registry, shared_config):
    """Results are returned in discovery step order"""
    from src.file_discovery import discover_artifacts

    artifacts = discover_artifacts(comprehensive_repo, tool_registry, shared_config)

    # Define expected step order
    step_order = [
        "tool_standard",
        "shared_in_tool_folder",
        "shared_in_root",
        "non_standard_root",
        "non_standard_other"
    ]

    # Get order of steps as they appear in results
    seen_steps = []
    for a in artifacts:
        step = a["discovery_step"]
        if step not in seen_steps:
            seen_steps.append(step)

    # Verify steps appear in correct order
    for i, step in enumerate(seen_steps):
        expected_idx = step_order.index(step)
        for prev_step in seen_steps[:i]:
            prev_idx = step_order.index(prev_step)
            assert prev_idx <= expected_idx, f"Step {step} appeared before {prev_step}"
