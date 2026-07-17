"""Tests for src.artifact_filtering — boilerplate vs AI artifact discrimination."""

from src.artifact_filtering import (
    DOC_FOLDER_NAMES,
    FILTER_FILE_PATTERN,
    is_boilerplate,
    is_in_doc_folder,
    is_protected_artifact,
    load_protected_patterns,
)


class TestIsBoilerplate:
    """is_boilerplate must catch generic project files but not AI artifacts."""

    def test_readme_is_boilerplate(self):
        assert is_boilerplate("README.md", "README.md") is True
        assert is_boilerplate("README", "README") is True
        assert is_boilerplate("readme.md", "readme.md") is True

    def test_license_changelog_contributing(self):
        assert is_boilerplate("LICENSE", "LICENSE") is True
        assert is_boilerplate("CHANGELOG.md", "CHANGELOG.md") is True
        assert is_boilerplate("CONTRIBUTING.md", "CONTRIBUTING.md") is True
        assert is_boilerplate("CODE_OF_CONDUCT.md", "CODE_OF_CONDUCT.md") is True
        assert is_boilerplate("SECURITY.md", "SECURITY.md") is True

    def test_pull_request_template_is_boilerplate(self):
        assert is_boilerplate(
            "pull_request_template.md",
            ".github/pull_request_template.md",
        ) is True
        assert is_boilerplate(
            "PULL_REQUEST_TEMPLATE.md",
            ".github/PULL_REQUEST_TEMPLATE.md",
        ) is True
        assert is_boilerplate(
            "issue_template.md",
            ".github/issue_template.md",
        ) is True

    def test_claude_md_is_not_boilerplate(self):
        """CLAUDE.md is whitelisted by Artifacts/*.json — must NOT be filtered."""
        assert is_boilerplate("CLAUDE.md", "CLAUDE.md") is False
        assert is_boilerplate("claude.md", "claude.md") is False

    def test_agents_md_is_not_boilerplate(self):
        """AGENTS.md (cross-tool standard) is whitelisted."""
        assert is_boilerplate("AGENTS.md", "AGENTS.md") is False

    def test_arbitrary_md_is_not_boilerplate(self):
        """Files that don't match the filter pattern simply pass through."""
        assert is_boilerplate("foo.md", "docs/foo.md") is False
        assert is_boilerplate("setup.md", "docs/setup.md") is False

    def test_empty_input(self):
        assert is_boilerplate("", "") is False
        assert is_boilerplate("", None) is False


class TestIsProtectedArtifact:
    """is_protected_artifact whitelist check."""

    def test_known_ai_files_protected(self):
        assert is_protected_artifact("CLAUDE.md") is True
        assert is_protected_artifact("AGENTS.md") is True
        assert is_protected_artifact("AGENTS.override.md") is True

    def test_agent_keyword_in_name_protected(self):
        assert is_protected_artifact("custom_agent.md") is True
        assert is_protected_artifact("my-agent-helper.md") is True

    def test_files_under_claude_dir_protected(self):
        assert is_protected_artifact("foo.md", ".claude/agents/foo.md") is True
        assert is_protected_artifact("anything.md", ".claude/commands/anything.md") is True

    def test_files_under_cursor_dir_protected(self):
        assert is_protected_artifact("rule.mdc", ".cursor/rules/rule.mdc") is True

    def test_random_file_not_protected(self):
        assert is_protected_artifact("README.md", "README.md") is False
        assert is_protected_artifact("setup.md", "docs/setup.md") is False


class TestLoadProtectedPatterns:
    """load_protected_patterns reads Artifacts/*.json."""

    def test_loads_default_patterns(self):
        exact, exts, prefixes = load_protected_patterns()
        # Spot-check a few canonical entries that must be present.
        assert "claude.md" in exact
        assert "agents.md" in exact
        assert ".mdc" in exts
        assert any("claude" in p for p in prefixes)

    def test_handles_missing_dir(self, tmp_path):
        """Non-existent dir returns empty sets, doesn't raise."""
        exact, exts, prefixes = load_protected_patterns(tmp_path / "nonexistent")
        assert exact == set()
        assert exts == set()
        assert prefixes == set()


def test_filter_pattern_constant_exposed():
    """notebook 2 imports FILTER_FILE_PATTERN; must remain accessible."""
    assert "readme" in FILTER_FILE_PATTERN
    assert "pull_request_template" in FILTER_FILE_PATTERN


class TestIsInDocFolder:
    """is_in_doc_folder catches doc/docs/documentation trees in any case."""

    def test_docs_lowercase(self):
        assert is_in_doc_folder("docs/getting-started.md") is True
        assert is_in_doc_folder("project/docs/api.md") is True

    def test_doc_singular(self):
        assert is_in_doc_folder("doc/Skill.md") is True

    def test_documentation_segment(self):
        assert is_in_doc_folder("documentation/architecture/adr.md") is True

    def test_case_insensitive(self):
        assert is_in_doc_folder("Documentation/intro.md") is True
        assert is_in_doc_folder("DOC/file.md") is True
        assert is_in_doc_folder("Docs/file.md") is True

    def test_nested_under_other_folders(self):
        # mixed real-world: agents/docs/src/foo.md still trips the filter
        assert is_in_doc_folder("agents/docs/src/foo.md") is True

    def test_windows_separator(self):
        assert is_in_doc_folder("project\\docs\\api.md") is True

    def test_substring_does_not_trigger(self):
        # 'documentation_helper.py' is not a doc-folder segment
        assert is_in_doc_folder("src/documentation_helper.py") is False
        assert is_in_doc_folder("src/docstring.md") is False
        assert is_in_doc_folder("internal-docs.md") is False

    def test_no_doc_segment(self):
        assert is_in_doc_folder("CLAUDE.md") is False
        assert is_in_doc_folder(".claude/agents/foo.md") is False
        assert is_in_doc_folder("src/main.py") is False

    def test_empty_or_none(self):
        assert is_in_doc_folder("") is False
        assert is_in_doc_folder(None) is False

    def test_constant_exposed(self):
        # Downstream callers may want to extend or inspect the set.
        assert "doc" in DOC_FOLDER_NAMES
        assert "docs" in DOC_FOLDER_NAMES
        assert "documentation" in DOC_FOLDER_NAMES
