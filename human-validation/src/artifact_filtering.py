"""Boilerplate-vs-AI-artifact discrimination.

Single source of truth, used by:
- notebooks/2. embedding_filtration.ipynb (Phase 1c file-pattern filter)
- src/maturity_scorer.py (drop boilerplate before classification)

The whitelist is auto-loaded from Artifacts/*.json so any AI-tool artifact
declared in those configs is protected from the boilerplate filter.
"""

import json
import re
from pathlib import Path
from typing import Optional, Tuple

# Case-insensitive regex matching project-boilerplate basenames
# (README, LICENSE, CHANGELOG, PR/issue templates, CONTRIBUTING, etc.).
FILTER_FILE_PATTERN = (
    r'^(readme|license|changelog|pull_request_template|merge_request_template'
    r'|issue_template|contributing|code_of_conduct|security|compass-webapp).*'
)
_FILTER_RE = re.compile(FILTER_FILE_PATTERN, re.IGNORECASE)


def _default_artifacts_dir() -> Path:
    """Resolve the analyzer's Artifacts/ directory regardless of cwd."""
    return Path(__file__).resolve().parent.parent / "Artifacts"


def load_protected_patterns(
    artifacts_dir: Optional[Path] = None,
) -> Tuple[set, set, set]:
    """Load AI tool artifact patterns from Artifacts/*.json files.

    Returns:
        (protected_exact, protected_extensions, protected_prefixes)
    """
    artifacts_dir = Path(artifacts_dir) if artifacts_dir else _default_artifacts_dir()
    protected_exact: set = set()
    protected_extensions: set = set()
    protected_prefixes: set = set()

    if not artifacts_dir.is_dir():
        return protected_exact, protected_extensions, protected_prefixes

    for json_file in artifacts_dir.glob("*.json"):
        with open(json_file, "r") as f:
            config = json.load(f)

        for root_file in config.get("root_files", []):
            protected_exact.add(root_file.lower())

        for folder in config.get("config_folders", []):
            protected_prefixes.add(folder.lower().rstrip("/"))

        for pattern in config.get("artifact_patterns", []):
            if "exact_path" in pattern:
                protected_exact.add(Path(pattern["exact_path"]).name.lower())
            if pattern.get("file_type") == "mdc":
                protected_extensions.add(".mdc")
            if "path_prefix" in pattern:
                protected_prefixes.add(pattern["path_prefix"].lower().rstrip("/"))

    return protected_exact, protected_extensions, protected_prefixes


# Lazy module-level cache so repeated calls don't re-read JSON.
_CACHED_PATTERNS: Optional[Tuple[set, set, set]] = None


def _get_patterns(artifacts_dir: Optional[Path] = None) -> Tuple[set, set, set]:
    global _CACHED_PATTERNS
    if artifacts_dir is not None:
        return load_protected_patterns(artifacts_dir)
    if _CACHED_PATTERNS is None:
        _CACHED_PATTERNS = load_protected_patterns()
    return _CACHED_PATTERNS


def is_protected_artifact(
    artifact_name: str,
    artifact_path: Optional[str] = None,
    artifacts_dir: Optional[Path] = None,
) -> bool:
    """True if the artifact is a known AI tool file (must not be filtered).

    Lifted from notebooks/2. embedding_filtration.ipynb so both the
    notebook and the maturity scorer share one definition.
    """
    if not artifact_name:
        return False

    protected_exact, protected_extensions, protected_prefixes = _get_patterns(artifacts_dir)
    name_lower = artifact_name.lower()

    if name_lower in protected_exact:
        return True

    if "agent" in name_lower:
        return True

    for ext in protected_extensions:
        if name_lower.endswith(ext):
            return True

    if artifact_path:
        path_lower = artifact_path.lower()
        for prefix in protected_prefixes:
            if prefix and prefix in path_lower:
                return True

    return False


BOILERPLATE_PATH_SEGMENTS = frozenset({
    ".github/issue_template",
    ".github/pull_request_template",
})


def _path_in_boilerplate_folder(artifact_path: Optional[str]) -> bool:
    """True iff the path contains a known boilerplate folder segment.

    Catches GitHub template folders whose file basenames don't match the
    boilerplate regex (e.g. .github/ISSUE_TEMPLATE/bug_report.md). Path
    comparison is case-insensitive and slash-normalised.
    """
    if not artifact_path:
        return False
    normalised = artifact_path.replace("\\", "/").lower()
    return any(seg in normalised for seg in BOILERPLATE_PATH_SEGMENTS)


def is_boilerplate(
    artifact_name: str,
    artifact_path: Optional[str] = None,
    artifacts_dir: Optional[Path] = None,
) -> bool:
    """True iff the file is project boilerplate (basename OR path-based).

    Two-pass:
      1. Path-based: any file inside `.github/ISSUE_TEMPLATE/` or
         `.github/PULL_REQUEST_TEMPLATE/` is boilerplate regardless of
         basename — GitHub's templated-issue feature lets users name files
         arbitrarily (bug_report.md, feature_request.md, …) and they all
         serve the same "issue template" role.
      2. Basename regex: matches README, LICENSE, CHANGELOG, generic
         {issue,pull_request,merge_request}_template.* etc., minus AI-tool
         artifacts protected by Artifacts/*.json.

    The basename whitelist check is name-only so files like
    `.github/pull_request_template.md` are correctly filtered even though
    `.github/` is registered as a github-copilot config folder.
    """
    if _path_in_boilerplate_folder(artifact_path):
        return True
    if not _FILTER_RE.match(artifact_name or ""):
        return False
    return not is_protected_artifact(artifact_name, artifact_path=None, artifacts_dir=artifacts_dir)


DOC_FOLDER_NAMES = frozenset({"doc", "docs", "documentation"})


def is_in_doc_folder(artifact_path: Optional[str]) -> bool:
    """True iff any path segment equals 'doc', 'docs', or 'documentation' (case-insensitive).

    Used by the maturity scorer's `ignore_doc_folders` sensitivity knob to
    drop documentation-tree files before signal combination. Empirically
    (msrc Jan 2024 – Nov 2025) such files account for 61.5% of the
    population and virtually all of the artifact-category false positives
    driven by content+path classifiers (~2,234 of 2,234 in-doc FP suspects
    had tool_name='unknown' — see
    notebooks/exploratory/documentation_folder_validation.ipynb).
    """
    if not artifact_path:
        return False
    for segment in artifact_path.replace("\\", "/").split("/"):
        if segment.lower() in DOC_FOLDER_NAMES:
            return True
    return False
