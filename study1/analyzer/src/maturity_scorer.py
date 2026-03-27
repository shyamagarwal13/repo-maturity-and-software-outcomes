"""AIME: AI Adoption Maturity Evaluator.

Scores a repository's AI tool adoption maturity (L1-L4) using three signals:
1. Tool detection — known AI tool patterns from Artifacts/*.json
2. Path semantic intent — embed artifact_path, classify against category templates
3. Content semantic classification — existing file embeddings vs category templates

Maturity levels:
- L1 Ad Hoc: No AI artifacts found
- L2 Grounded Prompting: Rules, configuration, architecture, code-style files
- L3 Agent-Augmented: Agents, commands, skills files
- L4 Agentic Orchestration: Flows, session-logs

Uses the same CATEGORY_TEMPLATES and embedding model as
embedding_multi_signal_classification.ipynb.
"""

import re
import pickle
import fnmatch
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.artifact_config_loader import load_json_configs, load_shared_config
from src.embedding_generator import (
    DEFAULT_TASK_PREFIX,
    generate_embeddings_batch,
)


# ============================================================================
# Constants
# ============================================================================

class MaturityLevel(IntEnum):
    """AI adoption maturity levels."""
    L1 = 1  # Ad Hoc
    L2 = 2  # Grounded Prompting
    L3 = 3  # Agent-Augmented
    L4 = 4  # Agentic Orchestration


MATURITY_LABELS = {
    MaturityLevel.L1: "Ad Hoc",
    MaturityLevel.L2: "Grounded Prompting",
    MaturityLevel.L3: "Agent-Augmented",
    MaturityLevel.L4: "Agentic Orchestration",
}

# 9 category templates — identical to embedding_multi_signal_classification.ipynb
CATEGORY_TEMPLATES = {
    "agents": (
        "A persona definition file that establishes an AI agent's identity, role, and behavioral boundaries. "
        "Contains YAML frontmatter with structured fields like name, type, model, tools, and capabilities. "
        "Defines delegation boundaries, domain expertise scope, and interaction protocols for a single autonomous agent."
    ),
    "commands": (
        "A short, self-contained prompt template that defines exactly one executable action a user can invoke. "
        "Typically under 25 lines with a slash-command trigger, parameterized $ARGUMENTS, and a single output. "
        "Not a multi-step orchestration or policy document — just one atomic, reusable operation like commit, review, or format."
    ),
    "flows": (
        "A multi-phase orchestration plan that coordinates several agents or steps through a complex workflow. "
        "Contains tables with worker assignments, timeline phases, and dependency mapping between stages. "
        "Agents are spawned, monitored, and their outputs synthesized — this is a project execution blueprint, not a set of rules or policies."
    ),
    "rules": (
        "A policy document of imperative directives that govern how an AI assistant must behave in a codebase. "
        "Uses mandatory language like NEVER, ALWAYS, MUST, and DO NOT to enforce constraints and conventions. "
        "Does not contain code examples, workflow orchestration tables, or step-by-step tutorials — only behavioral rules and project-level instructions."
    ),
    "skills": (
        "A long-form how-to guide (typically 200-600 lines) that teaches a specific technique or capability in depth. "
        "Includes trigger conditions, detailed step-by-step methodology, MCP tool usage, edge case handling, and validation criteria. "
        "Functions as reusable domain expertise that can be composed and extended, unlike short commands or behavioral rules."
    ),
    "architecture": (
        "A system design document describing software architecture with component diagrams, data flows, and deployment topology. "
        "Uses Mermaid, PlantUML, or ASCII diagrams with ADR-style decision records and C4 model levels. "
        "Covers infrastructure, service boundaries, scaling strategies, and technology stack rationale — not coding standards or runtime configuration."
    ),
    "code-style": (
        "A coding standards document with before-and-after code comparisons showing incorrect vs correct patterns. "
        "Contains inline code examples, linting rules, type safety requirements, naming conventions with specific casing, and coverage metrics. "
        "Focuses on how source code should be written at the syntax level — unlike behavioral rules which govern AI assistant conduct."
    ),
    "configuration": (
        "A machine-readable JSON, YAML, or TOML file with hierarchical key-value pairs, boolean flags, and nested settings objects. "
        "Defines tool servers, environment variables, permission scopes, file patterns, and feature toggles. "
        "Contains no prose paragraphs or natural language instructions — purely structured data for tool or environment configuration."
    ),
    "session-logs": (
        "A retrospective record of work completed by an AI agent, capturing task outcomes, status transitions, and timestamped activity. "
        "Contains structured metadata like status, acceptance criteria checklists, files modified, commits made, or transition logs with actor attribution. "
        "Backward-looking and observational — documents what was done and when, unlike rules that prescribe behavior or flows that plan future work."
    ),
}

CATEGORY_NAMES = list(CATEGORY_TEMPLATES.keys())

# Category → maturity level mapping
CATEGORY_TO_LEVEL: Dict[str, MaturityLevel] = {
    "rules": MaturityLevel.L2,
    "configuration": MaturityLevel.L2,
    "architecture": MaturityLevel.L2,
    "code-style": MaturityLevel.L2,
    "agents": MaturityLevel.L3,
    "commands": MaturityLevel.L3,
    "skills": MaturityLevel.L3,
    "flows": MaturityLevel.L4,
    "session-logs": MaturityLevel.L4,
}

# "instructions" from Artifacts/*.json maps to rules (L2)
ARTIFACT_CATEGORY_TO_TEMPLATE = {
    "instructions": "rules",
    "unknown": None,
}

# Threshold for "within threshold" category attribution
HYBRID_THRESHOLD = 0.03


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class FileClassification:
    """Per-file multi-signal classification result."""
    file_id: str
    artifact_path: str
    tool_name: str
    discovery_step: str

    # Tool detection signal
    tool_category: Optional[str] = None  # From Artifacts/*.json reverse-lookup

    # Content signal (from embeddings)
    content_primary: Optional[str] = None
    content_primary_score: float = 0.0
    content_secondary: Optional[str] = None
    content_secondary_score: float = 0.0
    content_scores: Dict[str, float] = field(default_factory=dict)

    # Path signal
    path_primary: Optional[str] = None
    path_primary_score: float = 0.0
    path_secondary: Optional[str] = None
    path_secondary_score: float = 0.0

    # Combined
    hybrid_score: int = 1  # Number of categories within threshold
    categories_within_threshold: List[str] = field(default_factory=list)
    signals_agree: bool = False
    assigned_category: Optional[str] = None
    assigned_maturity_level: Optional[int] = None

    def to_dict(self) -> dict:
        """Convert to flat dict for DataFrame construction."""
        return {
            "file_id": self.file_id,
            "artifact_path": self.artifact_path,
            "tool_name": self.tool_name,
            "discovery_step": self.discovery_step,
            "tool_category": self.tool_category,
            "content_primary": self.content_primary,
            "content_primary_score": self.content_primary_score,
            "content_secondary": self.content_secondary,
            "content_secondary_score": self.content_secondary_score,
            "path_primary": self.path_primary,
            "path_primary_score": self.path_primary_score,
            "path_secondary": self.path_secondary,
            "path_secondary_score": self.path_secondary_score,
            "hybrid_score": self.hybrid_score,
            "categories_within_threshold": "+".join(self.categories_within_threshold),
            "signals_agree": self.signals_agree,
            "assigned_category": self.assigned_category,
            "assigned_maturity_level": self.assigned_maturity_level,
        }


@dataclass
class CoherenceFlag:
    """A single coherence check result."""
    check: str
    status: str  # "green", "yellow", "red"
    message: str


@dataclass
class MaturityScore:
    """Repository-level maturity assessment."""
    overall_level: int  # 1-4
    overall_label: str
    confidence: float  # 0.0-1.0

    tools_detected: List[str]
    artifact_count: int

    # Evidence counts per level (primary + secondary)
    level_evidence: Dict[int, Dict[str, int]]  # {2: {"primary": 5, "secondary": 12}, ...}

    # Category summary
    category_counts: Dict[str, int]  # primary counts per category

    coherence_flags: List[CoherenceFlag]
    recommendations: List[str]

    # Detailed per-file results
    file_classifications: Optional[pd.DataFrame] = None

    def to_dict(self) -> dict:
        """Export as JSON-serializable dict."""
        return {
            "overall_level": self.overall_level,
            "overall_label": self.overall_label,
            "confidence": round(self.confidence, 3),
            "tools_detected": self.tools_detected,
            "artifact_count": self.artifact_count,
            "level_evidence": self.level_evidence,
            "category_counts": self.category_counts,
            "coherence_flags": [
                {"check": f.check, "status": f.status, "message": f.message}
                for f in self.coherence_flags
            ],
            "recommendations": self.recommendations,
        }


# ============================================================================
# Category Template Embedding
# ============================================================================

def embed_category_templates(model, task_prefix: str = DEFAULT_TASK_PREFIX) -> np.ndarray:
    """Embed the 9 category templates using the given model.

    Args:
        model: Loaded SentenceTransformer model.
        task_prefix: Task prefix for nomic models.

    Returns:
        2D array of shape (9, embedding_dim), rows ordered by CATEGORY_NAMES.
    """
    texts = [CATEGORY_TEMPLATES[cat] for cat in CATEGORY_NAMES]
    prefixed = [f"{task_prefix}: {t}" if task_prefix else t for t in texts]
    embeddings = []
    for text in prefixed:
        emb = model.encode(text)
        embeddings.append(np.array(emb))
    return np.vstack(embeddings)


# ============================================================================
# Signal 1: Tool Detection (reverse-lookup from Artifacts/*.json)
# ============================================================================

def _build_pattern_lookup(artifacts_dir: str) -> List[dict]:
    """Build a flat list of (tool_name, artifact_category, match_spec) entries.

    Args:
        artifacts_dir: Path to the Artifacts/ directory.

    Returns:
        List of dicts with keys: tool_name, artifact_category, match_type, match_value.
    """
    lookup = []
    tools = load_json_configs(artifacts_dir)
    shared = load_shared_config(artifacts_dir)
    if shared:
        tools[shared.tool_name] = shared

    for tool_name, tool_config in tools.items():
        for pattern in tool_config.artifact_patterns:
            entry = {
                "tool_name": tool_name,
                "artifact_category": pattern.artifact_category,
                "discovery_method": pattern.discovery_method.value,
            }
            if pattern.exact_path:
                entry["match_type"] = "exact"
                entry["match_value"] = pattern.exact_path
            elif pattern.glob_pattern:
                entry["match_type"] = "glob"
                entry["match_value"] = pattern.glob_pattern
            else:
                continue
            lookup.append(entry)

    return lookup


def _glob_match(path: str, pattern: str) -> bool:
    """Match a path against a glob pattern, handling ** for zero-or-more dirs.

    Python 3.10's PurePosixPath.match treats ** as one-or-more directories.
    We also try replacing **/ with nothing to match zero intermediate dirs.
    """
    from pathlib import PurePosixPath
    p = PurePosixPath(path)
    if p.match(pattern):
        return True
    # Also try without the **/ segment (matches zero intermediate directories)
    if "**/" in pattern:
        collapsed = pattern.replace("**/", "")
        if fnmatch.fnmatch(path, collapsed):
            return True
    return False


def _match_artifact_category(
    artifact_path: str,
    tool_name: str,
    pattern_lookup: List[dict],
) -> Optional[str]:
    """Resolve the artifact_category for a file using pattern matching.

    Args:
        artifact_path: Relative path of the artifact (e.g., ".claude/commands/sparc.md").
        tool_name: The tool_name from the CSV.
        pattern_lookup: Output of _build_pattern_lookup().

    Returns:
        artifact_category string or None if no match.
    """
    # Try matching against the file's own tool first, then shared, then all
    candidates = [e for e in pattern_lookup if e["tool_name"] == tool_name]
    candidates += [e for e in pattern_lookup if e["tool_name"] == "shared"]
    candidates += [e for e in pattern_lookup if e["tool_name"] not in (tool_name, "shared")]

    for entry in candidates:
        if entry["match_type"] == "exact" and artifact_path == entry["match_value"]:
            return entry["artifact_category"]
        if entry["match_type"] == "glob":
            if _glob_match(artifact_path, entry["match_value"]):
                return entry["artifact_category"]

    return None


def classify_by_tool_detection(
    artifacts_df: pd.DataFrame,
    artifacts_dir: str,
) -> pd.DataFrame:
    """Classify files using tool detection from Artifacts/*.json.

    Args:
        artifacts_df: DataFrame with columns: file_id, artifact_path, tool_name.
        artifacts_dir: Path to the Artifacts/ directory.

    Returns:
        DataFrame with columns: file_id, tool_category.
    """
    pattern_lookup = _build_pattern_lookup(artifacts_dir)

    results = []
    for _, row in artifacts_df.iterrows():
        tool_name = row.get("tool_name", "unknown")
        artifact_path = row.get("artifact_path", "")
        file_id = row.get("file_id", "")

        category = None
        if tool_name != "unknown":
            category = _match_artifact_category(artifact_path, tool_name, pattern_lookup)

        results.append({"file_id": file_id, "tool_category": category})

    return pd.DataFrame(results)


# ============================================================================
# Signal 2: Path Semantic Intent
# ============================================================================

def path_to_semantic_tokens(path: str) -> str:
    """Convert a file path to clean semantic tokens for embedding.

    Same logic as embedding_multi_signal_classification.ipynb Phase 4.

    Args:
        path: Artifact file path (e.g., ".claude/commands/sparc.md").

    Returns:
        Cleaned token string (e.g., "claude commands sparc").
    """
    # Remove file extension
    path = re.sub(r'\.[^/]+$', '', path)
    # Replace path separators with spaces
    path = path.replace('/', ' ').replace('\\', ' ')
    # Remove leading dots (hidden dirs like .claude)
    path = re.sub(r'(?:^|\s)\.', ' ', path)
    # Remove underscores, hyphens -> spaces
    path = path.replace('_', ' ').replace('-', ' ')
    # Collapse whitespace
    path = ' '.join(path.split())
    return path.strip()


def classify_by_path(
    artifact_paths: List[str],
    model,
    template_embeddings: np.ndarray,
    task_prefix: str = DEFAULT_TASK_PREFIX,
) -> pd.DataFrame:
    """Classify files by embedding their paths against category templates.

    Args:
        artifact_paths: List of artifact file paths.
        model: Loaded SentenceTransformer model.
        template_embeddings: Category template embeddings (9 x dim).
        task_prefix: Task prefix for nomic models.

    Returns:
        DataFrame with columns: path_primary, path_primary_score,
        path_secondary, path_secondary_score, path_margin.
    """
    if not artifact_paths:
        return pd.DataFrame(columns=[
            "path_primary", "path_primary_score",
            "path_secondary", "path_secondary_score", "path_margin",
        ])

    path_tokens = [path_to_semantic_tokens(p) for p in artifact_paths]
    path_embeddings = generate_embeddings_batch(
        path_tokens, model, batch_size=64,
        show_progress=False, task_prefix=task_prefix,
    )

    path_sim = cosine_similarity(path_embeddings, template_embeddings)
    sorted_idx = np.argsort(-path_sim, axis=1)

    rows = []
    for i in range(len(artifact_paths)):
        top1_idx = sorted_idx[i, 0]
        top2_idx = sorted_idx[i, 1]
        rows.append({
            "path_primary": CATEGORY_NAMES[top1_idx],
            "path_primary_score": float(path_sim[i, top1_idx]),
            "path_secondary": CATEGORY_NAMES[top2_idx],
            "path_secondary_score": float(path_sim[i, top2_idx]),
            "path_margin": float(path_sim[i, top1_idx] - path_sim[i, top2_idx]),
        })

    return pd.DataFrame(rows)


# ============================================================================
# Signal 3: Content Semantic Classification
# ============================================================================

def classify_by_content(
    file_embeddings: np.ndarray,
    template_embeddings: np.ndarray,
) -> pd.DataFrame:
    """Classify files by their content embeddings against category templates.

    Args:
        file_embeddings: 2D array (N x dim) of file content embeddings.
        template_embeddings: Category template embeddings (9 x dim).

    Returns:
        DataFrame with columns: content_primary, content_primary_score,
        content_secondary, content_secondary_score, content_margin,
        hybrid_score, categories_within_threshold, plus content_{cat} for each category.
    """
    if file_embeddings.size == 0:
        cols = (
            ["content_primary", "content_primary_score",
             "content_secondary", "content_secondary_score",
             "content_margin", "hybrid_score", "categories_within_threshold"]
            + [f"content_{cat}" for cat in CATEGORY_NAMES]
        )
        return pd.DataFrame(columns=cols)

    content_sim = cosine_similarity(file_embeddings, template_embeddings)
    sorted_idx = np.argsort(-content_sim, axis=1)

    rows = []
    for i in range(len(file_embeddings)):
        top1_idx = sorted_idx[i, 0]
        top2_idx = sorted_idx[i, 1]
        top1_score = float(content_sim[i, top1_idx])

        # Categories within threshold of top-1
        threshold = top1_score - HYBRID_THRESHOLD
        within = sorted([
            CATEGORY_NAMES[j]
            for j in range(len(CATEGORY_NAMES))
            if content_sim[i, j] >= threshold
        ])

        row = {
            "content_primary": CATEGORY_NAMES[top1_idx],
            "content_primary_score": top1_score,
            "content_secondary": CATEGORY_NAMES[top2_idx],
            "content_secondary_score": float(content_sim[i, top2_idx]),
            "content_margin": float(content_sim[i, top1_idx] - content_sim[i, top2_idx]),
            "hybrid_score": len(within),
            "categories_within_threshold": "+".join(within),
        }
        for j, cat in enumerate(CATEGORY_NAMES):
            row[f"content_{cat}"] = float(content_sim[i, j])

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================================
# Signal Combination
# ============================================================================

def _normalize_category(category: Optional[str]) -> Optional[str]:
    """Normalize artifact_category from JSON configs to template category names.

    Maps "instructions" -> "rules", "unknown" -> None, etc.
    """
    if category is None:
        return None
    if category in CATEGORY_TO_LEVEL:
        return category
    return ARTIFACT_CATEGORY_TO_TEMPLATE.get(category, None)


def combine_signals(
    artifacts_df: pd.DataFrame,
    tool_signal: pd.DataFrame,
    path_signal: pd.DataFrame,
    content_signal: pd.DataFrame,
) -> List[FileClassification]:
    """Combine three classification signals into per-file classifications.

    Priority for assigned_category:
    1. If tool_category is known and maps to a template category, use it
    2. If path and content agree, use the agreed category
    3. Use content_primary (content signal is richest)

    Args:
        artifacts_df: DataFrame with file_id, artifact_path, tool_name, discovery_step.
        tool_signal: DataFrame with file_id, tool_category.
        path_signal: DataFrame with path_primary, path_primary_score, etc.
        content_signal: DataFrame with content_primary, content_primary_score, etc.

    Returns:
        List of FileClassification objects.
    """
    classifications = []

    for i in range(len(artifacts_df)):
        row = artifacts_df.iloc[i]
        file_id = str(row.get("file_id", f"file_{i}"))

        fc = FileClassification(
            file_id=file_id,
            artifact_path=str(row.get("artifact_path", "")),
            tool_name=str(row.get("tool_name", "unknown")),
            discovery_step=str(row.get("discovery_step", "")),
        )

        # Tool signal
        if i < len(tool_signal):
            tc = tool_signal.iloc[i].get("tool_category")
            fc.tool_category = _normalize_category(tc) if tc else None

        # Content signal
        if i < len(content_signal):
            cs = content_signal.iloc[i]
            fc.content_primary = cs.get("content_primary")
            fc.content_primary_score = float(cs.get("content_primary_score", 0))
            fc.content_secondary = cs.get("content_secondary")
            fc.content_secondary_score = float(cs.get("content_secondary_score", 0))
            fc.hybrid_score = int(cs.get("hybrid_score", 1))
            cats_str = cs.get("categories_within_threshold", "")
            fc.categories_within_threshold = cats_str.split("+") if cats_str else []

            # Collect per-category content scores
            for cat in CATEGORY_NAMES:
                score = cs.get(f"content_{cat}")
                if score is not None:
                    fc.content_scores[cat] = float(score)

        # Path signal
        if i < len(path_signal):
            ps = path_signal.iloc[i]
            fc.path_primary = ps.get("path_primary")
            fc.path_primary_score = float(ps.get("path_primary_score", 0))
            fc.path_secondary = ps.get("path_secondary")
            fc.path_secondary_score = float(ps.get("path_secondary_score", 0))

        # Signal agreement
        fc.signals_agree = (
            fc.content_primary is not None
            and fc.path_primary is not None
            and fc.content_primary == fc.path_primary
        )

        # Determine assigned category
        if fc.tool_category and fc.tool_category in CATEGORY_TO_LEVEL:
            fc.assigned_category = fc.tool_category
        elif fc.signals_agree and fc.content_primary:
            fc.assigned_category = fc.content_primary
        elif fc.content_primary:
            fc.assigned_category = fc.content_primary
        elif fc.path_primary:
            fc.assigned_category = fc.path_primary

        # Maturity level from assigned category
        if fc.assigned_category and fc.assigned_category in CATEGORY_TO_LEVEL:
            fc.assigned_maturity_level = int(CATEGORY_TO_LEVEL[fc.assigned_category])

        classifications.append(fc)

    return classifications


# ============================================================================
# Coherence Checks
# ============================================================================

def _check_coherence(
    level_primary: Dict[int, int],
    level_secondary: Dict[int, int],
) -> List[CoherenceFlag]:
    """Run maturity coherence checks.

    The model is cumulative: higher levels should include lower-level foundations.

    Args:
        level_primary: {level: primary_count} for levels 2, 3, 4.
        level_secondary: {level: secondary_count} for levels 2, 3, 4.

    Returns:
        List of CoherenceFlag objects.
    """
    flags = []

    l2_primary = level_primary.get(2, 0)
    l2_secondary = level_secondary.get(2, 0)
    l3_primary = level_primary.get(3, 0)
    l4_primary = level_primary.get(4, 0)

    # L2 foundation check
    if l2_primary > 0:
        flags.append(CoherenceFlag(
            check="L2 foundation",
            status="green",
            message=f"L2 grounding present ({l2_primary} primary artifacts)",
        ))
    elif l2_secondary > 0:
        flags.append(CoherenceFlag(
            check="L2 foundation",
            status="yellow",
            message=f"L2 grounding embedded in other files ({l2_secondary} secondary), no standalone L2 artifacts",
        ))
    else:
        if l3_primary > 0 or l4_primary > 0:
            flags.append(CoherenceFlag(
                check="L2 foundation",
                status="red",
                message="No L2 grounding detected — higher levels present without foundation",
            ))

    # L3 builds on L2
    if l3_primary > 0 and l2_primary == 0 and l2_secondary == 0:
        flags.append(CoherenceFlag(
            check="L3 without L2",
            status="red",
            message="L3 agent artifacts present but no L2 grounding at all — anomaly",
        ))
    elif l3_primary > 0 and l2_primary == 0 and l2_secondary > 0:
        flags.append(CoherenceFlag(
            check="L3 without L2",
            status="yellow",
            message="L3 present, L2 grounding only as secondary signal — consider standalone grounding files",
        ))
    elif l3_primary > 0 and l2_primary > 0:
        flags.append(CoherenceFlag(
            check="L3 builds on L2",
            status="green",
            message="L3 agents build on L2 grounding — progressive adoption",
        ))

    # L4 builds on L3
    if l4_primary > 0 and l3_primary == 0:
        flags.append(CoherenceFlag(
            check="L4 without L3",
            status="red",
            message="L4 orchestration artifacts present but no L3 agent artifacts — anomaly",
        ))
    elif l4_primary > 0 and l3_primary > 0:
        flags.append(CoherenceFlag(
            check="L4 builds on L3",
            status="green",
            message="L4 orchestration builds on L3 agents — full maturity stack",
        ))

    return flags


# ============================================================================
# Confidence
# ============================================================================

def _compute_confidence(
    overall_level: int,
    artifact_count: int,
    signal_agreement_rate: float,
    coherence_flags: List[CoherenceFlag],
) -> float:
    """Compute confidence score for the maturity assessment.

    Args:
        overall_level: Determined maturity level (1-4).
        artifact_count: Total number of artifacts.
        signal_agreement_rate: Fraction of files where path and content agree.
        coherence_flags: List of coherence check results.

    Returns:
        Confidence score between 0.0 and 1.0.
    """
    if overall_level == 1:
        return 1.0  # No artifacts = definitively L1

    # Base confidence from artifact count (more artifacts = more confident)
    if artifact_count >= 20:
        count_score = 1.0
    elif artifact_count >= 10:
        count_score = 0.8
    elif artifact_count >= 5:
        count_score = 0.6
    else:
        count_score = 0.4

    # Signal agreement contribution
    agree_score = signal_agreement_rate

    # Coherence penalty
    red_flags = sum(1 for f in coherence_flags if f.status == "red")
    yellow_flags = sum(1 for f in coherence_flags if f.status == "yellow")
    coherence_score = max(0.0, 1.0 - 0.2 * red_flags - 0.1 * yellow_flags)

    # Weighted combination
    confidence = (
        0.35 * count_score
        + 0.35 * agree_score
        + 0.30 * coherence_score
    )

    return round(min(1.0, max(0.0, confidence)), 3)


# ============================================================================
# Aggregate Repo-Level Score
# ============================================================================

def aggregate_repo_maturity(
    file_classifications: List[FileClassification],
) -> MaturityScore:
    """Compute repository-level maturity score from per-file classifications.

    The repo's maturity level is the highest level with at least one
    confirmed primary artifact. Coherence flags note when higher levels
    exist without lower-level foundations.

    Args:
        file_classifications: List of FileClassification objects.

    Returns:
        MaturityScore object.
    """
    if not file_classifications:
        return MaturityScore(
            overall_level=1,
            overall_label=MATURITY_LABELS[MaturityLevel.L1],
            confidence=1.0,
            tools_detected=[],
            artifact_count=0,
            level_evidence={2: {"primary": 0, "secondary": 0},
                           3: {"primary": 0, "secondary": 0},
                           4: {"primary": 0, "secondary": 0}},
            category_counts={cat: 0 for cat in CATEGORY_NAMES},
            coherence_flags=[],
            recommendations=["No AI artifacts detected. Consider starting with a rules file (e.g., CLAUDE.md, .cursorrules)."],
        )

    # Count primary evidence per level
    level_primary: Dict[int, int] = {2: 0, 3: 0, 4: 0}
    level_secondary: Dict[int, int] = {2: 0, 3: 0, 4: 0}
    category_counts: Dict[str, int] = {cat: 0 for cat in CATEGORY_NAMES}

    for fc in file_classifications:
        # Primary category → primary level evidence
        if fc.assigned_category and fc.assigned_category in CATEGORY_TO_LEVEL:
            level = int(CATEGORY_TO_LEVEL[fc.assigned_category])
            level_primary[level] = level_primary.get(level, 0) + 1
            category_counts[fc.assigned_category] = category_counts.get(fc.assigned_category, 0) + 1

        # Secondary/within-threshold categories → secondary level evidence
        for cat in fc.categories_within_threshold:
            if cat != fc.assigned_category and cat in CATEGORY_TO_LEVEL:
                sec_level = int(CATEGORY_TO_LEVEL[cat])
                level_secondary[sec_level] = level_secondary.get(sec_level, 0) + 1

    # Tools detected
    tools = sorted(set(
        fc.tool_name for fc in file_classifications
        if fc.tool_name and fc.tool_name != "unknown"
    ))

    # Determine highest level with ≥1 primary artifact
    if level_primary.get(4, 0) > 0:
        overall_level = 4
    elif level_primary.get(3, 0) > 0:
        overall_level = 3
    elif level_primary.get(2, 0) > 0:
        overall_level = 2
    else:
        overall_level = 1

    # Coherence checks
    coherence_flags = _check_coherence(level_primary, level_secondary)

    # Signal agreement rate
    agreed = sum(1 for fc in file_classifications if fc.signals_agree)
    total = len(file_classifications)
    agreement_rate = agreed / total if total > 0 else 0.0

    # Confidence
    confidence = _compute_confidence(
        overall_level, total, agreement_rate, coherence_flags,
    )

    # Recommendations
    recommendations = _generate_recommendations(
        overall_level, level_primary, level_secondary,
        coherence_flags, category_counts,
    )

    # Level evidence dict
    level_evidence = {}
    for lvl in (2, 3, 4):
        level_evidence[lvl] = {
            "primary": level_primary.get(lvl, 0),
            "secondary": level_secondary.get(lvl, 0),
        }

    # Build file classifications DataFrame
    fc_df = pd.DataFrame([fc.to_dict() for fc in file_classifications])

    return MaturityScore(
        overall_level=overall_level,
        overall_label=MATURITY_LABELS[MaturityLevel(overall_level)],
        confidence=confidence,
        tools_detected=tools,
        artifact_count=total,
        level_evidence=level_evidence,
        category_counts=category_counts,
        coherence_flags=coherence_flags,
        recommendations=recommendations,
        file_classifications=fc_df,
    )


def _generate_recommendations(
    overall_level: int,
    level_primary: Dict[int, int],
    level_secondary: Dict[int, int],
    coherence_flags: List[CoherenceFlag],
    category_counts: Dict[str, int],
) -> List[str]:
    """Generate actionable recommendations based on the maturity assessment."""
    recs = []

    if overall_level == 1:
        recs.append(
            "Start with a rules/instructions file (CLAUDE.md, .cursorrules, copilot-instructions.md) "
            "to ground AI tools in project-specific context."
        )
        return recs

    # Red coherence flags → immediate recommendations
    for flag in coherence_flags:
        if flag.status == "red" and "L3 without L2" in flag.check:
            recs.append(
                "Add standalone grounding files (rules, configuration, architecture docs) "
                "to provide L2 foundation for your L3 agent artifacts."
            )
        if flag.status == "red" and "L4 without L3" in flag.check:
            recs.append(
                "Add agent definitions (L3) before scaling to orchestration (L4). "
                "Agents provide the building blocks that flows coordinate."
            )

    # Yellow flags
    for flag in coherence_flags:
        if flag.status == "yellow" and "L2 foundation" in flag.check:
            recs.append(
                "L2 grounding is embedded inside other files. Consider extracting "
                "shared rules into dedicated CLAUDE.md or .cursorrules for maintainability."
            )

    # Level-specific advancement suggestions
    if overall_level == 2:
        if category_counts.get("agents", 0) == 0 and category_counts.get("commands", 0) == 0:
            recs.append(
                "Advance to L3 by adding agent definitions or reusable commands "
                "that enable autonomous AI behaviors."
            )
    elif overall_level == 3:
        if level_primary.get(4, 0) == 0:
            recs.append(
                "Advance to L4 by creating workflow orchestration files (flows) "
                "that coordinate multiple agents through complex tasks."
            )

    # Category concentration warning
    total = sum(category_counts.values())
    if total > 0:
        max_cat_count = max(category_counts.values())
        concentration = max_cat_count / total
        if concentration > 0.8 and total >= 5:
            dominant = max(category_counts, key=category_counts.get)
            recs.append(
                f"Artifact adoption is concentrated in '{dominant}' ({max_cat_count}/{total}). "
                f"Consider diversifying across categories for deeper AI integration."
            )

    return recs


# ============================================================================
# Artifacts Map (Summary Table)
# ============================================================================

def build_artifacts_map(
    file_classifications: List[FileClassification],
) -> pd.DataFrame:
    """Build the per-category summary table (the "artifacts map").

    Args:
        file_classifications: List of FileClassification objects.

    Returns:
        DataFrame with one row per category, columns:
        category, primary_path, primary_content, secondary_content,
        agreement, maturity_level, total_primary.
    """
    rows = []
    for cat in CATEGORY_NAMES:
        path_primary = sum(
            1 for fc in file_classifications if fc.path_primary == cat
        )
        content_primary = sum(
            1 for fc in file_classifications if fc.content_primary == cat
        )
        secondary_content = sum(
            1 for fc in file_classifications
            if cat in fc.categories_within_threshold and fc.assigned_category != cat
        )
        agreement = sum(
            1 for fc in file_classifications
            if fc.path_primary == cat and fc.content_primary == cat
        )
        level = int(CATEGORY_TO_LEVEL[cat])

        rows.append({
            "category": cat,
            "primary_path": path_primary,
            "primary_content": content_primary,
            "secondary_content": secondary_content,
            "agreement": agreement,
            "maturity_level": level,
            "total_primary": content_primary,
        })

    return pd.DataFrame(rows)


def build_tool_category_matrix(
    file_classifications: List[FileClassification],
) -> pd.DataFrame:
    """Build tool × category heatmap data.

    Args:
        file_classifications: List of FileClassification objects.

    Returns:
        DataFrame with tools as index, categories as columns, counts as values.
    """
    known = [fc for fc in file_classifications if fc.tool_name != "unknown"]
    if not known:
        return pd.DataFrame()

    tools = sorted(set(fc.tool_name for fc in known))
    data = {cat: [0] * len(tools) for cat in CATEGORY_NAMES}

    tool_idx = {t: i for i, t in enumerate(tools)}
    for fc in known:
        if fc.assigned_category and fc.tool_name in tool_idx:
            cat = fc.assigned_category
            if cat in data:
                data[cat][tool_idx[fc.tool_name]] += 1

    return pd.DataFrame(data, index=tools)


# ============================================================================
# Report Generation
# ============================================================================

def generate_report(score: MaturityScore) -> dict:
    """Generate a JSON report from the maturity score.

    Args:
        score: MaturityScore object.

    Returns:
        JSON-serializable dict matching the AIME output schema.
    """
    report = score.to_dict()

    # Add derived diagnostics
    total = score.artifact_count
    if total > 0 and score.category_counts:
        max_count = max(score.category_counts.values())
        report["category_concentration"] = round(max_count / total, 3)
    else:
        report["category_concentration"] = 0.0

    if score.file_classifications is not None and not score.file_classifications.empty:
        agreed = score.file_classifications["signals_agree"].sum()
        report["signal_agreement_rate"] = round(agreed / total, 3)
    else:
        report["signal_agreement_rate"] = 0.0

    # Level stacking
    report["level_stacking"] = {
        f"L{lvl}": score.level_evidence.get(lvl, {}).get("primary", 0)
        + score.level_evidence.get(lvl, {}).get("secondary", 0)
        for lvl in (2, 3, 4)
    }

    return report


# ============================================================================
# Main Entry Point
# ============================================================================

def score_from_output_dir(
    output_path: str,
    repo_name: str,
    model,
    artifacts_dir: str = "Artifacts",
    task_prefix: str = DEFAULT_TASK_PREFIX,
) -> MaturityScore:
    """Score a single repository from its output directory.

    Loads CSVs/PKL from {output_path}/{repo_name}/, runs all three signals,
    and returns the maturity score.

    Args:
        output_path: Path to the output directory (e.g., "output/" or "embedding_output/").
        repo_name: Repository name (subdirectory name).
        model: Loaded SentenceTransformer model.
        artifacts_dir: Path to the Artifacts/ directory.
        task_prefix: Task prefix for nomic models.

    Returns:
        MaturityScore object.
    """
    repo_dir = Path(output_path) / repo_name

    # Load file artifacts CSV
    csv_files = list(repo_dir.glob("*_file_artifacts.csv"))
    if not csv_files:
        return MaturityScore(
            overall_level=1,
            overall_label=MATURITY_LABELS[MaturityLevel.L1],
            confidence=1.0,
            tools_detected=[],
            artifact_count=0,
            level_evidence={2: {"primary": 0, "secondary": 0},
                           3: {"primary": 0, "secondary": 0},
                           4: {"primary": 0, "secondary": 0}},
            category_counts={cat: 0 for cat in CATEGORY_NAMES},
            coherence_flags=[],
            recommendations=["No artifact data found for this repository."],
        )

    artifacts_df = pd.read_csv(csv_files[0])

    # Load embeddings PKL
    pkl_files = list(repo_dir.glob("*_embeddings.pkl"))
    file_embeddings = None
    embedding_file_ids = []
    if pkl_files:
        with open(pkl_files[0], "rb") as f:
            emb_data = pickle.load(f)
        file_embeddings = emb_data.get("embeddings")
        embedding_file_ids = list(emb_data.get("file_ids", []))

    # Embed category templates
    template_embeddings = embed_category_templates(model, task_prefix)

    # Signal 1: Tool detection
    tool_signal = classify_by_tool_detection(artifacts_df, artifacts_dir)

    # Signal 2: Path semantic intent
    path_signal = classify_by_path(
        artifacts_df["artifact_path"].tolist(),
        model, template_embeddings, task_prefix,
    )

    # Signal 3: Content semantic classification
    # Align file embeddings with artifacts_df by file_id
    if file_embeddings is not None and len(embedding_file_ids) > 0:
        # Build embedding lookup
        emb_lookup = {}
        for idx, fid in enumerate(embedding_file_ids):
            emb_lookup[fid] = file_embeddings[idx]

        # Align with artifacts_df
        aligned_embeddings = []
        for _, row in artifacts_df.iterrows():
            fid = row.get("file_id", "")
            if fid in emb_lookup:
                aligned_embeddings.append(emb_lookup[fid])
            else:
                # Zero vector for files without embeddings
                dim = file_embeddings.shape[1] if file_embeddings.ndim == 2 else 768
                aligned_embeddings.append(np.zeros(dim))

        aligned_emb_array = np.vstack(aligned_embeddings)
        content_signal = classify_by_content(aligned_emb_array, template_embeddings)
    else:
        content_signal = classify_by_content(np.array([]).reshape(0, 768), template_embeddings)
        # Pad with empty rows to match artifacts_df
        empty_rows = []
        for _ in range(len(artifacts_df)):
            row = {
                "content_primary": None,
                "content_primary_score": 0.0,
                "content_secondary": None,
                "content_secondary_score": 0.0,
                "content_margin": 0.0,
                "hybrid_score": 1,
                "categories_within_threshold": "",
            }
            for cat in CATEGORY_NAMES:
                row[f"content_{cat}"] = 0.0
            empty_rows.append(row)
        content_signal = pd.DataFrame(empty_rows)

    # Combine signals
    classifications = combine_signals(
        artifacts_df, tool_signal, path_signal, content_signal,
    )

    # Aggregate to repo-level score
    return aggregate_repo_maturity(classifications)
