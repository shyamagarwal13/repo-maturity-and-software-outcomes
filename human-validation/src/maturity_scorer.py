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
from src.artifact_filtering import is_boilerplate, is_in_doc_folder
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
        "An executable orchestration plan authored to be consumed by an AI runner — names specific phases with identifiers, assigns named agents or workers to each phase, declares explicit dependency edges between stages, and lists per-phase acceptance/exit criteria. "
        "The file directly drives execution: a runner could parse it and dispatch work. "
        "NOT a design document explaining how orchestration works, NOT a tutorial teaching about workflows, NOT a roadmap or release schedule, NOT a code recipe describing how an event sequence behaves, NOT a doc page about multi-agent systems — those describe orchestration; this IS an orchestration."
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
        "An actual log entry produced by a specific AI agent run — captures concrete artifacts of that run: a run/session/task identifier, real timestamps of state transitions that occurred, the names of files that were actually modified, real commit SHAs that were made, the agent's actor identity, and the outcome of acceptance criteria. "
        "The file is the OUTPUT of an executed agent run — a forensic record that exists because the run happened. "
        "NOT a documentation page describing how session logs work, NOT a tutorial about agent memory or logging, NOT an observability guide, NOT a state-machine reference manual, NOT a tracing-system explanation, NOT a retrospective design discussion, NOT an event-sourcing recipe — those describe logs; this IS a log."
    ),
    "general-documentation": (
        "User-facing software project documentation written for end users, contributors, or operators of a non-AI software system. "
        "Covers installation, API reference, usage tutorials, performance characteristics, troubleshooting, FAQs, design rationale, deployment guides, and operational concerns. "
        "Describes how the project itself works — does not configure, instruct, or orchestrate any AI assistant or agent. "
        "Not authored by an AI tool, not consumed by an AI tool: ordinary technical writing for humans about a software product."
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

# Dialect bridge: each Artifacts/*.json author chose category names that
# mirror their tool's terminology, so artifact_category uses a wider
# vocabulary than CATEGORY_TEMPLATES. This dict translates every dialect
# value not already in CATEGORY_TO_LEVEL into a template category, so
# tool detection produces level evidence consistently across tools.
# Mappings are grounded in each pattern's description in Artifacts/*.json.
ARTIFACT_CATEGORY_TO_TEMPLATE = {
    # → rules (L2): grounding / behavioral instructions for the AI
    "instructions": "rules",   # claude-code, github-copilot, shared (CLAUDE.md, copilot-instructions.md)
    "context":      "rules",   # gemini-cli (GEMINI.md persistent project context)
    "steering":     "rules",   # kiro (.kiro/steering/*.md guides Kiro's behavior)
    "guidelines":   "rules",   # jetbrains-ai (Junie coding standards & best practices)

    # → commands (L3): short reusable prompt templates
    "prompts":      "commands",  # github-copilot (.github/prompts/*.prompt.md)

    # → agents (L3): named agent/persona definitions
    "microagents":  "agents",    # openhands (.openhands/microagents/*.md)

    # → flows (L4): event/task-driven agent orchestration
    "workflows":    "flows",     # windsurf (.windsurf/workflows/*.md Cascade task sequences)
    "hooks":        "flows",     # kiro (.kiro/hooks/*.hook automated event→agent-action triggers)

    # Sentinel: explicit drop
    "unknown":      None,
}

# Threshold for "within threshold" category attribution
HYBRID_THRESHOLD = 0.03


@dataclass(frozen=True)
class ScoringConfig:
    """Tunable thresholds and decision rules for the semantic signals.

    Files whose semantic top-1 cosine score or top1-top2 margin fall below
    the per-signal gates contribute no semantic evidence — their score
    columns are still populated for diagnostics, but `content_primary` /
    `path_primary` is set to None, so they don't drive level attribution
    through the semantic branches of `combine_signals`. Tool-based
    attribution (Artifacts JSON pattern matching) is unaffected.

    Optional sensitivity knobs:
    - cross_level_disagreement_demote: off by default. When enabled, a
      cross-level disagreement between the content and path categories
      is resolved by picking the LOWER-level category (a conservative
      sensitivity variant). With the default (False), content wins on
      any disagreement — content embeddings are the richer signal and
      this avoids the asymmetric downward bias of always demoting.
    - strict_cap_to_l1_without_tool_attribution: on by default. The
      repo's overall_level is capped at L1 unless at least one file's
      tool_category resolved to a leveled template via Artifacts/*.json.
      Semantic content/path evidence alone cannot promote a repo. Turn
      off to recover the un-capped semantic level for sensitivity work.
    - filter_boilerplate: on by default. Drops project-boilerplate files
      (README, LICENSE, PR/issue templates, …) before / during signal
      combination via the shared is_boilerplate predicate. The filter is
      applied early in score_from_output_dir (saves classification work)
      and defensively in combine_signals (catches direct callers, e.g.
      sensitivity notebooks, that skip the early stage). Turn off only
      for diagnostics on the unfiltered input.
    - ignore_doc_folders: on by default. Drops every file whose path
      contains a segment named doc/docs/documentation (case-insensitive)
      before signal combination — same drop pattern as filter_boilerplate,
      gated by is_in_doc_folder. Motivation: empirical validation
      (notebooks/exploratory/documentation_folder_validation.ipynb) shows
      that 61.5% of MD files in the msrc cohort live under doc folders,
      and that virtually all (~2,234) artifact-category false positives
      there are driven by content+path semantic matches on doc content
      (every in-doc FP suspect had tool_name='unknown'). Turn off only
      for diagnostics on the unfiltered input — note this will reintroduce
      the FP surface measured in §3 of the validation notebook.
    """

    tau_content_score: float = 0.70
    tau_content_margin: float = 0.02
    tau_path_score: float = 0.70
    tau_path_margin: float = 0.02
    hybrid_threshold: float = HYBRID_THRESHOLD

    cross_level_disagreement_demote: bool = False
    strict_cap_to_l1_without_tool_attribution: bool = True
    filter_boilerplate: bool = True
    ignore_doc_folders: bool = True


DEFAULT_CONFIG = ScoringConfig()


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

    # LLM overlay (populated only when score_from_output_dir is called with
    # llm_overlay + blob_hash_for_files). aime_assigned_category preserves
    # the multi-signal label for side-by-side comparison; assigned_category
    # is overwritten by the LLM verdict (or set to None when llm_cut=True).
    aime_assigned_category: Optional[str] = None
    blob_hash: Optional[str] = None
    llm_rationale: Optional[str] = None
    llm_cut: bool = False
    llm_cut_reason: Optional[str] = None

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
            "aime_assigned_category": self.aime_assigned_category,
            "blob_hash": self.blob_hash,
            "llm_rationale": self.llm_rationale,
            "llm_cut": self.llm_cut,
            "llm_cut_reason": self.llm_cut_reason,
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

    # Detailed per-file results
    file_classifications: Optional[pd.DataFrame] = None

    # Diagnostics — number of files dropped by the boilerplate filter
    # before classification (README, LICENSE, PR templates, etc.).
    boilerplate_filtered: int = 0

    # Diagnostics — number of files dropped because their path lives inside
    # a doc/docs/documentation segment (config.ignore_doc_folders).
    doc_folder_filtered: int = 0

    # True iff at least one file was assigned to a leveled category by the
    # tool-detection signal (Artifacts/*.json pattern matching). Used by
    # downstream "strict mode" evaluation in notebook 8 to distinguish
    # repos with confirmed AI tool presence from repos whose level is
    # driven entirely by semantic (content/path) evidence.
    has_leveled_tool_attribution: bool = False

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
            "boilerplate_filtered": self.boilerplate_filtered,
            "doc_folder_filtered": self.doc_folder_filtered,
            "has_leveled_tool_attribution": self.has_leveled_tool_attribution,
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

    # Skip placeholder/catch-all patterns whose `artifact_category` is the
    # sentinel "unknown" (e.g. shared `**/*.md` used by the collector for
    # bulk MD discovery). Otherwise those would shadow specific patterns
    # like `**/CLAUDE.md`, `**/GEMINI.md` which live in tool-specific
    # configs that the candidate ordering visits *after* shared.
    for entry in candidates:
        if entry.get("artifact_category") == "unknown":
            continue
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

        # Always try path-pattern matching, even when the collector tagged the
        # file as 'unknown'. The collector only attaches a tool_name for files
        # matching an exact_path; nested canonical files (e.g. app/.../AGENTS.md,
        # nested CLAUDE.md/GEMINI.md) arrive as 'unknown' and would otherwise
        # be silently dropped from tool attribution. The pattern lookup itself
        # is the source of truth.
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
    config: ScoringConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Classify files by embedding their paths against category templates.

    Args:
        artifact_paths: List of artifact file paths.
        model: Loaded SentenceTransformer model.
        template_embeddings: Category template embeddings (9 x dim).
        task_prefix: Task prefix for nomic models.
        config: Scoring thresholds — files below tau_path_score or with
            margin below tau_path_margin have path_primary set to None.

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
        top1_score = float(path_sim[i, top1_idx])
        top2_score = float(path_sim[i, top2_idx])
        margin = top1_score - top2_score
        gated = top1_score < config.tau_path_score or margin < config.tau_path_margin
        rows.append({
            "path_primary": None if gated else CATEGORY_NAMES[top1_idx],
            "path_primary_score": top1_score,
            "path_secondary": CATEGORY_NAMES[top2_idx],
            "path_secondary_score": top2_score,
            "path_margin": margin,
        })

    return pd.DataFrame(rows)


# ============================================================================
# Signal 3: Content Semantic Classification
# ============================================================================

def classify_by_content(
    file_embeddings: np.ndarray,
    template_embeddings: np.ndarray,
    config: ScoringConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Classify files by their content embeddings against category templates.

    Args:
        file_embeddings: 2D array (N x dim) of file content embeddings.
        template_embeddings: Category template embeddings (9 x dim).
        config: Scoring thresholds — files below tau_content_score or with
            margin below tau_content_margin have content_primary set to None.

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
        top2_score = float(content_sim[i, top2_idx])
        margin = top1_score - top2_score

        # Categories within threshold of top-1
        threshold = top1_score - config.hybrid_threshold
        within = sorted([
            CATEGORY_NAMES[j]
            for j in range(len(CATEGORY_NAMES))
            if content_sim[i, j] >= threshold
        ])

        gated = top1_score < config.tau_content_score or margin < config.tau_content_margin

        row = {
            "content_primary": None if gated else CATEGORY_NAMES[top1_idx],
            "content_primary_score": top1_score,
            "content_secondary": CATEGORY_NAMES[top2_idx],
            "content_secondary_score": top2_score,
            "content_margin": margin,
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

def _coerce_str_or_none(value) -> Optional[str]:
    """Treat NaN / None / empty string as None; otherwise return str(value)."""
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    s = str(value)
    return s if s else None


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
    config: ScoringConfig = DEFAULT_CONFIG,
) -> List[FileClassification]:
    """Combine three classification signals into per-file classifications.

    Files matching the boilerplate predicate are dropped up front when
    `config.filter_boilerplate` is set (defensive — the primary drop happens
    earlier in `score_from_output_dir`). The same pattern applies to
    `config.ignore_doc_folders`: when enabled, files inside a
    doc/docs/documentation path segment are dropped up front (defensive —
    primary drop in `score_from_output_dir`).

    Priority chain for `assigned_category` (first match wins):
    1. `tool_category`, if it maps to a leveled template category.
    2. `content_primary`, when both content and path signals are non-gated
       and agree (`signals_agree`).
    3. Both signals non-gated but disagreeing:
         - if `config.cross_level_disagreement_demote` is on AND the
           categories live at different maturity levels → pick the
           LOWER-level category (conservative variant).
         - otherwise → prefer `content_primary` (richer signal).
    4. `content_primary` alone (path gated or missing).
    5. `path_primary` alone (content gated or missing).
    Files matching none of the above get no `assigned_category` and
    contribute no level evidence downstream.

    Also populated on each FileClassification: `signals_agree`,
    `assigned_maturity_level` (from CATEGORY_TO_LEVEL), `hybrid_score`,
    `categories_within_threshold`, and per-category `content_scores`.

    Args:
        artifacts_df: DataFrame with file_id, artifact_path, tool_name, discovery_step.
        tool_signal: DataFrame with file_id, tool_category.
        path_signal: DataFrame with path_primary, path_primary_score, etc.
        content_signal: DataFrame with content_primary, content_primary_score,
            categories_within_threshold, hybrid_score, content_{cat} columns.
        config: ScoringConfig — controls boilerplate filtering and the
            cross-level disagreement demote rule.

    Returns:
        List of FileClassification objects (one per surviving input row).
    """
    classifications = []

    for i in range(len(artifacts_df)):
        row = artifacts_df.iloc[i]
        file_id = str(row.get("file_id", f"file_{i}"))

        # Defensive boilerplate filter — primary drop happens earlier in
        # score_from_output_dir; this catches direct callers (e.g. notebooks
        # running classify_*/combine_signals manually for sensitivity work)
        # so the filter cannot be silently bypassed. Disable via
        # config.filter_boilerplate=False for diagnostics on raw input.
        if config.filter_boilerplate and is_boilerplate(
            str(row.get("artifact_name", "") or ""),
            str(row.get("artifact_path", "") or ""),
        ):
            continue

        # Defensive doc-folder filter — same shape as the boilerplate one.
        if config.ignore_doc_folders and is_in_doc_folder(
            str(row.get("artifact_path", "") or "")
        ):
            continue

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
            fc.content_primary = _coerce_str_or_none(cs.get("content_primary"))
            fc.content_primary_score = float(cs.get("content_primary_score", 0))
            fc.content_secondary = _coerce_str_or_none(cs.get("content_secondary"))
            fc.content_secondary_score = float(cs.get("content_secondary_score", 0))
            fc.hybrid_score = int(cs.get("hybrid_score", 1))
            cats_str = cs.get("categories_within_threshold", "")
            cats_str = "" if cats_str is None or (isinstance(cats_str, float) and pd.isna(cats_str)) else cats_str
            fc.categories_within_threshold = cats_str.split("+") if cats_str else []

            # Collect per-category content scores
            for cat in CATEGORY_NAMES:
                score = cs.get(f"content_{cat}")
                if score is not None and not (isinstance(score, float) and pd.isna(score)):
                    fc.content_scores[cat] = float(score)

        # Path signal
        if i < len(path_signal):
            ps = path_signal.iloc[i]
            fc.path_primary = _coerce_str_or_none(ps.get("path_primary"))
            fc.path_primary_score = float(ps.get("path_primary_score", 0))
            fc.path_secondary = _coerce_str_or_none(ps.get("path_secondary"))
            fc.path_secondary_score = float(ps.get("path_secondary_score", 0))

        # Signal agreement
        fc.signals_agree = (
            fc.content_primary is not None
            and fc.path_primary is not None
            and fc.content_primary == fc.path_primary
        )

        # Effective per-signal levels (None when gated or non-leveled, e.g. general-documentation)
        cp_level = CATEGORY_TO_LEVEL.get(fc.content_primary) if fc.content_primary else None
        pp_level = CATEGORY_TO_LEVEL.get(fc.path_primary) if fc.path_primary else None

        # Determine assigned category — priority chain unchanged for tool / agreement / single-signal,
        # but content+path disagreement is resolved by config.cross_level_disagreement_demote.
        if fc.tool_category and fc.tool_category in CATEGORY_TO_LEVEL:
            fc.assigned_category = fc.tool_category
        elif fc.signals_agree and fc.content_primary:
            fc.assigned_category = fc.content_primary
        elif fc.content_primary and fc.path_primary:
            # Both gates passed; categories disagree.
            if (config.cross_level_disagreement_demote
                    and cp_level is not None and pp_level is not None
                    and cp_level != pp_level):
                # Cross-level disagreement → keep the LOWER-level category.
                fc.assigned_category = fc.content_primary if cp_level < pp_level else fc.path_primary
            else:
                # Same-level disagreement, or one side has no level mapping → prefer content (richer signal).
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
    boilerplate_filtered: int = 0,
    doc_folder_filtered: int = 0,
    config: ScoringConfig = DEFAULT_CONFIG,
) -> MaturityScore:
    """Compute repository-level maturity score from per-file classifications.

    The repo's maturity level is the highest level with at least one
    confirmed primary artifact. Coherence flags note when higher levels
    exist without lower-level foundations.

    With config.strict_cap_to_l1_without_tool_attribution = True (default),
    the level is capped at L1 unless at least one file's tool_category
    resolved to a leveled template via Artifacts/*.json — semantic
    evidence alone cannot promote a repo above L1.

    Args:
        file_classifications: List of FileClassification objects.
        boilerplate_filtered: Count of files dropped by the boilerplate filter.
        doc_folder_filtered: Count of files dropped by the doc-folder filter.
        config: Scoring config (controls the strict cap and demote lever).

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
            boilerplate_filtered=boilerplate_filtered,
            doc_folder_filtered=doc_folder_filtered,
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

    # True if at least one file's tool_category resolved to a leveled
    # template (i.e., tool detection produced real level evidence — not
    # just a tool_name from a config-folder discovery). Used by the
    # downstream "strict" evaluation profile in notebook 8.
    has_leveled_tool_attribution = any(
        fc.tool_category and fc.tool_category in CATEGORY_TO_LEVEL
        for fc in file_classifications
    )

    # Determine highest level with ≥1 primary artifact
    if level_primary.get(4, 0) > 0:
        overall_level = 4
    elif level_primary.get(3, 0) > 0:
        overall_level = 3
    elif level_primary.get(2, 0) > 0:
        overall_level = 2
    else:
        overall_level = 1

    # Strict cap: repo can only stand above L1 if at least one file has
    # a leveled tool-category attribution (real AI tool config detected
    # via Artifacts/*.json). Semantic evidence alone cannot promote.
    if (config.strict_cap_to_l1_without_tool_attribution
            and not has_leveled_tool_attribution):
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
        file_classifications=fc_df,
        boilerplate_filtered=boilerplate_filtered,
        doc_folder_filtered=doc_folder_filtered,
        has_leveled_tool_attribution=has_leveled_tool_attribution,
    )


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
        # general-documentation has no level (it's the absorbing/unclassified bucket).
        level_enum = CATEGORY_TO_LEVEL.get(cat)
        level = int(level_enum) if level_enum is not None else None

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

def load_llm_overlay(
    labels_csv: str,
    anomalies_csv: Optional[str] = None,
) -> Dict[tuple, dict]:
    """Load LLM-as-judge labels into a `(repo, file_path, blob_hash) → entry` dict.

    Used by `score_from_output_dir` when called with `llm_overlay=...`. The
    returned dict is keyed by `(repo, file_path, blob_hash)` so that the
    same content blob across multiple snapshots resolves to a single label
    (matching the dedup done in `labeling_universe.parquet`).

    Entry shape:
        Success (file labeled):
            {"category": str, "rationale": str, "cut": False, "cut_reason": None}
        Anomaly (file could not be labeled — missing clone, binary blob, etc.):
            {"category": None, "rationale": None,
             "cut": True, "cut_reason": "anomaly_<reason>"}

    Args:
        labels_csv: Path to llm_prelabels_full_universe_*.csv produced by
            `sampling/label_full_universe.py`. Required columns:
            repo, file_path, blob_hash, llm_suggested_category, llm_rationale.
        anomalies_csv: Optional path to the companion anomalies CSV
            (llm_prelabels_full_universe_anomalies.csv). When provided,
            anomaly rows are merged in so the scorer can record a specific
            cut_reason instead of the generic `not_in_llm_csv`.

    Returns:
        Dict[(repo, file_path, blob_hash), entry].
    """
    overlay: Dict[tuple, dict] = {}
    labels = pd.read_csv(labels_csv, dtype={"blob_hash": str})
    # Keep only rows with a real category (drop pure error rows).
    labels = labels[labels["llm_suggested_category"].notna()
                    & (labels["llm_suggested_category"].astype(str).str.len() > 0)]
    for r in labels.itertuples(index=False):
        key = (r.repo, r.file_path, r.blob_hash)
        overlay[key] = {
            "category": r.llm_suggested_category,
            "rationale": getattr(r, "llm_rationale", None),
            "cut": False,
            "cut_reason": None,
        }
    if anomalies_csv:
        anoms = pd.read_csv(anomalies_csv, dtype={"blob_hash": str})
        for r in anoms.itertuples(index=False):
            key = (r.repo, r.file_path, r.blob_hash)
            # Don't clobber a successful label with an anomaly entry (shouldn't
            # happen, but be defensive — labels CSV wins).
            if key in overlay:
                continue
            overlay[key] = {
                "category": None,
                "rationale": None,
                "cut": True,
                "cut_reason": f"anomaly_{getattr(r, 'anomaly', 'unknown')}",
            }
    return overlay


def _apply_llm_overlay(
    classifications: List[FileClassification],
    repo: str,
    snapshot: str,
    overlay: Dict[tuple, dict],
    blob_hash_for_files: Dict[tuple, str],
) -> None:
    """Mutate FileClassification list in place: replace AIME's assigned_category
    with the LLM verdict, or mark the row as cut.

    Cut reasons:
        no_blob_hash   — (repo, snapshot, file_path) missing from
                         blob_hash_for_files (couldn't resolve git blob).
        not_in_llm_csv — blob hash resolved, but no LLM label or anomaly
                         record exists for it (labeling not yet run).
        anomaly_<reason> — blob hash resolved and the anomalies CSV recorded
                           a specific reason (clone_missing, blob_missing,
                           binary, hash_mismatch).
    """
    for fc in classifications:
        fc.aime_assigned_category = fc.assigned_category
        blob_hash = blob_hash_for_files.get((repo, snapshot, fc.artifact_path))
        fc.blob_hash = blob_hash
        if blob_hash is None:
            fc.assigned_category = None
            fc.assigned_maturity_level = None
            fc.llm_cut = True
            fc.llm_cut_reason = "no_blob_hash"
            continue
        entry = overlay.get((repo, fc.artifact_path, blob_hash))
        if entry is None:
            fc.assigned_category = None
            fc.assigned_maturity_level = None
            fc.llm_cut = True
            fc.llm_cut_reason = "not_in_llm_csv"
            continue
        if entry["cut"]:
            fc.assigned_category = None
            fc.assigned_maturity_level = None
            fc.llm_cut = True
            fc.llm_cut_reason = entry["cut_reason"]
            fc.llm_rationale = entry["rationale"]
            continue
        fc.assigned_category = entry["category"]
        fc.assigned_maturity_level = (
            int(CATEGORY_TO_LEVEL[entry["category"]])
            if entry["category"] in CATEGORY_TO_LEVEL else None
        )
        fc.llm_rationale = entry["rationale"]
        fc.llm_cut = False
        fc.llm_cut_reason = None


def score_from_output_dir(
    output_path: str,
    repo_name: str,
    model,
    artifacts_dir: str = "Artifacts",
    task_prefix: str = DEFAULT_TASK_PREFIX,
    config: ScoringConfig = DEFAULT_CONFIG,
    llm_overlay: Optional[Dict[tuple, dict]] = None,
    blob_hash_for_files: Optional[Dict[tuple, str]] = None,
    overlay_repo: Optional[str] = None,
    overlay_snapshot: Optional[str] = None,
    wl_steps: Optional[set] = None,
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
        config: Scoring thresholds for content / path semantic signals.
        llm_overlay: Optional dict from `load_llm_overlay()`. When provided
            together with `blob_hash_for_files`, the AIME `assigned_category`
            is replaced by the LLM verdict (`aime_assigned_category` preserves
            the original). Files without an LLM label are marked `llm_cut=True`
            with a `llm_cut_reason`. Default None → unchanged AIME behavior.
        blob_hash_for_files: Optional dict `(repo, snapshot, file_path) → blob_hash`
            used to look up LLM labels. Required when `llm_overlay` is set.
        overlay_repo: The repo name used as the first element of the
            `blob_hash_for_files` and `llm_overlay` keys. Defaults to
            `Path(output_path).name` (matches the MSRC layout where
            output_path is `.../msrc/<repo>` and repo_name is the snapshot).
        overlay_snapshot: The snapshot used as the second element of
            `blob_hash_for_files` keys. Defaults to `repo_name`.
        wl_steps: Optional whitelist of `discovery_step` values. When set,
            only files whose discovery_step is in this set are classified
            (e.g. the strict AI-artifact tiers {"tool_standard",
            "shared_in_tool_folder", "shared_in_root"} from notebooks 13/14).
            Default None → all discovered files, unchanged behavior.

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
        )

    artifacts_df = pd.read_csv(csv_files[0])

    # Restrict to whitelisted discovery tiers before any classification —
    # dropped rows must contribute no level evidence.
    if wl_steps is not None and not artifacts_df.empty:
        artifacts_df = artifacts_df[
            artifacts_df["discovery_step"].isin(wl_steps)
        ].reset_index(drop=True)

    # Drop project-boilerplate files (README, LICENSE, PR/issue templates, …)
    # before classification — they are not AI artifacts and cause spurious
    # level attribution via cosine noise. Gated by config.filter_boilerplate
    # so sensitivity analyses can opt out and inspect the unfiltered input.
    boilerplate_filtered = 0
    if config.filter_boilerplate and not artifacts_df.empty:
        mask = artifacts_df.apply(
            lambda r: not is_boilerplate(
                str(r.get("artifact_name", "") or ""),
                str(r.get("artifact_path", "") or ""),
                Path(artifacts_dir),
            ),
            axis=1,
        )
        boilerplate_filtered = int((~mask).sum())
        artifacts_df = artifacts_df[mask].reset_index(drop=True)

    # Drop files inside doc/docs/documentation trees before classification —
    # gated by config.ignore_doc_folders, on by default. See ScoringConfig
    # docstring for the empirical motivation.
    doc_folder_filtered = 0
    if config.ignore_doc_folders and not artifacts_df.empty:
        mask = artifacts_df["artifact_path"].fillna("").apply(
            lambda p: not is_in_doc_folder(str(p))
        )
        doc_folder_filtered = int((~mask).sum())
        artifacts_df = artifacts_df[mask].reset_index(drop=True)

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
        model, template_embeddings, task_prefix, config,
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

        if aligned_embeddings:
            aligned_emb_array = np.vstack(aligned_embeddings)
            content_signal = classify_by_content(aligned_emb_array, template_embeddings, config)
        else:
            content_signal = classify_by_content(np.array([]).reshape(0, 768), template_embeddings, config)
    else:
        content_signal = classify_by_content(np.array([]).reshape(0, 768), template_embeddings, config)
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
        artifacts_df, tool_signal, path_signal, content_signal, config,
    )

    # Apply LLM overlay if provided. Replaces AIME's assigned_category with
    # the LLM verdict (preserving the AIME label as aime_assigned_category)
    # and marks files without an LLM label as llm_cut. Both llm_overlay
    # and blob_hash_for_files must be passed together; either alone is a
    # configuration error.
    if (llm_overlay is not None) ^ (blob_hash_for_files is not None):
        raise ValueError(
            "llm_overlay and blob_hash_for_files must be provided together "
            "(got one without the other)."
        )
    if llm_overlay is not None and blob_hash_for_files is not None:
        repo_for_overlay = overlay_repo or Path(output_path).name
        snap_for_overlay = overlay_snapshot or repo_name
        _apply_llm_overlay(
            classifications,
            repo_for_overlay,
            snap_for_overlay,
            llm_overlay,
            blob_hash_for_files,
        )

    # Aggregate to repo-level score
    return aggregate_repo_maturity(
        classifications,
        boilerplate_filtered=boilerplate_filtered,
        doc_folder_filtered=doc_folder_filtered,
        config=config,
    )


def normalize_timeseries_appearance(
    df: pd.DataFrame,
    msrc_root: Path,
    snapshots: List[str],
) -> pd.DataFrame:
    """Enforce first-appearance semantics on a (repo, month) maturity timeseries.

    Three row classes emerge in the output:
      - NA rows (level=NaN, label="Repo Didn't Exist") for (repo, month) pairs
        where month is strictly before the repo's first appearance — i.e. the
        first snapshot for which `{repo}_file_artifacts.csv` exists under
        msrc_root. These replace the phantom-L1 rows the scoring loops emit
        when a snapshot directory is absent.
      - L1 rows backfilled for repos that exist in msrc_root but are entirely
        absent from `df` — typically repos where every collected artifact was
        removed by the AIME boilerplate / doc-folder pre-filter and therefore
        never produced a scored row.
      - Existing rows from `df` for (repo, month) pairs at or after first
        appearance, kept verbatim.

    Repos that have never produced a file_artifacts.csv (collector never
    succeeded on any snapshot) are dropped entirely — there is no signal that
    the repo ever existed during the study window.

    The returned frame has one row per (repo, snapshot) for every MSRC repo
    that appeared at least once, with `level` cast to nullable Int64 so the
    NA rows survive CSV serialization as empty cells.
    """
    msrc_root = Path(msrc_root)
    repos = sorted(p.name for p in msrc_root.iterdir() if p.is_dir())

    appearance: Dict[str, Optional[str]] = {}
    for repo in repos:
        first = None
        for s in snapshots:
            if (msrc_root / repo / s / f"{repo}_file_artifacts.csv").exists():
                first = s
                break
        appearance[repo] = first

    existing = {
        (str(r["repo"]), str(r["month"])): r.to_dict()
        for _, r in df.iterrows()
    }
    payload_cols = [c for c in df.columns if c not in ("repo", "month")]

    l1_defaults = {
        "level": 1,
        "label": MATURITY_LABELS[MaturityLevel.L1],
        "confidence": 1.0,
        "artifacts": 0,
        "tools": "",
        "l2_primary": 0,
        "l3_primary": 0,
        "l4_primary": 0,
        "has_leveled_tool_attribution": False,
        "_note": "no AI artifacts (all filtered or absent)",
    }
    na_defaults = {
        "level": pd.NA,
        "label": "Repo Didn't Exist",
        "_note": "repo did not exist at this snapshot",
    }

    def _row(repo: str, snap: str, defaults: Dict) -> Dict:
        row = {c: pd.NA for c in payload_cols}
        row["repo"] = repo
        row["month"] = snap
        for k, v in defaults.items():
            if k in row:
                row[k] = v
        return row

    out_rows: List[Dict] = []
    for repo in repos:
        first_seen = appearance[repo]
        if first_seen is None:
            continue
        for snap in snapshots:
            if snap < first_seen:
                out_rows.append(_row(repo, snap, na_defaults))
            elif (repo, snap) in existing:
                out_rows.append(existing[(repo, snap)])
            else:
                out_rows.append(_row(repo, snap, l1_defaults))

    out = pd.DataFrame(out_rows, columns=list(df.columns))
    if "level" in out.columns:
        out["level"] = pd.to_numeric(out["level"], errors="coerce").astype("Int64")
    return out.sort_values(["repo", "month"]).reset_index(drop=True)
