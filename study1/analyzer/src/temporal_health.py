"""Temporal health analysis for AIME maturity assessments.

Joins per-artifact git commit history (timeseries CSV) with AIME file
classifications to produce lifecycle classifications, author diversity
metrics, and per-category health verdicts.

ALL classified artifacts are included: those with git history get a lifecycle
classification (steady/burst/set-and-forget/abandoned); those without get
"no-history".
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


# ============================================================================
# Constants
# ============================================================================

# Category tiers for health verdict rules
GROUNDING_CATEGORIES = {"rules", "architecture", "configuration"}
AGENTIC_CATEGORIES = {"agents", "commands", "skills"}
FLOW_CATEGORIES = {"flows"}
SESSION_CATEGORIES = {"session-logs"}
CODE_STYLE_CATEGORIES = {"code-style"}

# Verdict matrix: tier -> lifecycle -> verdict
# Possible verdicts: "healthy", "warning", "concern"
_VERDICT_MATRIX = {
    "grounding": {
        "set-and-forget": "concern",
        "burst": "warning",
        "abandoned": "concern",
        "steady": "healthy",
    },
    "agentic": {
        "set-and-forget": "healthy",
        "burst": "healthy",
        "abandoned": "warning",
        "steady": "healthy",
    },
    "flows": {
        "set-and-forget": "healthy",
        "burst": "healthy",
        "abandoned": "concern",
        "steady": "healthy",
    },
    "session-logs": {
        "set-and-forget": "healthy",
        "burst": "healthy",
        "abandoned": "healthy",
        "steady": "healthy",
    },
    "code-style": {
        "set-and-forget": "healthy",
        "burst": "healthy",
        "abandoned": "healthy",
        "steady": "healthy",
    },
}


def _category_tier(category: str) -> str:
    """Map a category name to its tier for verdict lookup."""
    if category in GROUNDING_CATEGORIES:
        return "grounding"
    if category in CODE_STYLE_CATEGORIES:
        return "code-style"
    if category in AGENTIC_CATEGORIES:
        return "agentic"
    if category in FLOW_CATEGORIES:
        return "flows"
    if category in SESSION_CATEGORIES:
        return "session-logs"
    return "grounding"  # fallback


# ============================================================================
# Dataclass
# ============================================================================

@dataclass
class TemporalHealth:
    """Result of temporal health analysis."""
    has_timeseries: bool
    artifact_lifecycles: pd.DataFrame  # per-artifact lifecycle info (ALL artifacts)
    category_summaries: pd.DataFrame   # per-category aggregation
    health_verdicts: List[Dict]        # per-category verdict dicts
    author_diversity: Dict[str, int]   # category -> unique author count
    artifact_count: int = 0            # artifacts with commit history
    total_classified: int = 0          # total classified artifacts
    earliest_date: Optional[pd.Timestamp] = None
    horizon_date: Optional[pd.Timestamp] = None


# ============================================================================
# Functions
# ============================================================================

def load_timeseries(repo_dir: str, repo_name: str = "") -> Optional[pd.DataFrame]:
    """Search for *_artifact_timeseries.csv in repo_dir (and one level deeper).

    Parses commit_date to datetime. Returns None if not found or empty.
    """
    search_dir = Path(repo_dir)
    candidates = list(search_dir.glob("*_artifact_timeseries.csv"))

    # Also search one level deeper (handles nested layout)
    for child in search_dir.iterdir():
        if child.is_dir():
            candidates.extend(child.glob("*_artifact_timeseries.csv"))

    if not candidates:
        return None

    # Use the first match (usually only one)
    csv_path = candidates[0]
    df = pd.read_csv(csv_path)

    if df.empty:
        return None

    if "commit_date" in df.columns:
        df["commit_date"] = pd.to_datetime(df["commit_date"], utc=True, errors="coerce")

    return df


def classify_artifact_lifecycle(
    commits: pd.DataFrame,
    horizon_date: pd.Timestamp,
    burst_window_days: int = 30,
    abandonment_months: int = 6,
) -> str:
    """Classify a single artifact's commit history into a lifecycle type.

    Args:
        commits: DataFrame with commit_date column for one artifact.
        horizon_date: Reference date (usually last commit across all artifacts).
        burst_window_days: Max days span to be classified as "burst".
        abandonment_months: Months before horizon to be classified as "abandoned".

    Returns:
        One of: "set-and-forget", "burst", "abandoned", "steady".
    """
    if len(commits) == 0:
        return "set-and-forget"

    if len(commits) == 1:
        return "set-and-forget"

    dates = commits["commit_date"].dropna().sort_values()
    if len(dates) < 2:
        return "set-and-forget"

    first, last = dates.iloc[0], dates.iloc[-1]
    span_days = (last - first).days

    # All commits within burst window
    if span_days <= burst_window_days:
        return "burst"

    # Last commit too old relative to horizon
    abandonment_cutoff = horizon_date - pd.DateOffset(months=abandonment_months)
    if last < abandonment_cutoff:
        return "abandoned"

    return "steady"


def _build_health_verdicts(
    category_summaries: pd.DataFrame,
) -> List[Dict]:
    """Build per-category health verdicts based on dominant lifecycle."""
    verdicts = []
    for _, row in category_summaries.iterrows():
        category = row["category"]
        tier = _category_tier(category)

        # Find dominant lifecycle (only from artifacts with history)
        lifecycle_cols = ["steady", "burst", "set-and-forget", "abandoned"]
        counts = {lc: row.get(lc, 0) for lc in lifecycle_cols}
        with_history = sum(counts.values())

        if with_history == 0:
            # No artifacts have commit history — can't assess
            verdicts.append({
                "category": category,
                "tier": tier,
                "dominant_lifecycle": "no-history",
                "verdict": "unknown",
                "message": f"{category}: no commit history available — cannot assess temporal health.",
            })
            continue

        dominant = max(counts, key=counts.get)
        verdict = _VERDICT_MATRIX.get(tier, {}).get(dominant, "healthy")

        messages = {
            ("grounding", "concern", "set-and-forget"):
                f"{category}: grounding artifacts were created once and never updated — they may be stale.",
            ("grounding", "concern", "abandoned"):
                f"{category}: grounding artifacts haven't been updated in 6+ months — review for staleness.",
            ("grounding", "warning", "burst"):
                f"{category}: grounding artifacts were updated in a short burst — monitor for ongoing maintenance.",
            ("grounding", "healthy", "steady"):
                f"{category}: grounding artifacts are actively maintained.",
            ("agentic", "warning", "abandoned"):
                f"{category}: agentic artifacts haven't been updated in 6+ months.",
            ("agentic", "healthy", "set-and-forget"):
                f"{category}: agentic artifacts are stable — normal for mature agent definitions.",
            ("agentic", "healthy", "burst"):
                f"{category}: agentic artifacts had focused development — normal pattern.",
            ("agentic", "healthy", "steady"):
                f"{category}: agentic artifacts are actively maintained.",
            ("flows", "concern", "abandoned"):
                f"{category}: flow artifacts haven't been updated in 6+ months.",
            ("flows", "healthy", "set-and-forget"):
                f"{category}: flow artifacts are stable.",
            ("flows", "healthy", "burst"):
                f"{category}: flow artifacts had focused development.",
            ("flows", "healthy", "steady"):
                f"{category}: flow artifacts are actively maintained.",
            ("session-logs", "healthy", "burst"):
                f"{category}: session logs show burst activity — expected pattern.",
            ("session-logs", "healthy", "steady"):
                f"{category}: session logs are regularly updated.",
            ("session-logs", "healthy", "set-and-forget"):
                f"{category}: session logs are stable.",
            ("session-logs", "healthy", "abandoned"):
                f"{category}: session logs haven't been updated recently.",
            ("code-style", "healthy", "set-and-forget"):
                f"{category}: coding style artifacts are stable — expected for style guides and linting configs.",
            ("code-style", "healthy", "burst"):
                f"{category}: coding style artifacts had focused setup.",
            ("code-style", "healthy", "abandoned"):
                f"{category}: coding style artifacts are stable — no updates needed.",
            ("code-style", "healthy", "steady"):
                f"{category}: coding style artifacts are actively maintained.",
        }

        message = messages.get(
            (tier, verdict, dominant),
            f"{category}: {dominant} pattern detected ({verdict}).",
        )

        verdicts.append({
            "category": category,
            "tier": tier,
            "dominant_lifecycle": dominant,
            "verdict": verdict,
            "message": message,
        })

    return verdicts


def analyze_temporal_health(
    repo_dir: str,
    file_classifications: pd.DataFrame,
    repo_name: str = "",
) -> TemporalHealth:
    """Main entry point for temporal health analysis.

    Loads timeseries CSV, joins with file classifications, computes all metrics.
    Returns TemporalHealth(has_timeseries=False, ...) if no CSV found.
    """
    empty = TemporalHealth(
        has_timeseries=False,
        artifact_lifecycles=pd.DataFrame(),
        category_summaries=pd.DataFrame(),
        health_verdicts=[],
        author_diversity={},
    )

    ts = load_timeseries(repo_dir, repo_name)
    if ts is None or ts.empty:
        return empty

    if file_classifications is None or file_classifications.empty:
        return empty

    # Join timeseries with classifications on artifact_path
    fc_cols = ["artifact_path", "assigned_category"]
    fc_subset = file_classifications[fc_cols].drop_duplicates(subset=["artifact_path"])
    merged = ts.merge(fc_subset, on="artifact_path", how="left")

    # Drop rows without a category
    merged = merged.dropna(subset=["assigned_category"])
    if merged.empty:
        return empty

    # Date range from commit history
    earliest_date = merged["commit_date"].min()
    horizon_date = merged["commit_date"].max()

    # 1. Classify artifacts WITH commit history
    lifecycle_rows = []
    for artifact_path, group in merged.groupby("artifact_path"):
        lifecycle = classify_artifact_lifecycle(group, horizon_date)
        category = group["assigned_category"].iloc[0]
        authors = group["author_hash"].nunique() if "author_hash" in group.columns else 0
        lifecycle_rows.append({
            "artifact_path": artifact_path,
            "category": category,
            "lifecycle": lifecycle,
            "commit_count": len(group),
            "first_commit": group["commit_date"].min(),
            "last_commit": group["commit_date"].max(),
            "authors": authors,
        })

    n_with_history = len(lifecycle_rows)

    # 2. Add ALL classified artifacts without timeseries as "no-history"
    paths_with_history = {r["artifact_path"] for r in lifecycle_rows}
    for _, row in file_classifications.iterrows():
        path = row["artifact_path"]
        category = row.get("assigned_category")
        if path not in paths_with_history and pd.notna(category):
            lifecycle_rows.append({
                "artifact_path": path,
                "category": category,
                "lifecycle": "no-history",
                "commit_count": 0,
                "first_commit": pd.NaT,
                "last_commit": pd.NaT,
                "authors": 0,
            })

    artifact_lifecycles = pd.DataFrame(lifecycle_rows)

    # 3. Build category summaries (includes no-history column)
    cat_summary_rows = []
    for category, cat_group in artifact_lifecycles.groupby("category"):
        lc_counts = cat_group["lifecycle"].value_counts()
        cat_summary_rows.append({
            "category": category,
            "total_artifacts": len(cat_group),
            "total_commits": int(cat_group["commit_count"].sum()),
            "unique_authors": int(cat_group[cat_group["authors"] > 0]["authors"].max())
                if (cat_group["authors"] > 0).any() else 0,
            "steady": int(lc_counts.get("steady", 0)),
            "burst": int(lc_counts.get("burst", 0)),
            "set-and-forget": int(lc_counts.get("set-and-forget", 0)),
            "abandoned": int(lc_counts.get("abandoned", 0)),
            "no-history": int(lc_counts.get("no-history", 0)),
        })

    category_summaries = pd.DataFrame(cat_summary_rows)

    # 4. Author diversity from all available commit data
    author_diversity = {}
    for category, cat_group in merged.groupby("assigned_category"):
        if "author_hash" in cat_group.columns:
            author_diversity[category] = cat_group["author_hash"].nunique()
        else:
            author_diversity[category] = 0

    # 5. Build health verdicts
    health_verdicts = _build_health_verdicts(category_summaries)

    return TemporalHealth(
        has_timeseries=True,
        artifact_lifecycles=artifact_lifecycles,
        category_summaries=category_summaries,
        health_verdicts=health_verdicts,
        author_diversity=author_diversity,
        artifact_count=n_with_history,
        total_classified=len(file_classifications),
        earliest_date=earliest_date,
        horizon_date=horizon_date,
    )
