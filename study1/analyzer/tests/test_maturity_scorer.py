"""Tests for maturity_scorer module.

Uses mocked SentenceTransformer models to avoid actual model downloads.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

from src.maturity_scorer import (
    # Constants
    MaturityLevel,
    MATURITY_LABELS,
    CATEGORY_TEMPLATES,
    CATEGORY_NAMES,
    CATEGORY_TO_LEVEL,
    HYBRID_THRESHOLD,
    # Data classes
    FileClassification,
    CoherenceFlag,
    MaturityScore,
    # Functions
    embed_category_templates,
    path_to_semantic_tokens,
    classify_by_content,
    classify_by_path,
    classify_by_tool_detection,
    combine_signals,
    aggregate_repo_maturity,
    build_artifacts_map,
    build_tool_category_matrix,
    generate_report,
    _normalize_category,
    _check_coherence,
    _compute_confidence,
    _generate_recommendations,
)


# ============================================================================
# Helpers
# ============================================================================

def _mock_model(dim=768):
    """Create a mock SentenceTransformer that returns deterministic embeddings."""
    model = Mock()
    model.get_sentence_embedding_dimension.return_value = dim
    model.tokenizer = Mock()
    model.tokenizer.encode.return_value = list(range(100))

    def _encode(texts, **kwargs):
        if isinstance(texts, str):
            np.random.seed(hash(texts[:20]) % 2**31)
            return np.random.randn(dim).astype(np.float32)
        return np.random.randn(len(texts), dim).astype(np.float32)

    model.encode.side_effect = _encode
    return model


def _make_classification(
    file_id="f1",
    artifact_path=".claude/commands/test.md",
    tool_name="claude-code",
    assigned_category="commands",
    content_primary="commands",
    path_primary="commands",
    signals_agree=True,
    categories_within_threshold=None,
    content_scores=None,
) -> FileClassification:
    """Helper to build a FileClassification."""
    level = CATEGORY_TO_LEVEL.get(assigned_category)
    fc = FileClassification(
        file_id=file_id,
        artifact_path=artifact_path,
        tool_name=tool_name,
        discovery_step="tool_standard",
        assigned_category=assigned_category,
        assigned_maturity_level=int(level) if level else None,
        content_primary=content_primary,
        content_primary_score=0.7,
        content_secondary="rules",
        content_secondary_score=0.65,
        path_primary=path_primary,
        path_primary_score=0.7,
        path_secondary="rules",
        path_secondary_score=0.6,
        signals_agree=signals_agree,
        hybrid_score=len(categories_within_threshold) if categories_within_threshold else 1,
        categories_within_threshold=categories_within_threshold or [assigned_category],
        content_scores=content_scores or {},
    )
    return fc


# ============================================================================
# Test Constants
# ============================================================================

class TestConstants:
    """Tests for module-level constants."""

    def test_maturity_levels(self):
        assert MaturityLevel.L1 == 1
        assert MaturityLevel.L4 == 4

    def test_category_templates_count(self):
        assert len(CATEGORY_TEMPLATES) == 9

    def test_category_names_ordered(self):
        assert CATEGORY_NAMES == list(CATEGORY_TEMPLATES.keys())

    def test_all_categories_mapped(self):
        for cat in CATEGORY_NAMES:
            assert cat in CATEGORY_TO_LEVEL

    def test_level_mappings(self):
        assert CATEGORY_TO_LEVEL["rules"] == MaturityLevel.L2
        assert CATEGORY_TO_LEVEL["configuration"] == MaturityLevel.L2
        assert CATEGORY_TO_LEVEL["architecture"] == MaturityLevel.L2
        assert CATEGORY_TO_LEVEL["code-style"] == MaturityLevel.L2
        assert CATEGORY_TO_LEVEL["agents"] == MaturityLevel.L3
        assert CATEGORY_TO_LEVEL["commands"] == MaturityLevel.L3
        assert CATEGORY_TO_LEVEL["skills"] == MaturityLevel.L3
        assert CATEGORY_TO_LEVEL["flows"] == MaturityLevel.L4
        assert CATEGORY_TO_LEVEL["session-logs"] == MaturityLevel.L4

    def test_maturity_labels(self):
        assert MATURITY_LABELS[MaturityLevel.L1] == "Ad Hoc"
        assert MATURITY_LABELS[MaturityLevel.L4] == "Agentic Orchestration"


# ============================================================================
# Test Data Classes
# ============================================================================

class TestFileClassification:
    """Tests for FileClassification dataclass."""

    def test_to_dict(self):
        fc = _make_classification(categories_within_threshold=["commands", "rules"])
        d = fc.to_dict()
        assert d["file_id"] == "f1"
        assert d["assigned_category"] == "commands"
        assert d["categories_within_threshold"] == "commands+rules"
        assert d["signals_agree"] is True

    def test_defaults(self):
        fc = FileClassification(
            file_id="x", artifact_path="test.md",
            tool_name="unknown", discovery_step="non_standard_root",
        )
        assert fc.tool_category is None
        assert fc.content_primary is None
        assert fc.hybrid_score == 1
        assert fc.categories_within_threshold == []


class TestMaturityScore:
    """Tests for MaturityScore dataclass."""

    def test_to_dict_serializable(self):
        score = MaturityScore(
            overall_level=3,
            overall_label="Agent-Augmented",
            confidence=0.85,
            tools_detected=["claude-code"],
            artifact_count=10,
            level_evidence={2: {"primary": 3, "secondary": 5}},
            category_counts={"commands": 5, "rules": 3},
            coherence_flags=[CoherenceFlag("test", "green", "ok")],
            recommendations=["Do more"],
        )
        d = score.to_dict()
        assert d["overall_level"] == 3
        assert d["confidence"] == 0.85
        assert d["coherence_flags"][0]["status"] == "green"


# ============================================================================
# Test Category Template Embedding
# ============================================================================

class TestEmbedCategoryTemplates:
    """Tests for embed_category_templates function."""

    def test_returns_correct_shape(self):
        model = _mock_model()
        result = embed_category_templates(model)
        assert result.shape == (9, 768)

    def test_calls_encode_for_each_category(self):
        model = _mock_model()
        embed_category_templates(model)
        assert model.encode.call_count == 9

    def test_applies_task_prefix(self):
        model = _mock_model()
        embed_category_templates(model, task_prefix="clustering")
        first_call = model.encode.call_args_list[0]
        assert first_call[0][0].startswith("clustering: ")


# ============================================================================
# Test Path Tokenization
# ============================================================================

class TestPathToSemanticTokens:
    """Tests for path_to_semantic_tokens function."""

    def test_basic_path(self):
        assert path_to_semantic_tokens(".claude/commands/sparc.md") == "claude commands sparc"

    def test_removes_extension(self):
        result = path_to_semantic_tokens("test.json")
        assert ".json" not in result

    def test_replaces_separators(self):
        result = path_to_semantic_tokens("a/b/c.md")
        assert "/" not in result
        assert result == "a b c"

    def test_removes_leading_dots(self):
        result = path_to_semantic_tokens(".cursor/rules/test.mdc")
        assert result.startswith("cursor")

    def test_replaces_underscores_hyphens(self):
        result = path_to_semantic_tokens("my_file-name.md")
        assert "_" not in result
        assert "-" not in result
        assert result == "my file name"

    def test_collapses_whitespace(self):
        result = path_to_semantic_tokens(".claude/  /test.md")
        assert "  " not in result

    def test_nested_path(self):
        result = path_to_semantic_tokens("marketplace/plugins/web/agents/rush-runner.md")
        assert result == "marketplace plugins web agents rush runner"


# ============================================================================
# Test Signal 1: Tool Detection
# ============================================================================

class TestClassifyByToolDetection:
    """Tests for classify_by_tool_detection function."""

    @patch("src.maturity_scorer.load_shared_config")
    @patch("src.maturity_scorer.load_json_configs")
    def test_known_tool_match(self, mock_load, mock_shared):
        """Known tool with matching pattern resolves category."""
        from src.data_models import ToolConfig, ArtifactPattern, DiscoveryMethod, ArtifactStatus

        pattern = ArtifactPattern(
            pattern="CLAUDE.md",
            type="file",
            description="Test",
            file_type="markdown",
            status=ArtifactStatus.STABLE,
            is_standard=True,
            artifact_category="instructions",
            scope="project",
            discovery_method=DiscoveryMethod.EXACT_PATH,
            exact_path="CLAUDE.md",
        )
        tool = ToolConfig(
            tool_name="claude-code",
            artifact_patterns=[pattern],
        )
        mock_load.return_value = {"claude-code": tool}
        mock_shared.return_value = None

        df = pd.DataFrame([{
            "file_id": "f1",
            "artifact_path": "CLAUDE.md",
            "tool_name": "claude-code",
        }])

        result = classify_by_tool_detection(df, "Artifacts")
        assert result.iloc[0]["tool_category"] == "instructions"

    @patch("src.maturity_scorer.load_shared_config")
    @patch("src.maturity_scorer.load_json_configs")
    def test_unknown_tool_no_match(self, mock_load, mock_shared):
        """Unknown tool returns None category."""
        mock_load.return_value = {}
        mock_shared.return_value = None

        df = pd.DataFrame([{
            "file_id": "f1",
            "artifact_path": "README.md",
            "tool_name": "unknown",
        }])

        result = classify_by_tool_detection(df, "Artifacts")
        assert result.iloc[0]["tool_category"] is None

    @patch("src.maturity_scorer.load_shared_config")
    @patch("src.maturity_scorer.load_json_configs")
    def test_glob_pattern_match(self, mock_load, mock_shared):
        """Glob patterns match correctly."""
        from src.data_models import ToolConfig, ArtifactPattern, DiscoveryMethod, ArtifactStatus

        pattern = ArtifactPattern(
            pattern=".claude/commands/*.md",
            type="directory",
            description="Test",
            file_type="markdown",
            status=ArtifactStatus.STABLE,
            is_standard=True,
            artifact_category="commands",
            scope="project",
            discovery_method=DiscoveryMethod.GLOB,
            glob_pattern=".claude/commands/**/*.md",
            path_prefix=".claude/commands/",
            recursive=True,
        )
        tool = ToolConfig(
            tool_name="claude-code",
            artifact_patterns=[pattern],
        )
        mock_load.return_value = {"claude-code": tool}
        mock_shared.return_value = None

        df = pd.DataFrame([{
            "file_id": "f1",
            "artifact_path": ".claude/commands/sparc.md",
            "tool_name": "claude-code",
        }])

        result = classify_by_tool_detection(df, "Artifacts")
        assert result.iloc[0]["tool_category"] == "commands"


# ============================================================================
# Test Signal 2: Path Classification
# ============================================================================

class TestClassifyByPath:
    """Tests for classify_by_path function."""

    def test_returns_correct_columns(self):
        model = _mock_model()
        template_embs = np.random.randn(9, 768).astype(np.float32)
        result = classify_by_path(
            [".claude/commands/test.md"],
            model, template_embs,
        )
        assert "path_primary" in result.columns
        assert "path_primary_score" in result.columns
        assert "path_secondary" in result.columns
        assert "path_margin" in result.columns

    def test_empty_input(self):
        model = _mock_model()
        template_embs = np.random.randn(9, 768).astype(np.float32)
        result = classify_by_path([], model, template_embs)
        assert len(result) == 0

    def test_result_length_matches_input(self):
        model = _mock_model()
        template_embs = np.random.randn(9, 768).astype(np.float32)
        paths = ["a.md", "b.md", "c.md"]
        result = classify_by_path(paths, model, template_embs)
        assert len(result) == 3

    def test_primary_is_valid_category(self):
        model = _mock_model()
        template_embs = np.random.randn(9, 768).astype(np.float32)
        result = classify_by_path(["test.md"], model, template_embs)
        assert result.iloc[0]["path_primary"] in CATEGORY_NAMES


# ============================================================================
# Test Signal 3: Content Classification
# ============================================================================

class TestClassifyByContent:
    """Tests for classify_by_content function."""

    def test_returns_correct_columns(self):
        file_embs = np.random.randn(3, 768).astype(np.float32)
        template_embs = np.random.randn(9, 768).astype(np.float32)
        result = classify_by_content(file_embs, template_embs)
        assert "content_primary" in result.columns
        assert "content_primary_score" in result.columns
        assert "hybrid_score" in result.columns
        for cat in CATEGORY_NAMES:
            assert f"content_{cat}" in result.columns

    def test_empty_input(self):
        file_embs = np.array([]).reshape(0, 768)
        template_embs = np.random.randn(9, 768).astype(np.float32)
        result = classify_by_content(file_embs, template_embs)
        assert len(result) == 0

    def test_hybrid_score_at_least_1(self):
        file_embs = np.random.randn(5, 768).astype(np.float32)
        template_embs = np.random.randn(9, 768).astype(np.float32)
        result = classify_by_content(file_embs, template_embs)
        assert (result["hybrid_score"] >= 1).all()

    def test_primary_has_highest_score(self):
        file_embs = np.random.randn(3, 768).astype(np.float32)
        template_embs = np.random.randn(9, 768).astype(np.float32)
        result = classify_by_content(file_embs, template_embs)
        for _, row in result.iterrows():
            primary_cat = row["content_primary"]
            primary_score = row["content_primary_score"]
            for cat in CATEGORY_NAMES:
                assert row[f"content_{cat}"] <= primary_score + 1e-6


# ============================================================================
# Test Normalize Category
# ============================================================================

class TestNormalizeCategory:

    def test_standard_category(self):
        assert _normalize_category("rules") == "rules"
        assert _normalize_category("agents") == "agents"

    def test_instructions_maps_to_rules(self):
        assert _normalize_category("instructions") == "rules"

    def test_unknown_maps_to_none(self):
        assert _normalize_category("unknown") is None

    def test_none_input(self):
        assert _normalize_category(None) is None

    def test_unrecognized_maps_to_none(self):
        assert _normalize_category("nonexistent") is None


# ============================================================================
# Test Signal Combination
# ============================================================================

class TestCombineSignals:
    """Tests for combine_signals function."""

    def test_basic_combination(self):
        artifacts_df = pd.DataFrame([{
            "file_id": "f1",
            "artifact_path": ".claude/commands/test.md",
            "tool_name": "claude-code",
            "discovery_step": "tool_standard",
        }])
        tool_signal = pd.DataFrame([{"file_id": "f1", "tool_category": "commands"}])
        path_signal = pd.DataFrame([{
            "path_primary": "commands",
            "path_primary_score": 0.8,
            "path_secondary": "rules",
            "path_secondary_score": 0.6,
            "path_margin": 0.2,
        }])
        content_signal = pd.DataFrame([{
            "content_primary": "commands",
            "content_primary_score": 0.75,
            "content_secondary": "skills",
            "content_secondary_score": 0.70,
            "content_margin": 0.05,
            "hybrid_score": 2,
            "categories_within_threshold": "commands+skills",
            **{f"content_{cat}": 0.5 for cat in CATEGORY_NAMES},
        }])

        result = combine_signals(artifacts_df, tool_signal, path_signal, content_signal)
        assert len(result) == 1
        fc = result[0]
        assert fc.assigned_category == "commands"
        assert fc.assigned_maturity_level == 3

    def test_tool_category_takes_priority(self):
        """When tool_category is known, it overrides other signals."""
        artifacts_df = pd.DataFrame([{
            "file_id": "f1",
            "artifact_path": "CLAUDE.md",
            "tool_name": "claude-code",
            "discovery_step": "tool_standard",
        }])
        # Tool says "instructions" → normalized to "rules"
        tool_signal = pd.DataFrame([{"file_id": "f1", "tool_category": "instructions"}])
        path_signal = pd.DataFrame([{
            "path_primary": "agents",
            "path_primary_score": 0.7,
            "path_secondary": "rules",
            "path_secondary_score": 0.6,
            "path_margin": 0.1,
        }])
        content_signal = pd.DataFrame([{
            "content_primary": "agents",
            "content_primary_score": 0.72,
            "content_secondary": "rules",
            "content_secondary_score": 0.68,
            "content_margin": 0.04,
            "hybrid_score": 1,
            "categories_within_threshold": "agents",
            **{f"content_{cat}": 0.5 for cat in CATEGORY_NAMES},
        }])

        result = combine_signals(artifacts_df, tool_signal, path_signal, content_signal)
        fc = result[0]
        assert fc.assigned_category == "rules"
        assert fc.assigned_maturity_level == 2

    def test_agreement_used_when_no_tool(self):
        """When path and content agree and tool is unknown, use agreed category."""
        artifacts_df = pd.DataFrame([{
            "file_id": "f1",
            "artifact_path": "agents/helper.md",
            "tool_name": "unknown",
            "discovery_step": "non_standard_other",
        }])
        tool_signal = pd.DataFrame([{"file_id": "f1", "tool_category": None}])
        path_signal = pd.DataFrame([{
            "path_primary": "agents",
            "path_primary_score": 0.7,
            "path_secondary": "skills",
            "path_secondary_score": 0.6,
            "path_margin": 0.1,
        }])
        content_signal = pd.DataFrame([{
            "content_primary": "agents",
            "content_primary_score": 0.75,
            "content_secondary": "skills",
            "content_secondary_score": 0.70,
            "content_margin": 0.05,
            "hybrid_score": 1,
            "categories_within_threshold": "agents",
            **{f"content_{cat}": 0.5 for cat in CATEGORY_NAMES},
        }])

        result = combine_signals(artifacts_df, tool_signal, path_signal, content_signal)
        fc = result[0]
        assert fc.assigned_category == "agents"
        assert fc.signals_agree is True

    def test_content_fallback(self):
        """When signals disagree and tool is unknown, content_primary wins."""
        artifacts_df = pd.DataFrame([{
            "file_id": "f1",
            "artifact_path": "docs/setup.md",
            "tool_name": "unknown",
            "discovery_step": "non_standard_root",
        }])
        tool_signal = pd.DataFrame([{"file_id": "f1", "tool_category": None}])
        path_signal = pd.DataFrame([{
            "path_primary": "architecture",
            "path_primary_score": 0.6,
            "path_secondary": "rules",
            "path_secondary_score": 0.55,
            "path_margin": 0.05,
        }])
        content_signal = pd.DataFrame([{
            "content_primary": "rules",
            "content_primary_score": 0.7,
            "content_secondary": "architecture",
            "content_secondary_score": 0.65,
            "content_margin": 0.05,
            "hybrid_score": 2,
            "categories_within_threshold": "rules+architecture",
            **{f"content_{cat}": 0.5 for cat in CATEGORY_NAMES},
        }])

        result = combine_signals(artifacts_df, tool_signal, path_signal, content_signal)
        fc = result[0]
        assert fc.assigned_category == "rules"
        assert fc.signals_agree is False


# ============================================================================
# Test Coherence Checks
# ============================================================================

class TestCoherenceChecks:
    """Tests for _check_coherence function."""

    def test_progressive_adoption_all_green(self):
        flags = _check_coherence(
            {2: 5, 3: 10, 4: 2},
            {2: 3, 3: 1, 4: 0},
        )
        statuses = {f.check: f.status for f in flags}
        assert statuses.get("L2 foundation") == "green"
        assert statuses.get("L3 builds on L2") == "green"
        assert statuses.get("L4 builds on L3") == "green"

    def test_l3_without_l2_primary(self):
        flags = _check_coherence(
            {2: 0, 3: 10, 4: 0},
            {2: 5, 3: 0, 4: 0},
        )
        statuses = {f.check: f.status for f in flags}
        assert statuses.get("L2 foundation") == "yellow"
        assert statuses.get("L3 without L2") == "yellow"

    def test_l3_without_any_l2(self):
        flags = _check_coherence(
            {2: 0, 3: 10, 4: 0},
            {2: 0, 3: 0, 4: 0},
        )
        statuses = {f.check: f.status for f in flags}
        assert statuses.get("L2 foundation") == "red"
        assert statuses.get("L3 without L2") == "red"

    def test_l4_without_l3(self):
        flags = _check_coherence(
            {2: 5, 3: 0, 4: 2},
            {2: 0, 3: 0, 4: 0},
        )
        check_map = {f.check: f.status for f in flags}
        assert check_map.get("L4 without L3") == "red"

    def test_no_artifacts(self):
        """No artifacts produces no flags."""
        flags = _check_coherence(
            {2: 0, 3: 0, 4: 0},
            {2: 0, 3: 0, 4: 0},
        )
        assert len(flags) == 0


# ============================================================================
# Test Confidence
# ============================================================================

class TestComputeConfidence:

    def test_l1_always_1(self):
        """L1 (no artifacts) is always confidence 1.0."""
        c = _compute_confidence(1, 0, 0.0, [])
        assert c == 1.0

    def test_high_artifacts_high_agreement(self):
        c = _compute_confidence(3, 25, 0.9, [])
        assert c > 0.8

    def test_red_flags_reduce_confidence(self):
        flags = [CoherenceFlag("test", "red", "bad")]
        c_with_flag = _compute_confidence(3, 10, 0.7, flags)
        c_without_flag = _compute_confidence(3, 10, 0.7, [])
        assert c_with_flag < c_without_flag

    def test_bounded_0_1(self):
        c = _compute_confidence(3, 1, 0.0, [
            CoherenceFlag("a", "red", "x"),
            CoherenceFlag("b", "red", "y"),
            CoherenceFlag("c", "red", "z"),
        ])
        assert 0.0 <= c <= 1.0


# ============================================================================
# Test Aggregate Repo Maturity
# ============================================================================

class TestAggregateRepoMaturity:

    def test_empty_classifications_returns_l1(self):
        score = aggregate_repo_maturity([])
        assert score.overall_level == 1
        assert score.overall_label == "Ad Hoc"
        assert score.confidence == 1.0

    def test_l2_with_rules(self):
        fcs = [
            _make_classification(file_id="f1", assigned_category="rules",
                                content_primary="rules", path_primary="rules"),
            _make_classification(file_id="f2", assigned_category="configuration",
                                content_primary="configuration", path_primary="configuration"),
        ]
        score = aggregate_repo_maturity(fcs)
        assert score.overall_level == 2

    def test_l3_with_agents_and_l2(self):
        fcs = [
            _make_classification(file_id="f1", assigned_category="rules",
                                content_primary="rules", path_primary="rules"),
            _make_classification(file_id="f2", assigned_category="agents",
                                content_primary="agents", path_primary="agents"),
            _make_classification(file_id="f3", assigned_category="commands",
                                content_primary="commands", path_primary="commands"),
        ]
        score = aggregate_repo_maturity(fcs)
        assert score.overall_level == 3

    def test_l4_with_flows(self):
        """L4 artifacts present → repo is L4."""
        fcs = [
            _make_classification(file_id="f1", assigned_category="agents",
                                content_primary="agents", path_primary="agents"),
            _make_classification(file_id="f2", assigned_category="flows",
                                content_primary="flows", path_primary="flows"),
        ]
        score = aggregate_repo_maturity(fcs)
        assert score.overall_level == 4

    def test_tools_detected(self):
        fcs = [
            _make_classification(file_id="f1", tool_name="claude-code"),
            _make_classification(file_id="f2", tool_name="cursor"),
            _make_classification(file_id="f3", tool_name="unknown"),
        ]
        score = aggregate_repo_maturity(fcs)
        assert "claude-code" in score.tools_detected
        assert "cursor" in score.tools_detected
        assert "unknown" not in score.tools_detected

    def test_level_evidence_tracks_primary_and_secondary(self):
        fcs = [
            _make_classification(
                file_id="f1", assigned_category="agents",
                categories_within_threshold=["agents", "rules"],
            ),
        ]
        score = aggregate_repo_maturity(fcs)
        assert score.level_evidence[3]["primary"] == 1
        # "rules" is secondary → L2 secondary
        assert score.level_evidence[2]["secondary"] == 1

    def test_category_counts(self):
        fcs = [
            _make_classification(file_id="f1", assigned_category="commands"),
            _make_classification(file_id="f2", assigned_category="commands"),
            _make_classification(file_id="f3", assigned_category="rules"),
        ]
        score = aggregate_repo_maturity(fcs)
        assert score.category_counts["commands"] == 2
        assert score.category_counts["rules"] == 1

    def test_recommendations_for_l1(self):
        score = aggregate_repo_maturity([])
        assert any("rules" in r.lower() or "instructions" in r.lower()
                   for r in score.recommendations)


# ============================================================================
# Test Artifacts Map
# ============================================================================

class TestBuildArtifactsMap:

    def test_has_all_categories(self):
        fcs = [_make_classification(assigned_category="commands")]
        result = build_artifacts_map(fcs)
        assert len(result) == 9
        assert set(result["category"]) == set(CATEGORY_NAMES)

    def test_counts_primary_and_secondary(self):
        fcs = [
            _make_classification(
                file_id="f1", assigned_category="agents",
                content_primary="agents", path_primary="agents",
                categories_within_threshold=["agents", "rules"],
            ),
        ]
        result = build_artifacts_map(fcs)
        agents_row = result[result["category"] == "agents"].iloc[0]
        assert agents_row["primary_content"] == 1
        rules_row = result[result["category"] == "rules"].iloc[0]
        assert rules_row["secondary_content"] == 1


# ============================================================================
# Test Tool Category Matrix
# ============================================================================

class TestBuildToolCategoryMatrix:

    def test_basic_matrix(self):
        fcs = [
            _make_classification(file_id="f1", tool_name="claude-code",
                                assigned_category="commands"),
            _make_classification(file_id="f2", tool_name="cursor",
                                assigned_category="rules"),
        ]
        result = build_tool_category_matrix(fcs)
        assert "claude-code" in result.index
        assert "cursor" in result.index
        assert result.loc["claude-code", "commands"] == 1
        assert result.loc["cursor", "rules"] == 1

    def test_excludes_unknown_tools(self):
        fcs = [
            _make_classification(file_id="f1", tool_name="unknown"),
        ]
        result = build_tool_category_matrix(fcs)
        assert len(result) == 0


# ============================================================================
# Test Report Generation
# ============================================================================

class TestGenerateReport:

    def test_report_has_required_fields(self):
        score = MaturityScore(
            overall_level=3,
            overall_label="Agent-Augmented",
            confidence=0.85,
            tools_detected=["claude-code"],
            artifact_count=10,
            level_evidence={2: {"primary": 3, "secondary": 5},
                          3: {"primary": 5, "secondary": 2},
                          4: {"primary": 0, "secondary": 0}},
            category_counts={"commands": 5, "rules": 3, "agents": 2},
            coherence_flags=[],
            recommendations=["test"],
            file_classifications=pd.DataFrame([{"signals_agree": True}] * 10),
        )
        report = generate_report(score)
        assert "overall_level" in report
        assert "category_concentration" in report
        assert "signal_agreement_rate" in report
        assert "level_stacking" in report

    def test_signal_agreement_rate(self):
        fc_df = pd.DataFrame([
            {"signals_agree": True},
            {"signals_agree": True},
            {"signals_agree": False},
            {"signals_agree": False},
        ])
        score = MaturityScore(
            overall_level=2, overall_label="Grounded Prompting",
            confidence=0.5, tools_detected=[], artifact_count=4,
            level_evidence={}, category_counts={},
            coherence_flags=[], recommendations=[],
            file_classifications=fc_df,
        )
        report = generate_report(score)
        assert report["signal_agreement_rate"] == 0.5


# ============================================================================
# Test Recommendations
# ============================================================================

class TestGenerateRecommendations:

    def test_l1_recommends_grounding(self):
        recs = _generate_recommendations(
            1, {2: 0, 3: 0, 4: 0}, {2: 0, 3: 0, 4: 0},
            [], {cat: 0 for cat in CATEGORY_NAMES},
        )
        assert len(recs) >= 1
        assert any("rules" in r.lower() or "instructions" in r.lower() for r in recs)

    def test_l2_recommends_agents(self):
        recs = _generate_recommendations(
            2, {2: 5, 3: 0, 4: 0}, {2: 0, 3: 0, 4: 0},
            [], {"rules": 3, "configuration": 2, **{c: 0 for c in CATEGORY_NAMES if c not in ("rules", "configuration")}},
        )
        assert any("agent" in r.lower() or "l3" in r.lower() for r in recs)

    def test_concentration_warning(self):
        cats = {cat: 0 for cat in CATEGORY_NAMES}
        cats["commands"] = 20
        cats["rules"] = 1
        recs = _generate_recommendations(
            3, {2: 1, 3: 20, 4: 0}, {}, [],
            cats,
        )
        assert any("concentrated" in r.lower() for r in recs)

    def test_red_coherence_flag_creates_recommendation(self):
        flags = [CoherenceFlag("L3 without L2", "red", "anomaly")]
        recs = _generate_recommendations(
            3, {2: 0, 3: 10, 4: 0}, {2: 0}, flags,
            {cat: 0 for cat in CATEGORY_NAMES},
        )
        assert any("grounding" in r.lower() or "l2" in r.lower() for r in recs)
