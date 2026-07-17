"""Tests for maturity_scorer module.

Uses mocked SentenceTransformer models to avoid actual model downloads.
"""

import pickle
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
    # Config
    ScoringConfig,
    DEFAULT_CONFIG,
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
    score_from_output_dir,
    _normalize_category,
    _coerce_str_or_none,
    _check_coherence,
    _compute_confidence,
    _build_pattern_lookup,
    _glob_match,
    _match_artifact_category,
)


# Permissive config for legacy tests that use random embeddings —
# random Gaussian vectors produce cosines near zero, well below the
# default tau gates. These tests pre-date thresholding and assert on
# the unconditional argmax behavior.
PERMISSIVE_CONFIG = ScoringConfig(
    tau_content_score=-1.0,
    tau_content_margin=-1.0,
    tau_path_score=-1.0,
    tau_path_margin=-1.0,
    ignore_doc_folders=False,
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
    tool_category=None,
) -> FileClassification:
    """Helper to build a FileClassification."""
    level = CATEGORY_TO_LEVEL.get(assigned_category)
    fc = FileClassification(
        file_id=file_id,
        artifact_path=artifact_path,
        tool_name=tool_name,
        discovery_step="tool_standard",
        tool_category=tool_category,
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
        # 9 leveled categories + 1 absorbing "general-documentation" template.
        assert len(CATEGORY_TEMPLATES) == 10
        assert "general-documentation" in CATEGORY_TEMPLATES

    def test_category_names_ordered(self):
        assert CATEGORY_NAMES == list(CATEGORY_TEMPLATES.keys())

    def test_all_leveled_categories_mapped(self):
        # general-documentation is intentionally not in CATEGORY_TO_LEVEL —
        # it absorbs non-AI documents and contributes no level evidence.
        for cat in CATEGORY_NAMES:
            if cat == "general-documentation":
                assert cat not in CATEGORY_TO_LEVEL
            else:
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
        )
        d = score.to_dict()
        assert d["overall_level"] == 3
        assert d["confidence"] == 0.85
        assert d["coherence_flags"][0]["status"] == "green"
        assert d["boilerplate_filtered"] == 0

    def test_boilerplate_filtered_field(self):
        score = MaturityScore(
            overall_level=2,
            overall_label="Grounded Prompting",
            confidence=0.7,
            tools_detected=[],
            artifact_count=5,
            level_evidence={2: {"primary": 5, "secondary": 0}},
            category_counts={"rules": 5},
            coherence_flags=[],
            boilerplate_filtered=4,
        )
        assert score.boilerplate_filtered == 4
        assert score.to_dict()["boilerplate_filtered"] == 4


# ============================================================================
# Test Category Template Embedding
# ============================================================================

class TestEmbedCategoryTemplates:
    """Tests for embed_category_templates function."""

    def test_returns_correct_shape(self):
        model = _mock_model()
        result = embed_category_templates(model)
        assert result.shape == (len(CATEGORY_NAMES), 768)

    def test_calls_encode_for_each_category(self):
        model = _mock_model()
        embed_category_templates(model)
        assert model.encode.call_count == len(CATEGORY_NAMES)

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
        template_embs = np.random.randn(len(CATEGORY_NAMES), 768).astype(np.float32)
        result = classify_by_path(
            [".claude/commands/test.md"],
            model, template_embs, config=PERMISSIVE_CONFIG,
        )
        assert "path_primary" in result.columns
        assert "path_primary_score" in result.columns
        assert "path_secondary" in result.columns
        assert "path_margin" in result.columns

    def test_empty_input(self):
        model = _mock_model()
        template_embs = np.random.randn(len(CATEGORY_NAMES), 768).astype(np.float32)
        result = classify_by_path([], model, template_embs, config=PERMISSIVE_CONFIG)
        assert len(result) == 0

    def test_result_length_matches_input(self):
        model = _mock_model()
        template_embs = np.random.randn(len(CATEGORY_NAMES), 768).astype(np.float32)
        paths = ["a.md", "b.md", "c.md"]
        result = classify_by_path(paths, model, template_embs, config=PERMISSIVE_CONFIG)
        assert len(result) == 3

    def test_primary_is_valid_category(self):
        model = _mock_model()
        template_embs = np.random.randn(len(CATEGORY_NAMES), 768).astype(np.float32)
        result = classify_by_path(["test.md"], model, template_embs, config=PERMISSIVE_CONFIG)
        assert result.iloc[0]["path_primary"] in CATEGORY_NAMES

    def test_low_score_gates_to_none(self):
        """Random embeddings (cosine ~0) must be gated to path_primary=None under default config."""
        model = _mock_model()
        template_embs = np.random.randn(len(CATEGORY_NAMES), 768).astype(np.float32)
        result = classify_by_path(["test.md"], model, template_embs)  # default config
        assert result.iloc[0]["path_primary"] is None
        # Score is still recorded for diagnostics.
        assert "path_primary_score" in result.columns

    def test_high_score_passes_gate(self):
        """When the file embedding equals one template, cosine=1.0 → not gated."""
        # Construct a template set where path_token "test" deterministic embedding
        # already aligns with the chosen template. Easiest: embed an aligned vec.
        template_embs = np.zeros((len(CATEGORY_NAMES), 768), dtype=np.float32)
        template_embs[0, 0] = 1.0  # category 0 = "agents"
        # Mock model that returns the same one-hot vector for any input
        model = Mock()
        model.encode.return_value = np.array([1.0] + [0.0] * 767, dtype=np.float32)
        # generate_embeddings_batch is what classify_by_path calls under the hood;
        # patch it to return our fixed embedding.
        with patch("src.maturity_scorer.generate_embeddings_batch") as gen:
            gen.return_value = np.array([[1.0] + [0.0] * 767], dtype=np.float32)
            result = classify_by_path(["x.md"], model, template_embs)  # default config
        # cosine = 1.0 with category 0; margin to next ≥ 1.0 → not gated
        assert result.iloc[0]["path_primary"] == CATEGORY_NAMES[0]
        assert result.iloc[0]["path_primary_score"] >= 0.9


# ============================================================================
# Test Signal 3: Content Classification
# ============================================================================

class TestClassifyByContent:
    """Tests for classify_by_content function."""

    def test_returns_correct_columns(self):
        file_embs = np.random.randn(3, 768).astype(np.float32)
        template_embs = np.random.randn(len(CATEGORY_NAMES), 768).astype(np.float32)
        result = classify_by_content(file_embs, template_embs, config=PERMISSIVE_CONFIG)
        assert "content_primary" in result.columns
        assert "content_primary_score" in result.columns
        assert "hybrid_score" in result.columns
        for cat in CATEGORY_NAMES:
            assert f"content_{cat}" in result.columns

    def test_empty_input(self):
        file_embs = np.array([]).reshape(0, 768)
        template_embs = np.random.randn(len(CATEGORY_NAMES), 768).astype(np.float32)
        result = classify_by_content(file_embs, template_embs, config=PERMISSIVE_CONFIG)
        assert len(result) == 0

    def test_hybrid_score_at_least_1(self):
        file_embs = np.random.randn(5, 768).astype(np.float32)
        template_embs = np.random.randn(len(CATEGORY_NAMES), 768).astype(np.float32)
        result = classify_by_content(file_embs, template_embs, config=PERMISSIVE_CONFIG)
        assert (result["hybrid_score"] >= 1).all()

    def test_primary_has_highest_score(self):
        file_embs = np.random.randn(3, 768).astype(np.float32)
        template_embs = np.random.randn(len(CATEGORY_NAMES), 768).astype(np.float32)
        result = classify_by_content(file_embs, template_embs, config=PERMISSIVE_CONFIG)
        for _, row in result.iterrows():
            primary_score = row["content_primary_score"]
            for cat in CATEGORY_NAMES:
                assert row[f"content_{cat}"] <= primary_score + 1e-6

    def test_low_score_gates_to_none(self):
        """Default config gates random Gaussian embeddings (cosine ~0) → content_primary=None."""
        file_embs = np.random.randn(3, 768).astype(np.float32)
        template_embs = np.random.randn(len(CATEGORY_NAMES), 768).astype(np.float32)
        result = classify_by_content(file_embs, template_embs)  # default config
        assert result["content_primary"].isna().all() or (result["content_primary"] == None).all()
        # Score columns are still populated.
        assert "content_primary_score" in result.columns

    def test_high_score_passes_gate(self):
        """When file embedding equals a template (cosine=1), the gate passes."""
        template_embs = np.zeros((len(CATEGORY_NAMES), 768), dtype=np.float32)
        template_embs[2, 0] = 1.0
        file_embs = np.array([[1.0] + [0.0] * 767], dtype=np.float32)
        result = classify_by_content(file_embs, template_embs)  # default config
        assert result.iloc[0]["content_primary"] == CATEGORY_NAMES[2]
        assert result.iloc[0]["content_primary_score"] >= 0.9

    def test_margin_gate_blocks_close_call(self):
        """Two templates with near-identical scores should be gated by tau_margin."""
        # File vector aligned 0.7 to template[0] and 0.69 to template[1] → margin 0.01
        template_embs = np.zeros((len(CATEGORY_NAMES), 768), dtype=np.float32)
        template_embs[0, 0] = 1.0
        template_embs[1, 0] = 0.99
        template_embs[1, 1] = np.sqrt(1 - 0.99**2)  # unit vector
        file_embs = np.array([[1.0] + [0.0] * 767], dtype=np.float32)
        result = classify_by_content(file_embs, template_embs)  # default config tau_margin=0.02
        # cosines: 1.0 and 0.99 → margin 0.01 < 0.02 → gated
        assert result.iloc[0]["content_primary"] is None


# ============================================================================
# Test Normalize Category
# ============================================================================

class TestNormalizeCategory:

    def test_standard_category(self):
        assert _normalize_category("rules") == "rules"
        assert _normalize_category("agents") == "agents"

    def test_dialect_aliases_to_rules(self):
        # All four L2-grounding dialect names should resolve to "rules"
        assert _normalize_category("instructions") == "rules"
        assert _normalize_category("context") == "rules"
        assert _normalize_category("steering") == "rules"
        assert _normalize_category("guidelines") == "rules"

    def test_dialect_aliases_to_agents(self):
        assert _normalize_category("microagents") == "agents"

    def test_dialect_aliases_to_commands(self):
        assert _normalize_category("prompts") == "commands"

    def test_dialect_aliases_to_flows(self):
        assert _normalize_category("workflows") == "flows"
        assert _normalize_category("hooks") == "flows"

    def test_unknown_maps_to_none(self):
        assert _normalize_category("unknown") is None

    def test_none_input(self):
        assert _normalize_category(None) is None

    def test_unrecognized_maps_to_none(self):
        assert _normalize_category("nonexistent") is None

    def test_every_artifact_json_category_is_mapped(self):
        """Coverage guarantee: every artifact_category value used in any
        Artifacts/*.json must resolve via _normalize_category to either a
        leveled template name (in CATEGORY_TO_LEVEL) or explicit None.

        If a tool author adds a new dialect value to a JSON without updating
        ARTIFACT_CATEGORY_TO_TEMPLATE, this test fails — preventing silent
        drops of tool-detection evidence.
        """
        import json
        from pathlib import Path

        artifacts_dir = Path(__file__).resolve().parent.parent / "Artifacts"
        json_files = sorted(artifacts_dir.glob("*.json"))
        assert json_files, f"No Artifacts/*.json found at {artifacts_dir}"

        seen: dict[str, list[str]] = {}
        for jf in json_files:
            data = json.loads(jf.read_text())
            for pattern in data.get("artifact_patterns", []):
                cat = pattern.get("artifact_category")
                if cat is None:
                    continue
                seen.setdefault(cat, []).append(jf.name)

        unmapped = []
        for cat in seen:
            normalized = _normalize_category(cat)
            # "unknown" is the only category that may legitimately resolve to None
            if normalized is None and cat != "unknown":
                unmapped.append((cat, seen[cat]))
            elif normalized is not None and normalized not in CATEGORY_TO_LEVEL:
                unmapped.append((cat, seen[cat]))

        assert not unmapped, (
            "These artifact_category values from Artifacts/*.json have no "
            "leveled mapping (add them to ARTIFACT_CATEGORY_TO_TEMPLATE):\n  "
            + "\n  ".join(f"{cat!r}  used in: {sorted(set(files))}"
                          for cat, files in unmapped)
        )


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
        """Same-level disagreement (rules vs architecture, both L2) → content wins."""
        artifacts_df = pd.DataFrame([{
            "file_id": "f1",
            "artifact_path": "src/setup.md",
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

    def test_cross_level_disagreement_demote_when_enabled(self):
        """Opt-in lever: with demote enabled, content=L4 vs path=L2 → keep the lower-level (L2)."""
        cfg = ScoringConfig(cross_level_disagreement_demote=True)
        artifacts_df = pd.DataFrame([{
            "file_id": "f1",
            "artifact_path": "PERFORMANCE.md",
            "tool_name": "unknown",
            "discovery_step": "non_standard_root",
        }])
        tool_signal = pd.DataFrame([{"file_id": "f1", "tool_category": None}])
        path_signal = pd.DataFrame([{
            "path_primary": "configuration",  # L2
            "path_primary_score": 0.7,
            "path_secondary": "skills",
            "path_secondary_score": 0.65,
            "path_margin": 0.05,
        }])
        content_signal = pd.DataFrame([{
            "content_primary": "session-logs",  # L4
            "content_primary_score": 0.72,
            "content_secondary": "rules",
            "content_secondary_score": 0.65,
            "content_margin": 0.07,
            "hybrid_score": 1,
            "categories_within_threshold": "session-logs",
            **{f"content_{cat}": 0.5 for cat in CATEGORY_NAMES},
        }])

        result = combine_signals(artifacts_df, tool_signal, path_signal, content_signal, cfg)
        fc = result[0]
        # Lower level wins → configuration (L2), not session-logs (L4)
        assert fc.assigned_category == "configuration"
        assert fc.assigned_maturity_level == 2

    def test_cross_level_demote_default_off(self):
        """Default config: content keeps winning on cross-level disagreement (demote off)."""
        artifacts_df = pd.DataFrame([{
            "file_id": "f1", "artifact_path": "x.md",
            "tool_name": "unknown", "discovery_step": "non_standard_root",
        }])
        tool_signal = pd.DataFrame([{"file_id": "f1", "tool_category": None}])
        path_signal = pd.DataFrame([{
            "path_primary": "configuration", "path_primary_score": 0.7,
            "path_secondary": "skills", "path_secondary_score": 0.65, "path_margin": 0.05,
        }])
        content_signal = pd.DataFrame([{
            "content_primary": "session-logs", "content_primary_score": 0.72,
            "content_secondary": "rules", "content_secondary_score": 0.65, "content_margin": 0.07,
            "hybrid_score": 1, "categories_within_threshold": "session-logs",
            **{f"content_{cat}": 0.5 for cat in CATEGORY_NAMES},
        }])
        # No explicit cfg → DEFAULT_CONFIG (demote off)
        result = combine_signals(artifacts_df, tool_signal, path_signal, content_signal)
        assert result[0].assigned_category == "session-logs"
        assert result[0].assigned_maturity_level == 4

    def test_ignore_doc_folders_default_on(self):
        """Default config drops docs/ files; non-doc files keep being scored."""
        artifacts_df = pd.DataFrame([
            {"file_id": "f1", "artifact_path": "docs/framework/rules.md",
             "tool_name": "unknown", "discovery_step": "non_standard_other"},
            {"file_id": "f2", "artifact_path": ".claude/agents/foo.md",
             "tool_name": "unknown", "discovery_step": "non_standard_other"},
        ])
        tool_signal = pd.DataFrame([
            {"file_id": "f1", "tool_category": None},
            {"file_id": "f2", "tool_category": None},
        ])
        path_signal = pd.DataFrame([
            {"path_primary": "rules", "path_primary_score": 0.75,
             "path_secondary": None, "path_secondary_score": 0.0, "path_margin": 0.1},
            {"path_primary": "agents", "path_primary_score": 0.80,
             "path_secondary": None, "path_secondary_score": 0.0, "path_margin": 0.1},
        ])
        content_signal = pd.DataFrame([
            {"content_primary": "rules", "content_primary_score": 0.72,
             "content_secondary": None, "content_secondary_score": 0.0, "content_margin": 0.1,
             "hybrid_score": 1, "categories_within_threshold": "rules",
             **{f"content_{cat}": 0.5 for cat in CATEGORY_NAMES}},
            {"content_primary": "agents", "content_primary_score": 0.72,
             "content_secondary": None, "content_secondary_score": 0.0, "content_margin": 0.1,
             "hybrid_score": 1, "categories_within_threshold": "agents",
             **{f"content_{cat}": 0.5 for cat in CATEGORY_NAMES}},
        ])
        # No explicit cfg → DEFAULT_CONFIG (ignore_doc_folders on)
        result = combine_signals(artifacts_df, tool_signal, path_signal, content_signal)
        assert len(result) == 1
        assert result[0].artifact_path == ".claude/agents/foo.md"
        assert result[0].assigned_category == "agents"

    def test_ignore_doc_folders_opt_out(self):
        """Explicit ignore_doc_folders=False restores the pre-filter behavior."""
        cfg = ScoringConfig(ignore_doc_folders=False)
        artifacts_df = pd.DataFrame([{
            "file_id": "f1", "artifact_path": "docs/framework/rules.md",
            "tool_name": "unknown", "discovery_step": "non_standard_other",
        }])
        tool_signal = pd.DataFrame([{"file_id": "f1", "tool_category": None}])
        path_signal = pd.DataFrame([{
            "path_primary": "rules", "path_primary_score": 0.75,
            "path_secondary": None, "path_secondary_score": 0.0, "path_margin": 0.1,
        }])
        content_signal = pd.DataFrame([{
            "content_primary": "rules", "content_primary_score": 0.72,
            "content_secondary": None, "content_secondary_score": 0.0, "content_margin": 0.1,
            "hybrid_score": 1, "categories_within_threshold": "rules",
            **{f"content_{cat}": 0.5 for cat in CATEGORY_NAMES},
        }])
        result = combine_signals(artifacts_df, tool_signal, path_signal, content_signal, cfg)
        assert len(result) == 1
        assert result[0].assigned_category == "rules"


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
        # tool_category=rules on f1 provides has_leveled_tool_attribution → cap doesn't fire
        fcs = [
            _make_classification(file_id="f1", assigned_category="rules", tool_category="rules",
                                content_primary="rules", path_primary="rules"),
            _make_classification(file_id="f2", assigned_category="configuration",
                                content_primary="configuration", path_primary="configuration"),
        ]
        score = aggregate_repo_maturity(fcs)
        assert score.overall_level == 2

    def test_l3_with_agents_and_l2(self):
        fcs = [
            _make_classification(file_id="f1", assigned_category="rules", tool_category="rules",
                                content_primary="rules", path_primary="rules"),
            _make_classification(file_id="f2", assigned_category="agents", tool_category="agents",
                                content_primary="agents", path_primary="agents"),
            _make_classification(file_id="f3", assigned_category="commands", tool_category="commands",
                                content_primary="commands", path_primary="commands"),
        ]
        score = aggregate_repo_maturity(fcs)
        assert score.overall_level == 3

    def test_l4_with_flows(self):
        """L4 artifacts present + tool attribution → repo is L4."""
        fcs = [
            _make_classification(file_id="f1", assigned_category="agents", tool_category="agents",
                                content_primary="agents", path_primary="agents"),
            _make_classification(file_id="f2", assigned_category="flows", tool_category="flows",
                                content_primary="flows", path_primary="flows"),
        ]
        score = aggregate_repo_maturity(fcs)
        assert score.overall_level == 4

    def test_strict_cap_pins_to_l1_without_tool_attribution(self):
        """Semantic evidence alone (no tool_category) → capped at L1 by default."""
        fcs = [
            _make_classification(file_id="f1", assigned_category="agents", tool_category=None,
                                content_primary="agents", path_primary="agents"),
            _make_classification(file_id="f2", assigned_category="flows", tool_category=None,
                                content_primary="flows", path_primary="flows"),
        ]
        score = aggregate_repo_maturity(fcs)
        assert score.overall_level == 1
        assert score.overall_label == "Ad Hoc"
        assert score.has_leveled_tool_attribution is False

    def test_strict_cap_can_be_disabled(self):
        """Disabling the cap recovers the un-capped semantic level for sensitivity analysis."""
        cfg = ScoringConfig(strict_cap_to_l1_without_tool_attribution=False)
        fcs = [
            _make_classification(file_id="f1", assigned_category="flows", tool_category=None,
                                content_primary="flows", path_primary="flows"),
        ]
        score = aggregate_repo_maturity(fcs, config=cfg)
        assert score.overall_level == 4
        assert score.has_leveled_tool_attribution is False

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

# Recommendations live in report_generator.py and are tested there.

# ============================================================================
# Test Artifacts Map
# ============================================================================

class TestBuildArtifactsMap:

    def test_has_all_categories(self):
        fcs = [_make_classification(assigned_category="commands")]
        result = build_artifacts_map(fcs)
        assert len(result) == len(CATEGORY_NAMES)
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
            coherence_flags=[],
            file_classifications=fc_df,
        )
        report = generate_report(score)
        assert report["signal_agreement_rate"] == 0.5


# ============================================================================
# Test Internal Helpers (_coerce_str_or_none, _glob_match, _build_pattern_lookup)
# ============================================================================

class TestCoerceStrOrNone:
    """Tests for the small _coerce_str_or_none helper."""

    def test_none_input_returns_none(self):
        assert _coerce_str_or_none(None) is None

    def test_nan_returns_none(self):
        assert _coerce_str_or_none(float("nan")) is None

    def test_empty_string_returns_none(self):
        assert _coerce_str_or_none("") is None

    def test_non_empty_string_returned(self):
        assert _coerce_str_or_none("rules") == "rules"

    def test_non_string_value_coerced_to_str(self):
        # Non-string, non-NaN floats coerce to their string form
        assert _coerce_str_or_none(3.14) == "3.14"


class TestGlobMatch:
    """Tests for the internal _glob_match helper."""

    def test_simple_filename_match(self):
        # PurePosixPath.match handles single-segment glob.
        assert _glob_match("file.md", "*.md") is True

    def test_double_star_zero_intermediate_dirs(self):
        # The collapsed-pattern branch fires when **/ should match zero dirs.
        assert _glob_match("foo.md", "**/*.md") is True

    def test_double_star_with_intermediate_dirs(self):
        assert _glob_match(".claude/commands/sparc.md", ".claude/commands/**/*.md") is True

    def test_no_match_returns_false(self):
        assert _glob_match("docs/setup.txt", ".claude/commands/**/*.md") is False


class TestBuildPatternLookup:
    """Tests for _build_pattern_lookup — covers shared config + skip-on-no-pattern."""

    @patch("src.maturity_scorer.load_shared_config")
    @patch("src.maturity_scorer.load_json_configs")
    def test_shared_config_added_to_lookup(self, mock_load, mock_shared):
        """When a shared config exists it should be included under its tool_name."""
        from src.data_models import ToolConfig, ArtifactPattern, DiscoveryMethod, ArtifactStatus
        shared_pattern = ArtifactPattern(
            pattern="AGENTS.md",
            type="file",
            description="Shared",
            file_type="markdown",
            status=ArtifactStatus.STABLE,
            is_standard=True,
            artifact_category="instructions",
            scope="project",
            discovery_method=DiscoveryMethod.EXACT_PATH,
            exact_path="AGENTS.md",
        )
        shared = ToolConfig(tool_name="shared", artifact_patterns=[shared_pattern])
        mock_load.return_value = {}
        mock_shared.return_value = shared

        lookup = _build_pattern_lookup("Artifacts")
        assert any(e["tool_name"] == "shared" and e["match_value"] == "AGENTS.md"
                   for e in lookup)

    @patch("src.maturity_scorer.load_shared_config")
    @patch("src.maturity_scorer.load_json_configs")
    def test_pattern_without_exact_or_glob_is_skipped(self, mock_load, mock_shared):
        """A pattern with neither exact_path nor glob_pattern produces no lookup entry."""
        from src.data_models import ToolConfig, ArtifactPattern, DiscoveryMethod, ArtifactStatus
        # regex_pattern is a valid alternative but neither exact_path nor glob_pattern is set
        regex_only = ArtifactPattern(
            pattern="weird",
            type="file",
            description="Regex only",
            file_type="markdown",
            status=ArtifactStatus.STABLE,
            is_standard=True,
            artifact_category="rules",
            scope="project",
            discovery_method=DiscoveryMethod.REGEX,
            regex_pattern=r".*\.md$",
        )
        tool = ToolConfig(tool_name="weird", artifact_patterns=[regex_only])
        mock_load.return_value = {"weird": tool}
        mock_shared.return_value = None

        lookup = _build_pattern_lookup("Artifacts")
        # No entries for the regex-only pattern.
        assert all(e["tool_name"] != "weird" for e in lookup)


class TestMatchArtifactCategory:
    """Direct tests for the reverse-lookup helper."""

    @patch("src.maturity_scorer.load_shared_config")
    @patch("src.maturity_scorer.load_json_configs")
    def test_no_match_returns_none(self, mock_load, mock_shared):
        from src.data_models import ToolConfig, ArtifactPattern, DiscoveryMethod, ArtifactStatus
        pat = ArtifactPattern(
            pattern="CLAUDE.md", type="file", description="x", file_type="md",
            status=ArtifactStatus.STABLE, is_standard=True,
            artifact_category="instructions", scope="project",
            discovery_method=DiscoveryMethod.EXACT_PATH, exact_path="CLAUDE.md",
        )
        mock_load.return_value = {"claude-code": ToolConfig(
            tool_name="claude-code", artifact_patterns=[pat])}
        mock_shared.return_value = None
        lookup = _build_pattern_lookup("Artifacts")
        assert _match_artifact_category("docs/whatever.md", "claude-code", lookup) is None

    @patch("src.maturity_scorer.load_shared_config")
    @patch("src.maturity_scorer.load_json_configs")
    def test_falls_back_across_tools_when_owning_tool_misses(self, mock_load, mock_shared):
        """If a path doesn't match its own tool's patterns, the lookup walks
        the other tools — covering the third candidate-list extension."""
        from src.data_models import ToolConfig, ArtifactPattern, DiscoveryMethod, ArtifactStatus
        cursor_pat = ArtifactPattern(
            pattern=".cursorrules", type="file", description="x", file_type="md",
            status=ArtifactStatus.STABLE, is_standard=True,
            artifact_category="rules", scope="project",
            discovery_method=DiscoveryMethod.EXACT_PATH, exact_path=".cursorrules",
        )
        claude_pat = ArtifactPattern(
            pattern="CLAUDE.md", type="file", description="x", file_type="md",
            status=ArtifactStatus.STABLE, is_standard=True,
            artifact_category="instructions", scope="project",
            discovery_method=DiscoveryMethod.EXACT_PATH, exact_path="CLAUDE.md",
        )
        mock_load.return_value = {
            "cursor": ToolConfig(tool_name="cursor", artifact_patterns=[cursor_pat]),
            "claude-code": ToolConfig(tool_name="claude-code", artifact_patterns=[claude_pat]),
        }
        mock_shared.return_value = None
        lookup = _build_pattern_lookup("Artifacts")
        # Path is a Claude file, but tool_name is reported as "cursor" — match via fallback.
        result = _match_artifact_category("CLAUDE.md", "cursor", lookup)
        assert result == "instructions"


# ============================================================================
# Test combine_signals — additional fallback paths
# ============================================================================

class TestCombineSignalsFallbacks:

    def _empty_path_row(self):
        return {
            "path_primary": None,
            "path_primary_score": 0.0,
            "path_secondary": None,
            "path_secondary_score": 0.0,
            "path_margin": 0.0,
        }

    def _empty_content_row(self):
        return {
            "content_primary": None,
            "content_primary_score": 0.0,
            "content_secondary": None,
            "content_secondary_score": 0.0,
            "content_margin": 0.0,
            "hybrid_score": 1,
            "categories_within_threshold": "",
            **{f"content_{cat}": 0.0 for cat in CATEGORY_NAMES},
        }

    def test_path_only_fallback(self):
        """If only path_primary is present (content gated), use the path category."""
        artifacts_df = pd.DataFrame([{
            "file_id": "f1", "artifact_path": "agents/x.md",
            "tool_name": "unknown", "discovery_step": "non_standard_other",
        }])
        tool_signal = pd.DataFrame([{"file_id": "f1", "tool_category": None}])
        path_signal = pd.DataFrame([{
            "path_primary": "agents",
            "path_primary_score": 0.8,
            "path_secondary": "skills",
            "path_secondary_score": 0.5,
            "path_margin": 0.3,
        }])
        content_signal = pd.DataFrame([self._empty_content_row()])
        result = combine_signals(artifacts_df, tool_signal, path_signal, content_signal)
        fc = result[0]
        assert fc.assigned_category == "agents"
        assert fc.assigned_maturity_level == 3
        assert fc.signals_agree is False

    def test_no_signals_leaves_unassigned(self):
        """All signals empty → assigned_category remains None."""
        artifacts_df = pd.DataFrame([{
            "file_id": "f1", "artifact_path": "x.md",
            "tool_name": "unknown", "discovery_step": "non_standard_other",
        }])
        tool_signal = pd.DataFrame([{"file_id": "f1", "tool_category": None}])
        path_signal = pd.DataFrame([self._empty_path_row()])
        content_signal = pd.DataFrame([self._empty_content_row()])
        result = combine_signals(artifacts_df, tool_signal, path_signal, content_signal)
        fc = result[0]
        assert fc.assigned_category is None
        assert fc.assigned_maturity_level is None

    def test_general_documentation_assigned_but_unleveled(self):
        """general-documentation is in CATEGORY_NAMES but not in CATEGORY_TO_LEVEL —
        it should be assigned but produce no maturity level."""
        artifacts_df = pd.DataFrame([{
            "file_id": "f1", "artifact_path": "README.md",
            "tool_name": "unknown", "discovery_step": "non_standard_root",
        }])
        tool_signal = pd.DataFrame([{"file_id": "f1", "tool_category": None}])
        path_signal = pd.DataFrame([{
            "path_primary": "general-documentation", "path_primary_score": 0.8,
            "path_secondary": "rules", "path_secondary_score": 0.5, "path_margin": 0.3,
        }])
        content_signal = pd.DataFrame([{
            "content_primary": "general-documentation", "content_primary_score": 0.85,
            "content_secondary": "rules", "content_secondary_score": 0.5, "content_margin": 0.35,
            "hybrid_score": 1, "categories_within_threshold": "general-documentation",
            **{f"content_{cat}": 0.0 for cat in CATEGORY_NAMES},
        }])
        result = combine_signals(artifacts_df, tool_signal, path_signal, content_signal)
        fc = result[0]
        assert fc.assigned_category == "general-documentation"
        assert fc.assigned_maturity_level is None  # general-documentation has no level


# ============================================================================
# Test _compute_confidence small-count branch
# ============================================================================

class TestConfidenceSmallCounts:

    def test_few_artifacts_uses_low_count_score(self):
        """artifact_count < 5 should hit the count_score = 0.4 branch."""
        c_small = _compute_confidence(2, 1, 1.0, [])  # 1 artifact, perfect agreement, no flags
        c_medium = _compute_confidence(2, 5, 1.0, [])  # 5 artifacts → 0.6
        c_large = _compute_confidence(2, 20, 1.0, [])  # 20 artifacts → 1.0
        # Strict ordering: small count must produce strictly lower confidence.
        assert c_small < c_medium < c_large

    def test_yellow_flag_reduces_less_than_red(self):
        red = _compute_confidence(3, 10, 0.7, [CoherenceFlag("a", "red", "x")])
        yellow = _compute_confidence(3, 10, 0.7, [CoherenceFlag("a", "yellow", "x")])
        clean = _compute_confidence(3, 10, 0.7, [])
        assert red < yellow < clean


# ============================================================================
# Test aggregate_repo_maturity — additional branches
# ============================================================================

class TestAggregateRepoMaturityBranches:

    def test_unleveled_classifications_collapse_to_l1(self):
        """File classifications with no leveled category (e.g., general-documentation) →
        repo lands at L1, not by the strict cap but because no level evidence exists."""
        cfg = ScoringConfig(strict_cap_to_l1_without_tool_attribution=False)
        fc = _make_classification(
            file_id="f1",
            assigned_category="general-documentation",
            content_primary="general-documentation",
            path_primary="general-documentation",
            categories_within_threshold=["general-documentation"],
        )
        # Manually clear the level (general-documentation isn't in CATEGORY_TO_LEVEL anyway)
        fc.assigned_maturity_level = None
        score = aggregate_repo_maturity([fc], config=cfg)
        assert score.overall_level == 1
        assert score.level_evidence[2]["primary"] == 0
        assert score.level_evidence[3]["primary"] == 0
        assert score.level_evidence[4]["primary"] == 0


# ============================================================================
# Test generate_report — empty / no-classifications branch
# ============================================================================

class TestGenerateReportEmpty:

    def test_no_file_classifications_zero_agreement_rate(self):
        score = MaturityScore(
            overall_level=1, overall_label="Ad Hoc",
            confidence=1.0, tools_detected=[], artifact_count=0,
            level_evidence={2: {"primary": 0, "secondary": 0},
                            3: {"primary": 0, "secondary": 0},
                            4: {"primary": 0, "secondary": 0}},
            category_counts={cat: 0 for cat in CATEGORY_NAMES},
            coherence_flags=[],
            file_classifications=None,
        )
        report = generate_report(score)
        assert report["signal_agreement_rate"] == 0.0
        assert report["category_concentration"] == 0.0

    def test_empty_dataframe_zero_agreement_rate(self):
        score = MaturityScore(
            overall_level=1, overall_label="Ad Hoc",
            confidence=1.0, tools_detected=[], artifact_count=0,
            level_evidence={2: {"primary": 0, "secondary": 0},
                            3: {"primary": 0, "secondary": 0},
                            4: {"primary": 0, "secondary": 0}},
            category_counts={cat: 0 for cat in CATEGORY_NAMES},
            coherence_flags=[],
            file_classifications=pd.DataFrame(),
        )
        report = generate_report(score)
        assert report["signal_agreement_rate"] == 0.0


# ============================================================================
# Test build_artifacts_map — agreement and general-documentation branches
# ============================================================================

class TestBuildArtifactsMapBranches:

    def test_agreement_counted(self):
        fcs = [
            _make_classification(
                file_id="f1", assigned_category="rules",
                content_primary="rules", path_primary="rules",
                categories_within_threshold=["rules"],
            ),
            _make_classification(
                file_id="f2", assigned_category="rules",
                content_primary="rules", path_primary="agents",  # disagrees
                categories_within_threshold=["rules"],
                signals_agree=False,
            ),
        ]
        result = build_artifacts_map(fcs)
        rules_row = result[result["category"] == "rules"].iloc[0]
        assert rules_row["agreement"] == 1  # only f1 agrees on "rules"

    def test_general_documentation_has_no_level(self):
        result = build_artifacts_map([])
        gd_row = result[result["category"] == "general-documentation"].iloc[0]
        # pandas widens an int column with a None to float64 NaN — check via pd.isna
        assert pd.isna(gd_row["maturity_level"])

    def test_content_only_fallback_assigns_content(self):
        """If path is gated to None but content is present, content_primary is used."""
        artifacts_df = pd.DataFrame([{
            "file_id": "f1", "artifact_path": "x.md",
            "tool_name": "unknown", "discovery_step": "non_standard_other",
        }])
        tool_signal = pd.DataFrame([{"file_id": "f1", "tool_category": None}])
        path_signal = pd.DataFrame([{
            "path_primary": None, "path_primary_score": 0.0,
            "path_secondary": None, "path_secondary_score": 0.0, "path_margin": 0.0,
        }])
        content_signal = pd.DataFrame([{
            "content_primary": "rules", "content_primary_score": 0.85,
            "content_secondary": "agents", "content_secondary_score": 0.5, "content_margin": 0.35,
            "hybrid_score": 1, "categories_within_threshold": "rules",
            **{f"content_{cat}": 0.0 for cat in CATEGORY_NAMES},
        }])
        result = combine_signals(artifacts_df, tool_signal, path_signal, content_signal)
        assert result[0].assigned_category == "rules"
        assert result[0].assigned_maturity_level == 2


# ============================================================================
# Test score_from_output_dir — end-to-end on synthetic output dir
# ============================================================================

class TestScoreFromOutputDir:

    def _write_repo(self, tmp_path, repo, df, embeddings=None, file_ids=None):
        repo_dir = tmp_path / repo
        repo_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(repo_dir / f"{repo}_file_artifacts.csv", index=False)
        if embeddings is not None:
            with open(repo_dir / f"{repo}_embeddings.pkl", "wb") as f:
                pickle.dump({"embeddings": embeddings, "file_ids": file_ids}, f)
        return repo_dir

    def test_missing_csv_returns_l1(self, tmp_path):
        # Empty repo dir → no CSVs → early-return L1 path
        (tmp_path / "empty_repo").mkdir()
        model = _mock_model()
        score = score_from_output_dir(str(tmp_path), "empty_repo", model)
        assert score.overall_level == 1
        assert score.artifact_count == 0

    @patch("src.maturity_scorer.classify_by_tool_detection")
    @patch("src.maturity_scorer.is_boilerplate")
    def test_csv_no_embeddings_pads_content_signal(
        self, mock_boilerplate, mock_tool, tmp_path,
    ):
        """Without an embeddings PKL, the function must still produce a score
        by padding the content signal with zeros."""
        mock_boilerplate.return_value = False  # nothing filtered as boilerplate
        mock_tool.return_value = pd.DataFrame([
            {"file_id": "f1", "tool_category": "rules"},
        ])
        df = pd.DataFrame([{
            "file_id": "f1",
            "artifact_path": "CLAUDE.md",
            "artifact_name": "CLAUDE.md",
            "tool_name": "claude-code",
            "discovery_step": "tool_standard",
        }])
        self._write_repo(tmp_path, "repo_a", df)

        model = _mock_model()
        # Use a permissive config so the path signal can attribute, although
        # we mainly care that the function completes and returns a score.
        score = score_from_output_dir(
            str(tmp_path), "repo_a", model, config=PERMISSIVE_CONFIG,
        )
        assert score.artifact_count == 1
        # Tool category resolved → has_leveled_tool_attribution is True, so the
        # strict cap does not fire. Repo is at least L2 because tool_category=rules.
        assert score.overall_level >= 2
        assert score.has_leveled_tool_attribution is True

    @patch("src.maturity_scorer.classify_by_tool_detection")
    @patch("src.maturity_scorer.is_boilerplate")
    def test_boilerplate_filtered_count_recorded(
        self, mock_boilerplate, mock_tool, tmp_path,
    ):
        """Boilerplate files are dropped before classification and counted."""
        # First file (README) is boilerplate, second (CLAUDE.md) is not.
        # Accept *args because is_boilerplate is invoked with both 3-arg
        # (early filter in score_from_output_dir, includes artifacts_dir)
        # and 2-arg (defensive filter in combine_signals) call sites.
        mock_boilerplate.side_effect = lambda name, *args, **kw: name == "README.md"
        mock_tool.return_value = pd.DataFrame([
            {"file_id": "f2", "tool_category": "rules"},
        ])

        df = pd.DataFrame([
            {"file_id": "f1", "artifact_path": "README.md", "artifact_name": "README.md",
             "tool_name": "unknown", "discovery_step": "non_standard_root"},
            {"file_id": "f2", "artifact_path": "CLAUDE.md", "artifact_name": "CLAUDE.md",
             "tool_name": "claude-code", "discovery_step": "tool_standard"},
        ])
        self._write_repo(tmp_path, "repo_b", df)

        model = _mock_model()
        score = score_from_output_dir(
            str(tmp_path), "repo_b", model, config=PERMISSIVE_CONFIG,
        )
        assert score.boilerplate_filtered == 1
        assert score.artifact_count == 1  # only CLAUDE.md classified

    @patch("src.maturity_scorer.classify_by_tool_detection")
    @patch("src.maturity_scorer.is_boilerplate")
    def test_embeddings_present_but_all_rows_filtered_as_boilerplate(
        self, mock_boilerplate, mock_tool, tmp_path,
    ):
        """When the embeddings PKL exists but every artifact row is dropped as
        boilerplate, aligned_embeddings is empty → falls into the 0-row
        classify_by_content path (line 1291)."""
        mock_boilerplate.return_value = True  # everything is boilerplate
        mock_tool.return_value = pd.DataFrame(columns=["file_id", "tool_category"])
        df = pd.DataFrame([{
            "file_id": "f1", "artifact_path": "README.md", "artifact_name": "README.md",
            "tool_name": "unknown", "discovery_step": "non_standard_root",
        }])
        embs = np.ones((1, 768), dtype=np.float32)
        self._write_repo(tmp_path, "all_boilerplate", df, embeddings=embs, file_ids=["f1"])

        model = _mock_model()
        score = score_from_output_dir(
            str(tmp_path), "all_boilerplate", model, config=PERMISSIVE_CONFIG,
        )
        assert score.boilerplate_filtered == 1
        assert score.artifact_count == 0
        assert score.overall_level == 1

    @patch("src.maturity_scorer.classify_by_tool_detection")
    @patch("src.maturity_scorer.is_boilerplate")
    def test_score_returns_l4_for_full_stack(
        self, mock_boilerplate, mock_tool, tmp_path,
    ):
        """End-to-end sanity: a repo with rules + agents + flows lands at L4
        with all three levels reflected in level_evidence."""
        mock_boilerplate.return_value = False
        mock_tool.return_value = pd.DataFrame([
            {"file_id": "f1", "tool_category": "rules"},
            {"file_id": "f2", "tool_category": "agents"},
            {"file_id": "f3", "tool_category": "flows"},
        ])
        df = pd.DataFrame([
            {"file_id": "f1", "artifact_path": "CLAUDE.md", "artifact_name": "CLAUDE.md",
             "tool_name": "claude-code", "discovery_step": "tool_standard"},
            {"file_id": "f2", "artifact_path": ".claude/agents/x.md", "artifact_name": "x.md",
             "tool_name": "claude-code", "discovery_step": "tool_standard"},
            {"file_id": "f3", "artifact_path": ".claude/flows/y.md", "artifact_name": "y.md",
             "tool_name": "claude-code", "discovery_step": "tool_standard"},
        ])
        self._write_repo(tmp_path, "fullstack", df)
        model = _mock_model()
        score = score_from_output_dir(
            str(tmp_path), "fullstack", model, config=PERMISSIVE_CONFIG,
        )
        assert score.overall_level == 4
        assert score.overall_label == "Agentic Orchestration"
        assert score.level_evidence[2]["primary"] == 1
        assert score.level_evidence[3]["primary"] == 1
        assert score.level_evidence[4]["primary"] == 1
        # Tools detected and category counts populated correctly
        assert score.tools_detected == ["claude-code"]
        assert score.category_counts["rules"] == 1
        assert score.category_counts["agents"] == 1
        assert score.category_counts["flows"] == 1

    @patch("src.maturity_scorer.classify_by_tool_detection")
    @patch("src.maturity_scorer.is_boilerplate")
    def test_embeddings_aligned_by_file_id(
        self, mock_boilerplate, mock_tool, tmp_path,
    ):
        """When an embeddings PKL is present, embeddings should be aligned by
        file_id — covering the alignment branch and zero-vector fallback."""
        mock_boilerplate.return_value = False
        mock_tool.return_value = pd.DataFrame([
            {"file_id": "f1", "tool_category": "rules"},
            {"file_id": "f2", "tool_category": None},
        ])

        df = pd.DataFrame([
            {"file_id": "f1", "artifact_path": "CLAUDE.md", "artifact_name": "CLAUDE.md",
             "tool_name": "claude-code", "discovery_step": "tool_standard"},
            # f2 has no embedding → must fall back to zero vector
            {"file_id": "f2", "artifact_path": "docs/x.md", "artifact_name": "x.md",
             "tool_name": "unknown", "discovery_step": "non_standard_root"},
        ])
        # Only embed f1.
        embs = np.ones((1, 768), dtype=np.float32)
        self._write_repo(tmp_path, "repo_c", df, embeddings=embs, file_ids=["f1"])

        model = _mock_model()
        score = score_from_output_dir(
            str(tmp_path), "repo_c", model, config=PERMISSIVE_CONFIG,
        )
        assert score.artifact_count == 2
        # has_leveled_tool_attribution flips on (f1.tool_category=rules → L2)
        assert score.has_leveled_tool_attribution is True


# ============================================================================
# Mutation-targeted tests
#
# These tests pin down specific behaviors that mutation testing identified as
# under-asserted. Each assertion is chosen to fail for a specific class of
# mutation that survived a baseline cosmic-ray run on src/maturity_scorer.py.
# ============================================================================

class TestMutationKillers:

    def test_secondary_excludes_assigned_category(self):
        """In aggregate_repo_maturity, the assigned category must NOT be
        double-counted in level_evidence[*]['secondary'] even when it appears
        in categories_within_threshold. Targets: line 915 mutation
        `cat != fc.assigned_category` → `==` (or relational variants)."""
        fc = _make_classification(
            file_id="f1", assigned_category="agents",
            tool_category="agents",
            content_primary="agents", path_primary="agents",
            categories_within_threshold=["agents", "rules"],
        )
        score = aggregate_repo_maturity([fc])
        # L3 secondary stays 0 because "agents" is the primary, not secondary.
        assert score.level_evidence[3]["secondary"] == 0
        # Only "rules" — the within_threshold entry distinct from assigned —
        # contributes secondary.
        assert score.level_evidence[2]["secondary"] == 1

    def test_artifacts_map_secondary_excludes_assigned(self):
        """build_artifacts_map.secondary_content must skip rows where the
        category equals the assigned_category. Targets: line 1091
        `fc.assigned_category != cat` → relational variants."""
        fc = _make_classification(
            file_id="f1", assigned_category="rules",
            content_primary="rules", path_primary="rules",
            categories_within_threshold=["rules", "agents"],
        )
        result = build_artifacts_map([fc])
        rules_row = result[result["category"] == "rules"].iloc[0]
        # rules is assigned → secondary count must be 0 for it.
        assert rules_row["secondary_content"] == 0
        # agents is in within_threshold but not assigned → counted as secondary.
        agents_row = result[result["category"] == "agents"].iloc[0]
        assert agents_row["secondary_content"] == 1

    def test_level_stacking_uses_addition_not_other_op(self):
        """generate_report.level_stacking must compute primary+secondary,
        not primary*, primary**secondary, primary^secondary, etc.
        Values are chosen so 5+3=8 differs from |, &, ^, *, **, <<, >>, -.
        Targets: line 1174 ReplaceBinaryOperator_Add_* survivors."""
        score = MaturityScore(
            overall_level=2, overall_label="Grounded Prompting",
            confidence=0.7, tools_detected=[], artifact_count=8,
            level_evidence={
                2: {"primary": 5, "secondary": 3},  # 5 + 3 = 8 (unique under +)
                3: {"primary": 2, "secondary": 0},  # 2 + 0 = 2; * → 0; ** → 1
                4: {"primary": 0, "secondary": 0},
            },
            category_counts={cat: 0 for cat in CATEGORY_NAMES},
            coherence_flags=[],
            file_classifications=pd.DataFrame([{"signals_agree": True}] * 8),
        )
        report = generate_report(score)
        assert report["level_stacking"]["L2"] == 8
        assert report["level_stacking"]["L3"] == 2  # kills *, **, <<, >>
        assert report["level_stacking"]["L4"] == 0

    def test_compute_confidence_count_score_thresholds(self):
        """_compute_confidence weights count_score by artifact count buckets:
        20+, 10+, 5+, <5. Use exact agreement_rate=0 and no flags so the
        count_score weight (0.35) is the only non-zero contributor.
        Targets: NumberReplacer survivors on the bucket bounds and weights."""
        # bucket >=20 → count_score = 1.0 → confidence = 0.35*1.0 + 0.30*1.0 = 0.65
        c20 = _compute_confidence(2, 20, 0.0, [])
        # bucket [10,20) → count_score = 0.8 → confidence = 0.35*0.8 + 0.30*1.0 = 0.58
        c10 = _compute_confidence(2, 10, 0.0, [])
        # bucket [5,10) → count_score = 0.6 → 0.35*0.6 + 0.30*1.0 = 0.51
        c5 = _compute_confidence(2, 5, 0.0, [])
        # bucket <5 → count_score = 0.4 → 0.35*0.4 + 0.30*1.0 = 0.44
        c1 = _compute_confidence(2, 1, 0.0, [])

        assert c20 == 0.65
        assert c10 == 0.58
        assert c5 == 0.51
        assert c1 == 0.44

        # Right at the boundaries: 19 falls into the [10,20) bucket.
        assert _compute_confidence(2, 19, 0.0, []) == 0.58
        assert _compute_confidence(2, 9, 0.0, []) == 0.51
        assert _compute_confidence(2, 4, 0.0, []) == 0.44

    def test_compute_confidence_red_yellow_penalties(self):
        """Coherence penalties: -0.2 per red, -0.1 per yellow, floored at 0.
        Pin the exact arithmetic to kill mutations that swap signs/values."""
        # 2 reds → coherence_score = max(0, 1 - 0.4) = 0.6
        # confidence = 0.35*1 + 0.35*1 + 0.30*0.6 = 0.88
        assert _compute_confidence(3, 20, 1.0, [
            CoherenceFlag("a", "red", "x"),
            CoherenceFlag("b", "red", "y"),
        ]) == 0.88

        # 1 red + 2 yellows → 1 - 0.2 - 0.2 = 0.6
        # confidence = 0.7 + 0.30*0.6 = 0.88 (same)
        assert _compute_confidence(3, 20, 1.0, [
            CoherenceFlag("a", "red", "x"),
            CoherenceFlag("b", "yellow", "y"),
            CoherenceFlag("c", "yellow", "z"),
        ]) == 0.88

        # 6 reds → coherence_score floored at 0 → confidence = 0.7 + 0 = 0.7
        c = _compute_confidence(3, 20, 1.0, [
            CoherenceFlag(str(i), "red", "x") for i in range(6)
        ])
        assert c == 0.7

    def test_check_coherence_l3_yellow_requires_l2_secondary_only(self):
        """L3 with L2 secondary AND L2 primary should NOT trigger the yellow
        'L3 without L2' message — only when l2_primary == 0 AND l2_secondary > 0.
        Targets: ReplaceAndWithOr / Eq_LtE survivors on line 775."""
        # Has both L2 primary AND L2 secondary → green path, not yellow
        flags = _check_coherence(
            {2: 3, 3: 5, 4: 0},
            {2: 4, 3: 0, 4: 0},
        )
        statuses = {f.check: f.status for f in flags}
        assert statuses.get("L3 builds on L2") == "green"
        # No 'L3 without L2' check should be in the flags
        assert "L3 without L2" not in statuses

    # Recommendation-edge-case mutation killers moved to test_report_generator.py
    # (recommendations now live in report_generator.generate_recommendations).

    def test_combine_signals_handles_short_signal_dataframes(self):
        """If a signal DataFrame is shorter than artifacts_df, combine_signals
        falls back to defaults for missing rows. Targets: defensive index
        guards `i < len(tool_signal)` etc. on lines 650/655/673."""
        artifacts_df = pd.DataFrame([
            {"file_id": "f1", "artifact_path": "a.md",
             "tool_name": "unknown", "discovery_step": "non_standard_other"},
            {"file_id": "f2", "artifact_path": "b.md",
             "tool_name": "unknown", "discovery_step": "non_standard_other"},
        ])
        # Signals only contain one row each
        tool_signal = pd.DataFrame([{"file_id": "f1", "tool_category": "rules"}])
        path_signal = pd.DataFrame([{
            "path_primary": "rules", "path_primary_score": 0.8,
            "path_secondary": "agents", "path_secondary_score": 0.6,
            "path_margin": 0.2,
        }])
        content_signal = pd.DataFrame([{
            "content_primary": "rules", "content_primary_score": 0.85,
            "content_secondary": "agents", "content_secondary_score": 0.6,
            "content_margin": 0.25, "hybrid_score": 1,
            "categories_within_threshold": "rules",
            **{f"content_{cat}": 0.0 for cat in CATEGORY_NAMES},
        }])
        result = combine_signals(artifacts_df, tool_signal, path_signal, content_signal)
        assert len(result) == 2
        # First file gets the signals
        assert result[0].assigned_category == "rules"
        # Second file falls back to defaults — no signals, no assignment
        assert result[1].tool_category is None
        assert result[1].content_primary is None
        assert result[1].path_primary is None
        assert result[1].assigned_category is None

    def test_combine_signals_skips_nan_content_scores(self):
        """Per-category NaN scores in content_signal must be skipped without
        populating fc.content_scores with NaN values. Targets: line 669
        `score is not None and not pd.isna(score)` mutation variants."""
        artifacts_df = pd.DataFrame([{
            "file_id": "f1", "artifact_path": "x.md",
            "tool_name": "unknown", "discovery_step": "non_standard_other",
        }])
        tool_signal = pd.DataFrame([{"file_id": "f1", "tool_category": None}])
        path_signal = pd.DataFrame([{
            "path_primary": None, "path_primary_score": 0.0,
            "path_secondary": None, "path_secondary_score": 0.0, "path_margin": 0.0,
        }])
        # Mix valid score, NaN, and None across categories
        content_row = {
            "content_primary": "rules", "content_primary_score": 0.85,
            "content_secondary": None, "content_secondary_score": 0.0,
            "content_margin": 0.0, "hybrid_score": 1,
            "categories_within_threshold": "rules",
        }
        for i, cat in enumerate(CATEGORY_NAMES):
            if i == 0:
                content_row[f"content_{cat}"] = 0.85
            elif i == 1:
                content_row[f"content_{cat}"] = float("nan")
            else:
                content_row[f"content_{cat}"] = 0.5
        content_signal = pd.DataFrame([content_row])
        result = combine_signals(artifacts_df, tool_signal, path_signal, content_signal)
        fc = result[0]
        # The first category got 0.85 → present.
        assert fc.content_scores.get(CATEGORY_NAMES[0]) == 0.85
        # The NaN category was skipped → not in content_scores.
        assert CATEGORY_NAMES[1] not in fc.content_scores
        # Sanity: no NaN snuck in.
        for v in fc.content_scores.values():
            assert not (isinstance(v, float) and pd.isna(v))

    def test_combine_signals_handles_nan_categories_within_threshold(self):
        """A NaN value in 'categories_within_threshold' must be coerced to an
        empty string and produce []. Targets: line 663 IsNot / AddNot /
        AndOr mutation variants."""
        artifacts_df = pd.DataFrame([{
            "file_id": "f1", "artifact_path": "x.md",
            "tool_name": "unknown", "discovery_step": "non_standard_other",
        }])
        tool_signal = pd.DataFrame([{"file_id": "f1", "tool_category": None}])
        path_signal = pd.DataFrame([{
            "path_primary": None, "path_primary_score": 0.0,
            "path_secondary": None, "path_secondary_score": 0.0, "path_margin": 0.0,
        }])
        content_signal = pd.DataFrame([{
            "content_primary": None, "content_primary_score": 0.0,
            "content_secondary": None, "content_secondary_score": 0.0,
            "content_margin": 0.0, "hybrid_score": 1,
            "categories_within_threshold": float("nan"),
            **{f"content_{cat}": 0.0 for cat in CATEGORY_NAMES},
        }])
        result = combine_signals(artifacts_df, tool_signal, path_signal, content_signal)
        assert result[0].categories_within_threshold == []

    def test_aggregate_includes_categories_within_threshold_secondary(self):
        """Verify level_evidence['secondary'] increments per within-threshold
        secondary category (not per file). Targets: ZeroIterationForLoop on
        line 914 — the `for cat in fc.categories_within_threshold` loop."""
        # Single file with two within-threshold non-assigned categories,
        # both at distinct levels (rules=L2, flows=L4).
        fc = _make_classification(
            file_id="f1", assigned_category="agents",
            content_primary="agents", path_primary="agents",
            tool_category="agents",
            categories_within_threshold=["agents", "rules", "flows"],
        )
        score = aggregate_repo_maturity([fc])
        assert score.level_evidence[3]["primary"] == 1
        assert score.level_evidence[2]["secondary"] == 1  # rules
        assert score.level_evidence[4]["secondary"] == 1  # flows

    def test_secondary_excludes_assigned_category_lexical_reverse(self):
        """Symmetric to test_secondary_excludes_assigned_category but with a
        lexically-LATER assigned category. The original test could not
        distinguish `cat != assigned` from `cat > assigned` because
        ('rules' > 'agents') == ('rules' != 'agents'). This case flips the
        ordering: assigned='rules' (L2), within_threshold has 'agents' (L3,
        lexically earlier). Targets: line 915 ReplaceComparisonOperator
        NotEq_Gt / NotEq_GtE / NotEq_IsNot survivors."""
        fc = _make_classification(
            file_id="f1", assigned_category="rules",
            tool_category="rules",
            content_primary="rules", path_primary="rules",
            categories_within_threshold=["rules", "agents"],
        )
        score = aggregate_repo_maturity([fc])
        # Original: 'agents' != 'rules' is True → agents added to L3 secondary.
        # Mutation `>`: 'agents' > 'rules' is False → agents NOT added.
        # We assert agents IS counted as secondary, which fails under mutation.
        assert score.level_evidence[3]["secondary"] == 1
        assert score.level_evidence[2]["secondary"] == 0  # rules is assigned, skipped

    def test_score_from_output_dir_aligns_real_embeddings(self, tmp_path=None):
        """When an embeddings PKL is present with non-empty file_ids, the
        function must take the alignment branch — not fall into the
        zero-padding else branch. Asserts non-zero content_primary_score for
        the aligned row, which only happens if real cosine similarity is
        computed. Targets: score_from_output_dir line 1270 (`len > 0`
        comparison variants and `is not None` swaps), and the
        ZeroIterationForLoop mutants on the alignment loops (1273, 1278)."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pd.io.common.Path(tmp) if hasattr(pd.io.common, 'Path') else None
            from pathlib import Path
            tmp_path = Path(tmp)
            repo_dir = tmp_path / "real_emb"
            repo_dir.mkdir()
            df = pd.DataFrame([{
                "file_id": "f1", "artifact_path": "CLAUDE.md",
                "artifact_name": "CLAUDE.md",
                "tool_name": "claude-code", "discovery_step": "tool_standard",
            }])
            df.to_csv(repo_dir / "real_emb_file_artifacts.csv", index=False)
            # Use a real, non-zero embedding for f1 so cosine similarity is non-zero.
            embs = np.ones((1, 768), dtype=np.float32)
            with open(repo_dir / "real_emb_embeddings.pkl", "wb") as f:
                pickle.dump({"embeddings": embs, "file_ids": ["f1"]}, f)

            with patch("src.maturity_scorer.classify_by_tool_detection") as mock_tool, \
                 patch("src.maturity_scorer.is_boilerplate") as mock_b:
                mock_b.return_value = False
                mock_tool.return_value = pd.DataFrame([
                    {"file_id": "f1", "tool_category": "rules"},
                ])
                model = _mock_model()
                score = score_from_output_dir(
                    str(tmp_path), "real_emb", model, config=PERMISSIVE_CONFIG,
                )
            # If the alignment branch ran, the file's content_primary_score is
            # non-zero (real cosine). If it fell into the else (zero pad), the
            # content_primary_score is 0.0.
            row = score.file_classifications.iloc[0]
            assert row["content_primary_score"] != 0.0
            # Sanity: the alignment loop produced a populated content score.
            assert row["content_primary"] is not None or row["content_primary_score"] > 0

    def test_score_from_output_dir_pkl_with_embeddings_but_empty_file_ids(self, tmp_path=None):
        """If the PKL has `embeddings` populated but `file_ids` is empty,
        the function must take the else branch (zero-padded content_signal)
        — not enter the alignment branch and feed zero vectors to cosine.
        Targets: line 1270 Gt_GtE / NumberReplacer (`> -1`) /
        ReplaceAndWithOr survivors. All three only differ when
        file_embeddings is not None AND embedding_file_ids is empty."""
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            repo_dir = tmp_path / "empty_ids"
            repo_dir.mkdir()
            df = pd.DataFrame([{
                "file_id": "f1", "artifact_path": "CLAUDE.md",
                "artifact_name": "CLAUDE.md",
                "tool_name": "claude-code", "discovery_step": "tool_standard",
            }])
            df.to_csv(repo_dir / "empty_ids_file_artifacts.csv", index=False)
            # Embeddings array present but file_ids list empty.
            embs = np.ones((1, 768), dtype=np.float32)
            with open(repo_dir / "empty_ids_embeddings.pkl", "wb") as f:
                pickle.dump({"embeddings": embs, "file_ids": []}, f)

            with patch("src.maturity_scorer.classify_by_tool_detection") as mock_tool, \
                 patch("src.maturity_scorer.is_boilerplate") as mock_b:
                mock_b.return_value = False
                mock_tool.return_value = pd.DataFrame([
                    {"file_id": "f1", "tool_category": "rules"},
                ])
                model = _mock_model()
                score = score_from_output_dir(
                    str(tmp_path), "empty_ids", model, config=PERMISSIVE_CONFIG,
                )
            row = score.file_classifications.iloc[0]
            # Original (else branch): content_primary is the empty-pad None.
            # Mutated (alignment branch with zero vectors): content_primary
            # would be populated by classify_by_content's argmax over
            # cosine-of-zeros (which is 0 / NaN, but argmax still picks a
            # category index → non-None primary under PERMISSIVE_CONFIG).
            assert row["content_primary"] is None or pd.isna(row["content_primary"])
            assert row["content_primary_score"] == 0.0

    def test_match_artifact_category_prefers_owning_tool_over_other(self):
        """The pattern lookup walks own-tool patterns first. If two tools
        define different categories for the same exact_path, the one matching
        the file's tool_name must win. Targets: list-priority survivors in
        _match_artifact_category."""
        from src.data_models import ToolConfig, ArtifactPattern, DiscoveryMethod, ArtifactStatus

        same_path_a = ArtifactPattern(
            pattern="X.md", type="file", description="x", file_type="md",
            status=ArtifactStatus.STABLE, is_standard=True,
            artifact_category="rules", scope="project",
            discovery_method=DiscoveryMethod.EXACT_PATH, exact_path="X.md",
        )
        same_path_b = ArtifactPattern(
            pattern="X.md", type="file", description="x", file_type="md",
            status=ArtifactStatus.STABLE, is_standard=True,
            artifact_category="agents", scope="project",
            discovery_method=DiscoveryMethod.EXACT_PATH, exact_path="X.md",
        )
        with patch("src.maturity_scorer.load_json_configs") as mock_load, \
             patch("src.maturity_scorer.load_shared_config") as mock_shared:
            mock_load.return_value = {
                "tool_a": ToolConfig(tool_name="tool_a", artifact_patterns=[same_path_a]),
                "tool_b": ToolConfig(tool_name="tool_b", artifact_patterns=[same_path_b]),
            }
            mock_shared.return_value = None
            from src.maturity_scorer import _build_pattern_lookup, _match_artifact_category
            lookup = _build_pattern_lookup("Artifacts")
            # Owning tool=tool_b → must resolve to "agents", not "rules".
            assert _match_artifact_category("X.md", "tool_b", lookup) == "agents"
            # Owning tool=tool_a → must resolve to "rules".
            assert _match_artifact_category("X.md", "tool_a", lookup) == "rules"
