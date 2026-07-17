"""Tests for report_generator module."""

import base64
import os
import tempfile

import pandas as pd
import plotly.graph_objects as go
import pytest

from src.maturity_scorer import (
    CATEGORY_NAMES,
    CoherenceFlag,
    MaturityScore,
)
from src.report_generator import (
    LEVEL_DEFINITIONS,
    REPORT_CSS,
    REPORT_TEMPLATE_HTML,
    _compute_tool_breakdown,
    _markdown_to_html,
    fig_to_base64,
    generate_pdf_report,
    generate_recommendations,
)


# ============================================================================
# Test fixtures for generate_recommendations
# ============================================================================

def _score_for_recs(
    overall_level=2,
    level_primary=None,
    coherence_flags=None,
    category_counts=None,
    artifact_count=None,
):
    """Build a minimal MaturityScore that exercises generate_recommendations.

    Only the fields that generate_recommendations reads need realistic
    values; everything else gets a neutral default.
    """
    if level_primary is None:
        level_primary = {2: 0, 3: 0, 4: 0}
    if coherence_flags is None:
        coherence_flags = []
    if category_counts is None:
        category_counts = {cat: 0 for cat in CATEGORY_NAMES}
    if artifact_count is None:
        artifact_count = sum(category_counts.values())

    level_evidence = {
        lvl: {"primary": level_primary.get(lvl, 0), "secondary": 0}
        for lvl in (2, 3, 4)
    }
    return MaturityScore(
        overall_level=overall_level,
        overall_label=f"L{overall_level}",
        confidence=1.0,
        tools_detected=[],
        artifact_count=artifact_count,
        level_evidence=level_evidence,
        category_counts=category_counts,
        coherence_flags=coherence_flags,
    )


# ============================================================================
# Helpers
# ============================================================================

def _simple_figure() -> go.Figure:
    """Create a minimal Plotly figure for testing."""
    return go.Figure(go.Bar(x=[1, 2, 3], y=[4, 5, 6]))


def _make_score(with_fc=True) -> MaturityScore:
    """Create a minimal MaturityScore for testing."""
    fc_df = None
    if with_fc:
        fc_df = pd.DataFrame([
            {"file_id": "f1", "signals_agree": True, "assigned_category": "rules",
             "tool_name": "claude-code", "assigned_maturity_level": 2,
             "artifact_path": "CLAUDE.md"},
            {"file_id": "f2", "signals_agree": False, "assigned_category": "agents",
             "tool_name": "cursor", "assigned_maturity_level": 3,
             "artifact_path": ".cursorrules"},
        ])

    return MaturityScore(
        overall_level=3,
        overall_label="Agent-Augmented",
        confidence=0.75,
        tools_detected=["claude-code", "cursor"],
        artifact_count=2,
        level_evidence={
            2: {"primary": 1, "secondary": 2},
            3: {"primary": 1, "secondary": 0},
            4: {"primary": 0, "secondary": 0},
        },
        category_counts={cat: (1 if cat in ("rules", "agents") else 0) for cat in CATEGORY_NAMES},
        coherence_flags=[
            CoherenceFlag("L2 foundation", "green", "L2 grounding present"),
            CoherenceFlag("L3 builds on L2", "green", "Progressive adoption"),
        ],
        file_classifications=fc_df,
    )


def _make_figures() -> dict:
    """Create a minimal set of named figures for testing."""
    return {
        "gauge": _simple_figure(),
        "stacking": _simple_figure(),
        "categories": _simple_figure(),
        "agreement": _simple_figure(),
        "sunburst": _simple_figure(),
        "tool_category": _simple_figure(),
        "hybrid": _simple_figure(),
        "coherence": _simple_figure(),
    }


# ============================================================================
# Tests: fig_to_base64
# ============================================================================

class TestFigToBase64:

    def test_returns_valid_base64(self):
        fig = _simple_figure()
        result = fig_to_base64(fig)
        # Should be valid base64
        decoded = base64.b64decode(result)
        assert len(decoded) > 0

    def test_returns_png_header(self):
        fig = _simple_figure()
        result = fig_to_base64(fig)
        decoded = base64.b64decode(result)
        # PNG magic bytes: \x89PNG\r\n\x1a\n
        assert decoded[:4] == b'\x89PNG'

    def test_custom_dimensions(self):
        fig = _simple_figure()
        result = fig_to_base64(fig, width=400, height=200)
        decoded = base64.b64decode(result)
        assert decoded[:4] == b'\x89PNG'
        # Smaller dimensions should produce a smaller image
        result_large = fig_to_base64(fig, width=1200, height=800)
        assert len(result_large) > len(result)


# ============================================================================
# Tests: generate_pdf_report
# ============================================================================

class TestGeneratePdfReport:

    def test_creates_file(self):
        score = _make_score()
        figures = _make_figures()

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "test_report.pdf")
            result = generate_pdf_report(
                score=score,
                repo_name="test-repo",
                figures=figures,
                output_path=out_path,
            )
            assert result == out_path
            assert os.path.exists(out_path)
            assert os.path.getsize(out_path) > 0

    def test_pdf_starts_with_magic_bytes(self):
        score = _make_score()
        figures = _make_figures()

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "test_report.pdf")
            generate_pdf_report(
                score=score,
                repo_name="test-repo",
                figures=figures,
                output_path=out_path,
            )
            with open(out_path, "rb") as f:
                header = f.read(5)
            assert header == b"%PDF-"

    def test_without_llm_report(self):
        score = _make_score()
        figures = _make_figures()

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "test_no_llm.pdf")
            result = generate_pdf_report(
                score=score,
                repo_name="test-repo",
                figures=figures,
                output_path=out_path,
                llm_report=None,
            )
            assert os.path.exists(result)
            assert os.path.getsize(result) > 0

    def test_with_llm_report(self):
        score = _make_score()
        figures = _make_figures()
        llm_text = "## Summary\nThis repo is **great**.\n- Item 1\n- Item 2"

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "test_with_llm.pdf")
            result = generate_pdf_report(
                score=score,
                repo_name="test-repo",
                figures=figures,
                output_path=out_path,
                llm_report=llm_text,
            )
            assert os.path.exists(result)
            # File with LLM section should be larger than without
            size_with = os.path.getsize(result)

            out_path2 = os.path.join(tmpdir, "test_without_llm.pdf")
            generate_pdf_report(
                score=score,
                repo_name="test-repo",
                figures=figures,
                output_path=out_path2,
                llm_report=None,
            )
            size_without = os.path.getsize(out_path2)
            assert size_with > size_without

    def test_with_empty_figures(self):
        score = _make_score()
        figures = {}

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "test_empty_figs.pdf")
            result = generate_pdf_report(
                score=score,
                repo_name="test-repo",
                figures=figures,
                output_path=out_path,
            )
            assert os.path.exists(result)
            assert os.path.getsize(result) > 0

    def test_without_file_classifications(self):
        score = _make_score(with_fc=False)
        figures = _make_figures()

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "test_no_fc.pdf")
            result = generate_pdf_report(
                score=score,
                repo_name="test-repo",
                figures=figures,
                output_path=out_path,
            )
            assert os.path.exists(result)


# ============================================================================
# Tests: Template rendering
# ============================================================================

class TestTemplateRendering:

    def test_template_renders_all_sections(self):
        from jinja2 import Template

        score = _make_score()
        template = Template(REPORT_TEMPLATE_HTML)

        html = template.render(
            css=REPORT_CSS,
            repo_name="test-repo",
            score=score,
            date="2026-02-11",
            figures={},
            level_definitions=LEVEL_DEFINITIONS,
            signal_agreement_rate=0.5,
            category_concentration=0.3,
            dominant_category="rules",
            tool_artifact_counts=[],
            primary_tools=["claude-code", "cursor"],
            llm_report=None,
            llm_report_html=None,
            recommendations=generate_recommendations(score),
        )

        # Title page
        assert "AI Adoption Maturity Assessment" in html
        assert "test-repo" in html
        assert "Agent-Augmented" in html

        # About section
        assert "What is this document?" in html
        assert "Methodology" in html
        assert "Disclaimer" in html
        assert "Maturity Level Definitions" in html

        # Level definitions table
        assert "Ad Hoc" in html
        assert "Grounded Prompting" in html
        assert "Agent-Augmented" in html
        assert "Agentic Orchestration" in html

        # Chart sections
        assert "Maturity Level Gauge" in html
        assert "Level Stacking" in html
        assert "Category Distribution" in html
        assert "Signal Agreement Matrix" in html
        assert "Maturity Composition Sunburst" in html
        assert "Hybrid Score Distribution" in html
        assert "Coherence Dashboard" in html

        # Metrics
        assert "50.0%" in html  # signal agreement rate
        assert "30%" in html  # category concentration

        # Coherence flags
        assert "L2 foundation" in html
        assert "L3 builds on L2" in html

        # Recommendations: _make_score() is L3 with no L4 artifacts, so the
        # advancement rec ("Advance to L4 ...") fires.
        assert "Recommendations" in html
        assert "Advance to L4" in html

    def test_template_renders_llm_section_when_provided(self):
        from jinja2 import Template

        score = _make_score()
        template = Template(REPORT_TEMPLATE_HTML)

        html = template.render(
            css=REPORT_CSS,
            repo_name="test-repo",
            score=score,
            date="2026-02-11",
            figures={},
            level_definitions=LEVEL_DEFINITIONS,
            signal_agreement_rate=0.5,
            category_concentration=0.3,
            dominant_category="rules",
            tool_artifact_counts=[],
            primary_tools=["claude-code"],
            llm_report="Some analysis text",
            llm_report_html="<p>Some analysis text</p>",
        )

        assert "AI-Generated Analysis" in html
        assert "Claude Sonnet 4.5" in html
        assert "Some analysis text" in html

    def test_template_omits_llm_section_when_none(self):
        from jinja2 import Template

        score = _make_score()
        template = Template(REPORT_TEMPLATE_HTML)

        html = template.render(
            css=REPORT_CSS,
            repo_name="test-repo",
            score=score,
            date="2026-02-11",
            figures={},
            level_definitions=LEVEL_DEFINITIONS,
            signal_agreement_rate=0.5,
            category_concentration=0.3,
            dominant_category="rules",
            tool_artifact_counts=[],
            primary_tools=["claude-code", "cursor"],
            llm_report=None,
            llm_report_html=None,
        )

        assert "AI-Generated Analysis" not in html

    def test_primary_tools_displayed(self):
        from jinja2 import Template

        score = _make_score()
        template = Template(REPORT_TEMPLATE_HTML)

        html = template.render(
            css=REPORT_CSS,
            repo_name="test-repo",
            score=score,
            date="2026-02-11",
            figures={},
            level_definitions=LEVEL_DEFINITIONS,
            signal_agreement_rate=None,
            category_concentration=None,
            dominant_category=None,
            tool_artifact_counts=[],
            primary_tools=["claude-code"],
            llm_report=None,
            llm_report_html=None,
        )

        assert "claude-code" in html
        assert "Primary tool" in html

    def test_interpretation_text_for_agreement(self):
        from jinja2 import Template

        score = _make_score()
        template = Template(REPORT_TEMPLATE_HTML)

        html = template.render(
            css=REPORT_CSS,
            repo_name="test-repo",
            score=score,
            date="2026-02-11",
            figures={},
            level_definitions=LEVEL_DEFINITIONS,
            signal_agreement_rate=0.29,
            category_concentration=0.3,
            dominant_category="skills",
            tool_artifact_counts=[],
            primary_tools=["claude-code"],
            llm_report=None,
            llm_report_html=None,
        )

        assert "How to read this" in html
        assert "29%" in html

    def test_tool_breakdown_with_negligible(self):
        from jinja2 import Template

        score = _make_score()
        template = Template(REPORT_TEMPLATE_HTML)

        tool_counts = [
            {"tool": "claude-code", "count": 2000, "share": 0.998,
             "categories": ["rules", "agents", "flows"], "highest_level": "L4",
             "highest_level_num": 4, "is_negligible": False, "sample_paths": []},
            {"tool": "github-copilot", "count": 3, "share": 0.002,
             "categories": ["configuration"], "highest_level": "L2",
             "highest_level_num": 2, "is_negligible": True,
             "sample_paths": [".github/workflows/ci.yml", ".github/ISSUE_TEMPLATE/bug.md"]},
        ]

        html = template.render(
            css=REPORT_CSS,
            repo_name="test-repo",
            score=score,
            date="2026-02-11",
            figures={},
            level_definitions=LEVEL_DEFINITIONS,
            signal_agreement_rate=0.3,
            category_concentration=0.3,
            dominant_category="skills",
            tool_artifact_counts=tool_counts,
            primary_tools=["claude-code"],
            llm_report=None,
            llm_report_html=None,
        )

        assert "Tool Artifact Breakdown" in html
        assert "Negligible" in html
        assert "Substantive" in html
        assert "github-copilot" in html
        assert "false positive" in html.lower()
        assert "excluded from the maturity" in html
        assert ".github/workflows/ci.yml" in html


# ============================================================================
# Tests: _markdown_to_html
# ============================================================================

class TestMarkdownToHtml:

    def test_headers(self):
        assert "<h3>" in _markdown_to_html("## Summary")
        assert "<h3>" in _markdown_to_html("### Details")
        assert "<h3>" in _markdown_to_html("# Top")

    def test_bold(self):
        result = _markdown_to_html("This is **bold** text")
        assert "<strong>bold</strong>" in result

    def test_italic(self):
        result = _markdown_to_html("This is *italic* text")
        assert "<em>italic</em>" in result

    def test_code(self):
        result = _markdown_to_html("Use `code` here")
        assert "<code>code</code>" in result

    def test_list_items(self):
        result = _markdown_to_html("- Item 1\n- Item 2")
        assert "<ul>" in result
        assert "<li>Item 1</li>" in result
        assert "<li>Item 2</li>" in result
        assert "</ul>" in result

    def test_paragraphs(self):
        result = _markdown_to_html("Hello world")
        assert "<p>Hello world</p>" in result


# ============================================================================
# Tests: _compute_tool_breakdown
# ============================================================================

class TestComputeToolBreakdown:

    def test_basic_breakdown(self):
        score = _make_score()
        result = _compute_tool_breakdown(score)
        assert len(result) == 2
        tools = {r["tool"] for r in result}
        assert "claude-code" in tools
        assert "cursor" in tools

    def test_negligible_detection(self):
        """Tool with <5% share and <10 absolute count is negligible."""
        fc_df = pd.DataFrame(
            [{"file_id": f"f{i}", "signals_agree": True,
              "assigned_category": "rules", "tool_name": "claude-code",
              "assigned_maturity_level": 2,
              "artifact_path": f"CLAUDE.md"} for i in range(100)]
            + [{"file_id": "f100", "signals_agree": True,
                "assigned_category": "configuration", "tool_name": "copilot",
                "assigned_maturity_level": 2,
                "artifact_path": ".github/workflows/ci.yml"}]
        )
        score = MaturityScore(
            overall_level=2, overall_label="Grounded",
            confidence=0.7, tools_detected=["claude-code", "copilot"],
            artifact_count=101,
            level_evidence={2: {"primary": 101, "secondary": 0}},
            category_counts={"rules": 100, "configuration": 1},
            coherence_flags=[],
            file_classifications=fc_df,
        )
        result = _compute_tool_breakdown(score)
        copilot = [r for r in result if r["tool"] == "copilot"][0]
        assert copilot["is_negligible"] is True
        assert copilot["sample_paths"] == [".github/workflows/ci.yml"]
        claude = [r for r in result if r["tool"] == "claude-code"][0]
        assert claude["is_negligible"] is False
        assert claude["sample_paths"] == []  # only populated for negligible

    def test_excludes_unknown_tools(self):
        fc_df = pd.DataFrame([
            {"file_id": "f1", "signals_agree": True, "assigned_category": "rules",
             "tool_name": "unknown", "assigned_maturity_level": 2},
            {"file_id": "f2", "signals_agree": True, "assigned_category": "agents",
             "tool_name": "claude-code", "assigned_maturity_level": 3},
        ])
        score = MaturityScore(
            overall_level=3, overall_label="Agent-Augmented",
            confidence=0.7, tools_detected=["claude-code"],
            artifact_count=2,
            level_evidence={}, category_counts={},
            coherence_flags=[],
            file_classifications=fc_df,
        )
        result = _compute_tool_breakdown(score)
        assert len(result) == 1
        assert result[0]["tool"] == "claude-code"

    def test_no_file_classifications(self):
        score = _make_score(with_fc=False)
        result = _compute_tool_breakdown(score)
        assert result == []

    def test_sorted_substantive_first(self):
        """Substantive tools should come before negligible ones."""
        fc_df = pd.DataFrame(
            [{"file_id": f"f{i}", "signals_agree": True,
              "assigned_category": "rules", "tool_name": "claude-code",
              "assigned_maturity_level": 2} for i in range(50)]
            + [{"file_id": "fx", "signals_agree": True,
                "assigned_category": "configuration", "tool_name": "tiny-tool",
                "assigned_maturity_level": 2}]
        )
        score = MaturityScore(
            overall_level=2, overall_label="Grounded",
            confidence=0.7, tools_detected=["claude-code", "tiny-tool"],
            artifact_count=51,
            level_evidence={}, category_counts={},
            coherence_flags=[],
            file_classifications=fc_df,
        )
        result = _compute_tool_breakdown(score)
        assert result[0]["tool"] == "claude-code"
        assert result[0]["is_negligible"] is False
        assert result[1]["is_negligible"] is True


# ============================================================================
# Test generate_recommendations
# ============================================================================

class TestGenerateRecommendations:

    def test_l1_recommends_grounding(self):
        recs = generate_recommendations(_score_for_recs(overall_level=1))
        assert len(recs) >= 1
        assert any("rules" in r.lower() or "instructions" in r.lower() for r in recs)

    def test_l2_recommends_agents(self):
        cats = {"rules": 3, "configuration": 2,
                **{c: 0 for c in CATEGORY_NAMES if c not in ("rules", "configuration")}}
        recs = generate_recommendations(_score_for_recs(
            overall_level=2,
            level_primary={2: 5, 3: 0, 4: 0},
            category_counts=cats,
        ))
        assert any("agent" in r.lower() or "l3" in r.lower() for r in recs)

    def test_concentration_warning(self):
        cats = {cat: 0 for cat in CATEGORY_NAMES}
        cats["commands"] = 20
        cats["rules"] = 1
        recs = generate_recommendations(_score_for_recs(
            overall_level=3,
            level_primary={2: 1, 3: 20, 4: 0},
            category_counts=cats,
        ))
        assert any("concentrated" in r.lower() for r in recs)

    def test_red_l3_without_l2_creates_recommendation(self):
        recs = generate_recommendations(_score_for_recs(
            overall_level=3,
            level_primary={2: 0, 3: 10, 4: 0},
            coherence_flags=[CoherenceFlag("L3 without L2", "red", "anomaly")],
        ))
        assert any("grounding" in r.lower() or "l2" in r.lower() for r in recs)

    def test_red_l4_without_l3_creates_recommendation(self):
        recs = generate_recommendations(_score_for_recs(
            overall_level=4,
            level_primary={2: 5, 3: 0, 4: 2},
            coherence_flags=[CoherenceFlag("L4 without L3", "red", "anomaly")],
        ))
        assert any("agent" in r.lower() and ("l3" in r.lower() or "orchestration" in r.lower())
                   for r in recs)

    def test_yellow_l2_foundation_creates_recommendation(self):
        cats = {"agents": 5, **{c: 0 for c in CATEGORY_NAMES if c != "agents"}}
        recs = generate_recommendations(_score_for_recs(
            overall_level=3,
            level_primary={2: 0, 3: 5, 4: 0},
            coherence_flags=[CoherenceFlag("L2 foundation", "yellow", "embedded only")],
            category_counts=cats,
        ))
        assert any("extracting" in r.lower() or "claude.md" in r.lower() or "cursorrules" in r.lower()
                   for r in recs)

    def test_l3_recommends_l4(self):
        cats = {cat: 0 for cat in CATEGORY_NAMES}
        cats["agents"] = 3
        recs = generate_recommendations(_score_for_recs(
            overall_level=3,
            level_primary={2: 1, 3: 3, 4: 0},
            category_counts=cats,
        ))
        assert any("flow" in r.lower() or "l4" in r.lower() for r in recs)

    def test_grounding_rec_requires_red_status(self):
        """Yellow 'L3 without L2' or unrelated red flags must NOT trigger
        the standalone-grounding-files recommendation."""
        recs_yellow = generate_recommendations(_score_for_recs(
            overall_level=3,
            level_primary={2: 0, 3: 5, 4: 0},
            coherence_flags=[CoherenceFlag("L3 without L2", "yellow", "x")],
        ))
        assert not any("standalone grounding files" in r.lower() for r in recs_yellow)

        recs_unrelated = generate_recommendations(_score_for_recs(
            overall_level=3,
            level_primary={2: 5, 3: 5, 4: 0},
            coherence_flags=[CoherenceFlag("Some other check", "red", "x")],
        ))
        assert not any("standalone grounding files" in r.lower() for r in recs_unrelated)

    def test_l2_advancement_suppressed_when_agents_or_commands_present(self):
        """The 'Advance to L3' rec must NOT fire if either agents or commands
        is non-zero."""
        cats_a = {cat: 0 for cat in CATEGORY_NAMES}
        cats_a["agents"] = 1
        cats_a["rules"] = 5
        recs = generate_recommendations(_score_for_recs(
            overall_level=2,
            level_primary={2: 5, 3: 0, 4: 0},
            category_counts=cats_a,
        ))
        assert not any("advance to l3" in r.lower() for r in recs)

        cats_c = {cat: 0 for cat in CATEGORY_NAMES}
        cats_c["commands"] = 1
        cats_c["rules"] = 5
        recs = generate_recommendations(_score_for_recs(
            overall_level=2,
            level_primary={2: 5, 3: 0, 4: 0},
            category_counts=cats_c,
        ))
        assert not any("advance to l3" in r.lower() for r in recs)

    def test_concentration_threshold_boundaries(self):
        """Concentration warning fires only when (max/total) > 0.8 AND total >= 5."""
        # 4/5 = 0.8 → NOT > 0.8 → no warning
        cats = {cat: 0 for cat in CATEGORY_NAMES}
        cats["commands"] = 4
        cats["rules"] = 1
        recs = generate_recommendations(_score_for_recs(
            overall_level=3,
            level_primary={2: 1, 3: 4, 4: 0},
            category_counts=cats,
        ))
        assert not any("concentrated" in r.lower() for r in recs)

        # 5/5 = 1.0 → > 0.8 → warning fires
        cats2 = {cat: 0 for cat in CATEGORY_NAMES}
        cats2["commands"] = 5
        recs2 = generate_recommendations(_score_for_recs(
            overall_level=3,
            level_primary={2: 0, 3: 5, 4: 0},
            category_counts=cats2,
        ))
        assert any("concentrated" in r.lower() for r in recs2)

        # total=4 (below 5) — no warning even at 100% concentration
        cats3 = {cat: 0 for cat in CATEGORY_NAMES}
        cats3["commands"] = 4
        recs3 = generate_recommendations(_score_for_recs(
            overall_level=3,
            level_primary={2: 0, 3: 4, 4: 0},
            category_counts=cats3,
        ))
        assert not any("concentrated" in r.lower() for r in recs3)
