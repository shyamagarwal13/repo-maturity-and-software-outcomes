"""PDF report generator for AIME maturity assessments.

Renders a Jinja2 HTML template with embedded Plotly chart PNGs (via kaleido),
then converts to PDF using WeasyPrint.
"""

import base64
import io
import os
import platform
import re
import subprocess
from datetime import datetime
from typing import Dict, Optional

import plotly.graph_objects as go
from jinja2 import Template

from src.maturity_scorer import (
    CATEGORY_TO_LEVEL,
    MATURITY_LABELS,
    MaturityLevel,
    MaturityScore,
)


# ============================================================================
# Constants
# ============================================================================

LEVEL_COLORS = {
    2: "#3b82f6",  # blue
    3: "#f97316",  # orange
    4: "#22c55e",  # green
}

LEVEL_DEFINITIONS = [
    {
        "level": "L1",
        "name": "Ad Hoc",
        "description": "No AI-specific configuration artifacts found. Developers may use AI tools but without shared, version-controlled guidance.",
        "examples": "No artifacts detected",
    },
    {
        "level": "L2",
        "name": "Grounded Prompting",
        "description": "Project-specific rules, coding standards, architecture docs, or tool configurations provide shared context for AI assistants.",
        "examples": "rules, configuration, architecture, code-style",
    },
    {
        "level": "L3",
        "name": "Agent-Augmented",
        "description": "Autonomous agent personas, reusable commands, or skill definitions enable AI to perform complex tasks independently.",
        "examples": "agents, commands, skills",
    },
    {
        "level": "L4",
        "name": "Agentic Orchestration",
        "description": "Multi-agent workflows, orchestration plans, and session logs coordinate multiple AI agents through complex, phased tasks.",
        "examples": "flows, session-logs",
    },
]

REPORT_CSS = """
@page {
    size: A4;
    margin: 2cm 1.5cm;
    @bottom-center {
        content: "Page " counter(page) " of " counter(pages);
        font-size: 9pt;
        color: #6b7280;
    }
}

body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    font-size: 11pt;
    line-height: 1.5;
    color: #1f2937;
    max-width: 100%;
}

h1 {
    font-size: 22pt;
    color: #111827;
    margin-top: 0;
    border-bottom: 3px solid #3b82f6;
    padding-bottom: 8px;
}

h2 {
    font-size: 16pt;
    color: #1f2937;
    margin-top: 24px;
    border-bottom: 1px solid #e5e7eb;
    padding-bottom: 4px;
}

h3 {
    font-size: 13pt;
    color: #374151;
    margin-top: 16px;
}

.title-page {
    text-align: center;
    padding-top: 120px;
    page-break-after: always;
}

.title-page h1 {
    font-size: 28pt;
    border-bottom: none;
    margin-bottom: 8px;
}

.title-page .subtitle {
    font-size: 14pt;
    color: #6b7280;
    margin-bottom: 40px;
}

.badge {
    display: inline-block;
    font-size: 36pt;
    font-weight: bold;
    padding: 16px 32px;
    border-radius: 12px;
    margin: 20px 0;
}

.badge-l1 { background: #f3f4f6; color: #6b7280; }
.badge-l2 { background: #dbeafe; color: #1d4ed8; }
.badge-l3 { background: #ffedd5; color: #c2410c; }
.badge-l4 { background: #dcfce7; color: #15803d; }

.confidence {
    font-size: 14pt;
    color: #6b7280;
}

.tools-list {
    font-size: 11pt;
    color: #6b7280;
    margin-top: 12px;
}

.page-break { page-break-before: always; }

table {
    width: 100%;
    border-collapse: collapse;
    margin: 12px 0;
    font-size: 10pt;
}

th {
    background: #f9fafb;
    border: 1px solid #e5e7eb;
    padding: 8px 10px;
    text-align: left;
    font-weight: 600;
}

td {
    border: 1px solid #e5e7eb;
    padding: 6px 10px;
}

tr:nth-child(even) td {
    background: #f9fafb;
}

.chart-container {
    margin: 16px 0;
    text-align: center;
}

.chart-container img {
    max-width: 100%;
    height: auto;
}

.chart-description {
    font-size: 10pt;
    color: #6b7280;
    font-style: italic;
    margin: 4px 0 16px 0;
}

.metric-box {
    display: inline-block;
    background: #f9fafb;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 12px 20px;
    margin: 6px 8px 6px 0;
    text-align: center;
}

.metric-value {
    font-size: 20pt;
    font-weight: bold;
    color: #111827;
}

.metric-label {
    font-size: 9pt;
    color: #6b7280;
}

.flag-green { color: #15803d; }
.flag-yellow { color: #a16207; }
.flag-red { color: #dc2626; }

.flag-icon::before {
    content: "";
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    margin-right: 6px;
}

.flag-green .flag-icon::before { background: #22c55e; }
.flag-yellow .flag-icon::before { background: #eab308; }
.flag-red .flag-icon::before { background: #ef4444; }

.recommendation {
    background: #eff6ff;
    border-left: 4px solid #3b82f6;
    padding: 8px 12px;
    margin: 8px 0;
    font-size: 10pt;
}

.disclaimer {
    background: #fefce8;
    border: 1px solid #fde68a;
    border-radius: 6px;
    padding: 10px 14px;
    font-size: 9pt;
    color: #92400e;
}

.llm-report {
    background: #f9fafb;
    border: 1px solid #e5e7eb;
    padding: 14px 18px;
    font-size: 10pt;
    page-break-inside: auto;
    box-decoration-break: clone;
    -webkit-box-decoration-break: clone;
}

.about-section p {
    margin: 6px 0;
    font-size: 10pt;
}

.concern {
    background: #fef2f2;
    border-left: 4px solid #ef4444;
    padding: 8px 12px;
    margin: 8px 0;
    font-size: 10pt;
}

.verdict-healthy {
    color: #15803d;
    font-weight: 600;
}

.verdict-warning {
    color: #a16207;
    font-weight: 600;
}

.verdict-concern {
    color: #dc2626;
    font-weight: 600;
}

.verdict-icon::before {
    content: "";
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    margin-right: 6px;
}

.verdict-healthy .verdict-icon::before { background: #22c55e; }
.verdict-warning .verdict-icon::before { background: #eab308; }
.verdict-concern .verdict-icon::before { background: #ef4444; }

.verdict-unknown {
    color: #6b7280;
    font-weight: 600;
}

.verdict-unknown .verdict-icon::before { background: #9ca3af; }

.interpretation {
    font-size: 10pt;
    color: #4b5563;
    margin: 4px 0 12px 0;
    padding: 6px 10px;
    background: #f9fafb;
    border-radius: 4px;
}
"""

REPORT_TEMPLATE_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>{{ css }}</style>
</head>
<body>

{# ======== PAGE 1: TITLE ======== #}
<div class="title-page">
    <h1>AI Adoption Maturity Assessment</h1>
    <div class="subtitle">AIME Framework</div>
    <div style="margin: 30px 0;">
        <div style="font-size: 14pt; color: #6b7280;">Repository</div>
        <div style="font-size: 20pt; font-weight: bold; color: #111827;">{{ repo_name }}</div>
    </div>
    <div class="badge badge-l{{ score.overall_level }}">
        L{{ score.overall_level }} &mdash; {{ score.overall_label }}
    </div>
    <div class="confidence">Confidence: {{ "%.0f"|format(score.confidence * 100) }}%</div>
    <div class="tools-list">
        {% if primary_tools %}
        Primary tool{{ 's' if primary_tools|length > 1 else '' }}: {{ primary_tools|join(', ') }}
        {% else %}
        Tools detected: none
        {% endif %}
    </div>
    <div style="margin-top: 40px; font-size: 10pt; color: #9ca3af;">
        Generated {{ date }} &bull; {{ score.artifact_count }} artifacts analyzed
    </div>
</div>

{# ======== PAGE 2: ABOUT ======== #}
<div class="page-break about-section">
    <h2>About This Report</h2>

    <h3>What is this document?</h3>
    <p>
        This report assesses a code repository's AI tool adoption maturity using the
        <strong>AIME</strong> (AI Adoption Maturity Evaluator) framework. It analyzes
        version-controlled AI configuration files to determine how deeply AI tools are
        integrated into the development workflow.
    </p>

    <h3>Methodology</h3>
    <p>The assessment uses three independent signals, combined for a final classification:</p>
    <table>
        <tr><th>Signal</th><th>Description</th></tr>
        <tr><td><strong>Tool Detection</strong></td><td>Pattern-matches files against known AI tool configurations (14+ tools).</td></tr>
        <tr><td><strong>Path Semantic Intent</strong></td><td>Embeds file paths and classifies against 9 category templates using cosine similarity.</td></tr>
        <tr><td><strong>Content Classification</strong></td><td>Embeds file contents with nomic-embed-text-v1.5 (768-dim) and classifies against the same category templates.</td></tr>
    </table>

    <h3>Disclaimer</h3>
    <div class="disclaimer">
        <strong>Limitations:</strong>
        <ul style="margin: 4px 0; padding-left: 20px;">
            <li>Automated classification may misattribute files whose content spans multiple categories.</li>
            <li>The confidence score reflects signal agreement, not ground truth accuracy.</li>
            <li>The maturity level is a snapshot in time based on the repository's current state.</li>
            <li>Categories are based on semantic similarity to template descriptions, not manual review.</li>
        </ul>
    </div>

    <h3>Maturity Level Definitions</h3>
    <table>
        <tr><th>Level</th><th>Name</th><th>Description</th><th>Example Categories</th></tr>
        {% for defn in level_definitions %}
        <tr>
            <td><strong>{{ defn.level }}</strong></td>
            <td>{{ defn.name }}</td>
            <td>{{ defn.description }}</td>
            <td><em>{{ defn.examples }}</em></td>
        </tr>
        {% endfor %}
    </table>
</div>

{# ======== PAGE 3-4: ASSESSMENT RESULTS ======== #}
<div class="page-break">
    <h2>Assessment Results</h2>

    <h3>Maturity Level Gauge</h3>
    <p class="chart-description">
        Shows the determined maturity level (L1&ndash;L4) with confidence score.
        The gauge reflects the highest level with at least one confirmed primary artifact.
    </p>
    {% if figures.gauge %}
    <div class="chart-container">
        <img src="data:image/png;base64,{{ figures.gauge }}" alt="Maturity Level Gauge">
    </div>
    {% endif %}
    <div class="interpretation">
        <strong>What this means for you:</strong>
        {% if score.overall_level == 4 %}
        Your repository has reached the highest maturity level. You have multi-agent workflows
        and orchestration patterns in place. Focus on maintaining coherence and documenting
        orchestration patterns for team onboarding.
        {% elif score.overall_level == 3 %}
        Your repository uses AI agents and reusable commands. To advance to L4, introduce
        multi-step workflow orchestration files that coordinate multiple agents through complex tasks.
        {% elif score.overall_level == 2 %}
        Your repository provides foundational context to AI tools (rules, standards, configuration).
        To advance to L3, define agent personas or reusable commands that let AI act autonomously.
        {% else %}
        No AI configuration artifacts were found. Start by adding a rules file (CLAUDE.md, .cursorrules)
        to give AI tools project-specific context.
        {% endif %}
        <br><br>
        <strong>Confidence:</strong>
        {% if score.confidence >= 0.8 %}
        <strong style="color: #15803d;">High ({{ "%.0f"|format(score.confidence * 100) }}%)</strong> &mdash;
        multiple signals agree and the evidence base is strong.
        {% elif score.confidence >= 0.5 %}
        <strong style="color: #a16207;">Moderate ({{ "%.0f"|format(score.confidence * 100) }}%)</strong> &mdash;
        the assessment is directionally correct but some signals are mixed. Review the coherence
        flags below for specific gaps.
        {% else %}
        <strong style="color: #dc2626;">Low ({{ "%.0f"|format(score.confidence * 100) }}%)</strong> &mdash;
        signals disagree significantly. The level may not fully reflect the repository's actual
        AI integration. Manual review recommended.
        {% endif %}
    </div>

    <h3>Level Stacking</h3>
    <p class="chart-description">
        Compares primary vs secondary (embedded) evidence at each level.
        Primary = file's main classification. Secondary = file contains content
        characteristic of this level as a secondary signal.
    </p>
    {% if figures.stacking %}
    <div class="chart-container">
        <img src="data:image/png;base64,{{ figures.stacking }}" alt="Level Stacking">
    </div>
    {% endif %}
    <div class="interpretation">
        <strong>What this means for you:</strong>
        A healthy maturity stack shows a broad L2 foundation (rules, architecture, coding standards)
        supporting a narrower L3 layer (agents, commands, skills), with L4 orchestration at the top.
        {% set l2p = score.level_evidence.get(2, {}).get('primary', 0) %}
        {% set l3p = score.level_evidence.get(3, {}).get('primary', 0) %}
        {% set l4p = score.level_evidence.get(4, {}).get('primary', 0) %}
        {% if l2p > 0 and l3p > 0 and l4p > 0 %}
        This repository has artifacts at all three active levels &mdash; progressive adoption is in place.
        {% elif l2p > 0 and l3p > 0 %}
        L2 and L3 are present. Consider adding workflow orchestration (L4) to coordinate your agents.
        {% elif l2p > 0 %}
        Only L2 grounding is present. Define agent personas or slash commands (L3) to expand AI capabilities.
        {% endif %}
        If secondary counts are much larger than primary, it means many files embed grounding
        content alongside their main purpose &mdash; this is normal for well-documented projects.
    </div>

    <h3>Level Evidence Summary</h3>
    <table>
        <tr><th>Level</th><th>Primary</th><th>Secondary</th><th>Total</th></tr>
        {% for lvl in [2, 3, 4] %}
        {% set ev = score.level_evidence.get(lvl, {}) %}
        {% set p = ev.get('primary', 0) %}
        {% set s = ev.get('secondary', 0) %}
        <tr>
            <td><strong>L{{ lvl }}</strong></td>
            <td>{{ p }}</td>
            <td>{{ s }}</td>
            <td>{{ p + s }}</td>
        </tr>
        {% endfor %}
    </table>
</div>

{# ======== PAGE 5-6: CATEGORY BREAKDOWN ======== #}
<div class="page-break">
    <h2>Category Breakdown</h2>

    <h3>Category Distribution</h3>
    <p class="chart-description">
        Breaks down artifacts into 9 categories, comparing content-based vs path-based classification.
        Categories are ordered by maturity level (L4 top, L2 bottom). Discrepancies between content
        and path signals suggest multi-purpose files or unconventional directory organization.
    </p>
    {% if figures.categories %}
    <div class="chart-container">
        <img src="data:image/png;base64,{{ figures.categories }}" alt="Category Distribution">
    </div>
    <div class="interpretation">
        <strong>What this means for you:</strong>
        When content-based and path-based bars are similar for a category, files are well-organized
        &mdash; their location matches their purpose. Large discrepancies indicate files that serve
        a different purpose than their directory name suggests. For example, a file in a "rules" directory
        whose content is primarily "architecture" may benefit from being reorganized &mdash; or it may
        intentionally serve dual purpose.
    </div>
    {% endif %}

    <h3>Signal Agreement Matrix</h3>
    <p class="chart-description">
        Heatmap showing how path-based and content-based classification agree.
        Strong diagonal = well-organized repository where file location matches content.
        Off-diagonal cells reveal files placed in unexpected locations or serving multiple purposes.
    </p>
    {% if figures.agreement %}
    <div class="chart-container">
        <img src="data:image/png;base64,{{ figures.agreement }}" alt="Signal Agreement Matrix">
    </div>
    {% endif %}

    {% if signal_agreement_rate is not none %}
    <div class="metric-box">
        <div class="metric-value">{{ "%.1f"|format(signal_agreement_rate * 100) }}%</div>
        <div class="metric-label">Signal Agreement Rate</div>
    </div>
    <div class="interpretation">
        <strong>How to read this:</strong> Signal agreement measures how often a file's
        directory location suggests the same category as its actual content. A rate of
        {{ "%.0f"|format(signal_agreement_rate * 100) }}% means that
        {{ "%.0f"|format(signal_agreement_rate * 100) }}% of files are placed in directories
        that match their content type.
        {% if signal_agreement_rate >= 0.6 %}
        This is a strong signal &mdash; the repository is well-organized with files
        consistently placed where their content would be expected.
        {% elif signal_agreement_rate >= 0.35 %}
        This is a moderate signal. Some files serve multiple purposes or are organized
        by criteria other than content type (e.g., by feature or team). This is common
        in repos with many multi-purpose configuration files.
        {% else %}
        This is a low signal, which is typical for repositories where AI artifacts are deeply
        embedded within feature directories rather than organized by type. The content-based
        classification (not path) drives the maturity assessment, so a low agreement rate does
        not reduce the accuracy of the overall score &mdash; it reflects organizational style.
        {% endif %}
    </div>
    {% endif %}
</div>

{# ======== PAGE 7: COMPOSITION ======== #}
<div class="page-break">
    <h2>Composition Analysis</h2>

    <h3>Maturity Composition Sunburst</h3>
    <p class="chart-description">
        Hierarchical view: inner ring shows maturity levels, outer ring shows categories within
        each level. Slice size = artifact count. Reveals which categories dominate at each level.
    </p>
    {% if figures.sunburst %}
    <div class="chart-container">
        <img src="data:image/png;base64,{{ figures.sunburst }}" alt="Maturity Composition Sunburst">
    </div>
    <div class="interpretation">
        <strong>What this means for you:</strong>
        A balanced sunburst shows multiple categories at each level. If one category dominates
        an entire level (e.g., all L3 artifacts are "skills" with no "agents" or "commands"),
        consider whether adding the missing categories would strengthen your AI setup.
        A thin or missing level ring indicates a gap in the maturity stack.
    </div>
    {% endif %}

    <h3>Hybrid Score Distribution</h3>
    <p class="chart-description">
        Shows how many categories each file matches within a 0.03 cosine similarity threshold.
        Files matching 1 category are clearly focused on a single purpose; files matching
        many categories are diffuse and harder for AI tools to classify reliably.
    </p>
    {% if figures.hybrid %}
    <div class="chart-container">
        <img src="data:image/png;base64,{{ figures.hybrid }}" alt="Hybrid Score Distribution">
    </div>
    <div class="interpretation">
        <strong>What this means for you:</strong>
        <table>
            <tr><th>Categories matched</th><th>Meaning</th><th>Action</th></tr>
            <tr>
                <td><strong style="color: #15803d;">1 (Focused)</strong></td>
                <td>File has a clear, single purpose. AI tools can classify it reliably.</td>
                <td>Ideal state &mdash; no action needed.</td>
            </tr>
            <tr>
                <td><strong style="color: #a16207;">2 (Moderate)</strong></td>
                <td>File serves two related purposes. Common for files that combine rules with architecture.</td>
                <td>Acceptable. Consider splitting if the file grows large.</td>
            </tr>
            <tr>
                <td><strong style="color: #dc2626;">3+ (Diffuse)</strong></td>
                <td>File content spans many categories. It may be a "kitchen sink" document that tries to do too much.</td>
                <td>Consider splitting into focused files &mdash; e.g., separate rules from architecture from code-style.
                    Focused files are easier for AI tools to use as context.</td>
            </tr>
        </table>
    </div>
    {% endif %}

    {% if category_concentration is not none %}
    <div class="metric-box">
        <div class="metric-value">{{ "%.0f"|format(category_concentration * 100) }}%</div>
        <div class="metric-label">Category Concentration</div>
    </div>
    <div class="interpretation">
        <strong>How to read this:</strong> Category concentration shows what percentage
        of artifacts belong to the single most common category
        {% if dominant_category %}({{ dominant_category }}){% endif %}.
        {{ "%.0f"|format(category_concentration * 100) }}% means artifacts are
        {% if category_concentration >= 0.6 %}
        heavily concentrated in one category. This suggests the repository's AI integration
        focuses on a single use case. Consider whether other categories could add value.
        {% elif category_concentration >= 0.35 %}
        moderately spread across categories, with one category somewhat more common.
        This is a typical pattern for repositories with focused but diversified AI tooling.
        {% else %}
        well-distributed across multiple categories. This indicates broad, balanced AI
        adoption spanning different types of artifacts &mdash; a sign of mature, diversified integration.
        {% endif %}
    </div>
    {% endif %}
</div>

{# ======== PAGE 8: TOOL ANALYSIS ======== #}
<div class="page-break">
    <h2>Tool Analysis</h2>

    {% if tool_artifact_counts %}
    <h3>Tool Artifact Breakdown</h3>
    <table>
        <tr><th>Tool</th><th>Artifacts</th><th>Share</th><th>Highest Category Level</th><th>Status</th></tr>
        {% for tool_info in tool_artifact_counts %}
        <tr>
            <td><strong>{{ tool_info.tool }}</strong></td>
            <td>{{ tool_info.count }}</td>
            <td>{{ "%.1f"|format(tool_info.share * 100) }}%</td>
            <td>{{ tool_info.highest_level }}</td>
            <td>
                {% if tool_info.is_negligible %}
                <span style="color: #dc2626;">Negligible</span>
                {% else %}
                <span style="color: #15803d;">Substantive</span>
                {% endif %}
            </td>
        </tr>
        {% endfor %}
    </table>
    <div class="interpretation">
        <strong>What this means for you:</strong>
        {% set substantive = tool_artifact_counts|rejectattr('is_negligible')|list %}
        {% set negligible = tool_artifact_counts|selectattr('is_negligible')|list %}
        {% if substantive|length == 1 %}
        This repository's AI maturity is driven entirely by <strong>{{ substantive[0].tool }}</strong>
        ({{ substantive[0].count }} artifacts, {{ "%.0f"|format(substantive[0].share * 100) }}% of total).
        {% if negligible|length > 0 %}
        {{ negligible|length }} other tool{{ 's' if negligible|length > 1 else '' }}
        ({{ negligible|map(attribute='tool')|join(', ') }})
        {{ 'were' if negligible|length > 1 else 'was' }} detected but with negligible artifact counts
        &mdash; likely false positives from pattern matching rather than actual AI tool usage.
        {% endif %}
        {% elif substantive|length > 1 %}
        Multiple AI tools contribute substantively:
        {{ substantive|map(attribute='tool')|join(', ') }}.
        This multi-tool setup can provide complementary capabilities, but ensure configurations
        don&rsquo;t conflict.
        {% else %}
        No substantive AI tool contributions were detected.
        {% endif %}
    </div>

    {% for tool_info in tool_artifact_counts %}
    {% if tool_info.is_negligible %}
    <div class="concern">
        <strong>Possible false positive: {{ tool_info.tool }}</strong> &mdash;
        only {{ tool_info.count }} artifact{{ 's' if tool_info.count != 1 else '' }}
        detected ({{ "%.1f"|format(tool_info.share * 100) }}% of total).
        {% if tool_info.highest_level_num <= 2 %}
        These are foundational (L2) artifacts only.
        {% endif %}
        With so few files, these may be standard repository files (e.g., CI configs,
        GitHub Actions workflows) that matched the tool's detection pattern rather than
        actual AI tool configurations. <strong>This tool is excluded from the maturity
        assessment.</strong>
        {% if tool_info.sample_paths %}
        <br>Matched files: <em>{{ tool_info.sample_paths|join(', ') }}</em>
        {% endif %}
    </div>
    {% endif %}
    {% endfor %}
    {% endif %}

    <h3>Tool &times; Category Heatmap</h3>
    <p class="chart-description">
        Shows which AI tools contribute which artifact categories.
        Only includes files with known tool attribution.
        Reveals tool specialization patterns.
    </p>
    {% if figures.tool_category %}
    <div class="chart-container">
        <img src="data:image/png;base64,{{ figures.tool_category }}" alt="Tool × Category Heatmap">
    </div>
    <div class="interpretation">
        <strong>What this means for you:</strong>
        A row with artifacts spread across many categories indicates a versatile tool
        that serves multiple purposes. A row concentrated in one or two categories
        indicates a specialized tool. If a tool appears only in L2 categories
        (rules, configuration, architecture, code-style), it provides grounding context
        but no agentic capabilities &mdash; consider whether that tool could also
        generate agent definitions or workflow files.
    </div>
    {% endif %}
</div>

{# ======== PAGE 9: HEALTH & COHERENCE ======== #}
<div class="page-break">
    <h2>Health &amp; Coherence</h2>

    <h3>Coherence Dashboard</h3>
    <p class="chart-description">
        Validates that maturity adoption is progressive. Green = healthy,
        Yellow = minor gap, Red = anomaly. The cumulative model expects
        L2 foundations before L3, and L3 before L4.
    </p>
    {% if figures.coherence %}
    <div class="chart-container">
        <img src="data:image/png;base64,{{ figures.coherence }}" alt="Coherence Dashboard">
    </div>
    {% endif %}
    <div class="interpretation">
        <strong>What this means for you:</strong>
        Coherence checks validate that your AI adoption follows a progressive path.
        The AIME model expects L2 foundations (rules, configuration) to be in place before
        L3 agents, and L3 agents before L4 orchestration &mdash; skipping levels suggests
        configuration gaps that may reduce AI tool effectiveness.
        {% set reds = score.coherence_flags|selectattr('status', 'equalto', 'red')|list %}
        {% set yellows = score.coherence_flags|selectattr('status', 'equalto', 'yellow')|list %}
        {% if reds|length > 0 %}
        <br><br><strong style="color: #dc2626;">Action needed:</strong>
        {{ reds|length }} check{{ 's' if reds|length > 1 else '' }} failed. This typically
        means a higher maturity level is present without the expected foundation &mdash;
        for example, orchestration files exist but no grounding rules.
        Review the flags below and add the missing foundational artifacts.
        {% elif yellows|length > 0 %}
        <br><br><strong style="color: #a16207;">Minor gaps detected:</strong>
        {{ yellows|length }} check{{ 's' if yellows|length > 1 else '' }} show warnings.
        The maturity stack is mostly healthy but could be strengthened.
        {% else %}
        <br><br><strong style="color: #15803d;">All checks passed.</strong>
        Your maturity stack is progressive and coherent &mdash; each level builds
        on the previous one as expected.
        {% endif %}
    </div>

    {% if score.coherence_flags %}
    <h3>Coherence Flags</h3>
    {% for flag in score.coherence_flags %}
    <p class="flag-{{ flag.status }}">
        <span class="flag-icon"></span>
        <strong>{{ flag.check }}</strong>: {{ flag.message }}
    </p>
    {% endfor %}
    {% endif %}

    {% if score.recommendations %}
    <h3>Recommendations</h3>
    {% for rec in score.recommendations %}
    <div class="recommendation">{{ rec }}</div>
    {% endfor %}
    {% endif %}
</div>

{# ======== PAGE 10 (OPTIONAL): TEMPORAL HEALTH ======== #}
{% if temporal_health and temporal_health.has_timeseries %}
<div class="page-break">
    <h2>Temporal Health</h2>
    <p class="chart-description">
        Analyzes how AI artifacts evolve over time using git commit history.
        Different artifact types have different expected update patterns &mdash;
        grounding artifacts (rules, architecture, configuration) should be living documents,
        while agentic artifacts and code-style configs can be stable.
    </p>

    <div class="metric-box">
        <div class="metric-value">{{ temporal_health.artifact_count }}</div>
        <div class="metric-label">Artifacts with commit history</div>
    </div>
    <div class="metric-box">
        <div class="metric-value">{{ temporal_health.total_classified }}</div>
        <div class="metric-label">Total classified</div>
    </div>
    {% if temporal_health.earliest_date and temporal_health.horizon_date %}
    <div class="interpretation">
        <strong>Scope:</strong>
        <strong>{{ temporal_health.artifact_count }}</strong> of
        {{ temporal_health.total_classified }} classified artifacts have git commit history
        ({{ temporal_health.earliest_date.strftime('%Y-%m-%d') }}
        to {{ temporal_health.horizon_date.strftime('%Y-%m-%d') }}).
        The remaining {{ temporal_health.total_classified - temporal_health.artifact_count }}
        artifacts have no recorded commits and appear as &ldquo;no-history&rdquo; below.
    </div>
    {% endif %}

    <h3>Artifact Lifecycle Classification</h3>
    <p class="chart-description">
        All {{ temporal_health.total_classified }} artifacts are classified:
        <strong>steady</strong> (regular updates), <strong>burst</strong> (all changes within 30 days),
        <strong>set-and-forget</strong> (single commit only),
        <strong>abandoned</strong> (no updates in 6+ months),
        or <strong>no-history</strong> (no recorded commits).
    </p>
    {% if figures.lifecycle_bars %}
    <div class="chart-container">
        <img src="data:image/png;base64,{{ figures.lifecycle_bars }}" alt="Artifact Lifecycle Classification">
    </div>
    {% endif %}

    <table>
        <tr><th>Category Tier</th><th>set-and-forget</th><th>burst</th><th>abandoned</th><th>steady</th></tr>
        <tr>
            <td><strong>Grounding</strong> (rules, architecture, configuration)</td>
            <td style="color: #dc2626;">concern</td>
            <td style="color: #a16207;">warning</td>
            <td style="color: #dc2626;">concern</td>
            <td style="color: #15803d;">healthy</td>
        </tr>
        <tr>
            <td><strong>Code-style</strong> (linting, formatting, style guides)</td>
            <td style="color: #15803d;">healthy</td>
            <td style="color: #15803d;">healthy</td>
            <td style="color: #15803d;">healthy</td>
            <td style="color: #15803d;">healthy</td>
        </tr>
        <tr>
            <td><strong>Agentic</strong> (agents, commands, skills)</td>
            <td style="color: #15803d;">healthy</td>
            <td style="color: #15803d;">healthy</td>
            <td style="color: #a16207;">warning</td>
            <td style="color: #15803d;">healthy</td>
        </tr>
        <tr>
            <td><strong>Flows</strong></td>
            <td style="color: #15803d;">healthy</td>
            <td style="color: #15803d;">healthy</td>
            <td style="color: #dc2626;">concern</td>
            <td style="color: #15803d;">healthy</td>
        </tr>
        <tr>
            <td><strong>Session-logs</strong></td>
            <td style="color: #15803d;">healthy</td>
            <td style="color: #15803d;">healthy</td>
            <td style="color: #15803d;">healthy</td>
            <td style="color: #15803d;">healthy</td>
        </tr>
    </table>

    <h3>Lifecycle by Creation Period</h3>
    <p class="chart-description">
        For the {{ temporal_health.artifact_count }} artifacts with commit history, grouped by the month they
        were first committed. Shows how lifecycle patterns differ between older and newer artifacts.
    </p>
    {% if figures.lifecycle_evolution %}
    <div class="chart-container">
        <img src="data:image/png;base64,{{ figures.lifecycle_evolution }}" alt="Lifecycle by Creation Period">
    </div>
    {% endif %}
    <div class="interpretation">
        <strong>What this means for you:</strong>
        Older cohorts naturally show more &ldquo;abandoned&rdquo; artifacts (they&rsquo;ve had more time
        to fall behind). Recent cohorts dominated by &ldquo;set-and-forget&rdquo; suggest new artifacts
        are being created without ongoing maintenance plans. A healthy pattern shows steady artifacts
        distributed across all periods.
    </div>

    {% if temporal_health.health_verdicts %}
    <h3>Health Verdicts</h3>
    {% for v in temporal_health.health_verdicts %}
    <p class="verdict-{{ v.verdict }}">
        <span class="verdict-icon"></span>
        {{ v.message }}
    </p>
    {% endfor %}
    {% endif %}

    <h3>Author Diversity by Category</h3>
    <p class="chart-description">
        Unique contributors per category across the full recorded git history.
        Only categories with commit history are shown.
        Single-author grounding categories represent a bus-factor risk &mdash;
        if that person leaves, foundational AI configuration knowledge may be lost.
    </p>
    {% if figures.author_diversity %}
    <div class="chart-container">
        <img src="data:image/png;base64,{{ figures.author_diversity }}" alt="Author Diversity by Category">
    </div>
    {% endif %}
    <div class="interpretation">
        <strong>What this means for you:</strong>
        {% set grounding_single_author = [] %}
        {% for cat, count in temporal_health.author_diversity.items() %}
            {% if count <= 1 and cat in ['rules', 'architecture', 'configuration'] %}
                {% if grounding_single_author.append(cat) %}{% endif %}
            {% endif %}
        {% endfor %}
        {% if grounding_single_author|length > 0 %}
        <strong style="color: #dc2626;">Bus-factor risk:</strong>
        {{ grounding_single_author|join(', ') }}
        {{ 'has' if grounding_single_author|length == 1 else 'have' }}
        only a single contributor. If that person leaves, these foundational AI configurations
        may become unmaintained. Consider sharing ownership of these artifacts across the team.
        {% else %}
        All grounding categories have multiple contributors &mdash; good knowledge distribution.
        {% endif %}
    </div>
</div>
{% endif %}

{# ======== PAGE 11 (OPTIONAL): LLM ANALYSIS ======== #}
{% if llm_report %}
<div class="page-break">
    <h2>AI-Generated Analysis</h2>
    <p style="font-size: 9pt; color: #6b7280;">Generated by Claude Sonnet 4.5</p>
    <div class="llm-report">
        {{ llm_report_html }}
    </div>
</div>
{% endif %}

</body>
</html>"""


# ============================================================================
# Public API
# ============================================================================

def fig_to_base64(fig: go.Figure, width: int = 900, height: int = 450) -> str:
    """Convert a Plotly figure to a base64-encoded PNG string for HTML embedding.

    Args:
        fig: Plotly figure object.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        Base64-encoded PNG string.
    """
    img_bytes = fig.to_image(format="png", width=width, height=height)
    return base64.b64encode(img_bytes).decode("utf-8")


def _markdown_to_html(text: str) -> str:
    """Minimal markdown-to-HTML conversion for LLM reports.

    Handles headers, bold, italic, lists, and paragraphs.
    """
    lines = text.split("\n")
    html_lines = []
    in_list = False

    for line in lines:
        stripped = line.strip()

        # Headers
        if stripped.startswith("### "):
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            html_lines.append(f"<h3>{stripped[4:]}</h3>")
            continue
        if stripped.startswith("## "):
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            html_lines.append(f"<h3>{stripped[3:]}</h3>")
            continue
        if stripped.startswith("# "):
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            html_lines.append(f"<h3>{stripped[2:]}</h3>")
            continue

        # List items
        if stripped.startswith("- ") or stripped.startswith("* "):
            if not in_list:
                html_lines.append("<ul>")
                in_list = True
            item = stripped[2:]
            item = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', item)
            item = re.sub(r'\*(.+?)\*', r'<em>\1</em>', item)
            item = re.sub(r'`(.+?)`', r'<code>\1</code>', item)
            html_lines.append(f"<li>{item}</li>")
            continue

        # End list if non-list line
        if in_list and not stripped:
            html_lines.append("</ul>")
            in_list = False

        # Empty line
        if not stripped:
            continue

        # Paragraph with inline formatting
        p = stripped
        p = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', p)
        p = re.sub(r'\*(.+?)\*', r'<em>\1</em>', p)
        p = re.sub(r'`(.+?)`', r'<code>\1</code>', p)
        html_lines.append(f"<p>{p}</p>")

    if in_list:
        html_lines.append("</ul>")

    return "\n".join(html_lines)


def _compute_tool_breakdown(score: MaturityScore) -> list:
    """Compute per-tool artifact counts and flag negligible contributors.

    A tool is considered "negligible" if it contributes less than 5% of total
    artifacts AND fewer than 10 artifacts in absolute terms.

    Returns:
        List of dicts with keys: tool, count, share, categories,
        highest_level, highest_level_num, is_negligible.
    """
    if score.file_classifications is None or score.file_classifications.empty:
        return []

    fc_df = score.file_classifications
    total = len(fc_df)
    if total == 0:
        return []

    # Only count known tools (not "unknown")
    known = fc_df[fc_df["tool_name"] != "unknown"]
    tool_groups = known.groupby("tool_name")

    results = []
    for tool_name, group in tool_groups:
        count = len(group)
        share = count / total
        categories = sorted(
            group["assigned_category"].dropna().unique().tolist()
        )
        # Find highest maturity level among this tool's artifacts
        levels = group["assigned_maturity_level"].dropna()
        highest_level_num = int(levels.max()) if not levels.empty else 0
        highest_level = f"L{highest_level_num}" if highest_level_num > 0 else "N/A"

        is_negligible = share < 0.05 and count < 10

        # Collect sample file paths for negligible tools (for false-positive review)
        sample_paths = []
        if is_negligible and "artifact_path" in group.columns:
            sample_paths = group["artifact_path"].head(5).tolist()

        results.append({
            "tool": tool_name,
            "count": count,
            "share": share,
            "categories": categories,
            "highest_level": highest_level,
            "highest_level_num": highest_level_num,
            "is_negligible": is_negligible,
            "sample_paths": sample_paths,
        })

    # Sort: substantive tools first, then by count descending
    results.sort(key=lambda x: (x["is_negligible"], -x["count"]))
    return results


def generate_pdf_report(
    score: MaturityScore,
    repo_name: str,
    figures: Dict[str, go.Figure],
    output_path: str,
    llm_report: Optional[str] = None,
    temporal_health=None,
) -> str:
    """Generate a single-repo PDF report.

    Args:
        score: MaturityScore object from the evaluation.
        repo_name: Name of the repository.
        figures: Dict mapping chart names to Plotly figure objects.
            Expected keys: gauge, stacking, categories, agreement,
            sunburst, tool_category, hybrid, coherence,
            lifecycle_bars, lifecycle_evolution, author_diversity.
        output_path: File path for the output PDF.
        llm_report: Optional markdown text from LLM analysis.
        temporal_health: Optional TemporalHealth object from temporal analysis.

    Returns:
        The output file path.
    """
    # Convert figures to base64 PNGs
    figure_b64 = {}
    chart_sizes = {
        "gauge": (500, 350),
        "stacking": (800, 300),
        "categories": (900, 450),
        "agreement": (600, 500),
        "sunburst": (600, 500),
        "tool_category": (800, 400),
        "hybrid": (600, 350),
        "coherence": (700, 300),
        "lifecycle_bars": (900, 450),
        "lifecycle_evolution": (900, 400),
        "author_diversity": (700, 350),
    }
    for name, fig in figures.items():
        if fig is not None:
            w, h = chart_sizes.get(name, (900, 450))
            figure_b64[name] = fig_to_base64(fig, width=w, height=h)

    # Compute derived metrics
    signal_agreement_rate = None
    category_concentration = None
    dominant_category = None

    if score.file_classifications is not None and not score.file_classifications.empty:
        total = score.artifact_count
        if total > 0:
            agreed = score.file_classifications["signals_agree"].sum()
            signal_agreement_rate = agreed / total

    if score.artifact_count > 0 and score.category_counts:
        max_count = max(score.category_counts.values())
        category_concentration = max_count / score.artifact_count
        dominant_category = max(score.category_counts, key=score.category_counts.get)

    # Compute per-tool artifact counts and identify negligible tools
    tool_artifact_counts = _compute_tool_breakdown(score)
    primary_tools = [t["tool"] for t in tool_artifact_counts if not t["is_negligible"]]

    # Convert LLM report markdown to HTML
    llm_report_html = _markdown_to_html(llm_report) if llm_report else None

    # Render HTML
    template = Template(REPORT_TEMPLATE_HTML)
    html_content = template.render(
        css=REPORT_CSS,
        repo_name=repo_name,
        score=score,
        date=datetime.now().strftime("%Y-%m-%d"),
        figures=figure_b64,
        level_definitions=LEVEL_DEFINITIONS,
        signal_agreement_rate=signal_agreement_rate,
        category_concentration=category_concentration,
        dominant_category=dominant_category,
        tool_artifact_counts=tool_artifact_counts,
        primary_tools=primary_tools,
        llm_report=llm_report,
        llm_report_html=llm_report_html,
        temporal_health=temporal_health,
    )

    # Generate PDF — defer weasyprint import so the module loads without pango
    HTML = _import_weasyprint()
    HTML(string=html_content).write_pdf(output_path)

    return output_path


def _import_weasyprint():
    """Import weasyprint.HTML, ensuring Homebrew libs are discoverable on macOS.

    WeasyPrint depends on pango/gobject via cffi.dlopen(). When Python is
    launched from an Anaconda env (or any env that doesn't inherit
    Homebrew's DYLD paths), those shared libraries can't be found.
    We detect Homebrew's lib dir and prepend it to DYLD_FALLBACK_LIBRARY_PATH
    before the first import.
    """
    dyld_key = "DYLD_FALLBACK_LIBRARY_PATH"
    if platform.system() == "Darwin" and dyld_key not in os.environ:
        try:
            brew_prefix = subprocess.check_output(
                ["brew", "--prefix"], text=True, stderr=subprocess.DEVNULL,
            ).strip()
            lib_dir = os.path.join(brew_prefix, "lib")
            if os.path.isdir(lib_dir):
                os.environ[dyld_key] = lib_dir
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass  # No Homebrew — user must ensure libs are on the path

    from weasyprint import HTML as _HTML
    return _HTML
