# Annotator instructions — AI artifact labeling

Thanks for taking part. You're labeling files that an automated pipeline
flagged as AI-tool configuration artifacts. Your label is the ground truth
we'll measure the pipeline's per-category precision and recall against.

This document defines the 11 label options and the protocol. Read it once,
then keep it open while labeling — the category line is short on purpose so
you can scan and decide.

---

## What you're seeing

Your package contains this document and your Excel workbook.

Each row in the workbook is one file, with four columns:

- `repo` — the source repository (context only).
- `file` — the full path of the file inside that repo. The path alone
  often suggests the label (e.g. `.claude/commands/foo.md` → `commands`).
- `content` — a GitHub link to the exact version of the file you're
  labeling (pinned to a specific commit). Click it to read the file.
- `category` — the dropdown you fill in (11 options).

**Every row needs a category** — there is no "unsure" option; when torn,
pick the dominant function.

Label what the file *is by its content*. If the content contradicts the
path (e.g. a README pasted into `.cursorrules`), trust the content.

---

## The 11 labels

Nine target categories plus `general-documentation` and `none`. Pick
exactly one. If a file is a hybrid, pick the *dominant* function — what
the file is *for*, not what it references. Examples are illustrative, not
exhaustive.

### `rules`

A policy document of imperative directives that govern how an AI assistant
must behave in a codebase. Mandatory language (NEVER, ALWAYS, MUST, DO NOT).
No code examples, no orchestration, no tutorials — only behavioral rules.

Examples: `CLAUDE.md`, `AGENTS.md`, `.cursorrules`, `.cursor/rules/*.mdc`,
`.github/copilot-instructions.md`, `GEMINI.md`, `.kiro/steering/*.md`,
`.junie/guidelines.md`.

### `configuration`

A machine-readable JSON / YAML / TOML / `.cursorrules`-style file with
hierarchical key-value pairs, boolean flags, and nested settings. Defines
MCP servers, env vars, permission scopes, file patterns, feature toggles.
**No prose paragraphs** — purely structured data.

Examples: `.claude/settings.local.json`, `.mcp.json`, `.cursor/mcp.json`,
`.windsurf/config.toml`, MCP server config files.

### `architecture`

A system-design document with component diagrams, data flows, deployment
topology. Mermaid / PlantUML / ASCII diagrams, ADR-style decision records,
C4 model levels. Covers infrastructure, service boundaries, scaling.
*Not* coding standards or runtime config.

Examples: `docs/architecture.md`, `docs/architecture/decisions/0001-*.md`,
`ARCHITECTURE.md`, system-design markdown with embedded diagrams.

### `code-style`

A coding-standards document with before/after code comparisons (incorrect
vs. correct), inline code snippets, linting rules, type-safety rules, naming
conventions with specific casing, coverage metrics. Focuses on **how source
code should be written at the syntax level**.

Examples: `docs/code-style.md`, `STYLE_GUIDE.md`, files with extensive
fenced code blocks showing pattern A → pattern B refactors. Distinguish
from `rules`: `code-style` shows *code*; `rules` shows *imperatives about
AI assistant behavior*.

### `agents`

A persona definition for *one* AI agent. YAML frontmatter with `name`,
`type`, `model`, `tools`, `capabilities`. Defines the agent's identity,
delegation boundaries, domain scope, interaction protocol.

Examples: `.claude/agents/code-reviewer.md`,
`.openhands/microagents/*.md`, files starting with YAML frontmatter that
describes a single agent's role.

### `commands`

A short, self-contained prompt template defining **exactly one** executable
action. Typically <25 lines, slash-command trigger, parameterized
`$ARGUMENTS`, single output. Atomic and reusable — *not* multi-step
orchestration.

Examples: `.claude/commands/commit.md`, `.claude/commands/review.md`,
`.github/prompts/*.prompt.md`. Distinguish from `skills` (long-form,
methodology) and `flows` (multi-phase orchestration).

### `skills`

A long-form how-to guide (200–600 lines typical) teaching **one technique
or capability** in depth. Includes trigger conditions, step-by-step method,
MCP tool usage, edge cases, validation criteria. Reusable domain expertise.

Examples: `.claude/skills/{topic}/SKILL.md` files with extended methodology
sections.

### `flows`

An **executable orchestration plan** consumed by an AI runner — names
specific phases with identifiers, assigns named agents/workers per phase,
declares dependency edges, lists per-phase exit criteria. The file directly
*drives* execution — a runner could parse it and dispatch work.

Examples: `.windsurf/workflows/*.md`, `.kiro/hooks/*.hook`. **NOT** a doc
explaining orchestration, NOT a tutorial on multi-agent systems, NOT a
release roadmap, NOT a recipe describing event sequences. Those *describe*
flows; this *is* one.

### `session-logs`

An actual log entry produced by a specific agent run — captures concrete
artifacts of that run: a run/session/task ID, real timestamps of state
transitions, names of files actually modified, real commit SHAs, the
agent's actor identity, outcome of acceptance criteria. The file is the
**output** of an executed run.

Examples: log entries with timestamps, real SHAs, agent IDs.
**NOT** a doc page describing how session logs work, NOT a tutorial on
agent memory, NOT an observability guide.

### `general-documentation`

User-facing software-project documentation written for end users,
contributors, or operators of a **non-AI** software system. Installation,
API reference, usage tutorials, troubleshooting, FAQs, deployment guides.
Describes how the project itself works — *does not* configure, instruct,
or orchestrate any AI assistant.

Examples: `README.md`, `INSTALL.md`, `CONTRIBUTING.md` (note: most of
these are filtered as boilerplate before they reach you, but some land
in this set).

### `none`

No AI artifact at all — the file shouldn't be classified as any of the
above. This is your "the algorithm got nothing useful out of this file"
option.

Use `none` for: raw datasets, license texts, third-party vendored docs,
release notes, generic blog posts that happen to live in the repo,
non-English content with no AI-tooling relevance, files that are clearly
*about* AI tooling (a marketing page) but aren't *configuration for* one.

---

## Protocol (2 steps per item)

1. **Read the path.** Most items are decided here. `.claude/commands/X.md`
   is `commands`; `CLAUDE.md` is `rules`; `docs/architecture.md` is
   `architecture`.
2. **Open the `content` link.** Skim the file on GitHub to confirm or
   override your path-based guess before committing the label.

**Work independently.** Please don't discuss specific items with the other
raters until everyone has returned their workbook — independent judgments
are the point.

When in doubt: pick the dominant function.

Thanks again — these labels are the ground truth that makes the rest of
the work credible.
