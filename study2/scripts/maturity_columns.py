#!/usr/bin/env python3
"""
Add maturity level columns (l1–l4, l2+, subsets) to panel_event_monthly.csv.

Labels come from the **last calendar month column** in
`msrc_maturity_timeseries_wide.csv` (or `--label-column`), using the cell value
as MSRC level 1–4. Wide `repo` uses owner__repo → panel owner/repo. Only repos
that appear in agent_first.txt ∪ ide_first.txt get a level; values outside
{1,2,3,4} (e.g. -1, NA) omit that repo from the level map.

The CLI flag **`--old`** uses default inputs under **`data/data_old/`** (instead of
**`data/`**) and maps levels from **`repos-file - November 2025_org_maturity_scores.csv`**
(repo → level) instead of the MSRC wide sheet.

**`--new`** uses default inputs under **`data/data_new/`** (still MSRC wide → levels,
same pipeline as defaults under **`data/`**).

**`--llm`** uses defaults under **`data/data_llm/`** (panel, wide, matching,
``agent_first.txt``, ``ide_first.txt``).

**`--final`** uses the same layout under **`data/data_final/`**.

Legacy helper: `_build_full_repo_to_level` (repos-file CSV) is used when
`maturity_scores_path` is set.
"""

import argparse
from collections import Counter
from pathlib import Path
from typing import Optional, Union

import pandas as pd  # type: ignore[import-not-found]

BASE_DIR = Path(__file__).resolve().parent.parent  # study2/


def _resolve_from_base(path_like: Union[str, Path]) -> str:
    """Resolve relative paths against the replication package's `study2/` directory."""
    p = Path(path_like)
    if p.is_absolute():
        return str(p)
    return str((BASE_DIR / p).resolve())


def _extract_repo_short(full_name):
    """Basename after last '/' (maturity CSV and list files use this to align)."""
    if pd.isna(full_name) or full_name == "":
        return full_name
    s = str(full_name)
    if "/" in s:
        return s.split("/")[-1]
    return s


def _load_full_repo_names(agent_first_path, ide_first_path):
    """Union of owner/repo lines from agent_first.txt and ide_first.txt."""
    paths = [agent_first_path, ide_first_path]
    full_names = set()
    for p in paths:
        text = Path(p).read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            line = line.strip()
            if line:
                full_names.add(line)
    return full_names


def _maturity_ambiguous_short_names(df_maturity):
    """
    Basenames that appear both as a bare `repo` cell and as the part after
    `owner__` in a disambiguated row (e.g. cli, csv, docs, frontend). Bare rows
    for these must be ignored so we only use explicit owner__repo lines.
    """
    bare = set()
    suffix_after_double_underscore = set()
    for _, row in df_maturity.iterrows():
        r = row["repo"]
        if pd.isna(r):
            continue
        s = str(r).strip()
        if not s:
            continue
        if "__" in s:
            _, rest = s.split("__", 1)
            if rest:
                suffix_after_double_underscore.add(rest)
        else:
            bare.add(s)
    return bare & suffix_after_double_underscore


def _wide_repo_to_panel_repo(wide_repo_cell) -> Optional[str]:
    """Wide `owner__repo` → panel `owner/repo` (first `__` only)."""
    if pd.isna(wide_repo_cell):
        return None
    s = str(wide_repo_cell).strip()
    if not s or "__" not in s:
        return None
    return s.replace("__", "/", 1)


def _pick_last_month_column(df_wide: pd.DataFrame) -> str:
    """Rightmost column that parses as a calendar date (excluding `repo`)."""
    candidates: list[tuple[pd.Timestamp, str]] = []
    for c in df_wide.columns:
        if c == "repo":
            continue
        try:
            ts = pd.to_datetime(c)
        except (ValueError, TypeError):
            continue
        candidates.append((pd.Timestamp(ts), str(c)))
    if not candidates:
        raise ValueError(
            "No parseable date columns in MSRC wide file (need YYYY-MM-DD headers besides `repo`)."
        )
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def _build_full_repo_to_level_from_msrc_wide(
    wide_path: Union[str, Path],
    full_repo_set: set,
    label_column: Optional[str] = None,
) -> tuple[dict[str, int], str]:
    """
    Map full owner/repo -> level in {1,2,3,4} from one column of the wide MSRC CSV.

    Returns (full_repo_to_level, label_col_name_used).
    """
    df = pd.read_csv(wide_path, header=0)
    if "repo" not in df.columns:
        raise ValueError(f"MSRC wide file missing `repo` column: {wide_path}")

    label_col = label_column.strip() if label_column else None
    if label_col:
        if label_col not in df.columns:
            raise ValueError(
                f"--label-column {label_col!r} not in wide file columns "
                f"(examples: {list(df.columns)[:5]} …)"
            )
    else:
        label_col = _pick_last_month_column(df)

    full_repo_to_level: dict[str, int] = {}
    for _, row in df.iterrows():
        panel_repo = _wide_repo_to_panel_repo(row["repo"])
        if panel_repo is None or panel_repo not in full_repo_set:
            continue
        raw = row[label_col]
        if pd.isna(raw):
            continue
        try:
            k = int(float(raw))
        except (TypeError, ValueError):
            continue
        if k in (1, 2, 3, 4):
            full_repo_to_level[panel_repo] = k

    return full_repo_to_level, label_col


def _full_name_from_maturity_repo_cell(repo_cell):
    """
    If repo looks like owner__repo (double underscore), return owner/repo.
    Otherwise return None.
    """
    if pd.isna(repo_cell):
        return None
    s = str(repo_cell).strip()
    if not s or "__" not in s:
        return None
    owner, rest = s.split("__", 1)
    if not owner or not rest:
        return None
    return f"{owner}/{rest}"


def _build_full_repo_to_level(df_maturity, full_repo_set):
    """
    Map full owner/repo -> level.

    - Rows with `owner__repo` in the repo column map directly to owner/repo when
      that string is in full_repo_set.
    - Other rows use short `repo` and assign to every full name in full_repo_set
      with that basename, except bare short names that are ambiguous (see
      _maturity_ambiguous_short_names) — those rows are skipped.
    """
    short_to_fulls = {}
    for fn in full_repo_set:
        short = _extract_repo_short(fn)
        short_to_fulls.setdefault(short, []).append(fn)

    ambiguous_short = _maturity_ambiguous_short_names(df_maturity)
    full_repo_to_level = {}

    for _, row in df_maturity.iterrows():
        repo_cell = row["repo"]
        level = row["level"]

        explicit = _full_name_from_maturity_repo_cell(repo_cell)
        if explicit is not None:
            if explicit in full_repo_set:
                full_repo_to_level[explicit] = level
            continue

        short = str(repo_cell).strip() if not pd.isna(repo_cell) else ""
        if not short:
            continue
        if short in ambiguous_short:
            continue

        for fn in short_to_fulls.get(short, []):
            full_repo_to_level[fn] = level

    return full_repo_to_level


def add_maturity_columns(
    panel_path,
    matching_path,
    agent_first_path,
    ide_first_path,
    *,
    msrc_wide_path=None,
    maturity_scores_path=None,
    label_column=None,
    output_path=None,
):
    """
    Add maturity level columns to the panel data.

    Exactly one of ``msrc_wide_path`` or ``maturity_scores_path`` must be set.

    Args:
        panel_path: Path to panel_event_monthly.csv
        matching_path: Path to matching.csv
        agent_first_path: Path to agent_first.txt (full owner/repo lines)
        ide_first_path: Path to ide_first.txt (full owner/repo lines)
        msrc_wide_path: Path to msrc_maturity_timeseries_wide.csv (default pipeline)
        maturity_scores_path: repos-file org maturity scores CSV (legacy repo → level)
        label_column: Optional wide CSV column name (e.g. 2025-11-01); default = last month column
        output_path: Optional output path (defaults to overwriting input panel)
    """
    has_wide = msrc_wide_path is not None
    has_scores = maturity_scores_path is not None
    if has_wide == has_scores:
        raise ValueError("Set exactly one of msrc_wide_path or maturity_scores_path")

    print("Loading data files...")
    df_panel = pd.read_csv(panel_path, low_memory=False)
    df_matching = pd.read_csv(matching_path)
    full_repo_set = _load_full_repo_names(agent_first_path, ide_first_path)

    print(f"Panel data: {len(df_panel)} rows")
    print(f"Matching data: {len(df_matching)} rows")
    print(f"Full owner/repo in agent_first ∪ ide_first: {len(full_repo_set)}")

    # Step 1: full owner/repo -> level (MSRC wide last month, or legacy repos-file)
    if maturity_scores_path is not None:
        print(f"Maturity source (repos-file): {maturity_scores_path}")
        df_maturity = pd.read_csv(maturity_scores_path)
        print(f"Maturity score rows: {len(df_maturity)}")
        full_repo_to_level = _build_full_repo_to_level(df_maturity, full_repo_set)
        print(f"\nFull names with a maturity level (agent∪ide map): {len(full_repo_to_level)}")
        level_counts = df_maturity["level"].value_counts().sort_index()
        print("\nMaturity level distribution (short names in repos-file CSV):")
        for level in sorted(level_counts.index):
            print(f"  Level {level}: {level_counts[level]} repos")
    else:
        print(f"Maturity source (MSRC wide): {msrc_wide_path}")
        full_repo_to_level, label_col_used = _build_full_repo_to_level_from_msrc_wide(
            msrc_wide_path, full_repo_set, label_column=label_column
        )
        print(f"\nLabel column: {label_col_used!r}")
        print(
            f"Full names with MSRC level 1–4 in that column (and in agent∪ide): "
            f"{len(full_repo_to_level)}"
        )

        level_counts = Counter(full_repo_to_level.values())
        print("\nMaturity level distribution (wide last-month labels, agent∪ide only):")
        for level in sorted(level_counts.keys()):
            print(f"  Level {level}: {level_counts[level]} repos")

    repos_by_level = {
        L: {fn for fn, lv in full_repo_to_level.items() if lv == L}
        for L in (1, 2, 3, 4)
    }
    print(
        "\nUnique full names in maturity map (agent∪ide) per level "
        "(can differ from row counts above when CSV has duplicate/ambiguous rows):"
    )
    for level in (1, 2, 3, 4):
        print(f"  Level {level}: {len(repos_by_level[level])}")
    
    # Step 2: For each treatment repo, get its matched controls
    # Create mapping: treatment_repo -> list of matched controls
    treatment_to_controls = {}
    
    for _, row in df_matching[df_matching['group'] == 'treatment'].iterrows():
        repo_name = row['repo_name']
        controls = []
        
        for i in [1, 2, 3]:
            col_name = f'matched_control_{i}'
            if col_name in row.index:
                matched_control = row[col_name]
                if pd.notna(matched_control) and matched_control != '':
                    controls.append(matched_control)
        
        if controls:
            treatment_to_controls[repo_name] = controls
    
    print(f"\nTreatment repos with matched controls: {len(treatment_to_controls)}")
    
    # Step 3: Create reverse mapping: control_repo -> list of treatment repos it's matched to
    control_to_treatments = {}
    
    for treatment, controls in treatment_to_controls.items():
        for control in controls:
            if control not in control_to_treatments:
                control_to_treatments[control] = []
            control_to_treatments[control].append(treatment)
    
    print(f"Control repos matched to treatments: {len(control_to_treatments)}")
    
    # Step 3.5: Show AF/IF column statistics
    print("\n=== Existing AF/IF Column Statistics ===")
    
    # AF (Agent First) stats
    af_data = df_panel[df_panel['matched_agent_first_or_corresponding_matched_control'].fillna(False) == True]
    af_treatment = af_data[af_data['dataset_source'] == 'treatment']
    af_control = af_data[af_data['dataset_source'] == 'control']
    
    print(f"matched_agent_first_or_corresponding_matched_control (AF):")
    print(f"  Total: {len(af_data)} obs, {af_data['repo_name'].nunique()} repos")
    print(f"    Treatment: {af_treatment['repo_name'].nunique()} repos")
    print(f"    Control: {af_control['repo_name'].nunique()} repos")
    
    # IF (IDE First) stats
    if_data = df_panel[df_panel['matched_ide_first_or_corresponding_matched_control'].fillna(False) == True]
    if_treatment = if_data[if_data['dataset_source'] == 'treatment']
    if_control = if_data[if_data['dataset_source'] == 'control']
    
    print(f"matched_ide_first_or_corresponding_matched_control (IF):")
    print(f"  Total: {len(if_data)} obs, {if_data['repo_name'].nunique()} repos")
    print(f"    Treatment: {if_treatment['repo_name'].nunique()} repos")
    print(f"    Control: {if_control['repo_name'].nunique()} repos")
    
    # Combined AF OR IF
    af_or_if_data = df_panel[
        (df_panel['matched_agent_first_or_corresponding_matched_control'].fillna(False) == True) |
        (df_panel['matched_ide_first_or_corresponding_matched_control'].fillna(False) == True)
    ]
    af_or_if_treatment = af_or_if_data[af_or_if_data['dataset_source'] == 'treatment']
    af_or_if_control = af_or_if_data[af_or_if_data['dataset_source'] == 'control']
    
    print(f"AF OR IF (Combined):")
    print(f"  Total: {len(af_or_if_data)} obs, {af_or_if_data['repo_name'].nunique()} repos")
    print(f"    Treatment: {af_or_if_treatment['repo_name'].nunique()} repos")
    print(f"    Control: {af_or_if_control['repo_name'].nunique()} repos")
    
    # Step 4: Create functions to check maturity level for each level
    def get_level_column(row, level):
        """
        Returns True if:
        - Treatment repo (full owner/repo) has this maturity level
        - Control repo matched to at least one treatment (full owner/repo) with this level
        """
        repo_name = row['repo_name']
        dataset_source = row['dataset_source']
        
        if dataset_source == 'treatment':
            return full_repo_to_level.get(repo_name) == level
        
        elif dataset_source == 'control':
            if repo_name in control_to_treatments:
                for treatment in control_to_treatments[repo_name]:
                    if full_repo_to_level.get(treatment) == level:
                        return True
            return False
        
        else:
            return False
    
    # Step 5: Create all 4 level columns
    print("\n=== Creating maturity level columns ===")

    for level in [1, 2, 3, 4]:
        col_name = f'l{level}_treatment_or_matched_control'
        print(f"\nProcessing {col_name}...")
        
        df_panel[col_name] = df_panel.apply(
            lambda row: get_level_column(row, level),
            axis=1
        )
        
        # Summary stats
        treatment_rows = df_panel[df_panel['dataset_source'] == 'treatment']
        control_rows = df_panel[df_panel['dataset_source'] == 'control']
        
        treatment_true = treatment_rows[treatment_rows[col_name] == True]
        control_true = control_rows[control_rows[col_name] == True]
        
        print(f"  Treatment repos with level {level}: {treatment_true['repo_name'].nunique()}")
        print(f"  Control repos matched to level {level} treatments: {control_true['repo_name'].nunique()}")
        print(f"  Total observations with {col_name}=True: {len(df_panel[df_panel[col_name] == True])}")

        maturity_at_level = repos_by_level[level]
        treatment_repos_at_level = set(treatment_true['repo_name'].unique())
        only_in_maturity_map = sorted(maturity_at_level - treatment_repos_at_level)
        only_in_treatment_flag = sorted(treatment_repos_at_level - maturity_at_level)

        print(
            f"  Full names in maturity map at level {level} "
            f"(agent∪ide, n={len(maturity_at_level)}) "
            f"minus treatment repos with L{level} (n={len(only_in_maturity_map)}):"
        )
        if only_in_maturity_map:
            for r in only_in_maturity_map:
                print(f"    {r}")
        else:
            print("    (none)")

        print(
            f"  Treatment repos with L{level} minus full names in maturity map at level {level} "
            f"(n={len(only_in_treatment_flag)}):"
        )
        if only_in_treatment_flag:
            for r in only_in_treatment_flag:
                print(f"    {r}")
        else:
            print("    (none)")
    
    # Step 6: Create l2+, l3+, and l12 columns (combined levels)
    print("\n=== Creating combined level columns ===")
    
    # l12 = l1 OR l2
    df_panel['l12_treatment_or_matched_control'] = (
        df_panel['l1_treatment_or_matched_control'] |
        df_panel['l2_treatment_or_matched_control']
    )
    
    # l2+ = l2 OR l3 OR l4
    df_panel['l2+_treatment_or_matched_control'] = (
        df_panel['l2_treatment_or_matched_control'] |
        df_panel['l3_treatment_or_matched_control'] |
        df_panel['l4_treatment_or_matched_control']
    )
    
    # l3+ = l3 OR l4
    df_panel['l3+_treatment_or_matched_control'] = (
        df_panel['l3_treatment_or_matched_control'] |
        df_panel['l4_treatment_or_matched_control']
    )
    
    print(f"l12 observations: {len(df_panel[df_panel['l12_treatment_or_matched_control'] == True])}")
    print(f"l2+ observations: {len(df_panel[df_panel['l2+_treatment_or_matched_control'] == True])}")
    print(f"l3+ observations: {len(df_panel[df_panel['l3+_treatment_or_matched_control'] == True])}")
    
    # Step 7: Create full_subset columns (AND with AF OR IF)
    print("\n=== Creating full_subset columns (maturity AND (AF OR IF)) ===")
    
    # Create a combined AF OR IF column for easier computation
    df_panel['_af_or_if'] = (
        df_panel['matched_agent_first_or_corresponding_matched_control'].fillna(False) |
        df_panel['matched_ide_first_or_corresponding_matched_control'].fillna(False)
    )
    
    for level in ['l1', 'l2', 'l3', 'l4', 'l12', 'l2+', 'l3+']:
        base_col = f'{level}_treatment_or_matched_control'
        subset_col = f'{level}_full_subset'
        
        df_panel[subset_col] = df_panel[base_col] & df_panel['_af_or_if']
        
        # Detailed breakdown
        subset_data = df_panel[df_panel[subset_col] == True]
        treatment_data = subset_data[subset_data['dataset_source'] == 'treatment']
        control_data = subset_data[subset_data['dataset_source'] == 'control']
        
        n_true = len(subset_data)
        n_repos = subset_data['repo_name'].nunique()
        n_treatment = treatment_data['repo_name'].nunique()
        n_control = control_data['repo_name'].nunique()
        
        print(f"{subset_col}: {n_true} obs, {n_repos} repos (treatment: {n_treatment}, control: {n_control})")
    
    # Step 8: Create agent_subset columns (AND with AF only)
    print("\n=== Creating agent_subset columns (maturity AND AF) ===")
    
    for level in ['l1', 'l2', 'l3', 'l4', 'l12', 'l2+', 'l3+']:
        base_col = f'{level}_treatment_or_matched_control'
        subset_col = f'{level}_agent_subset'
        
        df_panel[subset_col] = (
            df_panel[base_col] & 
            df_panel['matched_agent_first_or_corresponding_matched_control'].fillna(False)
        )
        
        # Detailed breakdown
        subset_data = df_panel[df_panel[subset_col] == True]
        treatment_data = subset_data[subset_data['dataset_source'] == 'treatment']
        control_data = subset_data[subset_data['dataset_source'] == 'control']
        
        n_true = len(subset_data)
        n_repos = subset_data['repo_name'].nunique()
        n_treatment = treatment_data['repo_name'].nunique()
        n_control = control_data['repo_name'].nunique()
        
        print(f"{subset_col}: {n_true} obs, {n_repos} repos (treatment: {n_treatment}, control: {n_control})")
    
    # Step 9: Create ide_subset columns (AND with IF only)
    print("\n=== Creating ide_subset columns (maturity AND IF) ===")
    
    for level in ['l1', 'l2', 'l3', 'l4', 'l12', 'l2+', 'l3+']:
        base_col = f'{level}_treatment_or_matched_control'
        subset_col = f'{level}_ide_subset'
        
        df_panel[subset_col] = (
            df_panel[base_col] & 
            df_panel['matched_ide_first_or_corresponding_matched_control'].fillna(False)
        )
        
        # Detailed breakdown
        subset_data = df_panel[df_panel[subset_col] == True]
        treatment_data = subset_data[subset_data['dataset_source'] == 'treatment']
        control_data = subset_data[subset_data['dataset_source'] == 'control']
        
        n_true = len(subset_data)
        n_repos = subset_data['repo_name'].nunique()
        n_treatment = treatment_data['repo_name'].nunique()
        n_control = control_data['repo_name'].nunique()
        
        print(f"{subset_col}: {n_true} obs, {n_repos} repos (treatment: {n_treatment}, control: {n_control})")
    
    # Drop the temporary helper column
    df_panel.drop(columns=['_af_or_if'], inplace=True)
    
    # ============================================================================
    # SAVE OUTPUT
    # ============================================================================
    
    if output_path is None:
        output_path = panel_path
    
    df_panel.to_csv(output_path, index=False)
    print(f"\n✅ Saved updated panel data to: {output_path}")
    print(f"Total columns: {len(df_panel.columns)}")
    
    # List all added columns
    added_cols = [
        'l1_treatment_or_matched_control', 'l2_treatment_or_matched_control', 
        'l3_treatment_or_matched_control', 'l4_treatment_or_matched_control',
        'l2+_treatment_or_matched_control', 'l3+_treatment_or_matched_control',
        'l1_full_subset', 'l2_full_subset', 'l3_full_subset', 'l4_full_subset',
        'l2+_full_subset', 'l3+_full_subset',
        'l1_agent_subset', 'l2_agent_subset', 'l3_agent_subset', 'l4_agent_subset',
        'l2+_agent_subset', 'l3+_agent_subset',
        'l1_ide_subset', 'l2_ide_subset', 'l3_ide_subset', 'l4_ide_subset',
        'l2+_ide_subset', 'l3+_ide_subset'
    ]
    print(f"Added {len(added_cols)} columns:")
    for col in added_cols:
        print(f"  - {col}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Add maturity level columns (l1, l2, l3, l4, l2+, subsets) from "
            "MSRC wide last month (default data/ or --new data/data_new/ or "
            "--llm data/data_llm/ or --final data/data_final/), or --old from "
            "data/data_old/ with repos-file levels."
        )
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help=(
            "Use data/data_llm/ defaults (panel, wide, matching, agent_first, ide_first)"
        ),
    )
    parser.add_argument(
        "--final",
        action="store_true",
        help=(
            "Use data/data_final/ defaults (panel, wide, matching, agent_first, ide_first)"
        ),
    )
    parser.add_argument(
        "--old",
        action="store_true",
        help=(
            "Use data/data_old/ defaults (panel, matching, lists, maturity CSV) and "
            "repos-file repo→level mapping instead of MSRC wide time series."
        ),
    )
    parser.add_argument(
        "--new",
        action="store_true",
        help=(
            "Use data/data_new/ defaults (panel, matching, lists, MSRC wide CSV); "
            "same wide time-series labeling as plain defaults, different data directory."
        ),
    )
    parser.add_argument(
        "--panel",
        default=None,
        help="Path to panel_event_monthly.csv (default depends on --new / --old / neither)",
    )
    parser.add_argument(
        "--msrc-wide",
        dest="msrc_wide",
        default=None,
        help="Path to msrc_maturity_timeseries_wide.csv (ignored with --old; under data/data_new/ with --new)",
    )
    parser.add_argument(
        "--maturity-scores",
        dest="maturity_scores",
        default=None,
        help=(
            "Path to repos-file org maturity_scores.csv "
            "(default with --old: data_old/repos-file - November 2025_org_maturity_scores.csv)"
        ),
    )
    parser.add_argument(
        "--label-column",
        dest="label_column",
        default=None,
        help="Wide CSV column to use (e.g. 2025-11-01). Default: last date column (ignored with --old).",
    )
    parser.add_argument(
        "--matching",
        default=None,
        help="Path to matching.csv",
    )
    parser.add_argument(
        "--agent-first",
        dest="agent_first",
        default=None,
        help="Path to agent_first.txt",
    )
    parser.add_argument(
        "--ide-first",
        dest="ide_first",
        default=None,
        help="Path to ide_first.txt",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path (defaults to overwriting --panel)",
    )

    args = parser.parse_args()

    mode_flags = sum(bool(x) for x in (args.old, args.new, args.llm, args.final))
    if mode_flags > 1:
        raise SystemExit("Use only one of --old, --new, --llm, or --final.")

    if not args.old and args.maturity_scores:
        raise SystemExit(
            "Use --maturity-scores only together with --old. "
            "Default run uses MSRC wide (or pass --msrc-wide)."
        )

    if args.old:
        subdir = Path("data/data_old")
    elif args.final:
        subdir = Path("data/data_final")
    elif args.llm:
        subdir = Path("data/data_llm")
    elif args.new:
        subdir = Path("data/data_new")
    else:
        subdir = Path("data")

    def pick(cli: Optional[str], filename: str) -> str:
        if cli:
            return _resolve_from_base(cli)
        return str(BASE_DIR / subdir / filename)

    panel_path = pick(args.panel, "panel_event_monthly.csv")
    matching_path = pick(args.matching, "matching.csv")
    agent_first_path = pick(args.agent_first, "agent_first.txt")
    ide_first_path = pick(args.ide_first, "ide_first.txt")

    if args.old and args.label_column:
        print("Note: --label-column ignored with --old (repos-file defines levels).\n")

    output_path_res = _resolve_from_base(args.output) if args.output else None

    if args.old:
        if args.msrc_wide:
            raise SystemExit("Do not combine --msrc-wide with --old (mapping uses repos-file only).")
        maturity_scores_default = "repos-file - November 2025_org_maturity_scores.csv"
        scores_path = pick(args.maturity_scores, maturity_scores_default)
        add_maturity_columns(
            panel_path,
            matching_path,
            agent_first_path,
            ide_first_path,
            maturity_scores_path=scores_path,
            output_path=output_path_res,
        )
    else:
        wide_name = (
            "msrc_maturity_timeseries_llm_wide.csv"
            if (args.llm or args.final)
            else "msrc_maturity_timeseries_wide.csv"
        )
        wide_path = pick(args.msrc_wide, wide_name)
        if args.llm or args.final:
            mode = "final" if args.final else "llm"
            print(
                f"{mode.upper()} mode: panel={panel_path!r} → "
                f"{output_path_res or panel_path!r}, wide={wide_path!r}\n"
            )
        add_maturity_columns(
            panel_path,
            matching_path,
            agent_first_path,
            ide_first_path,
            msrc_wide_path=wide_path,
            label_column=args.label_column,
            output_path=output_path_res,
        )


if __name__ == "__main__":
    main()
