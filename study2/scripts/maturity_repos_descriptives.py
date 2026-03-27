#!/usr/bin/env python3
"""
Descriptive stats for repos_with_details.csv by maturity: level 1 vs level 2+ (2,3,4).

Uses the same full-repo maturity mapping as maturity_columns.py. Reports mean,
median, min, max for numeric columns (including repo age in days from repo_created
to a reference date, default end of November 2025), plus counts and percentages
of L1 vs L2+ within agent_first.txt and ide_first.txt.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from maturity_columns import (  # noqa: E402
    _build_full_repo_to_level,
    _load_full_repo_names,
)


def _parse_reference_timestamp(date_str: str) -> pd.Timestamp:
    """End-of-study date: calendar day in UTC (used as repo-age reference)."""
    ts = pd.Timestamp(date_str)
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    return ts.normalize()


def _compute_age_days(repo_created: pd.Series, reference: pd.Timestamp) -> pd.Series:
    """
    Whole calendar days from repo_created (UTC) to reference date (inclusive of
    reference day as end of period). Repos created after reference get NaN.
    """
    created = pd.to_datetime(repo_created, utc=True, errors="coerce")
    ref = reference
    if getattr(ref, "tzinfo", None) is None:
        ref = ref.tz_localize("UTC")
    days = (ref.normalize() - created.dt.normalize()).dt.days
    return days.where((days >= 0) & created.notna())


def _load_lines(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def _numeric_descriptive_columns(df: pd.DataFrame) -> list[str]:
    """Columns to summarize: numeric repo metrics, excluding ids and timestamps."""
    exclude = {
        "id",
        "name",
        "url",
        "primary_language",
        "repo_created",
        "first_agent_adopted_at",
        "agent_adopt_month",
        "agent_adopt_week",
    }
    cols = []
    for c in df.columns:
        if c.startswith("_"):
            continue
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
            continue
        coerced = pd.to_numeric(df[c], errors="coerce")
        if coerced.notna().sum() > 0:
            cols.append(c)
    return sorted(cols, key=str.lower)


def _describe_group(series: pd.Series) -> dict:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return {"n": 0, "mean": float("nan"), "median": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "n": int(len(s)),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "min": float(s.min()),
        "max": float(s.max()),
    }


def _print_descriptives_table(df: pd.DataFrame, label: str, cols: list[str]) -> None:
    print(f"\n=== {label} ===")
    print(f"Rows: {len(df)}")
    if len(df) == 0:
        return
    for c in cols:
        st = _describe_group(df[c])
        print(
            f"  {c}: n={st['n']}, mean={st['mean']:.4g}, median={st['median']:.4g}, "
            f"min={st['min']:.4g}, max={st['max']:.4g}"
        )


def _list_file_breakdown(
    label: str,
    repo_lines: list[str],
    full_repo_to_level: dict[str, int],
) -> None:
    total = len(repo_lines)
    l1 = sum(1 for r in repo_lines if full_repo_to_level.get(r) == 1)
    l2p = sum(
        1 for r in repo_lines if full_repo_to_level.get(r) in (2, 3, 4)
    )
    mapped = l1 + l2p
    unmapped = total - mapped

    print(f"\n=== {label} ===")
    print(f"Total repos in file: {total}")
    print(f"With maturity level (L1 or L2+): {mapped}")
    print(f"  Level 1: {l1} ({100.0 * l1 / total:.2f}% of all lines in file)")
    print(f"  Level 2+ (2,3,4): {l2p} ({100.0 * l2p / total:.2f}% of all lines in file)")
    print(f"  Unmapped (no level in maturity map): {unmapped} ({100.0 * unmapped / total:.2f}% of all lines in file)")
    if mapped > 0:
        print(
            f"  Of mapped only — L1: {100.0 * l1 / mapped:.2f}%, "
            f"L2+: {100.0 * l2p / mapped:.2f}%"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Descriptives for repos_with_details by maturity L1 vs L2+"
    )
    parser.add_argument(
        "--repos-details",
        default="../data/repos_with_details.csv",
        help="repos_with_details.csv (column 'name' = owner/repo)",
    )
    parser.add_argument(
        "--maturity-scores",
        default="../data/repos-file - November 2025_org_maturity_scores.csv",
        help="Maturity scores CSV (repo, level)",
    )
    parser.add_argument(
        "--agent-first",
        dest="agent_first",
        default="../data/agent_first.txt",
        help="agent_first.txt",
    )
    parser.add_argument(
        "--ide-first",
        dest="ide_first",
        default="../data/ide_first.txt",
        help="ide_first.txt",
    )
    parser.add_argument(
        "--output-table",
        default=None,
        help="Optional path to write a long-format CSV of descriptives",
    )
    parser.add_argument(
        "--reference-date",
        default="2025-11-30",
        help="Reference date (UTC calendar day) for age in days; default end of Nov 2025",
    )
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    repos_path = (base / args.repos_details).resolve()
    maturity_path = (base / args.maturity_scores).resolve()
    agent_path = (base / args.agent_first).resolve()
    ide_path = (base / args.ide_first).resolve()

    df_m = pd.read_csv(maturity_path)
    full_repo_set = _load_full_repo_names(agent_path, ide_path)
    full_repo_to_level = _build_full_repo_to_level(df_m, full_repo_set)

    df = pd.read_csv(repos_path)
    if "name" not in df.columns:
        raise SystemExit("repos_with_details.csv must have a 'name' column (owner/repo).")

    df = df.copy()
    if "repo_created" not in df.columns:
        raise SystemExit("repos_with_details.csv must have repo_created for age_days.")
    ref_ts = _parse_reference_timestamp(args.reference_date)
    df["age_days"] = _compute_age_days(df["repo_created"], ref_ts)

    df["_maturity_level"] = df["name"].map(lambda x: full_repo_to_level.get(str(x).strip() if pd.notna(x) else ""))
    df_l1 = df[df["_maturity_level"] == 1].copy()
    df_l2p = df[df["_maturity_level"].isin((2, 3, 4))].copy()
    df_unmapped = df[df["_maturity_level"].isna()].copy()

    numeric_cols = _numeric_descriptive_columns(df)

    print("Maturity mapping: same rules as maturity_columns.py (agent∪ide full names).")
    print(
        f"Repo age (age_days): whole days from repo_created to reference {args.reference_date} (UTC); "
        f"repos created after reference are excluded (NaN)."
    )
    print(f"repos_with_details rows: {len(df)}")
    print(f"  Level 1 (in details table): {len(df_l1)}")
    print(f"  Level 2+ (in details table): {len(df_l2p)}")
    print(f"  No maturity level for name: {len(df_unmapped)}")

    _print_descriptives_table(df_l1, "Level 1 — numeric descriptives", numeric_cols)
    _print_descriptives_table(df_l2p, "Level 2+ — numeric descriptives", numeric_cols)

    agent_lines = _load_lines(agent_path)
    ide_lines = _load_lines(ide_path)
    _list_file_breakdown("agent_first.txt", agent_lines, full_repo_to_level)
    _list_file_breakdown("ide_first.txt", ide_lines, full_repo_to_level)

    if args.output_table:
        rows = []
        for label, part in [("level_1", df_l1), ("level_2plus", df_l2p)]:
            for col in numeric_cols:
                st = _describe_group(part[col])
                rows.append(
                    {
                        "group": label,
                        "column": col,
                        **st,
                    }
                )
        out = pd.DataFrame(rows)
        out_path = Path(args.output_table)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_path, index=False)
        print(f"\nWrote table: {out_path}")


if __name__ == "__main__":
    main()
