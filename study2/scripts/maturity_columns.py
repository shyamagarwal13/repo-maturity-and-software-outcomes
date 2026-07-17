#!/usr/bin/env python3
"""Build the analysis panel from the original panel plus final maturity labels.

Inputs (relative to study2/):
  data/panel_event_monthly_original.csv  Agarwal et al.'s repository-month panel
  data/maturity_levels.csv               final RAMP level (1-4) per treated repository
  data/matching.csv                      treated repository -> matched controls

Output:
  data/panel_event_monthly.csv           panel with maturity/matching flag columns
                                         (l1..l4, l12, l2+, l3+ x
                                          treatment_or_matched_control,
                                          _full_subset, _agent_subset, _ide_subset)

Controls inherit the level(s) of the treated repositories they are matched to.
Treated repositories without a row in maturity_levels.csv (inaccessible at
re-collection) receive no level and are excluded from every maturity subset.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent.parent


def main() -> None:
    panel = pd.read_csv(HERE / "data" / "panel_event_monthly_original.csv", low_memory=False)
    levels = pd.read_csv(HERE / "data" / "maturity_levels.csv")
    matching = pd.read_csv(HERE / "data" / "matching.csv")

    level_map = dict(zip(levels.repo_name, levels.maturity_level))

    treatment_to_controls: dict[str, list[str]] = {}
    for _, row in matching[matching["group"] == "treatment"].iterrows():
        controls = [row[c] for c in ("matched_control_1", "matched_control_2", "matched_control_3")
                    if c in row.index and pd.notna(row[c]) and row[c] != ""]
        if controls:
            treatment_to_controls[row["repo_name"]] = controls

    control_to_treatments: dict[str, list[str]] = {}
    for treated, controls in treatment_to_controls.items():
        for control in controls:
            control_to_treatments.setdefault(control, []).append(treated)

    def level_flag(row, level: int) -> bool:
        repo, src = row["repo_name"], row["dataset_source"]
        if src == "treatment":
            return level_map.get(repo) == level
        if src == "control":
            return any(level_map.get(t) == level
                       for t in control_to_treatments.get(repo, []))
        return False

    for lv in (1, 2, 3, 4):
        panel[f"l{lv}_treatment_or_matched_control"] = panel.apply(
            lambda r: level_flag(r, lv), axis=1)

    panel["l12_treatment_or_matched_control"] = (
        panel["l1_treatment_or_matched_control"] | panel["l2_treatment_or_matched_control"])
    panel["l2+_treatment_or_matched_control"] = (
        panel["l2_treatment_or_matched_control"] | panel["l3_treatment_or_matched_control"]
        | panel["l4_treatment_or_matched_control"])
    panel["l3+_treatment_or_matched_control"] = (
        panel["l3_treatment_or_matched_control"] | panel["l4_treatment_or_matched_control"])

    af = panel["matched_agent_first_or_corresponding_matched_control"].fillna(False).astype(bool)
    if_ = panel["matched_ide_first_or_corresponding_matched_control"].fillna(False).astype(bool)
    af_or_if = af | if_

    for lv in ("l1", "l2", "l3", "l4", "l12", "l2+", "l3+"):
        base = panel[f"{lv}_treatment_or_matched_control"]
        panel[f"{lv}_full_subset"] = base & af_or_if
        panel[f"{lv}_agent_subset"] = base & af
        panel[f"{lv}_ide_subset"] = base & if_

    out = HERE / "data" / "panel_event_monthly.csv"
    panel.to_csv(out, index=False)

    treated = panel[panel.dataset_source == "treatment"]
    counts = {lv: treated[treated[f"l{lv}_treatment_or_matched_control"]].repo_name.nunique()
              for lv in (1, 2, 3, 4)}
    print(f"wrote {out.name}: {len(panel)} rows, {len(panel.columns)} cols; "
          f"treated level counts {counts}")


if __name__ == "__main__":
    main()
