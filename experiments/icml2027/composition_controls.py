"""Behavioral probe readouts: reference estimates separate from correlated controls."""
from __future__ import annotations

from collections import defaultdict

from llm_pruning.composition_analysis import paired_composition_summary


def reference_rows(rows):
    return [row for row in rows if row.get("case_role", "reference") == "reference"]


def control_summary(baseline, current, *, seed, bootstrap_samples, confidence):
    roles = sorted({row.get("case_role", "reference") for row in current} - {"reference"})
    if not roles:
        return None
    by_group = defaultdict(lambda: defaultdict(dict))
    for row in current:
        by_group[row["source_group_id"]][row["case_role"]][row["variant"]] = row
    reports = {}
    for role in roles:
        now = [row for row in current if row["case_role"] == role]
        base = [row for row in baseline if row["case_role"] == role]
        report = paired_composition_summary(base, now, seed=seed,
                                            bootstrap_samples=bootstrap_samples, confidence=confidence)
        pairs = [(probes["reference"]["composed"], probes[role]["composed"])
                 for probes in by_group.values()]
        changing = role == "changed_target"
        if any((a["target"] != b["target"]) != changing for a, b in pairs):
            raise ValueError("Behavioral probe target contract violated")
        report["composed_pair_both_correct_rate"] = sum(a["correct"] and b["correct"] for a, b in pairs) / len(pairs)
        valid = [(a, b) for a, b in pairs if a["valid"] and b["valid"]]
        report["valid_pair_count"] = len(valid)
        report["prediction_changed_rate_given_valid_pair"] = (
            sum(a["prediction"] != b["prediction"] for a, b in valid) / len(valid) if valid else None)
        report["expected_behavior"] = "change to new gold" if changing else "preserve gold"
        reports[role] = report
    return {"paired_probe_groups": len(by_group), "roles": reports,
            "interpretation": "Behavioral sensitivity/invariance, not neural causal localization. Wrong-but-invariant outputs do not pass both-correct readout."}
