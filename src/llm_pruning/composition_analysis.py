"""Paired full-cohort readouts with source-group bootstrap; no independence-product null."""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .linked_tasks import ALL_VARIANTS


def _index(rows):
    episodes = defaultdict(dict)
    for row in rows:
        episode = episodes[row["episode_id"]]
        if row["variant"] in episode or row["variant"] not in ALL_VARIANTS:
            raise ValueError("Duplicate/unknown decision variant")
        if type(row["correct"]) is not bool or type(row["valid"]) is not bool:
            raise ValueError("Decision correctness/validity must be boolean")
        episode[row["variant"]] = row
    if not episodes or any(set(e) != set(ALL_VARIANTS) for e in episodes.values()):
        raise ValueError("Every episode needs all five decisions")
    for episode in episodes.values():
        if len({row["source_group_id"] for row in episode.values()}) != 1:
            raise ValueError("Decision source groups disagree")
    return episodes


def paired_composition_summary(baseline, current, *, seed=0, bootstrap_samples=1000, confidence=.95):
    """Resample source groups jointly across variants and dense/pruned conditions.

    Output contrasts are descriptive, unadjusted pointwise intervals. They neither
    identify a neural mechanism nor guarantee power/equivalence/ICML significance.
    """
    if type(bootstrap_samples) is not int or bootstrap_samples < 1 or not 0 < confidence < 1:
        raise ValueError("Positive bootstrap_samples and confidence in (0,1) required")
    base, now = _index(baseline), _index(current)
    if set(base) != set(now):
        raise ValueError("Paired conditions must cover the same episodes")
    ids = sorted(base)
    groups = defaultdict(list)
    for i, episode_id in enumerate(ids):
        for variant in ALL_VARIANTS:
            if any(base[episode_id][variant][key] != now[episode_id][variant][key]
                   for key in ("source_group_id", "split", "knowledge_scope", "target")):
                raise ValueError("Paired decision provenance mismatch")
        groups[base[episode_id]["a"]["source_group_id"]].append(i)
    b = np.array([[base[e][v]["correct"] for v in ALL_VARIANTS] for e in ids], dtype=float)
    c = np.array([[now[e][v]["correct"] for v in ALL_VARIANTS] for e in ids], dtype=float)
    damage = b - c
    # Positive contrast = direct composition lost more than the control.
    vectors = {f"{v}_accuracy_damage": damage[:, i] for i, v in enumerate(ALL_VARIANTS)}
    vectors["composed_minus_staged_damage"] = damage[:, 2] - damage[:, 4]
    vectors["composed_minus_oracle_damage"] = damage[:, 2] - damage[:, 3]
    vectors["composed_minus_b_damage"] = damage[:, 2] - damage[:, 1]
    atomic_ok_composed_wrong = (c[:, 0] == 1) & (c[:, 1] == 1) & (c[:, 2] == 0)
    vectors["atomic_ok_composed_wrong_change"] = atomic_ok_composed_wrong.astype(float) - (
        (b[:, 0] == 1) & (b[:, 1] == 1) & (b[:, 2] == 0)).astype(float)
    values = np.column_stack(list(vectors.values()))
    rng = np.random.default_rng(seed)
    cluster_ids = sorted(groups)
    draws = []
    if len(groups) >= 2:
        for _ in range(bootstrap_samples):
            selected = rng.integers(0, len(cluster_ids), len(cluster_ids))
            indices = np.concatenate([groups[cluster_ids[i]] for i in selected])
            draws.append(values[indices].mean(axis=0))
    alpha = (1 - confidence) / 2
    bounds = np.quantile(np.asarray(draws), [alpha, 1 - alpha], axis=0) if draws else None
    contrasts = {name: {"estimate": float(values[:, i].mean()),
                         "ci_low": None if bounds is None else float(bounds[0, i]),
                         "ci_high": None if bounds is None else float(bounds[1, i])}
                 for i, name in enumerate(vectors)}
    return {"episode_count": len(ids), "source_group_count": len(groups),
            "accuracy": {v: float(c[:, i].mean()) for i, v in enumerate(ALL_VARIANTS)},
            "baseline_accuracy": {v: float(b[:, i].mean()) for i, v in enumerate(ALL_VARIANTS)},
            "malformed_rate": {v: sum(not now[e][v]["valid"] for e in ids) / len(ids) for v in ALL_VARIANTS},
            "atomic_ok_composed_wrong_rate": float(atomic_ok_composed_wrong.mean()), "paired": contrasts,
            "uncertainty": {"method": "paired_source_group_percentile_bootstrap", "seed": seed,
                            "samples": bootstrap_samples, "confidence": confidence,
                            "status": "estimated" if draws else "insufficient_independent_groups",
                            "multiplicity": "unadjusted exploratory pointwise intervals"},
            "interpretation": "Full cohort; staged uses each condition's A prediction. No product-of-accuracies null, causal or optimality claim."}
