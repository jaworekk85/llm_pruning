"""Paired crossed bootstrap and finite-pool control precision planning, no inference.

Intervals condition on the discovery-selected mask. Rows and control columns
are resampled jointly across metrics, preserving the shared-example pairing.
Finite-population variance scaling is approximate, not an exact randomization test.
"""
import math
from collections import defaultdict
from statistics import NormalDist

import numpy as np

from .causal_evaluation import CONTROL_STRATEGIES


METRICS = ("accuracy_damage", "loss_increase", "margin_damage")


def validate_uncertainty_settings(*, samples=2000, confidence=0.95, seed=0, multiplicity="bonferroni", comparison_count=None, desired_half_width=None, planning_metric="loss_increase"):
    if type(samples) is not int or samples < 100 or type(seed) is not int or seed < 0:
        raise ValueError("Use >=100 bootstrap samples and a nonnegative integer seed")
    if isinstance(confidence, bool) or not isinstance(confidence, (float, int)) or not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly between 0 and 1")
    if multiplicity not in {"none", "bonferroni"} or planning_metric not in METRICS:
        raise ValueError("Unsupported multiplicity or planning metric")
    if comparison_count is not None and (type(comparison_count) is not int or comparison_count < 1):
        raise ValueError("comparison_count must be a positive integer")
    if desired_half_width is not None and (isinstance(desired_half_width, bool) or not isinstance(desired_half_width, (int, float)) or not math.isfinite(desired_half_width) or desired_half_width <= 0):
        raise ValueError("desired_half_width must be finite and positive")


def _draw_contrasts(selected, controls, population, samples, rng):
    n, m = controls.shape[:2]
    estimate = selected.mean(axis=0) - controls.mean(axis=(0, 1))
    examples, masks, joint = [], [], []
    census = population == m
    mask_estimable = census or m >= 2
    factor = None
    if census:
        factor = 0.0
    elif m >= 2:
        factor = math.sqrt(m / (m - 1))
        if population is not None:
            factor *= math.sqrt(1 - m / population)
    fixed_controls = controls.mean(axis=0)
    for start in range(0, samples, 64):
        size = min(64, samples - start)
        # The same row weights apply to selected and every mask; a sampled mask
        # column applies to all sampled examples, not independently per example.
        row_weights = rng.multinomial(n, np.full(n, 1 / n), size=size) / n
        selected_boot = row_weights @ selected
        per_mask = (row_weights @ controls.reshape(n, m * len(METRICS))).reshape(size, m, len(METRICS))
        row_control_mean = per_mask.mean(axis=1)
        examples.append(selected_boot - row_control_mean)
        if mask_estimable:
            if census:
                masks.append(np.tile(estimate, (size, 1)))
                joint.append(selected_boot - row_control_mean)
            else:
                column_weights = rng.multinomial(m, np.full(m, 1 / m), size=size) / m
                boot_controls = np.einsum("bm,bmk->bk", column_weights, per_mask)
                joint.append(selected_boot - row_control_mean - factor * (boot_controls - row_control_mean))
                fixed_boot = column_weights @ fixed_controls
                masks.append(estimate - factor * (fixed_boot - fixed_controls.mean(axis=0)))
    return estimate, {
        "examples_only": np.concatenate(examples) if n >= 2 else None,
        "masks_only": np.concatenate(masks) if mask_estimable else None,
        "joint": np.concatenate(joint) if n >= 2 and mask_estimable else None,
    }


def _precision_plan(controls, population, desired, confidence, metric, split):
    if desired is None:
        return None
    if split != "validation":
        return {"status": "not_planned_on_test", "reason": "Freeze the mask budget using validation/pilot data, not test outcomes."}
    m = controls.shape[1]
    if population == m:
        return {"status": "exhaustive", "estimated_masks_needed": m, "current_control_standard_error": 0.0}
    if m < 2:
        return {"status": "insufficient_unique_masks", "estimated_masks_needed": None}
    mask_means = controls[:, :, METRICS.index(metric)].mean(axis=0)
    variance = float(mask_means.var(ddof=1))
    z = NormalDist().inv_cdf((1 + confidence) / 2)
    numerator = z * z * variance
    needed = max(2, math.ceil(numerator / (desired * desired + (numerator / population if population is not None else 0))))
    if population is not None:
        needed = min(population, needed)
    correction = math.sqrt(1 - m / population) if population is not None else 1.0
    return {
        "status": "pilot_estimate", "metric": metric, "desired_control_half_width": desired,
        "estimated_masks_needed": needed, "additional_unique_masks": max(0, needed - m),
        "current_control_standard_error": math.sqrt(variance / m) * correction,
        "zero_observed_mask_variance": variance == 0,
        "scope": "Mask-mean precision on these examples only; not power or a guarantee of future precision.",
    }


def analyze_uncertainty(rows, conditions, sampling=None, *, samples=2000, confidence=0.95, seed=0, multiplicity="bonferroni", comparison_count=None, desired_half_width=None, planning_metric="loss_increase"):
    settings = dict(samples=samples, confidence=confidence, seed=seed, multiplicity=multiplicity, comparison_count=comparison_count,
                    desired_half_width=desired_half_width, planning_metric=planning_metric)
    validate_uncertainty_settings(**settings)
    if conditions.get("baseline") != [] or not conditions.get("selected"):
        raise ValueError("Require baseline and selected conditions")
    sampling = sampling or {}
    grouped, metadata = defaultdict(dict), {}
    for row in rows:
        if row["condition"] not in conditions or row["split"] not in {"validation", "test"}:
            raise ValueError("Invalid paired row condition or split")
        group = (row["target_field"], row["target_value"], row["split"])
        identity = (row["condition"], row["record_id"])
        record_meta = (group, str(row["is_target"]))
        if row["record_id"] in metadata and metadata[row["record_id"]] != record_meta:
            raise ValueError("Record metadata changes across conditions")
        metadata[row["record_id"]] = record_meta
        if identity in grouped[group]:
            raise ValueError("Duplicate paired row")
        values = [float(row[key]) for key in METRICS]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Paired changes must be finite")
        grouped[group][identity] = values
    strategy_conditions = {}
    for strategy in sorted(CONTROL_STRATEGIES):
        names = [name for name in conditions if name.startswith(strategy + "_")]
        if names:
            unique = {}
            for name in names:
                mask = tuple(sorted(conditions[name]))
                if not mask or len(mask) != len(set(mask)):
                    raise ValueError("Control masks must be nonempty with unique components")
                unique.setdefault(mask, name)
            strategy_conditions[strategy] = (names, list(unique.values()))
    if not grouped or not strategy_conditions:
        raise ValueError("Require paired evaluation examples and random controls")
    actual_comparisons = len(grouped) * len(strategy_conditions) * len(METRICS)
    family_size = comparison_count or actual_comparisons
    if family_size < actual_comparisons:
        raise ValueError("comparison_count cannot be smaller than reported comparisons")
    level = 1 - (1 - confidence) / family_size if multiplicity == "bonferroni" else confidence
    if level == 1:
        raise ValueError("Comparison family too large for numeric confidence precision")
    alpha_tail = (1 - level) / 2
    rng = np.random.default_rng(seed)
    results = []
    for (field, value, split), table in sorted(grouped.items()):
        record_ids = sorted(record_id for name, record_id in table if name == "selected")
        if not record_ids or set(table) != {(name, record_id) for name in conditions for record_id in record_ids}:
            raise ValueError("Incomplete paired example coverage across conditions")
        selected = np.array([table[("selected", record_id)] for record_id in record_ids])
        for strategy, (all_names, names) in strategy_conditions.items():
            controls = np.array([[table[(name, record_id)] for name in names] for record_id in record_ids])
            info = sampling.get(strategy, {})
            population = info.get("available_masks")
            if population is not None and (type(population) is not int or population < len(names)):
                raise ValueError("Invalid finite mask population")
            if info and (info.get("sampling") != "uniform_without_replacement" or info.get("unique_masks") != len(names)):
                raise ValueError("Sampling metadata does not match unique controls")
            estimate, draws = _draw_contrasts(selected, controls, population, samples, rng)
            intervals = {}
            for metric_index, metric in enumerate(METRICS):
                intervals[metric] = {kind: None if array is None else {
                    "low": float(np.quantile(array[:, metric_index], alpha_tail)),
                    "high": float(np.quantile(array[:, metric_index], 1 - alpha_tail)),
                } for kind, array in draws.items()}
            warnings = []
            if len(record_ids) < 2:
                warnings.append("example_uncertainty_not_estimable")
            elif len(record_ids) < 20:
                warnings.append("small_example_pilot")
            if len(names) < 2 and population != len(names):
                warnings.append("mask_uncertainty_not_estimable")
            elif len(names) < 10 and population != len(names):
                warnings.append("small_mask_pilot")
            if not info:
                warnings.append("legacy_sampling_unknown_no_finite_pool_correction")
            if len(names) != len(all_names):
                warnings.append("duplicate_masks_collapsed")
            if len(names) >= 2 and population != len(names) and np.any(controls.mean(axis=0).var(axis=0) == 0):
                warnings.append("zero_observed_mask_spread_in_some_metrics")
            if samples * alpha_tail < 10:
                warnings.append("sparse_bootstrap_tails_increase_samples")
            results.append({
                "target_field": field, "target_value": value, "split": split, "strategy": strategy,
                "example_count": len(record_ids), "recorded_masks": len(all_names), "unique_masks": len(names),
                "available_masks": population, "exhaustive": population == len(names),
                "estimate": dict(zip(METRICS, estimate.tolist())), "intervals": intervals, "warnings": warnings,
                "precision_plan": _precision_plan(controls, population, desired_half_width, level, planning_metric, split),
            })
    return {"schema_version": 1, "method": "paired_crossed_percentile_bootstrap_v1", "settings": settings,
            "comparison_count": family_size, "interval_confidence": level,
            "interpretation": "Approximate intervals conditional on discovery selection and independent evaluation examples; no p-values or guaranteed power. Joint intervals resample shared examples and whole control masks.",
            "results": results}
