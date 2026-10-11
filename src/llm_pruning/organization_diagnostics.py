"""Descriptive concentration, overlap and repeated-discovery stability.

Keep signed primitive scores. Positive clipping is an explicit convention for
distribution summaries, never evidence that a functional module exists.
"""
from collections import defaultdict
from itertools import combinations
import math

import numpy as np

from .components import ComponentId


def _vector(scores):
    if not scores or any(not math.isfinite(float(v)) for v in scores.values()):
        raise ValueError("Scores must be a nonempty finite component vector")
    return np.array([scores[key] for key in sorted(scores)], dtype=np.float64)


def _top(scores, k):
    return [key for key in sorted(scores, key=lambda key: (-scores[key], key)) if scores[key] > 0][:k]


def concentration(scores, *, top_k=5):
    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k must be positive")
    raw = _vector(scores)
    positive = np.maximum(raw, 0)
    n, total = len(raw), float(positive.sum())
    ranked = _top(scores, top_k)
    warnings = []
    if np.count_nonzero(positive) < top_k:
        warnings.append("fewer_positive_components_than_top_k")
    ordered = sorted((v for v in scores.values() if v > 0), reverse=True)
    if len(ordered) > top_k and ordered[top_k - 1] == ordered[top_k]:
        warnings.append("top_k_boundary_tie_resolved_by_component_id")
    output = {"component_count": n, "positive_signal": total, "positive_component_count": int((positive > 0).sum()),
              "negative_component_count": int((raw < 0).sum()), "top_k": top_k, "top_components": ranked, "warnings": warnings}
    if total == 0:
        output.update(entropy=None, normalized_entropy=None, effective_components=None, gini=None, top_k_share=None)
        output["warnings"].append("no_positive_signal")
        return output
    p = positive / total
    entropy = float(-(p[p > 0] * np.log(p[p > 0])).sum())
    sorted_positive = np.sort(positive)
    gini = float(2 * np.dot(np.arange(1, n + 1), sorted_positive) / (n * total) - (n + 1) / n)
    output.update(entropy=entropy, normalized_entropy=entropy / math.log(n) if n > 1 else 0.,
                  effective_components=math.exp(entropy), gini=gini,
                  top_k_share=sum(scores[key] for key in ranked) / total)
    return output


def _cosine(a, b):
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return None if denominator == 0 else float(np.clip(np.dot(a, b) / denominator, -1, 1))


def _ranks(values):
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2
        start = stop
    return ranks


def overlap(left, right, *, top_k=5):
    if set(left) != set(right):
        raise ValueError("Compare identical component universes; never silently intersect")
    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k must be positive")
    a, b = _vector(left), _vector(right)
    sa, sb = set(_top(left, top_k)), set(_top(right, top_k))
    union = sa | sb
    pa, pb = np.maximum(a, 0), np.maximum(b, 0)
    weighted = None if pa.sum() == 0 or pb.sum() == 0 else float(np.minimum(pa / pa.sum(), pb / pb.sum()).sum())
    ra, rb = _ranks(a), _ranks(b)
    return {"top_k_jaccard": len(sa & sb) / len(union) if union else None,
            "left_top_count": len(sa), "right_top_count": len(sb), "weighted_overlap": weighted,
            "signed_cosine": _cosine(a, b), "spearman": _cosine(ra - ra.mean(), rb - rb.mean())}


def measurement_vectors(rows, records, component_ids, field, *, kind, token_region="answer", metric="loss_increase", conditions=None):
    """Stream rows into equal-example target-vs-other vectors; validate coverage.

    Activation rows must cover the full chosen region/inventory. Causal rows
    must come from unique singleton probes; combined masks cannot be decomposed.
    """
    if kind not in {"activation", "causal"} or token_region not in {"prompt", "answer"} or metric not in {"accuracy_damage", "loss_increase", "margin_damage"}:
        raise ValueError("Unsupported measurement settings")
    by_id = {r["id"]: r for r in records}
    if not by_id or len(by_id) != len(records) or any(r["split"] != "discovery" or not r.get(field) for r in records):
        raise ValueError("Unique labeled discovery records are required")
    if len(set(component_ids)) != len(component_ids) or not component_ids:
        raise ValueError("Inventory must contain unique components")
    for component in component_ids:
        ComponentId.parse(component)
    universe = set(component_ids)
    probe_map = {}
    if kind == "causal":
        if not conditions or conditions.get("baseline") != []:
            raise ValueError("Causal diagnostics require an unmasked baseline")
        for name, components in conditions.items():
            if name == "baseline":
                continue
            if len(components) != 1 or components[0] not in universe:
                raise ValueError("Causal diagnostics require singleton probes in the inventory")
            probe_map[name] = components[0]
        if not probe_map or len(set(probe_map.values())) != len(probe_map):
            raise ValueError("Causal diagnostics require unique single-component probes")
        universe = set(probe_map.values())
    sums = defaultdict(float)
    seen = set()
    baseline_ids = set()
    for row in rows:
        record_id = row["record_id"]
        if record_id not in by_id or row["split"] != "discovery" or row["target_field"] != field or row["target_value"] != by_id[record_id][field]:
            raise ValueError("Measurement metadata differs from discovery records")
        if kind == "activation":
            component = row["component_id"]
            if component not in universe or row["token_region"] not in {"prompt", "answer"}:
                raise ValueError("Activation is outside the inventory/token regions")
            if row["token_region"] != token_region:
                continue
            value = float(row["mean_abs"])
            if value < 0:
                raise ValueError("mean_abs must be nonnegative")
        else:
            value = float(row[metric])
            if row["condition"] == "baseline":
                if record_id in baseline_ids or value != 0:
                    raise ValueError("Duplicate or nonzero baseline damage")
                baseline_ids.add(record_id)
                continue
            if row["condition"] not in probe_map:
                raise ValueError("Unknown causal condition")
            component = probe_map[row["condition"]]
        if not math.isfinite(value) or (record_id, component) in seen:
            raise ValueError("Non-finite or duplicate measurement")
        seen.add((record_id, component))
        sums[(by_id[record_id][field], component)] += value
    if len(seen) != len(by_id) * len(universe) or (kind == "causal" and baseline_ids != set(by_id)):
        raise ValueError("Incomplete discovery measurement coverage")
    counts = defaultdict(int)
    for record in records:
        counts[record[field]] += 1
    if len(counts) < 2:
        raise ValueError("Diagnostics require at least two target-field values")
    vectors = []
    granularities = sorted({ComponentId.parse(c).kind for c in universe})
    for granularity in granularities:
        components = sorted(c for c in universe if ComponentId.parse(c).kind == granularity)
        for target in sorted(counts):
            target_means = {c: sums[(target, c)] / counts[target] for c in components}
            other_count = len(records) - counts[target]
            other_means = {c: sum(sums[(v, c)] for v in counts if v != target) / other_count for c in components}
            profiles = [("activation_selectivity", {c: (target_means[c] - other_means[c]) / (target_means[c] + other_means[c] + 1e-12) for c in components})] if kind == "activation" else [
                (f"causal_{metric}", target_means),
                (f"causal_{metric}_contrast", {c: target_means[c] - other_means[c] for c in components}),
            ]
            for score_kind, scores in profiles:
                vectors.append({"target_field": field, "target_value": target, "granularity": granularity,
                                "score_kind": score_kind, "token_region": token_region if kind == "activation" else None,
                                "target_count": counts[target], "other_count": other_count,
                                "measured_component_count": len(components),
                                "inventory_component_count": sum(ComponentId.parse(c).kind == granularity for c in component_ids),
                                "scores": scores, "target_means": target_means, "other_means": other_means})
    return vectors


def summarize_organization(vectors, *, top_k=5):
    """Within-target concentration and between-target overlap, by measurement."""
    results, pairs = [], []
    for vector in vectors:
        metadata = {key: value for key, value in vector.items() if key not in {"scores", "target_means", "other_means"}}
        results.append({**metadata, **concentration(vector["scores"], top_k=top_k)})
    for left, right in combinations(vectors, 2):
        keys = ("target_field", "granularity", "score_kind", "token_region")
        if any(left[key] != right[key] for key in keys):
            continue
        pairs.append({**{key: left[key] for key in keys}, "left_target": left["target_value"], "right_target": right["target_value"],
                      **overlap(left["scores"], right["scores"], top_k=top_k)})
    return {"concentration": results, "overlap": pairs,
            "definitions": {"distribution": "Positive scores normalized over the entire measured component universe, including zero mass.",
                            "top_k": "Up to k strictly positive components; score ties break by component ID.",
                            "weighted_overlap": "Sum of minima of normalized positive scores (1 minus total variation).",
                            "causal_contrast": "Target mean damage minus pooled non-target mean damage.",
                            "causal_scope": "Single-component damage conditional on an intact model; not additive subset importance.",
                            "zero_signal": "Undefined concentration/overlap returned as null, not evidence of localization.",
                            "interpretation": "Descriptive diagnostics; activation, causal importance and compressibility remain separate."}}


def stability(left, right, *, top_k=5):
    """Match entire profiles across discovery runs; no model/inventory alignment."""
    keys = ("target_field", "target_value", "granularity", "score_kind", "token_region")
    keyed = lambda vectors: {tuple(v[k] for k in keys): v for v in vectors}
    a, b = keyed(left), keyed(right)
    if set(a) != set(b):
        raise ValueError("Stability requires identical target/granularity/measurement profiles")
    return [{**dict(zip(keys, key)), **overlap(a[key]["scores"], b[key]["scores"], top_k=top_k)} for key in a]
