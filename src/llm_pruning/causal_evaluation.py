"""Reusable component-mask controls and paired held-out MC damage measurements."""
from collections import defaultdict
from dataclasses import replace
from math import comb, prod
import random
from statistics import mean, pstdev

from .component_masks import ComponentMask
from .mc_evaluation import batched, decisions_from_scores, score_candidate_batch


CONTROL_STRATEGIES = {"random", "layer_matched", "cost_matched", "layer_cost_matched"}
DAMAGE_DEFINITIONS = {
    "accuracy_damage": "Baseline accuracy minus masked accuracy; positive means damage.",
    "loss_increase": "Masked correct-answer loss minus baseline loss (nats); positive means damage.",
    "margin_damage": "Baseline correct-vs-best-wrong margin minus masked margin (nats).",
    "control_spread": "Mean and population std across recorded random masks, not a confidence interval.",
    "cost": "Input-channel count x downstream projection output width; a matching proxy, not physical pruning or runtime savings.",
}


def component_cost(registry, spec):
    module = dict(registry.model.named_modules())[spec.module_name]
    return (spec.channel_stop - spec.channel_start) * module.out_features


def _combination_at(pool, count, rank):
    chosen, start = [], 0
    for slots in range(count, 0, -1):
        for index in range(start, len(pool)):
            block = comb(len(pool) - index - 1, slots - 1)
            if rank < block:
                chosen.append(pool[index])
                start = index + 1
                break
            rank -= block
    return chosen


def build_conditions(registry, selected, *, seed=0, repeats=3, strategies=("random", "layer_cost_matched"), return_sampling=False):
    if type(seed) is not int or seed < 0 or type(repeats) is not int or repeats < 1:
        raise ValueError("seed must be nonnegative and repeats must be positive integers")
    if not strategies or len(set(strategies)) != len(strategies) or set(strategies) - CONTROL_STRATEGIES:
        raise ValueError("Provide unique supported control strategies")
    specs = [registry.resolve(value) for value in selected]
    if not specs or len({spec.component_id for spec in specs}) != len(specs):
        raise ValueError("Selected components must be nonempty and unique")
    selected_ids = {spec.component_id for spec in specs}
    pool = [spec for spec in registry.components if spec.component_id not in selected_ids]
    conditions = {"baseline": [], "selected": [str(spec.component_id) for spec in specs]}
    rng = random.Random(seed)
    sampling = {}
    costs = {spec.component_id: component_cost(registry, spec) for spec in registry.components}
    for strategy in strategies:
        def key(spec):
            return (
                spec.component_id.kind,
                spec.component_id.layer if strategy in {"layer_matched", "layer_cost_matched"} else None,
                costs[spec.component_id] if strategy in {"cost_matched", "layer_cost_matched"} else None,
            )
        required, available = defaultdict(list), defaultdict(list)
        for spec in specs:
            required[key(spec)].append(spec)
        for spec in pool:
            available[key(spec)].append(spec)
        for bucket, selected_specs in required.items():
            if len(available[bucket]) < len(selected_specs):
                raise ValueError(f"Insufficient disjoint {strategy} controls for {bucket}; reduce top_k or change grouping")
        buckets = list(required)
        sizes = [comb(len(available[bucket]), len(required[bucket])) for bucket in buckets]
        population = prod(sizes)
        count = min(repeats, population)
        # Floyd sampling works even when the finite space exceeds sys.maxsize.
        ranks = set()
        for upper in range(population - count, population):
            candidate = rng.randrange(upper + 1)
            ranks.add(upper if candidate in ranks else candidate)
        ranks = sorted(ranks)
        sampling[strategy] = {
            "requested_masks": repeats, "unique_masks": count,
            "available_masks": population, "exhaustive": count == population,
            "sampling": "uniform_without_replacement",
        }
        for repeat, rank in enumerate(ranks):
            control = []
            for bucket, size in zip(buckets, sizes, strict=True):
                rank, local_rank = divmod(rank, size)
                control.extend(_combination_at(available[bucket], len(required[bucket]), local_rank))
            conditions[f"{strategy}_{repeat:03d}"] = [str(spec.component_id) for spec in control]
    return (conditions, sampling) if return_sampling else conditions


def condition_budgets(registry, conditions):
    return {name: {
        "component_count": len(values),
        "channel_count": sum(registry.resolve(value).channel_stop - registry.resolve(value).channel_start for value in values),
        "projection_weight_elements": sum(component_cost(registry, registry.resolve(value)) for value in values),
    } for name, values in conditions.items()}


def evaluate_conditions(loaded, registry, grouped, config, conditions):
    """Same examples/candidates for each condition; remove mask hooks on failure."""
    if registry.model is not loaded.model or loaded.model.training:
        raise ValueError("Use an eval model and its own component registry")
    if conditions.get("baseline") != [] or not any(values for name, values in conditions.items() if name != "baseline"):
        raise ValueError("Conditions require an unmasked baseline and a nonempty intervention")
    if "selected" in conditions and not conditions["selected"]:
        raise ValueError("The selected mask must be nonempty")
    records = [record for items in grouped.values() for record in items]
    if not records or any(record.split != config.split for record in records):
        raise ValueError("Evaluation records must belong to the configured split")
    # Resolve every ID before starting any model evaluation.
    for values in conditions.values():
        for value in values:
            registry.resolve(value)
    by_id = {record.id: record for record in records}
    if len(by_id) != len(records):
        raise ValueError("Evaluation record IDs must be unique")
    scores = []
    candidates = [(record, index) for record in records for index in range(len(record.choices))]
    score_cache = {}
    for name, components in conditions.items():
        mask_key = tuple(sorted(components))
        if mask_key in score_cache:
            scores.extend(replace(row, condition=name) for row in score_cache[mask_key])
            continue
        condition_scores = []
        with ComponentMask(registry, components):
            for batch in batched(candidates, config.batch_size):
                condition_scores.extend(score_candidate_batch(
                    loaded, batch, name, config.scoring_mode, prompt_policy=config.prompt_policy,
                ))
        score_cache[mask_key] = condition_scores
        scores.extend(condition_scores)
    decisions = decisions_from_scores(scores, by_id)
    baseline = {decision.record_id: decision for decision in decisions if decision.condition == "baseline"}
    paired = []
    groups = defaultdict(list)
    for decision in decisions:
        reference = baseline[decision.record_id]
        value = getattr(by_id[decision.record_id], config.target.field)
        damage = {
            "accuracy_damage": int(reference.is_correct) - int(decision.is_correct),
            "loss_increase": decision.correct_loss - reference.correct_loss,
            "margin_damage": reference.margin - decision.margin,
        }
        paired.append({
            "record_id": decision.record_id, "split": config.split, "condition": decision.condition,
            "target_field": config.target.field, "target_value": value,
            "is_target": value == config.target.value, **damage,
        })
        groups[(decision.condition, value)].append((decision, damage))
    metrics = []
    for (name, value), items in groups.items():
        metrics.append({
            "condition": name, "target_field": config.target.field, "target_value": value,
            "is_target": value == config.target.value, "record_count": len(items),
            "accuracy": mean(int(item.is_correct) for item, _ in items),
            "correct_loss": mean(item.correct_loss for item, _ in items),
            "margin": mean(item.margin for item, _ in items),
            **{key: mean(damage[key] for _, damage in items) for key in ("accuracy_damage", "loss_increase", "margin_damage")},
        })
    contrasts = []
    for value in grouped:
        if "selected" not in conditions:
            continue
        selected_metrics = next(row for row in metrics if row["condition"] == "selected" and row["target_value"] == value)
        for strategy in sorted(CONTROL_STRATEGIES):
            controls = [row for row in metrics if row["condition"].startswith(strategy + "_") and row["target_value"] == value]
            # Avoid treating layer_cost_matched as layer_matched, etc.
            if not controls:
                continue
            contrast = {"target_field": config.target.field, "target_value": value, "is_target": value == config.target.value,
                        "strategy": strategy, "repeats": len(controls)}
            for key in ("accuracy_damage", "loss_increase", "margin_damage"):
                control_values = [row[key] for row in controls]
                contrast[f"mean_{key}"] = mean(control_values)
                contrast[f"std_{key}"] = pstdev(control_values)
                contrast[f"selected_minus_control_{key}"] = selected_metrics[key] - mean(control_values)
            contrasts.append(contrast)
    return scores, decisions, paired, metrics, contrasts
