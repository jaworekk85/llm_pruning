"""Independent target-calibrated pruning and same-example specialist frontiers."""
from dataclasses import asdict
import math

import torch

from .mc_evaluation import batched, decisions_from_scores, score_candidate_batch
from .mc_prompts import format_choice_prompt, prompt_policy_metadata
from .wanda import WandaPruning, model_cost


DEFAULT_BUDGETS = (1.0, .9, .75, .6, .5, .4)


def validate_frontier_settings(budgets, calibration_records, core_values=(), max_accuracy_drop=.02, core_min_accuracy=None):
    if not budgets or len(set(budgets)) != len(budgets) or 1.0 not in budgets:
        raise ValueError("Provide unique retained fractions including the dense baseline 1.0")
    if any(isinstance(b, bool) or not isinstance(b, (int, float)) or not math.isfinite(b) or not 0 < b <= 1 for b in budgets):
        raise ValueError("Retained fractions must be finite in (0, 1]")
    if type(calibration_records) is not int or calibration_records < 1:
        raise ValueError("calibration_records must be a positive integer")
    if len(set(core_values)) != len(core_values) or any(not isinstance(v, str) or not v.strip() for v in core_values):
        raise ValueError("core_values must be unique explicit labels")
    if isinstance(max_accuracy_drop, bool) or not isinstance(max_accuracy_drop, (int, float)) or not math.isfinite(max_accuracy_drop) or not 0 <= max_accuracy_drop <= 1:
        raise ValueError("max_accuracy_drop must be in [0, 1]")
    if core_min_accuracy is not None:
        if not core_values or isinstance(core_min_accuracy, bool) or not isinstance(core_min_accuracy, (int, float)) or not math.isfinite(core_min_accuracy) or not 0 <= core_min_accuracy <= 1:
            raise ValueError("core_min_accuracy requires explicit core_values and a value in [0, 1]")


def calibration_batches(loaded, records, config):
    """Prompt-only target examples; no correct answers or evaluation examples."""
    if not records or any(record.split != "discovery" or not config.target.matches(record) for record in records):
        raise ValueError("Wanda calibration requires target-only discovery records")
    protocol = prompt_policy_metadata(loaded.tokenizer, config.prompt_policy, config.scoring_mode)
    tokenizer = loaded.tokenizer
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Calibration requires a pad or EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    batches = []
    for start in range(0, len(records), config.batch_size):
        texts = [format_choice_prompt(record, policy=config.prompt_policy, tokenizer=tokenizer, scoring_mode=config.scoring_mode)
                 for record in records[start:start + config.batch_size]]
        encoded = tokenizer(texts, padding=True, truncation=False, add_special_tokens=protocol["add_special_tokens"], return_tensors="pt")
        ids, mask = encoded["input_ids"], encoded["attention_mask"]
        limit = getattr(loaded.model.config, "max_position_embeddings", None)
        if limit is not None and ids.shape[1] > limit:
            raise ValueError("Calibration exceeds context length; truncation is not allowed")
        positions = mask.long().cumsum(-1) - 1
        positions.masked_fill_(mask == 0, 0)
        batches.append({"input_ids": ids.cpu(), "attention_mask": mask.cpu(), "position_ids": positions.cpu()})
    return batches


def evaluate_frontier(loaded, config, grouped, calibration, *, budgets=DEFAULT_BUDGETS, core_values=(),
                      max_accuracy_drop=.02, core_min_accuracy=None, on_point=None):
    """Each point starts dense; no selection/localization scores enter pruning.

    on_point receives scores/decisions/paired deltas to stream without retaining
    every candidate across budgets. The returned frontier is descriptive; each
    point is the output of one specific pruning algorithm, not an optimum.
    """
    validate_frontier_settings(budgets, len(calibration), core_values, max_accuracy_drop, core_min_accuracy)
    if loaded.model.training or config.split not in {"validation", "test"}:
        raise ValueError("Frontiers require an eval model and validation/test examples")
    if config.target.value not in grouped or set(core_values) - set(grouped):
        raise ValueError("Target and all core_values must be present in evaluation")
    records = [r for items in grouped.values() for r in items]
    if not records or any(r.split != config.split for r in records):
        raise ValueError("Evaluation records must belong to the configured split")
    by_id = {r.id: r for r in records}
    if len(by_id) != len(records) or set(by_id) & {r.id for r in calibration}:
        raise ValueError("Evaluation IDs must be unique and disjoint from calibration")
    batches = calibration_batches(loaded, calibration, config)
    candidates = [(record, index) for record in records for index in range(len(record.choices))]
    panels = [("value", value, items) for value, items in grouped.items()]
    others = [r for r in records if not config.target.matches(r)]
    if others:
        panels.append(("non_target", "", others))
    if core_values:
        panels.append(("core", "", [r for r in records if getattr(r, config.target.field) in core_values]))
    frontier, reports = [], []
    baseline = {}
    dense_cost = model_cost(loaded.model)
    for budget in sorted(budgets, reverse=True):
        condition = f"retained_{budget:.12g}"
        with WandaPruning(loaded.model) as pruning:
            report = pruning.apply(batches, 1.0 - budget)
            cost = model_cost(loaded.model, pruning.groups)
            is_cuda = loaded.device.type == "cuda"
            if is_cuda:
                torch.cuda.synchronize(loaded.device)
                torch.cuda.reset_peak_memory_stats(loaded.device)
            scores = []
            for batch in batched(candidates, config.batch_size):
                scores.extend(score_candidate_batch(loaded, batch, condition, config.scoring_mode, prompt_policy=config.prompt_policy))
            decisions = decisions_from_scores(scores, by_id)
            peak = None
            if is_cuda:
                torch.cuda.synchronize(loaded.device)
                peak = torch.cuda.max_memory_allocated(loaded.device)
            indexed = {decision.record_id: decision for decision in decisions}
            if budget == 1:
                baseline = indexed
            paired = [{"retained_fraction": budget, "record_id": d.record_id, "split": config.split,
                       "target_field": config.target.field, "target_value": getattr(by_id[d.record_id], config.target.field),
                       "accuracy_damage": int(baseline[d.record_id].is_correct) - int(d.is_correct),
                       "loss_increase": d.correct_loss - baseline[d.record_id].correct_loss,
                       "margin_damage": baseline[d.record_id].margin - d.margin} for d in decisions]
            for panel, value, items in panels:
                current = [indexed[r.id] for r in items]
                reference = [baseline[r.id] for r in items]
                n = len(items)
                frontier.append({
                    "pruner": "wanda_sequential_v1", "retained_fraction": budget,
                    "realized_prunable_retained_fraction": cost["prunable_nonzero_count"] / cost["prunable_parameter_count"],
                    "realized_model_nonzero_fraction": cost["nonzero_parameter_count"] / cost["parameter_count"],
                    "panel": panel, "target_field": config.target.field, "target_value": value,
                    "is_target": panel == "value" and value == config.target.value,
                    "split": config.split, "record_count": n,
                    "accuracy": sum(int(d.is_correct) for d in current) / n,
                    "correct_loss": sum(d.correct_loss for d in current) / n,
                    "margin": sum(d.margin for d in current) / n,
                    "accuracy_damage": sum(int(b.is_correct) - int(d.is_correct) for b, d in zip(reference, current)) / n,
                    "loss_increase": sum(d.correct_loss - b.correct_loss for b, d in zip(reference, current)) / n,
                    "margin_damage": sum(b.margin - d.margin for b, d in zip(reference, current)) / n,
                    **cost, "evaluation_peak_cuda_allocated_bytes": peak,
                })
            reports.append({"retained_fraction": budget, **report})
            if on_point is not None:
                on_point(budget, [asdict(s) for s in scores], [asdict(d) for d in decisions], paired)
    feasible = []
    for row in frontier:
        if not row["is_target"] or row["accuracy_damage"] > max_accuracy_drop + 1e-12:
            continue
        core = next((r for r in frontier if r["retained_fraction"] == row["retained_fraction"] and r["panel"] == "core"), None)
        if core_min_accuracy is None or core["accuracy"] >= core_min_accuracy:
            feasible.append(row)
    minimum = min(feasible, key=lambda r: r["realized_prunable_retained_fraction"], default=None)
    return {"rows": frontier, "pruning_reports": reports, "dense_cost": dense_cost,
            "feasibility": {"max_accuracy_drop": max_accuracy_drop, "core_min_accuracy": core_min_accuracy,
                            "minimum_observed_retained_fraction": None if minimum is None else minimum["retained_fraction"],
                            "minimum_observed_prunable_nonzero_fraction": None if minimum is None else minimum["realized_prunable_retained_fraction"],
                            "interpretation": "Descriptive minimum among fixed tested budgets; no confidence/power guarantee or optimality claim."}}
