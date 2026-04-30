from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from llm_pruning.ablations import AblationManager
from llm_pruning.choice_records import ChoiceRecord, filter_choice_records, load_choice_records
from llm_pruning.mc_prompts import candidate_text, format_choice_prompt
from llm_pruning.models import DEFAULT_MODEL_NAME, load_model


@dataclass(frozen=True)
class ComponentScoreRow:
    domain: str
    component: str
    selectivity: float
    effect_size: float


@dataclass(frozen=True)
class ComponentSelection:
    top_target: list[str]
    random_controls: list[list[str]]


@dataclass(frozen=True)
class CandidateScore:
    record_id: str
    condition: str
    domain: str
    subject: str
    split: str
    choice_index: int
    is_correct: bool
    mean_logprob: float
    sum_logprob: float
    token_count: int


@dataclass(frozen=True)
class RecordDecision:
    record_id: str
    condition: str
    domain: str
    subject: str
    split: str
    answer_index: int
    predicted_index: int
    is_correct: bool
    correct_mean_logprob: float
    correct_loss: float
    margin: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate multiple-choice accuracy/log-likelihood under optional ablation."
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--choice-jsonl", type=Path, default=Path("data/prompt_sets/mmlu_mc.jsonl"))
    parser.add_argument("--split", choices=["discovery", "validation", "test"], default="validation")
    parser.add_argument("--domains", nargs="*", default=None)
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--max-records-per-domain", type=int, default=None)
    parser.add_argument("--component-scores", type=Path, default=None)
    parser.add_argument(
        "--granularity",
        choices=["none", "mlp_module", "mlp_neuron", "attention_head"],
        default="none",
    )
    parser.add_argument("--target-domain", default=None)
    parser.add_argument("--component-count", type=int, default=5)
    parser.add_argument("--random-control-repeats", type=int, default=0)
    parser.add_argument(
        "--control-strategy",
        choices=["random", "layer_matched"],
        default="random",
    )
    parser.add_argument(
        "--ranking-metric",
        choices=["selectivity", "effect_size"],
        default="selectivity",
    )
    parser.add_argument(
        "--scoring-mode",
        choices=["letter", "choice_text"],
        default="letter",
        help="Score answer letters (MMLU-style) or full answer choice text.",
    )
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-dir", type=Path, default=Path("results/mc_eval"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="auto")
    return parser.parse_args()


def load_component_scores(path: Path) -> list[ComponentScoreRow]:
    rows: list[ComponentScoreRow] = []
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            rows.append(
                ComponentScoreRow(
                    domain=row["domain"],
                    component=row["component"],
                    selectivity=float(row["selectivity"]),
                    effect_size=float(row["effect_size"]),
                )
            )
    if not rows:
        raise ValueError(f"No component scores found in {path}")
    return rows


def component_module_name(component: str) -> str:
    return component.rsplit("[", maxsplit=1)[0] if component.endswith("]") else component


def select_components(
    rows: list[ComponentScoreRow],
    target_domain: str,
    ranking_metric: str,
    component_count: int,
    random_seed: int,
    random_control_repeats: int,
    control_strategy: str,
) -> ComponentSelection:
    target_rows = [row for row in rows if row.domain == target_domain]
    if not target_rows:
        raise ValueError(f"No component scores found for domain: {target_domain}")

    ranked_rows = sorted(target_rows, key=lambda row: getattr(row, ranking_metric), reverse=True)
    selected_components = [
        row.component
        for row in ranked_rows
        if getattr(row, ranking_metric) > 0
    ][:component_count]
    if len(selected_components) < component_count:
        raise ValueError(
            f"Only found {len(selected_components)} positive {ranking_metric} components."
        )

    selected_set = set(selected_components)
    rng = random.Random(random_seed)
    random_controls: list[list[str]] = []

    if control_strategy == "random":
        pool = [row.component for row in target_rows if row.component not in selected_set]
        if len(pool) < component_count:
            raise ValueError("Not enough random-control components.")
        random_controls = [
            rng.sample(pool, k=component_count)
            for _repeat_index in range(random_control_repeats)
        ]
    else:
        components_by_module: dict[str, list[str]] = {}
        for row in target_rows:
            if row.component in selected_set:
                continue
            components_by_module.setdefault(component_module_name(row.component), []).append(row.component)

        selected_modules = [component_module_name(component) for component in selected_components]
        for _repeat_index in range(random_control_repeats):
            used: set[str] = set()
            control_set: list[str] = []
            for module in selected_modules:
                candidates = [
                    component
                    for component in components_by_module.get(module, [])
                    if component not in used
                ]
                if not candidates:
                    raise ValueError(f"No layer-matched control candidate for module {module}")
                sampled = rng.choice(candidates)
                control_set.append(sampled)
                used.add(sampled)
            random_controls.append(control_set)

    return ComponentSelection(top_target=selected_components, random_controls=random_controls)


def load_eval_records(args: argparse.Namespace) -> dict[str, list[ChoiceRecord]]:
    records = filter_choice_records(
        load_choice_records(args.choice_jsonl),
        split=args.split,
        domains=args.domains,
        subjects=args.subjects,
    )
    by_domain: dict[str, list[ChoiceRecord]] = {}
    for record in records:
        by_domain.setdefault(record.domain, []).append(record)

    if args.max_records_per_domain is None:
        return by_domain

    return {
        domain: domain_records[: args.max_records_per_domain]
        for domain, domain_records in by_domain.items()
    }


def ensure_padding_token(loaded) -> None:
    if loaded.tokenizer.pad_token is None:
        loaded.tokenizer.pad_token = loaded.tokenizer.eos_token


def batched(items: list[tuple[ChoiceRecord, int]], batch_size: int) -> list[list[tuple[ChoiceRecord, int]]]:
    if batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    return [items[start : start + batch_size] for start in range(0, len(items), batch_size)]


def score_candidate_batch(
    loaded,
    candidates: list[tuple[ChoiceRecord, int]],
    condition: str,
    scoring_mode: str,
) -> list[CandidateScore]:
    ensure_padding_token(loaded)
    prompts = [format_choice_prompt(record) for record, _choice_index in candidates]
    continuations = [
        candidate_text(record, choice_index, scoring_mode)
        for record, choice_index in candidates
    ]
    texts = [
        f"{prompt}{continuation}"
        for prompt, continuation in zip(prompts, continuations, strict=True)
    ]

    inputs = loaded.tokenizer(texts, return_tensors="pt", padding=True).to(loaded.device)
    labels = inputs["input_ids"].clone()
    labels[inputs["attention_mask"] == 0] = -100
    prompt_token_counts = [
        int(loaded.tokenizer(prompt, return_tensors="pt")["input_ids"].shape[-1])
        for prompt in prompts
    ]
    for row_index, prompt_token_count in enumerate(prompt_token_counts):
        labels[row_index, :prompt_token_count] = -100

    with torch.no_grad():
        logits = loaded.model(**inputs, use_cache=False).logits

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    losses = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view(shift_labels.shape)

    scores: list[CandidateScore] = []
    valid_tokens = shift_labels.ne(-100)
    for row_index, (record, choice_index) in enumerate(candidates):
        token_count = int(valid_tokens[row_index].sum().item())
        if token_count == 0:
            raise ValueError(f"No scored tokens for {record.id} choice {choice_index}")

        token_loss_sum = float(losses[row_index][valid_tokens[row_index]].sum().item())
        mean_logprob = -token_loss_sum / token_count
        scores.append(
            CandidateScore(
                record_id=record.id,
                condition=condition,
                domain=record.domain,
                subject=record.subject,
                split=record.split,
                choice_index=choice_index,
                is_correct=choice_index == record.answer_index,
                mean_logprob=mean_logprob,
                sum_logprob=-token_loss_sum,
                token_count=token_count,
            )
        )
    return scores


def evaluate_condition(
    loaded,
    records_by_domain: dict[str, list[ChoiceRecord]],
    condition: str,
    batch_size: int,
    scoring_mode: str,
) -> list[CandidateScore]:
    scores: list[CandidateScore] = []
    for domain, records in records_by_domain.items():
        print(f"Evaluating {condition}: {domain} ({len(records)} record(s))")
        candidates = [
            (record, choice_index)
            for record in records
            for choice_index in range(len(record.choices))
        ]
        for candidate_batch in batched(candidates, batch_size):
            scores.extend(
                score_candidate_batch(
                    loaded,
                    candidate_batch,
                    condition=condition,
                    scoring_mode=scoring_mode,
                )
            )
    return scores


def decisions_from_scores(scores: list[CandidateScore], records_by_id: dict[str, ChoiceRecord]) -> list[RecordDecision]:
    grouped: dict[tuple[str, str], list[CandidateScore]] = {}
    for score in scores:
        grouped.setdefault((score.condition, score.record_id), []).append(score)

    decisions: list[RecordDecision] = []
    for (condition, record_id), candidate_scores in grouped.items():
        record = records_by_id[record_id]
        ranked = sorted(candidate_scores, key=lambda score: score.mean_logprob, reverse=True)
        predicted = ranked[0]
        correct = [
            score
            for score in candidate_scores
            if score.choice_index == record.answer_index
        ][0]
        best_wrong = max(
            score.mean_logprob
            for score in candidate_scores
            if score.choice_index != record.answer_index
        )
        decisions.append(
            RecordDecision(
                record_id=record_id,
                condition=condition,
                domain=record.domain,
                subject=record.subject,
                split=record.split,
                answer_index=record.answer_index,
                predicted_index=predicted.choice_index,
                is_correct=predicted.choice_index == record.answer_index,
                correct_mean_logprob=correct.mean_logprob,
                correct_loss=-correct.mean_logprob,
                margin=correct.mean_logprob - best_wrong,
            )
        )
    return decisions


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def population_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / len(values))


def summarize_decisions(decisions: list[RecordDecision]) -> list[dict[str, object]]:
    baseline_by_domain = {
        domain: {
            "accuracy": mean([
                1.0 if decision.is_correct else 0.0
                for decision in decisions
                if decision.condition == "baseline" and decision.domain == domain
            ]),
            "correct_loss": mean([
                decision.correct_loss
                for decision in decisions
                if decision.condition == "baseline" and decision.domain == domain
            ]),
            "margin": mean([
                decision.margin
                for decision in decisions
                if decision.condition == "baseline" and decision.domain == domain
            ]),
        }
        for domain in sorted({decision.domain for decision in decisions})
    }

    rows: list[dict[str, object]] = []
    for condition in sorted({decision.condition for decision in decisions}):
        for domain in sorted({decision.domain for decision in decisions}):
            selected = [
                decision
                for decision in decisions
                if decision.condition == condition and decision.domain == domain
            ]
            if not selected:
                continue
            accuracy = mean([1.0 if decision.is_correct else 0.0 for decision in selected])
            correct_loss = mean([decision.correct_loss for decision in selected])
            margin = mean([decision.margin for decision in selected])
            rows.append(
                {
                    "condition": condition,
                    "domain": domain,
                    "record_count": len(selected),
                    "accuracy": accuracy,
                    "baseline_accuracy": baseline_by_domain[domain]["accuracy"],
                    "delta_accuracy": accuracy - baseline_by_domain[domain]["accuracy"],
                    "correct_loss": correct_loss,
                    "baseline_correct_loss": baseline_by_domain[domain]["correct_loss"],
                    "delta_correct_loss": correct_loss - baseline_by_domain[domain]["correct_loss"],
                    "margin": margin,
                    "baseline_margin": baseline_by_domain[domain]["margin"],
                    "delta_margin": margin - baseline_by_domain[domain]["margin"],
                }
            )
    return rows


def summarize_contrasts(summary_rows: list[dict[str, object]], target_domain: str | None) -> list[dict[str, object]]:
    random_conditions = sorted(
        {
            str(row["condition"])
            for row in summary_rows
            if str(row["condition"]).startswith("random_control_")
        }
    )
    by_condition_domain = {
        (str(row["condition"]), str(row["domain"])): row
        for row in summary_rows
    }
    rows: list[dict[str, object]] = []
    for domain in sorted({str(row["domain"]) for row in summary_rows}):
        top_row = by_condition_domain.get(("top_target", domain))
        if top_row is None:
            continue
        random_loss_deltas = [
            float(by_condition_domain[(condition, domain)]["delta_correct_loss"])
            for condition in random_conditions
            if (condition, domain) in by_condition_domain
        ]
        random_accuracy_deltas = [
            float(by_condition_domain[(condition, domain)]["delta_accuracy"])
            for condition in random_conditions
            if (condition, domain) in by_condition_domain
        ]
        random_margin_deltas = [
            float(by_condition_domain[(condition, domain)]["delta_margin"])
            for condition in random_conditions
            if (condition, domain) in by_condition_domain
        ]
        rows.append(
            {
                "domain": domain,
                "is_target_domain": domain == target_domain,
                "top_delta_accuracy": top_row["delta_accuracy"],
                "random_mean_delta_accuracy": mean(random_accuracy_deltas),
                "top_minus_random_delta_accuracy": float(top_row["delta_accuracy"]) - mean(random_accuracy_deltas),
                "top_delta_correct_loss": top_row["delta_correct_loss"],
                "random_mean_delta_correct_loss": mean(random_loss_deltas),
                "random_std_delta_correct_loss": population_std(random_loss_deltas),
                "top_minus_random_delta_correct_loss": float(top_row["delta_correct_loss"]) - mean(random_loss_deltas),
                "top_delta_margin": top_row["delta_margin"],
                "random_mean_delta_margin": mean(random_margin_deltas),
                "top_minus_random_delta_margin": float(top_row["delta_margin"]) - mean(random_margin_deltas),
                "random_control_repeats": len(random_loss_deltas),
            }
        )
    return rows


def write_dataclass_rows(rows, output_path: Path) -> None:
    if not rows:
        return
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].__dict__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def write_dict_rows(rows: list[dict[str, object]], output_path: Path) -> None:
    if not rows:
        return
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_components(components: dict[str, list[str]], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["condition", "component"])
        writer.writeheader()
        for condition, component_list in components.items():
            for component in component_list:
                writer.writerow({"condition": condition, "component": component})


def print_summary(rows: list[dict[str, object]], target_domain: str | None) -> None:
    print("\nMultiple-choice summary:")
    for row in rows:
        if row["condition"] == "baseline":
            continue
        marker = " < target" if row["domain"] == target_domain else ""
        print(
            f"  {row['condition']} | {row['domain']}: "
            f"delta_acc={row['delta_accuracy']:.4f}, "
            f"delta_loss={row['delta_correct_loss']:.4f}, "
            f"delta_margin={row['delta_margin']:.4f}{marker}"
        )


def main() -> None:
    args = parse_args()
    records_by_domain = load_eval_records(args)
    records_by_id = {
        record.id: record
        for records in records_by_domain.values()
        for record in records
    }

    condition_components: dict[str, list[str]] = {}
    if args.granularity != "none":
        if args.component_scores is None:
            raise ValueError("--component-scores is required when --granularity is not none.")
        if args.target_domain is None:
            raise ValueError("--target-domain is required when --granularity is not none.")
        selection = select_components(
            load_component_scores(args.component_scores),
            target_domain=args.target_domain,
            ranking_metric=args.ranking_metric,
            component_count=args.component_count,
            random_seed=args.random_seed,
            random_control_repeats=args.random_control_repeats,
            control_strategy=args.control_strategy,
        )
        condition_components["top_target"] = selection.top_target
        for repeat_index, components in enumerate(selection.random_controls):
            condition_components[f"random_control_{repeat_index:03d}"] = components

    loaded = load_model(args.model_name, device=args.device, dtype=args.dtype)
    candidate_scores = evaluate_condition(
        loaded,
        records_by_domain,
        condition="baseline",
        batch_size=args.batch_size,
        scoring_mode=args.scoring_mode,
    )

    for condition, components in condition_components.items():
        with AblationManager(loaded.model, components, granularity=args.granularity):
            candidate_scores.extend(
                evaluate_condition(
                    loaded,
                    records_by_domain,
                    condition=condition,
                    batch_size=args.batch_size,
                    scoring_mode=args.scoring_mode,
                )
            )

    decisions = decisions_from_scores(candidate_scores, records_by_id)
    summary_rows = summarize_decisions(decisions)
    contrast_rows = summarize_contrasts(summary_rows, args.target_domain)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_dataclass_rows(candidate_scores, args.output_dir / "candidate_scores.csv")
    write_dataclass_rows(decisions, args.output_dir / "decisions.csv")
    write_dict_rows(summary_rows, args.output_dir / "summary.csv")
    write_dict_rows(contrast_rows, args.output_dir / "contrasts.csv")
    if condition_components:
        write_components(condition_components, args.output_dir / "components.csv")
    print_summary(summary_rows, args.target_domain)
    print(f"\nWrote multiple-choice evaluation outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
