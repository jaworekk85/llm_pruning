from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from llm_pruning.ablations import AblationManager
from llm_pruning.models import DEFAULT_MODEL_NAME, format_chat_prompt, load_model
from llm_pruning.prompt_records import (
    PromptRecord,
    filter_prompt_records,
    group_records_by_domain,
    load_prompt_records,
)
from llm_pruning.prompts import (
    filter_domains,
    load_domain_prompts,
)


@dataclass(frozen=True)
class ComponentScoreRow:
    domain: str
    component: str
    selectivity: float
    effect_size: float


@dataclass(frozen=True)
class LossRecord:
    condition: str
    domain: str
    prompt_index: int
    loss: float


@dataclass(frozen=True)
class ComponentSelection:
    top_target: list[str]
    random_controls: list[list[str]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ablate selected model components and compare per-domain loss."
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--prompt-dir", type=Path, default=Path("data/prompts"))
    parser.add_argument("--prompt-jsonl", type=Path, default=Path("data/prompt_sets/seed_qa.jsonl"))
    parser.add_argument("--split", choices=["discovery", "validation", "test"], default="discovery")
    parser.add_argument("--domains", nargs="*", default=None)
    parser.add_argument("--max-prompts-per-domain", type=int, default=None)
    parser.add_argument(
        "--loss-scope",
        choices=["target", "prompt"],
        default="target",
        help=(
            "Use target-answer loss when JSONL records include targets. "
            "Use prompt only for legacy question-only smoke tests."
        ),
    )
    parser.add_argument("--component-scores", type=Path, required=True)
    parser.add_argument(
        "--granularity",
        choices=["mlp_module", "mlp_neuron", "attention_head"],
        required=True,
    )
    parser.add_argument("--target-domain", required=True)
    parser.add_argument("--component-count", type=int, default=5)
    parser.add_argument(
        "--random-control-repeats",
        type=int,
        default=10,
        help="Number of independently sampled random control component sets.",
    )
    parser.add_argument(
        "--ranking-metric",
        choices=["selectivity", "effect_size"],
        default="selectivity",
    )
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("results/ablation"))
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


def select_components(
    rows: list[ComponentScoreRow],
    target_domain: str,
    ranking_metric: str,
    component_count: int,
    random_seed: int,
    random_control_repeats: int,
) -> ComponentSelection:
    target_rows = [row for row in rows if row.domain == target_domain]
    if not target_rows:
        raise ValueError(f"No component scores found for domain: {target_domain}")

    ranked_rows = sorted(
        target_rows,
        key=lambda row: getattr(row, ranking_metric),
        reverse=True,
    )
    positive_rows = [row for row in ranked_rows if getattr(row, ranking_metric) > 0]
    selected_rows = positive_rows[:component_count]

    if len(selected_rows) < component_count:
        raise ValueError(
            f"Only found {len(selected_rows)} positive {ranking_metric} components "
            f"for {target_domain}, but {component_count} were requested."
        )

    selected_components = [row.component for row in selected_rows]

    selected_set = set(selected_components)
    random_pool = [row.component for row in target_rows if row.component not in selected_set]
    if len(random_pool) < component_count:
        raise ValueError("Not enough components left for random control selection.")

    rng = random.Random(random_seed)
    random_controls = [
        rng.sample(random_pool, k=component_count)
        for _repeat_index in range(random_control_repeats)
    ]
    return ComponentSelection(
        top_target=selected_components,
        random_controls=random_controls,
    )


def load_eval_records(args: argparse.Namespace) -> dict[str, list[PromptRecord]]:
    if args.prompt_jsonl is not None:
        records = load_prompt_records(args.prompt_jsonl)
        records = filter_prompt_records(records, split=args.split, domains=args.domains)
        prompts_by_domain = group_records_by_domain(records)
    else:
        prompt_text_by_domain = filter_domains(
            load_domain_prompts(args.prompt_dir),
            args.domains,
        )
        prompts_by_domain = {
            domain: [
                PromptRecord(
                    id=f"{domain}.legacy.{index:04d}",
                    domain=domain,
                    prompt=prompt,
                    split=args.split,
                    source_type="legacy_txt",
                    source_name=str(args.prompt_dir),
                    prompt_type="unknown",
                    difficulty="unknown",
                    language="en",
                )
                for index, prompt in enumerate(prompts)
            ]
            for domain, prompts in prompt_text_by_domain.items()
        }

    if args.max_prompts_per_domain is None:
        return prompts_by_domain

    return {
        domain: records[: args.max_prompts_per_domain]
        for domain, records in prompts_by_domain.items()
    }


def prompt_loss(loaded, question: str) -> float:
    prompt = format_chat_prompt(question)
    inputs = loaded.tokenizer(prompt, return_tensors="pt").to(loaded.device)
    labels = inputs["input_ids"].clone()

    with torch.no_grad():
        outputs = loaded.model(**inputs, labels=labels, use_cache=False)

    return float(outputs.loss.item())


def target_loss(loaded, question: str, target: str) -> float:
    prompt = format_chat_prompt(question)
    answer = target.strip()
    if loaded.tokenizer.eos_token:
        answer = f"{answer}{loaded.tokenizer.eos_token}"

    full_text = f"{prompt}{answer}"
    full_inputs = loaded.tokenizer(full_text, return_tensors="pt").to(loaded.device)
    prompt_inputs = loaded.tokenizer(prompt, return_tensors="pt").to(loaded.device)

    labels = full_inputs["input_ids"].clone()
    prompt_token_count = prompt_inputs["input_ids"].shape[-1]
    labels[:, :prompt_token_count] = -100

    with torch.no_grad():
        outputs = loaded.model(**full_inputs, labels=labels, use_cache=False)

    return float(outputs.loss.item())


def record_loss(loaded, record: PromptRecord, loss_scope: str) -> float:
    if loss_scope == "prompt":
        return prompt_loss(loaded, record.prompt)

    if record.target is None:
        raise ValueError(
            f"Record {record.id} has no target answer. "
            "Use --loss-scope prompt only for legacy smoke tests."
        )

    return target_loss(loaded, record.prompt, record.target)


def evaluate_condition(
    loaded,
    prompts_by_domain: dict[str, list[PromptRecord]],
    condition: str,
    loss_scope: str,
) -> list[LossRecord]:
    loss_records: list[LossRecord] = []
    for domain, prompt_records in prompts_by_domain.items():
        print(f"Evaluating {condition}: {domain} ({len(prompt_records)} prompt(s))")
        for prompt_index, prompt_record in enumerate(prompt_records):
            loss_records.append(
                LossRecord(
                    condition=condition,
                    domain=domain,
                    prompt_index=prompt_index,
                    loss=record_loss(loaded, prompt_record, loss_scope),
                )
            )
    return loss_records


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def population_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0

    avg = mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / len(values))


def summarize_losses(records: list[LossRecord]) -> list[dict[str, object]]:
    baseline_by_domain = {
        domain: mean([record.loss for record in records if record.condition == "baseline" and record.domain == domain])
        for domain in sorted({record.domain for record in records})
    }

    rows: list[dict[str, object]] = []
    for condition in sorted({record.condition for record in records}):
        for domain in sorted({record.domain for record in records}):
            losses = [
                record.loss
                for record in records
                if record.condition == condition and record.domain == domain
            ]
            if not losses:
                continue

            mean_loss = mean(losses)
            baseline_loss = baseline_by_domain[domain]
            rows.append(
                {
                    "condition": condition,
                    "domain": domain,
                    "prompt_count": len(losses),
                    "mean_loss": mean_loss,
                    "baseline_loss": baseline_loss,
                    "delta_loss": mean_loss - baseline_loss,
                }
            )

    return rows


def summarize_contrasts(
    summary_rows: list[dict[str, object]],
    target_domain: str,
) -> list[dict[str, object]]:
    by_condition_domain = {
        (str(row["condition"]), str(row["domain"])): float(row["delta_loss"])
        for row in summary_rows
    }
    domains = sorted({str(row["domain"]) for row in summary_rows})
    random_conditions = sorted(
        {
            str(row["condition"])
            for row in summary_rows
            if str(row["condition"]).startswith("random_control_")
        }
    )

    rows: list[dict[str, object]] = []
    for domain in domains:
        top_delta = by_condition_domain.get(("top_target", domain), 0.0)
        random_deltas = [
            by_condition_domain[(condition, domain)]
            for condition in random_conditions
            if (condition, domain) in by_condition_domain
        ]
        random_mean = mean(random_deltas)
        random_std = population_std(random_deltas)
        rows.append(
            {
                "domain": domain,
                "is_target_domain": domain == target_domain,
                "top_target_delta_loss": top_delta,
                "random_control_mean_delta_loss": random_mean,
                "random_control_std_delta_loss": random_std,
                "top_minus_random_delta_loss": top_delta - random_mean,
                "random_control_repeats": len(random_deltas),
            }
        )

    other_top_deltas = [
        float(row["top_target_delta_loss"])
        for row in rows
        if not bool(row["is_target_domain"])
    ]
    target_rows = [row for row in rows if bool(row["is_target_domain"])]
    if target_rows:
        target_row = target_rows[0]
        target_row["target_specificity_over_other_domains"] = (
            float(target_row["top_target_delta_loss"]) - mean(other_top_deltas)
        )
    for row in rows:
        row.setdefault("target_specificity_over_other_domains", "")

    return rows


def write_detailed(records: list[LossRecord], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["condition", "domain", "prompt_index", "loss"],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(record.__dict__)


def write_summary(rows: list[dict[str, object]], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "condition",
                "domain",
                "prompt_count",
                "mean_loss",
                "baseline_loss",
                "delta_loss",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_contrasts(rows: list[dict[str, object]], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "domain",
                "is_target_domain",
                "top_target_delta_loss",
                "random_control_mean_delta_loss",
                "random_control_std_delta_loss",
                "top_minus_random_delta_loss",
                "random_control_repeats",
                "target_specificity_over_other_domains",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_components(components: dict[str, list[str]], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["condition", "component"])
        writer.writeheader()
        for condition, component_list in components.items():
            for component in component_list:
                writer.writerow({"condition": condition, "component": component})


def print_summary(rows: list[dict[str, object]], target_domain: str) -> None:
    print("\nAblation summary:")
    for row in rows:
        if row["condition"] == "baseline":
            continue
        marker = " < target" if row["domain"] == target_domain else ""
        print(
            f"  {row['condition']} | {row['domain']}: "
            f"delta_loss={row['delta_loss']:.4f}{marker}"
        )


def main() -> None:
    args = parse_args()
    prompts_by_domain = load_eval_records(args)
    score_rows = load_component_scores(args.component_scores)
    selection = select_components(
        score_rows,
        target_domain=args.target_domain,
        ranking_metric=args.ranking_metric,
        component_count=args.component_count,
        random_seed=args.random_seed,
        random_control_repeats=args.random_control_repeats,
    )

    print("Selected components:")
    for component in selection.top_target:
        print(f"  top_target: {component}")
    for repeat_index, random_components in enumerate(selection.random_controls):
        for component in random_components:
            print(f"  random_control_{repeat_index:03d}: {component}")

    loaded = load_model(args.model_name, device=args.device, dtype=args.dtype)

    records = evaluate_condition(
        loaded,
        prompts_by_domain,
        condition="baseline",
        loss_scope=args.loss_scope,
    )

    with AblationManager(
        loaded.model,
        selection.top_target,
        granularity=args.granularity,
    ):
        records.extend(
            evaluate_condition(
                loaded,
                prompts_by_domain,
                condition="top_target",
                loss_scope=args.loss_scope,
            )
        )

    for repeat_index, random_components in enumerate(selection.random_controls):
        with AblationManager(
            loaded.model,
            random_components,
            granularity=args.granularity,
        ):
            records.extend(
                evaluate_condition(
                    loaded,
                    prompts_by_domain,
                    condition=f"random_control_{repeat_index:03d}",
                    loss_scope=args.loss_scope,
                )
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = summarize_losses(records)
    contrast_rows = summarize_contrasts(summary_rows, args.target_domain)
    write_detailed(records, args.output_dir / "losses.csv")
    write_summary(summary_rows, args.output_dir / "summary.csv")
    write_contrasts(contrast_rows, args.output_dir / "contrasts.csv")
    component_rows = {"top_target": selection.top_target}
    for repeat_index, random_components in enumerate(selection.random_controls):
        component_rows[f"random_control_{repeat_index:03d}"] = random_components
    write_components(component_rows, args.output_dir / "components.csv")
    print_summary(summary_rows, args.target_domain)
    print("\nContrasts against random controls:")
    for row in contrast_rows:
        marker = " < target" if row["is_target_domain"] else ""
        print(
            f"  {row['domain']}: top_minus_random="
            f"{row['top_minus_random_delta_loss']:.4f}{marker}"
        )
    print(f"\nWrote ablation outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
