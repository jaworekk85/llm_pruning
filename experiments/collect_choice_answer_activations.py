from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from llm_pruning.choice_records import ChoiceRecord, filter_choice_records, load_choice_records
from llm_pruning.hooks import ActivationCollector, ActivationRecord
from llm_pruning.mc_prompts import candidate_text, format_choice_prompt
from llm_pruning.metrics import selectivity_scores
from llm_pruning.models import DEFAULT_MODEL_NAME, load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect teacher-forced activations over answer tokens for multiple-choice records."
        )
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--choice-jsonl", type=Path, default=Path("data/prompt_sets/mmlu_mc.jsonl"))
    parser.add_argument("--split", choices=["discovery", "validation", "test"], default="discovery")
    parser.add_argument("--domains", nargs="*", default=None)
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--max-records-per-domain", type=int, default=None)
    parser.add_argument("--output-csv", type=Path, default=Path("results/mmlu_mc_answer_activations.csv"))
    parser.add_argument(
        "--granularity",
        choices=["mlp_module", "mlp_neuron", "attention_head"],
        default="attention_head",
    )
    parser.add_argument("--module-filter", default=None)
    parser.add_argument("--top-units-per-module", type=int, default=None)
    parser.add_argument(
        "--choice-scope",
        choices=["correct", "all"],
        default="correct",
        help="Collect activations for only the correct answer text or for all candidates.",
    )
    parser.add_argument(
        "--scoring-mode",
        choices=["letter", "choice_text"],
        default="choice_text",
        help="Answer continuation text to append after the prompt.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="auto")
    return parser.parse_args()


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


def choice_indices(record: ChoiceRecord, choice_scope: str) -> list[int]:
    if choice_scope == "correct":
        return [record.answer_index]
    return list(range(len(record.choices)))


def collect_records(args: argparse.Namespace) -> list[ActivationRecord]:
    records_by_domain = load_eval_records(args)
    loaded = load_model(args.model_name, device=args.device, dtype=args.dtype)
    collector = ActivationCollector(
        loaded.model,
        granularity=args.granularity,
        module_filter=args.module_filter,
        top_units_per_module=args.top_units_per_module,
    )

    records: list[ActivationRecord] = []
    collector.start()
    try:
        for domain, domain_records in records_by_domain.items():
            print(f"Collecting {domain}: {len(domain_records)} record(s)")
            for record_index, record in enumerate(domain_records):
                prompt = format_choice_prompt(record)
                prompt_inputs = loaded.tokenizer(prompt, return_tensors="pt")
                prompt_length = int(prompt_inputs["input_ids"].shape[1])

                for choice_index in choice_indices(record, args.choice_scope):
                    answer = candidate_text(record, choice_index, args.scoring_mode)
                    full_text = prompt + answer
                    inputs = loaded.tokenizer(full_text, return_tensors="pt").to(loaded.device)
                    prompt_choice_index = record_index * len(record.choices) + choice_index
                    collector.set_context(domain, prompt_choice_index)
                    collector.set_token_slice(prompt_length, None)
                    with torch.no_grad():
                        loaded.model(**inputs)
                    records.extend(collector.drain_records())
    finally:
        collector.clear_token_slice()
        collector.close()

    return records


def write_records(records: list[ActivationRecord], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "domain",
                "prompt_index",
                "granularity",
                "module_name",
                "unit_index",
                "mean_abs",
                "std",
                "max_abs",
                "numel",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(record.__dict__)


def print_selectivity(records: list[ActivationRecord], top_k: int) -> None:
    domains = sorted({record.domain for record in records})
    print("\nTop answer-token domain-selective components:")
    for domain in domains:
        print(f"\n{domain}")
        for module_name, score, target_mean, other_mean in selectivity_scores(records, domain)[:top_k]:
            print(
                f"  {module_name}: score={score:.4f}, "
                f"target_mean={target_mean:.6f}, other_mean={other_mean:.6f}"
            )


def main() -> None:
    args = parse_args()
    records = collect_records(args)
    write_records(records, args.output_csv)
    print(f"\nWrote {len(records)} answer-token activation records to {args.output_csv}")
    print_selectivity(records, args.top_k)


if __name__ == "__main__":
    main()
