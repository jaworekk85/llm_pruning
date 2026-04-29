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

from llm_pruning.hooks import ActivationCollector, ActivationRecord, summarize_tensor
from llm_pruning.metrics import selectivity_scores
from llm_pruning.models import DEFAULT_MODEL_NAME, format_chat_prompt, load_model
from llm_pruning.prompts import (
    filter_domains,
    load_domain_prompts,
    load_domain_prompts_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect activation summaries for domain prompt sets."
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--prompt-dir", type=Path, default=Path("data/prompts"))
    parser.add_argument(
        "--prompt-jsonl",
        type=Path,
        default=None,
        help="Metadata-rich JSONL prompt set. If set, this overrides --prompt-dir.",
    )
    parser.add_argument(
        "--split",
        choices=["discovery", "validation", "test"],
        default=None,
        help="Optional split filter when using --prompt-jsonl.",
    )
    parser.add_argument("--output-csv", type=Path, default=Path("results/activations.csv"))
    parser.add_argument(
        "--granularity",
        choices=["layer", "mlp_module", "mlp_neuron", "attention_head"],
        default="mlp_module",
        help="Model component level to measure.",
    )
    parser.add_argument(
        "--module-filter",
        default=None,
        help=(
            "Optional module suffix filter. Defaults depend on granularity: "
            ".mlp, .mlp.gate_proj, or .self_attn.o_proj."
        ),
    )
    parser.add_argument(
        "--top-units-per-module",
        type=int,
        default=None,
        help="For neuron/head runs, keep only the strongest units per module and prompt.",
    )
    parser.add_argument("--domains", nargs="*", default=None)
    parser.add_argument("--max-prompts-per-domain", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="auto")
    return parser.parse_args()


def collect_records(args: argparse.Namespace) -> list[ActivationRecord]:
    if args.prompt_jsonl is not None:
        prompts_by_domain = load_domain_prompts_jsonl(args.prompt_jsonl, split=args.split)
    else:
        prompts_by_domain = load_domain_prompts(args.prompt_dir)
    prompts_by_domain = filter_domains(prompts_by_domain, args.domains)

    loaded = load_model(args.model_name, device=args.device, dtype=args.dtype)

    if args.granularity == "layer":
        return collect_layer_records(loaded, prompts_by_domain, args.max_prompts_per_domain)

    collector = ActivationCollector(
        loaded.model,
        granularity=args.granularity,
        module_filter=args.module_filter,
        top_units_per_module=args.top_units_per_module,
    )
    records: list[ActivationRecord] = []

    collector.start()
    try:
        for domain, prompts in prompts_by_domain.items():
            selected_prompts = prompts[: args.max_prompts_per_domain]
            print(f"Collecting {domain}: {len(selected_prompts)} prompt(s)")

            for prompt_index, question in enumerate(selected_prompts):
                prompt = format_chat_prompt(question)
                inputs = loaded.tokenizer(prompt, return_tensors="pt").to(loaded.device)

                collector.set_context(domain, prompt_index)
                with torch.no_grad():
                    loaded.model(**inputs)
                records.extend(collector.drain_records())
    finally:
        collector.close()

    return records


def collect_layer_records(
    loaded,
    prompts_by_domain: dict[str, list[str]],
    max_prompts_per_domain: int | None,
) -> list[ActivationRecord]:
    records: list[ActivationRecord] = []

    for domain, prompts in prompts_by_domain.items():
        selected_prompts = prompts[:max_prompts_per_domain]
        print(f"Collecting {domain}: {len(selected_prompts)} prompt(s)")

        for prompt_index, question in enumerate(selected_prompts):
            prompt = format_chat_prompt(question)
            inputs = loaded.tokenizer(prompt, return_tensors="pt").to(loaded.device)

            with torch.no_grad():
                outputs = loaded.model(
                    **inputs,
                    output_hidden_states=True,
                    use_cache=False,
                )

            for hidden_index, hidden_state in enumerate(outputs.hidden_states):
                module_name = "embedding" if hidden_index == 0 else f"layer.{hidden_index - 1}"
                records.append(
                    summarize_tensor(
                        domain=domain,
                        prompt_index=prompt_index,
                        granularity="layer",
                        module_name=module_name,
                        unit_index=None,
                        tensor=hidden_state,
                    )
                )

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

    print("\nTop domain-selective components:")
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
    print(f"\nWrote {len(records)} activation records to {args.output_csv}")
    print_selectivity(records, args.top_k)


if __name__ == "__main__":
    main()
