"""Collect target-aware head/MLP-group summaries with separate prompt/answer regions."""
from __future__ import annotations

import argparse
import csv
import random
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for import_path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import torch

from llm_pruning.baseline_config import BaselineConfig
from llm_pruning.component_activations import (
    ActivationSample, ComponentActivation, ComponentActivationCollector, tokenize_prompt_answers,
)
from llm_pruning.components import ComponentRegistry
from llm_pruning.mc_prompts import candidate_text, format_choice_prompt, prompt_policy_metadata
from llm_pruning.models import load_model
from llm_pruning.record_selection import prepare_records
from llm_pruning.run_artifacts import RunArtifacts, file_fingerprint


MEASUREMENT = {
    "site": "Projection inputs: query-head outputs before o_proj; gated MLP channels before down_proj.",
    "token_regions": "Prompt and teacher-forced correct-answer input positions, separately; padding excluded.",
    "boundary": "Standalone prompt tokens must be an exact prefix of the full sequence; otherwise fail.",
    "prompt_policy": "Explicit configured policy; resolved template fingerprint in manifest.json.",
    "answer_policy": "correct_choice_teacher_forced; continuation from configured scoring_mode; not generation.",
    "statistics": "Per example/component/region over token x channel values in float32: mean, mean_abs, RMS, population std, max_abs.",
    "aggregation": "No cross-example pooling or component ranking; raw activation tensors are not saved.",
}


def validate_collection_settings(mlp_group_size: int, token_regions: tuple[str, ...]):
    if type(mlp_group_size) is not int or mlp_group_size <= 0:
        raise ValueError("mlp_group_size must be a positive integer")
    if (not token_regions or len(set(token_regions)) != len(token_regions)
            or set(token_regions) - {"prompt", "answer"}):
        raise ValueError("token_regions must contain unique prompt and/or answer values")


def run_collection(
    config: BaselineConfig, results_root: Path = REPO_ROOT / "results", *,
    mlp_group_size: int = 128, token_regions: tuple[str, ...] = ("prompt", "answer"),
) -> Path:
    validate_collection_settings(mlp_group_size, token_regions)
    output = results_root / config.experiment_name
    if output.exists():
        raise FileExistsError(f"Result directory already exists: {output}. Choose a new experiment_name.")
    grouped = prepare_records(config)
    selected = [record for records in grouped.values() for record in records]
    resolved_config = {
        **config.to_dict(), "mlp_group_size": mlp_group_size, "token_regions": list(token_regions),
    }
    with RunArtifacts(
        output, REPO_ROOT, resolved_config, dataset=file_fingerprint(config.dataset_path),
        model={"identifier": config.model_name, "requested_revision": config.revision},
        intervention_method="none", compression_policy=None, pruner=None,
        measurement_definitions=MEASUREMENT, run_kind="component_activations",
        selection={"split": config.split, "counts": {value: len(records) for value, records in grouped.items()}},
    ) as run:
        run.write_json("collection_records.json", [
            {key: getattr(record, key) for key in (
                "id", "split", "domain", "language", "capability", "task", "subject",
                "source_name", "source_type", "metadata", "answer_index",
            )} for record in selected
        ])
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        loaded = load_model(
            config.model_name, device=config.device, dtype=config.dtype,
            revision=config.revision, local_files_only=config.local_files_only,
        )
        protocol = prompt_policy_metadata(loaded.tokenizer, config.prompt_policy, config.scoring_mode)
        run.manifest["prompt_protocol"] = protocol
        registry = ComponentRegistry(loaded.model, mlp_group_size=mlp_group_size)
        collector = ComponentActivationCollector(registry, config.target)
        run.write_json("component_inventory.json", registry.to_dict())
        run.manifest["model"].update(
            resolved_revision=getattr(loaded.model.config, "_commit_hash", None),
            device=str(loaded.device), dtype=str(loaded.model.dtype), model_type=registry.model_type,
            attention_implementation=getattr(loaded.model.config, "_attn_implementation", None),
        )
        run.manifest["tokenizer"] = {
            "identifier": loaded.tokenizer.name_or_path, "padding_side": loaded.tokenizer.padding_side,
            "pad_token_id": loaded.tokenizer.pad_token_id,
            "special_tokens": loaded.tokenizer.special_tokens_map,
        }
        run.write_json("manifest.json", run.manifest)
        row_count = 0
        with (output / "activations.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(ComponentActivation.__dataclass_fields__))
            writer.writeheader()
            for start in range(0, len(selected), config.batch_size):
                records = selected[start:start + config.batch_size]
                inputs, masks = tokenize_prompt_answers(
                    loaded.tokenizer, [format_choice_prompt(record, policy=config.prompt_policy, tokenizer=loaded.tokenizer, scoring_mode=config.scoring_mode) for record in records],
                    [candidate_text(record, record.answer_index, config.scoring_mode) for record in records],
                    loaded.device, add_special_tokens=protocol["add_special_tokens"],
                )
                # Tokenization can resolve a missing pad token to EOS.
                run.manifest["tokenizer"]["pad_token_id"] = loaded.tokenizer.pad_token_id
                context_limit = getattr(loaded.model.config, "max_position_embeddings", None)
                if context_limit is not None and inputs["input_ids"].shape[1] > context_limit:
                    raise ValueError("Sequence exceeds model context length; truncation is not allowed")
                samples = [ActivationSample(record.id, record.split, getattr(record, config.target.field)) for record in records]
                rows = collector.collect(inputs, samples, {region: masks[region] for region in token_regions})
                writer.writerows(asdict(row) for row in rows)
                row_count += len(rows)
        run.complete([
            "# Component activation collection", "", "Status: complete.",
            f"Target: {config.target.field}={config.target.value}; split: {config.split}; seed: {config.seed}",
            f"Collected {len(selected)} examples across {len(grouped)} target-field values.",
            f"Components: {len(registry.components)}; MLP group size: {mlp_group_size}; regions: {', '.join(token_regions)}.",
            f"Summary rows: {row_count}; data: activations.csv; mapping: component_inventory.json.",
            "Selected IDs/source metadata: collection_records.json; definitions: manifest.json.",
            "Answers are teacher-forced correct continuations; no model generations were measured.",
            f"Prompt policy: {config.prompt_policy}; scientific prompt-quality validation remains necessary.",
            "Activation summaries do not establish causal importance, modularity or compressibility.",
        ], activation_row_count=row_count)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Shared target baseline JSON schema; use discovery for component discovery.")
    parser.add_argument("--results-root", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--mlp-group-size", type=int, default=128)
    parser.add_argument("--token-regions", nargs="+", choices=["prompt", "answer"], default=["prompt", "answer"])
    parser.add_argument("--validate-only", action="store_true", help="Check settings/config/data without loading a model or creating results.")
    args = parser.parse_args()
    validate_collection_settings(args.mlp_group_size, tuple(args.token_regions))
    config = BaselineConfig.load(args.config)
    if args.validate_only:
        grouped = prepare_records(config)
        print(f"Valid: {sum(map(len, grouped.values()))} records across {len(grouped)} values; split={config.split}; no model loaded.")
        return
    output = run_collection(config, args.results_root, mlp_group_size=args.mlp_group_size, token_regions=tuple(args.token_regions))
    print((output / "summary.md").read_text(encoding="utf-8"), end="")
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
