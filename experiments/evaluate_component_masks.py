"""Held-out target-aware MC mask evaluation from a verified discovery activation run."""
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
from llm_pruning.causal_evaluation import (
    CONTROL_STRATEGIES, DAMAGE_DEFINITIONS, build_conditions, condition_budgets, evaluate_conditions,
)
from llm_pruning.components import ComponentRegistry
from llm_pruning.discovery_selection import load_discovery_selection
from llm_pruning.mc_prompts import prompt_policy_metadata
from llm_pruning.mc_evaluation import MC_METRIC_DEFINITIONS
from llm_pruning.models import load_model
from llm_pruning.mask_uncertainty import analyze_uncertainty, validate_uncertainty_settings
from llm_pruning.record_selection import prepare_records
from llm_pruning.run_artifacts import RunArtifacts, file_fingerprint


def preflight(config, discovery_run, *, top_k=1, granularity="attention_head", token_region="answer", repeats=3, strategies=("random", "layer_cost_matched"), uncertainty_settings=None):
    validate_uncertainty_settings(**(uncertainty_settings or {}))
    if type(repeats) is not int or repeats < 1 or not strategies or len(set(strategies)) != len(strategies) or set(strategies) - CONTROL_STRATEGIES:
        raise ValueError("Use positive repeats and unique supported control strategies")
    grouped = prepare_records(config)
    declared_family = (uncertainty_settings or {}).get("comparison_count")
    if declared_family is not None and declared_family < len(grouped) * len(strategies) * 3:
        raise ValueError("comparison_count cannot be smaller than reported comparisons")
    records = [record for items in grouped.values() for record in items]
    selection = load_discovery_selection(discovery_run, config, records, top_k=top_k, granularity=granularity, token_region=token_region)
    return grouped, selection


def write_rows(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_causal(config, discovery_run, results_root=REPO_ROOT / "results", *, top_k=1, granularity="attention_head", token_region="answer", repeats=3, strategies=("random", "layer_cost_matched"), uncertainty_settings=None):
    uncertainty_settings = {"seed": config.seed, **(uncertainty_settings or {})}
    output = results_root / config.experiment_name
    if output.exists():
        raise FileExistsError(f"Result directory already exists: {output}. Choose a new experiment_name.")
    grouped, selection = preflight(config, discovery_run, top_k=top_k, granularity=granularity, token_region=token_region, repeats=repeats, strategies=strategies, uncertainty_settings=uncertainty_settings)
    records = [record for items in grouped.values() for record in items]
    resolved = {**config.to_dict(), "discovery_run": str(discovery_run.resolve()), "top_k": top_k,
                "granularity": granularity, "token_region": token_region, "control_repeats": repeats, "control_strategies": list(strategies), "uncertainty": uncertainty_settings}
    with RunArtifacts(
        output, REPO_ROOT, resolved, run_kind="held_out_causal_evaluation",
        dataset=file_fingerprint(config.dataset_path), model={"identifier": config.model_name, "requested_revision": config.revision},
        intervention_method="zero_component_projection_inputs", compression_policy=None, pruner=None,
        measurement_definitions=DAMAGE_DEFINITIONS, discovery_files=selection["source_files"],
        evaluation_metric_definitions=MC_METRIC_DEFINITIONS,
    ) as run:
        run.write_json("selection.json", {key: value for key, value in selection.items() if key != "inventory"})
        run.write_json("evaluation_records.json", [{key: getattr(record, key) for key in (
            "id", "split", "domain", "language", "capability", "task", "source_name", "source_type", "metadata",
        )} for record in records])
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        loaded = load_model(config.model_name, device=config.device, dtype=config.dtype, revision=config.revision, local_files_only=config.local_files_only)
        registry = ComponentRegistry(loaded.model, mlp_group_size=selection["inventory"]["mlp_group_size"])
        if registry.to_dict() != selection["inventory"]:
            raise ValueError("Loaded component inventory differs from discovery")
        protocol = prompt_policy_metadata(loaded.tokenizer, config.prompt_policy, config.scoring_mode)
        if protocol != selection["prompt_protocol"]:
            raise ValueError("Loaded tokenizer prompt protocol differs from discovery")
        resolved_revision = getattr(loaded.model.config, "_commit_hash", None)
        source_revision = selection["source_model"].get("resolved_revision")
        if source_revision is not None and source_revision != resolved_revision:
            raise ValueError("Resolved model revision differs from discovery")
        if selection["source_model"].get("dtype") != str(loaded.model.dtype):
            raise ValueError("Loaded model dtype differs from discovery")
        run.manifest["model"].update(resolved_revision=resolved_revision, device=str(loaded.device), dtype=str(loaded.model.dtype),
                                     revision_verified=source_revision is not None, model_type=registry.model_type,
                                     attention_implementation=getattr(loaded.model.config, "_attn_implementation", None))
        run.manifest["prompt_protocol"] = protocol
        run.write_json("component_inventory.json", registry.to_dict())
        conditions, sampling = build_conditions(registry, selection["selected"], seed=config.seed, repeats=repeats, strategies=strategies, return_sampling=True)
        run.write_json("conditions.json", {"components": conditions, "budgets": condition_budgets(registry, conditions), "sampling": sampling})
        run.manifest["control_sampling"] = sampling
        run.write_json("manifest.json", run.manifest)
        scores, decisions, paired, metrics, contrasts = evaluate_conditions(loaded, registry, grouped, config, conditions)
        write_rows(output / "candidate_scores.csv", [asdict(row) for row in scores])
        write_rows(output / "decisions.csv", [asdict(row) for row in decisions])
        write_rows(output / "paired_deltas.csv", paired)
        run.write_json("metrics.json", metrics)
        run.write_json("control_contrasts.json", contrasts)
        run.write_json("uncertainty.json", analyze_uncertainty(paired, conditions, sampling, **uncertainty_settings))
        target_result = next(row for row in metrics if row["condition"] == "selected" and row["is_target"])
        run.complete([
            "# Held-out component mask evaluation", "", "Status: complete.",
            f"Target: {config.target.field}={config.target.value}; evaluation split: {config.split}; seed: {config.seed}.",
            f"Discovery: {len(selection['discovery_ids'])} examples; {token_region}; {granularity}; top_k={top_k}.",
            f"Evaluated {len(records)} examples across {len(grouped)} values and {len(conditions)} conditions.",
            f"Target damage: accuracy={target_result['accuracy_damage']:.4f}; loss increase={target_result['loss_increase']:.4f}; margin={target_result['margin_damage']:.4f}.",
            f"Prompt policy: {config.prompt_policy}; scoring: {config.scoring_mode}.",
            "Per-value results: metrics.json; control contrasts: control_contrasts.json; example changes: paired_deltas.csv.",
            "Selected IDs/rankings/provenance: selection.json; all masks/cost proxies: conditions.json.",
            "Controls exclude selected components; matching cost is a projection-size proxy, not runtime savings.",
            "Approximate paired intervals/precision planning: uncertainty.json; no guaranteed power or scientific result from tiny fixtures.",
            "Use validation for protocol development; test only after freezing choices. Pin model revisions for research.",
        ], condition_count=len(conditions), unique_forward_masks=len({tuple(sorted(values)) for values in conditions.values()}), evaluation_record_count=len(records))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--discovery-run", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--granularity", choices=["attention_head", "mlp_group"], default="attention_head")
    parser.add_argument("--token-region", choices=["prompt", "answer"], default="answer")
    parser.add_argument("--control-repeats", type=int, default=3, help="Requested unique masks per strategy, capped at its finite pool. Default 3 is a technical pilot, not a powered design.")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--multiplicity", choices=["none", "bonferroni"], default="bonferroni")
    parser.add_argument("--comparison-count", type=int, help="Planned comparison family size, at least the reported target x strategy x metric count.")
    parser.add_argument("--desired-control-half-width", type=float, help="Validation-pilot precision target for control-mask mean; suggests a future mask budget, never stops on significance.")
    parser.add_argument("--planning-metric", choices=["accuracy_damage", "loss_increase", "margin_damage"], default="loss_increase")
    parser.add_argument("--control-strategies", nargs="+", choices=sorted(CONTROL_STRATEGIES), default=["random", "layer_cost_matched"])
    parser.add_argument("--validate-only", action="store_true", help="Check data/discovery provenance without loading a model; runtime layout/template/control pools checked later.")
    args = parser.parse_args()
    config = BaselineConfig.load(args.config)
    settings = {"top_k": args.top_k, "granularity": args.granularity, "token_region": args.token_region,
                "repeats": args.control_repeats, "strategies": tuple(args.control_strategies),
                "uncertainty_settings": {"samples": args.bootstrap_samples, "confidence": args.confidence, "seed": config.seed,
                                         "multiplicity": args.multiplicity, "comparison_count": args.comparison_count,
                                         "desired_half_width": args.desired_control_half_width, "planning_metric": args.planning_metric}}
    if args.validate_only:
        grouped, selection = preflight(config, args.discovery_run, **settings)
        print(f"Valid: {sum(map(len, grouped.values()))} evaluation examples; {len(selection['selected'])} discovery-selected components; no model loaded.")
        return
    output = run_causal(config, args.discovery_run, args.results_root, **settings)
    print((output / "summary.md").read_text(encoding="utf-8"), end="")
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
