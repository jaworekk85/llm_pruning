"""Target-calibrated Wanda with held-out MC frontiers; no localization rankings."""
import argparse
import csv
from dataclasses import replace
from pathlib import Path
import random
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch

from llm_pruning.baseline_config import BaselineConfig
from llm_pruning.compression_frontier import DEFAULT_BUDGETS, evaluate_frontier, validate_frontier_settings
from llm_pruning.mc_evaluation import MC_METRIC_DEFINITIONS
from llm_pruning.mc_prompts import prompt_policy_metadata
from llm_pruning.models import load_model
from llm_pruning.record_selection import prepare_records
from llm_pruning.run_artifacts import RunArtifacts, file_fingerprint


def preflight(config, *, budgets=DEFAULT_BUDGETS, calibration_records=128, core_values=(), max_accuracy_drop=.02, core_min_accuracy=None):
    validate_frontier_settings(budgets, calibration_records, core_values, max_accuracy_drop, core_min_accuracy)
    if config.split not in {"validation", "test"}:
        raise ValueError("Frontier evaluation requires validation or test")
    grouped = prepare_records(config)
    if set(core_values) - set(grouped):
        raise ValueError("All core_values must be explicit labels present in evaluation")
    discovery = prepare_records(replace(config, split="discovery", max_records_per_value=calibration_records))
    calibration = discovery[config.target.value]
    return grouped, calibration


def run_frontier(config, results_root=REPO_ROOT / "results", **settings):
    output = results_root / config.experiment_name
    if output.exists():
        raise FileExistsError(f"Result directory already exists: {output}")
    resolved_settings = {"budgets": DEFAULT_BUDGETS, "calibration_records": 128, "core_values": (),
                         "max_accuracy_drop": .02, "core_min_accuracy": None, **settings}
    grouped, calibration = preflight(config, **resolved_settings)
    records = [r for items in grouped.values() for r in items]
    with RunArtifacts(output, REPO_ROOT, {**config.to_dict(), **resolved_settings}, run_kind="compression_frontier",
                      dataset=file_fingerprint(config.dataset_path),
                      model={"identifier": config.model_name, "requested_revision": config.revision},
                      pruner="wanda_sequential_v1", intervention_method="zero_linear_weights",
                      compression_policy="target_prompt_calibration_unstructured_rowwise",
                      evaluation_metric_definitions=MC_METRIC_DEFINITIONS,
                      cost_definitions={"budget": "Retained fraction of eligible decoder projection weights; row rounding applies.",
                                        "memory": "Actual dense tensor storage bytes, not hypothetical sparse storage; excludes temporary tensors/CPU backups.",
                                        "cuda_peak": "Evaluation-only PyTorch allocated peak; not total process memory or pruning/calibration peak.",
                                        "quality": "Equal example weights within each value or explicit aggregate panel.",
                                        "core": "Explicit labels on the configured target field; null when undeclared."}) as run:
        for filename, items in (("calibration_records.json", calibration), ("evaluation_records.json", records)):
            run.write_json(filename, [{key: getattr(r, key) for key in ("id", "split", "domain", "language", "capability", "task", "source_name", "source_type", "metadata")} for r in items])
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        loaded = load_model(config.model_name, device=config.device, dtype=config.dtype, revision=config.revision, local_files_only=config.local_files_only)
        run.manifest["model"].update(resolved_revision=getattr(loaded.model.config, "_commit_hash", None),
                                     device=str(loaded.device), dtype=str(loaded.model.dtype), model_type=loaded.model.config.model_type,
                                     attention_implementation=getattr(loaded.model.config, "_attn_implementation", None))
        run.manifest["prompt_protocol"] = prompt_policy_metadata(loaded.tokenizer, config.prompt_policy, config.scoring_mode)
        run.write_json("manifest.json", run.manifest)

        def save_point(budget, scores, decisions, paired):
            for filename, rows in (("candidate_scores.csv", scores), ("decisions.csv", decisions), ("paired_deltas.csv", paired)):
                path = output / filename
                existed = path.exists()
                enriched = [{"retained_fraction": budget, **row} for row in rows]
                with path.open("a", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(enriched[0]))
                    if not existed:
                        writer.writeheader()
                    writer.writerows(enriched)

        result = evaluate_frontier(loaded, config, grouped, calibration,
                                   **{key: value for key, value in resolved_settings.items() if key != "calibration_records"}, on_point=save_point)
        with (output / "frontier.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(result["rows"][0]))
            writer.writeheader()
            writer.writerows(result["rows"])
        run.write_json("pruning_reports.json", result["pruning_reports"])
        run.write_json("feasibility.json", result["feasibility"])
        run.complete([
            "# Specialist pruning frontier", "", "Status: complete.",
            f"Target: {config.target.field}={config.target.value}; split: {config.split}; seed: {config.seed}.",
            f"Calibration: {len(calibration)} target-only discovery prompts, without correct answers.",
            f"Evaluation: {len(records)} examples; budgets: {sorted(resolved_settings['budgets'], reverse=True)}.",
            f"Core labels: {list(resolved_settings['core_values']) or 'not declared'}.",
            "Method: sequential-block Wanda; every budget starts from the dense model.",
            "Quality/cost curves: frontier.csv; paired examples: paired_deltas.csv; zero-pattern hashes: pruning_reports.json.",
            "Dense parameter/storage size remains unchanged; nonzeros are a separate sparsity measure.",
            "Feasibility is descriptive at fixed budgets; one pruner does not establish optimal compressibility.",
            "No latency, energy or sparse-kernel speedup was measured.",
        ], evaluation_record_count=len(records), calibration_record_count=len(calibration),
           frontier_point_count=len(resolved_settings["budgets"]), feasibility=result["feasibility"])
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--budgets", nargs="+", type=float, default=list(DEFAULT_BUDGETS), help="Retained projection-weight fractions; include 1.0.")
    parser.add_argument("--calibration-records", type=int, default=128)
    parser.add_argument("--core-values", nargs="*", default=[], help="Explicit core labels on the configured target field.")
    parser.add_argument("--max-accuracy-drop", type=float, default=.02)
    parser.add_argument("--core-min-accuracy", type=float)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    config = BaselineConfig.load(args.config)
    settings = {"budgets": tuple(args.budgets), "calibration_records": args.calibration_records,
                "core_values": tuple(args.core_values), "max_accuracy_drop": args.max_accuracy_drop, "core_min_accuracy": args.core_min_accuracy}
    if args.validate_only:
        grouped, calibration = preflight(config, **settings)
        print(f"Valid: {len(calibration)} calibration prompts; {sum(map(len, grouped.values()))} held-out examples; no model loaded.")
        return
    output = run_frontier(config, args.results_root, **settings)
    print((output / "summary.md").read_text(encoding="utf-8"), end="")
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
