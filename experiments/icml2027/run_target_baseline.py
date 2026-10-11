"""Config-driven baseline predictor evaluation for domains, languages, capabilities or tasks."""

from __future__ import annotations

import argparse
import csv
import random
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for import_path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import torch

from llm_pruning.baseline_config import BaselineConfig
from llm_pruning.mc_evaluation import MC_METRIC_DEFINITIONS, batched, decisions_from_scores, score_candidate_batch
from llm_pruning.models import load_model
from llm_pruning.mc_prompts import prompt_policy_metadata
from llm_pruning.record_selection import prepare_records
from llm_pruning.run_artifacts import RunArtifacts, file_fingerprint


METRIC_DEFINITIONS = MC_METRIC_DEFINITIONS


def write_rows(path: Path, rows: list) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0])))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def run_baseline(config: BaselineConfig, results_root: Path = REPO_ROOT / "results") -> Path:
    output_dir = results_root / config.experiment_name
    # Fail before dataset loading as well as before any expensive model work.
    if output_dir.exists():
        raise FileExistsError(f"Result directory already exists: {output_dir}. Choose a new experiment_name.")
    dataset = file_fingerprint(config.dataset_path)
    grouped = prepare_records(config)
    selected = [record for records in grouped.values() for record in records]
    with RunArtifacts(
        output_dir, REPO_ROOT, config.to_dict(), dataset=dataset,
        model={"identifier": config.model_name, "requested_revision": config.revision},
        intervention_method="none", compression_policy=None, pruner=None,
        evaluation_metric_definitions=METRIC_DEFINITIONS,
        evaluation={"split": config.split, "counts": {value: len(records) for value, records in grouped.items()}},
    ) as run:
        run.write_json("evaluation_records.json", [
            {"id": record.id, "split": record.split, "field": config.target.field,
             "value": getattr(record, config.target.field), "source_name": record.source_name}
            for record in selected
        ])
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        loaded = load_model(
            config.model_name, device=config.device, dtype=config.dtype,
            revision=config.revision, local_files_only=config.local_files_only,
        )
        run.manifest["prompt_protocol"] = prompt_policy_metadata(loaded.tokenizer, config.prompt_policy, config.scoring_mode)
        run.manifest["model"].update(
            resolved_revision=getattr(loaded.model.config, "_commit_hash", None),
            device=str(loaded.device), dtype=str(loaded.model.dtype),
        )
        run.write_json("manifest.json", run.manifest)
        scores = []
        for records in grouped.values():
            candidates = [(record, index) for record in records for index in range(len(record.choices))]
            for batch in batched(candidates, config.batch_size):
                scores.extend(score_candidate_batch(loaded, batch, "baseline", config.scoring_mode, prompt_policy=config.prompt_policy))
        by_id = {record.id: record for record in selected}
        decisions = decisions_from_scores(scores, by_id)
        decisions_by_id = {decision.record_id: decision for decision in decisions}
        metrics = []
        for value, records in grouped.items():
            group_decisions = [decisions_by_id[record.id] for record in records]
            count = len(group_decisions)
            metrics.append({
                "target_field": config.target.field, "target_value": value,
                "is_target": value == config.target.value, "record_count": count,
                "accuracy": sum(decision.is_correct for decision in group_decisions) / count,
                "correct_loss": sum(decision.correct_loss for decision in group_decisions) / count,
                "margin": sum(decision.margin for decision in group_decisions) / count,
            })
        write_rows(output_dir / "candidate_scores.csv", scores)
        write_rows(output_dir / "decisions.csv", decisions)
        run.write_json("metrics.json", metrics)
        target_metrics = next(row for row in metrics if row["is_target"])
        run.complete([
            "# Target baseline evaluation", "", "Status: complete.",
            f"Target: {config.target.field}={config.target.value}",
            f"Split: {config.split}; seed: {config.seed}; scoring: {config.scoring_mode}",
            f"Evaluated: {len(selected)} examples across {len(grouped)} target-field values.",
            f"Target: n={target_metrics['record_count']}; accuracy={target_metrics['accuracy']:.4f}; correct_loss={target_metrics['correct_loss']:.4f}; margin={target_metrics['margin']:.4f}",
            "Per-value metrics: metrics.json; sampled IDs: evaluation_records.json.",
            "Baseline quality only; this run does not measure causal importance or compressibility.",
            "Split checks detect ID and normalized-input reuse, not semantic overlap.",
        ])
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--validate-only", action="store_true", help="Check config and data without loading a model or creating results.")
    args = parser.parse_args()
    config = BaselineConfig.load(args.config)
    if args.validate_only:
        grouped = prepare_records(config)
        count = sum(len(records) for records in grouped.values())
        print(f"Valid: {config.target.field}={config.target.value}; split={config.split}; {count} records across {len(grouped)} values.")
        return
    output_dir = run_baseline(config, args.results_root)
    print((output_dir / "summary.md").read_text(encoding="utf-8"), end="")
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
