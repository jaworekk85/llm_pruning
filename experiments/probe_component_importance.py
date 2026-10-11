"""Single-component causal discovery probes; no test data or pruning decisions."""
import argparse
import csv
from dataclasses import asdict
from pathlib import Path
import random
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch

from llm_pruning.baseline_config import BaselineConfig
from llm_pruning.causal_evaluation import DAMAGE_DEFINITIONS, evaluate_conditions
from llm_pruning.components import ComponentId, ComponentRegistry
from llm_pruning.mc_evaluation import MC_METRIC_DEFINITIONS
from llm_pruning.mc_prompts import prompt_policy_metadata
from llm_pruning.models import load_model
from llm_pruning.record_selection import prepare_records
from llm_pruning.run_artifacts import RunArtifacts, file_fingerprint


def preflight(config, *, granularity="attention_head", components=None):
    if config.split != "discovery":
        raise ValueError("Organization probes require discovery, never validation/test")
    if granularity not in {"attention_head", "mlp_group"}:
        raise ValueError("Unsupported granularity")
    if components is not None:
        if not components or len(set(components)) != len(components) or any(ComponentId.parse(c).kind != granularity for c in components):
            raise ValueError("Use unique component IDs of the requested granularity")
    return prepare_records(config)


def run_probes(config, results_root=REPO_ROOT / "results", *, granularity="attention_head", components=None, mlp_group_size=128):
    output = results_root / config.experiment_name
    if output.exists():
        raise FileExistsError(f"Result directory already exists: {output}")
    if type(mlp_group_size) is not int or mlp_group_size < 1:
        raise ValueError("mlp_group_size must be positive")
    grouped = preflight(config, granularity=granularity, components=components)
    records = [r for items in grouped.values() for r in items]
    with RunArtifacts(output, REPO_ROOT, {**config.to_dict(), "granularity": granularity, "components": components, "mlp_group_size": mlp_group_size},
                      run_kind="single_component_discovery_probes", dataset=file_fingerprint(config.dataset_path),
                      model={"identifier": config.model_name, "requested_revision": config.revision},
                      intervention_method="zero_component_projection_inputs", pruner=None, compression_policy=None,
                      measurement_definitions=DAMAGE_DEFINITIONS, evaluation_metric_definitions=MC_METRIC_DEFINITIONS) as run:
        run.write_json("evaluation_records.json", [{key: getattr(r, key) for key in ("id", "split", "domain", "language", "capability", "task", "source_name", "source_type", "metadata")} for r in records])
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        loaded = load_model(config.model_name, device=config.device, dtype=config.dtype, revision=config.revision, local_files_only=config.local_files_only)
        registry = ComponentRegistry(loaded.model, mlp_group_size=mlp_group_size)
        selected = components if components is not None else [str(s.component_id) for s in registry.components if s.component_id.kind == granularity]
        for component in selected:
            registry.resolve(component)
        conditions = {"baseline": [], **{f"probe_{index:05d}": [component] for index, component in enumerate(selected)}}
        run.write_json("component_inventory.json", registry.to_dict())
        run.write_json("conditions.json", {"components": conditions})
        run.manifest["model"].update(resolved_revision=getattr(loaded.model.config, "_commit_hash", None), device=str(loaded.device),
                                     dtype=str(loaded.model.dtype), model_type=registry.model_type,
                                     attention_implementation=getattr(loaded.model.config, "_attn_implementation", None))
        run.manifest["prompt_protocol"] = prompt_policy_metadata(loaded.tokenizer, config.prompt_policy, config.scoring_mode)
        run.write_json("manifest.json", run.manifest)
        scores, decisions, paired, metrics, _ = evaluate_conditions(loaded, registry, grouped, config, conditions)
        for filename, rows in (("candidate_scores.csv", [asdict(r) for r in scores]), ("decisions.csv", [asdict(r) for r in decisions]), ("paired_deltas.csv", paired)):
            with (output / filename).open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        run.write_json("metrics.json", metrics)
        run.complete([
            "# Single-component discovery probes", "", "Status: complete.",
            f"Field: {config.target.field}; split: discovery; seed: {config.seed}.",
            f"Examples: {len(records)}; probed components: {len(selected)}; granularity: {granularity}.",
            "Each component is masked separately against the same unmasked baseline.",
            "Primitive paired damage: paired_deltas.csv; IDs and coverage: conditions.json/component_inventory.json.",
            "Single-component damage is conditional on the intact model; combined effects need not be additive.",
            "These discovery probes do not select or control the independent Wanda frontier.",
        ], evaluation_record_count=len(records), condition_count=len(conditions))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--granularity", choices=["attention_head", "mlp_group"], default="attention_head")
    parser.add_argument("--components", nargs="+", help="Explicit subset; omitted means all components at this granularity. Runtime grows with components.")
    parser.add_argument("--mlp-group-size", type=int, default=128)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    config = BaselineConfig.load(args.config)
    if args.mlp_group_size < 1:
        parser.error("mlp-group-size must be positive")
    if args.validate_only:
        grouped = preflight(config, granularity=args.granularity, components=args.components)
        print(f"Valid: {sum(map(len, grouped.values()))} discovery examples; component availability checked at runtime; no model loaded.")
        return
    output = run_probes(config, args.results_root, granularity=args.granularity, components=args.components, mlp_group_size=args.mlp_group_size)
    print((output / "summary.md").read_text(encoding="utf-8"), end="")
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
