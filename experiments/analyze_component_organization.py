"""Offline organization diagnostics from discovery activations or singleton probes."""
import argparse
import csv
from itertools import combinations
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from llm_pruning.baseline_config import BaselineConfig
from llm_pruning.organization_diagnostics import measurement_vectors, stability, summarize_organization
from llm_pruning.run_artifacts import RunArtifacts, file_fingerprint
from llm_pruning.targets import TargetSpec


def load_measurements(source, *, token_region="answer", metric="loss_increase"):
    source = source.resolve()
    files = {name: source / name for name in ("manifest.json", "component_inventory.json")}
    manifest = json.loads(files["manifest.json"].read_text(encoding="utf-8"))
    inventory = json.loads(files["component_inventory.json"].read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest["config"].get("split") != "discovery":
        raise ValueError("Organization diagnostics require a completed discovery run")
    field = manifest["config"]["target"]["field"]
    TargetSpec(field, manifest["config"]["target"]["value"])
    if not manifest.get("prompt_protocol") or not manifest.get("model") or not manifest.get("dataset", {}).get("sha256"):
        raise ValueError("Source must record prompt, model and dataset provenance")
    kind = manifest.get("run_kind")
    conditions = None
    if kind == "component_activations":
        if token_region not in manifest["config"].get("token_regions", []):
            raise ValueError("Requested token region was not collected")
        records_name, rows_name, kind = "collection_records.json", "activations.csv", "activation"
    elif kind == "single_component_discovery_probes":
        records_name, rows_name, kind = "evaluation_records.json", "paired_deltas.csv", "causal"
        files["conditions.json"] = source / "conditions.json"
        conditions = json.loads(files["conditions.json"].read_text(encoding="utf-8"))["components"]
    else:
        raise ValueError("Use discovery activations or single-component discovery probes")
    files[records_name], files[rows_name] = source / records_name, source / rows_name
    records = json.loads(files[records_name].read_text(encoding="utf-8"))
    with files[rows_name].open(encoding="utf-8", newline="") as handle:
        vectors = measurement_vectors(csv.DictReader(handle), records, [c["id"] for c in inventory["components"]], field,
                                      kind=kind, token_region=token_region, metric=metric, conditions=conditions)
    return {"path": str(source), "manifest": manifest, "inventory": inventory, "vectors": vectors,
            "source_files": {name: file_fingerprint(path) for name, path in files.items()}}


def run_analysis(sources, experiment_name, results_root=REPO_ROOT / "results", *, top_k=5, token_region="answer", metric="loss_increase", seed=0):
    # Reuse the existing path/seed validation without loading datasets or models.
    BaselineConfig(experiment_name, "offline-analysis", Path("unused"), TargetSpec("domain", "unused"), seed=seed)
    if type(top_k) is not int or top_k < 1 or not sources or len({path.resolve() for path in sources}) != len(sources):
        raise ValueError("Use positive top_k and unique source-run directories")
    output = results_root / experiment_name
    if output.exists():
        raise FileExistsError(f"Result directory already exists: {output}")
    loaded = [load_measurements(source, token_region=token_region, metric=metric) for source in sources]
    comparisons = []
    for left, right in combinations(loaded, 2):
        same_kind = left["manifest"]["run_kind"] == right["manifest"]["run_kind"]
        if not same_kind:
            continue  # Activation and causal vectors are different measurements.
        model_keys = ("identifier", "requested_revision", "resolved_revision", "dtype", "model_type", "attention_implementation")
        if left["inventory"] != right["inventory"] or left["manifest"]["prompt_protocol"] != right["manifest"]["prompt_protocol"] or any(
                left["manifest"]["model"].get(k) != right["manifest"]["model"].get(k) for k in model_keys):
            raise ValueError("Stability needs the same model revision, dtype, inventory and prompt protocol")
        warnings = []
        if not left["manifest"]["model"].get("resolved_revision"):
            warnings.append("resolved_model_revision_unverified")
        data_key = "activations.csv" if same_kind and left["manifest"]["run_kind"] == "component_activations" else "paired_deltas.csv"
        if left["source_files"][data_key]["sha256"] == right["source_files"][data_key]["sha256"]:
            warnings.append("identical_measurements_are_not_independent_replicates")
        comparisons.append({"left_source": left["path"], "right_source": right["path"], "warnings": warnings,
                            "profiles": stability(left["vectors"], right["vectors"], top_k=top_k)})
    resolved = {"experiment_name": experiment_name, "seed": seed, "sources": [str(p.resolve()) for p in sources],
                "top_k": top_k, "token_region": token_region, "causal_metric": metric}
    with RunArtifacts(output, REPO_ROOT, resolved, run_kind="component_organization_analysis", model_inference=False,
                      source_files=[s["source_files"] for s in loaded], models=[s["manifest"]["model"] for s in loaded]) as run:
        run.write_json("component_scores.json", [{"source": s["path"], "vectors": s["vectors"]} for s in loaded])
        reports = [{"source": s["path"], **summarize_organization(s["vectors"], top_k=top_k)} for s in loaded]
        run.write_json("diagnostics.json", {"sources": reports, "stability": comparisons})
        run.complete([
            "# Component organization diagnostics", "", "Status: complete; offline analysis, no model inference.",
            f"Discovery sources: {len(loaded)}; positive top-k: {top_k}; stability comparisons: {len(comparisons)}.",
            "Primitive signed scores/target-vs-other means: component_scores.json.",
            "Concentration, entropy/effective count, Gini, overlap and stability: diagnostics.json.",
            "Activation selectivity and single-component causal damage are reported separately.",
            "Subset probes report measured/inventory counts; concentration over a subset cannot describe the full model.",
            "Stability requires multiple matching discovery runs; a single run does not establish robustness.",
            "Positive-only summaries are descriptive; no modularity, significance or compressibility claim follows by itself.",
        ])
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--results-root", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--token-region", choices=["prompt", "answer"], default="answer")
    parser.add_argument("--causal-metric", choices=["loss_increase", "accuracy_damage", "margin_damage"], default="loss_increase")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    output = run_analysis(args.source_runs, args.experiment_name, args.results_root, top_k=args.top_k,
                          token_region=args.token_region, metric=args.causal_metric, seed=args.seed)
    print((output / "summary.md").read_text(encoding="utf-8"), end="")
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
