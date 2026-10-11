"""Analyze saved paired mask results and plan control precision without model inference."""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for import_path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from llm_pruning.mask_uncertainty import METRICS, analyze_uncertainty, validate_uncertainty_settings
from llm_pruning.run_artifacts import RunArtifacts, file_fingerprint


def run_analysis(source, experiment_name, results_root=REPO_ROOT / "results", **settings):
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_-]{0,99}", experiment_name):
        raise ValueError("Use a simple experiment_name")
    validate_uncertainty_settings(**settings)
    output = results_root / experiment_name
    if output.exists():
        raise FileExistsError(f"Result directory already exists: {output}")
    files = {name: source / name for name in ("manifest.json", "conditions.json", "paired_deltas.csv")}
    original = json.loads(files["manifest.json"].read_text(encoding="utf-8"))
    if original.get("status") != "complete" or original.get("run_kind") != "held_out_causal_evaluation":
        raise ValueError("Require a completed held-out causal evaluation")
    with RunArtifacts(
        output, REPO_ROOT, {"experiment_name": experiment_name, "source_run": str(source.resolve()), "seed": settings.get("seed", 0), **settings},
        run_kind="mask_uncertainty_analysis", source_files={name: file_fingerprint(path) for name, path in files.items()},
        model=original["model"], model_inference=False,
    ) as run:
        plan = json.loads(files["conditions.json"].read_text(encoding="utf-8"))
        with files["paired_deltas.csv"].open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        report = analyze_uncertainty(rows, plan["components"], plan.get("sampling"), **settings)
        run.write_json("uncertainty.json", report)
        run.complete([
            "# Mask uncertainty analysis", "", "Status: complete.",
            f"Source: {source.resolve()}", f"Target/strategy panels: {len(report['results'])}; comparisons: {report['comparison_count']}.",
            "Results: uncertainty.json; source files were not modified; no model inference performed.",
            "Intervals are approximate and conditional on discovery selection; small pilots are flagged.",
            "Precision planning uses validation data only and does not guarantee power or significance.",
        ])
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--results-root", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--multiplicity", choices=["none", "bonferroni"], default="bonferroni")
    parser.add_argument("--comparison-count", type=int)
    parser.add_argument("--desired-control-half-width", type=float)
    parser.add_argument("--planning-metric", choices=METRICS, default="loss_increase")
    args = parser.parse_args()
    output = run_analysis(args.source_run, args.experiment_name, args.results_root,
                          samples=args.bootstrap_samples, confidence=args.confidence, seed=args.seed, multiplicity=args.multiplicity,
                          comparison_count=args.comparison_count, desired_half_width=args.desired_control_half_width, planning_metric=args.planning_metric)
    print((output / "summary.md").read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
