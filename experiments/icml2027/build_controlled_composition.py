"""Build one controlled task family locally with paired probes; no model/downloads."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    sys.path.insert(0, str(path))

from experiments.icml2027.controlled_composition import BENCHMARK_ID, FAMILIES, build_controlled_benchmark
from llm_pruning.run_artifacts import RunArtifacts, file_fingerprint


def build_dataset(output_dir, **settings):
    if output_dir.exists():
        raise FileExistsError(f"Dataset directory already exists: {output_dir}")
    records = build_controlled_benchmark(**settings)
    config = {"builder": BENCHMARK_ID, **settings, "seed": settings.get("seed", 2027)}
    with RunArtifacts(output_dir, REPO_ROOT, config, run_kind="controlled_composition_dataset") as run:
        with (output_dir / "records.jsonl").open("x", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(asdict(record), ensure_ascii=False, allow_nan=False) + "\n")
        reference = [r for r in records if r.task == "a" and r.metadata["case_role"] == "reference"]
        audit = {"benchmark_id": BENCHMARK_ID,
            "groups_per_split": dict(Counter(r.split for r in reference)),
            "reference_layouts": dict(Counter(r.metadata["template_id"] for r in reference)),
            "reference_target_positions": dict(Counter(
                next(i for i, (name, _) in enumerate(r.metadata["symbolic_case"]["entities"])
                     if name == r.metadata["symbolic_case"]["target_entity"]) for r in reference)),
            "symbolic_labels_and_probe_contract": "passed",
            "pretrained_feasibility": "not_evaluated"}
        run.write_json("audit.json", audit)
        run.complete(["# Controlled composition benchmark", "", "Status: complete; no model inference.",
            f"Family: {settings.get('family', 'lookup_lookup')}; records: {len(records)}.",
            f"Independent source groups per split: {audit['groups_per_split']}.",
            "Each group: reference plus target/distractor/order/name probes; four static variants each.",
            "A/B/composed/oracle share exact evidence; executable gold and complete probes checked.",
            "Probe copies are correlated; primary readout uses reference cases only.",
            "Context use only; scientific readiness and model competence remain unvalidated."],
            dataset=file_fingerprint(output_dir / "records.jsonl"), **audit)
    return output_dir / "records.jsonl"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--family", choices=FAMILIES, default="lookup_lookup")
    parser.add_argument("--groups-per-split", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--namespace", default="composition_v1")
    parser.add_argument("--items", type=int, default=4)
    parser.add_argument("--fixture", action="store_true")
    args = vars(parser.parse_args())
    output = args.pop("output_dir")
    path = build_dataset(output, **args)
    print((path.parent / "summary.md").read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
