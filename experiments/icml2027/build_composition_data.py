"""Build/import linked cases locally; fresh manifest directory, no model or downloads."""
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

from experiments.icml2027.composition_data import BUILDER_ID, build_linked_data
from llm_pruning.run_artifacts import RunArtifacts, file_fingerprint


def build_dataset(output_dir, *, facts_path=None, episodes_per_split=128, seed=0, fixture=False):
    if output_dir.exists():
        raise FileExistsError(f"Dataset directory already exists: {output_dir}")
    facts = None
    if facts_path is not None:
        with facts_path.open(encoding="utf-8") as handle:
            facts = [json.loads(line) for line in handle if line.strip()]
    records = build_linked_data(facts, episodes_per_split=episodes_per_split, seed=seed, fixture=fixture)
    config = {"seed": seed, "episodes_per_split": episodes_per_split, "fixture": fixture,
              "facts_path": str(facts_path) if facts_path else None, "builder": BUILDER_ID}
    with RunArtifacts(output_dir, REPO_ROOT, config, run_kind="dataset_build",
                      input_source=file_fingerprint(facts_path) if facts_path else None) as run:
        with (output_dir / "records.jsonl").open("x", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(asdict(record), ensure_ascii=False, allow_nan=False) + "\n")
        counts = dict(Counter(r.split for r in records))
        run.complete(["# Linked task dataset", "", "Status: complete.",
                      f"Records: {len(records)}; per split: {counts}",
                      "Four linked variants per episode; source/fact groups never cross splits.",
                      "Context-supplied cases test computation over supplied facts, not factual storage.",
                      "Templates/difficulty/generalization need pretrained pilot validation."],
                     dataset=file_fingerprint(output_dir / "records.jsonl"), counts=counts)
    return output_dir / "records.jsonl"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--facts", type=Path)
    parser.add_argument("--episodes-per-split", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--fixture", action="store_true")
    args = parser.parse_args()
    path = build_dataset(args.output_dir, facts_path=args.facts, episodes_per_split=args.episodes_per_split,
                         seed=args.seed, fixture=args.fixture)
    print((path.parent / "summary.md").read_text(encoding="utf-8"), end="")
    print(f"Dataset: {path}")


if __name__ == "__main__":
    main()
