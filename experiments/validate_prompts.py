from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from llm_pruning.prompt_records import load_prompt_records, validate_prompt_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a JSONL prompt dataset.")
    parser.add_argument("--prompt-jsonl", type=Path, default=Path("data/prompt_sets/seed.jsonl"))
    parser.add_argument(
        "--require-targets",
        action="store_true",
        help="Require every record to include a target answer.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_prompt_records(args.prompt_jsonl)
    errors = validate_prompt_records(records, require_targets=args.require_targets)

    by_domain = Counter(record.domain for record in records)
    by_split = Counter(record.split for record in records)
    by_prompt_type = Counter(record.prompt_type for record in records)
    by_difficulty = Counter(record.difficulty for record in records)
    by_source = Counter(record.source_type for record in records)
    target_count = sum(1 for record in records if record.target)

    print(f"Loaded {len(records)} prompt records from {args.prompt_jsonl}")
    print(f"Domains: {dict(sorted(by_domain.items()))}")
    print(f"Splits: {dict(sorted(by_split.items()))}")
    print(f"Prompt types: {dict(sorted(by_prompt_type.items()))}")
    print(f"Difficulty: {dict(sorted(by_difficulty.items()))}")
    print(f"Source types: {dict(sorted(by_source.items()))}")
    print(f"Records with target answers: {target_count}/{len(records)}")
    print("Domain x split:")
    for domain in sorted(by_domain):
        split_counts = Counter(
            record.split for record in records if record.domain == domain
        )
        print(f"  {domain}: {dict(sorted(split_counts.items()))}")

    if errors:
        print("\nValidation errors:")
        for error in errors:
            print(f"  - {error}")
        raise SystemExit(1)

    print("\nPrompt dataset is valid.")


if __name__ == "__main__":
    main()
