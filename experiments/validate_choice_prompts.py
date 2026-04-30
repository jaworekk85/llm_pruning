from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from llm_pruning.choice_records import load_choice_records, validate_choice_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a multiple-choice JSONL dataset.")
    parser.add_argument("--choice-jsonl", type=Path, default=Path("data/prompt_sets/mmlu_mc.jsonl"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_choice_records(args.choice_jsonl)
    errors = validate_choice_records(records)

    by_domain = Counter(record.domain for record in records)
    by_split = Counter(record.split for record in records)
    by_subject = Counter(record.subject for record in records)

    print(f"Loaded {len(records)} choice records from {args.choice_jsonl}")
    print(f"Domains: {dict(sorted(by_domain.items()))}")
    print(f"Splits: {dict(sorted(by_split.items()))}")
    print("Subjects:")
    for subject, count in sorted(by_subject.items()):
        print(f"  {subject}: {count}")

    if errors:
        print("\nValidation errors:")
        for error in errors:
            print(f"  - {error}")
        raise SystemExit(1)

    print("\nChoice dataset is valid.")


if __name__ == "__main__":
    main()
