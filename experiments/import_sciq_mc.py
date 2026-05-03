from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from llm_pruning.choice_records import ChoiceRecord, validate_choice_records


PUBLIC_TO_LOCAL_SPLIT = {
    "train": "discovery",
    "validation": "validation",
    "test": "test",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import SciQ into the local multiple-choice JSONL schema."
    )
    parser.add_argument("--output", type=Path, default=Path("data/prompt_sets/sciq_mc.jsonl"))
    parser.add_argument("--dataset-name", default="allenai/sciq")
    parser.add_argument("--domain", default="science")
    parser.add_argument("--subject", default="sciq")
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument(
        "--max-records-per-split",
        type=int,
        default=None,
        help="Optional cap per local split, useful for smoke imports.",
    )
    return parser.parse_args()


def load_dataset_split(dataset_name: str, split: str):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "The 'datasets' package is required. Install it with: "
            "pip install -r requirements-data.txt"
        ) from exc

    return load_dataset(dataset_name, split=split)


def shuffled_choices(row: dict[str, Any], rng: random.Random) -> tuple[list[str], int]:
    correct_answer = str(row["correct_answer"]).strip()
    choices = [
        correct_answer,
        str(row["distractor1"]).strip(),
        str(row["distractor2"]).strip(),
        str(row["distractor3"]).strip(),
    ]
    indexed_choices = list(enumerate(choices))
    rng.shuffle(indexed_choices)
    shuffled = [choice for _old_index, choice in indexed_choices]
    answer_index = next(
        new_index
        for new_index, (old_index, _choice) in enumerate(indexed_choices)
        if old_index == 0
    )
    return shuffled, answer_index


def record_from_row(
    *,
    row: dict[str, Any],
    index: int,
    local_split: str,
    args: argparse.Namespace,
    rng: random.Random,
) -> ChoiceRecord:
    choices, answer_index = shuffled_choices(row, rng)
    support = str(row.get("support", "")).strip()
    notes = "Imported from SciQ official split."
    if support:
        notes = f"{notes} Support: {support}"

    return ChoiceRecord.from_dict(
        {
            "id": f"sciq.{local_split}.{index:05d}",
            "domain": args.domain,
            "source_name": args.dataset_name,
            "source_type": "benchmark_sciq",
            "subject": args.subject,
            "split": local_split,
            "question": str(row["question"]).strip(),
            "choices": choices,
            "answer_index": answer_index,
            "license": "unknown",
            "notes": notes,
        }
    )


def clean_records(records: list[ChoiceRecord]) -> list[ChoiceRecord]:
    cleaned: list[ChoiceRecord] = []
    seen_questions: set[tuple[str, str]] = set()
    dropped_duplicate_questions = 0
    dropped_bad_choices = 0

    for record in records:
        stripped_choices = [choice.strip() for choice in record.choices]
        if (
            any(not choice for choice in stripped_choices)
            or len(set(choice.lower() for choice in stripped_choices)) != len(stripped_choices)
        ):
            dropped_bad_choices += 1
            continue

        question_key = (record.domain, " ".join(record.question.lower().split()))
        if question_key in seen_questions:
            dropped_duplicate_questions += 1
            continue

        seen_questions.add(question_key)
        cleaned.append(record)

    if dropped_duplicate_questions or dropped_bad_choices:
        print(
            "Dropped records during import: "
            f"duplicate_questions={dropped_duplicate_questions}, "
            f"bad_choices={dropped_bad_choices}"
        )

    return cleaned


def cap_records(
    records: list[ChoiceRecord],
    max_records_per_split: int | None,
) -> list[ChoiceRecord]:
    if max_records_per_split is None:
        return records

    counts: Counter[str] = Counter()
    capped: list[ChoiceRecord] = []
    for record in records:
        if counts[record.split] >= max_records_per_split:
            continue
        capped.append(record)
        counts[record.split] += 1
    return capped


def build_records(args: argparse.Namespace) -> list[ChoiceRecord]:
    rng = random.Random(args.random_seed)
    records: list[ChoiceRecord] = []
    for public_split, local_split in PUBLIC_TO_LOCAL_SPLIT.items():
        print(f"Loading {args.dataset_name}/{public_split}")
        for row in load_dataset_split(args.dataset_name, public_split):
            records.append(
                record_from_row(
                    row=row,
                    index=len(records),
                    local_split=local_split,
                    args=args,
                    rng=rng,
                )
            )
    return cap_records(clean_records(records), args.max_records_per_split)


def write_records(records: list[ChoiceRecord], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record.__dict__, ensure_ascii=True, sort_keys=True))
            file.write("\n")


def print_summary(records: list[ChoiceRecord], output: Path) -> None:
    by_split = Counter(record.split for record in records)
    print(f"Wrote {len(records)} SciQ choice records to {output}")
    print(f"Splits: {dict(sorted(by_split.items()))}")

    errors = validate_choice_records(records)
    if errors:
        print("\nValidation errors:")
        for error in errors:
            print(f"  - {error}")
        raise SystemExit(1)


def main() -> None:
    args = parse_args()
    records = build_records(args)
    write_records(records, args.output)
    print_summary(records, args.output)


if __name__ == "__main__":
    main()
