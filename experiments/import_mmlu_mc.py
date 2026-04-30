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


DOMAIN_SUBJECTS = {
    "astronomy": [
        "astronomy",
    ],
    "math": [
        "abstract_algebra",
        "college_mathematics",
        "elementary_mathematics",
        "high_school_mathematics",
        "high_school_statistics",
    ],
    "medicine": [
        "anatomy",
        "clinical_knowledge",
        "college_medicine",
        "human_aging",
        "medical_genetics",
        "nutrition",
        "professional_medicine",
        "virology",
    ],
    "politics": [
        "high_school_government_and_politics",
        "international_law",
        "jurisprudence",
        "professional_law",
        "security_studies",
        "us_foreign_policy",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import MMLU subjects into the local multiple-choice JSONL schema."
    )
    parser.add_argument("--output", type=Path, default=Path("data/prompt_sets/mmlu_mc.jsonl"))
    parser.add_argument("--dataset-name", default="cais/mmlu")
    parser.add_argument("--domains", nargs="*", default=sorted(DOMAIN_SUBJECTS))
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument(
        "--discovery-fraction",
        type=float,
        default=0.6,
        help="Fraction of each subject assigned to local discovery.",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.2,
        help="Fraction of each subject assigned to local validation. The rest goes to local test.",
    )
    parser.add_argument(
        "--max-records-per-subject-split",
        type=int,
        default=None,
        help="Optional cap after local split assignment, useful for smoke imports.",
    )
    return parser.parse_args()


def load_dataset_split(dataset_name: str, subject: str, split: str):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "The 'datasets' package is required. Install it with: "
            "pip install -r requirements-data.txt"
        ) from exc

    return load_dataset(dataset_name, subject, split=split)


def answer_index(value: Any) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            return int(stripped)
        if len(stripped) == 1 and "A" <= stripped.upper() <= "Z":
            return ord(stripped.upper()) - ord("A")
    raise ValueError(f"Unsupported MMLU answer value: {value!r}")


def record_from_row(
    *,
    domain: str,
    subject: str,
    split: str,
    index: int,
    row: dict[str, Any],
    dataset_name: str,
) -> ChoiceRecord:
    return ChoiceRecord.from_dict(
        {
            "id": f"mmlu.{subject}.{split}.{index:05d}",
            "domain": domain,
            "source_name": dataset_name,
            "source_type": "benchmark_mmlu",
            "subject": subject,
            "split": split,
            "question": str(row["question"]).strip(),
            "choices": [str(choice).strip() for choice in row["choices"]],
            "answer_index": answer_index(row["answer"]),
            "license": "mit",
            "notes": "Imported from MMLU and assigned to deterministic local splits.",
        }
    )


def assign_local_splits(
    rows: list[dict[str, Any]],
    discovery_fraction: float,
    validation_fraction: float,
    rng: random.Random,
) -> list[tuple[str, dict[str, Any]]]:
    if discovery_fraction <= 0.0 or validation_fraction <= 0.0:
        raise ValueError("--discovery-fraction and --validation-fraction must be positive.")
    if discovery_fraction + validation_fraction >= 1.0:
        raise ValueError("Discovery plus validation fractions must be below 1.")

    shuffled = list(rows)
    rng.shuffle(shuffled)
    discovery_count = max(1, round(len(shuffled) * discovery_fraction))
    validation_count = max(1, round(len(shuffled) * validation_fraction))
    if discovery_count + validation_count >= len(shuffled):
        validation_count = max(1, len(shuffled) - discovery_count - 1)

    split_by_row_id: dict[int, str] = {}
    for row in shuffled[:discovery_count]:
        split_by_row_id[id(row)] = "discovery"
    for row in shuffled[discovery_count : discovery_count + validation_count]:
        split_by_row_id[id(row)] = "validation"
    for row in shuffled[discovery_count + validation_count :]:
        split_by_row_id[id(row)] = "test"

    return [(split_by_row_id[id(row)], row) for row in rows]


def cap_records(
    records: list[ChoiceRecord],
    max_records_per_subject_split: int | None,
) -> list[ChoiceRecord]:
    if max_records_per_subject_split is None:
        return records

    counts: Counter[tuple[str, str]] = Counter()
    capped: list[ChoiceRecord] = []
    for record in records:
        key = (record.subject, record.split)
        if counts[key] >= max_records_per_subject_split:
            continue
        capped.append(record)
        counts[key] += 1
    return capped


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


def build_records(args: argparse.Namespace) -> list[ChoiceRecord]:
    rng = random.Random(args.random_seed)
    records: list[ChoiceRecord] = []

    for domain in args.domains:
        subjects = DOMAIN_SUBJECTS.get(domain)
        if subjects is None:
            allowed = ", ".join(sorted(DOMAIN_SUBJECTS))
            raise ValueError(f"Unknown domain '{domain}'. Use one of: {allowed}.")

        for subject in subjects:
            print(f"Loading {args.dataset_name}/{subject}")
            public_rows: list[dict[str, Any]] = []
            for public_split in ["dev", "validation", "test"]:
                public_rows.extend(load_dataset_split(args.dataset_name, subject, public_split))

            for local_split, row in assign_local_splits(
                public_rows,
                args.discovery_fraction,
                args.validation_fraction,
                rng,
            ):
                records.append(
                    record_from_row(
                        domain=domain,
                        subject=subject,
                        split=local_split,
                        index=len(records),
                        row=row,
                        dataset_name=args.dataset_name,
                    )
                )

    return cap_records(clean_records(records), args.max_records_per_subject_split)


def write_records(records: list[ChoiceRecord], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record.__dict__, ensure_ascii=True, sort_keys=True))
            file.write("\n")


def print_summary(records: list[ChoiceRecord], output: Path) -> None:
    by_domain = Counter(record.domain for record in records)
    by_split = Counter(record.split for record in records)
    by_subject = Counter(record.subject for record in records)

    print(f"Wrote {len(records)} MMLU choice records to {output}")
    print(f"Domains: {dict(sorted(by_domain.items()))}")
    print(f"Splits: {dict(sorted(by_split.items()))}")
    print("Subject counts:")
    for subject, count in sorted(by_subject.items()):
        print(f"  {subject}: {count}")

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
