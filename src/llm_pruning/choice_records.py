from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = {
    "id",
    "domain",
    "source_name",
    "source_type",
    "subject",
    "split",
    "question",
    "choices",
    "answer_index",
}

ALLOWED_SPLITS = {"discovery", "validation", "test"}


@dataclass(frozen=True)
class ChoiceRecord:
    id: str
    domain: str
    source_name: str
    source_type: str
    subject: str
    split: str
    question: str
    choices: list[str]
    answer_index: int
    license: str = "unknown"
    notes: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChoiceRecord":
        missing = sorted(REQUIRED_FIELDS - set(data))
        if missing:
            raise ValueError(f"Choice record is missing required fields: {missing}")

        split = str(data["split"])
        if split not in ALLOWED_SPLITS:
            allowed = ", ".join(sorted(ALLOWED_SPLITS))
            raise ValueError(f"Invalid split '{split}'. Use one of: {allowed}.")

        choices = data["choices"]
        if not isinstance(choices, list) or len(choices) < 2:
            raise ValueError("Choice record must contain at least two choices.")

        normalized_choices = [str(choice) for choice in choices]
        answer_index = int(data["answer_index"])
        if answer_index < 0 or answer_index >= len(normalized_choices):
            raise ValueError(
                f"answer_index {answer_index} is outside choices length {len(normalized_choices)}."
            )

        return cls(
            id=str(data["id"]),
            domain=str(data["domain"]),
            source_name=str(data["source_name"]),
            source_type=str(data["source_type"]),
            subject=str(data["subject"]),
            split=split,
            question=str(data["question"]),
            choices=normalized_choices,
            answer_index=answer_index,
            license=str(data.get("license", "unknown")),
            notes=str(data.get("notes", "")),
        )


def load_choice_records(path: Path) -> list[ChoiceRecord]:
    records: list[ChoiceRecord] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(ChoiceRecord.from_dict(json.loads(line)))
            except Exception as exc:
                raise ValueError(f"Invalid choice record at {path}:{line_number}: {exc}") from exc
    return records


def filter_choice_records(
    records: list[ChoiceRecord],
    split: str | None = None,
    domains: list[str] | None = None,
    subjects: list[str] | None = None,
) -> list[ChoiceRecord]:
    domain_set = set(domains) if domains else None
    subject_set = set(subjects) if subjects else None
    return [
        record
        for record in records
        if (split is None or record.split == split)
        and (domain_set is None or record.domain in domain_set)
        and (subject_set is None or record.subject in subject_set)
    ]


def validate_choice_records(records: list[ChoiceRecord]) -> list[str]:
    errors: list[str] = []
    seen_ids: set[str] = set()
    seen_questions: dict[tuple[str, str], str] = {}

    for record in records:
        if record.id in seen_ids:
            errors.append(f"Duplicate id: {record.id}")
        seen_ids.add(record.id)

        key = (record.domain, " ".join(record.question.lower().split()))
        if key in seen_questions:
            errors.append(f"Duplicate question in domain: {record.id} duplicates {seen_questions[key]}")
        seen_questions[key] = record.id

        stripped_choices = [choice.strip() for choice in record.choices]
        if any(not choice for choice in stripped_choices):
            errors.append(f"Empty choice text: {record.id}")
        if len(set(choice.lower() for choice in stripped_choices)) != len(stripped_choices):
            errors.append(f"Duplicate choices: {record.id}")

    return errors
