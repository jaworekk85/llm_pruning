from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = {
    "id",
    "domain",
    "prompt",
    "split",
    "source_type",
    "source_name",
    "prompt_type",
    "difficulty",
    "language",
}

ALLOWED_SPLITS = {"discovery", "validation", "test"}


@dataclass(frozen=True)
class PromptRecord:
    id: str
    domain: str
    prompt: str
    split: str
    source_type: str
    source_name: str
    prompt_type: str
    difficulty: str
    language: str
    target: str | None = None
    license: str = "unknown"
    notes: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PromptRecord":
        missing = sorted(REQUIRED_FIELDS - set(data))
        if missing:
            raise ValueError(f"Prompt record is missing required fields: {missing}")

        if data["split"] not in ALLOWED_SPLITS:
            allowed = ", ".join(sorted(ALLOWED_SPLITS))
            raise ValueError(f"Invalid split '{data['split']}'. Use one of: {allowed}.")

        return cls(
            id=str(data["id"]),
            domain=str(data["domain"]),
            prompt=str(data["prompt"]),
            split=str(data["split"]),
            source_type=str(data["source_type"]),
            source_name=str(data["source_name"]),
            prompt_type=str(data["prompt_type"]),
            difficulty=str(data["difficulty"]),
            language=str(data["language"]),
            target=str(data["target"]) if "target" in data and data["target"] is not None else None,
            license=str(data.get("license", "unknown")),
            notes=str(data.get("notes", "")),
        )


def load_prompt_records(path: Path) -> list[PromptRecord]:
    records: list[PromptRecord] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                records.append(PromptRecord.from_dict(data))
            except Exception as exc:
                raise ValueError(f"Invalid prompt record at {path}:{line_number}: {exc}") from exc

    return records


def validate_prompt_records(
    records: list[PromptRecord],
    require_targets: bool = False,
) -> list[str]:
    errors: list[str] = []
    seen_ids: set[str] = set()
    seen_prompts: dict[str, str] = {}

    for record in records:
        if record.id in seen_ids:
            errors.append(f"Duplicate id: {record.id}")
        seen_ids.add(record.id)

        normalized_prompt = " ".join(record.prompt.lower().split())
        if normalized_prompt in seen_prompts:
            errors.append(
                f"Duplicate prompt text: {record.id} duplicates {seen_prompts[normalized_prompt]}"
            )
        seen_prompts[normalized_prompt] = record.id

        if not record.prompt.endswith("?"):
            errors.append(f"Prompt should be a question for now: {record.id}")

        if len(record.prompt.split()) < 3:
            errors.append(f"Prompt is very short: {record.id}")

        if require_targets and not record.target:
            errors.append(f"Record is missing target answer: {record.id}")

        if record.target is not None and len(record.target.split()) < 3:
            errors.append(f"Target answer is very short: {record.id}")

    return errors


def group_prompts_by_domain(
    records: list[PromptRecord],
    split: str | None = None,
) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for record in records:
        if split is not None and record.split != split:
            continue
        grouped.setdefault(record.domain, []).append(record.prompt)
    return grouped


def filter_prompt_records(
    records: list[PromptRecord],
    split: str | None = None,
    domains: list[str] | None = None,
) -> list[PromptRecord]:
    domain_set = set(domains) if domains else None
    return [
        record
        for record in records
        if (split is None or record.split == split)
        and (domain_set is None or record.domain in domain_set)
    ]


def group_records_by_domain(
    records: list[PromptRecord],
) -> dict[str, list[PromptRecord]]:
    grouped: dict[str, list[PromptRecord]] = {}
    for record in records:
        grouped.setdefault(record.domain, []).append(record)
    return grouped
