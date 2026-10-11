"""Select research targets without tying experiments to domains or one dataset schema."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, TypeVar

from llm_pruning.choice_records import choice_input_key
from llm_pruning.prompt_records import ALLOWED_SPLITS


TARGET_FIELDS = {"domain", "language", "capability", "task"}
Record = TypeVar("Record")


@dataclass(frozen=True)
class TargetSpec:
    field: str
    value: str

    def __post_init__(self) -> None:
        if self.field not in TARGET_FIELDS:
            raise ValueError(f"Unsupported target field: {self.field}")
        if not isinstance(self.value, str) or not self.value.strip():
            raise ValueError("Target value must be a non-empty string.")

    def matches(self, record: object) -> bool:
        return getattr(record, self.field, None) == self.value

    def select(self, records: Sequence[Record], split: str) -> list[Record]:
        if split not in ALLOWED_SPLITS:
            raise ValueError(f"Unsupported split: {split}")
        selected = [r for r in records if getattr(r, "split", None) == split and self.matches(r)]
        if not selected:
            raise ValueError(f"No records for {self.field}={self.value} on split {split}.")
        return selected


def group_records_by_target(records: Sequence[Record], field: str) -> dict[str, list[Record]]:
    """Require explicit labels; unknown languages/capabilities are never inferred."""
    if field not in TARGET_FIELDS:
        raise ValueError(f"Unsupported target field: {field}")
    grouped: dict[str, list[Record]] = {}
    for record in records:
        value = getattr(record, field, None)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Record {getattr(record, 'id', '?')} has no {field} label.")
        grouped.setdefault(value, []).append(record)
    return grouped


def validate_split_isolation(records: Sequence[object]) -> None:
    """Reject duplicate IDs and normalized inputs shared across splits.

    Checks the whole dataset before filtering or sampling. This detects exact
    input reuse, not paraphrases or semantic overlap; source-level audits remain
    necessary for research datasets. MC inputs include the question and choices
    (independent of option order), so common stems alone are not duplicates.
    """
    ids: set[str] = set()
    inputs: dict[object, tuple[str, str]] = {}
    for record in records:
        record_id = getattr(record, "id")
        split = getattr(record, "split")
        if split not in ALLOWED_SPLITS:
            raise ValueError(f"Invalid split for {record_id}: {split}")
        if record_id in ids:
            raise ValueError(f"Duplicate record id: {record_id}")
        ids.add(record_id)
        text = getattr(record, "question", None)
        if text is None:
            text = getattr(record, "prompt", None)
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Record {record_id} has no input text.")
        choices = getattr(record, "choices", None)
        normalized = (
            choice_input_key(text, choices) if choices is not None
            else " ".join(text.casefold().split())
        )
        previous = inputs.get(normalized)
        if previous is not None and previous[0] != split:
            raise ValueError(
                f"Split leakage: {record_id} ({split}) reuses input from "
                f"{previous[1]} ({previous[0]})."
            )
        inputs[normalized] = (split, record_id)
