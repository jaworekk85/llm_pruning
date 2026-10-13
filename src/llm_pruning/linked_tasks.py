"""Linked integer QA episodes on the existing PromptRecord metadata contract."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from string import Formatter

from .prompt_records import PromptRecord
from .short_answers import parse_integer
from .targets import validate_split_isolation


STATIC_VARIANTS = ("a", "b", "composed", "oracle")
ALL_VARIANTS = (*STATIC_VARIANTS, "staged")


def validate_linked_records(records: list[PromptRecord]) -> dict[str, dict[str, PromptRecord]]:
    """Audit the entire dataset before selection, including group leakage and completeness."""
    if not records:
        raise ValueError("Linked dataset is empty")
    validate_split_isolation(records)
    episodes = defaultdict(dict)
    group_splits = {}
    for record in records:
        metadata = record.metadata
        for key in ("episode_id", "source_group_id", "fact_group_id", "variant", "knowledge_scope",
                    "dataset_role", "staged_b_template", "gold_intermediate"):
            if not isinstance(metadata.get(key), str) or not metadata[key].strip():
                raise ValueError(f"{record.id}: missing nonempty metadata.{key}")
        if metadata["variant"] not in STATIC_VARIANTS:
            raise ValueError(f"{record.id}: unexpected static variant")
        if metadata["knowledge_scope"] not in {"context_supplied", "parametric_recall"}:
            raise ValueError("Unknown knowledge_scope")
        if metadata["dataset_role"] not in {"engineering_fixture", "controlled_benchmark", "imported_benchmark"}:
            raise ValueError("Unknown dataset_role")
        if parse_integer(record.target) is None or parse_integer(metadata["gold_intermediate"]) is None:
            raise ValueError(f"{record.id}: integer gold answers required")
        fields = [field for _, field, _, _ in Formatter().parse(metadata["staged_b_template"]) if field is not None]
        if fields != ["intermediate"] or "{intermediate}" not in metadata["staged_b_template"]:
            raise ValueError("staged_b_template must have exactly one plain {intermediate} placeholder")
        for key in ("episode_id", "source_group_id", "fact_group_id"):
            identity = (key, metadata[key])
            if identity in group_splits and group_splits[identity] != record.split:
                raise ValueError(f"Group split leakage: {key}={metadata[key]}")
            group_splits[identity] = record.split
        episode = episodes[metadata["episode_id"]]
        if metadata["variant"] in episode:
            raise ValueError("Duplicate episode variant")
        episode[metadata["variant"]] = record
    for episode_id, episode in episodes.items():
        if set(episode) != set(STATIC_VARIANTS):
            raise ValueError(f"Incomplete episode: {episode_id}")
        a = episode["a"]
        for record in episode.values():
            if record.split != a.split or any(record.metadata[k] != a.metadata[k] for k in (
                    "source_group_id", "fact_group_id", "knowledge_scope", "dataset_role",
                    "gold_intermediate", "staged_b_template")):
                raise ValueError("Inconsistent episode metadata")
        if parse_integer(a.target) != parse_integer(a.metadata["gold_intermediate"]):
            raise ValueError("A target must equal gold_intermediate")
        if len({parse_integer(episode[v].target) for v in ("b", "composed", "oracle")}) != 1:
            raise ValueError("B/composed/oracle must share the final target")
        if episode["b"].prompt != a.metadata["staged_b_template"].format(intermediate=a.metadata["gold_intermediate"]):
            raise ValueError("B prompt must be staged template with the gold intermediate")
    return dict(episodes)


def staged_record(episode: dict[str, PromptRecord], prediction: int) -> PromptRecord:
    if type(prediction) is not int:
        raise ValueError("Staged input must be a parsed integer")
    b = episode["b"]
    return replace(b, id=b.id + ".staged", prompt=b.metadata["staged_b_template"].format(intermediate=str(prediction)),
                   metadata={**b.metadata, "variant": "staged"})


def decision(record: PromptRecord, output: str, *, malformed_stage: bool = False) -> dict:
    prediction = None if malformed_stage else parse_integer(output)
    return {"record_id": record.id, "episode_id": record.metadata["episode_id"],
            "source_group_id": record.metadata["source_group_id"], "variant": record.metadata["variant"],
            "knowledge_scope": record.metadata["knowledge_scope"], "split": record.split,
            "prompt": record.prompt, "target": record.target, "output": output,
            "prediction": prediction, "valid": prediction is not None,
            "correct": prediction is not None and prediction == parse_integer(record.target),
            "status": "invalid_intermediate" if malformed_stage else ("valid" if prediction is not None else "malformed")}
