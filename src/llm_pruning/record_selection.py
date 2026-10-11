"""Shared MC dataset preflight and seeded per-target-value sampling."""
import random

from .baseline_config import BaselineConfig
from .choice_records import ChoiceRecord, load_choice_records, validate_choice_records
from .targets import group_records_by_target, validate_split_isolation


def prepare_records(config: BaselineConfig) -> dict[str, list[ChoiceRecord]]:
    records = load_choice_records(config.dataset_path)
    validate_split_isolation(records)
    errors = validate_choice_records(records)
    if errors:
        raise ValueError(f"Dataset validation failed ({len(errors)} errors): {'; '.join(errors[:5])}")
    config.target.select(records, split=config.split)
    grouped = group_records_by_target([record for record in records if record.split == config.split], config.target.field)
    rng = random.Random(config.seed)
    prepared = {}
    for value in sorted(grouped):
        records_for_value = sorted(grouped[value], key=lambda record: record.id)
        limit = config.max_records_per_value
        if limit is not None and len(records_for_value) > limit:
            records_for_value = sorted(rng.sample(records_for_value, limit), key=lambda record: record.id)
        prepared[value] = records_for_value
    return prepared
