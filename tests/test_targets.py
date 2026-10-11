from dataclasses import replace

import pytest

from llm_pruning.choice_records import ChoiceRecord
from llm_pruning.prompt_records import PromptRecord
from llm_pruning.targets import TargetSpec, group_records_by_target, validate_split_isolation


def prompt_record(**updates):
    data = {
        "id": "qa.1", "domain": "math", "prompt": "What is two plus two?",
        "split": "discovery", "source_type": "fixture", "source_name": "fixture",
        "prompt_type": "question", "difficulty": "basic", "language": "en",
        "capability": "arithmetic", "task": "qa", "metadata": {"source_id": "1"},
    }
    return PromptRecord.from_dict({**data, **updates})


def choice_record(**updates):
    data = {
        "id": "mc.1", "domain": "math", "subject": "arithmetic",
        "question": "What is two plus two?", "choices": ["Three", "Four"], "answer_index": 1,
        "split": "validation", "source_type": "fixture", "source_name": "fixture",
        "language": "en", "capability": "arithmetic", "task": "multiple_choice",
        "metadata": {"source_id": "1"},
    }
    return ChoiceRecord.from_dict({**data, **updates})


@pytest.mark.parametrize("field,value", [
    ("domain", "math"), ("language", "en"), ("capability", "arithmetic"), ("task", "qa"),
])
def test_target_selection_respects_field_and_split(field, value):
    discovery = prompt_record()
    held_out = replace(discovery, id="qa.2", split="test", prompt="What is three plus three?")
    other = replace(discovery, id="qa.3", **{field: "other"})
    spec = TargetSpec(field, value)
    assert spec.select([discovery, held_out, other], "discovery") == [discovery]
    assert spec.select([discovery, held_out, other], "test") == [held_out]
    with pytest.raises(ValueError, match="No records"):
        spec.select([discovery], "validation")


@pytest.mark.parametrize("factory", [prompt_record, choice_record])
def test_new_record_metadata_is_preserved(factory):
    record = factory()
    assert record.capability == "arithmetic"
    assert record.metadata == {"source_id": "1"}
    assert group_records_by_target([record], "language") == {"en": [record]}
    with pytest.raises(ValueError, match="JSON object"):
        factory(metadata=[])


def test_legacy_choice_record_does_not_infer_language():
    record = choice_record(language=None, capability=None, task=None)
    assert record.language == record.capability == record.task == ""
    assert TargetSpec("domain", "math").matches(record)
    with pytest.raises(ValueError, match="no language label"):
        group_records_by_target([record], "language")


@pytest.mark.parametrize("field,value", [("subject", "math"), ("domain", ""), ("language", " ")])
def test_invalid_target_rejected(field, value):
    with pytest.raises(ValueError):
        TargetSpec(field, value)


def test_cross_split_text_leakage_is_detected_across_domains():
    first = choice_record(split="discovery")
    second = replace(first, id="mc.2", domain="other", split="test", question=" WHAT is two  plus TWO? ")
    with pytest.raises(ValueError, match="Split leakage"):
        validate_split_isolation([first, second])


def test_duplicate_ids_and_unknown_splits_fail():
    first = prompt_record()
    with pytest.raises(ValueError, match="Duplicate record id"):
        validate_split_isolation([first, replace(first, prompt="Another different question?")])
    with pytest.raises(ValueError, match="Invalid split"):
        validate_split_isolation([replace(first, split="train")])


def test_clean_prompt_and_choice_splits_pass():
    validate_split_isolation([
        prompt_record(), choice_record(id="mc.2", question="What is three plus three?"),
        prompt_record(id="qa.3", split="test", prompt="What is four plus four?"),
    ])


def test_mc_stem_reuse_with_different_options_is_allowed():
    from llm_pruning.choice_records import validate_choice_records

    first = choice_record(split="discovery", question="Which of the following is true?")
    second = replace(first, id="mc.2", split="test", choices=["Five", "Six"])
    validate_split_isolation([first, second])
    assert validate_choice_records([first, second]) == []


def test_reordered_mc_options_cannot_hide_split_leakage():
    first = choice_record(split="discovery")
    second = replace(first, id="mc.2", split="test", choices=list(reversed(first.choices)))
    with pytest.raises(ValueError, match="Split leakage"):
        validate_split_isolation([first, second])
