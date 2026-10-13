"""Linked cases: semantic grouping, gold consistency, imports and strict config locks."""
from dataclasses import replace
import json

import pytest

from experiments.icml2027.build_composition_data import build_dataset
from experiments.icml2027.composition_config import CompositionConfig
from experiments.icml2027.composition_data import build_linked_data
from experiments.icml2027.run_composition_screen import preflight
from llm_pruning.linked_tasks import validate_linked_records


def dataset_config(tmp_path, *, fixture=True):
    path = build_dataset(tmp_path / "dataset", episodes_per_split=2, seed=7, fixture=fixture)
    return CompositionConfig("screen", path, "offline-random", fixture_run=fixture,
        prompt_policy="plain_short_v1", max_new_tokens=2, retained_fractions=(1., .5),
        calibration_modes=("composed", "mixed"), calibration_seeds=(7,),
        calibration_token_budget=64, bootstrap_samples=20)


def test_builder_reproducible_complete_split_before_variants_and_gold_formula(tmp_path):
    records = build_linked_data(episodes_per_split=7, seed=23)
    assert records == build_linked_data(episodes_per_split=7, seed=23)
    episodes = validate_linked_records(records)
    assert len(episodes) == 21
    assert {split: sum(e["a"].split == split for e in episodes.values())
            for split in ("discovery", "validation", "test")} == {"discovery": 7, "validation": 7, "test": 7}
    for episode in episodes.values():
        a, b = episode["a"], episode["b"]
        assert int(b.target) == int(a.target) * a.metadata["multiplier"] + a.metadata["offset"]
        assert all(r.metadata["knowledge_scope"] == "context_supplied" for r in episode.values())
        assert "correct intermediate" not in episode["composed"].prompt
        assert "correct intermediate" in episode["oracle"].prompt
    config = dataset_config(tmp_path)
    assert b'\r\n' not in config.dataset_path.read_bytes()  # Identical data bytes on Windows/Linux.
    assert len(preflight(config)[0]) == 2
    assert json.loads((config.dataset_path.parent / "manifest.json").read_text())["status"] == "complete"
    with pytest.raises(FileExistsError):
        build_dataset(config.dataset_path.parent, seed=8)


@pytest.mark.parametrize("problem", ["source_leak", "fact_leak", "episode_leak", "missing", "duplicate", "gold", "template", "b_prompt", "metadata"])
def test_invalid_linked_data_rejected_before_selection(problem):
    records = build_linked_data(episodes_per_split=2, seed=7)
    source = next(r for r in records if r.split == "discovery")
    index = next(i for i, r in enumerate(records) if r.split == "validation")
    record = records[index]
    if problem in {"source_leak", "fact_leak", "episode_leak"}:
        field = {"source_leak": "source_group_id", "fact_leak": "fact_group_id", "episode_leak": "episode_id"}[problem]
        records[index] = replace(record, metadata={**record.metadata, field: source.metadata[field]})
    elif problem == "missing":
        records.pop(index)
    elif problem == "duplicate":
        records.append(replace(record, id="new-id"))
    elif problem == "gold":
        records[index] = replace(record, target="999999")
    elif problem == "template":
        records[index] = replace(record, metadata={**record.metadata, "staged_b_template": "{intermediate!r}"})
    elif problem == "b_prompt":
        index = next(i for i, r in enumerate(records) if r.metadata["variant"] == "b")
        records[index] = replace(records[index], prompt="Changed stage? ")
    else:
        records[index] = replace(record, metadata={**record.metadata, "knowledge_scope": "unknown"})
    with pytest.raises(ValueError):
        validate_linked_records(records)


def test_import_parametric_facts_and_alias_groups():
    facts = [{"id": str(i), "source_group_id": str(i), "fact_group_id": str(i),
              "question": f"How many known units in object {i}?", "answer": str(i + 3),
              "knowledge_scope": "parametric_recall", "source_name": "import-test", "license": "test-only"}
             for i in range(3)]
    episodes = validate_linked_records(build_linked_data(facts, seed=3))
    assert all(e["a"].metadata["dataset_role"] == "imported_benchmark" for e in episodes.values())
    with pytest.raises(ValueError, match="same source group"):
        build_linked_data([*facts, {**facts[0], "id": "alias", "question": "Different phrasing?", "source_group_id": "other"}])
    with pytest.raises(ValueError, match="must not supply context"):
        build_linked_data([{**f, "context": "Answer supplied."} for f in facts])


def test_controlled_context_requires_entity_lookup_without_position_or_offset_shortcut():
    import re

    episodes = validate_linked_records(build_linked_data(episodes_per_split=128, seed=2027))
    positions, differences, target_parities = set(), set(), set()
    for episode in episodes.values():
        a, composed = episode["a"], episode["composed"]
        matches = re.findall(r"(Station-[0-9]+) holds ([0-9]+) supply units\.", a.prompt)
        assert len(matches) == 2
        entity = re.search(r"How many supply units does (Station-[0-9]+) hold\?", a.prompt)[1]
        position = next(i for i, (name, _) in enumerate(matches) if name == entity)
        assert int(matches[position][1]) == int(a.target)
        assert int(matches[1 - position][1]) != int(a.target)
        assert a.metadata["context_target_position"] == position
        positions.add(position)
        target_parities.add(int(entity.split('-')[1]) % 2)
        differences.add(int(matches[1 - position][1]) - int(a.target))
        assert composed.prompt.count('?') == 1
        assert 'without outputting it' in composed.prompt
        assert a.difficulty == 'lookup' and episode['b'].difficulty == 'affine'
    assert positions == {0, 1} and target_parities == {0, 1}
    assert min(differences) < 0 < max(differences)
    assert len(differences) > 20


def test_config_confirmation_binding_and_fixture_guard(tmp_path):
    config = dataset_config(tmp_path)
    with pytest.raises(ValueError, match="fixture_run"):
        preflight(replace(config, fixture_run=False))
    with pytest.raises(ValueError, match="frozen"):
        preflight(replace(config, split="test"))
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps(config.protocol()), encoding="utf-8")
    test = replace(config, split="test", test_protocol_path=protocol, experiment_name="confirmation")
    assert len(preflight(test)[0]) == 2
    with pytest.raises(ValueError, match="differs"):
        preflight(replace(test, calibration_token_budget=65))
    locked = json.loads(protocol.read_text())
    assert 'experiments/icml2027/composition_controls.py' in locked['implementation_sha256']
    assert 'experiments/icml2027/controlled_composition.py' in locked['implementation_sha256']
    changed = {**locked, 'implementation_sha256': {**locked['implementation_sha256'],
               'src/llm_pruning/short_answers.py': '0' * 64}}
    protocol.write_text(json.dumps(changed), encoding='utf-8')
    with pytest.raises(ValueError, match="differs"):
        preflight(test)
    protocol.write_text(json.dumps(locked), encoding='utf-8')
    with config.dataset_path.open("a", encoding="utf-8") as handle:
        handle.write("\n")  # Even unchanged semantics require an identical dataset fingerprint.
    with pytest.raises(ValueError, match="differs"):
        preflight(test)


@pytest.mark.parametrize("field,value", [("seed", True), ("batch_size", 0), ("confidence", float('nan')),
    ("retained_fractions", (.5,)), ("calibration_seeds", (0, 0)), ("revision", "main"),
    ("fixture_run", "false"), ("experiment_name", "../oops")])
def test_config_invalid(field, value, tmp_path):
    config = dataset_config(tmp_path)
    with pytest.raises(ValueError):
        replace(config, **{field: value})


def test_config_paths_unknown_fields_and_cohort_sampling(tmp_path):
    config = dataset_config(tmp_path)
    data = config.to_dict()
    data["dataset_path"] = "dataset/records.jsonl"
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    loaded = CompositionConfig.load(path)
    assert loaded.dataset_path == config.dataset_path.resolve()
    selected, _ = preflight(replace(loaded, max_eval_episodes=1))
    assert len(selected) == 1 and all(len(e) == 4 for e in selected.values())
    assert selected == preflight(replace(loaded, max_eval_episodes=1))[0]
    path.write_text(json.dumps({**data, "unknown": 1}), encoding="utf-8")
    with pytest.raises(ValueError):
        CompositionConfig.load(path)
