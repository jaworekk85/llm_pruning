import csv
import json
from dataclasses import replace
from unittest.mock import patch

import pytest

from experiments.icml2027.run_target_baseline import prepare_records, run_baseline
from llm_pruning.baseline_config import BaselineConfig
from llm_pruning.choice_records import ChoiceRecord
from llm_pruning.mc_evaluation import score_candidate_batch
from llm_pruning.models import load_model
from llm_pruning.targets import TargetSpec


def record_data(index, split="validation", language="en"):
    return {
        "id": f"fixture.{index}", "domain": "science", "subject": "fixture",
        "source_name": "offline_fixture", "source_type": "random_init_test",
        "split": split, "question": f"Question number {index}?",
        "choices": ["A short answer", "A somewhat longer answer"], "answer_index": index % 2,
        "language": language, "capability": "recall", "task": "multiple_choice",
    }


@pytest.fixture
def config(tmp_path):
    dataset = tmp_path / "choices.jsonl"
    records = [record_data(1, "discovery"), record_data(2), record_data(3, language="zu"), record_data(4, "test")]
    dataset.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
    return BaselineConfig("offline_baseline", "random-init-fixture", dataset, TargetSpec("language", "en"), device="cpu")


def test_config_paths_are_relative_to_config_and_unknown_fields_fail(config, tmp_path):
    config_path = tmp_path / "settings.json"
    data = config.to_dict()
    data["dataset_path"] = "choices.jsonl"
    config_path.write_text(json.dumps(data))
    assert BaselineConfig.load(config_path) == config
    data["batch_szie"] = 3
    config_path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="unknown=.*batch_szie"):
        BaselineConfig.load(config_path)


@pytest.mark.parametrize("updates", [
    {"experiment_name": "../escape"}, {"batch_size": 0}, {"batch_size": True},
    {"seed": -1}, {"max_records_per_value": -1}, {"local_files_only": "false"},
    {"split": "train"}, {"scoring_mode": "unknown"}, {"dtype": "int8"},
])
def test_config_rejects_invalid_values(config, updates):
    with pytest.raises(ValueError):
        replace(config, **updates)


def test_seeded_sampling_is_independent_of_dataset_order(config):
    records = [record_data(index, language="en" if index % 2 else "zu") for index in range(10)]
    config.dataset_path.write_text("\n".join(json.dumps(record) for record in records))
    limited = replace(config, max_records_per_value=2, seed=11)
    first = prepare_records(limited)
    config.dataset_path.write_text("\n".join(json.dumps(record) for record in reversed(records)))
    assert prepare_records(limited) == first
    assert all(len(group) == 2 for group in first.values())
    assert all(record.split == "validation" for group in first.values() for record in group)


def test_preflight_checks_unselected_splits_and_never_loads_model(config, tmp_path):
    records = [record_data(1, "discovery"), record_data(2), record_data(3, "test")]
    records[-1]["question"] = records[0]["question"].upper()
    config.dataset_path.write_text("\n".join(json.dumps(record) for record in records))
    with patch("experiments.icml2027.run_target_baseline.load_model") as loader:
        with pytest.raises(ValueError, match="Split leakage"):
            run_baseline(config, tmp_path / "results")
        loader.assert_not_called()
    assert not (tmp_path / "results").exists()


def test_offline_end_to_end_metrics_manifest_and_overwrite_guard(config, tmp_path, tiny_loaded):
    with patch("experiments.icml2027.run_target_baseline.load_model", return_value=tiny_loaded) as loader:
        output = run_baseline(config, tmp_path / "results")
        loader.assert_called_once_with("random-init-fixture", device="cpu", dtype="auto", revision=None, local_files_only=True)
        before = (output / "manifest.json").read_bytes()
        with pytest.raises(FileExistsError):
            run_baseline(config, tmp_path / "results")
        assert loader.call_count == 1
        assert (output / "manifest.json").read_bytes() == before
    manifest = json.loads((output / "manifest.json").read_text())
    metrics = json.loads((output / "metrics.json").read_text())
    assert manifest["status"] == "complete"
    assert len(manifest["dataset"]["sha256"]) == 64
    assert manifest["evaluation"]["counts"] == {"en": 1, "zu": 1}
    assert manifest["model"]["device"] == "cpu"
    assert manifest["model"]["dtype"] == "torch.float32"
    assert manifest["pruner"] is None
    assert "accuracy" in manifest["evaluation_metric_definitions"]
    assert len((output / "summary.md").read_text().splitlines()) <= 40
    assert {row["target_value"] for row in metrics} == {"en", "zu"}
    assert sum(row["is_target"] for row in metrics) == 1
    with (output / "decisions.csv").open() as handle:
        decisions = {row["record_id"]: row for row in csv.DictReader(handle)}
    for row in metrics:
        record_id = "fixture.2" if row["target_value"] == "en" else "fixture.3"
        assert row["correct_loss"] == pytest.approx(float(decisions[record_id]["correct_loss"]))
        assert row["accuracy"] == (decisions[record_id]["is_correct"] == "True")
    selected = json.loads((output / "evaluation_records.json").read_text())
    assert {row["id"] for row in selected} == {"fixture.2", "fixture.3"}


def test_model_load_failure_is_recorded(config, tmp_path):
    with patch("experiments.icml2027.run_target_baseline.load_model", side_effect=RuntimeError("no cached model")):
        with pytest.raises(RuntimeError, match="no cached model"):
            run_baseline(config, tmp_path / "results")
    output = tmp_path / "results" / config.experiment_name
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["error"]["type"] == "RuntimeError"
    assert (output / "evaluation_records.json").exists()


def test_batching_preserves_mc_likelihoods(tiny_loaded):
    record = ChoiceRecord.from_dict(record_data(5))
    candidates = [(record, 0), (record, 1)]
    together = score_candidate_batch(tiny_loaded, candidates, "baseline", "choice_text")
    separate = [score_candidate_batch(tiny_loaded, [candidate], "baseline", "choice_text")[0] for candidate in candidates]
    for batched_score, single_score in zip(together, separate):
        assert batched_score.mean_logprob == pytest.approx(single_score.mean_logprob, abs=1e-6)
        assert batched_score.token_count == single_score.token_count
        assert batched_score.token_count == len(record.choices[batched_score.choice_index]) + 1


def test_model_loader_pins_both_model_and_tokenizer_revision(tiny_loaded):
    with patch("llm_pruning.models.AutoTokenizer.from_pretrained", return_value=tiny_loaded.tokenizer) as tokenizer_loader:
        with patch("llm_pruning.models.AutoModelForCausalLM.from_pretrained", return_value=tiny_loaded.model) as model_loader:
            load_model("fixture", device="cpu", revision="fixed-commit", local_files_only=True)
    assert tokenizer_loader.call_args.kwargs["revision"] == "fixed-commit"
    assert model_loader.call_args.kwargs["revision"] == "fixed-commit"
    assert tokenizer_loader.call_args.kwargs["local_files_only"] is True
    assert model_loader.call_args.kwargs["local_files_only"] is True
