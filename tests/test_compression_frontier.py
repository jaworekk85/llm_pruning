"""Tiny random-model specialist frontiers, calibration isolation and artifacts."""
import csv
from dataclasses import replace
import json
from unittest.mock import patch

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from experiments.run_compression_frontier import main, preflight, run_frontier
from llm_pruning.baseline_config import BaselineConfig
from llm_pruning.compression_frontier import calibration_batches, evaluate_frontier
from llm_pruning.mc_evaluation import decisions_from_scores, score_candidate_batch
from llm_pruning.models import LoadedModel
from llm_pruning.record_selection import prepare_records
from llm_pruning.targets import TargetSpec


@pytest.fixture
def frontier_setup(tmp_path):
    rows = [{"id": f"item.{i}", "domain": "medicine" if i % 2 else "astronomy", "language": "en",
             "capability": "recall", "task": "mc", "subject": "fixture", "source_name": "offline", "source_type": "test",
             "split": split, "question": f"Question {i}?", "choices": ["Short", "Long answer"], "answer_index": i % 2}
            for i, split in enumerate(["discovery"] * 4 + ["validation"] * 4 + ["test"] * 4)]
    path = tmp_path / "choices.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return BaselineConfig("frontier", "offline-random", path, TargetSpec("domain", "medicine"), split="validation",
                          batch_size=4, device="cpu", prompt_policy="plain_mc_v1", scoring_mode="choice_text")


@pytest.mark.parametrize("family", ["llama", "qwen2"])
def test_complete_frontier_matches_baseline_and_restores_all_weights(frontier_setup, tiny_loaded, tmp_path, family):
    if family == "qwen2":
        config = Qwen2Config(vocab_size=tiny_loaded.model.config.vocab_size, hidden_size=16, intermediate_size=32,
                            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=1024)
        tiny_loaded = LoadedModel(Qwen2ForCausalLM(config).eval(), tiny_loaded.tokenizer, torch.device("cpu"))
    original = {name: p.clone() for name, p in tiny_loaded.model.named_parameters()}
    grouped = prepare_records(frontier_setup)
    records = [r for items in grouped.values() for r in items]
    expected_scores = score_candidate_batch(tiny_loaded, [(r, i) for r in records for i in range(2)], "baseline", "choice_text", prompt_policy="plain_mc_v1")
    expected = {d.record_id: d for d in decisions_from_scores(expected_scores, {r.id: r for r in records})}
    with patch("experiments.run_compression_frontier.load_model", return_value=tiny_loaded) as loader:
        output = run_frontier(frontier_setup, tmp_path / "results", budgets=(1., .75, .5), calibration_records=1,
                              core_values=("astronomy",), max_accuracy_drop=.1, core_min_accuracy=.1)
        assert loader.call_count == 1
        with pytest.raises(FileExistsError):
            run_frontier(frontier_setup, tmp_path / "results")
        assert loader.call_count == 1
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "complete" and manifest["pruner"] == "wanda_sequential_v1"
    assert manifest["calibration_record_count"] == 1
    calibration = json.loads((output / "calibration_records.json").read_text())
    assert all(r["split"] == "discovery" and r["domain"] == "medicine" for r in calibration)
    with (output / "frontier.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 12  # 2 values + pooled non-target + explicit core, at 3 budgets.
    assert {r["panel"] for r in rows} == {"value", "core", "non_target"}
    dense = [r for r in rows if float(r["retained_fraction"]) == 1]
    assert all(float(r["accuracy_damage"]) == float(r["loss_increase"]) == 0 for r in dense)
    assert len({r["resident_tensor_storage_bytes"] for r in rows}) == 1
    assert len({r["parameter_count"] for r in rows}) == 1
    assert len({r["prunable_nonzero_count"] for r in rows}) == 3
    assert all(r["evaluation_peak_cuda_allocated_bytes"] == "" for r in rows)
    with (output / "decisions.csv").open(newline="") as handle:
        decisions = list(csv.DictReader(handle))
    for d in decisions:
        if float(d["retained_fraction"]) == 1:
            assert float(d["correct_loss"]) == pytest.approx(expected[d["record_id"]].correct_loss)
    with (output / "paired_deltas.csv").open(newline="") as handle:
        paired = list(csv.DictReader(handle))
    assert len(paired) == 12
    assert any(abs(float(r["loss_increase"])) > 1e-6 for r in paired if float(r["retained_fraction"]) == .5)
    assert all(torch.equal(p, original[name]) for name, p in tiny_loaded.model.named_parameters())
    assert all(not module._forward_pre_hooks for module in tiny_loaded.model.modules())
    assert len((output / "summary.md").read_text().splitlines()) <= 40


def test_prompt_calibration_does_not_use_correct_answer(frontier_setup, tiny_loaded):
    _, records = preflight(frontier_setup)
    before = calibration_batches(tiny_loaded, records, frontier_setup)
    after = calibration_batches(tiny_loaded, [replace(r, answer_index=1 - r.answer_index) for r in records], frontier_setup)
    assert all(torch.equal(a[key], b[key]) for a, b in zip(before, after) for key in a)


@pytest.mark.parametrize("problem", ["split", "budget", "missing_baseline", "core", "count", "leak"])
def test_bad_protocol_rejected_without_model(frontier_setup, tmp_path, problem):
    config, settings = frontier_setup, {}
    if problem == "split":
        config = replace(config, split="discovery")
    elif problem == "budget":
        settings["budgets"] = (1, -1)
    elif problem == "missing_baseline":
        settings["budgets"] = (.5,)
    elif problem == "core":
        settings["core_values"] = ("unknown",)
    elif problem == "count":
        settings["calibration_records"] = 0
    else:
        records = [json.loads(line) for line in config.dataset_path.read_text().splitlines()]
        records[4]["question"], records[4]["choices"] = records[0]["question"], records[0]["choices"]
        config.dataset_path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    with patch("experiments.run_compression_frontier.load_model") as loader:
        with pytest.raises(ValueError):
            run_frontier(config, tmp_path / "results", **settings)
        loader.assert_not_called()
    assert not (tmp_path / "results" / "frontier").exists()


def test_failure_after_pruning_restores_and_marks_failed(frontier_setup, tiny_loaded, tmp_path):
    original = {name: p.clone() for name, p in tiny_loaded.model.named_parameters()}
    actual = score_candidate_batch
    def fail_pruned(loaded, candidates, condition, mode, **kwargs):
        if condition != "retained_1":
            raise RuntimeError("injected evaluation failure")
        return actual(loaded, candidates, condition, mode, **kwargs)
    with patch("experiments.run_compression_frontier.load_model", return_value=tiny_loaded), patch("llm_pruning.compression_frontier.score_candidate_batch", side_effect=fail_pruned):
        with pytest.raises(RuntimeError, match="injected"):
            run_frontier(frontier_setup, tmp_path / "results", budgets=(1, .5))
    assert all(torch.equal(p, original[name]) for name, p in tiny_loaded.model.named_parameters())
    manifest = json.loads((tmp_path / "results" / "frontier" / "manifest.json").read_text())
    assert manifest["status"] == "failed"


def test_frozen_test_and_no_implicit_core(frontier_setup, tiny_loaded):
    config = replace(frontier_setup, split="test")
    grouped, calibration = preflight(config)
    result = evaluate_frontier(tiny_loaded, config, grouped, calibration, budgets=(1, .5))
    assert all(row["split"] == "test" and row["panel"] != "core" for row in result["rows"])


def test_validate_only_does_not_load_or_create(frontier_setup, tmp_path, monkeypatch, capsys):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(frontier_setup.to_dict()), encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["frontier", "--config", str(path), "--validate-only"])
    with patch("experiments.run_compression_frontier.load_model") as loader:
        main()
        loader.assert_not_called()
    assert "no model loaded" in capsys.readouterr().out
