"""Offline controls, discovery isolation and complete held-out mask runs."""
import csv
import json
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from experiments.collect_component_activations import run_collection
from experiments.evaluate_component_masks import main, preflight, run_causal
from llm_pruning.baseline_config import BaselineConfig
from llm_pruning.causal_evaluation import build_conditions, condition_budgets
from llm_pruning.components import ComponentRegistry
from llm_pruning.mc_evaluation import score_candidate_batch
from llm_pruning.models import LoadedModel
from llm_pruning.targets import TargetSpec


def test_seeded_controls_are_disjoint_and_match_requested_budgets(tiny_model):
    registry = ComponentRegistry(tiny_model, mlp_group_size=4)
    selected = ["layer.0.attn.head.1", "layer.1.mlp.group.1"]
    strategies = ("random", "layer_matched", "cost_matched", "layer_cost_matched")
    conditions = build_conditions(registry, selected, seed=17, repeats=3, strategies=strategies)
    assert conditions == build_conditions(registry, selected, seed=17, repeats=3, strategies=strategies)
    assert conditions != build_conditions(registry, selected, seed=19, repeats=3, strategies=strategies)
    assert len(conditions) == 14
    budgets = condition_budgets(registry, conditions)
    assert budgets["selected"]["projection_weight_elements"] == 128
    selected_specs = [registry.resolve(value) for value in selected]
    for name, values in conditions.items():
        if name in {"baseline", "selected"}:
            continue
        assert len(values) == len(set(values)) == 2
        assert not set(values) & set(selected)
        specs = [registry.resolve(value) for value in values]
        assert sorted(spec.component_id.kind for spec in specs) == sorted(spec.component_id.kind for spec in selected_specs)
        if name.startswith(("layer_matched_", "layer_cost_matched_")):
            assert sorted((spec.component_id.kind, spec.component_id.layer) for spec in specs) == sorted((spec.component_id.kind, spec.component_id.layer) for spec in selected_specs)
        if name.startswith(("cost_matched_", "layer_cost_matched_")):
            assert budgets[name]["projection_weight_elements"] == budgets["selected"]["projection_weight_elements"]


def test_impossible_tail_group_control_fails_explicitly(tiny_model):
    registry = ComponentRegistry(tiny_model, mlp_group_size=4)
    with pytest.raises(ValueError, match="Insufficient disjoint"):
        build_conditions(registry, ["layer.0.mlp.group.4"], strategies=("layer_cost_matched",))


@pytest.fixture
def causal_setup(tmp_path, tiny_loaded):
    records = [{
        "id": f"fixture.{index}", "domain": "medicine" if index % 2 else "astronomy",
        "language": "en" if index % 2 else "zu", "capability": "recall", "task": "multiple_choice",
        "subject": "fixture", "source_name": "offline_fixture", "source_type": "random_init_test",
        "split": split, "question": f"Question number {index}?", "choices": ["Short", "Longer answer"],
        "answer_index": index % 2, "metadata": {"origin": "test"},
    } for index, split in enumerate(["discovery", "discovery", "validation", "validation", "test", "test"], start=1)]
    dataset = tmp_path / "choices.jsonl"
    dataset.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
    tiny_loaded.tokenizer.chat_template = "{% for message in messages %}<{{ message['role'] }}>{{ message['content'] }}\n{% endfor %}{% if add_generation_prompt %}<assistant>{% endif %}"
    config = BaselineConfig("discovery", "random-init", dataset, TargetSpec("language", "en"), split="discovery", device="cpu", batch_size=4, prompt_policy="tokenizer_chat_v1")
    with patch("experiments.collect_component_activations.load_model", return_value=tiny_loaded):
        source = run_collection(config, tmp_path / "results", mlp_group_size=8)
    return replace(config, experiment_name="causal", split="validation"), source


@pytest.mark.parametrize("granularity", ["attention_head", "mlp_group"])
def test_full_held_out_pipeline_and_paired_damage(causal_setup, tiny_loaded, tmp_path, granularity):
    config, source = causal_setup
    inputs = torch.tensor([[1, 4, 9]])
    with torch.no_grad():
        baseline_logits = tiny_loaded.model(inputs).logits.clone()
    with patch("experiments.evaluate_component_masks.load_model", return_value=tiny_loaded) as loader:
        output = run_causal(config, source, tmp_path / "results", granularity=granularity, repeats=2)
        assert loader.call_count == 1
        before = (output / "manifest.json").read_bytes()
        with pytest.raises(FileExistsError):
            run_causal(config, source, tmp_path / "results")
        assert loader.call_count == 1 and (output / "manifest.json").read_bytes() == before
    manifest = json.loads((output / "manifest.json").read_text())
    selection = json.loads((output / "selection.json").read_text())
    records = json.loads((output / "evaluation_records.json").read_text())
    metrics = json.loads((output / "metrics.json").read_text())
    contrasts = json.loads((output / "control_contrasts.json").read_text())
    conditions = json.loads((output / "conditions.json").read_text())
    with (output / "paired_deltas.csv").open(newline="") as handle:
        paired = list(csv.DictReader(handle))
    with (output / "decisions.csv").open(newline="") as handle:
        decisions = list(csv.DictReader(handle))
    assert manifest["status"] == "complete"
    expected_conditions = 4 if granularity == "attention_head" else 6
    assert manifest["condition_count"] == expected_conditions
    assert manifest["evaluation_record_count"] == 2
    assert manifest["prompt_protocol"]["policy"] == "tokenizer_chat_v1"
    assert set(selection["discovery_ids"]) == {"fixture.1", "fixture.2"}
    assert {record["id"] for record in records} == {"fixture.3", "fixture.4"}
    assert len(metrics) == len(paired) == len(decisions) == expected_conditions * 2
    uncertainty = json.loads((output / "uncertainty.json").read_text())
    assert uncertainty["method"] == "paired_crossed_percentile_bootstrap_v1"
    assert len(uncertainty["results"]) == 4
    assert len(contrasts) == 4
    baseline = {row["record_id"]: row for row in decisions if row["condition"] == "baseline"}
    by_condition_id = {(row["condition"], row["record_id"]): row for row in decisions}
    for row in paired:
        masked = by_condition_id[(row["condition"], row["record_id"])]
        reference = baseline[row["record_id"]]
        assert float(row["accuracy_damage"]) == int(reference["is_correct"] == "True") - int(masked["is_correct"] == "True")
        assert float(row["loss_increase"]) == pytest.approx(float(masked["correct_loss"]) - float(reference["correct_loss"]))
        assert float(row["margin_damage"]) == pytest.approx(float(reference["margin"]) - float(masked["margin"]))
    assert all(row["accuracy_damage"] == row["loss_increase"] == row["margin_damage"] == 0 for row in metrics if row["condition"] == "baseline")
    assert any(abs(row["loss_increase"]) > 1e-7 for row in metrics if row["condition"] == "selected")
    for name, values in conditions["components"].items():
        if name not in {"baseline", "selected"}:
            assert not set(values) & set(selection["selected"])
    assert all(not module._forward_pre_hooks for module in tiny_loaded.model.modules())
    with torch.no_grad():
        assert torch.equal(tiny_loaded.model(inputs).logits, baseline_logits)
    assert len((output / "summary.md").read_text().splitlines()) <= 40


@pytest.mark.parametrize("problem", ["discovery_split", "target", "policy", "scoring", "model", "revision"])
def test_protocol_mismatches_fail_before_model_loading(causal_setup, tmp_path, problem):
    config, source = causal_setup
    updates = {
        "discovery_split": {"split": "discovery"}, "target": {"target": TargetSpec("domain", "medicine")},
        "policy": {"prompt_policy": "plain_mc_v1"}, "scoring": {"scoring_mode": "letter"},
        "model": {"model_name": "another-model"}, "revision": {"revision": "changed"},
    }[problem]
    with patch("experiments.evaluate_component_masks.load_model") as loader:
        with pytest.raises(ValueError):
            run_causal(replace(config, **updates), source, tmp_path / "results")
        loader.assert_not_called()
    assert not (tmp_path / "results" / "causal").exists()


def test_frozen_test_split_is_supported(causal_setup, tiny_loaded, tmp_path):
    config, source = causal_setup
    with patch("experiments.evaluate_component_masks.load_model", return_value=tiny_loaded):
        output = run_causal(replace(config, split="test"), source, tmp_path / "results", repeats=1)
    records = json.loads((output / "evaluation_records.json").read_text())
    assert {record["id"] for record in records} == {"fixture.5", "fixture.6"}


def test_qwen2_discovery_to_causal_pipeline(causal_setup, tiny_loaded, tmp_path):
    config, _ = causal_setup
    torch.manual_seed(71)
    model = Qwen2ForCausalLM(Qwen2Config(
        vocab_size=len(tiny_loaded.tokenizer), hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        max_position_embeddings=1024, pad_token_id=0, eos_token_id=1,
    )).eval()
    loaded = LoadedModel(model, tiny_loaded.tokenizer, torch.device("cpu"))
    discovery_config = replace(config, experiment_name="qwen_discovery", model_name="random-qwen2", split="discovery")
    with patch("experiments.collect_component_activations.load_model", return_value=loaded):
        source = run_collection(discovery_config, tmp_path / "results", mlp_group_size=8)
    with patch("experiments.evaluate_component_masks.load_model", return_value=loaded):
        output = run_causal(replace(discovery_config, experiment_name="qwen_causal", split="validation"), source, tmp_path / "results", repeats=1)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "complete" and manifest["model"]["model_type"] == "qwen2"
    assert all(not module._forward_pre_hooks for module in model.modules())


def test_changed_dataset_fingerprint_is_rejected(causal_setup):
    config, source = causal_setup
    with config.dataset_path.open("a") as handle:
        handle.write("\n")  # Even semantically unchanged bytes belong to a different snapshot.
    with pytest.raises(ValueError, match="fingerprint differs"):
        preflight(config, source)


def test_incomplete_discovery_coverage_is_rejected(causal_setup):
    config, source = causal_setup
    path = source / "activations.csv"
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields, rows = reader.fieldnames, list(reader)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows[:-1])
    with pytest.raises(ValueError, match="Incomplete discovery"):
        preflight(config, source)


def test_rank_is_computed_only_from_discovery_and_requested_region(causal_setup):
    config, source = causal_setup
    path = source / "activations.csv"
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields, rows = reader.fieldnames, list(reader)
    for row in rows:
        if row["token_region"] == "answer" and ".attn.head." in row["component_id"]:
            row["mean_abs"] = 4 if (row["component_id"].endswith("head.0") == (row["record_id"] == "fixture.1")) else 1
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    _, selection = preflight(config, source)
    assert selection["selected"] == ["layer.0.attn.head.0"]
    assert selection["ranking"][0]["score"] == pytest.approx(0.6)
    assert selection["ranking"][1]["score"] == pytest.approx(-0.6)


def test_template_change_and_evaluation_failure_are_recorded(causal_setup, tiny_loaded, tmp_path):
    config, source = causal_setup
    original_template = tiny_loaded.tokenizer.chat_template
    tiny_loaded.tokenizer.chat_template += " "
    with patch("experiments.evaluate_component_masks.load_model", return_value=tiny_loaded):
        with pytest.raises(ValueError, match="prompt protocol differs"):
            run_causal(config, source, tmp_path / "results")
    manifest = json.loads((tmp_path / "results" / "causal" / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    tiny_loaded.tokenizer.chat_template = original_template
    config = replace(config, experiment_name="failing_eval")
    def fail_selected(loaded, candidates, condition, scoring_mode, **kwargs):
        if condition == "selected":
            raise RuntimeError("evaluation failed")
        return score_candidate_batch(loaded, candidates, condition, scoring_mode, **kwargs)
    with patch("experiments.evaluate_component_masks.load_model", return_value=tiny_loaded), patch(
        "llm_pruning.causal_evaluation.score_candidate_batch", side_effect=fail_selected,
    ):
        with pytest.raises(RuntimeError, match="evaluation failed"):
            run_causal(config, source, tmp_path / "results", repeats=1)
    assert all(not module._forward_pre_hooks for module in tiny_loaded.model.modules())
    manifest = json.loads((tmp_path / "results" / "failing_eval" / "manifest.json").read_text())
    assert manifest["status"] == "failed" and manifest["error"]["message"] == "evaluation failed"


def test_discovery_evaluation_id_overlap_is_rejected(causal_setup):
    config, source = causal_setup
    path = source / "collection_records.json"
    records = json.loads(path.read_text())
    records[0]["id"] = "fixture.3"
    path.write_text(json.dumps(records))
    with pytest.raises(ValueError, match="IDs overlap"):
        preflight(config, source)


def test_resolved_model_revision_mismatch_is_recorded(causal_setup, tiny_loaded, tmp_path):
    config, source = causal_setup
    path = source / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["model"]["resolved_revision"] = "different-immutable-revision"
    path.write_text(json.dumps(manifest))
    with patch("experiments.evaluate_component_masks.load_model", return_value=tiny_loaded):
        with pytest.raises(ValueError, match="Resolved model revision differs"):
            run_causal(config, source, tmp_path / "results")
    manifest = json.loads((tmp_path / "results" / "causal" / "manifest.json").read_text())
    assert manifest["status"] == "failed"


def test_causal_validate_only_loads_no_model(causal_setup, tmp_path, monkeypatch, capsys):
    config, source = causal_setup
    config_path = tmp_path / "causal.json"
    config_path.write_text(json.dumps(config.to_dict()))
    monkeypatch.setattr("sys.argv", ["evaluate_component_masks", "--config", str(config_path), "--discovery-run", str(source), "--validate-only"])
    with patch("experiments.evaluate_component_masks.load_model") as loader:
        main()
        loader.assert_not_called()
    assert "2 evaluation examples" in capsys.readouterr().out
    assert not (tmp_path / "results" / "causal").exists()


def test_invalid_uncertainty_family_fails_before_model_work(causal_setup, tmp_path):
    config, source = causal_setup
    with patch("experiments.evaluate_component_masks.load_model") as loader:
        with pytest.raises(ValueError, match="comparison_count"):
            run_causal(config, source, tmp_path / "results", uncertainty_settings={"comparison_count": 1})
        loader.assert_not_called()
    assert not (tmp_path / "results" / "causal").exists()
