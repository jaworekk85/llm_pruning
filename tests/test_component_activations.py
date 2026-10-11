"""Independent projection references, token boundary checks and offline run lifecycle."""
import csv
import json
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch
from tokenizers import Tokenizer, models, processors
from transformers import PreTrainedTokenizerFast

from experiments.collect_component_activations import main, run_collection
from llm_pruning.baseline_config import BaselineConfig
from llm_pruning.component_activations import (
    ActivationSample, ComponentActivationCollector, tokenize_prompt_answers,
)
from llm_pruning.component_masks import ComponentMask
from llm_pruning.components import ComponentRegistry
from llm_pruning.targets import TargetSpec


def sample_batch():
    inputs = {
        "input_ids": torch.tensor([[1, 4, 9, 0], [1, 3, 6, 8]]),
        "attention_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]]),
    }
    masks = {
        "prompt": torch.tensor([[1, 1, 0, 0], [1, 1, 0, 0]], dtype=torch.bool),
        "answer": torch.tensor([[0, 0, 1, 0], [0, 0, 1, 1]], dtype=torch.bool),
    }
    samples = [ActivationSample("first", "discovery", "en"), ActivationSample("second", "discovery", "zu")]
    return inputs, masks, samples


def test_all_summaries_match_independent_projection_capture(tiny_model):
    registry = ComponentRegistry(tiny_model, mlp_group_size=4)
    inputs, masks, samples = sample_batch()
    captured = {}
    handles = []
    for name, module in tiny_model.named_modules():
        if name.endswith(("self_attn.o_proj", "mlp.down_proj")):
            handles.append(module.register_forward_pre_hook(
                lambda _module, args, name=name: captured.update({name: args[0].detach().clone()})
            ))
    with torch.no_grad():
        before = tiny_model(**inputs).logits
    for handle in handles:
        handle.remove()
    rows = ComponentActivationCollector(registry, TargetSpec("language", "en")).collect(inputs, samples, masks)
    assert len(rows) == len(registry.components) * 2 * 2
    assert len({(row.record_id, row.component_id, row.token_region) for row in rows}) == len(rows)
    for row in rows:
        sample_index = 0 if row.record_id == "first" else 1
        spec = registry.resolve(row.component_id)
        block = captured[spec.module_name][sample_index, masks[row.token_region][sample_index], spec.channel_start:spec.channel_stop]
        values = block.double()
        assert row.token_count == block.shape[0]
        assert row.channel_count == block.shape[1]
        assert row.numel == block.numel()
        expected = [values.mean(), values.abs().mean(), values.square().mean().sqrt(), values.std(unbiased=False), values.abs().max()]
        assert [row.mean, row.mean_abs, row.rms, row.std, row.max_abs] == pytest.approx([value.item() for value in expected], rel=1e-5, abs=1e-7)
        assert row.is_target == (sample_index == 0)
        assert row.target_field == "language"
        assert row.target_value == samples[sample_index].target_value
        assert row.split == "discovery"
    with torch.no_grad():
        assert torch.equal(tiny_model(**inputs).logits, before)
    assert all(not module._forward_pre_hooks for module in tiny_model.modules())
    # In particular, the smaller final group must count only its two real channels.
    assert all(row.channel_count == 2 for row in rows if row.component_id.endswith("mlp.group.4"))


def test_collector_observes_the_same_channels_as_masks(tiny_model):
    registry = ComponentRegistry(tiny_model, mlp_group_size=4)
    inputs, masks, samples = sample_batch()
    selected = {"layer.0.attn.head.1", "layer.0.mlp.group.4"}
    with ComponentMask(registry, selected):
        rows = ComponentActivationCollector(registry, TargetSpec("language", "en")).collect(inputs, samples, masks)
    assert all(row.max_abs == row.rms == row.std == 0 for row in rows if row.component_id in selected)


@pytest.mark.parametrize("problem", ["padding", "overlap", "empty", "shape", "dtype", "labels", "training"])
def test_invalid_context_fails_before_forward_or_hook_registration(tiny_model, problem):
    registry = ComponentRegistry(tiny_model, mlp_group_size=4)
    inputs, masks, samples = sample_batch()
    if problem == "padding":
        masks["answer"][0, 3] = True
    elif problem == "overlap":
        masks["answer"][0, 0] = True
    elif problem == "empty":
        masks["answer"][0] = False
    elif problem == "shape":
        masks["prompt"] = masks["prompt"][:, :2]
    elif problem == "dtype":
        masks["prompt"] = masks["prompt"].long()
    elif problem == "labels":
        samples = samples[:1]
    else:
        tiny_model.train()
    with patch.object(tiny_model, "forward") as forward:
        with pytest.raises(ValueError):
            ComponentActivationCollector(registry, TargetSpec("language", "en")).collect(inputs, samples, masks)
        forward.assert_not_called()
    assert all(not module._forward_pre_hooks for module in tiny_model.modules())


def test_forward_failure_removes_hooks_and_collector_can_retry(tiny_model):
    registry = ComponentRegistry(tiny_model, mlp_group_size=4)
    collector = ComponentActivationCollector(registry, TargetSpec("language", "en"))
    inputs, masks, samples = sample_batch()
    with patch.object(tiny_model.model.layers[1].mlp.down_proj, "forward", side_effect=RuntimeError("forward failure")):
        with pytest.raises(RuntimeError, match="forward failure"):
            collector.collect(inputs, samples, masks)
    assert all(not module._forward_pre_hooks for module in tiny_model.modules())
    assert len(collector.collect(inputs, samples, masks)) == 72


def test_nonfinite_statistics_fail_instead_of_writing_invalid_values(tiny_model):
    registry = ComponentRegistry(tiny_model, mlp_group_size=4)
    projection = tiny_model.model.layers[0].self_attn.o_proj
    external = projection.register_forward_pre_hook(lambda _module, args: (torch.full_like(args[0], float("nan")),))
    inputs, masks, samples = sample_batch()
    with pytest.raises(ValueError, match="Non-finite"):
        ComponentActivationCollector(registry, TargetSpec("language", "en")).collect(inputs, samples, masks)
    assert len(projection._forward_pre_hooks) == 1
    external.remove()


@pytest.mark.parametrize("padding_side", ["left", "right"])
def test_regions_and_padding_do_not_change_per_example_summaries(tiny_loaded, padding_side):
    tiny_loaded.tokenizer.padding_side = padding_side
    registry = ComponentRegistry(tiny_loaded.model, mlp_group_size=7)
    collector = ComponentActivationCollector(registry, TargetSpec("language", "en"))
    prompts, answers = ["Question:", "A much longer question:"], [" X", " YZ"]
    inputs, masks = tokenize_prompt_answers(tiny_loaded.tokenizer, prompts, answers, tiny_loaded.device)
    samples = [ActivationSample(str(index), "discovery", "en") for index in range(2)]
    batched = collector.collect(inputs, samples, masks)
    for index in range(2):
        single_inputs, single_masks = tokenize_prompt_answers(tiny_loaded.tokenizer, [prompts[index]], [answers[index]], tiny_loaded.device)
        single = collector.collect(single_inputs, [samples[index]], single_masks)
        for combined, separate in zip([row for row in batched if row.record_id == str(index)], single, strict=True):
            assert combined.token_count == separate.token_count
            assert combined.numel == separate.numel
            assert [combined.mean, combined.mean_abs, combined.rms, combined.std, combined.max_abs] == pytest.approx(
                [separate.mean, separate.mean_abs, separate.rms, separate.std, separate.max_abs], rel=1e-5, abs=1e-7,
            )


def test_unstable_boundary_from_tokenizer_suffix_fails(tiny_loaded):
    tiny_loaded.tokenizer.backend_tokenizer.post_processor = processors.TemplateProcessing(
        single="$A <eos>", special_tokens=[("<eos>", tiny_loaded.tokenizer.eos_token_id)],
    )
    with pytest.raises(ValueError, match="Unstable prompt/answer token boundary"):
        tokenize_prompt_answers(tiny_loaded.tokenizer, ["Question:"], [" X"], tiny_loaded.device)


def test_merge_across_prompt_answer_boundary_fails():
    backend = Tokenizer(models.BPE(
        vocab={"<pad>": 0, "<unk>": 1, "a": 2, "b": 3, "ab": 4},
        merges=[("a", "b")], unk_token="<unk>",
    ))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<pad>", unk_token="<unk>")
    with pytest.raises(ValueError, match="Unstable prompt/answer token boundary"):
        tokenize_prompt_answers(tokenizer, ["a"], ["b"], torch.device("cpu"))


def test_prompt_only_collection_is_recorded(collection_config, tiny_loaded, tmp_path):
    with patch("experiments.collect_component_activations.load_model", return_value=tiny_loaded):
        output = run_collection(collection_config, tmp_path / "results", token_regions=("prompt",))
    with (output / "activations.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6  # 2 examples x (2 heads + one MLP group).
    assert {row["token_region"] for row in rows} == {"prompt"}
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["config"]["token_regions"] == ["prompt"]


@pytest.fixture
def collection_config(tmp_path):
    records = [{
        "id": f"sample.{index}", "domain": "medicine" if index % 2 else "astronomy",
        "language": "en" if index % 2 else "zu", "capability": "recall", "task": "multiple_choice",
        "subject": "fixture", "source_name": "offline_fixture", "source_type": "random_init_test",
        "split": split, "question": f"Question number {index}?",
        "choices": ["Short", "Longer answer"], "answer_index": index % 2,
        "metadata": {"origin": "unit_test"},
    } for index, split in enumerate(["discovery", "discovery", "validation", "test"], start=1)]
    dataset = tmp_path / "choices.jsonl"
    dataset.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
    return BaselineConfig("offline_activations", "random-init", dataset, TargetSpec("language", "en"), split="discovery", device="cpu", batch_size=2)


@pytest.mark.parametrize("field,value", [("domain", "medicine"), ("language", "en"), ("capability", "recall"), ("task", "multiple_choice")])
def test_offline_runner_metadata_inventory_and_overwrite_guard(collection_config, tiny_loaded, tmp_path, field, value):
    config = replace(collection_config, target=TargetSpec(field, value))
    with patch("experiments.collect_component_activations.load_model", return_value=tiny_loaded) as loader:
        output = run_collection(config, tmp_path / "results", mlp_group_size=7)
        loader.assert_called_once_with("random-init", device="cpu", dtype="auto", revision=None, local_files_only=True)
        before = (output / "manifest.json").read_bytes()
        with pytest.raises(FileExistsError):
            run_collection(config, tmp_path / "results", mlp_group_size=7)
        assert loader.call_count == 1
        assert (output / "manifest.json").read_bytes() == before
    manifest = json.loads((output / "manifest.json").read_text())
    inventory = json.loads((output / "component_inventory.json").read_text())
    metadata = json.loads((output / "collection_records.json").read_text())
    with (output / "activations.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == manifest["activation_row_count"] == 28
    assert manifest["status"] == "complete"
    assert manifest["config"]["token_regions"] == ["prompt", "answer"]
    assert manifest["config"]["mlp_group_size"] == 7
    assert len(manifest["dataset"]["sha256"]) == 64
    assert "commit" in manifest["git"] and "transformers" in manifest["versions"]
    assert manifest["model"]["model_type"] == "llama"
    assert {row["record_id"] for row in rows} == {"sample.1", "sample.2"}
    assert {row["split"] for row in rows} == {"discovery"}
    assert {row["target_field"] for row in rows} == {field}
    assert {row["token_region"] for row in rows} == {"prompt", "answer"}
    assert {row["component_id"] for row in rows} == {item["id"] for item in inventory["components"]}
    assert all(record["metadata"] == {"origin": "unit_test"} for record in metadata)
    assert len((output / "summary.md").read_text().splitlines()) <= 40


def test_preflight_audits_unselected_splits_and_never_loads_model(collection_config, tmp_path):
    records = [json.loads(line) for line in collection_config.dataset_path.read_text().splitlines()]
    records[-1]["question"] = records[0]["question"].upper()
    collection_config.dataset_path.write_text("\n".join(json.dumps(record) for record in records))
    with patch("experiments.collect_component_activations.load_model") as loader:
        with pytest.raises(ValueError, match="Split leakage"):
            run_collection(collection_config, tmp_path / "results")
        loader.assert_not_called()
    assert not (tmp_path / "results").exists()


def test_failure_records_error_and_requires_fresh_directory(collection_config, tmp_path):
    with patch("experiments.collect_component_activations.load_model", side_effect=RuntimeError("offline model unavailable")):
        with pytest.raises(RuntimeError, match="offline model unavailable"):
            run_collection(collection_config, tmp_path / "results")
    output = tmp_path / "results" / collection_config.experiment_name
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["error"]["message"] == "offline model unavailable"
    assert (output / "collection_records.json").exists()
    assert "Status: failed" in (output / "summary.md").read_text()


def test_validate_only_cli_creates_no_results_or_model(collection_config, tmp_path, monkeypatch, capsys):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(collection_config.to_dict()))
    monkeypatch.setattr("sys.argv", ["collect_component_activations", "--config", str(config_path), "--results-root", str(tmp_path / "results"), "--validate-only"])
    with patch("experiments.collect_component_activations.load_model") as loader:
        main()
        loader.assert_not_called()
    assert "2 records" in capsys.readouterr().out
    assert not (tmp_path / "results").exists()
