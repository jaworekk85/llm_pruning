"""Primitive mathematical references and offline activation/probe analysis chain."""
from dataclasses import replace
import json
from unittest.mock import patch

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from experiments.analyze_component_organization import load_measurements, run_analysis
from experiments.collect_component_activations import run_collection
from experiments.probe_component_importance import preflight, run_probes
from experiments.run_compression_frontier import run_frontier
from llm_pruning.baseline_config import BaselineConfig
from llm_pruning.models import LoadedModel
from llm_pruning.organization_diagnostics import concentration, measurement_vectors, overlap, stability
from llm_pruning.targets import TargetSpec


IDS = [f"layer.0.attn.head.{i}" for i in range(3)]


def test_concentration_preserves_zero_mass_and_undefined_signal():
    focused = concentration(dict(zip(IDS, [3., 0., -1.])), top_k=1)
    assert focused["entropy"] == 0 and focused["effective_components"] == 1
    assert focused["gini"] == pytest.approx(2 / 3)
    assert focused["negative_component_count"] == 1
    uniform = concentration(dict(zip(IDS, [1., 1., 1.])), top_k=1)
    assert uniform["normalized_entropy"] == pytest.approx(1)
    assert uniform["effective_components"] == pytest.approx(3)
    assert uniform["gini"] == 0 and uniform["top_k_share"] == pytest.approx(1 / 3)
    assert "top_k_boundary_tie_resolved_by_component_id" in uniform["warnings"]
    assert uniform["top_components"] == [IDS[0]]
    absent = concentration(dict(zip(IDS, [0., -1., -2.])))
    assert absent["entropy"] is absent["top_k_share"] is None
    assert "no_positive_signal" in absent["warnings"]


def test_overlap_and_tied_rank_references():
    a = dict(zip(IDS, [2., 1., 0.]))
    b = dict(zip(IDS, [0., 1., 2.]))
    result = overlap(a, b, top_k=1)
    assert result["top_k_jaccard"] == 0
    assert result["weighted_overlap"] == pytest.approx(1 / 3)
    assert result["signed_cosine"] == pytest.approx(1 / 5)
    assert result["spearman"] == pytest.approx(-1)
    tied = dict(zip(IDS, [1., 1., 3.]))
    assert overlap(tied, tied)["spearman"] == pytest.approx(1)
    constant = dict(zip(IDS, [1., 1., 1.]))
    assert overlap(constant, constant)["spearman"] is None
    empty = dict(zip(IDS, [0., 0., 0.]))
    assert overlap(empty, empty)["top_k_jaccard"] is None
    with pytest.raises(ValueError, match="universes"):
        overlap(a, {IDS[0]: 1})


def measurements():
    records = [{"id": "a", "split": "discovery", "domain": "one"},
               {"id": "b", "split": "discovery", "domain": "two"},
               {"id": "c", "split": "discovery", "domain": "two"}]
    rows = [{"record_id": r["id"], "split": r["split"], "target_field": "domain", "target_value": r["domain"],
             "component_id": component, "token_region": "answer", "mean_abs": value}
            for r, values in zip(records, ([4, 2, 0], [1, 2, 0], [1, 2, 0])) for component, value in zip(IDS, values)]
    return records, rows


def test_streamed_activation_selectivity_equal_example_reference():
    records, rows = measurements()
    vectors = measurement_vectors(iter(rows), records, IDS, "domain", kind="activation")
    one = next(v for v in vectors if v["target_value"] == "one")
    assert one["scores"][IDS[0]] == pytest.approx(.6)
    assert one["scores"][IDS[1]] == one["scores"][IDS[2]] == 0
    assert one["target_count"] == 1 and one["other_count"] == 2
    assert stability(vectors, vectors, top_k=1)[0]["weighted_overlap"] == pytest.approx(1)
    with pytest.raises(ValueError):
        stability(vectors, vectors[:1])


@pytest.mark.parametrize("problem", ["missing", "duplicate", "finite", "label", "heldout", "inventory"])
def test_corrupt_activation_coverage_or_metadata_rejected(problem):
    records, rows = measurements()
    if problem == "missing":
        rows.pop()
    elif problem == "duplicate":
        rows.append(rows[0])
    elif problem == "finite":
        rows[0]["mean_abs"] = float('nan')
    elif problem == "label":
        rows[0]["target_value"] = "wrong"
    elif problem == "heldout":
        records[0]["split"] = "test"
    else:
        rows[0]["component_id"] = "layer.99.attn.head.0"
    with pytest.raises(ValueError):
        measurement_vectors(rows, records, IDS, "domain", kind="activation")


def test_singleton_causal_damage_signed_and_subset_scope():
    records, _ = measurements()
    conditions = {"baseline": [], "p0": [IDS[0]], "p1": [IDS[1]]}
    rows = [{"record_id": r["id"], "split": "discovery", "target_field": "domain", "target_value": r["domain"],
             "condition": condition, "loss_increase": value}
            for r, values in zip(records, ([0, 2, -1], [0, 1, 0], [0, 1, 0])) for condition, value in zip(conditions, values)]
    vectors = measurement_vectors(rows, records, IDS, "domain", kind="causal", conditions=conditions)
    one = next(v for v in vectors if v["target_value"] == "one" and v["score_kind"] == "causal_loss_increase")
    assert one["scores"] == {IDS[0]: 2., IDS[1]: -1.}
    assert one["measured_component_count"] == 2 and one["inventory_component_count"] == 3
    contrast = next(v for v in vectors if v["target_value"] == "one" and v["score_kind"].endswith("contrast"))
    assert contrast["scores"][IDS[0]] == 1
    with pytest.raises(ValueError, match="singleton"):
        measurement_vectors(rows, records, IDS, "domain", kind="causal", conditions={"baseline": [], "selected": IDS[:2]})
    with pytest.raises(ValueError, match="unique"):
        measurement_vectors(rows, records, IDS, "domain", kind="causal", conditions={**conditions, "duplicate": [IDS[0]]})


@pytest.fixture
def diagnostic_setup(tmp_path):
    records = [{"id": f"fixture.{i}", "domain": "one" if i % 2 else "two", "language": "en", "capability": "recall", "task": "mc",
                "subject": "fixture", "source_name": "offline", "source_type": "test", "split": split,
                "question": f"Question {i}?", "choices": ["Yes", "No"], "answer_index": i % 2}
               for i, split in enumerate(["discovery"] * 4 + ["validation"] * 2 + ["test"] * 2)]
    path = tmp_path / "choices.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    return BaselineConfig("activations", "random-init", path, TargetSpec("domain", "one"), split="discovery",
                          device="cpu", batch_size=4, scoring_mode="letter", prompt_policy="plain_mc_v1")


@pytest.mark.parametrize("family", ["llama", "qwen2"])
def test_full_discovery_activation_probe_and_offline_diagnostics(diagnostic_setup, tiny_loaded, tmp_path, family):
    if family == "qwen2":
        tiny_loaded = LoadedModel(Qwen2ForCausalLM(Qwen2Config(vocab_size=tiny_loaded.model.config.vocab_size,
                                  hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
                                  num_key_value_heads=1, max_position_embeddings=1024)).eval(), tiny_loaded.tokenizer, torch.device("cpu"))
    config, root = diagnostic_setup, tmp_path / "results"
    original = {name: p.clone() for name, p in tiny_loaded.model.named_parameters()}
    with patch("experiments.collect_component_activations.load_model", return_value=tiny_loaded):
        activations = run_collection(config, root, mlp_group_size=8)
        repeated = run_collection(replace(config, experiment_name="repeat", seed=10), root, mlp_group_size=8)
    with patch("experiments.probe_component_importance.load_model", return_value=tiny_loaded) as loader:
        probes = run_probes(replace(config, experiment_name="probes"), root, components=IDS[:2], mlp_group_size=8)
        with pytest.raises(FileExistsError):
            run_probes(replace(config, experiment_name="probes"), root)
        assert loader.call_count == 1
    saved = {str(source): (source / "manifest.json").read_bytes() for source in (activations, repeated, probes)}
    output = run_analysis([activations, repeated, probes], "diagnostics", root, top_k=1)
    # The independent frontier shares the same protocol/data, without consuming
    # localization rankings or causal diagnostics. All discovery sources survive.
    with patch("experiments.run_compression_frontier.load_model", return_value=tiny_loaded):
        frontier = run_frontier(replace(config, experiment_name="frontier", split="validation"), root, budgets=(1, .5))
    assert json.loads((frontier / "manifest.json").read_text())["status"] == "complete"
    report = json.loads((output / "diagnostics.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "complete" and manifest["model_inference"] is False
    assert len(report["sources"]) == 3 and len(report["stability"]) == 1
    assert "identical_measurements_are_not_independent_replicates" in report["stability"][0]["warnings"]
    assert all((s / "manifest.json").read_bytes() == saved[str(s)] for s in (activations, repeated, probes))
    assert all(torch.equal(p, original[name]) for name, p in tiny_loaded.model.named_parameters())
    assert all(not module._forward_pre_hooks for module in tiny_loaded.model.modules())
    assert len((output / "summary.md").read_text().splitlines()) <= 40
    with pytest.raises(FileExistsError):
        run_analysis([activations], "diagnostics", root)


def test_probe_heldout_preflight_and_failed_discovery_source(diagnostic_setup, tmp_path):
    with patch("experiments.probe_component_importance.load_model") as loader:
        with pytest.raises(ValueError, match="discovery"):
            run_probes(replace(diagnostic_setup, split="test"), tmp_path / "results")
        loader.assert_not_called()
    source = tmp_path / "bad-source"
    source.mkdir()
    (source / "manifest.json").write_text(json.dumps({"status": "failed", "config": {"split": "discovery"}}))
    (source / "component_inventory.json").write_text('{}')
    with pytest.raises(ValueError, match="completed discovery"):
        load_measurements(source)


def test_stability_model_and_inventory_mismatch(diagnostic_setup, tiny_loaded, tmp_path):
    with patch("experiments.collect_component_activations.load_model", return_value=tiny_loaded):
        a = run_collection(diagnostic_setup, tmp_path / "results", mlp_group_size=8)
        b = run_collection(replace(diagnostic_setup, experiment_name="other"), tmp_path / "results", mlp_group_size=16)
    with pytest.raises(ValueError, match="same model"):
        run_analysis([a, b], "bad-analysis", tmp_path / "results")
    assert not (tmp_path / "results" / "bad-analysis").exists()
