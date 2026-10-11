"""Finite mask populations, preserved pairing and offline precision analysis."""
import csv
import itertools
import json
from math import comb
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from experiments.analyze_mask_uncertainty import run_analysis
from llm_pruning.causal_evaluation import _combination_at, build_conditions, evaluate_conditions
from llm_pruning.baseline_config import BaselineConfig
from llm_pruning.choice_records import ChoiceRecord
from llm_pruning.mc_evaluation import score_candidate_batch
from llm_pruning.targets import TargetSpec
from llm_pruning.components import ComponentId, ComponentRegistry, ComponentSpec
from llm_pruning.mask_uncertainty import analyze_uncertainty


def panel(offsets=(0, 1, 2), n=4, split="validation"):
    conditions = {"baseline": [], "selected": ["selected"]}
    conditions.update({f"random_{index:03d}": [f"mask{index}"] for index in range(len(offsets))})
    rows = []
    for index in range(n):
        for name in conditions:
            loss = 0 if name == "baseline" else index + 2 if name == "selected" else index + offsets[int(name.rsplit("_", 1)[1])]
            rows.append({"record_id": str(index), "split": split, "condition": name, "target_field": "language", "target_value": "en", "is_target": True,
                         "accuracy_damage": 0, "loss_increase": loss, "margin_damage": loss})
    return rows, conditions


def sampling(count, population):
    return {"random": {"sampling": "uniform_without_replacement", "unique_masks": count, "available_masks": population}}


def test_combinatorial_unranking_matches_complete_small_space():
    pool = list(range(7))
    expected = list(itertools.combinations(pool, 3))
    assert [tuple(_combination_at(pool, 3, rank)) for rank in range(comb(7, 3))] == expected


def test_controls_cap_at_finite_pool_and_never_duplicate(tiny_model):
    registry = ComponentRegistry(tiny_model, mlp_group_size=4)
    conditions, info = build_conditions(registry, ["layer.0.attn.head.0"], repeats=100,
                                       strategies=("layer_cost_matched",), return_sampling=True)
    masks = [tuple(sorted(values)) for name, values in conditions.items() if name.startswith("layer_cost_matched_")]
    assert len(masks) == len(set(masks)) == 3
    assert info["layer_cost_matched"]["available_masks"] == 3
    assert info["layer_cost_matched"]["unique_masks"] == 3
    assert info["layer_cost_matched"]["requested_masks"] == 100
    assert info["layer_cost_matched"]["exhaustive"] is True


def test_huge_mask_space_is_sampled_without_enumeration():
    model = torch.nn.Module()
    model.projection = torch.nn.Linear(1000, 1)
    specs = [ComponentSpec(ComponentId(0, "attention_head", index), "projection", index, index + 1, 1000) for index in range(1000)]
    by_id = {str(spec.component_id): spec for spec in specs}
    registry = SimpleNamespace(model=model, components=specs, resolve=lambda value: by_id[value])
    conditions, info = build_conditions(registry, list(by_id)[:100], repeats=4, strategies=("random",), return_sampling=True)
    assert info["random"]["available_masks"] == comb(900, 100) > 2**63
    assert len({tuple(sorted(values)) for name, values in conditions.items() if name.startswith("random_")}) == 4


def test_pairing_cancels_shared_example_variation_and_census_mask_variation():
    rows, conditions = panel()
    report = analyze_uncertainty(rows, conditions, sampling(3, 3), samples=500, multiplicity="none")
    result = report["results"][0]
    assert result["estimate"]["loss_increase"] == 1
    assert result["exhaustive"] is True
    for kind in ("joint", "examples_only", "masks_only"):
        assert result["intervals"]["loss_increase"][kind] == {"low": 1, "high": 1}


def test_exhaustive_masks_do_not_remove_example_uncertainty():
    rows, conditions = panel(offsets=(0, 0))
    for row in rows:
        if row["condition"] != "selected":
            row["loss_increase"] = row["margin_damage"] = 0
    report = analyze_uncertainty(rows, conditions, sampling(2, 2), samples=1000, multiplicity="none")
    intervals = report["results"][0]["intervals"]["loss_increase"]
    assert intervals["masks_only"] == {"low": 3.5, "high": 3.5}
    assert intervals["joint"] == intervals["examples_only"]
    assert intervals["joint"]["high"] > intervals["joint"]["low"]


def test_one_nonexhaustive_mask_does_not_produce_joint_certainty():
    rows, conditions = panel(offsets=(1,))
    result = analyze_uncertainty(rows, conditions, sampling(1, 10), samples=100, desired_half_width=0.1)["results"][0]
    assert result["intervals"]["loss_increase"]["joint"] is None
    assert result["intervals"]["loss_increase"]["masks_only"] is None
    assert "mask_uncertainty_not_estimable" in result["warnings"]
    assert result["precision_plan"]["status"] == "insufficient_unique_masks"


def test_single_example_cannot_estimate_joint_uncertainty():
    rows, conditions = panel(n=1)
    result = analyze_uncertainty(rows, conditions, sampling(3, 3), samples=100)["results"][0]
    assert result["intervals"]["loss_increase"]["joint"] is None
    assert "example_uncertainty_not_estimable" in result["warnings"]


def test_shared_mask_uncertainty_does_not_shrink_as_if_examples_were_masks():
    widths = []
    for n in (4, 200):
        rows, conditions = panel(offsets=(0, 2), n=n)
        result = analyze_uncertainty(rows, conditions, sampling(2, 20), samples=500, multiplicity="none")["results"][0]
        interval = result["intervals"]["loss_increase"]["joint"]
        widths.append(interval["high"] - interval["low"])
    assert widths[0] > 0
    assert widths[0] == pytest.approx(widths[1])


def test_identical_masks_across_strategies_reuse_inference(tiny_loaded, tmp_path):
    registry = ComponentRegistry(tiny_loaded.model)
    record = ChoiceRecord("fixture", "science", "offline", "test", "test", "validation", "Which answer?", ["One", "Two"], 0)
    config = BaselineConfig("fixture", "random-init", tmp_path / "unused", TargetSpec("domain", "science"), batch_size=2)
    conditions = {"baseline": [], "selected": ["layer.0.attn.head.0"], "random_000": ["layer.0.attn.head.1"], "layer_cost_matched_000": ["layer.0.attn.head.1"]}
    with patch("llm_pruning.causal_evaluation.score_candidate_batch", wraps=score_candidate_batch) as scorer:
        scores, *_ = evaluate_conditions(tiny_loaded, registry, {"science": [record]}, config, conditions)
    assert scorer.call_count == 3
    assert len(scores) == 8


def test_duplicate_masks_are_collapsed_for_legacy_artifacts():
    rows, conditions = panel(offsets=(0, 0, 2))
    conditions["random_001"] = conditions["random_000"]
    result = analyze_uncertainty(rows, conditions, samples=300)["results"][0]
    assert result["recorded_masks"] == 3 and result["unique_masks"] == 2
    assert result["estimate"]["loss_increase"] == 1
    assert "duplicate_masks_collapsed" in result["warnings"]


def test_precision_plan_tightens_budget_and_uses_finite_pool():
    rows, conditions = panel()
    loose = analyze_uncertainty(rows, conditions, sampling(3, 100), samples=300, multiplicity="none", desired_half_width=0.2)
    tight = analyze_uncertainty(rows, conditions, sampling(3, 100), samples=300, multiplicity="none", desired_half_width=0.05)
    plan = loose["results"][0]["precision_plan"]
    assert plan["estimated_masks_needed"] == 49
    assert plan["current_control_standard_error"] == pytest.approx((0.97 / 3)**0.5)
    assert plan["estimated_masks_needed"] < tight["results"][0]["precision_plan"]["estimated_masks_needed"] <= 100
    assert loose == analyze_uncertainty(rows, conditions, sampling(3, 100), samples=300, multiplicity="none", desired_half_width=0.2)


def test_test_results_do_not_drive_new_mask_budget():
    rows, conditions = panel(split="test")
    result = analyze_uncertainty(rows, conditions, sampling(3, 100), samples=100, desired_half_width=0.1)["results"][0]
    assert result["precision_plan"]["status"] == "not_planned_on_test"


@pytest.mark.parametrize("problem", ["missing", "duplicate", "nan", "metadata", "family"])
def test_invalid_or_incomplete_paired_data_is_rejected(problem):
    rows, conditions = panel()
    kwargs = {}
    if problem == "missing":
        rows.pop()
    elif problem == "duplicate":
        rows.append(dict(rows[0]))
    elif problem == "nan":
        rows[0]["loss_increase"] = float("nan")
    elif problem == "metadata":
        rows[0]["target_value"] = "zu"
    else:
        kwargs["comparison_count"] = 1
    with pytest.raises(ValueError):
        analyze_uncertainty(rows, conditions, samples=100, **kwargs)


def test_offline_analysis_is_exclusive_and_preserves_source(tmp_path):
    rows, conditions = panel()
    source = tmp_path / "source"
    source.mkdir()
    (source / "manifest.json").write_text(json.dumps({"status": "complete", "run_kind": "held_out_causal_evaluation", "model": {"identifier": "offline-fixture"}}))
    (source / "conditions.json").write_text(json.dumps({"components": conditions, "sampling": sampling(3, 3)}))
    with (source / "paired_deltas.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    before = {path.name: path.read_bytes() for path in source.iterdir()}
    with patch("llm_pruning.models.load_model", side_effect=AssertionError("No model inference allowed")):
        output = run_analysis(source, "precision", tmp_path / "results", samples=200)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "complete" and manifest["model_inference"] is False
    assert manifest["versions"]["numpy"] is not None
    assert (output / "uncertainty.json").exists()
    assert {path.name: path.read_bytes() for path in source.iterdir()} == before
    with pytest.raises(FileExistsError):
        run_analysis(source, "precision", tmp_path / "results", samples=200)
