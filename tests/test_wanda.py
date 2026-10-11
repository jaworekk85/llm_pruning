"""Offline algorithm/reference, padding, sequential propagation and restoration."""
import pytest
import torch

from llm_pruning.wanda import WandaPruning, model_cost, wanda_indices


def test_row_ranking_matches_independent_reference_and_stable_ties():
    weight = torch.tensor([[1., 4., -2., 0.], [2., 1., 2., 2.]])
    norm = torch.tensor([2., 0.5, 1., 1.])
    expected = torch.tensor([[3, 0], [1, 2]])
    assert torch.equal(wanda_indices(weight, norm, .5), expected)
    assert wanda_indices(weight, norm, 0).shape == (2, 0)
    assert wanda_indices(weight, norm, 1).shape == weight.shape
    with pytest.raises(ValueError):
        wanda_indices(weight, torch.tensor([1., float('nan'), 1., 1.]), .5)


def calibration(model, padded=False):
    ids = torch.tensor([[1, 4, 9, 7], [1, 3, 5, 8]])
    mask = torch.ones_like(ids)
    if padded:
        ids = torch.cat((torch.zeros(2, 2, dtype=torch.long), ids), dim=1)
        mask = torch.cat((torch.zeros(2, 2, dtype=torch.long), mask), dim=1)
    positions = mask.cumsum(-1) - 1
    positions.masked_fill_(mask == 0, 0)
    return [{"input_ids": ids, "attention_mask": mask, "position_ids": positions}]


def test_sequential_pruning_exact_row_budget_padding_and_dense_storage(tiny_model):
    original = {name: param.detach().clone() for name, param in tiny_model.named_parameters()}
    cost = model_cost(tiny_model)
    inputs = calibration(tiny_model)
    with WandaPruning(tiny_model) as pruning:
        report = pruning.apply(inputs, .5)
        masked = {name: module.weight.clone() for group in pruning.groups for name, module in group.items()}
        for group in pruning.groups:
            for name, module in group.items():
                assert torch.all(module.weight.eq(0).sum(dim=1) == module.in_features // 2)
        assert len(report["projections"]) == 14
        assert model_cost(tiny_model)["prunable_nonzero_count"] < cost["prunable_nonzero_count"]
        assert model_cost(tiny_model)["resident_tensor_storage_bytes"] == cost["resident_tensor_storage_bytes"]
    assert all(torch.equal(param, original[name]) for name, param in tiny_model.named_parameters())
    with WandaPruning(tiny_model) as pruning:
        pruning.apply(calibration(tiny_model, padded=True), .5)
        assert all(torch.equal(module.weight, masked[name]) for group in pruning.groups for name, module in group.items())
    assert all(not module._forward_pre_hooks for module in tiny_model.modules())


def test_first_block_reference_and_next_block_sees_pruned_prefix(tiny_model):
    inputs = calibration(tiny_model)
    records = []
    layer = tiny_model.model.layers[1]
    handle = layer.register_forward_pre_hook(lambda module, args: records.append(args[0].detach().clone()))
    with torch.no_grad():
        tiny_model(**inputs[0], use_cache=False)
    dense_second_input = records.pop()
    with WandaPruning(tiny_model) as pruning:
        first = pruning.groups[0]["model.layers.0.self_attn.q_proj"]
        captured = []
        h = first.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach().clone()))
        weight = first.weight.detach().clone()
        pruning.apply(inputs, .5)
        h.remove()
        norm = captured[0].float().square().sum((0, 1)).sqrt()
        expected = weight.clone()
        expected.scatter_(1, wanda_indices(weight, norm, .5), 0)
        assert torch.equal(first.weight, expected)
        assert not torch.allclose(records[0], dense_second_input)
    handle.remove()


def test_exception_restores_partial_pruning_and_own_hooks(tiny_model):
    original = {name: value.clone() for name, value in tiny_model.named_parameters()}
    def fail(module, args):
        raise RuntimeError("injected later block failure")
    handle = tiny_model.model.layers[1].register_forward_pre_hook(fail)
    with pytest.raises(RuntimeError, match="injected"):
        with WandaPruning(tiny_model) as pruning:
            pruning.apply(calibration(tiny_model), .5)
    assert all(torch.equal(value, original[name]) for name, value in tiny_model.named_parameters())
    assert handle.id in tiny_model.model.layers[1]._forward_pre_hooks
    handle.remove()
    assert all(not module._forward_pre_hooks for module in tiny_model.modules())


@pytest.mark.parametrize("sparsity", [-1, 1.1, float('nan'), True])
def test_invalid_settings_do_not_mutate(tiny_model, sparsity):
    with pytest.raises(ValueError):
        with WandaPruning(tiny_model) as pruning:
            pruning.apply(calibration(tiny_model), sparsity)


def test_sliding_attention_requires_an_explicit_replay_adapter(tiny_model):
    tiny_model.config.use_sliding_window = True
    with pytest.raises(ValueError, match="sliding"):
        WandaPruning(tiny_model)


def test_invalid_padding_and_empty_calibration_cleanup(tiny_model):
    batches = calibration(tiny_model)
    batches[0]["attention_mask"][0, 0] = 2
    with WandaPruning(tiny_model) as pruning:
        with pytest.raises(ValueError, match="binary"):
            pruning.apply(batches, .5)
        with pytest.raises(ValueError, match="nonempty"):
            pruning.apply([], .5)
    assert all(not module._forward_pre_hooks for module in tiny_model.modules())
