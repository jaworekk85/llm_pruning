"""Offline architecture/intervention checks; no pretrained weights or downloads."""
import copy
import json

import pytest
import torch

from llm_pruning.component_masks import ComponentMask
from llm_pruning.components import ComponentId, ComponentRegistry


def logits(model):
    with torch.no_grad():
        return model(torch.tensor([[1, 4, 9, 7], [1, 3, 6, 8]])).logits


def test_inventory_is_stable_and_complete_with_gqa_and_partial_group(tiny_model):
    registry = ComponentRegistry(tiny_model, mlp_group_size=4)
    assert len(registry.components) == 18
    assert registry.to_dict() == ComponentRegistry(tiny_model, mlp_group_size=4).to_dict()
    assert json.loads(json.dumps(registry.to_dict())) == registry.to_dict()
    for layer in range(2):
        for kind, width in [("attention_head", 16), ("mlp_group", 18)]:
            channels = [
                channel for spec in registry.components
                if spec.component_id.layer == layer and spec.component_id.kind == kind
                for channel in range(spec.channel_start, spec.channel_stop)
            ]
            assert channels == list(range(width))
    assert registry.resolve("layer.0.attn.head.3").channel_start == 12
    tail = registry.resolve("layer.1.mlp.group.4")
    assert (tail.channel_start, tail.channel_stop) == (16, 18)
    assert tail.module_name == "model.layers.1.mlp.down_proj"
    for spec in registry.components:
        assert ComponentId.parse(str(spec.component_id)) == spec.component_id


@pytest.mark.parametrize("value", [
    "model.layers.0.mlp[0]", "layer.-1.attn.head.0", "layer.0.attn.head.04",
    "layer.0.mlp.neuron.0", "layer.0.mlp.group.0\n",
])
def test_noncanonical_ids_fail(value):
    with pytest.raises(ValueError, match="Invalid component ID"):
        ComponentId.parse(value)


@pytest.mark.parametrize("args", [(True, "mlp_group", 0), (0, "unknown", 0), (0, "attention_head", -1)])
def test_invalid_component_fields_fail(args):
    with pytest.raises(ValueError):
        ComponentId(*args)


@pytest.mark.parametrize("size", [0, -1, True, 1.5])
def test_invalid_group_size_fails(tiny_model, size):
    with pytest.raises(ValueError, match="positive integer"):
        ComponentRegistry(tiny_model, mlp_group_size=size)


def test_unsupported_architecture_and_projection_layout_fail(tiny_model):
    original_type = tiny_model.config.model_type
    tiny_model.config.model_type = "gemma"
    with pytest.raises(ValueError, match="Unsupported model type"):
        ComponentRegistry(tiny_model)
    tiny_model.config.model_type = original_type
    tiny_model.model.layers[0].mlp.down_proj = torch.nn.Linear(17, 16)
    with pytest.raises(ValueError, match="projection layout"):
        ComponentRegistry(tiny_model)


def test_invalid_selection_fails_before_any_hooks_are_installed(tiny_model):
    registry = ComponentRegistry(tiny_model, mlp_group_size=4)
    for invalid in ["layer.2.attn.head.0", "layer.0.attn.head.4", "layer.0.mlp.group.5"]:
        with pytest.raises(ValueError, match="Unknown component ID"):
            ComponentMask(registry, ["layer.0.attn.head.0", invalid])
    assert all(not module._forward_pre_hooks for module in tiny_model.modules())


@pytest.mark.parametrize("implementation", ["eager", "sdpa"])
def test_masks_match_independently_zeroed_weight_columns_and_restore(tiny_model, implementation):
    tiny_model.config._attn_implementation = implementation
    registry = ComponentRegistry(tiny_model, mlp_group_size=4)
    selected = ["layer.0.attn.head.1", "layer.0.mlp.group.4", "layer.1.mlp.group.2"]
    reference = copy.deepcopy(tiny_model)
    reference_modules = dict(reference.named_modules())
    with torch.no_grad():
        for component in selected:
            spec = registry.resolve(component)
            reference_modules[spec.module_name].weight[:, spec.channel_start:spec.channel_stop] = 0
    baseline = logits(tiny_model)
    expected = logits(reference)
    assert not torch.allclose(expected, baseline, atol=1e-7, rtol=1e-7)
    parameters = {name: value.clone() for name, value in tiny_model.state_dict().items()}
    mask = ComponentMask(registry, [*selected, selected[0]])
    with mask:
        mask.start()  # Idempotent start must not duplicate hooks.
        assert len(mask._handles) == 3
        torch.testing.assert_close(logits(tiny_model), expected, atol=1e-6, rtol=1e-5)
    assert torch.equal(logits(tiny_model), baseline)
    for name, value in tiny_model.state_dict().items():
        assert torch.equal(value, parameters[name])
    with mask:  # Reusable after close.
        torch.testing.assert_close(logits(tiny_model), expected, atol=1e-6, rtol=1e-5)


def test_empty_mask_is_exact_identity(tiny_model):
    registry = ComponentRegistry(tiny_model)
    baseline = logits(tiny_model)
    with ComponentMask(registry, []):
        assert torch.equal(logits(tiny_model), baseline)


@pytest.mark.parametrize("kind", ["attention_head", "mlp_group"])
def test_all_units_mask_only_projection_contribution_including_bias(tiny_model, kind):
    registry = ComponentRegistry(tiny_model, mlp_group_size=4)
    specs = [spec for spec in registry.components if spec.component_id.layer == 0 and spec.component_id.kind == kind]
    projection = dict(tiny_model.named_modules())[specs[0].module_name]
    values = torch.randn(2, 3, specs[0].input_width)
    original = values.clone()
    expected = torch.nn.functional.linear(torch.zeros_like(values), projection.weight, projection.bias)
    with ComponentMask(registry, [spec.component_id for spec in specs]):
        torch.testing.assert_close(projection(values), expected)
        torch.testing.assert_close(projection(input=values), expected)
    assert torch.equal(values, original)


def test_mask_preserves_autograd_and_zeros_only_selected_input_gradients(tiny_model):
    registry = ComponentRegistry(tiny_model, mlp_group_size=4)
    projection = tiny_model.model.layers[0].mlp.down_proj
    values = torch.randn(2, 18, requires_grad=True)
    with ComponentMask(registry, ["layer.0.mlp.group.1"]):
        projection(values).sum().backward()
    expected = projection.weight.sum(dim=0).detach().repeat(2, 1)
    expected[:, 4:8] = 0
    torch.testing.assert_close(values.grad, expected)
    assert torch.count_nonzero(projection.weight.grad[:, 4:8]) == 0


def test_exception_cleanup_preserves_unrelated_hooks(tiny_model):
    registry = ComponentRegistry(tiny_model)
    projection = tiny_model.model.layers[0].self_attn.o_proj
    unrelated = projection.register_forward_pre_hook(lambda _module, _args: None)
    baseline = logits(tiny_model)
    with pytest.raises(RuntimeError, match="test failure"):
        with ComponentMask(registry, ["layer.0.attn.head.0"]):
            logits(tiny_model)
            raise RuntimeError("test failure")
    assert len(projection._forward_pre_hooks) == 1
    assert torch.equal(logits(tiny_model), baseline)
    unrelated.remove()


def test_partial_registration_failure_rolls_back(tiny_model, monkeypatch):
    registry = ComponentRegistry(tiny_model)
    projection = tiny_model.model.layers[1].self_attn.o_proj

    def fail(*_args, **_kwargs):
        raise RuntimeError("registration failed")

    monkeypatch.setattr(projection, "register_forward_pre_hook", fail)
    mask = ComponentMask(registry, ["layer.0.attn.head.0", "layer.1.attn.head.0"])
    with pytest.raises(RuntimeError, match="registration failed"):
        mask.start()
    assert all(not module._forward_pre_hooks for module in tiny_model.modules())


def test_reentering_same_context_is_rejected_without_losing_outer_mask(tiny_model):
    registry = ComponentRegistry(tiny_model)
    mask = ComponentMask(registry, ["layer.0.attn.head.0"])
    with mask:
        with pytest.raises(RuntimeError, match="already active"):
            with mask:
                pass
        assert len(mask._handles) == 1
    assert not mask._handles
