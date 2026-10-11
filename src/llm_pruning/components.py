"""Stable component inventory for HF Llama/Qwen2 causal LMs.

MLP groups are contiguous partitions of *intermediate* channels, not inferred
functional modules. Save the inventory alongside masks: group IDs only have
meaning for a fixed architecture and group size.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import torch


_ID_PATTERN = re.compile(
    r"layer\.(0|[1-9]\d*)\.(attn\.head|mlp\.group)\.(0|[1-9]\d*)"
)


@dataclass(frozen=True)
class ComponentId:
    layer: int
    kind: str
    index: int

    def __post_init__(self) -> None:
        if self.kind not in {"attention_head", "mlp_group"}:
            raise ValueError(f"Unsupported component kind: {self.kind}")
        for value in (self.layer, self.index):
            if type(value) is not int or value < 0:
                raise ValueError("Component indices must be nonnegative integers")

    def __str__(self) -> str:
        unit = "attn.head" if self.kind == "attention_head" else "mlp.group"
        return f"layer.{self.layer}.{unit}.{self.index}"

    @classmethod
    def parse(cls, value: str) -> ComponentId:
        match = _ID_PATTERN.fullmatch(value)
        if match is None:
            raise ValueError(f"Invalid component ID: {value!r}")
        layer, unit, index = match.groups()
        kind = "attention_head" if unit == "attn.head" else "mlp_group"
        return cls(int(layer), kind, int(index))


@dataclass(frozen=True)
class ComponentSpec:
    component_id: ComponentId
    module_name: str
    channel_start: int
    channel_stop: int
    input_width: int

    def to_dict(self) -> dict:
        return {
            "id": str(self.component_id), "module_name": self.module_name,
            "channel_start": self.channel_start, "channel_stop": self.channel_stop,
            "input_width": self.input_width,
        }


class ComponentRegistry:
    """Validated, model-bound inventory; unsupported layouts fail explicitly.

    Head IDs refer to query-head outputs before ``o_proj`` (also with GQA).
    Group IDs refer to gated MLP channels before ``down_proj``. They do not
    refer to gate logits or residual-stream dimensions.
    """

    def __init__(self, model: torch.nn.Module, *, mlp_group_size: int = 128):
        if type(mlp_group_size) is not int or mlp_group_size <= 0:
            raise ValueError("mlp_group_size must be a positive integer")
        config = model.config
        if config.model_type not in {"llama", "qwen2"}:
            raise ValueError(f"Unsupported model type: {config.model_type}")
        self.model = model
        self.mlp_group_size = mlp_group_size
        self.model_type = config.model_type
        modules = dict(model.named_modules())
        layers = modules.get("model.layers")
        if layers is None or len(layers) != config.num_hidden_layers:
            raise ValueError("Expected HF causal LM layout at model.layers")
        specs = []
        heads = config.num_attention_heads
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // heads
        for layer in range(config.num_hidden_layers):
            prefix = f"model.layers.{layer}"
            attn_name = f"{prefix}.self_attn.o_proj"
            down_name = f"{prefix}.mlp.down_proj"
            expected_shapes = {
                attn_name: (heads * head_dim, config.hidden_size),
                down_name: (config.intermediate_size, config.hidden_size),
                f"{prefix}.mlp.gate_proj": (config.hidden_size, config.intermediate_size),
                f"{prefix}.mlp.up_proj": (config.hidden_size, config.intermediate_size),
            }
            for name, (width, output_width) in expected_shapes.items():
                module = modules.get(name)
                if not isinstance(module, torch.nn.Linear) or (
                    module.in_features, module.out_features
                ) != (width, output_width):
                    raise ValueError(f"Unsupported projection layout: {name}")
            for head in range(heads):
                specs.append(ComponentSpec(
                    ComponentId(layer, "attention_head", head), attn_name,
                    head * head_dim, (head + 1) * head_dim, heads * head_dim,
                ))
            for group, start in enumerate(range(0, config.intermediate_size, mlp_group_size)):
                specs.append(ComponentSpec(
                    ComponentId(layer, "mlp_group", group), down_name, start,
                    min(start + mlp_group_size, config.intermediate_size),
                    config.intermediate_size,
                ))
        self.components = tuple(specs)
        self._by_id = {str(spec.component_id): spec for spec in specs}

    def resolve(self, component: str | ComponentId) -> ComponentSpec:
        key = str(ComponentId.parse(component) if isinstance(component, str) else component)
        try:
            return self._by_id[key]
        except KeyError:
            raise ValueError(f"Unknown component ID: {key}") from None

    def to_dict(self) -> dict:
        """JSON-ready mapping to persist with model revision and run metadata."""
        return {
            "schema_version": 1, "model_type": self.model_type,
            "mlp_group_size": self.mlp_group_size,
            "grouping": "contiguous_intermediate_channels",
            "components": [spec.to_dict() for spec in self.components],
        }
