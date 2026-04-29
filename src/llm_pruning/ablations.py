from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import torch


COMPONENT_PATTERN = re.compile(r"^(?P<module>.+)\[(?P<unit>\d+)\]$")


@dataclass(frozen=True)
class ComponentRef:
    module_name: str
    unit_index: int | None = None


def parse_component_ref(component: str) -> ComponentRef:
    match = COMPONENT_PATTERN.match(component)
    if match is None:
        return ComponentRef(module_name=component, unit_index=None)

    return ComponentRef(
        module_name=match.group("module"),
        unit_index=int(match.group("unit")),
    )


def replace_first_tensor(value: Any, transform) -> tuple[Any, bool]:
    if isinstance(value, torch.Tensor):
        return transform(value), True

    if isinstance(value, tuple):
        updated = []
        replaced = False
        for item in value:
            if replaced:
                updated.append(item)
                continue
            new_item, replaced = replace_first_tensor(item, transform)
            updated.append(new_item)
        return tuple(updated), replaced

    if isinstance(value, list):
        updated = []
        replaced = False
        for item in value:
            if replaced:
                updated.append(item)
                continue
            new_item, replaced = replace_first_tensor(item, transform)
            updated.append(new_item)
        return updated, replaced

    return value, False


class AblationManager:
    def __init__(
        self,
        model: torch.nn.Module,
        components: list[str],
        granularity: str,
    ) -> None:
        self.model = model
        self.components = [parse_component_ref(component) for component in components]
        self.granularity = granularity
        self._handles: list[Any] = []

    def __enter__(self) -> "AblationManager":
        self.start()
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()

    def start(self) -> None:
        if self._handles:
            return

        modules = dict(self.model.named_modules())

        if self.granularity == "mlp_module":
            self._register_mlp_module_hooks(modules)
        elif self.granularity == "mlp_neuron":
            self._register_unit_hooks(modules)
        elif self.granularity == "attention_head":
            self._register_attention_head_hooks(modules)
        else:
            raise ValueError(
                "Ablation supports granularity values: "
                "mlp_module, mlp_neuron, attention_head."
            )

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def _register_mlp_module_hooks(self, modules: dict[str, torch.nn.Module]) -> None:
        for component in self.components:
            if component.unit_index is not None:
                raise ValueError(f"MLP module ablation does not use unit indices: {component}")

            module = modules.get(component.module_name)
            if module is None:
                raise ValueError(f"Unknown module: {component.module_name}")

            self._handles.append(module.register_forward_hook(self._zero_output_hook))

    def _register_unit_hooks(self, modules: dict[str, torch.nn.Module]) -> None:
        units_by_module: dict[str, list[int]] = defaultdict(list)
        for component in self.components:
            if component.unit_index is None:
                raise ValueError(f"Unit ablation requires unit index: {component}")
            units_by_module[component.module_name].append(component.unit_index)

        for module_name, unit_indices in units_by_module.items():
            module = modules.get(module_name)
            if module is None:
                raise ValueError(f"Unknown module: {module_name}")
            self._handles.append(
                module.register_forward_hook(self._zero_unit_output_hook(unit_indices))
            )

    def _register_attention_head_hooks(self, modules: dict[str, torch.nn.Module]) -> None:
        heads_by_module: dict[str, list[int]] = defaultdict(list)
        for component in self.components:
            if component.unit_index is None:
                raise ValueError(f"Attention head ablation requires head index: {component}")
            heads_by_module[component.module_name].append(component.unit_index)

        num_heads = int(getattr(self.model.config, "num_attention_heads"))
        for module_name, head_indices in heads_by_module.items():
            module = modules.get(module_name)
            if module is None:
                raise ValueError(f"Unknown module: {module_name}")
            self._handles.append(
                module.register_forward_pre_hook(
                    self._zero_attention_head_input_hook(head_indices, num_heads)
                )
            )

    @staticmethod
    def _zero_output_hook(
        _module: torch.nn.Module,
        _inputs: tuple[Any, ...],
        output: Any,
    ) -> Any:
        updated, replaced = replace_first_tensor(output, torch.zeros_like)
        return updated if replaced else output

    @staticmethod
    def _zero_unit_output_hook(unit_indices: list[int]):
        def hook(
            _module: torch.nn.Module,
            _inputs: tuple[Any, ...],
            output: Any,
        ) -> Any:
            def transform(tensor: torch.Tensor) -> torch.Tensor:
                values = tensor.clone()
                valid_indices = [
                    unit_index
                    for unit_index in unit_indices
                    if 0 <= unit_index < values.shape[-1]
                ]
                if valid_indices:
                    values[..., valid_indices] = 0
                return values

            updated, replaced = replace_first_tensor(output, transform)
            return updated if replaced else output

        return hook

    @staticmethod
    def _zero_attention_head_input_hook(head_indices: list[int], num_heads: int):
        def hook(_module: torch.nn.Module, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
            def transform(tensor: torch.Tensor) -> torch.Tensor:
                if tensor.ndim != 3:
                    return tensor

                batch_size, seq_len, hidden_size = tensor.shape
                if hidden_size % num_heads != 0:
                    raise ValueError(
                        f"Cannot split hidden size {hidden_size} into {num_heads} heads."
                    )

                values = tensor.clone()
                head_dim = hidden_size // num_heads
                by_head = values.reshape(batch_size, seq_len, num_heads, head_dim)
                valid_indices = [
                    head_index
                    for head_index in head_indices
                    if 0 <= head_index < num_heads
                ]
                if valid_indices:
                    by_head[:, :, valid_indices, :] = 0
                return by_head.reshape(batch_size, seq_len, hidden_size)

            updated, replaced = replace_first_tensor(inputs, transform)
            return updated if replaced else inputs

        return hook
