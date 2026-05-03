from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class ActivationRecord:
    domain: str
    prompt_index: int
    granularity: str
    module_name: str
    unit_index: int | None
    mean_abs: float
    std: float
    max_abs: float
    numel: int


def first_tensor(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value

    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = first_tensor(item)
            if tensor is not None:
                return tensor

    if isinstance(value, dict):
        for item in value.values():
            tensor = first_tensor(item)
            if tensor is not None:
                return tensor

    return None


def summarize_tensor(
    *,
    domain: str,
    prompt_index: int,
    granularity: str,
    module_name: str,
    tensor: torch.Tensor,
    unit_index: int | None = None,
) -> ActivationRecord:
    values = tensor.detach().float()
    return ActivationRecord(
        domain=domain,
        prompt_index=prompt_index,
        granularity=granularity,
        module_name=module_name,
        unit_index=unit_index,
        mean_abs=values.abs().mean().item(),
        std=values.std(unbiased=False).item(),
        max_abs=values.abs().max().item(),
        numel=values.numel(),
    )


def summarize_units(
    *,
    domain: str,
    prompt_index: int,
    granularity: str,
    module_name: str,
    tensor: torch.Tensor,
    top_units_per_module: int | None = None,
) -> list[ActivationRecord]:
    values = tensor.detach().float()
    if values.ndim < 1:
        return []

    flat = values.reshape(-1, values.shape[-1])
    unit_means = flat.abs().mean(dim=0)

    if top_units_per_module is None:
        unit_indices = range(flat.shape[-1])
    else:
        k = min(top_units_per_module, flat.shape[-1])
        unit_indices = torch.topk(unit_means, k=k).indices.tolist()

    records: list[ActivationRecord] = []
    for unit_index in unit_indices:
        unit_values = flat[:, unit_index]
        records.append(
            summarize_tensor(
                domain=domain,
                prompt_index=prompt_index,
                granularity=granularity,
                module_name=module_name,
                unit_index=int(unit_index),
                tensor=unit_values,
            )
        )
    return records


def summarize_attention_heads(
    *,
    domain: str,
    prompt_index: int,
    module_name: str,
    tensor: torch.Tensor,
    num_heads: int,
    top_units_per_module: int | None = None,
) -> list[ActivationRecord]:
    values = tensor.detach().float()
    if values.ndim != 3:
        return []

    batch_size, seq_len, hidden_size = values.shape
    if hidden_size % num_heads != 0:
        raise ValueError(
            f"Cannot split {module_name} hidden size {hidden_size} into {num_heads} heads."
        )

    head_dim = hidden_size // num_heads
    by_head = values.reshape(batch_size, seq_len, num_heads, head_dim)
    head_scores = by_head.abs().mean(dim=(0, 1, 3))

    if top_units_per_module is None:
        head_indices = range(num_heads)
    else:
        k = min(top_units_per_module, num_heads)
        head_indices = torch.topk(head_scores, k=k).indices.tolist()

    records: list[ActivationRecord] = []
    for head_index in head_indices:
        records.append(
            summarize_tensor(
                domain=domain,
                prompt_index=prompt_index,
                granularity="attention_head",
                module_name=module_name,
                unit_index=int(head_index),
                tensor=by_head[:, :, head_index, :],
            )
        )
    return records


class ActivationCollector:
    def __init__(
        self,
        model: torch.nn.Module,
        granularity: str = "mlp_module",
        module_filter: str | None = None,
        top_units_per_module: int | None = None,
    ) -> None:
        self.model = model
        self.granularity = granularity
        self.module_filter = module_filter
        self.top_units_per_module = top_units_per_module
        self.records: list[ActivationRecord] = []
        self._handles: list[Any] = []
        self._domain = "unknown"
        self._prompt_index = -1
        self._token_start: int | None = None
        self._token_end: int | None = None

    def start(self) -> None:
        if self._handles:
            return

        if self.granularity == "mlp_module":
            module_filter = self.module_filter or ".mlp"
            for name, module in self.model.named_modules():
                if name.endswith(module_filter):
                    handle = module.register_forward_hook(self._make_module_hook(name))
                    self._handles.append(handle)

        elif self.granularity == "mlp_neuron":
            module_filter = self.module_filter or ".mlp.gate_proj"
            for name, module in self.model.named_modules():
                if name.endswith(module_filter):
                    handle = module.register_forward_hook(self._make_unit_hook(name))
                    self._handles.append(handle)

        elif self.granularity == "attention_head":
            module_filter = self.module_filter or ".self_attn.o_proj"
            num_heads = int(getattr(self.model.config, "num_attention_heads"))
            for name, module in self.model.named_modules():
                if name.endswith(module_filter):
                    handle = module.register_forward_pre_hook(
                        self._make_attention_head_hook(name, num_heads)
                    )
                    self._handles.append(handle)

        else:
            raise ValueError(f"Unsupported hook granularity: {self.granularity}")

        if not self._handles:
            raise ValueError(
                f"No modules matched granularity={self.granularity}, "
                f"module_filter={self.module_filter}"
            )

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def set_context(self, domain: str, prompt_index: int) -> None:
        self._domain = domain
        self._prompt_index = prompt_index

    def set_token_slice(self, token_start: int | None, token_end: int | None = None) -> None:
        self._token_start = token_start
        self._token_end = token_end

    def clear_token_slice(self) -> None:
        self._token_start = None
        self._token_end = None

    def drain_records(self) -> list[ActivationRecord]:
        records = self.records
        self.records = []
        return records

    def _slice_tokens(self, tensor: torch.Tensor) -> torch.Tensor:
        if self._token_start is None or tensor.ndim < 2:
            return tensor
        return tensor[:, self._token_start : self._token_end, ...]

    def _make_module_hook(self, module_name: str):
        def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            tensor = first_tensor(output)
            if tensor is None:
                return
            tensor = self._slice_tokens(tensor)

            self.records.append(
                summarize_tensor(
                    domain=self._domain,
                    prompt_index=self._prompt_index,
                    granularity=self.granularity,
                    module_name=module_name,
                    unit_index=None,
                    tensor=tensor,
                )
            )

        return hook

    def _make_unit_hook(self, module_name: str):
        def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            tensor = first_tensor(output)
            if tensor is None:
                return
            tensor = self._slice_tokens(tensor)

            self.records.extend(
                summarize_units(
                    domain=self._domain,
                    prompt_index=self._prompt_index,
                    granularity=self.granularity,
                    module_name=module_name,
                    tensor=tensor,
                    top_units_per_module=self.top_units_per_module,
                )
            )

        return hook

    def _make_attention_head_hook(self, module_name: str, num_heads: int):
        def hook(_module: torch.nn.Module, inputs: tuple[Any, ...]) -> None:
            tensor = first_tensor(inputs)
            if tensor is None:
                return
            tensor = self._slice_tokens(tensor)

            self.records.extend(
                summarize_attention_heads(
                    domain=self._domain,
                    prompt_index=self._prompt_index,
                    module_name=module_name,
                    tensor=tensor,
                    num_heads=num_heads,
                    top_units_per_module=self.top_units_per_module,
                )
            )

        return hook
