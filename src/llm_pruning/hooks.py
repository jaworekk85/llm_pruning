from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class ActivationRecord:
    domain: str
    prompt_index: int
    module_name: str
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


class ActivationCollector:
    def __init__(self, model: torch.nn.Module, module_filter: str = "mlp") -> None:
        self.model = model
        self.module_filter = module_filter
        self.records: list[ActivationRecord] = []
        self._handles: list[Any] = []
        self._domain = "unknown"
        self._prompt_index = -1

    def start(self) -> None:
        if self._handles:
            return

        for name, module in self.model.named_modules():
            if self.module_filter in name:
                handle = module.register_forward_hook(self._make_hook(name))
                self._handles.append(handle)

        if not self._handles:
            raise ValueError(f"No modules matched filter: {self.module_filter}")

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def set_context(self, domain: str, prompt_index: int) -> None:
        self._domain = domain
        self._prompt_index = prompt_index

    def drain_records(self) -> list[ActivationRecord]:
        records = self.records
        self.records = []
        return records

    def _make_hook(self, module_name: str):
        def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            tensor = first_tensor(output)
            if tensor is None:
                return

            values = tensor.detach().float()
            record = ActivationRecord(
                domain=self._domain,
                prompt_index=self._prompt_index,
                module_name=module_name,
                mean_abs=values.abs().mean().item(),
                std=values.std().item(),
                max_abs=values.abs().max().item(),
                numel=values.numel(),
            )
            self.records.append(record)

        return hook
