"""Reversible zero masks at the projections identified by ComponentRegistry.

These are causal interventions, not physical compression or hardware speedups.
Use in eager mode; compiled/distributed/quantized wrappers are not validated.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

import torch

from .components import ComponentId, ComponentRegistry, ComponentSpec


class ComponentMask:
    """Temporarily zero selected query-head outputs or gated MLP groups.

    Validate the complete selection before registering hooks. Duplicate IDs
    are harmless. Empty selections are identity interventions. Closing the
    context, including on exceptions, removes only this mask's hooks.
    """

    def __init__(self, registry: ComponentRegistry, components: Iterable[str | ComponentId]):
        self.registry = registry
        self.components = tuple(dict.fromkeys(registry.resolve(value) for value in components))
        self._handles = []
        self._active = False

    def __enter__(self) -> ComponentMask:
        if self._active:
            raise RuntimeError("ComponentMask is already active")
        self.start()
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()

    def start(self) -> None:
        if self._active:
            return
        modules = dict(self.registry.model.named_modules())
        by_module = defaultdict(list)
        for spec in self.components:
            by_module[spec.module_name].append(spec)
        try:
            for name, specs in by_module.items():
                module = modules[name]
                if module.in_features != specs[0].input_width:
                    raise ValueError(f"Projection layout changed: {name}")
                self._handles.append(module.register_forward_pre_hook(
                    self._make_hook(specs), with_kwargs=True,
                ))
        except Exception:
            self.close()
            raise
        self._active = True

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._active = False

    @staticmethod
    def _make_hook(specs: list[ComponentSpec]):
        def hook(_module, args, kwargs):
            tensor = args[0] if args else kwargs.get("input")
            if not isinstance(tensor, torch.Tensor) or tensor.ndim < 1:
                raise ValueError("Expected a tensor projection input")
            if tensor.shape[-1] != specs[0].input_width:
                raise ValueError("Unexpected projection input width")
            masked = tensor.clone()
            for spec in specs:
                masked[..., spec.channel_start:spec.channel_stop] = 0
            if args:
                return (masked, *args[1:]), kwargs
            return args, {**kwargs, "input": masked}
        return hook
