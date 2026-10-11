"""Compact target-aware summaries at the same component sites as zero masks."""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

from .components import ComponentRegistry
from .prompt_records import ALLOWED_SPLITS
from .targets import TargetSpec
from .token_regions import tokenize_prompt_answers


@dataclass(frozen=True)
class ActivationSample:
    record_id: str
    split: str
    target_value: str

    def __post_init__(self):
        if self.split not in ALLOWED_SPLITS:
            raise ValueError(f"Unsupported split: {self.split}")
        if any(not isinstance(value, str) or not value.strip()
               for value in (self.record_id, self.target_value)):
            raise ValueError("record_id and target_value must be non-empty strings")


@dataclass(frozen=True)
class ComponentActivation:
    record_id: str
    split: str
    target_field: str
    target_value: str
    is_target: bool
    component_id: str
    token_region: str
    token_count: int
    channel_count: int
    numel: int
    mean: float
    mean_abs: float
    rms: float
    std: float
    max_abs: float


class ComponentActivationCollector:
    """Summarize one full-sequence batch in eval/no-grad mode, without retaining tensors.

    Inputs require [batch, sequence] input_ids and a binary attention_mask.
    Explicit boolean prompt/answer masks must be disjoint, nonempty per sample
    and exclude padding. Token regions denote consumed tokens, not shifted
    next-token prediction positions. Hooks are installed only for this call.
    """

    def __init__(self, registry: ComponentRegistry, target: TargetSpec):
        self.registry = registry
        self.target = target

    def collect(
        self, inputs: Mapping[str, torch.Tensor], samples: Sequence[ActivationSample],
        token_masks: Mapping[str, torch.Tensor],
    ) -> list[ComponentActivation]:
        ids, attention = inputs.get("input_ids"), inputs.get("attention_mask")
        if not isinstance(ids, torch.Tensor) or ids.ndim != 2 or ids.numel() == 0:
            raise ValueError("input_ids must be a nonempty [batch, sequence] tensor")
        if not isinstance(attention, torch.Tensor) or attention.shape != ids.shape:
            raise ValueError("attention_mask must match input_ids")
        if not torch.all((attention == 0) | (attention == 1)):
            raise ValueError("attention_mask must be binary")
        if len(samples) != ids.shape[0] or len({s.record_id for s in samples}) != len(samples):
            raise ValueError("Provide one uniquely identified sample per batch row")
        if not token_masks or set(token_masks) - {"prompt", "answer"}:
            raise ValueError("Token regions must be prompt and/or answer")
        occupied = torch.zeros_like(attention, dtype=torch.bool)
        masks = {}
        for region, mask in token_masks.items():
            if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or mask.shape != ids.shape:
                raise ValueError("Token masks must be boolean tensors matching input_ids")
            mask = mask.to(attention.device)
            if not mask.any(dim=1).all():
                raise ValueError(f"Empty {region} region in a sample")
            if (mask & ~attention.bool()).any():
                raise ValueError("Token regions must exclude padding")
            if (occupied & mask).any():
                raise ValueError("Token regions must not overlap")
            occupied |= mask
            masks[region] = mask
        if self.registry.model.training:
            raise ValueError("Activation collection requires model.eval()")
        if inputs.get("past_key_values") is not None:
            raise ValueError("Collect full sequences without past_key_values")

        by_module = defaultdict(list)
        for spec in self.registry.components:
            by_module[spec.module_name].append(spec)
        modules = dict(self.registry.model.named_modules())
        rows = []
        seen = set()

        def make_hook(name, specs):
            def hook(_module, args, kwargs):
                if name in seen:
                    raise ValueError(f"Projection called more than once: {name}")
                seen.add(name)
                tensor = args[0] if args else kwargs.get("input")
                if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != (*ids.shape, specs[0].input_width):
                    raise ValueError(f"Unexpected projection input shape: {name}")
                for row_index, sample in enumerate(samples):
                    for region in sorted(masks):
                        values = tensor.detach()[row_index, masks[region][row_index].to(tensor.device)].float()
                        stats = []
                        for spec in specs:
                            block = values[:, spec.channel_start:spec.channel_stop]
                            stats.append(torch.stack((
                                block.mean(), block.abs().mean(), block.square().mean().sqrt(),
                                block.std(unbiased=False), block.abs().max(),
                            )))
                        # Transfer only the compact statistics once per sample/region/module.
                        for spec, measures in zip(specs, torch.stack(stats).cpu().tolist(), strict=True):
                            if not all(math.isfinite(value) for value in measures):
                                raise ValueError(f"Non-finite activation summary: {spec.component_id}")
                            channels = spec.channel_stop - spec.channel_start
                            tokens = values.shape[0]
                            rows.append(ComponentActivation(
                                sample.record_id, sample.split, self.target.field, sample.target_value,
                                sample.target_value == self.target.value, str(spec.component_id),
                                region, tokens, channels, tokens * channels, *measures,
                            ))
            return hook

        handles = []
        try:
            for name, specs in by_module.items():
                handles.append(modules[name].register_forward_pre_hook(make_hook(name, specs), with_kwargs=True))
            with torch.no_grad():
                self.registry.model(**{**inputs, "use_cache": False})
            if seen != set(by_module):
                raise ValueError("Some registered component projections were not executed")
        finally:
            for handle in handles:
                handle.remove()
        sample_order = {sample.record_id: index for index, sample in enumerate(samples)}
        component_order = {str(spec.component_id): index for index, spec in enumerate(self.registry.components)}
        return sorted(rows, key=lambda row: (sample_order[row.record_id], component_order[row.component_id], row.token_region))
