"""Reversible sequential-block Wanda for ordinary HF Llama/Qwen2 linear weights.

Rank |weight| * input L2 norm within each output row, prune floor(sparsity *
input_width) entries, then propagate calibration through the pruned block.
Padding is excluded. No component rankings, recovery training or sparse kernels.
"""
from __future__ import annotations

import hashlib
import math

import torch

from .components import ComponentRegistry


def wanda_indices(weight, input_norm, sparsity):
    """Stable row-wise ranking; ties remove lower input indices first."""
    if isinstance(sparsity, bool) or not isinstance(sparsity, (int, float)) or not math.isfinite(sparsity) or not 0 <= sparsity <= 1:
        raise ValueError("sparsity must be finite and in [0, 1]")
    if weight.ndim != 2 or input_norm.shape != (weight.shape[1],):
        raise ValueError("Expected a matrix and one norm per input channel")
    if not torch.isfinite(weight).all() or not torch.isfinite(input_norm).all() or (input_norm < 0).any():
        raise ValueError("Wanda weights/norms must be finite; norms nonnegative")
    count = math.floor(weight.shape[1] * sparsity + 1e-10)
    metric = weight.detach().float().abs() * input_norm.to(weight.device).float()
    if not torch.isfinite(metric).all():
        raise ValueError("Non-finite Wanda importance")
    return torch.argsort(metric, dim=1, stable=True)[:, :count]


def projection_layers(model):
    """Explicit architecture adapter: decoder q/k/v/o and gate/up/down only."""
    ComponentRegistry(model)  # Validate supported standard decoder layout.
    if getattr(model, "is_quantized", False):
        raise ValueError("Quantized models are unsupported")
    if getattr(model.config, "use_sliding_window", False) or any(
            getattr(layer, "attention_type", "full_attention") != "full_attention" for layer in model.model.layers):
        raise ValueError("Mixed/sliding attention masks need a separate Wanda replay adapter")
    if len({parameter.device for parameter in model.parameters()}) != 1:
        raise ValueError("Use a single-device model; distributed/offloaded layouts are unsupported")
    groups = []
    seen = set()
    for index, layer in enumerate(model.model.layers):
        names = [f"self_attn.{part}_proj" for part in ("q", "k", "v", "o")]
        names += [f"mlp.{part}_proj" for part in ("gate", "up", "down")]
        modules = dict(layer.named_modules())
        group = {}
        for name in names:
            module = modules.get(name)
            if type(module) is not torch.nn.Linear or not module.weight.is_floating_point():
                raise ValueError(f"Unsupported Wanda projection: {index}.{name}")
            if id(module.weight) in seen:
                raise ValueError("Shared projection weights are unsupported")
            seen.add(id(module.weight))
            group[f"model.layers.{index}.{name}"] = module
        if {name for name, module in modules.items() if isinstance(module, torch.nn.Linear)} != set(names):
            raise ValueError("Unexpected decoder projections; add an explicit architecture adapter")
        groups.append(group)
    return groups


def model_cost(model, groups=None):
    """Count actual dense tensor storage and nonzeros separately, including ties."""
    groups = groups if groups is not None else projection_layers(model)
    parameters = list(model.parameters())
    tensors = parameters + list(model.buffers())
    storage_bytes = {}
    for tensor in tensors:
        storage = tensor.untyped_storage()
        key = (str(tensor.device), storage.data_ptr())
        storage_bytes[key] = storage.nbytes()
    projection_weights = [module.weight for group in groups for module in group.values()]
    return {
        "parameter_count": sum(t.numel() for t in parameters),
        "nonzero_parameter_count": sum(int(torch.count_nonzero(t)) for t in parameters),
        "parameter_bytes": sum(t.numel() * t.element_size() for t in parameters),
        "resident_tensor_storage_bytes": sum(storage_bytes.values()),
        "prunable_parameter_count": sum(t.numel() for t in projection_weights),
        "prunable_nonzero_count": sum(int(torch.count_nonzero(t)) for t in projection_weights),
        "storage_format": "dense",
    }


def _move(value, device):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device)
    if isinstance(value, tuple):
        return tuple(_move(item, device) for item in value)
    if isinstance(value, list):
        return [_move(item, device) for item in value]
    if isinstance(value, dict):
        return {key: _move(item, device) for key, item in value.items()}
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise ValueError(f"Unsupported decoder argument type: {type(value).__name__}")


class _Captured(Exception):
    pass


class WandaPruning:
    """One pruning point per context; CPU backups restore weights even on failure.

    Calibration batches contain input_ids, binary attention_mask and optional
    position_ids, on the model device. Activations/decoder kwargs are cached on
    CPU; only one batch and one block's statistics need accelerator memory.
    """

    def __init__(self, model):
        if model.training:
            raise ValueError("Wanda requires an eval model")
        self.model = model
        self.groups = projection_layers(model)
        self.backups = []
        self.active = False
        self.applied = False
        self.report = None

    def __enter__(self):
        if self.active or self.applied:
            raise RuntimeError("Use a fresh WandaPruning context for each budget")
        self.active = True
        return self

    def __exit__(self, exc_type, exc, traceback):
        with torch.no_grad():
            for module, original in reversed(self.backups):
                module.weight.copy_(original.to(module.weight.device))
        self.backups.clear()
        self.active = False
        return False

    @torch.no_grad()
    def apply(self, batches, sparsity):
        if not self.active or self.applied:
            raise RuntimeError("Apply once inside a WandaPruning context")
        # Validate before any forward or mutation, including empty calibration.
        wanda_indices(torch.ones(1, 1), torch.ones(1), sparsity)
        batches = list(batches)
        if not batches:
            raise ValueError("Calibration batches must be nonempty")
        for batch in batches:
            if set(batch) - {"input_ids", "attention_mask", "position_ids"} or not {"input_ids", "attention_mask"} <= set(batch):
                raise ValueError("Calibration requires input_ids and attention_mask; no cache or labels")
            ids, mask = batch["input_ids"], batch["attention_mask"]
            if ids.ndim != 2 or ids.shape != mask.shape or not ((mask == 0) | (mask == 1)).all() or not mask.bool().any(dim=1).all():
                raise ValueError("Calibration requires matching [batch, tokens] inputs and binary nonempty masks")
            limit = getattr(self.model.config, "max_position_embeddings", None)
            if limit is not None and ids.shape[1] > limit:
                raise ValueError("Calibration exceeds context length; truncation is not allowed")
        self.applied = True
        if sparsity == 0:
            self.report = {"method": "wanda_sequential_v1", "requested_sparsity": 0, "projections": []}
            return self.report
        device = next(self.model.parameters()).device
        cached = []

        def capture(module, args, kwargs):
            if args:
                if len(args) != 1:
                    raise ValueError("Unexpected positional decoder arguments")
                hidden = args[0]
            else:
                kwargs = dict(kwargs)
                hidden = kwargs.pop("hidden_states")
            cached.append((hidden.detach().cpu(), _move(kwargs, "cpu")))
            raise _Captured()

        handle = self.model.model.layers[0].register_forward_pre_hook(capture, with_kwargs=True)
        try:
            for batch in batches:
                try:
                    self.model(**_move(batch, device), use_cache=False)
                except _Captured:
                    pass
        finally:
            handle.remove()
        if len(cached) != len(batches):
            raise ValueError("Calibration failed to capture every batch")
        masks = [batch["attention_mask"].bool().cpu() for batch in batches]
        del batches
        report = []
        for layer, group in zip(self.model.model.layers, self.groups, strict=True):
            squares = {name: torch.zeros(module.in_features, device=device, dtype=torch.float32) for name, module in group.items()}
            calls = {name: 0 for name in group}
            current_mask = None

            def hook_for(name):
                def collect(module, args, kwargs):
                    value = args[0] if args else kwargs["input"]
                    if value.shape[:2] != current_mask.shape or value.ndim != 3:
                        raise ValueError("Unexpected projection input shape")
                    real = value.detach()[current_mask].float()
                    if not torch.isfinite(real).all():
                        raise ValueError("Non-finite calibration activation")
                    squares[name].add_(real.square().sum(dim=0))
                    calls[name] += 1
                return collect

            handles = []
            try:
                for name, module in group.items():
                    handles.append(module.register_forward_pre_hook(hook_for(name), with_kwargs=True))
                for (hidden, kwargs), mask in zip(cached, masks, strict=True):
                    current_mask = mask.to(device)
                    layer(hidden.to(device), **_move(kwargs, device))
            finally:
                for handle in handles:
                    handle.remove()
            if any(count != len(cached) for count in calls.values()):
                raise ValueError("Missing or repeated projection during calibration")
            for name, module in group.items():
                self.backups.append((module, module.weight.detach().cpu().clone()))
                norm = squares[name].sqrt()
                # Limit temporary ranking memory to 128 output rows.
                removed = 0
                for start in range(0, module.out_features, 128):
                    weight = module.weight[start:start + 128]
                    indices = wanda_indices(weight, norm, sparsity)
                    weight.scatter_(1, indices, 0)
                    removed += indices.numel()
                digest = hashlib.sha256(module.weight.eq(0).cpu().numpy().tobytes()).hexdigest()
                report.append({"module": name, "weight_elements": module.weight.numel(), "selected_zero_elements": removed,
                               "nonzero_elements": int(torch.count_nonzero(module.weight)), "zero_pattern_sha256": digest})
            # The next block sees the effect of every already-pruned block.
            updated = []
            for hidden, kwargs in cached:
                output = layer(hidden.to(device), **_move(kwargs, device))
                next_hidden = output[0] if isinstance(output, tuple) else output
                updated.append((next_hidden.detach().cpu(), kwargs))
            cached = updated
        self.report = {"method": "wanda_sequential_v1", "requested_sparsity": sparsity, "projections": report,
                       "padding": "excluded", "ranking": "per_output_row_stable", "calibration": "sequential_blocks"}
        return self.report
