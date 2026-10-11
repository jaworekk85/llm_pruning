"""Validated JSON configuration for the first generic target evaluation pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from llm_pruning.prompt_records import ALLOWED_SPLITS
from llm_pruning.mc_prompts import PROMPT_POLICIES
from llm_pruning.targets import TargetSpec


@dataclass(frozen=True)
class BaselineConfig:
    experiment_name: str
    model_name: str
    dataset_path: Path
    target: TargetSpec
    split: str = "validation"
    revision: str | None = None
    device: str | None = None
    dtype: str = "auto"
    local_files_only: bool = True
    scoring_mode: str = "choice_text"
    batch_size: int = 16
    max_records_per_value: int | None = None
    seed: int = 0
    prompt_policy: str = "legacy_mc_v1"

    @classmethod
    def load(cls, path: Path) -> "BaselineConfig":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Config must be a JSON object.")
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(data) - allowed)
        missing = sorted({"experiment_name", "model_name", "dataset_path", "target"} - set(data))
        if unknown or missing:
            raise ValueError(f"Invalid config fields; unknown={unknown}, missing={missing}")
        target = data["target"]
        if not isinstance(target, dict) or set(target) != {"field", "value"}:
            raise ValueError("target must contain exactly field and value.")
        data["target"] = TargetSpec(**target)
        if not isinstance(data["dataset_path"], str) or not data["dataset_path"].strip():
            raise ValueError("dataset_path must be a non-empty string.")
        dataset_path = Path(data["dataset_path"])
        # Config-relative paths keep the run independent of the working directory.
        data["dataset_path"] = (path.resolve().parent / dataset_path).resolve()
        return cls(**data)

    def __post_init__(self) -> None:
        name = self.experiment_name
        if not isinstance(name, str) or not name or len(name) > 100:
            raise ValueError("experiment_name must have 1–100 characters.")
        if name[0] in ".-" or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name):
            raise ValueError("experiment_name must be a simple directory name using letters, digits, _ or -.")
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ValueError("model_name must be a non-empty string.")
        if not isinstance(self.target, TargetSpec):
            raise ValueError("target must be a TargetSpec.")
        if self.split not in ALLOWED_SPLITS:
            raise ValueError(f"Unsupported split: {self.split}")
        if self.scoring_mode not in {"letter", "choice_text"}:
            raise ValueError(f"Unsupported scoring_mode: {self.scoring_mode}")
        if self.prompt_policy not in PROMPT_POLICIES:
            raise ValueError(f"Unsupported prompt_policy: {self.prompt_policy}")
        if self.dtype not in {"auto", "float32", "float16", "bfloat16"}:
            raise ValueError(f"Unsupported dtype: {self.dtype}")
        for key in ("revision", "device"):
            value = getattr(self, key)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{key} must be null or a non-empty string.")
        if not isinstance(self.local_files_only, bool):
            raise ValueError("local_files_only must be boolean.")
        for key in ("batch_size", "max_records_per_value", "seed"):
            value = getattr(self, key)
            if key == "max_records_per_value" and value is None:
                continue
            if type(value) is not int or value < (0 if key == "seed" else 1):
                raise ValueError(f"{key} must be an integer >= {0 if key == 'seed' else 1}.")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["dataset_path"] = str(self.dataset_path)
        return data
