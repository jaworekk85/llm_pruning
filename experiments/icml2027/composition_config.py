"""Small strict config for the linked-task screen, with explicit confirmation lock."""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from llm_pruning.run_artifacts import file_fingerprint


@dataclass(frozen=True)
class CompositionConfig:
    experiment_name: str
    dataset_path: Path
    model_name: str
    revision: str | None = None
    device: str = "cpu"
    dtype: str = "float32"
    local_files_only: bool = True
    split: str = "validation"
    seed: int = 2027
    prompt_policy: str = "tokenizer_chat_v1"
    batch_size: int = 4
    max_new_tokens: int = 16
    max_eval_episodes: int | None = None
    max_eval_source_groups: int | None = None
    retained_fractions: tuple = (1., .9, .75, .5)
    calibration_modes: tuple = ("composed", "mixed")
    calibration_seeds: tuple = (11, 23, 37)
    calibration_token_budget: int = 4096
    knowledge_scope: str = "context_supplied"
    bootstrap_samples: int = 1000
    confidence: float = .95
    max_atomic_loss: float = .05
    minimum_extra_loss: float = .10
    min_baseline_accuracy: float = .70
    fixture_run: bool = False
    test_protocol_path: Path | None = None

    def __post_init__(self):
        if not isinstance(self.experiment_name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,99}", self.experiment_name):
            raise ValueError("experiment_name must be a simple directory name")
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ValueError("model_name required")
        if self.split not in {"validation", "test"} or self.knowledge_scope not in {"context_supplied", "parametric_recall"}:
            raise ValueError("Invalid evaluation split/knowledge_scope")
        if self.prompt_policy not in {"plain_short_v1", "tokenizer_chat_v1"} or self.dtype not in {"auto", "float32", "float16", "bfloat16"}:
            raise ValueError("Invalid prompt policy/dtype")
        for key in ("local_files_only", "fixture_run"):
            if type(getattr(self, key)) is not bool:
                raise ValueError(f"{key} must be boolean")
        for key in ("seed", "batch_size", "max_new_tokens", "calibration_token_budget", "bootstrap_samples", "max_eval_episodes", "max_eval_source_groups"):
            value = getattr(self, key)
            if key in {"max_eval_episodes", "max_eval_source_groups"} and value is None:
                continue
            if type(value) is not int or value < (0 if key == "seed" else 1):
                raise ValueError(f"Invalid integer setting: {key}")
        if self.max_eval_episodes is not None and self.max_eval_source_groups is not None:
            raise ValueError("Choose an episode cap or a source-group cap, not both")
        for key in ("retained_fractions", "calibration_modes", "calibration_seeds"):
            value = getattr(self, key)
            if not isinstance(value, (list, tuple)) or not value or len(value) != len(set(value)):
                raise ValueError(f"{key} must be nonempty and unique")
        if 1. not in self.retained_fractions or any(type(b) not in (int, float) or not math.isfinite(b) or not 0 < b <= 1 for b in self.retained_fractions):
            raise ValueError("Retained fractions must include 1 and lie in (0,1]")
        if set(self.calibration_modes) - {"composed", "mixed"} or any(type(s) is not int or s < 0 for s in self.calibration_seeds):
            raise ValueError("Invalid calibration modes/seeds")
        for key in ("max_atomic_loss", "minimum_extra_loss", "min_baseline_accuracy", "confidence"):
            value = getattr(self, key)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value < 1:
                raise ValueError(f"{key} must be in (0,1)")
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("device required")
        if self.revision is not None and (not isinstance(self.revision, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", self.revision)):
            raise ValueError("revision must be an immutable 40-digit commit SHA or null while preparing")

    @classmethod
    def load(cls, path):
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or set(data) - set(cls.__dataclass_fields__):
            raise ValueError("Config must be an object with only known fields")
        if not {"experiment_name", "dataset_path", "model_name"} <= set(data):
            raise ValueError("Missing experiment_name/dataset_path/model_name")
        for key in ("dataset_path", "test_protocol_path"):
            if key not in data or data[key] is None:
                continue
            if not isinstance(data[key], str) or not data[key].strip():
                raise ValueError(f"{key} must be a nonempty path")
            data[key] = (path.resolve().parent / data[key]).resolve()
        return cls(**data)

    def to_dict(self):
        data = asdict(self)
        for key in ("retained_fractions", "calibration_modes", "calibration_seeds"):
            data[key] = list(data[key])
        data["dataset_path"] = str(self.dataset_path)
        data["test_protocol_path"] = str(self.test_protocol_path) if self.test_protocol_path else None
        return data

    def protocol(self):
        settings = self.to_dict()
        for key in ("experiment_name", "dataset_path", "split", "test_protocol_path"):
            settings.pop(key)
        encoded = json.dumps(settings, sort_keys=True, separators=(",", ":"))
        root = Path(__file__).resolve().parents[2]
        paths = ["experiments/icml2027/composition_config.py", "experiments/icml2027/run_composition_screen.py",
                 "src/llm_pruning/short_answers.py", "src/llm_pruning/linked_tasks.py",
                 "src/llm_pruning/composition_analysis.py", "src/llm_pruning/wanda.py",
                 "experiments/icml2027/composition_controls.py", "experiments/icml2027/controlled_composition.py"]
        implementation = {name: file_fingerprint(root / name)["sha256"] for name in paths}
        return {"schema_version": 1, "chosen_on_split": "validation", "implementation_sha256": implementation,
                "dataset_sha256": file_fingerprint(self.dataset_path)["sha256"],
                "settings": settings, "settings_sha256": hashlib.sha256(encoded.encode()).hexdigest()}

    def validate_confirmation(self):
        if self.split == "test":
            if self.test_protocol_path is None:
                raise ValueError("Test requires a frozen validation protocol")
            if json.loads(self.test_protocol_path.read_text(encoding="utf-8")) != self.protocol():
                raise ValueError("Test config/dataset differs from frozen validation protocol")
