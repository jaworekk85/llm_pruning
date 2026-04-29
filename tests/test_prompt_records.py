from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from llm_pruning.prompt_records import (
    PromptRecord,
    group_prompts_by_domain,
    validate_prompt_records,
)


class PromptRecordTests(unittest.TestCase):
    def test_record_requires_valid_split(self) -> None:
        data = {
            "id": "math.example.0001",
            "domain": "math",
            "prompt": "What is a prime number?",
            "split": "invalid",
            "source_type": "manual_seed",
            "source_name": "test",
            "prompt_type": "definition",
            "difficulty": "basic",
            "language": "en",
        }

        with self.assertRaises(ValueError):
            PromptRecord.from_dict(data)

    def test_group_prompts_by_split(self) -> None:
        records = [
            PromptRecord.from_dict(
                {
                    "id": "math.example.0001",
                    "domain": "math",
                    "prompt": "What is a prime number?",
                    "split": "discovery",
                    "source_type": "manual_seed",
                    "source_name": "test",
                    "prompt_type": "definition",
                    "difficulty": "basic",
                    "language": "en",
                }
            ),
            PromptRecord.from_dict(
                {
                    "id": "math.example.0002",
                    "domain": "math",
                    "prompt": "What does probability measure?",
                    "split": "test",
                    "source_type": "manual_seed",
                    "source_name": "test",
                    "prompt_type": "definition",
                    "difficulty": "basic",
                    "language": "en",
                }
            ),
        ]

        grouped = group_prompts_by_domain(records, split="discovery")

        self.assertEqual(grouped, {"math": ["What is a prime number?"]})

    def test_validate_requires_targets_when_requested(self) -> None:
        record = PromptRecord.from_dict(
            {
                "id": "math.example.0001",
                "domain": "math",
                "prompt": "What is a prime number?",
                "split": "discovery",
                "source_type": "manual_seed",
                "source_name": "test",
                "prompt_type": "definition",
                "difficulty": "basic",
                "language": "en",
            }
        )

        errors = validate_prompt_records([record], require_targets=True)

        self.assertIn("Record is missing target answer: math.example.0001", errors)

    def test_record_accepts_target_answer(self) -> None:
        record = PromptRecord.from_dict(
            {
                "id": "math.example.0001",
                "domain": "math",
                "prompt": "What is a prime number?",
                "target": "A prime number is divisible only by one and itself.",
                "split": "discovery",
                "source_type": "manual_seed",
                "source_name": "test",
                "prompt_type": "definition",
                "difficulty": "basic",
                "language": "en",
            }
        )

        self.assertEqual(
            record.target,
            "A prime number is divisible only by one and itself.",
        )


if __name__ == "__main__":
    unittest.main()
