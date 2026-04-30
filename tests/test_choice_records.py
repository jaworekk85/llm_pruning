from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from llm_pruning.choice_records import ChoiceRecord, validate_choice_records


class ChoiceRecordTests(unittest.TestCase):
    def test_record_requires_answer_index_in_range(self) -> None:
        with self.assertRaises(ValueError):
            ChoiceRecord.from_dict(
                {
                    "id": "mmlu.test.0001",
                    "domain": "math",
                    "source_name": "cais/mmlu",
                    "source_type": "benchmark_mmlu",
                    "subject": "abstract_algebra",
                    "split": "validation",
                    "question": "Which answer is correct?",
                    "choices": ["A", "B"],
                    "answer_index": 3,
                }
            )

    def test_record_accepts_valid_choice_question(self) -> None:
        record = ChoiceRecord.from_dict(
            {
                "id": "mmlu.test.0001",
                "domain": "math",
                "source_name": "cais/mmlu",
                "source_type": "benchmark_mmlu",
                "subject": "abstract_algebra",
                "split": "validation",
                "question": "Which answer is correct?",
                "choices": ["A", "B"],
                "answer_index": 1,
            }
        )

        self.assertEqual(record.choices[1], "B")
        self.assertEqual(record.answer_index, 1)

    def test_validate_detects_duplicate_choices(self) -> None:
        record = ChoiceRecord.from_dict(
            {
                "id": "mmlu.test.0001",
                "domain": "math",
                "source_name": "cais/mmlu",
                "source_type": "benchmark_mmlu",
                "subject": "abstract_algebra",
                "split": "validation",
                "question": "Which answer is correct?",
                "choices": ["same", "same"],
                "answer_index": 0,
            }
        )

        self.assertIn("Duplicate choices: mmlu.test.0001", validate_choice_records([record]))


if __name__ == "__main__":
    unittest.main()
