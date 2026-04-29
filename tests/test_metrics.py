from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from llm_pruning.hooks import ActivationRecord
from llm_pruning.metrics import (
    component_scores,
    concentration_scores,
    leave_one_out_domain_decodability,
)


def record(domain: str, prompt_index: int, component: str, value: float) -> ActivationRecord:
    return ActivationRecord(
        domain=domain,
        prompt_index=prompt_index,
        granularity="test",
        module_name=component,
        unit_index=None,
        mean_abs=value,
        std=0.0,
        max_abs=value,
        numel=1,
    )


class MetricTests(unittest.TestCase):
    def setUp(self) -> None:
        self.records = [
            record("agriculture", 0, "component_a", 10.0),
            record("agriculture", 0, "component_b", 1.0),
            record("agriculture", 1, "component_a", 9.0),
            record("agriculture", 1, "component_b", 1.0),
            record("math", 0, "component_a", 1.0),
            record("math", 0, "component_b", 10.0),
            record("math", 1, "component_a", 1.0),
            record("math", 1, "component_b", 9.0),
        ]

    def test_component_selectivity_ranks_domain_component_first(self) -> None:
        scores = component_scores(self.records, "agriculture")

        self.assertEqual(scores[0].component, "component_a")
        self.assertGreater(scores[0].selectivity, 0.0)
        self.assertLess(scores[-1].selectivity, 0.0)

    def test_concentration_entropy_detects_single_positive_component(self) -> None:
        agriculture = [
            score
            for score in concentration_scores(self.records, score_kind="selectivity")
            if score.domain == "agriculture"
        ][0]

        self.assertEqual(agriculture.entropy, 0.0)
        self.assertEqual(agriculture.effective_components, 1.0)
        self.assertEqual(agriculture.top_5_share, 1.0)

    def test_leave_one_out_decodability(self) -> None:
        score = leave_one_out_domain_decodability(self.records)

        self.assertEqual(score.correct, 4)
        self.assertEqual(score.total, 4)
        self.assertEqual(score.accuracy, 1.0)


if __name__ == "__main__":
    unittest.main()
