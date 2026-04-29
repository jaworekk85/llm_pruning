from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from llm_pruning.ablations import parse_component_ref


class AblationTests(unittest.TestCase):
    def test_parse_module_component(self) -> None:
        component = parse_component_ref("model.layers.20.mlp")

        self.assertEqual(component.module_name, "model.layers.20.mlp")
        self.assertIsNone(component.unit_index)

    def test_parse_indexed_component(self) -> None:
        component = parse_component_ref("model.layers.20.self_attn.o_proj[12]")

        self.assertEqual(component.module_name, "model.layers.20.self_attn.o_proj")
        self.assertEqual(component.unit_index, 12)


if __name__ == "__main__":
    unittest.main()
