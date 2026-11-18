"""Length and terminal-token behavior of the long-context correctness probe."""
import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location(
    "long_accuracy", Path(__file__).resolve().parents[2] / "live/long_accuracy.py"
)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


class LongAccuracyTests(unittest.TestCase):
    def test_only_terminal_eos_is_removed(self):
        self.assertEqual(probe.terminal_normalized([1, 9, 2, 9], "stop", [9]), [1, 9, 2])
        self.assertEqual(probe.terminal_normalized([1, 9], "length", [9]), [1, 9])
        self.assertEqual(probe.terminal_normalized([], "stop", [9]), [])

    def test_shorter_sequence_reports_exact_common_prefix(self):
        self.assertEqual(probe.common_prefix([1, 2, 3], [1, 2]), 2)
        self.assertEqual(probe.common_prefix([1, 2, 3], [1, 0, 3]), 1)

    def test_logprob_gap_requires_matching_diagnostic_prefix_and_selected_id(self):
        actual = {"tokens": [1, 2]}
        diagnostic = {"tokens": [1, 3], "logprobs": {"top_logprobs": [{"yes": -1, "no": -2}] * 2}}
        self.assertEqual(probe.riley_margin(actual, diagnostic, 0), 1)
        self.assertIsNone(probe.riley_margin(actual, diagnostic, 1))
