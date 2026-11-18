"""Length and terminal-token behavior of the long-context correctness probe."""
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
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

    def test_holdout_covers_boundary_neighbors_and_context_limit(self):
        fixture = Path(__file__).resolve().parents[2] / "evidence/20261119-w4-holdout/holdout.json.gz"
        cases, _ = probe.read_workload(fixture)
        self.assertEqual([len(case.expected_ids) for case in cases],
                         [255, 256, 257, 511, 512, 513, 1023, 1024, 1025, 7836, 7917])
        self.assertEqual(max(len(case.expected_ids) + case.limit for case in cases), 8173)

    def test_reference_rejects_changed_prompt_ids_and_wrong_input_digest(self):
        case = probe.Case("test", "a pinned prompt", 32, [1, 2])
        workload = probe.Workload([case], "fixture-digest")
        row = {"prompt_sha256": probe.sha256(case.prompt.encode()).hexdigest(), "prompt_tokens": [1, 2]}
        document = {"baseline": [row], "diagnostic": [{}], "workload_sha256": "fixture-digest"}
        with TemporaryDirectory() as directory:
            path = Path(directory) / "capture.json"
            path.write_text(json.dumps(document))
            self.assertEqual(probe.read_capture(path, workload), document)
            with self.assertRaisesRegex(ValueError, "file digest"):
                probe.read_capture(path, workload._replace(sha256="wrong-digest"))
            row["prompt_tokens"][1] = 3
            path.write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError, "prompt IDs differ"):
                probe.read_capture(path, workload)
