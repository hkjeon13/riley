"""Exercise the trace binding with a simulated committed mixed-step plan."""

from contextlib import chdir
from pathlib import Path
import runpy
import sys
import tarfile
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
VERIFY = runpy.run_path(str(HERE / "verify.py"))
TIMELINE = runpy.run_path(str(HERE.parent / "20261124-w3-rag-prefill-steps/verify.py"))
Step = TIMELINE["Step"]


class MixedStepTraceTests(unittest.TestCase):
    def audit(self, enabled, change=None):
        logs = [
            "W3-MIXED-CHOICES 1 0",
            f"W3-MIXED-CHOICE 3 {int(enabled)} 3 3500 0 2 3500 1023 1023",
            "W3-MIXED-DECODE 2 1:2",
            "W3-MIXED-DECODE 3 1:3",
        ]
        if change is not None:
            logs = change(logs)
        pick, target = (2, 2046) if enabled else (3, 1023)
        steps = [
            Step(1, 1_000_000, 2_000_000, 2_000_000, 1_000_000, 1, 0, [(1, 1, 1)]),
            Step(2, 3_000_000, 5_000_000, 5_000_000, 2_000_000, 1024, 1, [(2, 1023, 1023)]),
            Step(3, 6_000_000, 9_000_000, 9_000_000, 3_000_000, 1024, 1, [(pick, 1023, target)]),
        ]
        choices, decodes = VERIFY["read_events"](logs, TIMELINE["put_unique"])
        return VERIFY["check_steps"](steps, choices, decodes, {1: 0, 2: 0, 3: 0},
                                     {1: {"request_index": 4, "prompt_tokens": 1, "completion_tokens": 3}},
                                     {2: "rag", 3: "rag"}, enabled)

    def test_shadow_and_applied_choice_match_committed_contexts(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                row, = self.audit(enabled)
                self.assertEqual(row["selected_prefix_tokens"], 1023 if enabled else 0)
                self.assertEqual(row["gpu_ms"], 3)
                self.assertEqual(row["logical_pair_delta"], 1023 * 1023)
                self.assertEqual(row["decode"], [
                    {"scheduler_request": 1, "kv_tokens": 3, "request_index": 4, "commit_gap_ms": 4}
                ])

    def test_rejects_decoding_or_choice_with_discontinuous_context(self):
        with self.assertRaisesRegex(ValueError, "discontinuous decode cursor"):
            self.audit(True, lambda logs: [*logs[:-1], "W3-MIXED-DECODE 3 1:4"])
        with self.assertRaisesRegex(ValueError, "whole-budget prefill cursor"):
            self.audit(False, lambda logs: [logs[0], logs[1].replace("1023 1023", "1024 1023"), *logs[2:]])

    def test_rejects_missing_and_uncommitted_records(self):
        with self.assertRaisesRegex(ValueError, "decode context coverage"):
            self.audit(True, lambda logs: logs[:-1])
        with self.assertRaisesRegex(ValueError, "missing its committed step"):
            self.audit(True, lambda logs: [*logs, "W3-MIXED-DECODE 4 1:4"])
        with self.assertRaisesRegex(ValueError, "duplicate trace key 3"):
            self.audit(True, lambda logs: [*logs, logs[-1]])
        with self.assertRaisesRegex(ValueError, "incomplete mixed-prefill choices"):
            self.audit(True, lambda logs: ["W3-MIXED-CHOICES 1 1", *logs[1:]])

    def test_trace_configuration_preserves_schedules_and_cpu_placement(self):
        evidence = HERE.parent
        with TemporaryDirectory() as scratch, chdir(scratch):
            with tarfile.open(evidence / "20261124-w3-rag-admission/raw-results.tar.gz") as source:
                source.extract("arm.py", filter="data")
            with tarfile.open(evidence / "20261124-w3-separated-client/initial-results.tar.gz") as fixed:
                Path("fixed.json").write_bytes(fixed.extractfile("schedules.json").read())
            waves = Path("waves")
            waves.mkdir()
            with tarfile.open(evidence / "20261124-w3-chunk-pair/raw-results.tar.gz") as source:
                files = ("cells.json", *(f"gpu{gpu}/chunk-1024/{name}.json"
                                         for gpu in (6, 7) for name in ("wave-sync", "wave-50ms")))
                source.extractall(waves, members=[source.getmember(name) for name in files], filter="data")
            with tarfile.open(evidence / "20261124-w3-wide-prefill-order/raw-results.tar.gz") as previous:
                files = [row for row in previous.getmembers()
                         if row.name.startswith("tools/") or row.name in ("configure.py", "replay.py")]
                previous.extractall(members=files, filter="data")
                script = (HERE / "run.sh").read_text().split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
                exec(compile(script, "run.sh configure fragment", "exec"), {})
                with patch.object(sys, "path", [scratch, str(Path(scratch) / "tools"), *sys.path]):
                    runpy.run_path("configure.py")["configure"](Path("fixed.json"), waves)
                before = previous.extractfile("arm.py").read()
                anchor = b'        "--env", f"RILEY_W3_PREFILL_FIFO='
                self.assertEqual(before.count(anchor), 1)
                self.assertEqual(Path("arm.py").read_bytes(), before.replace(
                    anchor, b'        "--env", "RILEY_W3_RAG_STEPS=1",\n' + anchor))
                for filename in ("schedules.json", "cpu-placement.json"):
                    self.assertEqual(Path(filename).read_bytes(), previous.extractfile(filename).read())


if __name__ == "__main__":
    unittest.main()
