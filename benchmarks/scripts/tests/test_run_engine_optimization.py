from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch


SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("engine_optimization", SCRIPTS / "run_engine_optimization.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)
sys.path.pop(0)


class EngineRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.binding = {"workload": {"concurrency": 1, "prompt_tokens": 128, "output_tokens": 32,
                                   "sampling_id": "greedy", "model_id": "model", "model_revision": "rev",
                                   "dtype": "bf16", "warmups": 5, "measured_iterations": 30},
                        "input_token_ids": [10] * 128, "generated_token_ids": [20] * 32}
        w = self.binding["workload"]
        self.row = {"trial_index": 1, "status": "success", "failure_count": 0,
                    "model_id": "model", "model_revision": "rev", "dtype": "bf16", "warm_state": "warm",
                    "environment_id": "legacy-v1", "metrics": {"batch_wall_ms": 40},
                    "workload": {key: w[key] for key in ("concurrency", "prompt_tokens", "output_tokens", "sampling_id")},
                    "requests": [{"status": "success", "prompt_tokens": 128, "requested_output_tokens": 32,
                                  "generated_tokens": 32, "prompt_token_ids_sha256": runner.shared.token_digest(self.binding["input_token_ids"]),
                                  "generated_token_ids_sha256": runner.shared.token_digest(self.binding["generated_token_ids"]),
                                  "ttft_ms": 7, "mean_tpot_ms": 1, "end_to_end_ms": 38}]}
        self.row["workload"]["warm_state"] = "warm"
        self.rows = [dict(copy.deepcopy(self.row), trial_index=index) for index in range(1, 31)]

    def test_all_tokens_checked_and_legacy_raw_not_rewritten(self):
        before = copy.deepcopy(self.rows)
        normalized, receipt = runner.validate_vllm_rows(self.rows, self.binding)
        self.assertEqual(self.rows, before)
        self.assertEqual(len(normalized), 30)
        self.assertTrue(receipt["tokens_exact"])
        self.assertEqual(receipt["raw_environment_ids"], ["legacy-v1"])
        self.assertFalse(receipt["raw_canonical_eligible"])
        self.assertEqual(receipt["aggregate_output_tokens_per_second"], 800)

    def test_missing_duplicated_failed_or_wrong_token_samples_rejected(self):
        mutations = [lambda rows: rows.pop(),
                     lambda rows: rows[-1].update(trial_index=1),
                     lambda rows: rows[-1].update(failure_count=1),
                     lambda rows: rows[-1]["requests"][0].update(generated_tokens=31),
                     lambda rows: rows[-1]["requests"][0].update(generated_token_ids_sha256="wrong"),
                     lambda rows: rows[-1].update(model_revision="different")]
        for mutate in mutations:
            rows = copy.deepcopy(self.rows)
            mutate(rows)
            with self.assertRaises(ValueError):
                runner.validate_vllm_rows(rows, self.binding)

    def test_nonfinite_or_impossible_times_rejected(self):
        for args in [(float("nan"), 1, 40), (7, float("inf"), 40), (7, -1, 40), (41, 1, 40), (7, 2, 40)]:
            with self.assertRaises(ValueError):
                runner.validate_request_times(*args, 32)

    def test_summary_keeps_true_metrics_and_throughput_boundaries(self):
        normalized, extra = runner.validate_vllm_rows(self.rows, self.binding)
        summary = runner.summarize(normalized, 32, extra)
        self.assertEqual(summary["ttft_ms"], {"median": 7, "p95": 7, "p99": 7})
        self.assertEqual(summary["tpot_ms"]["median"], 1)
        self.assertEqual(summary["aggregate_output_tokens_per_second"], 800)
        self.assertAlmostEqual(summary["output_tokens_per_request_service_second"], 32000 / 38)

    def plan_fixture(self):
        binary = self.root / "riley"
        binary.write_text("pinned")
        python = Path(sys.executable).resolve()
        riley = [str(binary), "--warmups", "5", "--measured-iterations", "30", "--git-commit", "abc",
                 "--git-dirty", "false", "--executable-sha256", runner.shared.digest(binary),
                 "--environment-id", "actual-v2", "--concurrency", "1", "--prompt-tokens", "128", "--output-tokens", "32"]
        vllm = [str(python), "--warm-state", "warm", "--concurrency", "1", "--prompt-tokens", "128", "--output-tokens", "32"]
        return {"source_root": str(self.root), "source_commit": "abc", "preflight_environment_id": "actual-v2",
                "immutable_files": {str(binary): runner.shared.digest(binary), str(python): runner.shared.digest(python)},
                "reference_checker_python": str(python),
                "engine_lanes": {"riley": {"argv": riley}, "vllm": {"argv": vllm}}}

    def test_engine_plan_requires_pinned_binary_and_matching_arguments(self):
        plan = self.plan_fixture()
        with patch.object(runner.shared, "validate_artifacts"):
            runner.validate_plan(plan, self.root / "r", self.root / "b", {}, self.binding)
            plan["engine_lanes"]["riley"]["argv"][2] = "4"
            with self.assertRaisesRegex(ValueError, "repetition"):
                runner.validate_plan(plan, self.root / "r", self.root / "b", {}, self.binding)
            plan["engine_lanes"]["riley"]["argv"][2] = "5"
            del plan["immutable_files"][str(Path(sys.executable).resolve())]
            with self.assertRaisesRegex(ValueError, "not immutable-bound"):
                runner.validate_plan(plan, self.root / "r", self.root / "b", {}, self.binding)

    def input_files(self):
        plan = self.plan_fixture()
        paths = [self.root / name for name in ("plan.json", "request.json", "binding.json")]
        for path, value in zip(paths, (plan, {}, self.binding)):
            runner.shared.write_json(path, value)
        return ["--plan", str(paths[0]), "--request", str(paths[1]), "--binding", str(paths[2])]

    def test_preparation_does_not_launch_or_query_gpu(self):
        args = self.input_files() + ["--output", str(self.root / "prepared")]
        with patch.object(runner, "validate_plan"), patch.object(runner, "run_lane") as lane, \
             patch.object(runner.shared, "preflight") as preflight:
            self.assertEqual(runner.main(args), 0)
        lane.assert_not_called()
        preflight.assert_not_called()
        receipt = json.loads((self.root / "prepared/preparation.json").read_text())
        self.assertFalse(receipt["measurement_started"])
        self.assertFalse(receipt["condition"]["canonical_qualification"])
        self.assertEqual(receipt["condition"]["host_environment_id"], "actual-v2")

    def test_failure_saved_without_summary(self):
        args = self.input_files() + ["--output", str(self.root / "failed"), "--measure"]
        with patch.object(runner, "validate_plan"), patch.object(runner.shared, "preflight"), \
             patch.object(runner, "run_lane", side_effect=ValueError("wrong tokens")):
            with self.assertRaisesRegex(ValueError, "wrong tokens"):
                runner.main(args)
        self.assertFalse((self.root / "failed/summary.json").exists())
        self.assertTrue((self.root / "failed/pair-01-riley/failure.json").exists())
        self.assertFalse(json.loads((self.root / "failed/completion.json").read_text())["completed"])

    def test_failed_child_is_stopped_and_no_checker_runs(self):
        plan = self.plan_fixture()
        process = MagicMock()
        process.wait.return_value = 9
        process.pid = 123
        with patch.object(runner.subprocess, "Popen", return_value=process), \
             patch.object(runner.shared, "stop_owned_process") as stop, \
             patch.object(runner.subprocess, "run") as checker:
            with self.assertRaises(runner.subprocess.CalledProcessError):
                runner.run_lane(plan, "riley", 1, self.binding, self.root / "binding.json", self.root, 5)
        stop.assert_called_once_with(process)
        checker.assert_not_called()

    def test_five_alternating_pairs_complete_with_envelope_and_hashes(self):
        args = self.input_files() + ["--output", str(self.root / "pairs"), "--measure"]
        normalized, extra = runner.validate_vllm_rows(self.rows, self.binding)
        summary = runner.summarize(normalized, 32, extra)
        roles = []
        def lane(plan, role, index, *args):
            roles.append((index, role))
            return copy.deepcopy(summary)
        with patch.object(runner, "validate_plan"), patch.object(runner.shared, "preflight"), \
             patch.object(runner, "run_lane", side_effect=lane):
            runner.main(args)
        self.assertEqual(roles, [(1, "riley"), (1, "vllm"), (2, "vllm"), (2, "riley"),
                                 (3, "riley"), (3, "vllm"), (4, "vllm"), (4, "riley"), (5, "riley"), (5, "vllm")])
        result = json.loads((self.root / "pairs/summary.json").read_text())
        self.assertEqual(result["paired_ratio_medians"]["tpot_ms"], 1)
        self.assertTrue((self.root / "pairs/raw-sha256.json").exists())


if __name__ == "__main__":
    unittest.main()
