"""Orchestration failure boundaries use real CPU child processes, never a GPU."""
from __future__ import annotations

import contextlib
import copy
import importlib.util
import io
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

SOURCE = Path(__file__).resolve().parents[2] / "live/compare_runner.py"
SPEC = importlib.util.spec_from_file_location("serving_compare_runner", SOURCE)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)

SCENARIOS = {"closed-c1": ("short", "--requests 2 --warmup 0 --concurrency 1")}


def valid_summary():
    expected = runner.workload_arguments(SCENARIOS["closed-c1"][1])
    result = {"config": {**expected, "address": "127.0.0.1:18300", "model": "m"},
              "requests": 2, "ok": 2, "errors": 0, "rejected_429": 0, "other_failures": [],
              "warmup_requests": 0, "warmup_ok": 0, "warmup_errors": 0,
              "wall_s": 1, "output_tok_s": 4, "requests_s": 2,
              "prompt_tokens_mean": 20, "completion_tokens_mean": 2, "itl_p99_ms": 1}
    result.update({f"{name}_{q}_ms": 1 for name in ("ttft", "tpot", "e2e") for q in ("p50", "p95", "p99")})
    return result


class CompareTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="riley comparison with spaces ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.output = self.root / "result.json"
        self.args = runner.parser(SCENARIOS).parse_args(
            [str(self.root), "new-run", "model with spaces", "0", "candidate",
             "--cooldown-seconds", "0", "--shutdown-timeout", "0.2"]
        )

    def test_model_paths_and_json_are_single_argv_items(self):
        self.args.riley_short = '--example "value with spaces"'
        locations = runner.paths(self.args)
        argv = runner.server_command(self.args, locations, "r", "short")
        self.assertEqual(argv[argv.index("--model") + 1], str(self.root / "models/model with spaces"))
        self.assertEqual(argv[argv.index("--example") + 1], "value with spaces")
        parsed = runner.workload_arguments('--concurrency 8 --extra \'{"logprobs": true, "top_logprobs": 5}\'')
        self.assertTrue(json.loads(parsed["extra"])["logprobs"])

    def test_result_rejects_missing_counts_config_nonfinite_and_partial_data(self):
        expected = runner.workload_arguments(SCENARIOS["closed-c1"][1])
        mutations = [
            lambda d: d.update(ok=1), lambda d: d.update(requests=1),
            lambda d: d.update(errors=1), lambda d: d.update(rejected_429=1),
            lambda d: d.update(warmup_errors=1),
            lambda d: d.update(ttft_p95_ms=float("nan")),
            lambda d: d.update(output_tok_s=float("inf")),
            lambda d: d.update(completion_tokens_mean=1000),
            lambda d: d["config"].update(model="wrong"),
            lambda d: d["config"].update(warmup=32),
            lambda d: d.pop("tpot_p50_ms"),
        ]
        self.output.write_text(json.dumps(valid_summary()))
        runner.validate_result(self.output, expected, "127.0.0.1:18300")
        for change in mutations:
            value = copy.deepcopy(valid_summary())
            change(value)
            self.output.write_text(json.dumps(value))
            with self.subTest(change=change):
                with self.assertRaises(ValueError):
                    runner.validate_result(self.output, expected, "127.0.0.1:18300")
        for raw in ('{', '{"config":{},"config":{}}', 'null'):
            self.output.write_text(raw)
            with self.assertRaises(ValueError):
                runner.validate_result(self.output, expected, "127.0.0.1:18300")
        self.output.unlink()
        with self.assertRaises(FileNotFoundError):
            runner.validate_result(self.output, expected, "127.0.0.1:18300")

    def install_loader(self, raw, exit_code=0):
        path = self.root / "fake load with spaces.py"
        path.write_text("import pathlib,sys\n"
                        "out=pathlib.Path(sys.argv[sys.argv.index('--out')+1])\n"
                        f"out.write_text({raw!r})\n"
                        f"raise SystemExit({exit_code})\n")
        self.args.load_script = path

    def run_with_cpu_loader(self):
        process = mock.Mock(returncode=0)
        process.poll.return_value = None
        with mock.patch.object(runner, "start_server", return_value=process), \
             mock.patch.object(runner, "stop_process") as stop, \
             contextlib.redirect_stdout(io.StringIO()) as output:
            try:
                result = runner.run(self.args, SCENARIOS)
            finally:
                self.stdout = output.getvalue()
                self.stops = stop.call_count
        return result

    def test_all_declared_cells_are_validated_before_success(self):
        self.install_loader(json.dumps(valid_summary()))
        result = self.run_with_cpu_loader()
        self.assertEqual(result["status"], "passed")
        self.assertEqual([cell["tag"] for cell in result["cells"]], ["r1", "v1", "v2", "r2"])
        loads = [row for row in result["commands"] if row["kind"] == "load"]
        self.assertEqual(len(loads), 4)
        self.assertEqual(loads[0]["argv"][1], str(self.args.load_script))
        self.assertIn("--out", loads[0]["argv"])
        self.assertIn("COMPARE_DONE", self.stdout)
        with self.assertRaises(FileExistsError):
            self.run_with_cpu_loader()

    def test_nonzero_child_cannot_qualify_even_with_valid_json(self):
        self.install_loader(json.dumps(valid_summary()), exit_code=7)
        with self.assertRaisesRegex(runner.ComparisonError, "exited 7"):
            self.run_with_cpu_loader()
        self.assertNotIn("COMPARE_DONE", self.stdout)
        manifest = json.loads((runner.paths(self.args)["out"] / "run.json").read_text())
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(manifest["cells"], [])
        self.assertGreaterEqual(self.stops, 1)

    def test_zero_exit_with_truncated_json_fails_and_retains_logs(self):
        self.install_loader('{')
        with self.assertRaises(ValueError):
            self.run_with_cpu_loader()
        out = runner.paths(self.args)["out"]
        self.assertEqual(json.loads((out / "run.json").read_text())["status"], "failed")
        self.assertTrue((out / "r1-closed-c1.load.log").is_file())
        self.assertNotIn("COMPARE_DONE", self.stdout)

    def test_invalid_scenarios_order_and_timeouts_do_not_start_a_server(self):
        for key, value in (("scenarios", "closed-c1,closed-c1"), ("scenarios", "missing"),
                           ("order", "rX"), ("label", "../old"),
                           ("scenario_timeout", float("nan"))):
            args = copy.copy(self.args)
            setattr(args, key, value)
            with self.subTest(key=key, value=value):
                with self.assertRaises(runner.ComparisonError):
                    runner.validate_arguments(args, SCENARIOS)

    @unittest.skipUnless(os.name == "posix", "process-group lifecycle is POSIX-specific")
    def test_readiness_timeout_reaps_owned_process(self):
        processes = []
        popen = runner.subprocess.Popen
        def spawn(*args, **kwargs):
            child = popen(*args, **kwargs)
            processes.append(child)
            return child
        with mock.patch.object(runner.subprocess, "Popen", side_effect=spawn), \
             mock.patch.object(runner.urllib.request, "urlopen", side_effect=OSError("not ready")):
            with self.assertRaisesRegex(runner.ComparisonError, "readiness timeout"):
                runner.start_server([sys.executable, "-c", "import time; time.sleep(30)"],
                                    dict(os.environ), self.root / "server.log", "http://unused", .15, .2)
        self.assertIsNotNone(processes[0].returncode)

    @unittest.skipUnless(os.name == "posix", "process-group lifecycle is POSIX-specific")
    def test_load_timeout_stops_descendant_heartbeat_and_reaps_leader(self):
        heartbeat = self.root / "heartbeat"
        code = "import pathlib,time; p=pathlib.Path(" + repr(str(heartbeat)) + "); "
        code += "exec('while True:\\n p.write_text(str(time.monotonic()))\\n time.sleep(.01)')"
        parent = "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c'," + repr(code) + "]); time.sleep(30)"
        with self.assertRaisesRegex(runner.ComparisonError, "load timeout"):
            runner.run_load([sys.executable, "-c", parent], dict(os.environ), self.root / "load.log", .4, .2)
        self.assertTrue(heartbeat.is_file())
        before = heartbeat.read_text()
        time.sleep(.1)
        self.assertEqual(heartbeat.read_text(), before)
