from __future__ import annotations

import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import MagicMock, patch


SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("serving_concurrency", SCRIPTS / "run_serving_concurrency.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.source = self.root / "source"
        self.source.mkdir()
        self.request = {"prompt": "Hello", "prompt_token_ids": [10] * 128, "prompt_tokens": 128,
                        "requested_output_tokens": 32, "temperature": 0, "top_p": 1,
                        "cache_policy": "prefix-cache-off", "http_eos_policy": "natural-eos"}
        self.request_path = self.write("request.json", self.request)
        self.qualification_path = self.write("qualification.json", {"passed": True, "source_commit": "abc", "source_clean": True})
        self.binding = {"source": {"git_commit": "abc", "git_dirty": False,
                                  "correctness_gate_id": "g04-vllm-smol-p128-v1",
                                  "correctness_report_sha256": runner.shared.digest(self.qualification_path)},
                        "environment": {"host": {"environment_id": "test-env"}},
                        "workload": {"concurrency": 1, "sampling_id": "greedy", "prompt_tokens": 128,
                                     "output_tokens": 32, "workload_id": "g04-c1-p128-o32"},
                        "input_token_ids": [10] * 128, "generated_token_ids": [20] * 32}
        self.binding_path = self.write("binding.json", self.binding)
        self.riley = self.root / "riley"
        self.vllm = self.root / "vllm"
        self.riley.write_text("qualified binary")
        self.vllm.write_text("qualified vllm entry")
        service = self.source / "crates/riley-server/src/service.rs"
        service.parent.mkdir(parents=True)
        service.write_text("impl Default for ServerConfig { worker_threads: 8, }")
        self.parent = {"measurement_mode": "http", "source_root": str(self.source), "source_commit": "abc",
                       "request_path": str(self.request_path), "binding_path": str(self.binding_path),
                       "qualification_path": str(self.qualification_path), "preflight_environment_id": "test-env",
                       "immutable_files": {str(p): runner.shared.digest(p) for p in
                                           (self.request_path, self.binding_path, self.qualification_path, self.riley, self.vllm, service)},
                       "http_lanes": {
                           "riley": {"argv": [str(self.riley), "serve", "--bind", "127.0.0.1:19341",
                                                "--max-active-sequences", "1", "--batch-token-budget", "128",
                                                "--prefill-chunk-tokens", "128"],
                                     "env": {}, "port": 19341, "expected_output_text": "answer"},
                           "vllm": {"argv": [str(self.vllm), "serve", "/model", "--port", "19342",
                                              "--max-num-seqs", "1", "--max-num-batched-tokens", "128"],
                                    "env": {}, "port": 19342, "expected_output_text": "answer"}}}
        self.parent_path = self.write("parent.json", self.parent)
        self.plan = {"schema_version": runner.PLAN_SCHEMA, "parent_c1_plan": runner.evidence(self.parent_path),
                     "immutable_files": {str(p.resolve()): runner.shared.digest(p) for p in
                                         (Path(runner.__file__), Path(runner.shared.__file__), self.parent_path)},
                     "workload": {"id": "queued-c2-p128-o32-screen-v1", "offered_concurrency": 2,
                                  "arrival_policy": "closed-loop-refill", "retained_requests_per_process": 20,
                                  "warmups_per_worker_per_transport": 2, "pairs": 1, "purpose": "screening"},
                     "base_environment": {"PATH": "/usr/bin", "HOME": str(self.root)},
                     "startup_timeout_seconds": 60, "request_timeout_seconds": 120,
                     "http_lanes": copy.deepcopy(self.parent["http_lanes"]),
                     "vllm_runtime": {"engine_version": "0.27.1", "enforce_eager": False,
                                      "enable_chunked_prefill": True, "compilation_mode": "VLLM_COMPILE",
                                      "cudagraph_mode": "FULL_AND_PIECEWISE"}}
        riley, vllm = self.plan["http_lanes"]["riley"], self.plan["http_lanes"]["vllm"]
        riley.update(active_capacity=1, waiting_capacity=64, http_workers=8, token_budget=128)
        riley["argv"].extend(["--max-waiting-requests", "64"])
        vllm.update(active_capacity=2, token_budget=256)
        vllm["argv"][vllm["argv"].index("--max-num-seqs") + 1] = "2"
        vllm["argv"][vllm["argv"].index("--max-num-batched-tokens") + 1] = "256"
        self.plan_path = self.write("plan.json", self.plan)

    def write(self, name, value):
        path = self.root / name
        path.write_text(json.dumps(value))
        return path

    def validate(self):
        with patch.object(runner.shared.subprocess, "check_output", side_effect=["abc\n", ""]):
            return runner.validate_manifest(self.plan, self.request_path, self.binding_path, self.request, self.binding)

    def test_parent_is_validated_without_reinterpreting_c1(self):
        original = copy.deepcopy(self.parent)
        with patch.object(runner.shared, "validate_artifacts", wraps=runner.shared.validate_artifacts) as validate:
            self.assertEqual(self.validate(), original)
            self.assertEqual(validate.call_args.args[0], original)
            self.assertEqual(validate.call_args.args[-1]["workload"]["concurrency"], 1)
        self.assertEqual(runner.read(self.parent_path), original)

    def test_prepare_only_does_not_start_server_or_gpu_probe(self):
        with patch.object(runner.shared.subprocess, "check_output", side_effect=["abc\n", ""]), \
             patch.object(runner.subprocess, "Popen") as launch, patch.object(runner, "preflight") as preflight:
            self.assertEqual(runner.main(["--plan", str(self.plan_path), "--request", str(self.request_path),
                                         "--binding", str(self.binding_path), "--output", str(self.root / "prepared")]), 0)
        launch.assert_not_called()
        preflight.assert_not_called()
        receipt = runner.read(self.root / "prepared/preparation.json")
        self.assertFalse(receipt["measurement_started"])
        self.assertEqual(receipt["workload"]["offered_concurrency"], 2)
        self.assertEqual(receipt["parent_qualification_use"], "c1 source/token reference only; no concurrency evidence")
        with self.assertRaises(FileExistsError), patch.object(runner.shared.subprocess, "check_output", side_effect=["abc\n", ""]):
            runner.main(["--plan", str(self.plan_path), "--request", str(self.request_path),
                         "--binding", str(self.binding_path), "--output", str(self.root / "prepared")])

    def test_changed_source_binary_parent_or_runner_is_rejected(self):
        self.riley.write_text("changed")
        with self.assertRaisesRegex(ValueError, "immutable artifact changed"):
            self.validate()
        self.riley.write_text("qualified binary")
        self.plan["immutable_files"][str(Path(runner.__file__).resolve())] = "0" * 64
        with self.assertRaisesRegex(ValueError, "immutable artifact changed"):
            self.validate()

    def test_plan_byte_change_during_lane_rejects_identical_parsed_json(self):
        def measured(*args):
            self.plan_path.write_text(self.plan_path.read_text() + "\n")
            return {}
        output = self.root / "measurement"
        with patch.object(runner.shared.subprocess, "check_output", side_effect=["abc\n", ""] * 2), \
             patch.object(runner, "preflight"), patch.object(runner, "run_lane", side_effect=measured):
            with self.assertRaisesRegex(ValueError, "plan changed during lane"):
                runner.main(["--plan", str(self.plan_path), "--request", str(self.request_path),
                             "--binding", str(self.binding_path), "--output", str(output), "--measure"])
        self.assertFalse(runner.read(output / "completion.json")["completed"])
        self.assertEqual(runner.read(self.plan_path), self.plan)

    def test_new_scope_and_counts_are_required(self):
        for fields in ({"offered_concurrency": 9}, {"offered_concurrency": True}, {"pairs": 0},
                       {"retained_requests_per_process": 1}, {"warmups_per_worker_per_transport": 0},
                       {"purpose": "tail-study"}, {"arrival_policy": "open-loop"}, {"id": "g04-c1-p128-o32"}):
            with self.subTest(fields=fields):
                before = copy.deepcopy(self.plan["workload"])
                self.plan["workload"].update(fields)
                with self.assertRaises(ValueError): self.validate()
                self.plan["workload"] = before
        self.plan["schema_version"] = "riley.g04.measurement-preparation.v1"
        with self.assertRaisesRegex(ValueError, "cannot be relabelled"): self.validate()

    def test_capacity_command_reference_and_environment_drift_rejected(self):
        original = copy.deepcopy(self.plan)
        mutations = [lambda p: p["http_lanes"]["riley"].update(active_capacity=2),
                     lambda p: p["http_lanes"]["riley"].update(waiting_capacity=4),
                     lambda p: p["http_lanes"]["vllm"].update(active_capacity=1),
                     lambda p: p["http_lanes"]["vllm"]["argv"].extend(["--enforce-eager"]),
                     lambda p: p["http_lanes"]["vllm"].update(expected_output_text="new answer"),
                     lambda p: p["base_environment"].update(RILEY_OWNED_GRAPH_PROFILE="1"),
                     lambda p: p["base_environment"].update(LD_PRELOAD="unqualified.so"),
                     lambda p: p["http_lanes"]["vllm"]["env"].update(UNKNOWN="1")]
        for mutation in mutations:
            self.plan = copy.deepcopy(original)
            mutation(self.plan)
            with self.assertRaises(ValueError): self.validate()

    def test_vllm_actual_startup_matches_and_rejects_eager_or_capacity_drift(self):
        path = self.root / "server.log"
        args = {"max_num_seqs": 2, "max_num_batched_tokens": 256, "max_model_len": 160,
                "dtype": "bfloat16", "gpu_memory_utilization": 0.3, "enable_prefix_caching": False}
        record = "Initializing a V1 LLM engine (v0.27.1) with config: enforce_eager=False, enable_chunked_prefill=True, enable_prefix_caching=False, CompilationMode.VLLM_COMPILE, CUDAGraphMode.FULL_AND_PIECEWISE"
        text = "INFO non-default args: " + repr(args) + "\nINFO " + record
        path.write_text(text)
        self.assertTrue(runner.validate_vllm_startup(path, self.plan["http_lanes"]["vllm"], self.plan["vllm_runtime"])["validated"])
        for invalid in (text.replace("enforce_eager=False", "enforce_eager=True"),
                        text.replace("'max_num_seqs': 2", "'max_num_seqs': 1"),
                        text.replace("FULL_AND_PIECEWISE", "NONE"), text + "\nINFO " + record):
            path.write_text(invalid)
            with self.assertRaises(ValueError):
                runner.validate_vllm_startup(path, self.plan["http_lanes"]["vllm"], self.plan["vllm_runtime"])

    def test_worker_phase_exception_always_cleans_owned_server(self):
        directory = self.root / "lane"
        directory.mkdir()
        process = MagicMock(pid=98765, returncode=0)
        with patch.object(runner.shared, "check_port"), patch.object(runner.shared, "wait_ready"), \
             patch.object(runner.subprocess, "Popen", return_value=process) as launch, \
             patch.object(runner, "run_phase", side_effect=RuntimeError("worker failed")), \
             patch.object(runner.shared, "stop_owned_process") as stop:
            with self.assertRaisesRegex(RuntimeError, "worker failed"):
                runner.run_lane(self.plan, self.parent, "riley", self.request, directory)
        stop.assert_called_once_with(process)
        self.assertEqual(launch.call_args.kwargs["env"], self.plan["base_environment"])
        self.assertTrue(runner.read(directory / "process-exit.json")["owned_session_cleanup_finished"])

    def test_warmup_must_observe_offered_concurrency_before_retained_requests(self):
        directory = self.root / "lane"
        directory.mkdir()
        process = MagicMock(pid=98765, returncode=0)
        accounting = {"phase": "warmup-nonstream", "completed": True, "observed_max_request_in_flight": 1}
        with patch.object(runner.shared, "check_port"), patch.object(runner.shared, "wait_ready"), \
             patch.object(runner.subprocess, "Popen", return_value=process), \
             patch.object(runner, "run_phase", return_value=([], accounting)) as phase, \
             patch.object(runner.shared, "stop_owned_process") as stop:
            with self.assertRaisesRegex(ValueError, "warmup-nonstream did not exercise"):
                runner.run_lane(self.plan, self.parent, "riley", self.request, directory)
        self.assertEqual(phase.call_count, 1)
        stop.assert_called_once_with(process)
        self.assertEqual(runner.read(directory / "warmup-nonstream-accounting.json"), accounting)

    def test_startup_receipt_pins_snapshot_while_live_log_grows(self):
        directory = self.root / "lane"
        directory.mkdir()
        process = MagicMock(pid=98765, returncode=0)
        def ready(*args):
            (directory / "server.log").write_text("startup config\n")
        def cleanup(*args):
            with (directory / "server.log").open("a") as log:
                log.write("shutdown after requests\n")
        def validate(path, *args):
            return {"validated": True, "log": runner.evidence(path)}
        with patch.object(runner.shared, "check_port"), patch.object(runner.shared, "wait_ready", side_effect=ready), \
             patch.object(runner.subprocess, "Popen", return_value=process), \
             patch.object(runner, "validate_vllm_startup", side_effect=validate), \
             patch.object(runner, "run_phase", side_effect=RuntimeError("stop fixture")), \
             patch.object(runner.shared, "stop_owned_process", side_effect=cleanup):
            with self.assertRaisesRegex(RuntimeError, "stop fixture"):
                runner.run_lane(self.plan, self.parent, "vllm", self.request, directory)
        receipt = runner.read(directory / "vllm-startup.json")
        self.assertEqual(receipt["log"], runner.evidence(directory / "vllm-startup.log"))
        self.assertNotEqual(receipt["log"]["sha256"], runner.shared.digest(directory / "server.log"))


class PhaseTests(unittest.TestCase):
    def request(self, delay=.012):
        start = time.perf_counter_ns()
        time.sleep(delay)
        return {"started_ns": start, "finished_ns": time.perf_counter_ns(), "event_ns": [start + 1000], "text_exact": True}

    def test_workers_overlap_refill_and_cover_unique_indices(self):
        rows, accounting = runner.run_phase(self.request, 4, 20, "retained")
        self.assertEqual([row["index"] for row in rows], list(range(20)))
        self.assertEqual(len({row["request_id"] for row in rows}), 20)
        self.assertEqual(accounting["observed_max_request_in_flight"], 4)
        self.assertTrue(accounting["completed"])
        self.assertLessEqual(accounting["phase_started_ns"], min(row["started_ns"] for row in rows))
        summary = runner.summarize(rows[::-1], accounting, 32)
        expected = 640e9 / (max(r["finished_ns"] for r in rows) - min(r["started_ns"] for r in rows))
        self.assertEqual(summary["output_tokens_per_wall_second"], expected)
        self.assertNotIn("output_tokens_per_request_service_second", summary)

    def test_warmups_are_per_worker_and_separate(self):
        rows, accounting = runner.run_phase(self.request, 4, 12, "warmup-stream", per_worker=True)
        self.assertTrue(accounting["completed"])
        self.assertTrue(all(row["warmup"] for row in rows))
        self.assertEqual([sum(row["worker_id"] == worker for row in rows) for worker in range(4)], [3] * 4)
        with self.assertRaisesRegex(ValueError, "non-retained"):
            runner.summarize(rows, accounting, 32)

    def test_request_failure_stops_refill_drains_and_accounts(self):
        lock = threading.Lock()
        calls = 0
        def request():
            nonlocal calls
            with lock:
                calls += 1
                index = calls
            if index == 1:
                time.sleep(.003)
                raise ValueError("output mismatch")
            return self.request(.015)
        rows, accounting = runner.run_phase(request, 4, 40, "retained")
        self.assertFalse(accounting["completed"])
        self.assertEqual(accounting["failed"], 1)
        self.assertLessEqual(accounting["attempted"], 4)
        self.assertEqual(accounting["not_started"] + accounting["attempted"], 40)
        self.assertEqual(len(rows), calls)
        self.assertEqual(next(row for row in rows if row["status"] == "failed")["error"], "output mismatch")

    def test_common_wall_uses_extents_and_rejects_unobserved_concurrency(self):
        rows = [{"status": "success", "phase": "retained", "started_ns": 1000, "finished_ns": 4000, "event_ns": [1200]},
                {"status": "success", "phase": "retained", "started_ns": 2000, "finished_ns": 3000, "event_ns": [2100]}]
        phase = {"completed": True, "offered_concurrency": 2, "observed_max_request_in_flight": 2}
        result = runner.summarize(rows[::-1], phase, 32)
        self.assertEqual(result["retained_wall_ns"], 3000)
        phase["observed_max_request_in_flight"] = 1
        with self.assertRaisesRegex(ValueError, "never observed"): runner.summarize(rows, phase, 32)

    def test_total_request_watchdog_breaks_trickling_peer_and_records_timeout(self):
        disconnected = threading.Event()
        def request():
            # A socket read timeout alone would never fire while a peer trickles.
            self.assertTrue(disconnected.wait(timeout=2))
            raise ConnectionError("owned server stopped")
        began = time.monotonic()
        rows, accounting = runner.run_phase(request, 2, 10, "retained", request_timeout=.04,
                                            abort_owned=disconnected.set)
        self.assertLess(time.monotonic() - began, 1)
        self.assertFalse(accounting["completed"])
        self.assertEqual(accounting["attempted"], 2)
        self.assertEqual(accounting["failed"], 2)
        self.assertTrue(accounting["watchdog_timed_out_request_indices"])
        self.assertEqual(len(rows), 2)

    def test_unexpected_worker_error_is_not_lost(self):
        with patch.object(runner.threading.Barrier, "wait", side_effect=threading.BrokenBarrierError):
            rows, accounting = runner.run_phase(self.request, 2, 10, "retained")
        self.assertFalse(accounting["completed"])
        self.assertTrue(accounting["worker_errors"])
        self.assertEqual(accounting["not_started"], 10)
        self.assertEqual(rows, [])

    def test_real_loopback_streaming_parity_at_concurrency(self):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args): pass
            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                time.sleep(.01)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                row = {"choices": [{"index": 0, "text": "answer", "finish_reason": "length"}],
                       "usage": {"prompt_tokens": 128, "completion_tokens": 32}}
                self.wfile.write(b"data: " + json.dumps(row).encode() + b"\n\ndata: [DONE]\n\n")
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            request = {"prompt": "Hello", "prompt_tokens": 128, "requested_output_tokens": 32}
            rows, accounting = runner.run_phase(lambda: runner.shared.http_request(server.server_port, "test", request, "answer", True),
                                                4, 12, "retained")
            self.assertTrue(accounting["completed"])
            self.assertEqual(accounting["observed_max_request_in_flight"], 4)
            self.assertTrue(all(row["token_counts_validation"] == "response-usage-exact" for row in rows))
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
