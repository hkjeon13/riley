from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
import socket
import tempfile
import sys
import unittest
from unittest.mock import MagicMock, patch


SCRIPT = Path(__file__).resolve().parents[1] / "run_serving_optimization.py"
SPEC = importlib.util.spec_from_file_location("serving_optimization", SCRIPT)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class Response:
    status = 200

    def __init__(self, payload, *, streaming=False):
        self.payload = payload
        self.streaming = streaming

    def read(self):
        return json.dumps(self.payload).encode()

    def __iter__(self):
        return iter(self.payload)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.request = {
            "prompt": "Hello", "prompt_token_ids": [10], "prompt_tokens": 1,
            "requested_output_tokens": 2, "temperature": 0, "top_p": 1,
            "repetition_penalty": 1, "cache_policy": "prefix-cache-off", "http_eos_policy": "natural-eos",
        }
        self.binding = {"source": {"git_commit": "abc", "git_dirty": False},
                        "environment": {"host": {"environment_id": "test-env"},
                                        "gpu": {"device_index": 0, "uuid": "GPU-test"}},
                        "workload": {"concurrency": 1, "sampling_id": "greedy", "prompt_tokens": 1, "output_tokens": 2},
                        "input_token_ids": [10], "generated_token_ids": [20, 30]}
        self.request_path = self.root / "request.json"
        self.binding_path = self.root / "binding.json"
        runner.write_json(self.request_path, self.request)
        runner.write_json(self.binding_path, self.binding)
        self.binary = self.root / "binary"
        self.binary.write_text("fake binary")
        lane = {"argv": [str(self.binary)], "env": {}, "port": 12345, "expected_output_text": "answer"}
        self.plan = {"source_root": str(self.root), "source_commit": "abc", "preflight_environment_id": "test-env",
                     "immutable_files": {str(p): runner.digest(p) for p in [self.request_path, self.binding_path, self.binary]},
                     "http_lanes": {"riley": dict(lane), "vllm": dict(lane)}}
        self.plan_path = self.root / "plan.json"
        runner.write_json(self.plan_path, self.plan)

    def validate(self):
        with patch.object(runner.subprocess, "check_output", side_effect=["abc\n", ""]):
            runner.validate_artifacts(self.plan, self.request_path, self.binding_path, self.request, self.binding)

    def test_preparation_never_launches_or_queries_gpu(self):
        with patch.object(runner.subprocess, "check_output", side_effect=["abc\n", ""]), \
             patch.object(runner.subprocess, "Popen") as launch, \
             patch.object(runner, "preflight") as preflight:
            code = runner.main(["--plan", str(self.plan_path), "--request", str(self.request_path),
                                "--binding", str(self.binding_path), "--output", str(self.root / "prepared")])
        self.assertEqual(code, 0)
        launch.assert_not_called()
        preflight.assert_not_called()
        receipt = json.loads((self.root / "prepared/preparation.json").read_text())
        self.assertFalse(receipt["measurement_started"])
        self.assertFalse(receipt["condition"]["canonical_qualification"])
        self.assertFalse(receipt["http_generated_token_ids_verified"])

    def test_changed_binary_rejected(self):
        self.binary.write_text("changed")
        with self.assertRaisesRegex(ValueError, "immutable artifact changed"):
            self.validate()

    def test_unbound_explicit_input_rejected(self):
        del self.plan["immutable_files"][str(self.binding_path)]
        with self.assertRaisesRegex(ValueError, "input is not immutable-bound"):
            self.validate()

    def test_dirty_source_rejected(self):
        with patch.object(runner.subprocess, "check_output", side_effect=["abc\n", " M owned.rs"]):
            with self.assertRaisesRegex(ValueError, "pinned clean snapshot"):
                runner.validate_artifacts(self.plan, self.request_path, self.binding_path, self.request, self.binding)

    def test_token_contract_and_environment_cannot_drift(self):
        self.binding["input_token_ids"] = [99]
        with self.assertRaisesRegex(ValueError, "token contract"):
            self.validate()
        self.binding["input_token_ids"] = [10]
        self.plan["preflight_environment_id"] = "legacy-env"
        with self.assertRaisesRegex(ValueError, "environment differs"):
            self.validate()

    def test_active_listener_rejected_and_time_wait_allowed(self):
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            port = listener.getsockname()[1]
            with self.assertRaises(OSError):
                runner.check_port(port)
            with socket.create_connection(("127.0.0.1", port)) as client:
                accepted, _ = listener.accept()
                accepted.close()
                client.recv(1)
        runner.check_port(port)

    def test_preflight_refuses_foreign_process_without_killing(self):
        sample = {"compute_pids": ["321"], "memory_used_mib": 100, "temperature_c": 30}
        with patch.object(runner, "gpu_snapshot", return_value=sample), \
             patch.object(runner.subprocess, "run") as host_check, \
             patch.object(runner.os, "killpg") as kill:
            with self.assertRaisesRegex(ValueError, "foreign CUDA"):
                runner.preflight(self.plan, self.binding, self.root)
        host_check.assert_not_called()
        kill.assert_not_called()
        self.assertEqual(json.loads((self.root / "cooldown.json").read_text()), [sample])

    def test_preflight_cools_and_records_gui_limits(self):
        warm = {"compute_pids": [], "memory_used_mib": 400, "temperature_c": 49}
        cool = {**warm, "temperature_c": 48}
        with patch.object(runner, "gpu_snapshot", side_effect=[warm, cool, cool]), \
             patch.object(runner.time, "sleep") as sleep, \
             patch.object(runner.subprocess, "run") as host_check:
            runner.preflight(self.plan, self.binding, self.root)
        sleep.assert_called_once_with(2)
        env = host_check.call_args.kwargs["env"]
        self.assertEqual(env["RILEY_PREFLIGHT_ENVIRONMENT_ID"], "test-env")
        self.assertNotIn("RILEY_MAX_IDLE_MEMORY_MIB", env)
        self.assertNotIn("RILEY_MAX_START_TEMPERATURE_C", env)
        self.assertEqual(runner.CONDITION["host_profile_id"], "rtx4090-ubuntu22-driver580-host-v3")
        self.assertFalse(json.loads((self.root / "live-condition.json").read_text())["canonical_qualification"])
        with self.assertRaisesRegex(ValueError, "512"):
            runner.check_idle({"compute_pids": [], "memory_used_mib": 513})

    def request_with(self, response, streaming):
        connection = MagicMock()
        connection.getresponse.return_value = response
        with patch.object(runner.http.client, "HTTPConnection", return_value=connection):
            result = runner.http_request(12345, "fixture", self.request, "answer", streaming)
        connection.close.assert_called_once()
        return result

    def test_nonstream_usage_text_and_finish_are_checked(self):
        payload = {"choices": [{"text": "answer", "finish_reason": "length"}],
                   "usage": {"prompt_tokens": 1, "completion_tokens": 2}}
        row = self.request_with(Response(payload), False)
        self.assertEqual(row["token_counts_validation"], "response-usage-exact")
        payload["usage"]["completion_tokens"] = 1
        with self.assertRaisesRegex(ValueError, "token counts"):
            self.request_with(Response(payload), False)
        payload["usage"]["completion_tokens"] = 2
        payload["choices"][0]["finish_reason"] = "stop"
        with self.assertRaisesRegex(ValueError, "length finish"):
            self.request_with(Response(payload), False)

    def stream(self, text="answer", finish="length", done=True):
        value = {"choices": [{"text": text, "finish_reason": finish, "index": 0}]}
        return Response([b"data: " + json.dumps(value).encode() + b"\n"] + ([b"data: [DONE]\n"] if done else []))

    def test_stream_requires_exact_text_done_and_length(self):
        row = self.request_with(self.stream(), True)
        self.assertEqual(row["event_unit"], "sse-text-event")
        self.assertEqual(row["token_ids_validation"], "unavailable-in-http-response")
        for response in [self.stream(text="wrong"), self.stream(finish="stop"), self.stream(done=False)]:
            with self.assertRaises(ValueError):
                self.request_with(response, True)

    def test_warmup_failure_stops_owned_server_and_never_measures(self):
        process = MagicMock()
        with patch.object(runner, "check_port"), patch.object(runner, "wait_ready"), \
             patch.object(runner.subprocess, "Popen", return_value=process) as launch, \
             patch.object(runner, "http_request", side_effect=[{}, {}, ValueError("warmup mismatch")]) as request, \
             patch.object(runner, "stop_owned_process") as stop:
            with self.assertRaisesRegex(ValueError, "warmup mismatch"):
                runner.run_http(self.plan, "riley", self.request, self.root, 1, 30, 10)
        stop.assert_called_once_with(process)
        self.assertTrue(launch.call_args.kwargs["start_new_session"])
        self.assertEqual(request.call_count, 3)
        self.assertFalse((self.root / "http-streaming.jsonl").exists())

    def test_failure_is_recorded_and_not_aggregated(self):
        output = self.root / "failed"
        with patch.object(runner, "validate_artifacts"), \
             patch.object(runner, "preflight", side_effect=ValueError("GPU busy")):
            with self.assertRaisesRegex(ValueError, "GPU busy"):
                runner.main(["--plan", str(self.plan_path), "--request", str(self.request_path),
                             "--binding", str(self.binding_path), "--output", str(output), "--measure"])
        self.assertFalse(json.loads((output / "completion.json").read_text())["completed"])
        self.assertTrue((output / "pair-01-riley/failure.json").exists())
        self.assertFalse((output / "summary.json").exists())

    def test_real_http_process_streaming_and_cleanup(self):
        server = self.root / "server.py"
        server.write_text(r'''import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args): pass
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b'{}')
    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        value = {"choices": [{"index": 0, "text": "answer", "finish_reason": "length"}],
                 "usage": {"prompt_tokens": 1, "completion_tokens": 2}}
        self.send_response(200); self.end_headers()
        if request['stream']:
            value.pop('usage')
            self.wfile.write(b'data: ' + json.dumps(value).encode() + b'\n\ndata: [DONE]\n\n')
        else:
            self.wfile.write(json.dumps(value).encode())
HTTPServer(('127.0.0.1', int(sys.argv[1])), Handler).serve_forever()
''')
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        lane = self.plan["http_lanes"]["riley"]
        lane["port"] = port
        lane["argv"] = [sys.executable, str(server), str(port)]
        rows = runner.run_http(self.plan, "riley", self.request, self.root, 2, 3, 5)
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(row["text_exact"] for row in rows))
        self.assertEqual(len((self.root / "warmups.jsonl").read_text().splitlines()), 2)
        self.assertEqual(len((self.root / "http-streaming.jsonl").read_text().splitlines()), 3)
        with self.assertRaises(OSError):
            socket.create_connection(("127.0.0.1", port), timeout=.2)
        runner.check_port(port)

    def test_aggregation_uses_request_and_wall_boundaries(self):
        rows = [{"started_ns": 0, "finished_ns": 10_000_000, "event_ns": [2_000_000]},
                {"started_ns": 20_000_000, "finished_ns": 40_000_000, "event_ns": [25_000_000]}]
        summary = runner.summarize(rows, 2)
        self.assertEqual(summary["e2e_ms"], {"median": 15, "p95": 20, "p99": 20})
        self.assertEqual(summary["output_tokens_per_wall_second"], 100)
        self.assertAlmostEqual(summary["output_tokens_per_request_service_second"], 4000 / 30)


if __name__ == "__main__":
    unittest.main()
