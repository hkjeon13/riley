"""Check the fleet CLI's transport and schedule accounting using two local test replicas."""

import json
import re
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from collections import Counter
from contextlib import ExitStack
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import agent_fleet_replay
import replica_workflow
from agent_workflow import fixture


class TestReplica(BaseHTTPRequestHandler):
    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        instruction = next(
            m["content"]
            for m in req["messages"]
            if m["role"] == "user" and m["content"].startswith("Review case")
        )
        seed = int(re.search(r"Review case (\d+)", instruction)[1])
        role = req["tool_choice"]
        name = role["function"]["name"] if isinstance(role, dict) else "final"
        self.server.received.append((seed, name))
        if name == "final":
            docs = fixture(seed, 2)[2]
            a, b = [d["p95_ms"] for d in docs]
            pieces = [
                {"content": f"Before: {a} ms. After: {b} ms. Reduction: {a - b} ms. "},
                {"content": "report-A and report-B."},
            ]
            finish = "stop"
        else:
            args = (
                {}
                if name == "list_reports"
                else {"first": "report-A", "second": "report-B"}
            )
            pieces = [
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": f"server-{seed}-{name}",
                            "type": "function",
                            "function": {"name": name, "arguments": json.dumps(args)},
                        }
                    ]
                }
            ]
            finish = "tool_calls"
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for delta in pieces + [{}]:
            event = {
                "id": f"reply-{seed}-{name}",
                "choices": [
                    {
                        "index": 0,
                        "delta": delta,
                        "finish_reason": None if delta else finish,
                    }
                ],
            }
            if not delta:
                event["usage"] = {"prompt_tokens": 500, "completion_tokens": 64}
            self.wfile.write(("data: " + json.dumps(event) + "\n\n").encode())
        self.wfile.write(b"data: [DONE]\n\n")

    def log_message(self, *args):
        pass


class FleetTests(unittest.TestCase):
    def test_cli_sends_only_model_work_and_repeated_waves_use_unique_sessions(self):
        with ExitStack() as stack:
            tmp = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            servers = []
            for _ in range(2):
                server = ThreadingHTTPServer(("127.0.0.1", 0), TestReplica)
                server.received = []
                th = threading.Thread(target=server.serve_forever, daemon=True)
                th.start()
                stack.callback(server.server_close)
                stack.callback(th.join)
                stack.callback(server.shutdown)
                servers.append(server)
            inventory = tmp / "inventory.json"
            inventory.write_text(
                json.dumps(
                    {
                        "model": "m",
                        "replicas": [
                            {
                                "url": f"http://127.0.0.1:{s.server_port}/v1",
                                "node": f"test-{i}",
                                "gpu_count": 1,
                            }
                            for i, s in enumerate(servers)
                        ],
                    }
                )
            )
            for mode, waves, requests in [(True, [12, 12], 24), (False, [1, 2], 9)]:
                for s in servers:
                    s.received.clear()
                out = tmp / f"{mode}.json"
                cmd = [
                    sys.executable,
                    agent_fleet_replay.__file__,
                    str(inventory),
                    str(out),
                    "--waves",
                    *map(str, waves),
                ]
                if mode:
                    cmd.append("--dispatch-fixed-tools")
                subprocess.run(cmd, check=True, capture_output=True, timeout=30)
                result = json.loads(out.read_text())
                rs = [r for w in result["waves"] for r in w["records"]]
                self.assertEqual(len({r["workflow_id"] for r in rs}), sum(waves))
                self.assertEqual(len({r["seed"] for r in rs}), sum(waves))
                if not mode:
                    self.assertEqual([r["seed"] for r in rs], [761, 771, 772])
                self.assertTrue(all(r["ok"] for r in rs), [r.get("error") for r in rs])
                self.assertTrue(all(r["start_ns"] >= r["scheduled_ns"] for r in rs))
                received = [x for s in servers for x in s.received]
                self.assertEqual(len(received), requests)
                self.assertEqual(Counter(n for _, n in received)["final"], sum(waves))
                for w in result["waves"]:
                    self.assertEqual(w["scheduled_slo_completed"], w["submitted"])
                    self.assertAlmostEqual(
                        w["scheduled_warm_goodput_per_gpu_hour"],
                        w["scheduled_slo_completed"] * 3600 / (2 * w["window_s"]),
                    )
                    self.assertGreaterEqual(
                        w["scheduled_last_p95_ms"], w["last_p95_ms"]
                    )

    def test_schedule_lag_can_turn_a_completed_job_into_a_deadline_miss(self):
        def late(*args, **kwargs):
            return {
                "start_ns": time.monotonic_ns() + 2_000_000,
                "ok": True,
                "first_final_ms": 9999,
                "last_final_ms": 10050,
                "final_gap_p95_ms": 10,
            }

        router = unittest.mock.Mock()
        with patch.object(replica_workflow, "replay", late):
            w = replica_workflow.run_waves(router, "m", [1], 1)[0]
        self.assertEqual((w["slo_completed"], w["scheduled_slo_completed"]), (1, 0))
        self.assertGreater(w["warm_goodput_per_gpu_hour"], 0)
        self.assertEqual(w["scheduled_warm_goodput_per_gpu_hour"], 0)


if __name__ == "__main__":
    unittest.main()
