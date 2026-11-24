"""Regression cases for the optional cross-client load sampler and placement policy."""

import json
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import Mock, patch

from replica_load import ReplicaLoad, read_load
from replica_workflow import Router


class LoadTest(unittest.TestCase):
    def test_http_and_expiration(self):
        checks = []
        response = [
            200,
            {
                "ready": True,
                "accepting": True,
                "active_requests": 6,
                "waiting_requests": 1,
            },
        ]

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                checks.append((self.path, self.headers.get("Authorization")))
                self.send_response(response[0])
                self.end_headers()
                self.wfile.write(json.dumps(response[1]).encode())

            def log_message(self, *args):
                pass

        with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
            service = threading.Thread(target=server.serve_forever)
            service.start()
            endpoint = (f"http://127.0.0.1:{server.server_port}/gateway/v1", "fixture")
            try:
                with ReplicaLoad([endpoint]) as telemetry:
                    deadline = time.monotonic() + 2
                    while (
                        telemetry.snapshot()[0] is None and time.monotonic() < deadline
                    ):
                        time.sleep(0.01)
                    self.assertEqual(telemetry.snapshot()[0]["active"], 6)
                self.assertFalse(telemetry.thread.is_alive())
                self.assertEqual(checks[0], ("/gateway/readyz", "Bearer fixture"))
                response[0] = 503
                self.assertFalse(read_load(endpoint)[1]["available"])
                response[0] = 401
                self.assertIsNone(read_load(endpoint))
                response[0] = 200
                response[1]["waiting_requests"] = False
                self.assertIsNone(read_load(endpoint))
                with patch(
                    "replica_load.time.monotonic",
                    return_value=telemetry.samples[0][0] + 0.501,
                ):
                    self.assertEqual(telemetry.snapshot(), [None])
            finally:
                server.shutdown()
                service.join()

    def test_missing_sample_fallback(self):
        ready = {"available": True, "active": 0, "waiting": 0}
        request = {
            "model": "m",
            "tools": [],
            "tool_choice": "none",
            "messages": [{"role": "user", "content": "abc"}],
        }
        for samples, expected in [
            ([None, ready], 1),
            ([ready, None], 0),
            ([None, None], 0),
            ([None, dict(ready, active=8)], 0),
            ([None, dict(ready, waiting=1)], 0),
            ([dict(ready, available=False), None], 1),
        ]:
            with self.subTest(samples=samples):
                sampler = Mock()
                sampler.snapshot.return_value = samples
                router = Router(
                    [("a", None), ("b", None)], "fresh-capacity", [8, 8], sampler
                )
                row = {"usage": {"prompt_tokens": 1}}
                with patch("replica_workflow.stream"):
                    router.sender(0)("", request, row, None)
                self.assertEqual(row["route"]["replica"], expected)

    def test_cold_recompute_deferral(self):
        idle = {"available": True, "active": 0, "waiting": 0}
        reading = dict(idle, active=1)
        for idle_after, credit, expect in [
            (0, 3000, 0),
            (1, 3000, 0.05),
            (999, 3000, 1.5),
            (999, 100, 0),
        ]:
            with self.subTest(idle_after=idle_after, credit=credit):
                sampler = Mock()
                router = Router(
                    [("a", None), ("b", None)], "defer-cold-prefill", [8, 8], sampler
                )
                router.credits[0]["old"] = credit
                clock = [0.0]

                def sleep(seconds, router=router, clock=clock):
                    self.assertTrue(router.lock.acquire(blocking=False))
                    router.lock.release()
                    self.assertEqual(router.active, [0, 0])
                    clock[0] += seconds

                sampler.snapshot.side_effect = (
                    lambda clock=clock, idle_after=idle_after: [
                        None,
                        reading if clock[0] < idle_after * 0.05 else idle,
                    ]
                )
                row = {}
                with (
                    patch(
                        "replica_workflow.time.monotonic",
                        side_effect=lambda clock=clock: clock[0],
                    ),
                    patch("replica_workflow.time.sleep", side_effect=sleep),
                ):
                    router.defer_cold_prefill(["old"], 5000, row)
                self.assertAlmostEqual(row.get("routing_hold_ms", 0), expect * 1000)
                self.assertEqual("initial_readiness" in row, expect > 0)

    def test_poll_outcomes(self):
        endpoint = ("http://localhost/test/v1", None)
        for status, raw, error, expected in [
            (429, b"{}", None, "http_error"),
            (503, b"{}", None, "invalid"),
            (200, b"not JSON", None, "invalid"),
            (200, b" " * 65537, None, "invalid"),
            (None, b"", TimeoutError(), "timeout"),
            (None, b"", ConnectionResetError(), "transport_error"),
        ]:
            with (
                self.subTest(expected=expected),
                patch("replica_load.http.client.HTTPConnection") as factory,
            ):
                connection = factory.return_value
                connection.getresponse.return_value.status = status
                connection.getresponse.return_value.read.return_value = raw
                connection.getresponse.side_effect = error
                observe = Mock()
                self.assertIsNone(read_load(endpoint, observe))
                start, end, http, outcome = observe.call_args.args
                self.assertGreaterEqual(end, start)
                self.assertEqual((http, outcome), (status, expected))
                connection.close.assert_called_once()

    def test_placement(self):
        request = {
            "model": "m",
            "tools": [],
            "tool_choice": "none",
            "messages": [{"role": "user", "content": "abc"}],
        }
        ready = {"available": True, "active": 0, "waiting": 0}
        offline = dict(ready, available=False)
        for local, observed, want in [
            ([0, 0], [None, None], 0),
            ([4, 4], [dict(ready, active=4), ready], 0),
            ([0, 0], [dict(ready, active=8), ready], 1),
            ([0, 0], [dict(ready, waiting=1), ready], 1),
            ([8, 0], [None, None], 1),
            ([0, 0], [offline, None], 1),
            ([0, 0], [offline, offline], None),
        ]:
            with self.subTest(local=local, observed=observed):
                sampler = Mock()
                sampler.snapshot.return_value = observed
                router = Router(
                    [("a", None), ("b", None)], "observed-capacity", [8, 8], sampler
                )
                router.active = local.copy()
                row = {"usage": {"prompt_tokens": 10}}
                with patch("replica_workflow.stream") as stream:
                    if want is None:
                        with self.assertRaises(ConnectionError):
                            router.sender(0)("", request, row, None)
                        stream.assert_not_called()
                    else:
                        router.sender(0)("", request, row, None)
                        self.assertEqual(row["route"]["replica"], want)
                self.assertEqual(router.active, local)


if __name__ == "__main__":
    unittest.main()
