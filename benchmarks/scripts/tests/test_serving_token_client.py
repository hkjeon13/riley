from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import socket
import sys
import threading
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import serving_token_client as client


def reference(tokens=(10, 11, 12), text="가", finish="length"):
    return client.TokenReference("fixture", (1, 2), tokens, text, finish)


def envelope(choices, *, usage=None, identity="cmpl-fixture"):
    result = {"id": identity, "model": "fixture", "object": "text_completion", "created": 42,
              "choices": choices}
    if usage is not None:
        result["usage"] = usage
    return result


def choice(text="", tokens=None, prompt=None, finish=None):
    value = {"index": 0, "text": text, "finish_reason": finish}
    if tokens is not None:
        value["token_ids"] = tokens
    if prompt is not None:
        value["prompt_token_ids"] = prompt
    return value


def frames(*, combined=False, tokens=(10, 11, 12), text="가", identity="cmpl-fixture", finish="length"):
    result = []
    for index, token in enumerate(tokens):
        delta = text if index == min(1, len(tokens) - 1) else ""
        result.append(envelope([choice(delta, [token], [1, 2] if index == 0 else None,
                                      finish if combined and index == len(tokens) - 1 else None)], identity=identity))
    if not combined:
        result.append(envelope([choice(finish=finish)], identity=identity))
    result.append(envelope([], usage={"prompt_tokens": 2, "completion_tokens": len(tokens),
                                     "total_tokens": len(tokens) + 2}, identity=identity))
    result.append("[DONE]")
    return result


def payload(value):
    return value.encode() if isinstance(value, str) else json.dumps(value, ensure_ascii=False).encode()


def wire(values):
    return b"".join(b"data: " + payload(value) + b"\n\n" for value in values)


class ParserTests(unittest.TestCase):
    def parse(self, values, *, ref=None, mode="strict"):
        parser = client.TokenResponseParser(ref or reference(), streaming=True, started_ns=100, mode=mode)
        for index, value in enumerate(values):
            parser.feed_sse(payload(value), 110 + 10 * index)
        return parser, parser.finish()

    def test_both_terminal_shapes_preserve_empty_token_times_and_exclude_usage(self):
        for combined in (False, True):
            parser, row = self.parse(frames(combined=combined))
            self.assertEqual(row["token_arrival_ns"], [110, 120, 130])
            self.assertEqual(row["metrics"]["token_ttft_ns"], 10)
            self.assertEqual(row["metrics"]["token_tpot_ns"], 10)
            self.assertEqual(row["metrics"]["token_itl_ns"], [10, 10])
            self.assertEqual(row["metrics"]["first_text_ns"], 20)
            self.assertEqual(row["token_ids"], [10, 11, 12])
            self.assertTrue(row["protocol_valid"] and row["reference_match"])
            self.assertFalse(row["correctness_qualified"] or row["performance_qualified"])
            with self.assertRaises(client.ProtocolError):
                parser.feed_sse(b"[DONE]", 200)

    def test_single_invisible_token_has_no_tpot_or_first_visible_text(self):
        _, row = self.parse(frames(tokens=(10,), text=""), ref=reference((10,), ""))
        self.assertEqual(row["metrics"]["token_ttft_ns"], 10)
        self.assertIsNone(row["metrics"]["token_tpot_ns"])
        self.assertIsNone(row["metrics"]["first_text_ns"])
        self.assertEqual(row["metrics"]["token_itl_ns"], [])

    def test_nonstream_validates_exact_ids_and_usage_without_token_times(self):
        parser = client.TokenResponseParser(reference(), streaming=False, started_ns=100)
        parser.feed_nonstream(payload(envelope([choice("가", [10, 11, 12], [1, 2], "length")],
                                              usage={"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5})), 150)
        row = parser.finish()
        self.assertEqual(row["metrics"]["e2e_ns"], 50)
        self.assertIsNone(row["metrics"]["token_ttft_ns"])
        self.assertIsNone(row["metrics"]["token_tpot_ns"])
        self.assertEqual(row["token_arrival_ns"], [])

    def test_observation_mode_exposes_mismatch_without_qualifying_it(self):
        changed = frames(tokens=(10, 99, 12), text="나")
        _, row = self.parse(changed, mode="observe")
        self.assertTrue(row["protocol_valid"] and row["observation_only"])
        self.assertFalse(row["reference_match"] or row["correctness_qualified"] or row["performance_qualified"])
        self.assertEqual(row["reference_comparison"]["output_token_ids"]["first_mismatch_index"], 1)
        with self.assertRaises(client.ReferenceMismatch):
            self.parse(changed)
        with self.assertRaises(client.ProtocolError):
            self.parse(frames(tokens=(10, 99)), mode="observe")

    def test_malformed_identity_ids_usage_and_order_fail_in_both_modes(self):
        cases = []
        for key, value in (("id", ""), ("model", "other"), ("object", "wrong"), ("created", True)):
            case = frames(); case[0][key] = value; cases.append(case)
        case = frames(); case[1]["id"] = "other"; cases.append(case)
        for key, value in (("index", 1), ("index", False), ("token_ids", [10, 11]),
                           ("token_ids", [True]), ("token_ids", [-1]), ("token_ids", [2**32]),
                           ("prompt_token_ids", None), ("prompt_token_ids", [])):
            case = frames(); case[0]["choices"][0][key] = value; cases.append(case)
        case = frames(); case[1]["choices"][0]["prompt_token_ids"] = [1, 2]; cases.append(case)
        case = frames(); del case[1]["choices"][0]["token_ids"]; cases.append(case)
        case = frames(); case[0]["choices"][0]["finish_reason"] = "length"; cases.append(case)
        case = frames(); case[-2]["usage"]["total_tokens"] = 999; cases.append(case)
        case = frames(); case[-2]["usage"]["prompt_tokens"] = True; cases.append(case)
        case = frames(); case.insert(0, case[-2]); cases.append(case)
        case = frames(); case.insert(-1, copy.deepcopy(case[-2])); cases.append(case)
        case = frames(); case.insert(-2, envelope([choice(finish="length")])); cases.append(case)
        cases.extend([frames()[:-1], frames()[:-2] + ["[DONE]"], [envelope([], usage={})],
                      [{"error": {"message": "cancelled"}}]])
        for mode in ("strict", "observe"):
            for index, case in enumerate(cases):
                with self.subTest(mode=mode, case=index), self.assertRaises((client.ProtocolError, ValueError)):
                    self.parse(case, mode=mode)

    def test_raw_failure_payload_and_clock_checks(self):
        for bad in (b"{", b'{"choices":[],"choices":[]}', b'\xff', b'{"created":NaN}'):
            parser = client.TokenResponseParser(reference(), streaming=True, started_ns=100)
            with self.assertRaises((client.ProtocolError, ValueError)):
                parser.feed_sse(bad, 110)
            self.assertEqual(len(parser.frames), 1)
        parser = client.TokenResponseParser(reference(), streaming=True, started_ns=100)
        with self.assertRaises(client.ProtocolError):
            parser.feed_sse(payload(frames()[0]), 99)

    def test_framer_handles_split_utf8_crlf_and_same_read_timestamps(self):
        framer = client.SSEFramer()
        raw = wire(frames()).replace(b"\n", b"\r\n")
        split = raw.index("가".encode()) + 1
        first = framer.feed(raw[:split], 110)
        second = framer.feed(raw[split:], 120)
        framer.finish()
        self.assertEqual(len(first), 1)
        self.assertEqual(len(second), 5)
        self.assertEqual({stamp for _, stamp in second}, {120})
        parser = client.TokenResponseParser(reference(), streaming=True, started_ns=100)
        for data, stamp in first + second:
            parser.feed_sse(data, stamp)
        self.assertEqual(parser.finish()["token_arrival_ns"], [110, 120, 120])
        framer = client.SSEFramer(10)
        with self.assertRaises(client.ProtocolError):
            framer.feed(b"data: " + b"x" * 20, 1)
        framer = client.SSEFramer()
        framer.feed(b"data: unfinished\n", 1)
        with self.assertRaises(client.ProtocolError):
            framer.finish()


class FixtureServer(ThreadingHTTPServer):
    daemon_threads = True
    def __init__(self):
        super().__init__(("127.0.0.1", 0), FixtureHandler)
        self.lock, self.next_id = threading.Lock(), 0
        self.release, self.entered = threading.Event(), threading.Event()
        self.bodies = []


class FixtureHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *_):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        with self.server.lock:
            self.server.next_id += 1
            identity = "cmpl-" + str(self.server.next_id)
            self.server.bodies.append(body)
        prompt = body["prompt"]
        self.server.entered.set()
        if prompt == "header-trickle":
            try:
                for part in (b"HTTP/1.1 200 OK\r\n",) + (b"X-Test: incomplete",) * 20:
                    self.connection.sendall(part)
                    if self.server.release.wait(.025): break
            except OSError:
                pass
            self.close_connection = True
            return
        streaming = body["stream"]
        self.send_response(429 if prompt == "http-error" else 200)
        self.send_header("Content-Type", "text/event-stream" if streaming else "application/json")
        self.send_header("Connection", "close")
        if prompt == "truncated-chunk":
            self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        try:
            if prompt == "http-error":
                self.wfile.write(b'{"error":{"message":"overloaded"}}'); self.wfile.flush()
                return
            values = frames(identity=identity, combined=prompt == "vllm",
                            tokens=(10, 99, 12) if prompt == "mismatch" else (10, 11, 12))
            if prompt == "stop":
                values = frames(identity=identity, tokens=(10,), text="", finish="stop")
            if streaming:
                if prompt == "truncated-chunk":
                    raw = wire(values)
                    self.wfile.write(f"{len(raw):x}\r\n".encode() + raw + b"\r\n")
                    self.wfile.flush()
                    return  # Deliberately omit the required terminating chunk.
                if prompt in ("trickle", "stall"):
                    self.wfile.write(wire(values[:1])); self.wfile.flush()
                    if prompt == "stall":
                        self.server.release.wait(2)
                    else:
                        for _ in range(100):
                            self.wfile.write(b" "); self.wfile.flush()
                            if self.server.release.wait(.01): break
                    return
                if prompt == "overlap": time.sleep(.03)
                self.wfile.write(wire(values)); self.wfile.flush()
                if prompt == "late-done":
                    time.sleep(.04)
                    self.wfile.write(b"data: [DONE]\n\n"); self.wfile.flush()
            else:
                if prompt == "overlap": time.sleep(.03)
                result = envelope([choice("가", [10, 11, 12], [1, 2], "length")],
                                  usage={"prompt_tokens":2,"completion_tokens":3,"total_tokens":5}, identity=identity)
                self.wfile.write(payload(result)); self.wfile.flush()
        except OSError:
            pass
        finally:
            self.close_connection = True


class LoopbackTests(unittest.TestCase):
    def setUp(self):
        self.server = FixtureServer()
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": .01}, daemon=True)
        self.thread.start()
        self.port = self.server.server_address[1]

    def tearDown(self):
        self.server.release.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=1)

    @staticmethod
    def body(prompt="ok", max_tokens=3):
        return {"model":"fixture", "prompt":prompt, "max_tokens":max_tokens, "temperature":0, "top_p":1}

    def test_both_transports_and_vllm_shape_have_exact_raw_ids(self):
        with client.TokenHttpClient() as http:
            for streaming, prompt in ((False, "ok"), (True, "ok"), (True, "vllm")):
                row = http.request(self.port, self.body(prompt), reference(), streaming=streaming)
                self.assertEqual(row["status"], "success", row)
                self.assertTrue(row["protocol_valid"] and row["transport_complete"] and row["reference_match"])
                self.assertTrue(row["owned_connection_closed"])
                self.assertEqual(http.active_requests, 0)
                self.assertLessEqual(row["finished_ns"], row["call_finished_ns"])
                self.assertEqual(len(row["token_arrival_ns"]), 3 if streaming else 0)
        for body in self.server.bodies:
            self.assertIs(body["return_token_ids"], True)
            self.assertEqual(body.get("stream_options"), {"include_usage":True} if body["stream"] else None)

    def test_strict_mismatch_fails_but_observation_is_explicitly_unqualified(self):
        with client.TokenHttpClient() as http:
            for mode in ("strict", "observe"):
                row = http.request(self.port, self.body("mismatch"), reference(), streaming=True, mode=mode)
                self.assertEqual(row["status"], "failed" if mode == "strict" else "success")
                self.assertTrue(row["protocol_valid"])
                self.assertFalse(row["reference_match"] or row["correctness_qualified"] or row["performance_qualified"])

    def test_stop_reference_can_be_shorter_than_requested_budget(self):
        with client.TokenHttpClient() as http:
            row = http.request(self.port, self.body("stop", 32), reference((10,), "", "stop"), streaming=True)
            self.assertEqual(row["status"], "success", row)
            self.assertEqual(row["token_ids"], [10])
            self.assertEqual(self.server.bodies[0]["max_tokens"], 32)

    def test_total_deadline_bounds_trickling_body_and_headers_and_preserves_partial(self):
        with client.TokenHttpClient() as http:
            for prompt in ("trickle", "header-trickle"):
                started = time.monotonic()
                row = http.request(self.port, self.body(prompt), reference(), streaming=True, timeout_seconds=.12)
                self.assertEqual(row["status"], "failed")
                self.assertEqual(row["error"]["type"], "TimeoutError", row)
                self.assertLess(time.monotonic() - started, 1)
                self.assertEqual(http.active_requests, 0)
                if prompt == "trickle":
                    self.assertEqual(row["token_ids"], [10])
                    self.assertEqual(len(row["token_arrival_ns"]), 1)
                    self.assertFalse(row["protocol_valid"])

    def test_client_close_aborts_only_its_owned_pending_request(self):
        http = client.TokenHttpClient().__enter__()
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(http.request, self.port, self.body("stall"), reference(), streaming=True, timeout_seconds=2)
            self.assertTrue(self.server.entered.wait(1))
            http.close()
            row = future.result(timeout=1)
            self.assertEqual(row["status"], "failed")
            self.assertEqual(row["error"]["type"], "CancelledError")
        self.assertEqual(http.active_requests, 0)
        http.close()

    def test_late_payload_after_done_is_rejected_without_retiming_done(self):
        with client.TokenHttpClient() as http:
            row = http.request(self.port, self.body("late-done"), reference(), streaming=True)
        self.assertEqual(row["status"], "failed")
        self.assertIn("after DONE", row["error"]["message"])
        self.assertLess(row["done_ns"], row["call_finished_ns"])
        self.assertFalse(row["protocol_valid"])

    def test_error_http_body_is_retained(self):
        import base64
        with client.TokenHttpClient() as http:
            row = http.request(self.port, self.body("http-error"), reference(), streaming=True)
        self.assertEqual(row["http_status"], 429)
        self.assertIn(b"overloaded", base64.b64decode(row["non_sse_body_base64"]))

    def test_truncated_http_chunk_after_done_is_not_a_complete_response(self):
        with client.TokenHttpClient() as http:
            row = http.request(self.port, self.body("truncated-chunk"), reference(), streaming=True)
        self.assertEqual(row["status"], "failed", row)
        self.assertEqual(row["token_ids"], [10,11,12])
        self.assertIsNotNone(row["done_ns"])
        self.assertFalse(row["protocol_valid"] or row["transport_complete"])

    def test_callback_failure_is_accounted_without_deadlock(self):
        def fail():
            raise RuntimeError("adapter failed before request")
        with client.TokenHttpClient() as http:
            rows, accounting = client.run_phase(http, fail, concurrency=2, count=10, phase="retained")
        self.assertFalse(accounting["completed"])
        self.assertEqual(accounting["failed"], len(rows))
        self.assertEqual(accounting["unresolved_attempts"], 0)
        self.assertEqual(accounting["not_started"] + len(rows), 10)
        self.assertTrue(all(row["error"]["type"] == "RuntimeError" for row in rows))

    def test_warmup_and_retained_closed_loop_accounting_observes_concurrency(self):
        with client.TokenHttpClient() as http:
            for phase, streaming in (("warmup-nonstream", False), ("warmup-stream", True), ("retained", True)):
                rows, accounting = client.run_phase(http,
                    lambda: http.request(self.port, self.body("overlap"), reference(), streaming=streaming),
                    concurrency=4, count=8, phase=phase, per_worker=phase != "retained")
                self.assertTrue(accounting["completed"], accounting)
                self.assertEqual(accounting["observed_max_request_in_flight"], 4)
                self.assertEqual(accounting["attempted"], 8)
                self.assertEqual(len({row["response_identity"]["id"] for row in rows}), 8)
                self.assertLessEqual(accounting["phase_started_ns"], min(row["started_ns"] for row in rows))
                summary = client.summarize_phase(rows, accounting)
                self.assertEqual(summary["successful_output_tokens"], 24)
                self.assertFalse(summary["correctness_qualified"] or summary["performance_qualified"])
                self.assertEqual(summary["token_ttft_ms"] is None, not streaming)
                if phase != "retained":
                    self.assertEqual(sorted(sum(row["worker_id"] == worker for row in rows) for worker in range(4)), [2]*4)

    def test_phase_stops_refill_and_retains_failed_attempts(self):
        with client.TokenHttpClient() as http:
            rows, accounting = client.run_phase(http,
                lambda: http.request(self.port, self.body("trickle"), reference(), streaming=True, timeout_seconds=.1),
                concurrency=2, count=20, phase="retained")
            self.assertFalse(accounting["completed"])
            self.assertEqual(accounting["unresolved_attempts"], 0)
            self.assertEqual(accounting["attempted"] + accounting["not_started"], 20)
            self.assertEqual(accounting["failed"], len(rows))
            self.assertEqual(accounting["succeeded"], 0)
            self.assertTrue(all(row["token_ids"] == [10] for row in rows))
            self.assertEqual(http.active_requests, 0)
            summary = client.summarize_phase(rows, accounting)
            self.assertEqual(summary["successful_output_tokens_per_wall_second"], 0)
            self.assertGreater(summary["observed_output_tokens_including_partial_failures"], 0)


if __name__ == "__main__":
    unittest.main()
