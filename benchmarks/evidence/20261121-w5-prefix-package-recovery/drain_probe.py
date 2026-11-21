"""Witness a real Riley SSE response across a loopback admission cutoff."""

import asyncio
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from threading import Event, Thread
import time

from load import one

HOST = "127.0.0.1"
CHAT = "/v1/chat/completions"
GAUGES = ("active_requests", "waiting_requests", "kv_allocated_blocks")


def request(port, path, body=None):
    wire = HTTPConnection(HOST, port, timeout=30)
    try:
        if body is None:
            wire.request("GET", path)
        else:
            wire.request("POST", path, body=json.dumps(body), headers={"Content-Type": "application/json"})
        response = wire.getresponse()
        return response.status, response.read()
    finally:
        wire.close()


def document(port, path, body=None):
    status, data = request(port, path, body)
    if status != 200:
        raise RuntimeError(f"{path} returned HTTP {status}")
    return json.loads(data)


def gateway(port, upstream, admitted):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, _format, *_args):
            return

        def do_POST(self):
            if self.path != CHAT:
                self.send_error(404)
                return
            if not admitted.is_set():
                self.send_error(503, "recovery drain in progress")
                return
            body = self.rfile.read(int(self.headers["Content-Length"]))
            backend = HTTPConnection(HOST, upstream, timeout=30)
            try:
                backend.request("POST", CHAT, body=body, headers={"Content-Type": "application/json"})
                response = backend.getresponse()
                self.send_response(response.status)
                self.send_header("Content-Type", response.getheader("Content-Type", "application/json"))
                self.send_header("Connection", "close")
                self.end_headers()
                for line in iter(response.readline, b""):
                    self.wfile.write(line)
            finally:
                backend.close()

    return ThreadingHTTPServer((HOST, port), Handler)


async def active_drain(front, backend, admission, body):
    record = {}
    started = time.perf_counter()
    async with asyncio.TaskGroup() as tasks:
        stream = tasks.create_task(one(HOST, front, body, started, record, capture_hashes=True))
        async with asyncio.timeout(20):
            while True:
                metrics = await asyncio.to_thread(document, backend, "/metrics")
                if metrics["active_requests"] > 0:
                    break
                if stream.done():
                    raise RuntimeError("no live request observed before closing admissions")
                await asyncio.sleep(0.02)
        admission.clear()
        gate_ns = time.monotonic_ns()
        status, response = await asyncio.to_thread(request, front, CHAT, body)
        if status != 503 or b"recovery drain in progress" not in response:
            raise RuntimeError("a new request was not rejected by the test gateway")
        active = await asyncio.to_thread(document, backend, "/metrics")
        if active["active_requests"] != 1 or active["waiting_requests"] != 0:
            raise RuntimeError("expected only the original stream to remain active after cutoff")
        active_ns = time.monotonic_ns()
        await asyncio.wait_for(stream, timeout=30)
    completed_ns = time.monotonic_ns()
    if record.get("error") is not False:
        raise RuntimeError(f"in-flight SSE did not complete with usage and DONE: {record.get('detail')}")
    deadline = time.monotonic() + 15
    while True:
        drained = await asyncio.to_thread(document, backend, "/metrics")
        if all(drained[key] == 0 for key in GAUGES):
            break
        if time.monotonic() >= deadline:
            raise RuntimeError("the server did not drain its requests and KV")
        await asyncio.sleep(0.1)
    return {"gate_closed_ns": gate_ns, "rejected_http_status": status, "active_after_gate_ns": active_ns,
            "active_after_gate": {key: active[key] for key in GAUGES}, "sse_completed_ns": completed_ns,
            "drained_ns": time.monotonic_ns(), "drained": {key: drained[key] for key in GAUGES},
            "stream": {key: record[key] for key in ("input_sha256", "output_sha256", "prompt_tokens",
                                                    "completion_tokens", "finish")},
            "final": {"metrics": drained}}


def run(front, backend):
    initial = document(backend, "/metrics")
    prefix = "A Riley recovery drill verifies the same assistant on the restored image. " * 8
    body = {"model": "m", "messages": [{"role": "system", "content": prefix},
                                      {"role": "user", "content": "Reply with one short sentence."}],
            "temperature": 0, "max_tokens": 24}
    document(backend, CHAT, body)
    before = document(backend, "/metrics")["prefix_cache"]["hit_tokens"]
    reply = document(backend, CHAT, body)
    cached = reply["usage"]["prompt_tokens_details"]["cached_tokens"]
    after = document(backend, "/metrics")["prefix_cache"]["hit_tokens"]
    if type(cached) is not int or cached <= 0 or after - before != cached:
        raise RuntimeError("cached prompt usage differs from the server's hit counter")
    body.update(messages=[{"role": "user", "content": "List all integers from 1 through 500, in order, separated by commas."}],
                max_tokens=192, stream=True, stream_options={"include_usage": True})
    admission = Event()
    admission.set()
    with gateway(front, backend, admission) as listener:
        worker = Thread(target=listener.serve_forever)
        worker.start()
        try:
            report = asyncio.run(active_drain(front, backend, admission, body))
        finally:
            listener.shutdown()
            worker.join()
    return {"initial": {"metrics": initial}, "cached_prompt_tokens": cached,
            "cache_hit_delta": after - before, **report}
