"""Token-aware HTTP observations, without campaign or numerical qualification.

Use ``with TokenHttpClient() as client`` and call ``client.request(port, payload,
reference, streaming=True)``. Payload explicitly supplies model/prompt/max_tokens
and sampling controls. This module requires raw IDs and, for SSE, final usage.
``mode='strict'`` rejects reference differences. ``mode='observe'`` records them
without qualifying correctness or performance. Both reject malformed transport and
output counts outside the fixed reference shape. A generated SSE frame may carry
multiple IDs; every ID receives that frame's actual arrival timestamp. No
interpolation or inferred engine-generation time is introduced.

``run_phase`` refills C<=8 clients, retaining all failed/partial rows. A single
client watchdog bounds each call, including trickling headers/body, by shutting
down only its owned sockets. The caller owns server processes and their cleanup.
Do not reinterpret old c1 receipts: new controllers must pin this module, source,
model, binary, request, reference, environment and their own workload manifest.

Frame times are perf_counter_ns observations immediately after read1 returns,
before framing/JSON validation. Frames completed by the same read share its
timestamp. IDs within a grouped frame also share that timestamp, so their
within-frame delivery ITLs are zero. Group sizes are reported separately from
coalesced reads. These are client delivery times, not CUDA or scheduler commit
times. Version1 observations and failed campaigns retain their original meaning.
"""
from __future__ import annotations

import base64
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import hashlib
import http.client
import json
import math
import socket
import statistics
import threading
import time


SCHEMA = "riley.http-token-observation.v2"
PHASE_SCHEMA = "riley.http-token-phase.v2"
MODES = ("strict", "observe")


class ProtocolError(ValueError):
    """A response cannot supply valid fixed-workload token-delivery observations."""


class ReferenceMismatch(ProtocolError):
    """Complete transport differed from the explicit immutable reference."""


def require(condition, message):
    if not condition:
        raise ProtocolError(message)


def unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def parse_json(payload):
    def invalid_constant(value):
        raise ProtocolError("nonfinite JSON: " + value)
    return json.loads(payload, object_pairs_hook=unique_pairs, parse_constant=invalid_constant)


def ids(value, label):
    require(isinstance(value, (list, tuple)) and all(type(token) is int and 0 <= token < 2**32 for token in value),
            label + " must contain unsigned model token IDs")
    return list(value)


def digest_json(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class TokenReference:
    model: str
    prompt_token_ids: tuple[int, ...]
    output_token_ids: tuple[int, ...]
    text: str
    finish_reason: str = "length"
    sha256: str = field(init=False)

    def __post_init__(self):
        require(isinstance(self.model, str) and self.model, "reference model missing")
        require(isinstance(self.text, str), "reference text must be a string")
        prompt, output = ids(self.prompt_token_ids, "reference prompt"), ids(self.output_token_ids, "reference output")
        require(0 < len(prompt) <= 1048576 and 0 < len(output) <= 65536, "reference shape is outside bounded limits")
        require(self.finish_reason in ("length", "stop"), "reference successful finish reason required")
        object.__setattr__(self, "prompt_token_ids", tuple(prompt))
        object.__setattr__(self, "output_token_ids", tuple(output))
        object.__setattr__(self, "sha256", digest_json({"model": self.model, "prompt_token_ids": prompt,
                           "output_token_ids": output, "text": self.text, "finish_reason": self.finish_reason}))


class SSEFramer:
    """Bounded UTF-8-independent framing; no JSON parsing before timestamping."""

    def __init__(self, maximum_frame_bytes=8 * 1024 * 1024):
        self.maximum = maximum_frame_bytes
        self.pending = bytearray()
        self.data = []
        self.data_bytes = 0

    def feed(self, chunk, arrived_ns):
        self.pending.extend(chunk)
        frames = []
        while True:
            end = self.pending.find(b"\n")
            if end < 0:
                break
            line = bytes(self.pending[:end])
            del self.pending[:end + 1]
            if line.endswith(b"\r"):
                line = line[:-1]
            require(len(line) + self.data_bytes <= self.maximum, "SSE frame exceeds byte bound")
            if not line:
                if self.data:
                    frames.append((b"\n".join(self.data), arrived_ns))
                self.data, self.data_bytes = [], 0
            elif line.startswith(b":"):
                continue
            elif line == b"data" or line.startswith(b"data:"):
                value = line[5:] if line.startswith(b"data:") else b""
                if value.startswith(b" "):
                    value = value[1:]
                self.data.append(value)
                self.data_bytes += len(value) + 1
            else:
                raise ProtocolError("unsupported SSE field")
        require(len(self.pending) + self.data_bytes <= self.maximum, "SSE frame exceeds byte bound")
        return frames

    def finish(self):
        require(not self.pending and not self.data, "unfinished SSE frame")


def _comparison(actual, expected):
    if actual == expected:
        return {"matches": True, "first_mismatch_index": None}
    first = next((i for i, (left, right) in enumerate(zip(actual, expected)) if left != right),
                 min(len(actual), len(expected)))
    return {"matches": False, "first_mismatch_index": first}


class TokenResponseParser:
    """Strict single-choice parser shared by direct tests and the HTTP client."""

    def __init__(self, reference, *, streaming, started_ns, mode="strict"):
        require(isinstance(reference, TokenReference), "explicit token reference required")
        require(mode in MODES and type(streaming) is bool, "invalid observation mode/transport")
        require(type(started_ns) is int and started_ns >= 0, "invalid request start clock")
        self.reference, self.mode, self.streaming = reference, mode, streaming
        self.started_ns = started_ns
        self.frames, self.token_times, self.text_times = [], [], []
        self.token_frame_counts = []
        self.tokens, self.prompt, self.text_parts = [], None, []
        self.identity, self.usage, self.finish_reason = None, None, None
        self.done_ns, self.finished_ns = None, None
        self.protocol_valid = False
        self.last_arrival = started_ns

    def _record(self, payload, arrived_ns):
        require(type(arrived_ns) is int and arrived_ns >= self.last_arrival, "arrival clock moved backwards")
        require(len(self.frames) < len(self.reference.output_token_ids) + 8, "too many non-token frames")
        self.last_arrival = arrived_ns
        try:
            self.frames.append({"arrived_ns": arrived_ns, "data": payload.decode("utf-8")})
        except UnicodeDecodeError:
            self.frames.append({"arrived_ns": arrived_ns, "invalid_utf8_base64": base64.b64encode(payload).decode("ascii")})
            raise ProtocolError("response payload is not UTF-8") from None

    def _metadata(self, value):
        require(isinstance(value, dict) and "error" not in value, "server error or non-object response")
        require(isinstance(value.get("id"), str) and 0 < len(value["id"]) <= 1024, "response ID missing or oversized")
        require(value.get("model") == self.reference.model, "response model identity differs")
        require(value.get("object") == "text_completion", "response object differs")
        require(type(value.get("created")) is int and value["created"] >= 0, "response creation metadata invalid")
        identity = {key: value[key] for key in ("id", "model", "object", "created")}
        if self.identity is None:
            self.identity = identity
        require(self.identity == identity, "response identity changed within stream")
        require(isinstance(value.get("choices"), list), "response choices missing")

    def _usage(self, usage):
        require(isinstance(usage, dict), "final usage missing")
        expected = {"prompt_tokens": len(self.reference.prompt_token_ids),
                    "completion_tokens": len(self.reference.output_token_ids),
                    "total_tokens": len(self.reference.prompt_token_ids) + len(self.reference.output_token_ids)}
        require(all(type(usage.get(key)) is int and usage[key] == count for key, count in expected.items()),
                "usage does not match fixed prompt/output counts")
        require(self.prompt is not None and len(self.prompt) == expected["prompt_tokens"]
                and len(self.tokens) == expected["completion_tokens"], "usage does not match observed IDs")
        self.usage = usage

    def _choice(self, choice, arrived_ns, *, complete=False):
        require(isinstance(choice, dict) and type(choice.get("index")) is int and choice["index"] == 0,
                "exactly choice index zero required")
        require(isinstance(choice.get("text"), str), "choice text missing")
        token_ids = choice.get("token_ids")
        tokens = [] if token_ids is None else ids(token_ids, "generated token_ids")
        prompt = choice.get("prompt_token_ids")
        if tokens:
            require(self.finish_reason is None and self.usage is None, "token arrived after terminal event")
            if not self.tokens:
                self.prompt = ids(prompt, "first generated frame prompt_token_ids")
                require(len(self.prompt) == len(self.reference.prompt_token_ids), "prompt ID count differs")
            else:
                require(prompt is None, "prompt IDs repeated after first generated frame")
            require(len(self.tokens) + len(tokens) <= len(self.reference.output_token_ids), "too many output IDs")
            self.tokens.extend(tokens)
            self.text_parts.append(choice["text"])
            if self.streaming:
                self.token_times.extend([arrived_ns] * len(tokens))
                self.token_frame_counts.append(len(tokens))
                if choice["text"]:
                    self.text_times.append(arrived_ns)
        else:
            require(not complete, "nonstream response omitted output IDs")
            require(not choice["text"] and prompt is None, "tokenless choice carries unattributed text/prompt IDs")
        finish = choice.get("finish_reason")
        if finish is not None:
            require(finish in ("length", "stop") and self.finish_reason is None, "invalid or repeated finish reason")
            require(len(self.tokens) == len(self.reference.output_token_ids), "finish before fixed output count")
            self.finish_reason = finish
        elif not tokens:
            require(self.finish_reason is None and self.usage is None, "empty choice after finish")

    def feed_sse(self, payload, arrived_ns):
        require(self.streaming, "SSE payload supplied to nonstream parser")
        self._record(payload, arrived_ns)
        require(self.done_ns is None, "payload after DONE")
        if payload == b"[DONE]":
            require(self.finish_reason is not None and self.usage is not None, "DONE before finish and final usage")
            self.done_ns = self.finished_ns = arrived_ns
            self.protocol_valid = True
            return
        value = parse_json(payload)
        self._metadata(value)
        choices = value["choices"]
        if not choices:
            require(self.finish_reason is not None and self.usage is None, "usage before finish or repeated usage")
            self._usage(value.get("usage"))
        else:
            require(len(choices) == 1 and value.get("usage") is None, "expected a single choice or final usage-only frame")
            self._choice(choices[0], arrived_ns)

    def feed_nonstream(self, payload, arrived_ns):
        require(not self.streaming and not self.frames, "repeated/wrong nonstream response")
        self._record(payload, arrived_ns)
        value = parse_json(payload)
        self._metadata(value)
        require(len(value["choices"]) == 1, "nonstream response requires one choice")
        self._choice(value["choices"][0], arrived_ns, complete=True)
        require(self.finish_reason is not None, "nonstream finish missing")
        self._usage(value.get("usage"))
        self.finished_ns, self.protocol_valid = arrived_ns, True

    def snapshot(self):
        text = "".join(self.text_parts)
        comparison = {"prompt_token_ids": _comparison(self.prompt or [], list(self.reference.prompt_token_ids)),
                      "output_token_ids": _comparison(self.tokens, list(self.reference.output_token_ids)),
                      "text": {"matches": text == self.reference.text},
                      "finish_reason": {"matches": self.finish_reason == self.reference.finish_reason}}
        reference_match = self.protocol_valid and all(item["matches"] for item in comparison.values())
        complete = self.protocol_valid
        metrics = {"e2e_ns": self.finished_ns - self.started_ns if complete else None,
                   "token_ttft_ns": self.token_times[0] - self.started_ns if complete and self.token_times else None,
                   "token_tpot_ns": ((self.token_times[-1] - self.token_times[0]) / (len(self.token_times) - 1)
                                     if complete and len(self.token_times) > 1 else None),
                   "token_itl_ns": [right - left for left, right in zip(self.token_times, self.token_times[1:])]
                                   if complete and self.streaming else [],
                   "first_text_ns": self.text_times[0] - self.started_ns if complete and self.text_times else None}
        delivery_groups = {
            "frame_token_counts": self.token_frame_counts.copy(),
            "generated_frame_count": len(self.token_frame_counts),
            "multi_token_frame_count": sum(count > 1 for count in self.token_frame_counts),
            "tokens_in_multi_token_frames": sum(count for count in self.token_frame_counts if count > 1),
            "within_frame_zero_itl_count": sum(count - 1 for count in self.token_frame_counts),
            "max_frame_token_count": max(self.token_frame_counts, default=0),
        } if self.streaming else None
        return {"schema_version": SCHEMA, "mode": self.mode, "streaming": self.streaming,
                "reference_sha256": self.reference.sha256, "started_ns": self.started_ns,
                "finished_ns": self.finished_ns, "response_identity": self.identity,
                "frames": self.frames.copy(), "token_arrival_ns": self.token_times.copy(),
                "token_delivery_groups": delivery_groups,
                "text_arrival_ns": self.text_times.copy(), "prompt_token_ids": self.prompt,
                "token_ids": self.tokens.copy(), "text": text, "finish_reason": self.finish_reason,
                "usage": self.usage, "done_ns": self.done_ns, "protocol_valid": complete,
                "reference_match": reference_match, "reference_comparison": comparison,
                "correctness_qualified": False, "performance_qualified": False,
                "observation_only": self.mode == "observe", "metrics": metrics,
                "timing_scope": "client-observed complete SSE payloads; all IDs in a frame share arrival time; no interpolation or engine/CUDA token times"}

    def finish(self):
        require(self.protocol_valid, "incomplete response (finish/usage/DONE missing)")
        result = self.snapshot()
        if self.mode == "strict" and not result["reference_match"]:
            raise ReferenceMismatch("complete response differs from explicit token/text reference")
        return result


@dataclass
class _Call:
    deadline_ns: int
    socket: socket.socket | None = None
    reason: str | None = None

    def abort(self, reason):
        self.reason = self.reason or reason
        if self.socket is not None:
            try:
                self.socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass


class TokenHttpClient:
    """Reusable concurrent loopback client; one watchdog, no server ownership."""

    def __init__(self, *, maximum_response_bytes=32 * 1024 * 1024, maximum_frame_bytes=8 * 1024 * 1024):
        require(type(maximum_response_bytes) is int and type(maximum_frame_bytes) is int
                and 0 < maximum_frame_bytes <= maximum_response_bytes,
                "invalid response byte bounds")
        self.maximum_response_bytes, self.maximum_frame_bytes = maximum_response_bytes, maximum_frame_bytes
        self._lock, self._changed = threading.Lock(), threading.Event()
        self._calls, self._closed, self._thread = {}, False, None

    def __enter__(self):
        with self._lock:
            require(self._thread is None and not self._closed, "client cannot be entered twice")
            self._thread = threading.Thread(target=self._watch, name="token-http-deadlines", daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *_):
        self.close()

    @property
    def active_requests(self):
        with self._lock:
            return len(self._calls)

    def _watch(self):
        while True:
            with self._lock:
                if self._closed:
                    return
                now = time.perf_counter_ns()
                for call in self._calls.values():
                    if now >= call.deadline_ns:
                        call.abort("total request deadline exceeded")
                remaining = [call.deadline_ns - now for call in self._calls.values() if call.reason is None]
                wait = min(.05, max(.001, min(remaining) / 1e9)) if remaining else .05
                self._changed.clear()
            self._changed.wait(wait)

    def abort_pending(self, reason="phase aborted"):
        with self._lock:
            for call in self._calls.values():
                call.abort(reason)
        self._changed.set()

    def close(self):
        with self._lock:
            self._closed = True
            for call in self._calls.values():
                call.abort("client closed")
        self._changed.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join()

    def request(self, port, payload, reference, *, streaming, mode="strict", timeout_seconds=120):
        require(type(port) is int and 0 < port < 65536, "invalid loopback port")
        require(type(timeout_seconds) in (int, float) and math.isfinite(timeout_seconds)
                and 0 < timeout_seconds <= 120, "invalid total request deadline")
        require(isinstance(reference, TokenReference) and mode in MODES and type(streaming) is bool,
                "explicit reference/mode/transport required")
        require(isinstance(payload, dict) and payload.get("model") == reference.model
                and isinstance(payload.get("prompt"), str) and type(payload.get("max_tokens")) is int
                and len(reference.output_token_ids) <= payload["max_tokens"] <= 65536
                and (reference.finish_reason == "stop" or payload["max_tokens"] == len(reference.output_token_ids)),
                "payload differs from reference model/output shape")
        body = dict(payload)
        require(body.get("return_token_ids", True) is True, "raw token observations must be enabled")
        require(body.get("stream", streaming) is streaming, "payload transport flag differs")
        body.update(stream=streaming, return_token_ids=True)
        if streaming:
            require(body.get("stream_options", {"include_usage": True}) == {"include_usage": True},
                    "strict final-only usage option required")
            require(type(body.get("stream_options", {"include_usage": True})["include_usage"]) is bool,
                    "usage option must be boolean")
            body["stream_options"] = {"include_usage": True}
        else:
            require(body.get("stream_options") is None, "nonstream requests cannot contain stream options")
            body.pop("stream_options", None)
        encoded = json.dumps(body, ensure_ascii=False, allow_nan=False).encode()
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout_seconds)
        started = time.perf_counter_ns()
        parser = TokenResponseParser(reference, streaming=streaming, started_ns=started, mode=mode)
        call = _Call(started + int(timeout_seconds * 1e9))
        key = id(call)
        response, error, raw_bytes, status, headers_ns = None, None, 0, None, None
        chunk, cleanup_errors, transport_complete = b"", [], False
        framer, nonstream = SSEFramer(self.maximum_frame_bytes), bytearray()
        with self._lock:
            require(self._thread is not None and not self._closed, "request requires an entered, open client")
            self._calls[key] = call
        self._changed.set()
        try:
            connection.connect()
            with self._lock:
                call.socket = connection.sock
                if time.perf_counter_ns() >= call.deadline_ns:
                    call.abort("total request deadline exceeded")
                if call.reason:
                    call.abort(call.reason)
                    raise TimeoutError(call.reason)
            connection.request("POST", "/v1/completions", encoded,
                               {"Content-Type": "application/json", "Connection": "close"})
            response = connection.getresponse()
            headers_ns, status = time.perf_counter_ns(), response.status
            content_type = response.getheader("Content-Type", "").split(";", 1)[0].strip().lower()
            expected_content = "text/event-stream" if streaming else "application/json"
            valid_head = status == 200 and content_type == expected_content
            while True:
                chunk = response.read1(65536)
                arrived = time.perf_counter_ns()  # Before decoding/framing any payload in this read.
                if call.reason or arrived >= call.deadline_ns:
                    raise TimeoutError(call.reason or "total request deadline exceeded")
                if not chunk:
                    break
                raw_bytes += len(chunk)
                require(raw_bytes <= self.maximum_response_bytes, "response exceeds byte bound")
                if streaming and valid_head:
                    for frame, timestamp in framer.feed(chunk, arrived):
                        parser.feed_sse(frame, timestamp)
                else:
                    nonstream.extend(chunk)
            require(status == 200, "HTTP status " + str(status))
            require(content_type == expected_content, "HTTP content type differs")
            require(response.length in (None, 0), "HTTP body ended before Content-Length")
            if streaming:
                framer.finish()
            else:
                parser.feed_nonstream(bytes(nonstream), arrived)
            transport_complete = True
            result = parser.finish()
            if call.reason or time.perf_counter_ns() >= call.deadline_ns:
                raise TimeoutError(call.reason or "total request deadline exceeded")
            result["status"] = "success"
        except Exception as failure:
            error = {"type": type(failure).__name__, "message": str(failure)}
            if call.reason:
                error = {"type": "TimeoutError" if "deadline" in call.reason else "CancelledError", "message": call.reason}
            result = parser.snapshot()
            result["status"] = "failed"
        finally:
            for resource in (response, connection):
                if resource is not None:
                    try:
                        resource.close()
                    except Exception as failure:
                        cleanup_errors.append({"type": type(failure).__name__, "message": str(failure)})
            with self._lock:
                self._calls.pop(key, None)
            self._changed.set()
        if cleanup_errors:
            result["status"] = "failed"
            error = error or {"type": "CleanupError", "message": "owned HTTP resource cleanup failed"}
        result.update(error=error, http_status=status, headers_received_ns=headers_ns,
                      call_finished_ns=time.perf_counter_ns(), raw_body_bytes=raw_bytes,
                      total_deadline_seconds=timeout_seconds, owned_connection_closed=not cleanup_errors,
                      cleanup_errors=cleanup_errors, request=body,
                      request_body_sha256=hashlib.sha256(encoded).hexdigest())
        result["parser_protocol_valid"] = result["protocol_valid"]
        result["protocol_valid"] = result["protocol_valid"] and transport_complete
        result["transport_complete"] = transport_complete
        if result["finished_ns"] is None:
            result["finished_ns"] = result["call_finished_ns"]
        if error is not None:
            result["failure_read_base64"] = base64.b64encode(chunk).decode("ascii")
            result["non_sse_body_base64"] = base64.b64encode(nonstream).decode("ascii")
            result["unfinished_payload_base64"] = base64.b64encode(bytes(framer.pending) if streaming else bytes(nonstream)).decode("ascii")
            result["unfinished_sse_data_base64"] = [base64.b64encode(part).decode("ascii") for part in framer.data]
        return result


def overlap_peak(rows, start_key="started_ns", end_key="finished_ns"):
    active = peak = 0
    events = [(row[start_key], 1) for row in rows] + [(row[end_key], -1) for row in rows]
    for _, delta in sorted(events):
        active += delta
        require(active >= 0, "invalid request interval")
        peak = max(peak, active)
    require(active == 0, "unbalanced request intervals")
    return peak


def run_phase(client, request_one, *, concurrency, count, phase, per_worker=False):
    """Closed-loop refill with explicit partial accounting; no process launches.

    request_one is a no-argument call to this client's request method. Warmups
    use per_worker=True and count=C*warmups. C overlap is recorded, never inferred
    from thread count. ``completed`` requires actual C overlap for every phase.
    """
    require(isinstance(client, TokenHttpClient), "owned deadline client required")
    require(type(concurrency) is int and 1 <= concurrency <= 8 and type(count) is int
            and count >= concurrency and count <= 100000, "invalid phase size/concurrency")
    require(not per_worker or count % concurrency == 0, "warmup count must divide evenly across clients")
    require(isinstance(phase, str) and phase, "phase identity missing")
    lock, stop = threading.Lock(), threading.Event()
    state = {"next": 0, "started_ns": time.perf_counter_ns(), "errors": []}
    barrier = threading.Barrier(concurrency + 1, action=lambda: state.update(started_ns=time.perf_counter_ns()))
    rows = []

    def worker(worker_id):
        completed = 0
        barrier.wait(timeout=30)
        while not stop.is_set():
            with lock:
                if stop.is_set() or state["next"] >= count or (per_worker and completed >= count // concurrency):
                    return
                index = state["next"]
                state["next"] += 1
            call_started = time.perf_counter_ns()
            try:
                row = request_one()
                require(row["schema_version"] == SCHEMA and row["status"] in ("success", "failed"), "not a token-client row")
                require(call_started <= row["started_ns"] <= row["finished_ns"] <= row["call_finished_ns"], "request timestamps invalid")
                require(all(row["started_ns"] <= timestamp <= row["finished_ns"] for timestamp in row["token_arrival_ns"]),
                        "token timestamp lies outside request")
            except Exception as error:
                row = {"schema_version": SCHEMA, "status": "failed", "started_ns": call_started,
                       "finished_ns": time.perf_counter_ns(), "call_finished_ns": time.perf_counter_ns(),
                       "error": {"type": type(error).__name__, "message": str(error)}, "token_ids": [],
                       "reference_match": False, "protocol_valid": False}
            row.update(index=index, worker_id=worker_id, phase=phase, warmup=phase != "retained", call_started_ns=call_started)
            with lock:
                rows.append(row)
            completed += 1
            if row["status"] != "success":
                stop.set()  # No retries; already in-flight calls retain their own deadlines.

    pool = ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="token-http-load")
    futures = []
    interrupted = None
    try:
        futures = [pool.submit(worker, worker_id) for worker_id in range(concurrency)]
        barrier.wait(timeout=30)
        for future in as_completed(futures):
            try:
                future.result()
            except BaseException as error:
                state["errors"].append({"type": type(error).__name__, "message": str(error)})
                stop.set()
                client.abort_pending("worker failed")
                if not isinstance(error, Exception):
                    interrupted = type(error).__name__
    except BaseException as error:
        state["errors"].append({"type": type(error).__name__, "message": str(error)})
        stop.set()
        client.abort_pending("phase interrupted")
        if not isinstance(error, Exception):
            interrupted = type(error).__name__
    finally:
        stop.set()
        barrier.abort()
        pool.shutdown(wait=True, cancel_futures=True)
    finished = time.perf_counter_ns()
    rows.sort(key=lambda row: row["index"])
    successful = [row for row in rows if row["status"] == "success"]
    response_ids = [row["response_identity"]["id"] for row in successful]
    duplicates = sorted(identity for identity, occurrences in Counter(response_ids).items() if occurrences > 1)
    references = sorted({row["reference_sha256"] for row in successful})
    modes = sorted({row["mode"] for row in successful})
    transports = sorted({row["streaming"] for row in successful})
    observed = overlap_peak(rows) if rows else 0
    accounting = {"schema_version": PHASE_SCHEMA, "phase": phase, "warmup": phase != "retained",
                  "offered_concurrency": concurrency, "requested": count, "attempted": state["next"],
                  "succeeded": len(successful), "failed": len(rows) - len(successful),
                  "unresolved_attempts": state["next"] - len(rows), "not_started": count - state["next"],
                  "reference_matches": sum(row["reference_match"] is True for row in rows),
                  "protocol_valid_responses": sum(row["protocol_valid"] is True for row in rows),
                  "observed_max_request_in_flight": observed,
                  "observed_max_call_in_flight": overlap_peak(rows, "call_started_ns", "call_finished_ns") if rows else 0,
                  "observed_max_successful_request_in_flight": overlap_peak(successful) if successful else 0,
                  "phase_started_ns": state["started_ns"], "phase_finished_ns": finished,
                  "completed": len(successful) == count and observed == concurrency and not state["errors"] and not duplicates
                               and len(references) == len(modes) == len(transports) == 1,
                  "duplicate_response_ids": duplicates,
                  "reference_sha256s": references, "modes": modes, "streaming_transports": transports,
                  "worker_errors": state["errors"], "interrupted": interrupted,
                  "correctness_qualified": False, "performance_qualified": False,
                  "failure_policy": "stop refill; drain owned in-flight calls with total deadlines; no retries"}
    if rows:
        require(accounting["phase_started_ns"] <= min(row["started_ns"] for row in rows)
                and max(row["call_finished_ns"] for row in rows) <= finished, "phase does not enclose request calls")
    return rows, accounting


def _distribution(values):
    if not values:
        return None
    ordered = sorted(values)
    return {"samples": len(values), "median": statistics.median(values),
            "p95": ordered[max(0, math.ceil(.95 * len(values)) - 1)],
            "p99": ordered[max(0, math.ceil(.99 * len(values)) - 1)], "max": ordered[-1]}


def summarize_phase(rows, accounting):
    """Descriptive complete/partial statistics; never an acceptance decision."""
    require(rows and len(rows) == accounting["attempted"] - accounting["unresolved_attempts"], "phase rows differ")
    first, last = min(row["started_ns"] for row in rows), max(row["finished_ns"] for row in rows)
    require(accounting["phase_started_ns"] <= first < last <= accounting["phase_finished_ns"], "invalid common wall interval")
    good = [row for row in rows if row["status"] == "success" and row["protocol_valid"]]
    output_tokens = sum(len(row["token_ids"]) for row in good)
    phase_wall = accounting["phase_finished_ns"] - accounting["phase_started_ns"]
    result = {**accounting, "common_wall_started_ns": first, "common_wall_finished_ns": last,
              "common_wall_ns": last - first, "successful_output_tokens": output_tokens,
              "observed_output_tokens_including_partial_failures": sum(len(row["token_ids"]) for row in rows),
              "successful_output_tokens_per_wall_second": output_tokens * 1e9 / (last - first),
              "common_wall_scope": "first request start through last terminal delivery (or failed-call completion)",
              "phase_wall_ns": phase_wall,
              "successful_output_tokens_per_phase_wall_second": output_tokens * 1e9 / phase_wall,
              "phase_wall_scope": "barrier release through all workers drained, including validation and EOF cleanup",
              "descriptive_statistics_only": True, "timing_scope": "client delivery; incomplete phases are not comparable",
              "tail_sample_warning": "P95/P99 are empirical sample statistics, not a stability or winner qualification"}
    for name in ("e2e_ns", "token_ttft_ns", "token_tpot_ns", "first_text_ns"):
        result[name.removesuffix("_ns") + "_ms"] = _distribution(
            [row["metrics"][name] / 1e6 for row in good if row["metrics"][name] is not None])
    result["token_itl_ms"] = _distribution([delta / 1e6 for row in good for delta in row["metrics"]["token_itl_ns"]])
    return result
