"""Check a paired C32 transport trace from the same diagnostic binary and GPU."""

from hashlib import sha256
import json
from math import isclose, isfinite
from pathlib import Path
import re
import sys

SERVER_MARKS = (
    "http_parsed", "transport_submit", "transport_admitted", "head_write_started",
    "head_write_returned", "visible", "text_write_returned",
)
CLIENT_REGIONS = (
    "write_to_status_line", "status_line_to_headers_end", "headers_end_to_first_data_line",
    "first_data_line_to_first_text",
)
IDENTITY = ("input_sha256", "prompt_ids_sha256", "token_ids_sha256", "output_sha256",
            "finish", "prompt_tokens", "completion_tokens", "cached_tokens")
SCOPE = ("source_revision", "source_archive_sha256", "binary_sha256", "build_result_sha256",
         "model_sha256", "image_id", "gpu_uuid", "http_workers", "steady_budget", "server_arguments")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_arm(folder, limit):
    receipt = json.loads((folder / "receipt.json").read_bytes())
    require(receipt["trace_mode"] == "transport" and receipt["stage_trace"] is True
            and receipt["short_burst_decode_limit"] == limit
            and receipt["client_exit"] == receipt["shutdown_exit"] == 0, "unexpected trace mode or failed replay")
    hashes = receipt["artifacts_sha256"]
    checked = {}
    for name in ("workload.json", "server.log", "shutdown.json"):
        raw = (folder / name).read_bytes()
        require(sha256(raw).hexdigest() == hashes[name], f"changed {name}")
        checked[name] = raw
    shutdown = json.loads(checked["shutdown.json"])
    require(all(shutdown[key] == 0 for key in ("active_requests", "waiting_requests", "kv_allocated_blocks"))
            and shutdown["allocation"] and not any(shutdown["allocation"].values()), "resources remained after shutdown")
    phases, losses = {}, []
    for line in checked["server.log"].decode().splitlines():
        if line.startswith("W3-REQUEST-PHASE "):
            _, request, name, stamp = line.split()
            existing = phases.setdefault(int(request), {})
            require(name in SERVER_MARKS and name not in existing, "unknown or duplicate transport mark")
            existing[name] = int(stamp)
        elif line.startswith("W3-DROPPED "):
            losses.append(int(line.split()[1]))
    require(losses == [0], "the transport trace dropped records")
    rows = json.loads(checked["workload.json"])["short"]["rows"]
    require(len(rows) == 96 and len({row["response_id"] for row in rows}) == 96, "invalid C32 request coverage")
    timings = []
    for row in rows:
        request = re.fullmatch(r"cmpl-([a-f0-9]{16})", row["response_id"])
        require(request is not None and row["status"] == 200 and row["error"] is False
                and row["cached_tokens"] == 0, "expected a complete uncached C32 response")
        timings.append(join(row, phases[int(request[1], 16)]))
    return receipt, rows, timings


def join(row, server):
    require(set(server) == set(SERVER_MARKS), "incomplete transport trace")
    marks = [server[name] for name in SERVER_MARKS]
    require(marks == sorted(marks), "server transport marks are out of order")
    parsed, submit, admitted, head_start, head_end, visible, written = (stamp / 1_000_000 for stamp in marks)
    request_start = row["start"] * 1000
    client_write = row["sent_at"] * 1000
    client_text = request_start + row["ttft"] * 1000
    client = row["transport_ms"]
    require(set(client) == set(CLIENT_REGIONS) and all(isfinite(delta) and delta >= 0 for delta in client.values()),
            "invalid client transport timing")
    require(isclose(client_write + sum(client.values()), client_text, rel_tol=0, abs_tol=0.00002),
            "client phase durations do not match first text")
    status_line = client_write + client["write_to_status_line"]
    # Monotonic clocks are shared by both peers on the Linux host. A socket
    # write can be visible to the peer before the writing thread returns.
    require(request_start <= parsed and head_start <= status_line + 0.00002 and visible <= client_text + 0.00002,
            "client and server timestamps do not describe the same exchange")
    return {
        "ttft": row["ttft"] * 1000,
        "parts": {
            "client_before_write": client_write - request_start,
            "write_to_server_parse": parsed - client_write,
            "server_parse_to_submit": submit - parsed,
            "backend_submit": admitted - submit,
            "admitted_to_first_visible": visible - admitted,
            "first_server_visible_to_client": client_text - visible,
        },
        "overlapping_observations": {
            **client,
            "server_head_write": head_end - head_start,
            "server_first_text_write": written - visible,
            "write_return_to_client_text": client_text - written,
        },
    }


def compare(base, candidate, quantile):
    rank = min(int(len(base) * quantile), len(base) - 1)
    a, b = (sorted(range(len(arm)), key=lambda index: arm[index]["ttft"])[rank]
            for arm in (base, candidate))
    old, fixed_request, new = base[a], base[b], candidate[b]
    changes = {key: new["parts"][key] - fixed_request["parts"][key] for key in new["parts"]}
    rank_shift = fixed_request["ttft"] - old["ttft"]
    delta = new["ttft"] - old["ttft"]
    require(isclose(delta, sum(changes.values()) + rank_shift, rel_tol=0, abs_tol=0.00002),
            "measured phases and rank shift do not explain the quantile change")
    return {
        "baseline_ms": old["ttft"], "candidate_ms": new["ttft"],
        "baseline_rank_request": a + 1, "candidate_rank_request": b + 1,
        "paired_ms_before": fixed_request, "paired_ms_after": new,
        "paired_deltas_ms": changes, "baseline_rank_shift_ms": rank_shift,
    }


def verify(baseline, candidate):
    old, new = read_arm(baseline, 0), read_arm(candidate, 8)
    require(all(old[0][key] == new[0][key] for key in SCOPE), "the compared replay changed its serving scope")
    require(all(all(a[key] == b[key] for key in IDENTITY) for a, b in zip(old[1], new[1], strict=True)),
            "the compared replay changed the request or response")
    return {
        "status": "verified", "matched_requests": len(old[1]),
        "scope": {key: old[0][key] for key in SCOPE},
        "arm_artifacts_sha256": [old[0]["artifacts_sha256"], new[0]["artifacts_sha256"]],
        "p50": compare(old[2], new[2], .50), "p95": compare(old[2], new[2], .95),
    }


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: verify_transport.py BASELINE_TRACE_DIRECTORY CANDIDATE_TRACE_DIRECTORY")
    print(json.dumps(verify(*map(Path, sys.argv[1:])), indent=2, allow_nan=False))
