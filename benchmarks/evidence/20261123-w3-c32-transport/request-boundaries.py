"""Attribute client-visible delay, including controls with server tracing disabled."""

from hashlib import sha256
import json
from math import isclose, isfinite
from pathlib import Path
import sys

from verify_transport import IDENTITY, CLIENT_REGIONS, compare, read_arm, require


def read_request_boundaries(folder, trace_mode):
    receipt = json.loads((folder / "receipt.json").read_bytes())

    def read_file(name):
        data = (folder / name).read_bytes()
        require(sha256(data).hexdigest() == receipt["artifacts_sha256"][name], f"{name} changed in {folder.name}")
        return data

    workload = json.loads(read_file("workload.json"))["short"]
    timeline = read_file("server.log").decode().splitlines()
    snapshots, observations = [], {}
    for line in timeline:
        if line.startswith("W3-OBS-SNAPSHOT "):
            snapshots.append(tuple(map(int, line.split()[1:])))
        elif line.startswith("W3-OBS "):
            _, request, admit, visible, tokens, status = line.split()
            require(request not in observations and status == "Completed", "unexpected request observation")
            observations[request] = tuple(map(int, (admit, visible, tokens)))
    rows = workload["rows"]
    stats = workload["statistics"]
    require(snapshots == [(164, 0)] and len(observations) == 164 and len(rows) == stats["ok"] == 96,
            "incomplete short C32 request observations")
    traced = None
    if trace_mode == "transport":
        _, _, traced = read_arm(folder, receipt["short_burst_decode_limit"])
    result = []
    for index, row in enumerate(rows):
        admit, visible, tokens = observations[row["response_id"]]
        require(row["status"] == 200 and not row["error"] and tokens == row["completion_tokens"],
                "server observation does not belong to the measured response")
        client = row["transport_ms"]
        require(set(client) == set(CLIENT_REGIONS) and all(isfinite(ms) and ms >= 0 for ms in client.values()),
                "missing client transport timing")
        before_write = 1000 * (row["sent_at"] - row["start"])
        ttft_ms = 1000 * row["ttft"]
        require(isclose(before_write + sum(client.values()), ttft_ms, rel_tol=0, abs_tol=0.00002),
                "client boundaries do not add up to TTFT")
        phases = {
            "client_before_write": before_write,
            "backend_submit": admit / 1_000_000,
            "admitted_to_first_visible": visible / 1_000_000,
            # These are residuals. A delayed client read can move time
            # across the boundary, so the terms may be signed.
            "write_to_status_less_submit": client["write_to_status_line"] - admit / 1_000_000,
            "status_to_first_text_less_visible": sum(client[key] for key in CLIENT_REGIONS[1:]) - visible / 1_000_000,
        }
        require(isclose(sum(phases.values()), ttft_ms, rel_tol=0, abs_tol=0.00002), "phase sum differs from TTFT")
        result.append({
            "request": index + 1,
            "ttft_ms": ttft_ms,
            "other_after_write_ms": phases["write_to_status_less_submit"] + phases["status_to_first_text_less_visible"],
            "phases_ms": phases,
            "client_transport_ms": client,
            **({"absolute_trace": traced[index]["parts"]} if traced is not None else {}),
        })
    order = sorted(result, key=lambda row: row["ttft_ms"])
    largest_other = max(result, key=lambda row: row["other_after_write_ms"])
    return {
        "n": len(result),
        "p50_ttft": order[int(.50 * len(order))],
        "p95_ttft": order[int(.95 * len(order))],
        "largest_unassigned_after_write": largest_other,
        "write_error_p95_ms": stats["write_error_p95_ms"],
        "workload_sha256": receipt["artifacts_sha256"]["workload.json"],
        "server_log_sha256": receipt["artifacts_sha256"]["server.log"],
    }, rows, result


def report(root):
    total = 0
    findings = {}
    for gpu in (6, 7):
        finding, paired, seen = {}, {}, None
        for mode in ("control", "transport"):
            timing_rows = []
            for limit in (0, 8):
                name = f"private-short-d{limit}-{mode}-gpu{gpu}"
                row, requests, timings = read_request_boundaries(root / name, mode)
                current = [tuple(request[key] for key in IDENTITY) for request in requests]
                require(seen is None or seen == current, "input or output changed across controls and traces")
                seen = current
                finding[name] = row
                timing_rows.append([{"ttft": item["ttft_ms"], "parts": item["phases_ms"]} for item in timings])
                total += len(requests)
            paired[mode] = {label: compare(*timing_rows, quantile)
                            for label, quantile in (("p50", .50), ("p95", .95))}
        findings[str(gpu)] = {"arms": finding, "paired": paired}
    return {"status": "verified", "short_requests": total, "remote_evidence": str(root), "gpus": findings}


if __name__ == "__main__":
    print(json.dumps(report(Path(sys.argv[1])), indent=2, allow_nan=False))
