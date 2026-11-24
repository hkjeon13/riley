"""Estimate fixed-schedule dense GEMM headroom in the captured Coder-14B trace."""

import gzip
from hashlib import sha256
import json
from pathlib import Path
from statistics import median

from summarize import LAYERS, SIGNATURES, TRACE_SHA256


HERE = Path(__file__).resolve().parent
PROFILE = HERE.parent / "20261119-w3-rag-profile" / "coder14-profile"
SOURCE = "https://www.nvidia.com/en-us/data-center/a100/"
PEAK_TFLOPS = 312
ROWS = 1024
MLP = ("gate_up", "down")


def read(path):
    return json.loads(path.read_bytes())


def p95(values):
    ordered = sorted(values)
    return ordered[int(len(ordered) * .95)]


def estimate():
    raw = HERE / "step-times.json.gz"
    with gzip.open(raw, "rt") as handle:
        profile = json.load(handle)
    receipt = read(PROFILE / "receipt.json")
    workload_path = PROFILE / "rag-c8-s1.json"
    workload = read(workload_path)
    legacy = read(PROFILE / "rag-c8-s1.profile.json")
    if profile["trace_sha256"] != receipt["sqlite_sha256"] or profile["trace_sha256"] != TRACE_SHA256:
        raise ValueError("the trace identity differs from the server receipt")
    floor_ns = {name: 2 * ROWS * LAYERS * k * n / (PEAK_TFLOPS * 1000)
                for name, (_, _, (k, n)) in SIGNATURES.items()}
    steps = profile["steps"]
    if (len(steps) != 492 or any(step["span_ns"] < step["kernel_union_ns"] for step in steps)
            or any(a["start_ns"] + a["span_ns"] > b["start_ns"] for a, b in zip(steps, steps[1:]))):
        raise ValueError("unexpected 1,024-row step coverage")
    clocks = profile["clock_marks_ns"]
    offsets = []
    for anchor in workload["anchors"]:
        before, after = anchor["before_ns"], anchor["after_ns"]
        if not 0 <= after - before <= 1_000_000:
            raise ValueError("unusable clock bracket")
        offsets.append(clocks[anchor["name"]] - (before + after) // 2)
    if len(offsets) != 2 or abs(offsets[1] - offsets[0]) > 1_000_000:
        raise ValueError("clock calibration differs across the workload")
    offset = sum(offsets) // 2

    def headroom(step, names):
        return sum(max(0, step[name] - floor_ns[name]) for name in names)

    requests = []
    for request, prior in zip(workload["measured"], legacy["measurements"], strict=True):
        if request["request_index"] != prior["request_index"]:
            raise ValueError("request order changed")
        begin = round(request["sent_at"] * 1e9) + offset
        end = round((request["start"] + request["ttft"]) * 1e9) + offset
        overlaps = [(step, min(end, step["start_ns"] + step["span_ns"]) - max(begin, step["start_ns"]))
                    for step in steps]
        overlaps = [(step, ns) for step, ns in overlaps if ns > 0]
        requests.append({
            "request_index": request["request_index"],
            "ttft_ms": request["ttft"] * 1000,
            "overlapping_wide_steps": len(overlaps),
            "mlp_optimistic_savings_ms": sum(min(ns, headroom(step, MLP)) for step, ns in overlaps) / 1e6,
            "all_four_optimistic_savings_ms": sum(
                min(ns, headroom(step, SIGNATURES)) for step, ns in overlaps
            ) / 1e6,
        })
    if len(requests) != 64:
        raise ValueError("expected 64 requests in the recorded RAG C8 cell")
    measured_p95 = p95(row["ttft_ms"] for row in requests)
    if abs(workload["statistics"]["ttft_p95_ms"] - measured_p95) > 0.00001:
        raise ValueError("the reconstructed p95 differs from the original analysis")
    p95_request = next(row for row in requests if row["ttft_ms"] == measured_p95)
    tail = sorted(requests, key=lambda row: row["ttft_ms"])[-7:]
    return {
        "trace_sha256": TRACE_SHA256,
        "step_times_sha256": sha256(raw.read_bytes()).hexdigest(),
        "workload_sha256": sha256(workload_path.read_bytes()).hexdigest(),
        "measured_wide_steps": len(steps),
        "measured_layers": len(steps) * LAYERS,
        "median_step_ms": profile["median_span_ms"],
        "median_projection_ms_per_step": profile["median_projection_ms_per_step"],
        "dense_peak_tflops": PEAK_TFLOPS,
        "dense_peak_source": SOURCE,
        "ideal_projection_ms_per_step": {name: ns / 1e6 for name, ns in floor_ns.items()},
        "median_mlp_optimistic_savings_ms_per_step": median(headroom(step, MLP) for step in steps) / 1e6,
        "median_all_four_optimistic_savings_ms_per_step": (
            median(headroom(step, SIGNATURES) for step in steps) / 1e6
        ),
        "rag_c8_seed1_p95_request": p95_request,
        "rag_c8_seed1_slowest_seven": tail,
        "rag_c8_seed1_optimistic_p95_ms": {
            target: p95(row["ttft_ms"] - row[f"{target}_optimistic_savings_ms"] for row in requests)
            for target in ("mlp", "all_four")
        },
    }


if __name__ == "__main__":
    print(json.dumps(estimate(), indent=2, allow_nan=False))
