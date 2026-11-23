"""Compare Coder-14B engines only for replay cells with matched inputs and write timing."""
import argparse
import json
import math
from pathlib import Path
import runpy

BASE = runpy.run_path(str(Path(__file__).resolve().parent.parent / "20261124-w3-package-coder14/verify.py"))
require = BASE["require"]
read = BASE["read"]
digest = BASE["digest"]
IMAGE = "sha256:1729f15a677422d552b9a3fd2352dfeb08dd4cccd3558680a016058d27191098"
REVISION = "0511b9280f62c3ee87c7a1344444423d8b4623c4"
BINARY = "8d4f8455bc0b9cccf8df3ee2feb2c606ffe2cf02dbf4bcdae6dcc2b033577e4b"
CPU = "bc10cbc7ffdc0164690f1e7ab1401cf26c50476f556ee7286b69b0e4dc382ae7"
MAX_WRITE_ERROR_P95_MS = 3.0


def p95(values):
    return sorted(values)[int(len(values) * .95)]


def replay_cell(out, label, plan, receipt):
    rows, stats = BASE["records"](out, label, plan["cell"], receipt)
    result = read(out / f"{label}.json")
    expected = plan["schedule"]
    require(min(item["offset_ns"] for item in expected) == 0, "schedule origin changed")
    first = min(row["sent_at"] for row in rows)
    errors, mismatches = [], []
    for row, archived in zip(rows, expected, strict=True):
        require(row["request_index"] == archived["request_index"]
                and row["input_sha256"] == archived["input_sha256"]
                and row["scheduled_write_ns"] == archived["offset_ns"]
                and row["start"] <= row["sent_at"] <= row["start"] + row["ttft"],
                f"RAG request or write schedule drifted in {label}")
        error = 1e3 * (row["sent_at"] - first - archived["offset_ns"] / 1e9)
        require(math.isclose(error, row["write_error_ms"], abs_tol=1e-9), "write error cannot be reproduced")
        errors.append(error)
        if any(row[key] != archived[key] for key in ("prompt_tokens", "completion_tokens", "finish")):
            mismatches.append(row["request_index"])
    require(mismatches == result["length_mismatches"]
            and p95([abs(error) for error in errors]) ==
            result["statistics"]["relative_write_error_abs_p95_ms"], "invalid replay receipt")
    return {"rows": rows, "statistics": stats, "write_errors_ms": errors, "length_mismatches": mismatches,
            "reported_itl_p99_ms": result["statistics"]["itl_p99_ms"]}


def arm(stage, index, engine, plans, holder):
    out = stage / f"gpu{index}/coder14-{engine}"
    receipt, plan = read(out / "receipt.json"), read(out / "plan.json")
    require(receipt["status"] == "passed" and receipt["gpu"] == BASE["GPUS"][index]
            and receipt["holder_pid"] == holder["pid"] and receipt["common_kv_blocks"] == 2576
            and receipt["kv_cache_bytes"] == 8103395328 and plan["cells"] == plans,
            f"GPU {index} {engine} did not use the pinned suite")
    server = plan["server"]
    if engine == "riley":
        identity = ("image_id", "source_revision", "riley_binary_sha256", "cpu_report_sha256", "checkpoint_sha256")
        require(tuple(receipt[key] for key in identity) == (IMAGE, REVISION, BINARY, CPU, BASE["CHECKPOINT"]),
                "the tested Riley package differs")
        require(server[server.index("--batch-token-budget") + 1] == "1024"
                and server[server.index("--prefill-chunk-tokens") + 1] == "1024"
                and "--prefix-cache-blocks" not in server, "Riley scheduler settings differ")
        shutdown = read(out / "shutdown.json")
        require(receipt["shutdown_exit"] == 0 and shutdown["allocation"] and not any(shutdown["allocation"].values())
                and all(shutdown[key] == 0 for key in ("active_requests", "waiting_requests", "kv_allocated_blocks")),
                "Riley retained native resources")
    else:
        require(receipt["holder_start_ticks"] == holder["start_ticks"] and receipt["kv_capacity_matches"]
                and receipt["reported_kv_tokens"] == [41216] and read(out / "server-exit.json")["exit"] in (0, -15),
                "vLLM capacity, GPU reservation or shutdown differs")
        require(server[server.index("--max-num-batched-tokens") + 1] == "2048"
                and "--no-enable-prefix-caching" in server, "vLLM scheduler settings differ")
    return {label: replay_cell(out, label, spec, receipt) for label, spec in plans.items()}


def pair(riley, vllm):
    riley_stats, vllm_stats = (run["statistics"] for run in (riley, vllm))
    riley_p95, vllm_p95 = (p95([abs(value) for value in run["write_errors_ms"]]) for run in (riley, vllm))
    between = p95([abs(first - second) for first, second in zip(riley["write_errors_ms"], vllm["write_errors_ms"], strict=True)])
    matched = not (riley["length_mismatches"] or vllm["length_mismatches"]) and max(riley_p95, vllm_p95, between) <= MAX_WRITE_ERROR_P95_MS
    ratios = {key: riley_stats[key] / vllm_stats[key] for key in BASE["METRICS"]}
    return {
        "requests_per_engine": len(riley["rows"]), "valid_matched_comparison": matched,
        "length_mismatches": {"riley": riley["length_mismatches"], "vllm": vllm["length_mismatches"]},
        "abs_write_error_p95_ms": {"riley": riley_p95, "vllm": vllm_p95, "between_engines": between},
        "riley": riley_stats, "vllm": vllm_stats, "riley_over_vllm": ratios,
        "ttft_p95_stage_met": ratios["ttft_p95_ms"] <= 1.40 if matched else None,
        "ttft_p95_final_met": ratios["ttft_p95_ms"] <= 1.15 if matched else None,
        "identical_output_text": sum(a["output_sha256"] == b["output_sha256"]
                                     for a, b in zip(riley["rows"], vllm["rows"], strict=True)),
    }


def write_components(rows):
    first_write = min(rows, key=lambda row: row["sent_at"])
    initial_connect_s = first_write["sent_at"] - first_write["start"]
    before_start, after_start = [], []
    above_limit = before_start_dominates = 0
    for row in rows:
        scheduled_ms = row["scheduled_write_ns"] / 1e6
        before = 1e3 * (row["start"] - first_write["start"]) - scheduled_ms
        after = 1e3 * (row["sent_at"] - row["start"] - initial_connect_s)
        error = row["write_error_ms"]
        require(math.isclose(before + after, error, abs_tol=1e-8), "write timing components do not add up")
        before_start.append(abs(before))
        after_start.append(abs(after))
        if abs(error) > MAX_WRITE_ERROR_P95_MS:
            above_limit += 1
            before_start_dominates += int(abs(before) > abs(after))
    return {"before_request_start_abs_p95_ms": p95(before_start),
            "start_to_write_relative_abs_p95_ms": p95(after_start),
            "start_to_write_abs_p95_ms": p95([1e3 * (row["sent_at"] - row["start"]) for row in rows]),
            "requests_beyond_3ms": above_limit, "beyond_3ms_dominated_by_request_start": before_start_dominates}


def verify(stage):
    require((stage / "campaign.exit").read_text().strip() == "0" and
            (stage / "vllm-version.txt").read_text().strip() == "0.27.1+cu129", "RAG replay did not complete")
    require(digest(stage / "schedules.json") ==
            "9b4db373c4b9b7307dd92e63460d37674986873cf391438b7bfb850bf17b8572", "archived write schedule differs")
    frozen = read(stage / "schedules.json")
    BASE["confirm_order"](stage)
    holders = {index: BASE["reservations"](stage, index) for index in BASE["ORDER"]}
    results = {}
    for index, engines in BASE["ORDER"].items():
        plans = frozen["schedules"][str(index)]
        expected = {f"rag-c8-s{seed}-{origin}-arrivals" for seed in (1, 2) for origin in ("riley", "vllm")}
        require(set(plans) == expected, "both seed and timeline origins are required")
        runs = {engine: arm(stage, index, engine, plans, holders[index]) for engine in engines}
        results[index] = {
            label: {**pair(runs["riley"][label], runs["vllm"][label]),
                    "write_components": {engine: write_components(runs[engine][label]["rows"]) for engine in engines}}
            for label in plans
        }
    return {"source_revision": REVISION, "image_id": IMAGE, "vllm_version": "0.27.1+cu129",
            "max_write_error_p95_ms": MAX_WRITE_ERROR_P95_MS, "by_gpu": results,
            "origin_differences": frozen["paired_origin_differences"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.stage), indent=2, allow_nan=False))
