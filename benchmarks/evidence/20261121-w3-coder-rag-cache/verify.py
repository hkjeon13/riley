"""Independently pair the raw requests, validate controls, and calculate regressions."""

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from statistics import fmean
import sys

from current_binding import check_reservations
from run_arm import GPU, HOST_USER, MODEL_SHA, SERVE, STOCK_IMAGE
from verify_a100_candidate import digest, read, require
from workload import REQUEST_COUNTS, WARM_BLOCKS, make_workloads


ARMS = ("off-first", "on-first", "on-second", "off-second")
OUTPUT = ("prompt_tokens", "completion_tokens", "finish", "prompt_ids_sha256", "token_ids_sha256", "output_sha256")


def p(values, quantile):
    values = sorted(values)
    return values[min(int(len(values) * quantile), len(values) - 1)]


def check_cell(dataset, bodies, name):
    count = REQUEST_COUNTS[name]
    samples = dataset["rows"]
    stats = dataset["statistics"]
    require(len(samples) == stats["ok"] == count and stats["errors"] == stats["rejected_429"] == 0,
            f"{name} completion or count differs")
    require(all(row["error"] is False and row["status"] == 200 and row["finish"] in ("length", "stop")
                and row["input_sha256"] == sha256(json.dumps(body).encode()).hexdigest()
                and 0 < row["completion_tokens"] <= body["max_tokens"]
                and 0 <= row["cached_tokens"] < row["prompt_tokens"] and row["cached_tokens"] % 16 == 0
                for row, body in zip(samples, bodies, strict=True)), f"{name} input or usage changed")
    metrics = dataset["metrics"]
    require(metrics["active_requests"] == metrics["waiting_requests"] == metrics["kv_allocated_blocks"] == 0,
            f"{name} failed to drain")
    ttft_ms = [row["ttft"] * 1000 for row in samples]
    latencies = {"ttft_p50_ms": p(ttft_ms, .50), "ttft_p95_ms": p(ttft_ms, .95),
                 "tpot_p50_ms": 1000 * p([row["tpot"] for row in samples], .50),
                 "output_tok_s": sum(row["completion_tokens"] for row in samples) / stats["wall_s"]}
    require(all(abs(stats[key] - value) < .00001 for key, value in latencies.items()), "summary differs from raw rows")
    cached = [row["cached_tokens"] for row in samples]
    timings = {}
    if name == "rag":
        origin = samples[0]["sent_at"] - samples[0]["send_lateness_ms"] / 1000
        lateness = [(row["sent_at"] - (origin + (index // 8) * 6 + (index % 8) * .050)) * 1000
                    for index, row in enumerate(samples)]
        require(all(abs(actual - row["send_lateness_ms"]) < .001
                    for actual, row in zip(lateness, samples, strict=True)), "fixed arrival timetable changed")
        timings["send_error_p95_ms"] = p([abs(value) for value in lateness], .95)
        require(timings["send_error_p95_ms"] <= 5, "fixed-arrival send error exceeded the replay guard")
    return {**latencies, **timings, "requests": count, "cached_requests": sum(value > 0 for value in cached),
            "cached_min": min(cached), "cached_mean": fmean(cached), "cached_max": max(cached),
            "prompt_tokens_min": min(row["prompt_tokens"] for row in samples),
            "prompt_tokens_max": max(row["prompt_tokens"] for row in samples)}


def check_arm(stage, arm, workloads, cpu):
    path = stage / arm
    receipt = read(path / "receipt.json")
    stock = read(cpu)
    cache = arm.startswith("on-")
    require(receipt["status"] == "passed" and receipt["image_id"] == STOCK_IMAGE
            and receipt["source_revision"] == stock["source_revision"]
            and receipt["source_archive_sha256"] == stock["source_archive_sha256"]
            and receipt["cpu_report_sha256"] == digest(cpu) and receipt["binary_sha256"] == stock["binaries"]["/opt/riley/bin/riley"]
            and receipt["gpu_uuid"] == GPU and receipt["model_sha256"] == MODEL_SHA and receipt["shutdown_exit"] == 0
            and receipt["container_user"] == HOST_USER and not HOST_USER.startswith("0:")
            and receipt["server_arguments"] == [*SERVE, *(["--prefix-cache-blocks", str(WARM_BLOCKS)] if cache else [])],
            "runtime identity or configuration differs")
    require(all(digest(path / name) == value for name, value in receipt["artifacts_sha256"].items()),
            "arm artifacts differ")
    reservation = check_reservations(path, "after")
    require(reservation == check_reservations(path, "before") and receipt["gpu_holder"] == str(reservation["7"]["pid"]),
            "reservation ownership changed")
    shutdown = read(path / "shutdown.json")
    require(all(shutdown[key] == 0 for key in ("active_requests", "waiting_requests", "kv_allocated_blocks"))
            and shutdown["allocation"] and all(value == 0 for value in shutdown["allocation"].values()),
            "GPU or KV allocations remained at exit")
    raw = read(path / "workload.json")
    cells = {name: check_cell(raw[name], bodies, name) for name, bodies in workloads.items()}
    require(cache or not any(cell["cached_requests"] for cell in cells.values()), "cache-off condition served cached tokens")
    return raw, {"cells": cells, "reservation": reservation, "receipt_sha256": digest(path / "receipt.json"),
                 "workload_sha256": digest(path / "workload.json"),
                 "shutdown": {key: shutdown[key] for key in ("active_requests", "waiting_requests",
                                                             "kv_allocated_blocks", "allocation")}}


def compare(before, after, previous, current):
    result = {}
    for name in REQUEST_COUNTS:
        baseline, candidate = before[name]["rows"], after[name]["rows"]
        matched = sum(all(left[key] == right[key] for key in OUTPUT) for left, right in zip(baseline, candidate, strict=True))
        prior, updated = previous["cells"][name], current["cells"][name]
        changes = {key: 100 * (updated[key] / prior[key] - 1)
                   for key in ("ttft_p50_ms", "ttft_p95_ms", "tpot_p50_ms", "output_tok_s")}
        result[name] = {"matched": matched, "requests": len(baseline), "change_percent": changes}
    return result


def short_distribution(before, after):
    off, on = before["short"]["rows"], after["short"]["rows"]
    result = {"cached_tokens_histogram": dict(sorted(Counter(row["cached_tokens"] for row in on).items()))}
    for key, indexes in (("first_32", range(32)), ("remaining_64", range(32, len(off)))):
        old = [1000 * off[index]["ttft"] for index in indexes]
        new = [1000 * on[index]["ttft"] for index in indexes]
        differences = [actual - baseline for baseline, actual in zip(old, new, strict=True)]
        result[key] = {"off_p50_ms": p(old, .50), "on_p50_ms": p(new, .50),
                       "paired_difference_p50_ms": p(differences, .50),
                       "slower_requests": sum(value > 0 for value in differences)}
    return result


def verify(stage):
    def order(path):
        return [line.split() for line in (path / "campaign.log").read_text().splitlines()
                if line.startswith(("START ", "END "))]

    initial = stage if (stage / "off-first").is_dir() else stage.parent / "w3-coder-rag-cache-20261121.kXvzFrJJ"
    invalid_attempt = None
    if initial == stage:
        history = order(stage)
    else:
        attempted = order(initial)
        require([row[:2] for row in attempted[-2:]] == [["START", "on-second"], ["END", "on-second"]]
                and attempted[-1][-1] == "exit=1" and "fixed arrival replay missed the send error guard"
                in (initial / "on-second/client.log").read_text(), "missing evidence of invalidated send timing")
        history = [*attempted[:-2], *order(stage)]
        invalid_attempt = {"directory": str(initial / "on-second"), "reason": "fixed_arrival_send_p95_exceeded_5_ms",
                           "client_log_sha256": digest(initial / "on-second/client.log")}
    require([row[:2] for row in history] == [[phase, arm] for arm in ARMS for phase in ("START", "END")]
            and all(row[-1] == "exit=0" for row in history if row[0] == "END")
            and [row[2] for row in history] == sorted(row[2] for row in history), "ABBA sequence changed")
    workloads = make_workloads()
    cpu = stage.parent / "w5-current-package-20261121.f6SEbjvA/cpu-evidence/report.json"
    raw, results = {}, {}
    for arm in ARMS:
        root = initial if arm.endswith("-first") else stage
        raw[arm], results[arm] = check_arm(root, arm, workloads, cpu)
    require(all(row["reservation"] == results["off-first"]["reservation"] for row in results.values()),
            "reservations changed between arms")
    comparisons = (("off_then_on", "off-first", "on-first"), ("on_then_off", "off-second", "on-second"))
    pairs = {label: compare(raw[off], raw[on], results[off], results[on]) for label, off, on in comparisons}
    short = {label: short_distribution(raw[off], raw[on]) for label, off, on in comparisons}
    control = compare(raw["off-first"], raw["off-second"], results["off-first"], results["off-second"])
    candidate = compare(raw["on-first"], raw["on-second"], results["on-first"], results["on-second"])
    checks = {
        "exact_outputs": all(cell["matched"] == cell["requests"] for source in (*pairs.values(), control, candidate)
                             for cell in source.values()),
        "warmed_all_four_prefixes": all(results[arm]["cells"]["rag"]["cached_min"] >= 1536 for arm in ARMS[1:3]),
        "rag_ttft_p95_improved_in_both_orders": all(pair["rag"]["change_percent"]["ttft_p95_ms"] < 0 for pair in pairs.values()),
        "decode_tpot_regression_below_1_percent": all(pair[name]["change_percent"]["tpot_p50_ms"] < 1
                                                    for pair in pairs.values() for name in ("rag", "short")),
        "short_ttft_and_throughput_regression_below_2_percent": all(
            pair["short"]["change_percent"]["ttft_p50_ms"] < 2
            and pair["short"]["change_percent"]["ttft_p95_ms"] < 2
            and pair["short"]["change_percent"]["output_tok_s"] > -2 for pair in pairs.values()),
    }
    return {"first_pair_remote_evidence": str(initial), "reverse_pair_remote_evidence": str(stage),
            "invalid_attempt": invalid_attempt, "image_id": STOCK_IMAGE, "model_sha256": MODEL_SHA,
            "run_order": history, "cache_blocks": WARM_BLOCKS, "runs": results, "pairs": pairs,
            "short_ttft_by_submission": short, "checks": checks,
            "fixed_arrival_workload_meets_trial_guards": all(checks.values())}


if __name__ == "__main__":
    print(json.dumps(verify(Path(sys.argv[1])), indent=2, allow_nan=False))
