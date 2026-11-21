"""Verify whether internal cache bypass restores the intended startup coverage."""

import json
from pathlib import Path
import re
import sys

import comparison_base
from check_distribution_gpu import current_batch_shapes, prefix_cache_evidence
from run_case import GPUS, HOST_USER, MODEL_SHA, MODE_LOG, PRIVATE_LOG, SERVE, STOCK_IMAGE, WARM_BLOCKS
from verify_a100_candidate import digest, read, require
from warm_reference import checked_reservation, same_outputs
from workload import REQUEST_COUNTS, make_workloads

ORDER = ("off-cold", "off-warm", "shared", "private")
ORDERS = {"7": ORDER, "6": ORDER[::-1]}
CONTRASTS = {"shared_vs_off_warm": ("off-warm", "shared"),
             "private_vs_shared": ("shared", "private"),
             "private_vs_off_warm": ("off-warm", "private"),
             "private_vs_off_cold": ("off-cold", "private")}
STARTUP = re.compile(r"riley startup warm-up: ([0-9]+) internal requests in ([0-9]+\.[0-9]+) s")
ZERO_STATE = ("active_requests", "waiting_requests", "kv_allocated_blocks")


def check_build(build):
    tests = (build / "cpu-tests.log").read_text()
    names = ("w3_private_warmup_neither_reads_nor_publishes_cached_prefixes", "w3_prefix_cache_minimum_prompt_probe")
    require(all(f"test {name} ... ok" in tests for name in names)
            and "test result: ok. 2 passed; 0 failed" in tests, "diagnostic CPU regressions are incomplete")
    toolchain = (build / "toolchain.txt").read_text()
    require(toolchain.startswith("rustc 1.85.0") and "V12.4." in toolchain, "diagnostic toolchain differs")
    files = {name: check for check, name in
             (row.split("  ", 1) for row in (build / "build-result.sha256").read_text().splitlines())}
    require(all(digest(build / name) == checksum for name, checksum in files.items()), "build artifact differs")
    original = (build / "first-build/cpu-tests.log").read_text()
    require((build / "first-build/build.exit").read_text().strip() == "101" and "error[E0027]" in original
            and "prefix_cache_eligible" in original, "initial compiler failure does not match the preserved copy path")
    return {"input_sha256": digest(build / "build-inputs.sha256"), "result_sha256": digest(build / "build-result.sha256"),
            "artifacts": files, "cpu_checks": list(names), "first_cpu_build": {"exit": 101, "error": "E0027",
                                                                            "log_sha256": digest(build / "first-build/cpu-tests.log")}}


def check_case(root, case, index, owner, stock, parent, build_checks, original, workloads):
    enabled = case in ("shared", "private")
    warm = case != "off-cold"
    folder = root / f"{case}-gpu{index}"
    receipt = read(folder / "receipt.json")
    argv = [*SERVE, "--startup-warmup", "step-shapes-v1" if warm else "disabled"]
    if enabled:
        argv += ["--prefix-cache-blocks", str(WARM_BLOCKS)]
    require((folder / "exit").read_text().strip() == "0" and receipt["client_exit"] == receipt["shutdown_exit"] == 0
            and receipt["readiness"]["ready"] is True and receipt["readiness"]["accepting"] is True
            and receipt["startup_case"] == case and receipt["mode"] == ("filtered" if enabled else "off")
            and receipt["min_prompt_tokens"] == (128 if enabled else 0)
            and receipt["gpu_uuid"] == GPUS[index] and receipt["container_user"] == HOST_USER
            and receipt["server_arguments"] == argv and receipt["image_id"] == STOCK_IMAGE
            and receipt["source_revision"] == stock["source_revision"]
            and receipt["source_archive_sha256"] == stock["source_archive_sha256"]
            and receipt["parent_report_sha256"] == digest(parent)
            and receipt["build_result_sha256"] == build_checks["result_sha256"]
            and receipt["binary_sha256"] == build_checks["artifacts"]["target/release/riley"]
            and receipt["model_sha256"] == MODEL_SHA, "runtime did not use the requested diagnostic case")
    require(all(checked_reservation(folder / f"reservation-{when}.json", index) == owner for when in ("before", "after")),
            "one case changed the GPU lease")
    artifacts = receipt["artifacts_sha256"]
    require(set(artifacts) == {"client.log", "workload.json", "shutdown.json", "server.log"}
            and all(digest(folder / name) == sha for name, sha in artifacts.items()), "raw result changed")
    log = (folder / "server.log").read_text()
    require([line for line in log.splitlines() if line.startswith(PRIVATE_LOG)]
            == ([f"{PRIVATE_LOG}{str(case == 'private').lower()}"] if warm else [])
            and [line for line in log.splitlines() if line.startswith(MODE_LOG)]
            == ([f"{MODE_LOG}128"] if enabled else []), "actual startup or filter marker differs")
    match = STARTUP.findall(log)
    require(len(match) == int(warm), "startup time is missing or unexpected")
    raw = read(folder / "workload.json")
    require(raw.keys() == {"startup", *REQUEST_COUNTS}, "unexpected workload phases")
    startup = raw["startup"]["metrics"]
    shutdown = read(folder / "shutdown.json")
    require(all(startup[key] == shutdown[key] == 0 for key in ZERO_STATE)
            and shutdown["allocation"] and all(amount == 0 for amount in shutdown["allocation"].values()),
            "startup or shutdown left allocations in use")
    cache = prefix_cache_evidence({"initial": raw["startup"], "rss_samples": [raw["prime"], raw["rag"]],
                                   "final": raw["short"]}, shutdown, WARM_BLOCKS if enabled else 0)
    require(cache["status"] == "passed", "prefix cache snapshots or cleanup differ")
    cells = {name: comparison_base.check_cell(raw[name], bodies, name) for name, bodies in workloads.items()}
    exact = {name: same_outputs(original[name], raw[name]) for name in REQUEST_COUNTS}
    require(exact == REQUEST_COUNTS and cells["short"]["cached_requests"] == 0
            and (not enabled or cells["rag"]["cached_requests"] == 64 and cells["rag"]["cached_min"] >= 1536)
            and (enabled or not any(cell["cached_requests"] for cell in cells.values())),
            "diagnostic changed output or cache reuse")
    ready_shapes = current_batch_shapes(raw["startup"])
    short_end = current_batch_shapes(raw["short"])
    short_start = current_batch_shapes(raw["rag"])
    short_gpu = {str(rows): {"steps": values[0] - short_start[rows][0],
                             "gpu_ns": values[1] - short_start[rows][1]}
                 for rows, values in short_end.items() if values[0] > short_start[rows][0]}
    ttft = [row["ttft"] * 1000 for row in raw["short"]["rows"]]
    return raw, {"cells": cells, "exact_stock_outputs": exact,
                 "startup_cache": startup["prefix_cache"], "cache_lifetime": cache,
                 "startup_seconds_in_log": float(match[0][1]) if match else None,
                 "startup_requests_in_log": int(match[0][0]) if match else 0,
                 "container_start_to_ready_ms": receipt["container_start_to_ready_ms"],
                 "ready_shape_steps": {str(rows): values[0] for rows, values in ready_shapes.items() if values[0]},
                 "short_shapes_missing_at_ready": [rows for rows, values in short_end.items()
                                                   if values[0] > short_start[rows][0] and not ready_shapes[rows][0]],
                 "short_gpu_by_shape": short_gpu,
                 "short_first_32_p95_ms": comparison_base.p(ttft[:32], .95),
                 "short_remaining_64_p95_ms": comparison_base.p(ttft[32:], .95),
                 "receipt_sha256": digest(folder / "receipt.json"), "workload_sha256": artifacts["workload.json"],
                 "shutdown_sha256": artifacts["shutdown.json"]}


def verify(stage):
    evidence = stage.parent
    build = evidence / "w3-private-warm-build-20261121.xrfPErXB"
    parent = evidence / "w5-prefix-metrics-package-20261121.nEFyze5I/cpu-evidence/report.json"
    stock = read(parent)
    require(stock["status"] == "passed" and stock["source_revision"] == "3951d90c8df381daef2ab88fd18ac9e8a83c4554"
            and stock["image_id"] == STOCK_IMAGE, "diagnostic parent does not match the committed source")
    original = evidence / "w3-coder-rag-cache-20261121.kXvzFrJJ/off-first/workload.json"
    reference_hash = digest(original)
    require(reference_hash == "a77aa2092bd229a40038622d1a2de6768d2a8a6fdca159fbff42c20d9d6cfb7b", "reference output changed")
    data = read(original)
    build_checks = check_build(build)
    timeline = [line.split() for line in (stage / "campaign.log").read_text().splitlines() if line.startswith(("START ", "END "))]
    require(len(timeline) == 16 and all(row[-1] == "exit=0" for row in timeline if row[0] == "END"), "incomplete campaign")
    workloads = make_workloads()
    arms, comparisons, cohorts, reservations = {}, {}, {}, {}
    for index, order in ORDERS.items():
        expected_events = [[phase, f"{case}-gpu{index}"] for case in order for phase in ("START", "END")]
        actual_events = [row for row in timeline if row[1].endswith(f"-gpu{index}")]
        require([row[:2] for row in actual_events] == expected_events
                and [row[2] for row in actual_events] == sorted(row[2] for row in actual_events), "wrong case order")
        owner = checked_reservation(stage / f"gpu{index}-before.json", index)
        require(checked_reservation(stage / f"gpu{index}-after.json", index) == owner, "campaign replaced a GPU holder")
        reservations.update(owner)
        raw = {}
        for case in order:
            raw[case], arms[f"{case}-gpu{index}"] = check_case(stage, case, index, owner, stock, parent,
                                                             build_checks, data, workloads)
        comparisons[index] = {key: comparison_base.compare(raw[before], raw[after], arms[f"{before}-gpu{index}"],
                                                              arms[f"{after}-gpu{index}"])
                              for key, (before, after) in CONTRASTS.items()}
        cohorts[index] = {key: comparison_base.short_distribution(raw[before], raw[after])
                          for key, (before, after) in CONTRASTS.items()}

    def passes(bundle):
        short = bundle["short"]["change_percent"]
        rag = bundle["rag"]["change_percent"]
        return (short["ttft_p50_ms"] < 2 and short["ttft_p95_ms"] < 2 and short["tpot_p50_ms"] < 1
                and short["output_tok_s"] > -2 and rag["ttft_p95_ms"] < 0 and rag["tpot_p50_ms"] < 1)

    return {"status": "passed", "source_revision": stock["source_revision"], "image_id": STOCK_IMAGE,
            "remote_evidence": str(stage), "remote_build": str(build), "build": build_checks,
            "model_sha256": MODEL_SHA, "stock_workload_sha256": reference_hash, "reservations": reservations,
            "order": timeline, "runs": arms, "comparisons": comparisons, "short_cohorts": cohorts,
            "both_devices_pass": {key: all(passes(row[key]) for row in comparisons.values())
                                  for key in ("shared_vs_off_warm", "private_vs_off_warm", "private_vs_off_cold")}}


if __name__ == "__main__":
    print(json.dumps(verify(Path(sys.argv[1])), indent=2, allow_nan=False))
