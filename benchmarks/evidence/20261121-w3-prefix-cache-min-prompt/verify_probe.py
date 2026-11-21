"""Verify the three-condition GPU comparison against the previous stock-image outputs."""

import json
from pathlib import Path
import sys

from check_distribution_gpu import batch_shape_evidence, current_batch_shapes
from comparison_base import OUTPUT, check_cell, compare, short_distribution
from current_binding import check_reservations
from run_arm import GPU, HOST_USER, MODEL_SHA, SERVE, STOCK_IMAGE
from run_probe import ARMS as PROBE_ARMS, MODE_LOG
from verify_a100_candidate import digest, read, require
from workload import REQUEST_COUNTS, WARM_BLOCKS, make_workloads


def short_gpu_shapes(raw):
    start, end = raw["rag"], raw["short"]
    summary = batch_shape_evidence({"initial": start, "rss_samples": [], "final": end})
    require(summary["status"] == "passed", summary.get("detail", "invalid batch-shape evidence"))
    before, after = current_batch_shapes(start), current_batch_shapes(end)
    buckets = {}
    for dense, (hits, total, _) in after.items():
        steps, gpu_ns = hits - before[dense][0], total - before[dense][1]
        if steps:
            buckets[str(dense)] = {"steps": steps, "gpu_execution_ns": gpu_ns}
    return {"committed_steps": summary["committed_steps"], "gpu_execution_ns": summary["gpu_execution_ns"],
            "by_dense_rows": buckets}


def probe_arm(stage, arm, build, workloads, cpu):
    path = stage / arm
    receipt = read(path / "receipt.json")
    stock = read(cpu)
    mode = arm.split("-")[0]
    minimum = 128 if mode == "filtered" else 0
    expected = [*SERVE, *(["--prefix-cache-blocks", str(WARM_BLOCKS)] if mode != "off" else [])]
    require((path / "exit").read_text().strip() == "0" and receipt["client_exit"] == receipt["shutdown_exit"] == 0,
            f"{arm} did not complete")
    require(receipt["mode"] == mode and receipt["min_prompt_tokens"] == minimum
            and receipt["image_id"] == STOCK_IMAGE and receipt["source_revision"] == stock["source_revision"]
            and receipt["source_archive_sha256"] == stock["source_archive_sha256"]
            and receipt["parent_report_sha256"] == digest(cpu)
            and receipt["build_result_sha256"] == digest(build / "build-result.sha256")
            and receipt["binary_sha256"] == digest(build / "target/release/riley")
            and receipt["model_sha256"] == MODEL_SHA and receipt["gpu_uuid"] == GPU
            and receipt["container_user"] == HOST_USER and receipt["server_arguments"] == expected,
            f"{arm} source or startup settings differ")
    require(receipt["readiness"]["ready"] is True and receipt["readiness"]["accepting"] is True,
            "service never became ready")
    artifacts = receipt["artifacts_sha256"]
    require(set(artifacts) == {"client.log", "workload.json", "shutdown.json", "server.log"}
            and all(digest(path / name) == value for name, value in artifacts.items()), "runtime evidence changed")
    markers = [line for line in (path / "server.log").read_text().splitlines() if line.startswith(MODE_LOG)]
    require(markers == ([] if mode == "off" else [f"{MODE_LOG}{minimum}"]), "runtime flag was not applied")
    reservation = check_reservations(path, "after")
    require(reservation == check_reservations(path, "before"), "GPU reservation ownership changed")
    shutdown = read(path / "shutdown.json")
    require(all(shutdown[key] == 0 for key in ("active_requests", "waiting_requests", "kv_allocated_blocks"))
            and shutdown["allocation"] and all(value == 0 for value in shutdown["allocation"].values()),
            "GPU or KV allocated state remained at shutdown")
    raw = read(path / "workload.json")
    cells = {name: check_cell(raw[name], payloads, name) for name, payloads in workloads.items()}
    return raw, {"cells": cells, "short_gpu_shapes": short_gpu_shapes(raw),
                 "reservation": reservation, "receipt_sha256": digest(path / "receipt.json"),
                 "workload_sha256": artifacts["workload.json"], "binary_sha256": receipt["binary_sha256"],
                 "shutdown": {key: shutdown[key] for key in ("active_requests", "waiting_requests",
                                                              "kv_allocated_blocks", "allocation")}}


def verify_probe(stage):
    build = stage.parent / "w3-prefix-min-build-20261121.vdX2vOQA"
    prior = stage.parent / "w3-coder-rag-cache-20261121.kXvzFrJJ/off-first"
    cpu = stage.parent / "w5-current-package-20261121.f6SEbjvA/cpu-evidence/report.json"
    progression = [line.split() for line in (stage / "campaign.log").read_text().splitlines()
                   if line.startswith(("START ", "END "))]
    require([row[:2] for row in progression] == [[phase, arm] for arm in PROBE_ARMS for phase in ("START", "END")]
            and [row[2] for row in progression] == sorted(row[2] for row in progression)
            and all(row[-1] == "exit=0" for row in progression if row[0] == "END"),
            "comparison order or exit status changed")
    workloads = make_workloads()
    raw, results = {}, {}
    for arm in PROBE_ARMS:
        raw[arm], results[arm] = probe_arm(stage, arm, build, workloads, cpu)
    require(all(item["reservation"] == results["off-first"]["reservation"] for item in results.values()),
            "GPU reservation owners changed across the comparison")

    stock = read(prior / "receipt.json")
    stock_path = prior / "workload.json"
    require(stock["image_id"] == STOCK_IMAGE and stock["model_sha256"] == MODEL_SHA
            and stock["binary_sha256"] == read(cpu)["binaries"]["/opt/riley/bin/riley"]
            and stock["artifacts_sha256"]["workload.json"] == digest(stock_path), "stock-image outputs differ from receipt")
    original = read(stock_path)
    stock_matches = {arm: {
        name: sum(all(left[key] == right[key] for key in OUTPUT)
                  for left, right in zip(original[name]["rows"], raw[arm][name]["rows"], strict=True))
        for name in workloads} for arm in PROBE_ARMS}

    comparisons, short = {}, {}
    for label, before, after in (("plain_vs_off", "off", "plain"), ("filtered_vs_off", "off", "filtered"),
                                 ("filtered_vs_plain", "plain", "filtered")):
        comparisons[label] = {}
        short[label] = {}
        for order in ("first", "second"):
            off, on = f"{before}-{order}", f"{after}-{order}"
            comparisons[label][order] = compare(raw[off], raw[on], results[off], results[on])
            short[label][order] = short_distribution(raw[off], raw[on])

    filtered = comparisons["filtered_vs_off"]
    guards = {
        "exact_stock_outputs": all(cells == REQUEST_COUNTS for cells in stock_matches.values()),
        "off_no_hits": all(results[f"off-{order}"]["cells"][name]["cached_requests"] == 0
                           for order in ("first", "second") for name in workloads),
        "filtered_short_no_hits": all(results[f"filtered-{order}"]["cells"]["short"]["prompt_tokens_max"] < 128
                                      and results[f"filtered-{order}"]["cells"]["short"]["cached_requests"] == 0
                                      and results[f"plain-{order}"]["cells"]["short"]["cached_requests"] > 0
                                      for order in ("first", "second")),
        "filtered_preserves_warm_reuse": all(
            all(left["cached_tokens"] == right["cached_tokens"] >= 1536
                for left, right in zip(raw[f"plain-{order}"]["rag"]["rows"],
                                        raw[f"filtered-{order}"]["rag"]["rows"], strict=True))
            for order in ("first", "second")),
        "rag_ttft_p95_improved_in_both_orders": all(row["rag"]["change_percent"]["ttft_p95_ms"] < 0
                                                   for row in filtered.values()),
        "decode_tpot_regression_below_1_percent": all(row[name]["change_percent"]["tpot_p50_ms"] < 1
                                                    for row in filtered.values() for name in ("rag", "short")),
        "short_ttft_and_throughput_regression_below_2_percent": all(
            row["short"]["change_percent"]["ttft_p50_ms"] < 2 and row["short"]["change_percent"]["ttft_p95_ms"] < 2
            and row["short"]["change_percent"]["output_tok_s"] > -2 for row in filtered.values()),
    }
    return {"remote_evidence": str(stage), "remote_build": str(build), "image_id": STOCK_IMAGE,
            "stock_workload_sha256": digest(stock_path), "run_order": progression, "runs": results,
            "stock_matches": stock_matches, "comparisons": comparisons, "short_ttft_by_submission": short, "guards": guards,
            "filtered_meets_trial_guards": all(guards.values())}


if __name__ == "__main__":
    print(json.dumps(verify_probe(Path(sys.argv[1])), indent=2, allow_nan=False))
