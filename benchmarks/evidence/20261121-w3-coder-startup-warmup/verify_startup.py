"""Validate actual startup shape coverage and compare the first short C32 on each device."""

import json
from pathlib import Path
import re
import sys
from unittest.mock import patch

import comparison_base
from run_startup import GPUS, POLICIES
from verify_a100_candidate import digest, read, require
import verify_probe
from warm_reference import checked_reservation, same_outputs
from workload import REQUEST_COUNTS, make_workloads

ORDER = ("off-cold", "filtered-cold", "filtered-warm", "off-warm")
ORDERS = {"7": ORDER, "6": ORDER[::-1]}
CONTRASTS = {
    "filtered_cold_vs_off_cold": ("off-cold", "filtered-cold"),
    "filtered_warm_vs_off_warm": ("off-warm", "filtered-warm"),
    "filtered_warm_vs_off_cold": ("off-cold", "filtered-warm"),
    "filtered_warm_vs_filtered_cold": ("filtered-cold", "filtered-warm"),
    "off_warm_vs_off_cold": ("off-cold", "off-warm"),
}
STARTUP_LOG = re.compile(r"riley startup warm-up: ([0-9]+) internal requests in ([0-9]+\.[0-9]+) s")


def checked_startup(path, raw, condition):
    require(raw.keys() == {"startup", *REQUEST_COUNTS}, "unexpected startup workload phases")
    sample = raw["startup"]
    metrics = sample["metrics"]
    require(all(metrics[key] == 0 for key in ("active_requests", "waiting_requests", "kv_allocated_blocks")),
            "startup requests did not drain before serving")
    shapes = verify_probe.current_batch_shapes(sample)
    hits = {str(rows): counters[0] for rows, counters in shapes.items() if counters[0]}
    startup_lines = STARTUP_LOG.findall((path / "server.log").read_text())
    require(len(startup_lines) == int(condition == "warm"), "wrong startup mode in the actual server log")
    ready_ms = read(path / "readiness.json")["container_start_to_ready_ms"]
    require(0 < ready_ms < 700_000, "warmup ready time is outside the startup timeout")
    warm_seconds = None
    requests = 0
    if startup_lines:
        count, duration = startup_lines[0]
        requests, warm_seconds = int(count), float(duration)
        require(requests > 0 and 0 < warm_seconds < ready_ms / 1000 and hits,
                "startup warmup produced no observable work")
    return {"native_requests_in_log": requests, "warmup_seconds_in_log": warm_seconds,
            "container_start_to_ready_ms": ready_ms, "ready_shape_steps": hits}


def verify(stage):
    evidence = stage.parent
    baseline = evidence / "w3-coder-rag-cache-20261121.kXvzFrJJ/off-first"
    build = evidence / "w3-prefix-min-build-20261121.vdX2vOQA"
    parent = evidence / "w5-current-package-20261121.f6SEbjvA/cpu-evidence/report.json"
    original = read(baseline / "workload.json")
    reference_hash = digest(baseline / "workload.json")
    require(reference_hash == "a77aa2092bd229a40038622d1a2de6768d2a8a6fdca159fbff42c20d9d6cfb7b"
            == read(baseline / "receipt.json")["artifacts_sha256"]["workload.json"], "stock outputs changed")
    timeline = [row.split() for row in (stage / "campaign.log").read_text().splitlines()
                if row.startswith(("START ", "END "))]
    require(len(timeline) == 16 and all(row[-1] == "exit=0" for row in timeline if row[0] == "END"),
            "startup campaign is incomplete")
    workloads = make_workloads()
    arms, comparisons, cohorts, reservations = {}, {}, {}, {}
    for index, progression in ORDERS.items():
        names = [f"{condition}-gpu{index}" for condition in progression]
        events = [event for event in timeline if event[1].endswith(f"-gpu{index}")]
        require([row[:2] for row in events] == [[phase, name] for name in names for phase in ("START", "END")]
                and [row[2] for row in events] == sorted(row[2] for row in events), "startup comparison sequence changed")
        owner = checked_reservation(stage / f"gpu{index}-before.json", index)
        require(owner == checked_reservation(stage / f"gpu{index}-after.json", index), "GPU holder changed")
        reservations.update(owner)
        data = {}
        for name, case in zip(names, progression, strict=True):
            mode, condition = case.split("-")

            def reservation(path, when):
                return checked_reservation(path / f"reservation-{when}.json", index)

            arguments = [*verify_probe.SERVE, "--startup-warmup", POLICIES[condition]]
            with (patch.object(verify_probe, "GPU", GPUS[index]), patch.object(verify_probe, "SERVE", arguments),
                  patch.object(verify_probe, "check_reservations", reservation)):
                dataset, arm = verify_probe.probe_arm(stage, name, build, workloads, parent)
            require(arm["reservation"] == owner, "arm changed the GPU holder")
            arm["startup"] = checked_startup(stage / name, dataset, condition)
            arm["short_shapes_missing_at_ready"] = sorted(
                set(arm["short_gpu_shapes"]["by_dense_rows"]) - set(arm["startup"]["ready_shape_steps"]), key=int)
            arm["stock_matches"] = {key: same_outputs(original[key], dataset[key]) for key in REQUEST_COUNTS}
            require(arm["stock_matches"] == REQUEST_COUNTS and arm["cells"]["short"]["cached_requests"] == 0,
                    "changed stock output or unexpected short cache use")
            rag = arm["cells"]["rag"]
            require((rag["cached_requests"] == 64 and rag["cached_min"] >= 1536) if mode == "filtered"
                    else rag["cached_requests"] == 0, "RAG cache reuse differs from the trial")
            arms[name], data[case] = arm, dataset
        comparisons[index] = {label: comparison_base.compare(data[base], data[test], arms[f"{base}-gpu{index}"],
                                                                arms[f"{test}-gpu{index}"])
                              for label, (base, test) in CONTRASTS.items()}
        cohorts[index] = {label: comparison_base.short_distribution(data[base], data[test])
                          for label, (base, test) in CONTRASTS.items()}

    def passes(bundle):
        cold = bundle["short"]["change_percent"]
        rag = bundle["rag"]["change_percent"]
        return (cold["ttft_p50_ms"] < 2 and cold["ttft_p95_ms"] < 2 and cold["tpot_p50_ms"] < 1
                and cold["output_tok_s"] > -2 and rag["ttft_p95_ms"] < 0 and rag["tpot_p50_ms"] < 1)

    named = ("filtered_cold_vs_off_cold", "filtered_warm_vs_off_warm", "filtered_warm_vs_off_cold")
    guards = {label: all(passes(row[label]) for row in comparisons.values()) for label in named}
    return {"status": "passed", "source_revision": read(parent)["source_revision"], "image_id": verify_probe.STOCK_IMAGE,
            "diagnostic_binary_sha256": digest(build / "target/release/riley"), "model_sha256": verify_probe.MODEL_SHA,
            "stock_workload_sha256": reference_hash, "order": timeline, "reservations": reservations, "runs": arms,
            "comparisons": comparisons, "short_cohorts": cohorts, "both_devices_pass": guards, "remote_evidence": str(stage)}


if __name__ == "__main__":
    print(json.dumps(verify(Path(sys.argv[1])), indent=2, allow_nan=False))
