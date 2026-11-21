"""Bind cache counter evidence to the exact binary, checkpoint and GPU."""

import json
from pathlib import Path
import sys

from current_binding import check_reservations
from run_probe import GPU, IMAGE, MODEL_SHA, MODES, RILEY, SERVER
from soak import idle
from verify_a100_candidate import digest, read, require


def verify(stage):
    build = stage.parent / "w5-prefix-metrics-build-20261121.5MLLtayu"
    prior = stage.parent / "w5-prefix-metrics-gpu-20261121.8DIRyrNl"
    cpu = stage.parent / "w5-current-package-20261121.f6SEbjvA/cpu-evidence/report.json"
    base = read(cpu)
    expected_binary = digest(build / "target/release/riley")
    timeline = [line.split() for line in (prior / "campaign.log").read_text().splitlines()
                if line.startswith(("START stock ", "END stock "))]
    timeline.extend(line.split() for line in (stage / "campaign.log").read_text().splitlines()
                    if line.startswith(("START ", "END ")))
    require([item[:2] for item in timeline] == [[phase, mode] for mode in MODES for phase in ("START", "END")]
            and all(item[-1] == "exit=0" for item in timeline if item[0] == "END"), "campaign incomplete")
    data, summaries = {}, {}
    for mode, capacity in MODES.items():
        path = (prior if mode == "stock" else stage) / mode
        receipt = read(path / "receipt.json")
        binary = base["binaries"][RILEY] if mode == "stock" else expected_binary
        argv = [*SERVER, *(["--prefix-cache-blocks", str(capacity)] if capacity else [])]
        require(receipt["mode"] == mode and receipt["image_id"] == IMAGE and receipt["gpu_uuid"] == GPU
                and receipt["checkpoint_sha256"] == MODEL_SHA and receipt["binary_sha256"] == binary
                and receipt["source_archive_sha256"] == base["source_archive_sha256"]
                and receipt["cpu_report_sha256"] == digest(cpu) and receipt["source_revision"] == base["source_revision"]
                and receipt["server_arguments"] == argv and receipt["client_exit"] == receipt["shutdown_exit"] == 0
                and receipt["ready"]["ready"] and receipt["ready"]["accepting"], "live receipt differs")
        hashes = receipt["artifacts_sha256"]
        require(set(hashes) == {"client.log", "operational.json", "shutdown.json", "server.log"}
                and all(digest(path / name) == expected for name, expected in hashes.items()), "raw artifacts changed")
        reservation = check_reservations(path, "after")
        require(reservation == check_reservations(path, "before"), "GPU reservation owner changed")
        result = read(path / "operational.json")
        require(result["status"] == "passed" and result["prefix_cache_blocks"] == capacity
                and all(value is True for value in result["checks"].values()), "cache probe did not pass")
        closed = read(path / "shutdown.json")
        require(idle(closed) and closed["allocation"] and all(value == 0 for value in closed["allocation"].values()),
                "shutdown did not release requests or allocations")
        live = result["metrics"]["final"]
        for snapshot in (*result["metrics"].values(), closed):
            if mode == "stock":
                require("prefix_cache" not in snapshot, "stock capture unexpectedly contains the new metric")
        cache_evidence = None
        if mode != "stock":
            cache_evidence = result["prefix_cache_metrics"]
            cache = cache_evidence["samples"]
            start, end = live["prefix_cache"], closed["prefix_cache"]
            require(cache_evidence["status"] == "passed" and cache["final"] == start
                    and end == {**start, "retained_blocks": 0}, "shutdown did not preserve lifetime cache counters")
            labels = ["before", *result["prefix_requests"], "drained", "final"]
            require(list(cache) == labels, "cache samples are missing or unordered")
            actual = {name: cache[name]["hit_tokens"] - cache[prev]["hit_tokens"]
                      for prev, name in zip(labels, labels[1:-2])}
            actual["recovery"] = cache["final"]["hit_tokens"] - cache["drained"]["hit_tokens"]
            expected = {name: row["cached_tokens"] for name, row in result["prefix_requests"].items()}
            expected["recovery"] = result["overload"]["recovery"]["cached_tokens"]
            require(actual == expected == cache_evidence["hit_token_deltas"]
                    and all(item["retained_blocks"] <= capacity and item["metrics_degraded"] is False
                            for item in cache.values()), "cache counter delta, capacity or health differs")
            if capacity:
                require(cache["pressure"]["evicted_blocks"] > cache["warm"]["evicted_blocks"]
                        and start["retained_blocks"] > 0, "missing eviction or retained cache evidence")
            else:
                require(all(not any(s[key] for key in ("retained_blocks", "hit_tokens", "evicted_blocks"))
                            for s in cache.values()), "disabled cache changed one of its metrics")
        data[mode] = result
        summaries[mode] = {"operational_sha256": hashes["operational.json"], "receipt_sha256": digest(path / "receipt.json"),
                           "binary_sha256": binary, "reservation": reservation,
                           "cache": cache_evidence, "shutdown_prefix_cache": closed.get("prefix_cache"),
                           "overload": {key: result["overload"][key] for key in ("completed", "rejected", "counter_delta")}}

    old_metrics = set(data["stock"]["metrics"]["final"])
    for mode in ("cache-off", "cache-on"):
        require(set(data[mode]["metrics"]["final"]) == old_metrics | {"prefix_cache"},
                "one of the previous metrics fields changed")
        require(summaries[mode]["reservation"] == summaries["stock"]["reservation"], "GPU holders changed across runs")
    require(data["stock"]["prefix_requests"] == data["cache-on"]["prefix_requests"]
            and data["stock"]["overload"]["recovery"] == data["cache-on"]["overload"]["recovery"],
            "cache-on requests differ from the previous image")
    stable_fields = ("prompt_sha256", "visible_output_sha256", "finish", "prompt_tokens")
    off, stock = (data[mode] for mode in ("cache-off", "stock"))
    old, new = ([*result["prefix_requests"].values(), result["overload"]["recovery"]] for result in (stock, off))
    require(all(all(left[key] == right[key] for key in stable_fields)
                for left, right in zip(old, new, strict=True)), "cache-off requests did not reproduce the stock output")
    initial = prior / "cache-on"
    initial_artifacts = read(initial / "receipt.json")["artifacts_sha256"]
    require(all(digest(initial / name) == initial_artifacts[name] for name in ("operational.json", "shutdown.json")),
            "initial failed attempt changed")
    initial_live = read(initial / "operational.json")["metrics"]["final"]["prefix_cache"]
    initial_closed = read(initial / "shutdown.json")["prefix_cache"]
    require(initial_live["hit_tokens"] > initial_closed["hit_tokens"] == 0
            and initial_live["evicted_blocks"] > initial_closed["evicted_blocks"] == 0,
            "initial report does not show the shutdown counter loss")
    runtime_sources = {line.split()[1]: line.split()[0]
                       for line in (build / "runtime-source.sha256").read_text().splitlines()}
    return {"status": "passed", "remote_evidence": str(stage), "remote_build": str(build),
            "initial_attempt": {"remote_evidence": str(prior), "live": initial_live, "shutdown": initial_closed},
            "source_archive_sha256": base["source_archive_sha256"], "image_id": IMAGE, "model_sha256": MODEL_SHA,
            "runtime_sources": runtime_sources,
            "build_result_sha256": digest(build / "build-result.sha256"), "patched_binary_sha256": expected_binary,
            "unchanged_metrics_fields": sorted(old_metrics), "match_to_stock_per_candidate": len(old), "runs": summaries}


if __name__ == "__main__":
    print(json.dumps(verify(Path(sys.argv[1])), indent=2, allow_nan=False))
