"""Bind address-redacted smaps deltas to the strict RSS gate and the paired soak."""
from argparse import ArgumentParser
import gzip
from hashlib import sha256
import json
from pathlib import Path

from compare_cache_off_repeat import json_file, require, require_matched_soaks, require_restored_reservation, run
from compare_smaps_rollup import check_rollup

RSS_GATE = "rss_growth_second_half_at_most_zero"
SMAPS_FIELDS = {"Rss", "Anonymous", "Private_Dirty", "AnonHugePages"}


def source_digest(path, target):
    digest, name = path.read_text().split()
    require(len(digest) == 64 and name == target, f"unexpected source digest for {target}")
    return digest


def read_bound_trace(trace_dir):
    cache = trace_dir / "cache-off"
    receipt = json_file(cache, "receipt.json")
    archive = cache / "soak.json.gz"
    raw = gzip.decompress(archive.read_bytes()) if archive.exists() else (cache / "soak.json").read_bytes()
    shutdown_raw = (cache / "shutdown-metrics.json").read_bytes()
    require(sha256(raw).hexdigest() == receipt["soak_sha256"]
            and sha256(shutdown_raw).hexdigest() == receipt["shutdown_metrics_sha256"],
            "trace or shutdown bytes differ from receipt")
    soak, shutdown = json.loads(raw), json.loads(shutdown_raw)
    fields = ("active_requests", "waiting_requests", "kv_allocated_blocks")
    require(not any(shutdown[key] for key in fields) and not any(shutdown["allocation"].values()),
            "shutdown left request or CUDA resources")
    require(receipt["mode"] == "cache-off" and receipt["prefix_cache_blocks"] == 0
            and receipt["server_exit"] == 0 and receipt["observed_active"],
            "expected a dedicated cache-off run and clean CUDA shutdown")
    require(soak["seconds"] == receipt["duration_seconds"] == 1800
            and soak["rate"] == 8 and soak["shared_prefixes"] == 8, "unaligned workload profile")

    base = next((row for row in soak["rss_samples"] if row["at_seconds"] >= soak["seconds"] / 2), None)
    require(base is not None and [row for row in soak["rss_samples"] if "smaps_rollup" in row] == [base],
            "the exact sample did not start the second half")
    start, end = base["smaps_rollup"], soak["final"]["smaps_rollup"]
    check_rollup(start)
    check_rollup(end)
    delta = end["rss_kib"] - start["rss_kib"]
    anon_delta = end["anonymous_kib"] - start["anonymous_kib"]
    require(soak["smaps_rollup_second_half"] ==
            {"baseline": start, "drain": end, "rss_growth_kib": delta, "anonymous_growth_kib": anon_delta},
            "strict RSS comparison differs from its measurements")
    require(not soak["problem_count"]
            and all(type(value) is bool for value in soak["checks"].values())
            and {key for key, passed in soak["checks"].items() if not passed} == ({RSS_GATE} if delta > 0 else set())
            and soak["checks"].get(RSS_GATE) is (delta <= 0),
            "a workload check other than the strict RSS gate failed")
    expected_exit = int(delta > 0)
    expected_status = "failed" if expected_exit else "passed"
    require(int((trace_dir / "campaign.exit").read_text()) == receipt["client_exit"] == expected_exit
            and soak["status"] == receipt["status"] == expected_status,
            "campaign and exact RSS gate exits disagree")
    return receipt, soak, delta, anon_delta


def compare(reference, trace, inputs, *, runner_digest_file=None):
    base_receipt, base_soak, _ = run(reference / "cache-off", reference)
    receipt, soak, delta, anon_delta = read_bound_trace(trace)
    require(set(soak["checks"]) == (set(base_soak["checks"]) - {"precise_rss_coverage"}) | {RSS_GATE},
            "workload check vocabulary changed")
    cmd = receipt["client_argv"]
    require(cmd.count("--require-zero-rss-growth") == 1, "strict RSS gate not enabled")
    normalized = dict(receipt, client_argv=[arg for arg in cmd if arg != "--require-zero-rss-growth"])
    hashes = {
        "soak.py": sha256((inputs / "attribute_anon_rss.py").read_bytes()).hexdigest(),
        "run_soak.py": source_digest(runner_digest_file or inputs / "strict-rss-runner.sha256", "tools/run_soak.py"),
    }
    require_matched_soaks(base_receipt, base_soak, normalized, soak, hashes)
    require_restored_reservation(trace)
    holder = json_file(trace, "idle-after.json")
    require(holder["pid"] == receipt["holder_pid"] and holder["start_ticks"] == receipt["holder_start_ticks"]
            and holder["gpu"] == receipt["gpu_uuid"] and holder["physical_index"] == 7
            and holder["phase"] == "idle" and holder["allocation_bytes"] == holder["torch_reserved_bytes"] == 20 * 1024**3
            and not holder["other_processes"], "GPU 7 did not restore its reserved holder")
    require("tools/upstream_soak.py: OK" in (trace / "campaign.log").read_text().splitlines(),
            "the upstream soak source was not verified by the campaign")

    mapping = json_file(trace / "cache-off", "anon-attribution.json")
    start, end = [mapping[f"{point}_by_class_kib"] for point in ("baseline", "drain")]
    require(all(set(entry) == SMAPS_FIELDS and all(type(v) is int and v >= 0 for v in entry.values())
                for group in (start, end) for entry in group.values()), "malformed mapping totals")
    expected = {label: {key: end.get(label, {}).get(key, 0) - start.get(label, {}).get(key, 0)
                        for key in SMAPS_FIELDS} for label in start.keys() | end.keys()}
    require(mapping["delta_by_class_kib"] == expected, "mapping deltas differ from raw totals")
    return {
        "strict_soak_status": soak["status"],
        "rollup_rss_delta_kib": delta,
        "rollup_anonymous_delta_kib": anon_delta,
        "mapping_rss_delta_kib": sum(row["Rss"] for row in expected.values()),
        "mapping_anonymous_delta_kib": sum(row["Anonymous"] for row in expected.values()),
        "delta_by_class_kib": mapping["delta_by_class_kib"],
        "largest_retained_anonymous_changes": mapping["largest_retained_anonymous_changes"],
        "mapping_read_ms": mapping["read_ms"],
        "late_p95_change_percent_from_reference": {
            key: (soak["latency_windows"]["late"][key] / base_soak["latency_windows"]["late"][key] - 1) * 100
            for key in ("ttft_ms_p95", "e2e_ms_p95")
        },
    }


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("trace", type=Path)
    parser.add_argument("inputs", type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.reference, args.trace, args.inputs), indent=2, allow_nan=False))
