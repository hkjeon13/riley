"""Verify allocator probe provenance and reconstruct its mapping-level diagnostic."""

from argparse import ArgumentParser
import gzip
from hashlib import sha256
import json
from pathlib import Path

from allocator_probe import read_mappings, summarize


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def require(condition, problem):
    if not condition:
        raise ValueError(problem)


def follow_growing_regions(maps):
    def by_start(regions):
        indexed = {(meta[0], *meta[2:]): (meta, fields["Anonymous"], label)
                   for meta, (label, fields) in regions.items()}
        require(len(indexed) == len(regions), "ambiguous mapping start and metadata")
        return indexed

    mid, drain, pretrim, posttrim = (by_start(maps[point]) for point in ("midpoint", "drain", "pretrim", "posttrim"))
    matched_starts = mid.keys() & drain.keys()
    growth_groups = {kind: {"maps_growing": 0, "growth_kib": 0, "maps_with_lower_anonymous_rss_after_trim": 0,
                            "reclaimed_kib": 0, "maps_still_above_midpoint": 0, "still_above_midpoint_kib": 0}
                     for kind in ("same_extent_during_soak", "changed_extent_during_soak")}
    for key in matched_starts:
        old_meta, old_anon, label = mid[key]
        drain_meta, drain_anon, _ = drain[key]
        growth = drain_anon - old_anon
        if growth <= 0 or label != "anonymous-private":
            continue
        group = growth_groups["same_extent_during_soak" if drain_meta == old_meta else "changed_extent_during_soak"]
        group["maps_growing"] += 1
        group["growth_kib"] += growth
        if key in pretrim and key in posttrim and pretrim[key][0] == posttrim[key][0]:
            decrease = pretrim[key][1] - posttrim[key][1]
            group["maps_with_lower_anonymous_rss_after_trim"] += int(decrease > 0)
            group["reclaimed_kib"] += decrease
            residual = posttrim[key][1] - old_anon
            group["maps_still_above_midpoint"] += int(residual > 0)
            group["still_above_midpoint_kib"] += max(0, residual)
    return {
        "matched_starts_and_metadata": len(matched_starts),
        "anonymous_net_growth_of_matched_kib": sum(drain[key][1] - mid[key][1] for key in matched_starts),
        "anonymous_at_unmatched_midpoint_kib": sum(mid[key][1] for key in mid.keys() - drain.keys()),
        "anonymous_at_unmatched_drain_kib": sum(drain[key][1] for key in drain.keys() - mid.keys()),
        "growing_anonymous_private_regions": growth_groups,
    }


def verify(stage, baseline):
    require((stage / "campaign.exit").read_text().strip() == "0", "allocator campaign did not complete")
    original = read(baseline)
    original_soak_path = baseline.with_name("soak.json")
    require(digest(original_soak_path) == original["artifacts_sha256"]["soak.json"]
            and original["mixed_soak"]["mode"] == "cache-off", "untrimmed reference evidence changed")
    original_soak = read(original_soak_path)
    evidence = stage / "gpu-evidence"
    report = read(evidence / "report.json")
    provenance = ("image_id", "source_revision", "source_archive_sha256", "riley_binary_sha256",
                  "cpu_report_sha256", "checkpoint_sha256", "gpu_uuid", "soak_server_arguments")
    require(all(report[key] == original[key] for key in provenance), "packaged cache-off inputs changed")
    require(report["api_checks"] == original["api_checks"] == 40 and report["tool_checks_enabled"],
            "packaged API coverage changed")
    for line in (stage / "result.sha256").read_text().splitlines():
        expected, name = line.split()
        require(digest(stage / name) == expected, f"campaign evidence changed: {name}")
    for file, expected in report["artifacts_sha256"].items():
        require(file == Path(file).name and digest(evidence / file) == expected, f"artifact changed: {file}")
    soak = read(evidence / "soak.json")
    probe = read(evidence / "allocator-trim.json")
    marker = soak["smaps_rollup_second_half"]
    gate = soak["checks"]["rss_growth_second_half_at_most_zero"]
    require(soak["seconds"] == 1800 and soak["rate"] == 8 and soak["shared_prefixes"] == 8
            and soak["checks"]["idle_after_drain"] and soak["problem_count"] == 0
            and marker["rss_growth_kib"] == probe["original_rss_growth_kib"]
            and type(marker["rss_growth_kib"]) is int
            and gate == probe["original_rss_gate"] == (marker["rss_growth_kib"] <= 0),
            "the probe diverges from the pre-trim soak")
    require(soak["rss_source"] == {"container_id": probe["container_id"], "host_pid": probe["riley_host_pid"],
                                  "container_pid": 1}
            and probe["in_process_trim"]["pid"] == 1 and probe["in_process_trim"]["malloc_trim_return"] in (0, 1),
            "the allocator probe was not attached to the measured server")

    points = ("midpoint", "drain", "pretrim", "posttrim")
    maps = {point: read_mappings(gzip.decompress((evidence / f"smaps-{point}.txt.gz").read_bytes()).decode().splitlines())
            for point in points}
    reconstructed = summarize(maps)
    require(all(probe[key] == value for key, value in reconstructed.items()), "mapping aggregate cannot be reproduced")
    before, after = (read(stage / f"idle-{point}.json") for point in ("before", "after"))
    require(all(before[key] == after[key] for key in before if key != "updated_ns")
            and after["phase"] == "idle" and after["allocation_bytes"] == after["torch_reserved_bytes"] == 20 * 1024**3
            and after["physical_index"] == 6 and not after["other_processes"],
            "A100 6 did not restore its 20 GiB reservation")
    shutdown = read(evidence / "shutdown.json")
    require(report["shutdown_exit"] == 0 and shutdown["allocation"]
            and all(shutdown[key] == 0 for key in ("active_requests", "waiting_requests", "kv_allocated_blocks"))
            and all(value == 0 for value in shutdown["allocation"].values()), "shutdown did not release resources")
    totals = reconstructed["totals_kib"]
    late_p95 = {}
    for metric in ("ttft_ms_p95", "e2e_ms_p95"):
        reference = original_soak["latency_windows"]["late"][metric]
        diagnostic = soak["latency_windows"]["late"][metric]
        require(reference > 0 and diagnostic > 0, "missing late latency samples")
        late_p95[metric] = {"untrimmed": reference, "diagnostic": diagnostic,
                            "difference_percent": (diagnostic / reference - 1) * 100}
    return {
        "provenance": {key: report[key] for key in provenance},
        "gpu_report_sha256": digest(evidence / "report.json"),
        "original_gpu_report_sha256": digest(baseline),
        "diagnostic_report_sha256": digest(evidence / "allocator-trim.json"),
        "original_uninstrumented_cache_off_rss_growth_kib": original["mixed_soak"]["rss"]["rss_growth_kib"],
        "pre_trim_strict_rss_gate_passed": gate,
        "diagnostic_pre_trim_rss_growth_kib": marker["rss_growth_kib"],
        "same_outcome_counts": soak["counts"] == original["mixed_soak"]["counts"],
        "diagnostic_soak_failed_checks": report["mixed_soak"]["failed_checks"],
        "diagnostic_soak_p95_change_percent": soak["p95_change_percent"],
        "late_p95_ms": late_p95,
        "anonymous_drain_to_pretrim_kib": totals["pretrim"]["Anonymous"] - totals["drain"]["Anonymous"],
        "anonymous_reclaimed_during_probe_kib": totals["pretrim"]["Anonymous"] - totals["posttrim"]["Anonymous"],
        "reservation_restored": True,
        "glibc": probe["in_process_trim"],
        "mappings": reconstructed,
        "region_analysis": follow_growing_regions(maps),
        "raw_capture_sha256": {point: digest(evidence / f"smaps-{point}.txt.gz") for point in points},
    }


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("stage", type=Path)
    parser.add_argument("baseline", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.stage, args.baseline), indent=2, allow_nan=False))
