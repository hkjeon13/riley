"""Match the two original RSS reads to the in-process trim and nearby mappings."""

from argparse import ArgumentParser
from collections import defaultdict
from hashlib import sha256
import gzip
import json
from pathlib import Path
import re

from allocator_probe import read_mappings
from periodic_trim import PERIOD_MS, read_telemetry


def require(condition, message):
    if not condition:
        raise ValueError(message)


def trim_position(start_ns, end_ns, trims):
    preceding = [call for call in trims if call["completed_ns"] <= start_ns]
    following = [call for call in trims if call["started_ns"] >= end_ns]
    overlap = [call["index"] for call in trims
               if call["completed_ns"] > start_ns and call["started_ns"] < end_ns]
    require(preceding and start_ns - preceding[-1]["completed_ns"] < 2 * PERIOD_MS * 1_000_000,
            "no nearby preceding trim on the same clock")
    last = preceding[-1]
    return {"preceding_trim": last["index"], "preceding_trim_returned": last["returned"],
            "milliseconds_since_trim_at_start": (start_ns - last["completed_ns"]) / 1e6,
            "milliseconds_since_trim_at_end": (end_ns - last["completed_ns"]) / 1e6,
            "overlapping_trims": overlap, "following_trim": following[0]["index"] if following else None}


def read_run(out):
    raw = (out / "soak.log").read_text().splitlines()
    clocks = [line.removeprefix("W5-CLK ") for line in raw if line.startswith("W5-CLK ")]
    require(len(clocks) == 1 and re.fullmatch(r"time:\[\d+\]", clocks[0]), "missing clock namespace proof")
    samples = [json.loads(line.removeprefix("W5-RSS-SAMPLE ")) for line in raw if line.startswith("W5-RSS-SAMPLE ")]
    require([row["point"] for row in samples] == ["midpoint", "drain"], "expected two scored RSS reads")
    soak = json.loads((out / "soak.json").read_bytes())
    report = json.loads((out / "report.json").read_bytes())
    require(soak["seconds"] == 1800 and soak["rate"] == 8 and soak["shared_prefixes"] == 8,
            "unexpected diagnostic workload")
    for name, digest in report["artifacts_sha256"].items():
        require(name == Path(name).name and sha256((out / name).read_bytes()).hexdigest() == digest,
                f"artifact checksum mismatch: {name}")
    exact = soak["smaps_rollup_second_half"]
    require(report["mixed_soak"]["rss"] == exact and report["shutdown_exit"] == 0,
            "soak result was not consumed or shutdown retained state")
    telemetry_file = out / "periodic-trim.jsonl"
    telemetry = read_telemetry(telemetry_file, pid=1, period_ms=PERIOD_MS, duration_seconds=1800)
    require(telemetry == report["periodic_trim"], "trim telemetry differs from the GPU checker")
    trims = [json.loads(line) for line in telemetry_file.read_text().splitlines()]
    capture = {}
    coordinates = {}
    for sample, key in zip(samples, ("baseline", "drain"), strict=True):
        point = sample["point"]
        require(sample["container_pid"] == soak["rss_source"]["container_pid"] == 1
                and sample["host_pid"] == soak["rss_source"]["host_pid"]
                and 0 < sample["rollup_start_ns"] <= sample["rollup_end_ns"]
                <= sample["map_start_ns"] <= sample["map_end_ns"],
                f"identity or sampling order differs at {point}")
        require(all(sample[field] == exact[key][field] for field in ("rss_kib", "anonymous_kib")),
                f"observer and original RSS checker disagree at {point}")
        name = sample["maps"]
        require(name == f"smaps-maps-{point}.txt.gz", f"unexpected map archive at {point}")
        content = gzip.decompress((out / name).read_bytes()).decode()
        capture[point] = read_mappings(content.splitlines())
        map_rss = sum(values["Rss"] for _, values in capture[point].values())
        map_anon = sum(values["Anonymous"] for _, values in capture[point].values())
        coordinates[point] = {
            "rollup_read": trim_position(sample["rollup_start_ns"], sample["rollup_end_ns"], trims),
            "map_read": trim_position(sample["map_start_ns"], sample["map_end_ns"], trims),
            "rollup_and_maps_rss_difference_kib": map_rss - sample["rss_kib"],
            "rollup_and_maps_anonymous_difference_kib": map_anon - sample["anonymous_kib"],
        }
    require(samples[0]["map_end_ns"] < samples[1]["rollup_start_ns"], "RSS reads overlap across the soak")

    halfway, drain = (capture[name] for name in ("midpoint", "drain"))
    deltas = defaultdict(lambda: {"retained_delta_kib": 0, "lost_kib": 0, "added_kib": 0})
    for identity, (label, old) in halfway.items():
        if identity in drain:
            deltas[label]["retained_delta_kib"] += drain[identity][1]["Anonymous"] - old["Anonymous"]
        elif old["Anonymous"]:
            deltas[label]["lost_kib"] += old["Anonymous"]
    for identity, (label, new) in drain.items():
        if identity not in halfway and new["Anonymous"]:
            deltas[label]["added_kib"] += new["Anonymous"]
    return {"mode": report["mixed_soak"]["mode"], "verdict": soak["status"], "original_rss": exact,
            "sample_bounds": coordinates, "same_extent_mapping_changes_by_class": dict(deltas),
            "diagnostic_scope": "Map reads follow the scored RSS samples and can observe different pages."}


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    print(json.dumps(read_run(args.path), indent=2, allow_nan=False))
