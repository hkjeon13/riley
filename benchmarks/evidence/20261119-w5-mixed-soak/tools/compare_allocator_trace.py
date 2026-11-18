"""Validate the cache-off allocator trace against the matched repeat and summarize it."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from statistics import median

from compare_cache_off_repeat import json_file, require, run, same_args


RSS = ("rss_kib", "anonymous_kib", "file_kib", "shared_kib")
ARENA = ("arena_bytes", "arena_in_use_bytes", "arena_free_bytes", "malloc_mmap_bytes")


def normalized_client(argv):
    result = same_args(argv, ("--pid", "--out"))
    result[2] = Path(result[2]).name
    result[3] = "loopback"
    return result


def summarize_samples(samples):
    summary = {}
    for field in (*RSS, *ARENA):
        values = [sample[field] for sample in samples]
        summary[field] = {"min": min(values), "median": median(values), "max": max(values)}
    return summary


def check_samples(samples, pid, expected_count):
    require(len(samples) == expected_count and len(samples) >= 120, "allocator trace coverage differs")
    require(all(sample["pid"] == pid and isfinite(sample["unix_seconds"]) for sample in samples),
            "allocator trace contains an unexpected PID or timestamp")
    require(all(0 < b["unix_seconds"] - a["unix_seconds"] <= 45 for a, b in zip(samples, samples[1:])),
            "allocator trace timestamps have a gap or are not ordered")
    require(all(type(sample[key]) is int and sample[key] >= 0 for sample in samples for key in (*RSS, *ARENA)),
            "allocator trace contains an invalid counter")
    require(all(sample["rss_kib"] == sum(sample[key] for key in RSS[1:])
                and sample["arena_bytes"] == sample["arena_in_use_bytes"] + sample["arena_free_bytes"]
                for sample in samples), "allocator or RSS component totals differ")


def compare(reference, traced, expected_runner_hash):
    base, base_soak, base_summary = run(reference / "cache-off", reference)
    current, current_soak, current_summary = run(traced / "cache-off", traced)
    require((traced / "campaign.exit").read_text().strip() == "0", "allocator campaign did not finish")
    require(all(base[key] == current[key] for key in ("gpu_uuid", "holder_pid", "holder_start_ticks", "riley_source")),
            "allocator trace changed device, holder, or source")
    hashes = [{Path(path).name: digest for path, digest in receipt["sha256"].items()}
              for receipt in (base, current)]
    require(hashes[0] | {"run_soak.py": expected_runner_hash} == hashes[1], "unexpected serving or runner inputs")
    require(same_args(base["server_argv"], ("--bind",)) == same_args(current["server_argv"], ("--bind",))
            and normalized_client(base["client_argv"]) == normalized_client(current["client_argv"]),
            "workload command or serving flags changed")
    require(base_soak["counts"] == current_soak["counts"], "measured workload outcomes differ")
    library_hash, filename = (traced / "library.sha256").read_text().split()
    require(filename == "sample_allocator.so" and library_hash == current["allocator_sampler_sha256"],
            "observed sampler differs from the checked library")
    raw = (traced / "cache-off" / "allocator.jsonl").read_bytes()
    require(sha256(raw).hexdigest() == current["allocator_jsonl_sha256"], "allocator trace digest differs")
    samples = [json.loads(line) for line in raw.splitlines()]
    argv = current["client_argv"]
    pid = int(argv[argv.index("--pid") + 1])
    check_samples(samples, pid, current["allocator_samples"])

    # The write follows the final soak snapshot. Keep the comparison windows away from their boundaries.
    finish_ns = int((traced / "soak-output-mtime-ns").read_text())
    estimated_start = finish_ns / 1e9 - current_soak["final"]["at_seconds"]
    windows = {}
    for label, start, end in (("mid", 930, 990), ("late", 1710, 1770)):
        selected = [sample for sample in samples if start <= sample["unix_seconds"] - estimated_start < end]
        require(len(selected) >= 2, f"allocator trace lacks {label} window coverage")
        windows[label] = {"estimated_elapsed_seconds": [start, end], "samples": len(selected),
                          "ranges": summarize_samples(selected)}
    late_latency = {}
    for key in ("ttft_ms_p95", "e2e_ms_p95"):
        b, c = (soak["latency_windows"]["late"][key] for soak in (base_soak, current_soak))
        late_latency[key] = {"repeat": b, "instrumented": c, "change_percent": (c / b - 1) * 100}
    return {"repeat": base_summary, "instrumented": current_summary, "late_p95_ms": late_latency,
            "window_clock": "Estimated from soak output mtime and the final monotonic snapshot",
            "allocator_jsonl_sha256": current["allocator_jsonl_sha256"], "samples": len(samples),
            "windows": windows,
            "late_minus_mid_medians": {key: windows["late"]["ranges"][key]["median"]
                                       - windows["mid"]["ranges"][key]["median"] for key in (*RSS, *ARENA)}}


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("traced", type=Path)
    parser.add_argument("runner_hashes", type=Path)
    args = parser.parse_args()
    runner_hash, runner_path = args.runner_hashes.read_text().split()
    require(runner_path == "tools/run_soak.py", "expected the patched soak runner hash")
    print(json.dumps(compare(args.reference, args.traced, runner_hash), indent=2, allow_nan=False))
