"""Cross-check committed work exported by /metrics with per-iteration trace data."""

import argparse
import hashlib
import json
from pathlib import Path

from verify_model_decode import check_frozen_requests, natural, read


def verify(root):
    for entry in (root / "inputs.sha256").read_text().splitlines():
        digest, path = entry.split(maxsplit=1)
        assert hashlib.sha256((root / path).read_bytes()).hexdigest() == digest, path
    folder = root / "measured"
    assert (folder / "binary.sha256").read_text().split()[0] == (
        root / "trace-server.sha256"
    ).read_text().split()[0]
    before, after = [read(folder / f"{name}.json") for name in ("before", "after")]
    phases = ("prefill_only", "decode_only", "mixed")
    fields = ("iterations", "prefill_tokens", "decode_tokens", "gpu_execution_ns")
    counts = {phase: dict.fromkeys(fields, 0) for phase in phases}
    raw = (folder / "server.log").read_bytes()
    start, stop = before["log_offset"], after["log_offset"]
    assert 0 <= start < stop <= len(raw)
    for line in raw[start:stop].decode().splitlines():
        if line.startswith("AGENT_STEP "):
            _, _, _, gpu_ns, members = line.split(maxsplit=4)
            tokens = {"p": 0, "d": 0}
            for _, kind, rows, *_ in json.loads(members):
                assert rows > 0
                tokens[kind] += rows
            assert sum(tokens.values()) > 0
            phase = (
                "mixed"
                if all(tokens.values())
                else "prefill_only" if tokens["p"] else "decode_only"
            )
            for key, value in zip(
                fields, (1, tokens["p"], tokens["d"], int(gpu_ns)), strict=True
            ):
                counts[phase][key] += value

    live = read(folder / "metrics.json")
    shutdown = read(folder / "shutdown.json")
    snapshots = [before["metrics"], after["metrics"], live, shutdown]
    for snapshot in snapshots:
        assert snapshot["active_requests"] == snapshot["waiting_requests"] == 0
        assert not snapshot["batch_shapes"]["metrics_degraded"]
    first, last = [s["batch_shapes"] for s in snapshots[:2]]
    for phase in phases:
        assert counts[phase]["iterations"] > 0
        for key, expected in counts[phase].items():
            assert (
                last["by_phase"][phase][key] - first["by_phase"][phase][key] == expected
            ), (phase, key)
    for field, bucket_key in (
        ("iterations", "hit_count"),
        ("iterations", "latency_sample_count"),
        ("gpu_execution_ns", "gpu_execution_ns_total"),
    ):
        totals = [sum(b[bucket_key] for b in s["buckets"]) for s in (first, last)]
        assert totals[1] - totals[0] == sum(c[field] for c in counts.values())
    assert (
        last["by_phase"]
        == live["batch_shapes"]["by_phase"]
        == shutdown["batch_shapes"]["by_phase"]
    )
    assert shutdown["kv_allocated_blocks"] == 0
    assert read(folder / "lease-after.json")["phase"] == "idle"
    check_frozen_requests(root, folder)
    workloads = read(folder / "natural.json")
    seeds = (761, 762, 833, 861, 902, 931)
    checks = [
        natural(work, seed, 2 if seed == 761 else 96, 0 if seed == 761 else 100)
        for seed, work in zip(seeds, workloads, strict=True)
    ]
    assert all(c["verified_tool_calls"] == 2 for c in checks)
    return {
        "phase_deltas": counts,
        "committed_iterations": sum(c["iterations"] for c in counts.values()),
        "exact_phase_counter_matches": len(phases) * len(fields),
        "existing_bucket_totals_match": True,
        "shutdown_phase_counters_match": True,
        "natural_workflows": len(checks),
        "basic_valid_answers": sum(c["valid"] for c in checks),
        "verified_tool_calls": sum(c["verified_tool_calls"] for c in checks),
        "server_binary_sha256": (folder / "binary.sha256").read_text().split()[0],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    print(json.dumps(verify(parser.parse_args().directory), indent=2))
