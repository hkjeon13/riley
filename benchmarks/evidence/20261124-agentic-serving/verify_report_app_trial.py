"""Verify the single-GPU report app crossover from the preserved run directory."""

import argparse
import itertools
import json
from pathlib import Path

from benchmarks.live.agent_workflow import percentile
from benchmarks.live.report_app_trial import WAVES, summarize


def read(path):
    return json.loads(path.read_text())


def process_ids(path):
    return {
        int(line.split(",", 1)[0])
        for line in path.read_text().splitlines()
        if "," in line
    }


def run_details(root, name, holder_pid):
    directory = root / name
    run = read(directory / "workflows.json")
    assert run["warm"]["ok"], name
    for wave, (size, base) in zip(run["waves"], WAVES, strict=True):
        assert [row["case"] for row in wave] == list(range(base, base + size)), name
        assert [row["reader"] for row in wave] == [
            index % 4 == 0 for index in range(size)
        ], name
        assert [row["scheduled_ns"] - wave[0]["scheduled_ns"] for row in wave] == [
            30_000_000 * index for index in range(size)
        ], name
    jobs = [row for wave in run["waves"] for row in wave]
    for row in jobs:
        events = row["events_ns"]
        start = row["scheduled_ns"]
        assert start <= row["sent_ns"] <= events[0] <= events[-1] <= row["done_ns"], (
            name
        )
        assert row["first_ms"] == (events[0] - start) / 1e6, name
        assert row["last_ms"] == (events[-1] - start) / 1e6, name
        assert row["gap_ms"] == percentile(
            [(b - a) / 1e6 for a, b in itertools.pairwise(events)], 0.95
        ), name
        assert row["slo"] == (
            row["ok"]
            and row["first_ms"] <= 5000
            and row["last_ms"] <= 8000
            and row["gap_ms"] <= 80
        ), name
    summary = summarize([row for wave in run["waves"][-2:] for row in wave])
    assert summary == run["c16"], name
    assert all(row["receipt_ok"] for row in jobs), name
    assert all(row["http_status"] == 200 and "error" not in row for row in jobs), name
    assert (directory / "server-exit.txt").read_text().strip() == "0", name
    shutdown = read(directory / "shutdown.json")
    reclaimed = {
        key: shutdown[key]
        for key in (
            "active_requests",
            "waiting_requests",
            "kv_allocated_blocks",
            "allocation",
            "counters",
        )
    }
    assert not any(
        reclaimed[key]
        for key in ("active_requests", "waiting_requests", "kv_allocated_blocks")
    ), name
    assert not any(reclaimed["allocation"].values()), name
    assert not any(reclaimed["counters"].values()), name
    server_pid = int((directory / "server-pid.txt").read_text())
    sampled = process_ids(directory / "compute-during.txt")
    assert sampled == {server_pid, holder_pid}, (name, sampled)
    assert process_ids(directory / "compute-after.csv") == {holder_pid}, name
    return jobs, {
        "c16": summary,
        "all_waves_valid": sum(row["ok"] for row in jobs),
        "all_waves_attempted": len(jobs),
        "gpu_sample_count": sum(
            line.isdigit()
            for line in (directory / "compute-during.txt").read_text().splitlines()
        ),
        "foreign_gpu_processes": sorted(sampled - {server_pid, holder_pid}),
        "shutdown": reclaimed,
    }


def verify_pair(baseline, improved, a, b):
    for left, right in zip(baseline, improved, strict=True):
        assert (
            left["case"] == right["case"]
            and left["request_sha256"] == right["request_sha256"]
        )
        control, candidate = left["receipt"], right["receipt"]
        assert control["source_count"] == candidate["source_count"]
        assert control["source_sha256"] == candidate["source_sha256"]
        if left["reader"]:
            assert control["dispatch"] == candidate["dispatch"] == ["model"] * 2
            assert control["forwarded_sha256"] == candidate["forwarded_sha256"]
        else:
            assert control["source_count"] == control["forwarded_count"] == 64
            assert control["source_sha256"] == control["forwarded_sha256"]
            assert candidate["forwarded_count"] == 2
            assert control["dispatch"] == ["model"] * 2
            assert candidate["dispatch"] == ["application"] * 2
    return {
        "matched_request_and_catalog_pairs": len(baseline),
        "new_invalid_cases": [
            a["case"]
            for a, b in zip(baseline, improved, strict=True)
            if a["ok"] and not b["ok"]
        ],
        "lower_complete_p95": b["complete_p95_ms"] < a["complete_p95_ms"],
        "more_slo": b["slo"] > a["slo"],
        "reader_first_p95_ratio": b["reader_first_p95_ms"] / a["reader_first_p95_ms"],
        "reader_gap_p95_ratio": b["reader_gap_p95_ms"] / a["reader_gap_p95_ms"],
    }


def verify(root):
    before, after = (read(root / f"lease-{when}.json") for when in ("before", "after"))
    assert before["pid"] == after["pid"] and before["gpu"] == after["gpu"]
    assert before["physical_index"] == after["physical_index"] == 6
    assert all(
        x["status"] == "running" and x["phase"] == "idle" and not x["other_processes"]
        for x in (before, after)
    )
    order = ("a1", "b1", "b2", "a2")
    details = {name: run_details(root, name, before["pid"]) for name in order}
    runs = {name: details[name][1] for name in order}
    pairs = {
        f"{a}/{b}": verify_pair(
            details[a][0], details[b][0], runs[a]["c16"], runs[b]["c16"]
        )
        for a, b in (("a1", "b1"), ("a2", "b2"))
    }
    return {
        "device": (root / "device.csv").read_text().splitlines()[1],
        "binary_sha256": (root / "identity.sha256")
        .read_text()
        .splitlines()[0]
        .split()[0],
        "checkpoint_manifest_sha256": (root / "identity.sha256")
        .read_text()
        .splitlines()[1]
        .split()[0],
        "catalog_sha256": (root / "catalog.sha256").read_text().split()[0],
        "order": order,
        "runs": runs,
        "pairs": pairs,
        "lease_restored": True,
        "qualified": all(
            pair["lower_complete_p95"]
            and pair["more_slo"]
            and not pair["new_invalid_cases"]
            and pair["reader_first_p95_ratio"] <= 1.05
            and pair["reader_gap_p95_ratio"] <= 1.05
            for pair in pairs.values()
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(verify(args.directory), indent=2) + "\n")
