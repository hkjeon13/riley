"""Replay a completed series at the evidence host and emit the timing summary."""

from argparse import ArgumentParser
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys


def replay(stage):
    saved = (stage / "verified.json").read_bytes()
    environment = {**os.environ, "PYTHONPATH": str(stage / "harness/benchmarks/live")}
    reconstructed = subprocess.check_output(
        [sys.executable, "-B", str(stage / "verify_rss_series.py"), str(stage)],
        env=environment,
    )
    if saved != reconstructed or (stage / "campaign.exit").read_text().strip() != "0":
        raise ValueError("the completed RSS series receipt did not reproduce")
    record = json.loads(reconstructed)
    raw = json.loads((stage / "gpu-evidence/soak.json").read_bytes())
    baseline = next(row for row in raw["rss_samples"] if row["at_seconds"] >= raw["seconds"] / 2)
    final = raw["final"]
    record["original_sample_seconds"] = {"baseline": baseline["at_seconds"], "drain": final["at_seconds"]}
    record["original_sample_gauges"] = {
        label: {name: sample["metrics"][name] for name in ("active_requests", "waiting_requests", "kv_allocated_blocks")}
        for label, sample in (("baseline", baseline), ("drain", final))
    }
    record["request_totals"] = {
        outcome: sum(counts.get(outcome, 0) for counts in raw["counts"].values())
        for outcome in ("completed", "disconnected")
    }
    record["original_p95_change_percent"] = raw["p95_change_percent"]
    record["original_latency_ms"] = {
        metric: raw["latency_windows"]["late"][metric] for metric in ("ttft_ms_p95", "e2e_ms_p95")
    }
    record["receipt_replay_matches"] = True
    record["remote_verified_receipt_sha256"] = sha256(saved).hexdigest()
    return record


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("stage", type=Path)
    args = parser.parse_args()
    print(json.dumps(replay(args.stage), indent=2, allow_nan=False))
