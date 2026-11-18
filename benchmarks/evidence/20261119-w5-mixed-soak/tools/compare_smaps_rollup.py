"""Validate paired Linux status and smaps_rollup readings from the cache-off soak."""

from argparse import ArgumentParser
import json
from math import isfinite
from pathlib import Path

from compare_cache_off_repeat import require, require_matched_soaks, require_restored_reservation, run


def check_rollup(sample):
    require(isinstance(sample, dict) and sample.keys() == {"rss_kib", "anonymous_kib", "read_ms"},
            "missing or malformed smaps rollup")
    require(type(sample["rss_kib"]) is int and type(sample["anonymous_kib"]) is int
            and 0 <= sample["anonymous_kib"] <= sample["rss_kib"] and sample["rss_kib"] > 0,
            "invalid smaps rollup RSS")
    require(type(sample["read_ms"]) in (int, float) and isfinite(sample["read_ms"]) and sample["read_ms"] >= 0,
            "invalid smaps rollup timing")


def compare(reference, traced, expected_soak_hash):
    require((traced / "campaign.exit").read_text().strip() == "0", "smaps rollup campaign did not finish")
    base, base_soak, base_summary = run(reference / "cache-off", reference)
    current, current_soak, current_summary = run(traced / "cache-off", traced)
    require_matched_soaks(base, base_soak, current, current_soak, {"soak.py": expected_soak_hash})
    require_restored_reservation(traced)

    snapshots = current_soak["rss_samples"]
    baseline = next(sample for sample in snapshots if sample["at_seconds"] >= current_soak["seconds"] / 2)
    final = current_soak["final"]
    require([sample for sample in snapshots if "smaps_rollup" in sample] == [baseline]
            and current_soak["checks"].get("precise_rss_coverage") is True,
            "precise measurements did not bracket the measured second half")
    start_rollup, end_rollup = baseline["smaps_rollup"], final.get("smaps_rollup")
    check_rollup(start_rollup)
    check_rollup(end_rollup)
    rss_growth = end_rollup["rss_kib"] - start_rollup["rss_kib"]
    anonymous_growth = end_rollup["anonymous_kib"] - start_rollup["anonymous_kib"]
    precise = {"baseline": start_rollup, "drain": end_rollup,
               "rss_growth_kib": rss_growth, "anonymous_growth_kib": anonymous_growth}
    require(current_soak["smaps_rollup_second_half"] == precise, "rollup growth differs from raw snapshots")

    late_latency = {}
    for key in ("ttft_ms_p95", "e2e_ms_p95"):
        before, after = (result["latency_windows"]["late"][key] for result in (base_soak, current_soak))
        late_latency[key] = {"repeat": before, "instrumented": after, "change_percent": (after / before - 1) * 100}
    return {"repeat": base_summary, "instrumented": current_summary,
            "smaps_rollup_second_half": precise, "smaps_rss_zero_growth": rss_growth <= 0,
            "status_and_rollup_verdicts_agree": current_summary["rss_zero_growth"] == (rss_growth <= 0),
            "late_p95_ms": late_latency}


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("traced", type=Path)
    parser.add_argument("soak_hashes", type=Path)
    args = parser.parse_args()
    soak_hash, soak_path = args.soak_hashes.read_text().split()
    require(soak_path == "tools/soak.py", "expected the patched soak client hash")
    print(json.dumps(compare(args.reference, args.traced, soak_hash), indent=2, allow_nan=False))
