"""Match the 7-second tcache comparison and retain the original soak decisions."""

from argparse import ArgumentParser
import json
from pathlib import Path

from periodic_trim import PERIOD_MS, TCACHE_TUNABLE
from verify_periodic_trim import IDENTITY, compare, digest, read, read_soak, require


MODES = ("cache-off", "cache-on")


def cpu_preflight(stage):
    baseline, disabled = (read(stage / f"probe-{mode}.json") for mode in ("default", "disabled"))
    for item in (baseline, disabled):
        require(item["uid"] == 65532 and item["glibc"] == "2.35" and item["request_bytes"] == 800
                and item["usable_bytes"] >= item["request_bytes"] and not item["other_tunable_present"],
                "smoke did not use the expected container user, libc, and allocation size")
    require(not baseline["explicitly_disabled"] and disabled["explicitly_disabled"]
            and baseline["uordblks_before"] == baseline["uordblks_after"]
            and disabled["uordblks_before"] - disabled["uordblks_after"] >= disabled["request_bytes"],
            "the allocator behavior did not confirm the configuration change")
    return {"default": baseline, "disabled": disabled}


def compare_mode(stage, reference, mode):
    candidate, baseline = (root / mode / "gpu-evidence" for root in (stage, reference))
    base_report, base_soak = read_soak(baseline)
    report, soak = read_soak(candidate)
    require(all(base_report[key] == report[key] for key in IDENTITY),
            "the trials differ in image, binary, GPU, model or server arguments")
    base_gate, gate = (compare(root, mode) for root in (reference, stage))
    require(base_gate["periodic_trim"]["period_ms"] == gate["periodic_trim"]["period_ms"] == 7000,
            "the trim period changed")
    marker = "W5-TCACHE-ACTIVE pid=1 " + TCACHE_TUNABLE
    for directory, expected in ((baseline, False), (candidate, True)):
        lines = (directory / "soak.log").read_text().splitlines()
        require((marker in lines) == expected and f"W5-TRIM-ACTIVE pid=1 period_ms={PERIOD_MS}" in lines,
                "unexpected runtime allocator configuration")
    require(base_soak["counts"] == soak["counts"], "request outcomes are different")
    late = {}
    for field in ("ttft_ms_p95", "e2e_ms_p95"):
        before, after = (item["latency_windows"]["late"][field] for item in (base_soak, soak))
        require(before > 0 and after > 0, f"invalid p95 {field}")
        late[field] = {"default": before, "disabled": after, "change_percent": (after / before - 1) * 100}
    return {
        "base_7second_report_sha256": digest(baseline / "report.json"),
        "disabled_report_sha256": digest(candidate / "report.json"),
        "same_20gib_reservation_restored": gate["same_20gib_reservation_restored"],
        "default_failed_checks": base_gate["candidate_failed_checks"],
        "disabled_failed_checks": gate["candidate_failed_checks"],
        "default_rss": base_gate["candidate_rss"],
        "disabled_rss": gate["candidate_rss"],
        "default_return_calls": base_gate["periodic_trim"],
        "disabled_return_calls": gate["periodic_trim"],
        "late_p95_ms": late,
    }


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("stage", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()
    proof = {"cpu_probe": cpu_preflight(args.stage)}
    if not args.cpu_only:
        proof["modes"] = {
            mode: compare_mode(args.stage, args.reference, mode)
            for mode in MODES if (args.stage / mode / "exit").is_file()
        }
    print(json.dumps(proof, indent=2, allow_nan=False))
