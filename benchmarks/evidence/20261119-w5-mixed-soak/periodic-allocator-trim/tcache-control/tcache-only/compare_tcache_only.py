"""Compare the unmodified image, tcache-only, and tcache with periodic trim."""

from argparse import ArgumentParser
import json
from pathlib import Path

from compare_tcache import MODES, cpu_preflight
from verify_periodic_trim import CONTROLS, IDENTITY, digest, read, read_soak, require


PROOF = "W5-TCACHE-ONLY pid=1 GLIBC_TUNABLES=glibc.malloc.tcache_count=0"


def compare_mode(stage, combined, mode):
    arm = stage / mode
    paths = {"unmodified": stage.parent / CONTROLS[mode],
             "tcache_only": arm / "gpu-evidence",
             "tcache_with_trim": combined / mode / "gpu-evidence"}
    records = {setting: read_soak(path) for setting, path in paths.items()}
    base, baseline_soak = records["unmodified"]
    require(all(all(report[key] == base[key] for key in IDENTITY) and report["mixed_soak"]["mode"] == mode
                and soak["counts"] == baseline_soak["counts"] for report, soak in records.values()),
            f"{mode}: image, workload, server arguments or outcomes differ")
    for point in ("ready", "after-soak"):
        value = (paths["tcache_only"] / f"allocator-{point}.txt").read_text().strip()
        require(value == PROOF, f"{mode}: tcache-only process proof missing at {point}")
    candidate, _ = records["tcache_only"]
    require("periodic_trim" not in candidate and not (paths["tcache_only"] / "periodic-trim.jsonl").exists(),
            f"{mode}: periodic trimming was present in tcache-only trial")
    status = int((arm / "exit").read_text())
    require(status == int(candidate["status"] == "failed"), f"{mode}: strict soak verdict and exit differ")
    before, after = (read(arm / f"idle-{point}.json") for point in ("before", "after"))
    require(all(before[key] == after[key] for key in before if key != "updated_ns")
            and after["physical_index"] == 6 and after["pid"] == 2431228 and after["phase"] == "idle"
            and not after["other_processes"] and after["allocation_bytes"] == after["torch_reserved_bytes"] == 20 * 1024**3,
            f"{mode}: known 20 GiB reservation was not restored")
    summaries = {}
    for setting, (report, soak) in records.items():
        late = {name: soak["latency_windows"]["late"][name] for name in ("ttft_ms_p95", "e2e_ms_p95")}
        require(all(value > 0 for value in late.values()), f"{mode}: {setting} lacks p95 sample")
        summaries[setting] = {"report_sha256": digest(paths[setting] / "report.json"), "status": report["status"],
                              "failed_checks": report["mixed_soak"]["failed_checks"],
                              "rss": report["mixed_soak"]["rss"], "late_p95_ms": late}
    return {"same_20gib_reservation_restored": True, "same_request_outcomes": True,
            "checks_api": candidate["api_checks"], "configurations": summaries}


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("stage", type=Path)
    parser.add_argument("combined", type=Path)
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()
    proof = {"cpu_probe": cpu_preflight(args.stage)}
    if not args.cpu_only:
        proof["modes"] = {mode: compare_mode(args.stage, args.combined, mode)
                          for mode in MODES if (args.stage / mode / "exit").is_file()}
    print(json.dumps(proof, indent=2, allow_nan=False))
