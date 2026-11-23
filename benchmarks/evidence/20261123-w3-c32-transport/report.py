"""Join the transport pair and retain the non-server-traced controls."""

from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import sys

from verify_transport import IDENTITY, SCOPE, require, verify


def read(path):
    return json.loads(path.read_bytes())


def transport_report(stage):
    gpu_results = {}
    for index in (6, 7):
        folders = {(limit, trace): stage / f"private-short-d{limit}-{trace}-gpu{index}"
                   for limit in (0, 8) for trace in ("control", "transport")}
        baseline, candidate = (folders[(limit, "transport")] for limit in (0, 8))
        traced = verify(baseline, candidate)
        for limit in (0, 8):
            control, trace = (folders[(limit, mode)] for mode in ("control", "transport"))
            cpu_report = read(control / "receipt.json")
            trace_report = read(trace / "receipt.json")
            require(cpu_report["trace_mode"] == "control" and cpu_report["stage_trace"] is False
                    and cpu_report["client_exit"] == cpu_report["shutdown_exit"] == 0
                    and cpu_report["short_burst_decode_limit"] == limit
                    and all(cpu_report[key] == trace_report[key] for key in SCOPE), "control scope differs")
            require(not any(line.startswith("W3-REQUEST-PHASE ") for line in
                            (control / "server.log").read_text().splitlines()), "control logged server timestamps")
            requests = []
            for folder, receipt in ((control, cpu_report), (trace, trace_report)):
                workload = (folder / "workload.json").read_bytes()
                require(sha256(workload).hexdigest() == receipt["artifacts_sha256"]["workload.json"], "workload changed")
                samples = json.loads(workload)["short"]["rows"]
                require(len(samples) == 96 and all(row["status"] == 200 and not row["error"]
                        and isfinite(row["ttft"]) and row["ttft"] > 0 for row in samples), "short replay failed")
                requests.append(samples)
            before, after = requests
            require(all(all(a[key] == b[key] for key in IDENTITY) for a, b in zip(before, after, strict=True)),
                    "trace changed a short response")
            times = sorted(row["ttft"] * 1000 for row in before)
            traced[f"control_d{limit}"] = {"n": len(times), "p50_ms": times[int(.50 * len(times))],
                                           "p95_ms": times[int(.95 * len(times))]}
        lease_before, lease_after = (read(stage / f"gpu{index}-{phase}.json") for phase in ("before", "after"))
        for lease in (lease_before, lease_after):
            require(lease["status"] == "running" and lease["phase"] == "idle" and lease["physical_index"] == index
                    and lease["gpu"] == traced["scope"]["gpu_uuid"] and lease["allocation_bytes"] == 20 * 2**30
                    and lease["other_processes"] == [], "GPU holder did not reserve 20 GiB")
        require(all(lease_before[key] == lease_after[key] for key in ("pid", "start_ticks")), "GPU holder changed")
        gpu_results[str(index)] = traced
    return {"status": "verified", "remote_evidence": str(stage), "gpus": gpu_results}


if __name__ == "__main__":
    print(json.dumps(transport_report(Path(sys.argv[1])), indent=2, allow_nan=False))
