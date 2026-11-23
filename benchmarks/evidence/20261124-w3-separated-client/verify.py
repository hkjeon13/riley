"""Check CPU placement, timestamps and the existing fixed-arrival performance gates."""

import argparse
from hashlib import sha256
import gzip
import json
from math import isfinite
from pathlib import Path
import re
import runpy
import tarfile

EVIDENCE = Path(__file__).resolve().parent.parent
RAG = runpy.run_path(str(EVIDENCE / "20261124-w3-coder-arrivals" / "verify-replay.py"))
C32 = runpy.run_path(str(EVIDENCE / "20261124-w3-coder-c32" / "verify.py"))
BASE = RAG["BASE"]
require, read = (RAG[key] for key in ("require", "read"))


def source_schedules():
    with tarfile.open(EVIDENCE / "20261124-w3-coder-arrivals/raw-results.tar.gz", "r:gz") as archive:
        rag = archive.extractfile("schedules.json").read()
    c32 = gzip.decompress((EVIDENCE / "20261124-w3-coder-c32/schedules.json.gz").read_bytes())
    for raw, expected in ((rag, "9b4db373c4b9b7307dd92e63460d37674986873cf391438b7bfb850bf17b8572"),
                          (c32, C32["SCHEDULE_SHA"])):
        require(sha256(raw).hexdigest() == expected, "frozen input schedules changed")
    left, right = (json.loads(raw)["schedules"] for raw in (rag, c32))
    return {index: {**left[index], **right[index]} for index in left}


def placement(stage, gpu, engine, label, pinned_cpu):
    out = stage / f"gpu{gpu}/coder14-{engine}"
    source = read(out / f"{label}.json")
    timing = source["client_timing"]
    epoch = timing["epoch"]
    require(timing["cpu_affinity"] == [pinned_cpu] and isfinite(epoch)
            and len(timing["schedstat_delta"]) == 3
            and all(type(counter) is int and counter >= 0 for counter in timing["schedstat_delta"]),
            "client did not run on its selected CPU, or scheduling counters are invalid")
    wake, dispatch = [], []
    for row in source["measured"]:
        require(isfinite(row["wake_at"]) and epoch <= row["wake_at"] <= row["start"] <= row["sent_at"],
                "invalid client wake, start or write order")
        wake.append(abs((row["wake_at"] - epoch) * 1e3 - row["scheduled_write_ns"] / 1e6))
        dispatch.append((row["start"] - row["wake_at"]) * 1e3)
    return {"client_cpu": pinned_cpu, "wake_error_abs_p95_ms": RAG["p95"](wake),
            "wake_to_start_p95_ms": RAG["p95"](dispatch), "schedstat_delta": timing["schedstat_delta"]}


def check_attempts(stage, retry):
    exit_code = (stage / "campaign.exit").read_text().strip()
    if exit_code == "0":
        require(retry is None, "unexpected retry for a successful run")
        BASE["confirm_order"](stage)
        return (6, 7), None, None
    log = (stage / "campaign.log").read_text()
    events = re.findall(r"^(START|END) gpu([67]) (riley|vllm) (\S+?)(?: exit=(\d+))?$", log, re.MULTILINE)
    require(exit_code == "1" and len(events) == 8
            and "TimeoutError: server readiness" in (stage / "gpu6/coder14-vllm/client.log").read_text(),
            "unexpected original replay failure")
    finish = None
    for gpu, engines in BASE["ORDER"].items():
        local = [event for event in events if int(event[1]) == gpu]
        actual = [(phase, engine, code) for phase, _, engine, when, code in local]
        expected = [(phase, engine, code) for engine in engines
                    for phase, code in (("START", ""), ("END", "1" if (gpu, engine) == (6, "vllm") else "0"))]
        require(actual == expected and [event[3] for event in local] == sorted(event[3] for event in local),
                f"GPU {gpu} original run order changed")
        if gpu == 6:
            finish = local[-1][3]
    failure = {"gpu": 6, "engine": "vllm", "exit": 1, "reason": "server readiness timeout", "ended_at": finish}
    if retry is None:
        return (7,), failure, None
    retry_exit = (retry / "campaign.exit").read_text().strip()
    events = re.findall(r"^(START|END) gpu6 vllm (\S+?)(?: exit=(\d+))?$",
                        (retry / "campaign.log").read_text(), re.MULTILINE)
    require(retry_exit in {"0", "1"} and len(events) == 2
            and events[0][0] == "START" and events[0][2] == ""
            and events[1][0] == "END" and events[1][2] == retry_exit
            and finish < events[0][1] <= events[1][1], "GPU 6 retry order or exit differs")
    for name in ("runtime.sha256", "schedules.json", "cpu-placement.json", "vllm-version.txt"):
        require((stage / name).read_bytes() == (retry / name).read_bytes(), f"retry changed {name}")
    original = BASE["reservations"](stage, 6)
    for phase in ("before", "after"):
        row = read(retry / f"gpu6/{phase}.json")
        require(row["status"] == "running" and row["phase"] == "idle" and row["physical_index"] == 6
                and row["allocation_bytes"] == row["torch_reserved_bytes"] == 20 * 2**30 and not row["other_processes"]
                and all(row[key] == original[key] for key in ("pid", "start_ticks", "gpu")),
                "GPU 6 retry reservation differs")
    retry_status = {"gpu": 6, "engine": "vllm", "exit": int(retry_exit),
                    "started_at": events[0][1], "ended_at": events[1][1]}
    if retry_exit == "1":
        attempt = retry / "gpu6/coder14-vllm"
        require("TimeoutError: server readiness" in (attempt / "client.log").read_text()
                and not any((attempt / f"{label}.json").exists() for label in source_schedules()["6"]),
                "unexpected GPU 6 retry failure")
        retry_status["reason"] = "server readiness timeout"
        return (7,), failure, retry_status
    return (6, 7), failure, retry_status


def verify(stage, retry=None):
    require((stage / "vllm-version.txt").read_text().strip() == "0.27.1+cu129", "vLLM version differs")
    plans = read(stage / "schedules.json")["schedules"]
    require(plans == source_schedules(), "combined schedules differ from the two source suites")
    cpus = read(stage / "cpu-placement.json")
    require(cpus["clients"] == {"6": 44, "7": 45}
            and cpus["siblings"] == {"44": "44,92", "45": "45,93"}
            and not ({44, 45, 92, 93} & set(cpus["server_cpus"])) and bool(cpus["server_cpus"]),
            "server and client CPU cores overlap")
    gpu_indexes, failure, retry_status = check_attempts(stage, retry)
    results = {}
    for gpu in gpu_indexes:
        engines = BASE["ORDER"][gpu]
        runs = {engine: retry if (gpu, engine) == (6, "vllm") and retry else stage for engine in engines}
        holder = BASE["reservations"](stage, gpu)
        schedule = plans[str(gpu)]
        for engine, relative in (("riley", "plan.json"), ("vllm", "cpu-placement.json")):
            record = read(runs[engine] / f"gpu{gpu}/coder14-{engine}/{relative}")
            require(record["server_cpu_affinity"] == cpus["server_cpus"], "server used an unexpected CPU set")
        arms = {engine: RAG["arm"](runs[engine], gpu, engine, schedule, holder) for engine in engines}
        comparisons = {}
        for label in schedule:
            riley, vllm = (arms[engine][label] for engine in ("riley", "vllm"))
            result = RAG["pair"](riley, vllm)
            if label.startswith("c32-"):
                del result["ttft_p95_stage_met"], result["ttft_p95_final_met"]
                result["tpot_p50_goal_met"] = (result["riley_over_vllm"]["tpot_p50_ms"] <= 1.03
                                               if result["valid_matched_comparison"] else None)
            result["client_timing"] = {engine: placement(runs[engine], gpu, engine, label, cpus["clients"][str(gpu)])
                                       for engine in engines}
            result["content_gaps"] = {engine: C32["content_gaps"](arms[engine][label]) for engine in engines}
            comparisons[label] = result
        results[gpu] = comparisons
    report = {"source_revision": RAG["REVISION"], "cpu_placement": cpus, "initial_failure": failure,
              "max_write_error_p95_ms": RAG["MAX_WRITE_ERROR_P95_MS"], "by_gpu": results}
    if retry_status is not None:
        report["retry"] = retry_status
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", type=Path)
    parser.add_argument("--retry", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.stage, args.retry), indent=2, allow_nan=False))
