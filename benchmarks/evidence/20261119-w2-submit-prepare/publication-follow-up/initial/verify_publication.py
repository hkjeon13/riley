"""Verify the same workloads and classify graph gaps for explicit publication modes."""

from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import sys

from preflight import BINARY, MODES, ORDER, TRACE_MODES


HOLDERS = {6: (2431228, "6339570869"), 7: (2431237, "6339570875")}
FIELDS = ("request_index", "input_sha256", "output_sha256", "prompt_tokens", "completion_tokens", "finish")
METRICS = ("ttft_p95_ms", "tpot_p50_ms", "itl_p99_ms", "output_tok_s")


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def read(path):
    raw = path.read_bytes()
    return json.loads(raw), sha256(raw).hexdigest()


def status(folder, index, period):
    row, _ = read(folder / f"gpu{index}-{period}.json")
    keys = ("physical_index", "pid", "start_ticks", "phase", "allocation_bytes", "torch_reserved_bytes", "other_processes")
    record = {key: row[key] for key in keys}
    require(tuple(record.values()) == (index, *HOLDERS[index], "idle", 20 * 1024**3, 20 * 1024**3, []),
            "GPU did not retain its original reservation")
    return record


def check_arm(stage, prior, name):
    folder = stage / f"smol-{name}"
    trace = name in TRACE_MODES
    policy = "after-submit-v1" if name.startswith("after-") else "immediate-v1"
    receipt, receipt_hash = read(folder / "receipt.json")
    plan, plan_hash = read(folder / "plan.json")
    frozen, _ = read(prior / "smol-base-first" / "plan.json")
    address, old_address = plan["address"], frozen["address"]
    server = plan["server"]
    args = [old_address if arg == address else arg for arg in server[1:]]
    require(address.startswith("127.0.0.1:") and args[-2:] == ["--event-publication", policy]
            and args[:-2] == frozen["server"][1:-2], f"{name}: serving arguments other than policy changed")
    require(receipt["status"] == "passed" and receipt["mode"] == name and receipt["common_kv_blocks"] == 9006
            and receipt["model"] == "SmolLM2-1.7B-Instruct" and receipt["nvml_samples"] > 0
            and receipt["holder_pid"] == HOLDERS[7][0] and receipt["holder_start_ticks"] == HOLDERS[7][1],
            f"{name}: incomplete execution receipt")
    require(receipt["sha256"][server[0]] == BINARY and receipt["sha256"][str(folder / "plan.json")] == plan_hash,
            f"{name}: unexpected binary or plan")
    require(plan["profile"] is trace and plan["check_token_ids"] is not trace and
            plan["env"] == ({"RILEY_HOST_STEP_TRACE": "1"} if trace else {}), "instrumentation changed")
    require(read(folder / "server-exit.json")[0]["exit"] == 0, "server did not shut down normally")
    holders = {str(index): status(folder, index, "after") for index in HOLDERS}
    require(all(holders[str(index)] == status(folder, index, "before") for index in HOLDERS), "holder changed")

    cases = {}
    names = ("reason-c32",) if trace else ("reason-c32", "long-c32")
    require(tuple(plan["cells"]) == names and set(receipt["results"]) == set(names), "invalid cells")
    for label in names:
        result, result_hash = read(folder / f"{label}.json")
        count, warmup = (32, 0) if trace else (64, 32)
        reference = frozen["cells"][label]
        spec = {**reference, "requests": count, "warmup": warmup}
        stats = result["statistics"]
        rows = result["measured"]
        require(plan["cells"][label] == stats["config"] == spec and receipt["results"][label] == result_hash
                and stats["ok"] == len(rows) == count and stats["warmup_ok"] == warmup
                and stats["errors"] == stats["rejected_429"] == 0
                and [row["request_index"] for row in rows] == list(range(warmup, warmup + count)),
                f"{name}: incomplete workload")
        metrics = {key: stats[key] for key in METRICS}
        require(all(isfinite(value) and value > 0 for value in metrics.values()), "invalid metrics")
        cases[label] = ({"metrics": metrics, "requests": count, "sha256": result_hash},
                        [tuple(row[key] for key in FIELDS) for row in rows])
    base = {"policy": policy, "receipt_sha256": receipt_hash, "holders": holders}
    if trace:
        summary, summary_hash = read(folder / "sync-gaps.json")
        require(summary["sqlite_sha256"] == receipt["sqlite_sha256"] and summary["adjacent_matched_gaps"] >= 2,
                f"{name}: graph lane not bound to the profile receipt")
        base["trace"] = {**summary, "summary_sha256": summary_hash, "nsys_sha256": next(
            digest for key, digest in receipt["sha256"].items() if Path(key).name == "nsys")}
    else:
        probe, probe_hash = read(folder / "token-id-probe.json")
        require(probe_hash == receipt["token_id_probe_sha256"] and probe["invariant"]
                and len(probe["c1"]) == len(probe["c8"]) == 8 and probe["c1"] == probe["c8"],
                f"{name}: token-ID probe incomplete")
        base["token_id_probe_sha256"] = probe_hash
    return base, cases


def verify(stage, prior):
    events = [line.split() for line in (stage / "campaign.log").read_text().splitlines()
              if line.startswith(("START ", "END "))]
    expected = [(event, name) for name in ORDER for event in ("START", "END")]
    require([tuple(row[:2]) for row in events] == expected and all(row[3:] == ["exit=0"] for row in events[1::2])
            and [row[2] for row in events] == sorted(row[2] for row in events), "incomplete or overlapping run")
    verified = {name: check_arm(stage, prior, name) for name in ORDER}
    arms = {name: base for name, (base, _) in verified.items()}
    require(len({arms[name]["token_id_probe_sha256"] for name in MODES}) == 1, "token ID probes differ")
    require(len({arms[name]["trace"]["nsys_sha256"] for name in TRACE_MODES}) == 1, "Nsight tool differs")
    profiles = {}
    serving = {}
    for phase, modes, results in (("serving", MODES, serving), ("profile", TRACE_MODES, profiles)):
        labels = verified[modes[0]][1]
        for label in labels:
            cases = {name: verified[name][1][label] for name in modes}
            require(all(case[1] == cases[modes[0]][1] for case in cases.values()), "response content or usage differs")
            pairs = {}
            for order in ("first", "second"):
                left, right = (cases[f"{policy}-{order}" + ("-trace" if phase == "profile" else "")][0]["metrics"]
                               for policy in ("immediate", "after"))
                pairs[order] = {key: (right[key] / left[key] - 1) * 100 for key in METRICS}
            results[label] = {"pairs_after_vs_immediate_percent": pairs, "arms": {key: value[0] for key, value in cases.items()}}
    return {"binary_sha256": BINARY, "execution_order": ORDER, "chronology": events, "arms": arms,
            "normal_serving": serving, "profiled_serving": profiles, "same_complete_streams": True}


if __name__ == "__main__":
    print(json.dumps(verify(*map(Path, sys.argv[1:])), indent=2, allow_nan=False))
