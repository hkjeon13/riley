"""Compare the single-download-fence candidate to the exact baseline on A100."""

from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import sys


BINARY = "306a4b25f35ddd531fea9b7219484d5c10c5752084e312140dcfd15f5b1902b8"
ORDER = ("base-first", "candidate-first", "candidate-second", "base-second")


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


def check_arm(stage, prior, name, binary_hash):
    folder = stage / f"smol-{name}"
    receipt, receipt_hash = read(folder / "receipt.json")
    plan, plan_hash = read(folder / "plan.json")
    frozen, _ = read(prior / "smol-base-first" / "plan.json")
    address, old_address = plan["address"], frozen["address"]
    server = plan["server"]
    args = [old_address if arg == address else arg for arg in server[1:]]
    require(address.startswith("127.0.0.1:") and args[-2:] == ["--event-publication", "after-submit-v1"]
            and args[:-2] == frozen["server"][1:-2], f"{name}: serving arguments other than policy changed")
    require(receipt["status"] == "passed" and receipt["mode"] == name and receipt["common_kv_blocks"] == 9006
            and receipt["model"] == "SmolLM2-1.7B-Instruct" and receipt["nvml_samples"] > 0
            and receipt["holder_pid"] == HOLDERS[7][0] and receipt["holder_start_ticks"] == HOLDERS[7][1],
            f"{name}: incomplete execution receipt")
    require(receipt["sha256"][server[0]] == binary_hash and receipt["sha256"][str(folder / "plan.json")] == plan_hash,
            f"{name}: unexpected binary or plan")
    require(not plan["profile"] and plan["check_token_ids"] and plan["env"] == {}, "instrumentation changed")
    require(read(folder / "server-exit.json")[0]["exit"] == 0, "server did not shut down normally")
    holders = {str(index): status(folder, index, "after") for index in HOLDERS}
    require(all(holders[str(index)] == status(folder, index, "before") for index in HOLDERS), "holder changed")

    cases = {}
    names = ("reason-c32", "long-c32")
    require(tuple(plan["cells"]) == names and set(receipt["results"]) == set(names), "invalid cells")
    for label in names:
        result, result_hash = read(folder / f"{label}.json")
        count, warmup = 64, 32
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
    probe, probe_hash = read(folder / "token-id-probe.json")
    require(probe_hash == receipt["token_id_probe_sha256"] and probe["invariant"]
            and len(probe["c1"]) == len(probe["c8"]) == 8 and probe["c1"] == probe["c8"],
            f"{name}: token-ID probe incomplete")
    logprobs, logprob_hash = read(folder / "logprob-probe.json")
    require(set(logprobs) == {"c1", "c8"} and all(len(entries) == 8 and all(
        row["logprob_rows"] > 0 and row["logprob_rows"] == row["usage"]["completion_tokens"] for row in entries
    ) for entries in logprobs.values()), f"{name}: logprobs probe incomplete")
    info = {"receipt_sha256": receipt_hash, "holders": holders, "token_id_probe_sha256": probe_hash,
            "logprob_probe_sha256": logprob_hash}
    return info, cases, logprobs


def verify(stage, prior):
    events = [line.split() for line in (stage / "campaign.log").read_text().splitlines()
              if line.startswith(("START ", "END "))]
    expected = [(event, name) for name in ORDER for event in ("START", "END")]
    require([tuple(row[:2]) for row in events] == expected and all(row[3:] == ["exit=0"] for row in events[1::2])
            and [row[2] for row in events] == sorted(row[2] for row in events), "incomplete or overlapping run")
    candidate, candidate_path = (stage / "candidate.sha256").read_text().split()
    require(candidate_path == "bin/candidate/riley" and candidate != BINARY
            and sha256((stage / candidate_path).read_bytes()).hexdigest() == candidate,
            "candidate binary is missing or matches the baseline")
    verified = {name: check_arm(stage, prior, name, BINARY if name.startswith("base-") else candidate) for name in ORDER}
    arms = {name: info for name, (info, _, _) in verified.items()}
    serving = {}
    for label in ("reason-c32", "long-c32"):
        cases = {name: verified[name][1][label] for name in ORDER}
        same = all(case[1] == cases[ORDER[0]][1] for case in cases.values())
        pairs = {}
        for order in ("first", "second"):
            left, right = (cases[f"{kind}-{order}"][0]["metrics"] for kind in ("base", "candidate"))
            delta = {key: (right[key] / left[key] - 1) * 100 for key in METRICS}
            guards = {"decode_tpot_under_one_percent_regression": delta["tpot_p50_ms"] < 1}
            if label == "reason-c32":
                guards.update(short_ttft_under_two_percent_regression=delta["ttft_p95_ms"] < 2,
                              short_throughput_under_two_percent_regression=delta["output_tok_s"] > -2)
            pairs[order] = {"candidate_percent_change": delta, "guards": guards}
        serving[label] = {"same_outputs": same, "pairs": pairs, "arms": {key: value[0] for key, value in cases.items()}}
    expected_probabilities = verified[ORDER[0]][2]
    full_probabilities_match = {
        mode: {name: [index for index, (left, right) in enumerate(zip(
            verified[name][2][mode], expected_probabilities[mode], strict=True)) if left != right]
               for name in ORDER}
        for mode in expected_probabilities
    }
    return {"scope": "Use one stream fence after staging the CUDA timing end marker and result downloads.",
            "binaries": {"baseline": BINARY, "candidate": candidate}, "execution_order": ORDER, "chronology": events,
            "arms": arms, "normal_serving": serving,
            "all_token_id_probes_match": len({row["token_id_probe_sha256"] for row in arms.values()}) == 1,
            "streamed_logprobs_differing_prompt_indices": full_probabilities_match}


if __name__ == "__main__":
    print(json.dumps(verify(*map(Path, sys.argv[1:])), indent=2, allow_nan=False))
