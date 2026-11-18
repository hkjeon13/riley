"""Recompute the explicit-mode comparison from the leaf run receipts."""
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(sys.argv[1])
CELL = "reason-c32"
ARMS = {"immediate1": ("immediate-v1", 1), "after1": ("after-submit-v1", 1),
        "after2": ("after-submit-v1", 2), "immediate2": ("immediate-v1", 2)}
METRICS = ("output_tok_s", "ttft_p50_ms", "ttft_p95_ms", "ttft_p99_ms",
           "tpot_p50_ms", "tpot_p95_ms", "tpot_p99_ms", "itl_p99_ms")
OUTPUT = ("output_sha256", "finish", "prompt_tokens", "completion_tokens")
IDENTITY = ("request_index", "input_sha256")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_arm(arm, policy, seed):
    directory = ROOT / arm
    receipt = json.loads((directory / "receipt.json").read_text())
    holder = json.loads((directory / "idle-after.json").read_text())
    argv = receipt["server_argv"]
    require(argv.count("--event-publication") == 1
            and argv[argv.index("--event-publication") + 1] == policy
            and receipt["event_publication"] == policy and receipt["seed"] == seed,
            f"{arm}: event publication and seed differ from the declared arm")
    require(receipt["status"] == "passed" and receipt["observed_active"] and receipt["server_exit"] == 0,
            f"{arm}: execution state")
    require(holder["phase"] == "idle" and holder["allocation_bytes"] == 20 * 2**30
            and holder["gpu"] == receipt["gpu_uuid"] and not holder["other_processes"]
            and (holder["pid"], holder["start_ticks"]) == (receipt["holder_pid"], receipt["holder_start_ticks"]),
            f"{arm}: holder state")
    for name, key in ((f"{CELL}.json", "summary_sha256"), (f"{CELL}.records.json", "records_sha256")):
        actual_hash = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        require(actual_hash == receipt["cells"][CELL][key], f"{arm}: hash mismatch for {name}")
    result = json.loads((directory / f"{CELL}.json").read_text())
    rows = json.loads((directory / f"{CELL}.records.json").read_text())
    require(result["ok"] == result["requests"] == len(rows) == 64
            and result["warmup_ok"] == result["warmup_requests"] == 32
            and result["errors"] == result["warmup_errors"] == result["rejected_429"] == 0,
            f"{arm}: request totals")
    require([row["request_index"] for row in rows] == list(range(32, 96)), f"{arm}: request indexes")
    require(all(math.isfinite(result[k]) and result[k] > 0 for k in METRICS), f"{arm}: nonpositive metrics")
    return {k: result[k] for k in METRICS}, rows, receipt


results = {arm: read_arm(arm, policy, seed) for arm, (policy, seed) in ARMS.items()}
digests = set()
for _, _, receipt in results.values():
    digests.add(tuple(sorted(receipt["sha256"].items())))
require(len(digests) == 1, "binary, model and tool bindings differ among arms")
comparisons = []
for seed in (1, 2):
    before, base_rows, _ = results[f"immediate{seed}"]
    after, candidate_rows, _ = results[f"after{seed}"]
    require(all(all(left[k] == right[k] for k in IDENTITY)
                for left, right in zip(base_rows, candidate_rows)), f"seed {seed}: different requests")
    matches = sum(all(left[k] == right[k] for k in OUTPUT)
                  for left, right in zip(base_rows, candidate_rows))
    differences = {key: (after[key] / before[key] - 1) * 100 for key in METRICS}
    comparisons.append({"seed": seed, "matched_text_finish_usage": matches, "compared": len(base_rows),
                        "immediate": before, "after_submit": after, "percent_change": differences})
report = {"cell": CELL, "model": "SmolLM2-1.7B-Instruct", "physical_gpu": 7, "order": list(ARMS),
          "comparisons": comparisons}
print(json.dumps(report, indent=2, allow_nan=False))
