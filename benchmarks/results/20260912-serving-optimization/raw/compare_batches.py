#!/usr/bin/env python3
"""Compare completed baseline/candidate/batch2 HTTP and engine summaries.

Usage: python3 compare_batches.py ROOT > comparison.json
ROOT contains {baseline,candidate,batch2}-{http,engine}/summary.json.
Missing or unfinished campaigns are pending. Invalid evidence is reported as
invalid and returns exit status 1. This script reads summaries without running
benchmarks or modifying any campaign artifact.
"""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


CAMPAIGNS = ("baseline", "candidate", "batch2")
ROLES = ("riley", "vllm")
LATENCY_FIELDS = {
    "http": ("e2e_ms", "first_text_event_ms"),
    "engine": ("ttft_ms", "tpot_ms", "e2e_ms"),
}
THROUGHPUT_FIELDS = {
    "http": "output_tokens_per_wall_second",
    "engine": "output_tokens_per_request_service_second",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def positive(value, label):
    require(not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and value > 0, "invalid positive metric: " + label)
    return value


def distribution(values):
    return {"process_values": values, "median": statistics.median(values),
            "range": [min(values), max(values)]}


def request_identity(summary):
    hashes = {digest for path, digest in summary["inputs"].items()
              if Path(path).name == "request.json"}
    require(len(hashes) == 1, "summary lacks an unambiguous request identity")
    return next(iter(hashes))


def summarize_campaign(summary, mode):
    require(summary["schema_version"] == ("riley.serving-optimization.v1" if mode == "http"
                                            else "riley.engine-optimization.v1"),
            "wrong runner summary schema")
    require(summary["measurement_started"] is True, "summary does not attest measurement")
    require(summary["pairs"] == 5 and summary["iterations"] == 30,
            "campaign must contain five pairs of thirty retained requests")
    warmups = summary["warmups_per_transport" if mode == "http" else "warmups"]
    require(warmups == 5, "campaign warmup count differs")
    pairs = sorted(summary["pair_results"], key=lambda pair: pair["index"])
    require([pair["index"] for pair in pairs] == list(range(1, 6)),
            "campaign pair indices must be exactly 1 through 5")
    for pair in pairs:
        require(pair["order"] == (list(ROLES) if pair["index"] % 2 else list(reversed(ROLES))),
                "campaign process order differs from alternating AB/BA")
        for role in ROLES:
            lane = pair[role]
            require(lane["requests"] == 30, "process has an incomplete retained sample")
            if mode == "engine":
                require(lane["tokens_exact"] is True, "engine exact-token check is absent")
            for field in LATENCY_FIELDS[mode]:
                vals = [positive(lane[field][stat], role + "." + field + "." + stat)
                        for stat in ("median", "p95", "p99")]
                require(vals == sorted(vals), "latency percentile ordering differs: " + field)
            positive(lane[THROUGHPUT_FIELDS[mode]], role + " throughput")
    fields = {}
    for field in (*LATENCY_FIELDS[mode], THROUGHPUT_FIELDS[mode]):
        latency = field in LATENCY_FIELDS[mode]
        stats = {}
        for stat in ("median", "p95", "p99") if latency else ("rate",):
            values = {role: [pair[role][field][stat] if latency else pair[role][field]
                             for pair in pairs] for role in ROLES}
            values["riley_over_vllm"] = [riley / vllm for riley, vllm
                                          in zip(values["riley"], values["vllm"])]
            stats[stat] = {role: distribution(samples) for role, samples in values.items()}
        fields[field] = {"unit": "ms" if latency else "output_tokens/s",
                         "better": "lower" if latency else "higher", "statistics": stats}
    return {
        "status": "complete", "source_commit": summary["source_commit"],
        "condition": summary["condition"], "request_sha256": request_identity(summary),
        "runner_sha256": summary["runner_sha256"],
        "shared_runner_sha256": summary.get("shared_runner_sha256"),
        "pair_indices": [pair["index"] for pair in pairs],
        "processes_per_lane": 5, "retained_requests_per_process": 30,
        "retained_requests_per_lane": 150, "warmups": warmups,
        "metrics": fields,
    }


def load_campaign(path, mode):
    if not path.exists():
        return {"status": "pending", "summary_path": str(path), "reason": "summary absent"}
    try:
        raw = path.read_bytes()
        summary = json.loads(raw)
        if summary.get("completed") is not True:
            return {"status": "pending", "summary_path": str(path),
                    "reason": "campaign does not attest completion"}
        result = summarize_campaign(summary, mode)
        return {"summary_path": str(path), "summary_sha256": hashlib.sha256(raw).hexdigest(),
                **result}
    except (KeyError, TypeError, ValueError, OSError) as error:
        return {"status": "invalid", "summary_path": str(path), "reason": str(error)}


def changes(new, previous):
    if new["status"] != "complete" or previous["status"] != "complete":
        return {"status": "invalid" if "invalid" in (new["status"], previous["status"])
                else "pending", "reason": "both completed campaigns are required"}
    if new["condition"] != previous["condition"] or new["request_sha256"] != previous["request_sha256"]:
        return {"status": "incomparable", "reason": "condition or request identity differs"}
    if (new["runner_sha256"] != previous["runner_sha256"]
            or new["shared_runner_sha256"] != previous["shared_runner_sha256"]):
        return {"status": "incomparable", "reason": "measurement runner identity differs"}
    fields = {}
    for field, current in new["metrics"].items():
        old = previous["metrics"][field]
        fields[field] = {
            stat: {role + "_change_pct": 100 * (values[role]["median"]
                     / old["statistics"][stat][role]["median"] - 1)
                   for role in (*ROLES, "riley_over_vllm")}
            for stat, values in current["statistics"].items()
        }
    return {"status": "computed", "metrics": fields}


def compare(root):
    result = {
        "schema_version": "riley.optimization-batches-comparison.v1",
        "scope": "SmolLM2-135M BF16 c1 P128 O32; GUI retained; five fresh AB/BA process pairs",
        "campaign_labels": {"baseline": "original baseline", "candidate": "rejected batch1",
                            "batch2": "P128 parallel prefill batch"},
        "goal_complete": False, "high_concurrency_claim": False,
        "canonical_qualification": False,
        "aggregation": {
            "process_values": "Arrays follow pair_indices; each lane has one fresh process per pair.",
            "latency": "Each value is a retained process median, P95 or P99; summary median/range spans processes.",
            "paired_ratio": "Riley/vLLM within the same pair, then median/range; not a ratio of campaign medians.",
            "changes": "100 * (new/previous - 1), using medians of process statistics or same-pair ratios; campaigns are separate trials.",
            "tails": "Nearest-rank tails from 30 requests per process, not pooled 150-request percentiles or high-concurrency stability.",
            "evidence": "Completed runner summaries only; raw artifacts are not revalidated by this analysis.",
        },
        "modes": {},
    }
    for mode in ("http", "engine"):
        campaigns = {name: load_campaign(root / (name + "-" + mode) / "summary.json", mode)
                     for name in CAMPAIGNS}
        statuses = [campaign["status"] for campaign in campaigns.values()]
        status = ("invalid" if "invalid" in statuses else "complete" if all(x == "complete" for x in statuses)
                  else "pending" if all(x == "pending" for x in statuses) else "partial")
        semantics = ({"latency": "HTTP E2E and first nonempty text event; first text is not token-aligned TTFT.",
                      "token_ttft": "unmeasured", "tpot": "unmeasured",
                      "throughput": "Observed retained-request wall interval, including gaps between requests."}
                     if mode == "http" else
                     {"latency": "Engine request TTFT, mean TPOT and E2E from exact-token-validated runs.",
                      "throughput": "Output tokens divided by sum of request E2E service durations; excludes inter-request gaps."})
        result["modes"][mode] = {
            "status": status, "semantics": semantics, "campaigns": campaigns,
            "changes": {
                "rejected_batch1_vs_baseline": changes(campaigns["candidate"], campaigns["baseline"]),
                "batch2_vs_baseline": changes(campaigns["batch2"], campaigns["baseline"]),
                "batch2_vs_rejected_batch1": changes(campaigns["batch2"], campaigns["candidate"]),
            },
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    result = compare(args.root.resolve())
    print(json.dumps(result, indent=2, allow_nan=False))
    return int(any(mode["status"] == "invalid" for mode in result["modes"].values()))


if __name__ == "__main__":
    raise SystemExit(main())
