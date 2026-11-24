"""Verify the registered 512-row experiment on both mixed and isolated traffic."""

import argparse
import json
import tarfile
from pathlib import Path

from verify_coder_fleet import percentile, read, summarize


def wide_steps(root, label, replica):
    buckets = read(root / f"shapes-{label}-{replica}.json")["buckets"]
    return next(r["hit_count"] for r in buckets if r["dense_rows"] == 512)


def check(root, baseline_dir, history):
    evidence = Path(__file__).resolve().parent
    mixed = summarize(root, evidence / "protected-raw.tar.gz")
    baseline = read(evidence / "coder-mixed-fleet-verified.json")
    reference = read(evidence / "coder-fleet-reference-verified.json")
    samples = []
    for i, arm in enumerate(mixed["arms"]):
        repeated = baseline["arms"][0 if i == 0 else 5]
        vllm = reference["arms"][i]
        ratios = {}
        for typ in ("short", "rag"):
            x = arm["classes"][typ]
            prev = repeated["classes"][typ]
            ref = vllm["classes"][typ]
            ratios[typ] = {
                "riley_first_ratio": x["ttft_p95_ms"] / prev["ttft_p95_ms"],
                "riley_last_ratio": x["last_p95_ms"] / prev["last_p95_ms"],
                "vllm_first_ratio": x["ttft_p95_ms"] / ref["ttft_p95_ms"],
                "vllm_last_ratio": x["last_p95_ms"] / ref["last_p95_ms"],
                "vllm_gap_ratio": x["request_gap_p95_across_requests_p95_ms"]
                / ref["request_gap_p95_across_requests_p95_ms"],
            }
        valid = all(
            c["arrival_error_p95_ms"] <= 5
            for a in (arm, repeated, vllm)
            for c in a["classes"].values()
        )
        faithful = all(
            c["matches_reference"] == c["n"] for c in arm["classes"].values()
        )
        latency = (
            ratios["rag"]["vllm_first_ratio"] <= 1.15
            and ratios["rag"]["riley_first_ratio"] <= 0.95
            and ratios["short"]["riley_last_ratio"] <= 1.05
            and all(
                r["vllm_last_ratio"] <= 1 and r["vllm_gap_ratio"] <= 1
                for r in ratios.values()
            )
        )
        deltas = []
        for replica in (0, 1):
            deltas.append(
                wide_steps(root, str(i), replica)
                - wide_steps(root, "warm" if i == 0 else "0", replica)
            )
        samples.append(
            {
                "ratios": ratios,
                "timing_valid": valid,
                "candidate_hashes_match": faithful,
                "latency_targets_met": latency,
                "new_512_steps_by_replica": deltas,
                "pass": latency and valid and faithful and min(deltas) > 0,
            }
        )
    isolated = {}
    with tarfile.open(history) as archive:

        def prior(gpu, engine, name):
            return json.load(
                archive.extractfile(f"gpu{gpu}/coder14-{engine}/{name}.json")
            )

        for gpu in (6, 7):
            label = f"single{gpu}"
            cases = {}
            for name, cell in read(root / label / "plan.json")["cells"].items():
                reference = prior(gpu, "riley", name)["measured"]
                measured = read(root / label / f"{name}.json")
                rows = measured["measured"]
                if len(rows) != len(reference) or len(rows) != len(cell["schedule"]):
                    raise ValueError("missing protected requests")
                same = 0
                for old, new, src in zip(
                    reference, rows, cell["schedule"], strict=True
                ):
                    if (
                        old["input_sha256"] != new["input_sha256"]
                        or src["input_sha256"] != new["input_sha256"]
                        or src["offset_ns"] != new["scheduled_write_ns"]
                    ):
                        raise ValueError("different input schedule")
                    same += all(
                        old[x] == new[x]
                        for x in (
                            "output_sha256",
                            "prompt_tokens",
                            "completion_tokens",
                            "finish",
                        )
                    )
                first = 1000 * percentile([r["ttft"] for r in rows])
                case = {
                    "n": len(rows),
                    "matched": same,
                    "ttft_p95_ms": first,
                    "e2e_p95_ms": 1000 * percentile([r["e2e"] for r in rows]),
                    "tpot_p50_ms": measured["statistics"]["tpot_p50_ms"],
                    "ttft_p50_ms": measured["statistics"]["ttft_p50_ms"],
                    "output_tok_s": measured["statistics"]["output_tok_s"],
                    "arrival_error_p95_ms": percentile(
                        [abs(r["write_error_ms"]) for r in rows]
                    ),
                }
                if gpu == 7:
                    frozen = prior(7, "vllm", name)
                    case["frozen_vllm_ttft_p95_ms"] = frozen["statistics"][
                        "ttft_p95_ms"
                    ]
                    case["frozen_vllm_ttft_ratio"] = (
                        first / case["frozen_vllm_ttft_p95_ms"]
                    )
                    case["paired_write_delta_p95_ms"] = percentile(
                        [
                            abs(a["write_error_ms"] - b["write_error_ms"])
                            for a, b in zip(rows, frozen["measured"], strict=True)
                        ]
                    )
                if gpu == 7 and name == "c32-vllm-arrivals":
                    with tarfile.open(evidence / "protected-raw.tar.gz") as t:
                        previous = {}
                        for arm in ("protected-a", "protected-d"):
                            base = json.load(
                                t.extractfile(f"promotion/{arm}/{name}.json")
                            )["statistics"]
                            previous[arm] = {
                                metric: case[metric] / base[metric]
                                for metric in (
                                    "tpot_p50_ms",
                                    "ttft_p50_ms",
                                    "output_tok_s",
                                )
                            }
                    case["over_prior_256_same_gpu"] = previous
                cases[name] = case
            isolated[label] = cases
    for gpu in (6, 7):
        server = root / f"gpu{gpu}"
        args = (server / "args").read_text().splitlines()
        for option, value in (
            ("--batch-token-budget", "512"),
            ("--prefill-chunk-tokens", "512"),
            ("--prefill-burst-max-decode-rows", "8"),
            ("--prefill-burst-steady-budget", "256"),
            ("--prefill-burst-min-prompt-tokens", "512"),
        ):
            if args[args.index(option) + 1] != value:
                raise ValueError("wrong scheduler settings")
        if (server / "binary.sha256").read_bytes() != (
            baseline_dir / f"gpu{gpu}/binary.sha256"
        ).read_bytes():
            raise ValueError("unmatched server executable")
    return {
        "mixed": mixed,
        "registered_mixed_comparisons": samples,
        "isolated": isolated,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("baseline", type=Path)
    p.add_argument("history", type=Path)
    a = p.parse_args()
    print(json.dumps(check(a.root, a.baseline, a.history), indent=2))
