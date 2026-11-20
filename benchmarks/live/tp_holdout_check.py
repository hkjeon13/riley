#!/usr/bin/env python3
"""Compare token-exact C1, C8, and C32 responses for a two-GPU holdout."""

from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import time
from urllib.request import urlopen

from long_accuracy import complete, read_workload


OUTPUT_TOKENS = 32


def run_cell(address, model, cases, concurrency):
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        requests = [pool.submit(complete, address, model, case.prompt,
                                expected=case.expected_ids, limit=OUTPUT_TOKENS) for case in cases]
        peak = 0
        while not all(request.done() for request in requests):
            with urlopen(f"http://{address}/metrics", timeout=5) as response:
                metrics = json.load(response)
            peak = max(peak, metrics["active_requests"])
            time.sleep(0.05)
        results = [request.result() for request in requests]
    if not (1 <= peak <= concurrency) or concurrency > 1 and peak == 1:
        raise ValueError(f"C{concurrency}: active request observations do not establish the requested comparison")
    return results, peak


def check(address, model, workload_path):
    workload = read_workload(workload_path)
    cases = workload.cases
    # Include every boundary in C32. Repeat the seven shortest inputs so a
    # flood of duplicate near-8K requests does not dominate the KV pool.
    if len(cases) != 11:
        raise ValueError("the pinned TP=2 holdout must have eleven boundary cases")
    short = sorted(cases, key=lambda case: len(case.expected_ids))[:7]
    modes = {}
    reference, peak = run_cell(address, model, cases, 1)
    modes["c1"] = {"requests": len(reference), "observed_peak_active": peak}
    for concurrency, scheduled in ((8, cases), (32, [*cases, *(short * 3)])):
        results, peak = run_cell(address, model, scheduled, concurrency)
        expected = dict(zip((case.label for case in cases), reference, strict=True))
        for case, actual in zip(scheduled, results, strict=True):
            if actual != expected[case.label]:
                raise ValueError(f"C{concurrency}: output or usage differs for {case.label}")
        modes[f"c{concurrency}"] = {"requests": len(results), "exact_matches": len(results),
                                  "observed_peak_active": peak}
    return {
        "status": "passed",
        "workload_sha256": workload.sha256,
        "output_token_limit": OUTPUT_TOKENS,
        "modes": modes,
        "cases": [{"label": case.label, "prompt_tokens": len(result["prompt_tokens"]),
                   "generated_ids": len(result["tokens"]), "finish": result["finish"],
                   "response_sha256": sha256(json.dumps(result, sort_keys=True, ensure_ascii=False).encode()).hexdigest()}
                  for case, result in zip(cases, reference, strict=True)],
    }


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("address")
    parser.add_argument("model")
    parser.add_argument("workload", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summary = check(args.address, args.model, args.workload)
    with args.out.open("x", encoding="utf-8") as output:
        json.dump(summary, output, indent=2, allow_nan=False)
        output.write("\n")


if __name__ == "__main__":
    main()
