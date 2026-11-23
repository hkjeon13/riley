"""Audit matched switch-off controls and estimate causal pair work from the prior trace."""

from contextlib import ExitStack
import json
from math import isclose
from pathlib import Path
import runpy
from statistics import median
import tarfile
import tempfile

from check_opportunities import placements, summarize


HERE = Path(__file__).resolve().parent
BROAD = HERE.parent / "20261124-w3-rag-prefill-order"
METRICS = ("ttft_p50_ms", "ttft_p95_ms", "tpot_p50_ms")
REQUEST_FIELDS = ("request_index", "input_sha256", "prompt_tokens", "completion_tokens",
                  "finish", "output_sha256", "scheduled_write_ns")


def measured_step_times(steps, prompt_lengths):
    """Group committed GPU times by decode rows and the prefill cursor."""
    widths = (1, 8)
    intervals = ((0, 512), (512, 1536), (1536, 2560))
    groups = {width: {lower: [] for lower, _ in intervals} for width in widths}
    for step in steps:
        if step.rows != 1024 or step.decode not in groups or len(step.prefill) != 1:
            continue
        request, rows, target = step.prefill[0]
        if request not in prompt_lengths:
            continue
        cursor = target - rows
        for lower, upper in intervals:
            if lower <= cursor < upper:
                groups[step.decode][lower].append(step.gpu / 1e6)
                break
    return {
        width: [
            {"prefix_tokens_min": lower, "prefix_tokens_max": upper - 1, "steps": len(values),
             "gpu_ms_min": min(values), "gpu_ms_median": median(values), "gpu_ms_max": max(values)}
            for lower, upper in intervals
            if (values := group[lower])
        ]
        for width, group in groups.items()
    }


def estimate_pairs(steps, submitted, prompt_lengths, activation):
    details = []
    for step, head, budget, original_past, eligible in placements(steps, submitted, prompt_lengths, activation):
        candidate, candidate_past = eligible[0]
        # For Q new causal queries at position P, the pair count is Q*P + Q*(Q+1)/2.
        original_pairs = budget * original_past + budget * (budget + 1) // 2
        candidate_pairs = original_pairs + budget * (candidate_past - original_past)
        details.append({
            "step_id": step.id, "decode_rows": step.decode, "prefill_rows": budget,
            "original_request": head, "candidate_request": candidate, "eligible_requests": len(eligible),
            "original_prefix_tokens": original_past, "candidate_prefix_tokens": candidate_past,
            "causal_pairs_delta": candidate_pairs - original_pairs,
            "causal_pairs_delta_pct": 100 * (candidate_pairs / original_pairs - 1),
            "any_eligible_nonincreasing": any(past <= original_past for _, past in eligible),
        })
    return {
        "opportunities": len(details),
        "increased_pairs": sum(row["causal_pairs_delta"] > 0 for row in details),
        "same_pairs": sum(row["causal_pairs_delta"] == 0 for row in details),
        "decreased_pairs": sum(row["causal_pairs_delta"] < 0 for row in details),
        "any_eligible_nonincreasing": sum(row["any_eligible_nonincreasing"] for row in details),
        "measured_1024_row_steps_by_decode_and_prefix": measured_step_times(steps, prompt_lengths),
        "details": details,
    }


def matched_rows(left, right):
    pairs = list(zip(left["measured"], right["measured"], strict=True))
    if any(any(a[key] != b[key] for key in REQUEST_FIELDS) for a, b in pairs):
        raise ValueError("unmatched requests, schedules or outputs")
    return pairs


def per_token_ms(row):
    span = row["e2e"] - row["ttft"]
    if span < 0 or not isclose(span, sum(row["content_gaps_s"]), abs_tol=1e-8):
        raise ValueError("client content gaps do not reconstruct decode span")
    return 1000 * span / max(row["completion_tokens"] - 1, 1)


def check_statistics(capture, percentile):
    rows = capture["measured"]
    ttft = [row["ttft"] * 1000 for row in rows]
    decode = [per_token_ms(row) for row in rows]
    computed = (percentile(ttft, .50), percentile(ttft, .95), percentile(decode, .50))
    for metric, expected in zip(METRICS, computed, strict=True):
        if not isclose(capture["statistics"][metric], expected, abs_tol=1e-6):
            raise ValueError(f"raw requests do not reconstruct {metric}")
    return decode


def timing_audit(broad, narrow, report, verifier, limit):
    read, percentile = verifier["read"], verifier["percentile"]
    results = {}
    for gpu, baseline in report["gpus"].items():
        broad_arm, narrow_arm = (root / f"gpu{gpu}" / "control" for root in (broad, narrow))
        plans = [read(arm / "plan.json") for arm in (broad_arm, narrow_arm)]
        if any(plans[0][key] != plans[1][key] for key in ("server", "cells", "server_cpu_affinity")):
            raise ValueError(f"GPU {gpu} switch-off settings or CPU placement differ")
        results[gpu] = {}
        for label, comparison in baseline["cells"].items():
            a, b = [read(arm / f"{label}.json") for arm in (broad_arm, narrow_arm)]
            pairs = matched_rows(a, b)
            write_error = max(
                percentile([abs(x["write_error_ms"]) for x in a["measured"]], .95),
                percentile([abs(x["write_error_ms"]) for x in b["measured"]], .95),
                percentile([abs(x["write_error_ms"] - y["write_error_ms"]) for x, y in pairs], .95),
            )
            candidate = read(narrow_arm.parent / "fifo" / f"{label}.json")
            matched_rows(b, candidate)
            check_statistics(a, percentile)
            control_decode = check_statistics(b, percentile)
            candidate_decode = check_statistics(candidate, percentile)
            deltas_ms = [new - old for old, new in zip(control_decode, candidate_decode, strict=True)]
            results[gpu][label] = {
                "matched": len(pairs), "control_to_control_write_error_p95_ms": write_error,
                "control_to_control_valid": write_error <= limit,
                "control_to_control_delta_pct": {
                    metric: 100 * (b["statistics"][metric] / a["statistics"][metric] - 1) for metric in METRICS
                },
                "candidate_comparison_timing_valid": comparison["gates"]["timing"],
                "candidate_decode_slower_requests": sum(delta > 0 for delta in deltas_ms),
                "candidate_decode_slower_by_more_than_1ms_per_token": sum(delta > 1 for delta in deltas_ms),
            }
    return results


def main():
    verifier = runpy.run_path(str(BROAD / "verify.py"))
    read = verifier["read"]
    scope = summarize(estimate_pairs)
    scope["causal_pair_estimate_by_schedule"] = scope.pop("placements_by_decode_rows")
    registrations = [directory / "registration.json" for directory in (BROAD, HERE)]
    with ExitStack() as artifacts:
        roots, reports = [], []
        for directory, registration in zip((BROAD, HERE), registrations, strict=True):
            root = Path(artifacts.enter_context(tempfile.TemporaryDirectory()))
            with tarfile.open(directory / "raw-results.tar.gz", "r:gz") as archive:
                archive.extractall(root, filter="data")
            report = verifier["verify"](root, registration)
            replayed = json.dumps(report, indent=2, allow_nan=False) + "\n"
            if replayed != (directory / "verified.json").read_text():
                raise ValueError(f"stored result does not replay in {directory.name}")
            roots.append(root)
            reports.append(report)
        if (reports[0]["base_revision"] != reports[1]["base_revision"]
                or reports[0]["binary_sha256"] == reports[1]["binary_sha256"]
                or any(reports[0]["gpus"][gpu]["order"] != reports[1]["gpus"][gpu]["order"]
                       for gpu in reports[1]["gpus"])):
            raise ValueError("expected same base, arm order, and distinct diagnostic binaries")
        limits = {read(file)["write_error_p95_limit_ms"] for file in registrations}
        limit, = limits
        timings = timing_audit(*roots, reports[1], verifier, limit)
    return {
        "base_revision": reports[0]["base_revision"],
        "broad_binary_sha256": reports[0]["binary_sha256"],
        "narrow_binary_sha256": reports[1]["binary_sha256"],
        "same_binary_control_repeat": False,
        "control_and_candidate_timings": timings,
        "frozen_trace": scope,
    }


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, allow_nan=False))
