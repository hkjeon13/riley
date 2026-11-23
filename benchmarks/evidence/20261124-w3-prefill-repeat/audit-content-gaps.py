"""Reconcile the same-binary replay's slower requests with client SSE gaps."""

from hashlib import sha256
import json
from math import isclose
from pathlib import Path
import tarfile


HERE = Path(__file__).resolve().parent
MODES = ("control-first", "fifo", "control-second")
MATCH = ("request_index", "prompt_tokens", "completion_tokens", "input_sha256",
         "output_sha256", "finish", "scheduled_write_ns")
LONG_GAP_S = 0.120
FOCUS_PER_CELL = 5


def capture(archive, gpu, mode, label):
    prefix = f"gpu{gpu}/{mode}"
    raw = archive.extractfile(f"{prefix}/{label}.json").read()
    receipt = json.load(archive.extractfile(f"{prefix}/receipt.json"))
    if sha256(raw).hexdigest() != receipt["results"][label]:
        raise ValueError(f"unbound request capture at {prefix}/{label}")
    return json.loads(raw)["measured"]


def intervals(row):
    gaps = row["content_gaps_s"]
    if (len(gaps) != row["completion_tokens"] - 1 or len(gaps) == 0
            or not isclose(sum(gaps), row["e2e"] - row["ttft"], abs_tol=1e-8)
            or any(gap < 0 for gap in gaps)):
        raise ValueError(f"client gap accounting differs for request {row['request_index']}")
    return gaps


def summarize(captures, comparison):
    cases = (comparison["candidate_vs_first"], comparison["candidate_vs_second"])
    if not all(cell["gates"]["timing"] and cell["gates"]["output"] for cell in cases):
        raise ValueError("focus comparisons must share a qualified client schedule and output")
    rows = []
    for first, candidate, second in zip(*captures, strict=True):
        if any(any(row[key] != first[key] for key in MATCH) for row in (candidate, second)):
            raise ValueError("request identity or generated output drifted")
        gaps = dict(zip(MODES, (intervals(row) for row in (first, candidate, second)), strict=True))
        tpot = {mode: 1000 * sum(events) / len(events) for mode, events in gaps.items()}
        counts = {mode: sum(gap >= LONG_GAP_S for gap in events) for mode, events in gaps.items()}
        short_total = {mode: 1000 * sum(gap for gap in events if gap < LONG_GAP_S)
                       for mode, events in gaps.items()}
        long_total = {mode: 1000 * sum(gap for gap in events if gap >= LONG_GAP_S)
                      for mode, events in gaps.items()}
        rows.append({
            "request_index": first["request_index"], "prompt_tokens": first["prompt_tokens"],
            "completion_tokens": first["completion_tokens"], "tpot_ms": tpot,
            "ttft_ms": {mode: 1000 * row["ttft"] for mode, row in zip(MODES, (first, candidate, second), strict=True)},
            "candidate_tpot_ms_over_slower_control": tpot["fifo"] - max(tpot["control-first"], tpot["control-second"]),
            "long_gaps": counts, "long_gap_total_ms": long_total, "short_gap_total_ms": short_total,
        })
    slower = [row for row in rows if row["candidate_tpot_ms_over_slower_control"] > 0]
    focus = sorted(slower, key=lambda row: (-row["candidate_tpot_ms_over_slower_control"], row["request_index"]))[
        :FOCUS_PER_CELL]
    return {
        "requests": len(rows), "candidate_slower_than_both_requests": len(slower),
        "candidate_long_gaps_above_both_requests": sum(row["long_gaps"]["fifo"] >
            max(row["long_gaps"]["control-first"], row["long_gaps"]["control-second"]) for row in rows),
        "long_gap_count": {mode: sum(row["long_gaps"][mode] for row in rows) for mode in MODES},
        "long_gap_total_ms": {mode: sum(row["long_gap_total_ms"][mode] for row in rows) for mode in MODES},
        "short_gap_total_ms": {mode: sum(row["short_gap_total_ms"][mode] for row in rows) for mode in MODES},
        "focus_request_indexes": [row["request_index"] for row in focus],
        "rows": rows,
    }


def audit():
    prior = json.loads((HERE / "verified.json").read_bytes())
    results = {}
    with tarfile.open(HERE / "raw-results.tar.gz", "r:gz") as archive:
        for gpu, group in prior["gpus"].items():
            for label, comparison in group["cells"].items():
                first, second = (comparison[key]["gates"] for key in ("candidate_vs_first", "candidate_vs_second"))
                if not all(gates["timing"] and gates["output"] and not gates["decode"] for gates in (first, second)):
                    continue
                captures = [capture(archive, gpu, mode, label) for mode in MODES]
                results.setdefault(gpu, {})[label] = summarize(captures, comparison)
    return {"binary_sha256": prior["binary_sha256"], "long_gap_s": LONG_GAP_S, "focus_per_cell": FOCUS_PER_CELL,
            "decode_regressions_against_both_controls": results}


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2, allow_nan=False))
