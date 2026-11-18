"""Join measured TTFT windows and CUDA kernels on a validated Nsight clock."""
import argparse
from bisect import bisect_left
from collections import defaultdict
import json
from pathlib import Path
import re
import sqlite3
import statistics

from profile_workload import save


def union_ns(intervals):
    elapsed = end = 0
    for start, stop in sorted(intervals):
        if stop > end:
            elapsed += stop - max(start, end)
            end = stop
    return elapsed


PATTERNS = (
    ("attention", re.compile(r"attention|(?:paged|gqa).*(?:split|merge)|fmha|flash_fwd|softmax", re.I)),
    ("gemm", re.compile(r"gemm|matmul|(?:ampere|sm80).*bf16|s16816|cutlass|xmma", re.I)),
)


def name_group(name):
    for group, expression in PATTERNS:
        if expression.search(name):
            return group
    return "other"


def anchors(db):
    columns = {row[1] for row in db.execute("PRAGMA table_info(NVTX_EVENTS)")}
    if "start" not in columns or not {"text", "textId"} & columns:
        raise ValueError("Nsight export lacks NVTX clock marks")
    if {"text", "textId"} <= columns:
        query = """
            SELECT n.start, coalesce(n.text, s.value) FROM NVTX_EVENTS n
            LEFT JOIN StringIds s ON s.id = n.textId
        """
    elif "text" in columns:
        query = "SELECT start,text FROM NVTX_EVENTS"
    else:
        query = "SELECT n.start,s.value FROM NVTX_EVENTS n JOIN StringIds s ON s.id = n.textId"
    result = {}
    for start, tag in db.execute(query):
        if not tag or not tag.startswith("RILEY_W3_CLOCK_"):
            continue
        if tag in result:
            raise ValueError(f"duplicated NVTX anchor {tag}")
        result[tag] = start
    return result


def clock_offset(markers, stamps):
    offsets, spans = [], []
    for stamp in stamps:
        if stamp["name"] not in markers:
            raise ValueError(f"missing NVTX anchor {stamp['name']}")
        low, high = stamp["before_ns"], stamp["after_ns"]
        if high < low:
            raise ValueError("invalid CPU clock bracket")
        spans.append(high - low)
        offsets.append(markers[stamp["name"]] - (low + high) // 2)
    if len(offsets) != 2 or max(spans) > 1_000_000 or abs(offsets[1] - offsets[0]) > 1_000_000:
        raise ValueError(f"uncertain trace clock alignment: brackets={spans} offsets={offsets}")
    return (offsets[0] + offsets[1]) // 2, spans, abs(offsets[1] - offsets[0])


def summarize(path, run, server_pid):
    with sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True) as db:
        marks = anchors(db)
        records = db.execute("""
            SELECT k.globalPid, k.start, k.end, s.value
            FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id = k.shortName
            ORDER BY k.start,k.end
        """)
        # Nsight encodes the PID in bits 24..47 of globalTid/globalPid for
        # this export format. Refuse ambiguous trace owners.
        kernels = defaultdict(list)
        for process, start, end, name in records:
            if end <= start:
                continue
            if process == server_pid or ((process >> 24) & 0xffffff) == server_pid:
                kernels[process].append((start, end, name_group(name), name))
    if len(kernels) != 1:
        raise ValueError(f"cannot isolate the Riley CUDA process: {list(kernels)}")
    spans = next(iter(kernels.values()))
    starts = [span[0] for span in spans]
    longest = max(end - start for start, end, _, _ in spans)
    offset, anchor_sizes, alignment = clock_offset(marks, run["anchors"])
    measurements = []
    for row in run["measured"]:
        begin = round(row["sent_at"] * 1e9) + offset
        first = round((row["start"] + row["ttft"]) * 1e9) + offset
        if not begin < first:
            raise ValueError("first visible token must follow request send")
        lo = bisect_left(starts, begin - longest)
        hi = bisect_left(starts, first)
        intervals = defaultdict(list)
        by_name = defaultdict(list)
        for i in range(lo, hi):
            start, end, group, name = spans[i]
            if end <= begin:
                continue
            clipped = (max(start, begin), min(end, first))
            intervals[group].append(clipped)
            by_name[name].append(clipped)
        busy = union_ns(interval for group in intervals.values() for interval in group)
        if not busy:
            raise ValueError(f"no server GPU time for request {row['request_index']}")
        milliseconds = lambda elapsed: elapsed / 1e6
        measurements.append({
            "request_index": row["request_index"], "prompt_tokens": row["prompt_tokens"],
            "sent_to_first_ms": milliseconds(first - begin),
            "gpu_kernel_union_ms": milliseconds(busy),
            "time_outside_gpu_kernels_ms": milliseconds(first - begin - busy),
            "gpu_categories_union_ms": {k: milliseconds(union_ns(v)) for k, v in sorted(intervals.items())},
            "top_kernels_by_union_ms": dict(sorted(
                ((name, milliseconds(union_ns(periods))) for name, periods in by_name.items()),
                key=lambda pair: pair[1], reverse=True,
            )[:10]),
        })
    return {"scope": "GPU time inside request windows; concurrent requests share these intervals",
            "clock_bracket_ns": anchor_sizes, "clock_offset_change_ns": alignment,
            "server_pid": server_pid, "requests": len(measurements),
            "gpu_busy_share_median": statistics.median(
                row["gpu_kernel_union_ms"] / row["sent_to_first_ms"] for row in measurements),
            "measurements": measurements}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite", type=Path)
    parser.add_argument("workload", type=Path)
    parser.add_argument("server_pid", type=int)
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    result = summarize(args.sqlite, json.loads(args.workload.read_bytes()), args.server_pid)
    save(args.result, result)
