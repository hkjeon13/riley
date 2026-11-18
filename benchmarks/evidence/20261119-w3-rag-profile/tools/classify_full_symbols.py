"""Classify the same measured GPU windows using full Nsight symbols."""

from argparse import ArgumentParser
from bisect import bisect_left
from collections import defaultdict
from hashlib import file_digest
import json
from pathlib import Path
import sqlite3
from statistics import median

from profile_summary import anchors, clock_offset, name_group, union_ns


def category(symbol):
    if "riley_fi::" in symbol:
        if any(name in symbol for name in ("strided_paged_kernel", "merge_parts_kernel")):
            return "attention"
        if "plan_kernel" in symbol:
            return "attention_metadata"
    if "kv_write" in symbol:
        return "qkv_bias_rope_kv_write"
    return name_group(symbol)


def classify(database, arm):
    receipt = json.loads((arm / "receipt.json").read_text())
    server_pid = int((arm / "server.pid").read_text())
    with database.open("rb") as file:
        if file_digest(file, "sha256").hexdigest() != receipt["sqlite_sha256"]:
            raise ValueError("SQLite input differs from the captured trace")
    with sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True) as db:
        marks = anchors(db)
        symbols = dict(db.execute("SELECT id,value FROM StringIds"))
        query = """
            SELECT globalPid,start,end,demangledName
            FROM CUPTI_ACTIVITY_KIND_KERNEL
            WHERE end > start AND (globalPid = ? OR ((globalPid >> 24) & 16777215) = ?)
            ORDER BY start,end
        """
        names = {}
        owners = set()
        spans = []
        for owner, start, end, symbol_id in db.execute(query, (server_pid, server_pid)):
            owners.add(owner)
            if symbol_id not in names:
                names[symbol_id] = category(symbols[symbol_id])
            spans.append((start, end, names[symbol_id], symbol_id))
        if len(owners) != 1:
            raise ValueError(f"expected a single GPU trace owner, got {len(owners)}")
    starts = [span[0] for span in spans]
    longest = max(end - start for start, end, _, _ in spans)
    results = {}
    for name in receipt["results"]:
        workload = json.loads((arm / f"{name}.json").read_text())
        legacy = json.loads((arm / f"{name}.profile.json").read_text())
        offset, _, _ = clock_offset(marks, workload["anchors"])
        rows = []
        for request, prior in zip(workload["measured"], legacy["measurements"], strict=True):
            if request["request_index"] != prior["request_index"]:
                raise ValueError(f"{name}: mismatched request order")
            begin = round(request["sent_at"] * 1e9) + offset
            first = round((request["start"] + request["ttft"]) * 1e9) + offset
            left, right = bisect_left(starts, begin - longest), bisect_left(starts, first)
            classified = defaultdict(list)
            for start, end, group, _ in spans[left:right]:
                if end > begin:
                    classified[group].append((max(begin, start), min(first, end)))
            all_intervals = (interval for intervals in classified.values() for interval in intervals)
            if abs(union_ns(all_intervals) - prior["gpu_kernel_union_ms"] * 1e6) > 1:
                raise ValueError(f"{name}: classified kernel window differs from initial measurement")
            rows.append({group: union_ns(intervals) / 1e6 for group, intervals in classified.items()})
        groups = sorted({group for row in rows for group in row})
        results[name] = {"requests": len(rows),
                         "kernel_category_ms_p50": {group: median(row.get(group, 0) for row in rows)
                                                    for group in groups}}
    sample_counts = defaultdict(int)
    for _, _, _, symbol_id in spans:
        sample_counts[symbol_id] += 1
    return {"scope": "Kernel union by reviewed full symbol within each request window",
            "sqlite_sha256": receipt["sqlite_sha256"], "server_pid": server_pid,
            "cells": results,
            "symbols": [{"name": symbols[symbol_id], "category": group, "launches": sample_counts[symbol_id]}
                        for symbol_id, group in sorted(names.items())]}


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    parser.add_argument("arm", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    output = classify(args.database, args.arm)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, allow_nan=False)
        handle.write("\n")
