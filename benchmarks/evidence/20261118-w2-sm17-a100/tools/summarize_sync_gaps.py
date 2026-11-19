"""Measure captured CUDA synchronization and DMA between matched decode graphs."""

from argparse import ArgumentParser
from collections import Counter, defaultdict
from hashlib import sha256
import itertools
import json
from pathlib import Path
import sqlite3
from statistics import median, quantiles


GRAPH = """
SELECT k.globalPid, k.correlationId, k.deviceId, k.contextId, r.globalTid, r.start, r.end,
       k.start, k.end, k.gridX, kernel.value
FROM CUPTI_ACTIVITY_KIND_KERNEL k
JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.correlationId = k.correlationId
JOIN StringIds launch ON launch.id = r.nameId
JOIN StringIds kernel ON kernel.id = k.shortName
WHERE launch.value LIKE 'cudaGraphLaunch%'
ORDER BY k.globalPid, k.correlationId, k.start
"""

SYNC = """
SELECT r.start, r.end, api.value, r.returnValue
FROM CUPTI_ACTIVITY_KIND_RUNTIME r
JOIN StringIds api ON api.id = r.nameId
WHERE r.globalTid = ?
AND (api.value LIKE 'cudaStreamSynchronize%' OR api.value LIKE 'cudaEventSynchronize%'
     OR api.value LIKE 'cudaDeviceSynchronize%')
ORDER BY r.start
"""

DMA = """
SELECT start, end
FROM CUPTI_ACTIVITY_KIND_MEMCPY
WHERE globalPid = ? AND deviceId = ? AND contextId = ?
ORDER BY start
"""


def digest(path):
    value = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def stats(ns):
    if len(ns) < 2:
        raise ValueError("insufficient interval coverage")
    return {"count": len(ns), "median_us": median(ns) / 1000,
            "p95_us": quantiles(ns, n=20, method="inclusive")[18] / 1000}


def overlap(intervals, start, end):
    return [(max(start, left), min(end, right)) for left, right in intervals
            if left < end and start < right]


def union_ns(intervals):
    until, total = 0, 0
    for start, end in sorted(intervals):
        total += max(0, end - max(start, until))
        until = max(until, end)
    return total


def graphed_iterations(db):
    output = []
    for _, group in itertools.groupby(db.execute(GRAPH), key=lambda row: row[:2]):
        rows = list(group)
        lanes = {(row[0], row[2], row[3], row[4], row[5], row[6]) for row in rows}
        if len(lanes) != 1:
            raise ValueError("one graph spans multiple contexts or CPU submissions")
        lane, = lanes
        embedding = [row[9] for row in rows if row[10] == "embedding_kernel"]
        argmax = [row[9] for row in rows if row[10] == "bf16_argmax_kernel"]
        output.append((lane, rows[0][7], max(row[8] for row in rows),
                       embedding == [256] and argmax == [32], len(rows)))
    return sorted(output, key=lambda step: (step[0][0], step[1]))


def summarize(path):
    times = defaultdict(list)
    counters = Counter()
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro&immutable=1", uri=True) as db:
        steps = graphed_iterations(db)
        matched = [step for step in steps if step[3]]
        lanes = {step[0][:4] for step in matched}
        if len(lanes) != 1:
            raise ValueError("matched graph coverage is not from one trace lane")
        pid, device, context, thread = next(iter(lanes))
        dma = list(db.execute(DMA, (pid, device, context)))
        sync = defaultdict(list)
        for start, end, api, status in db.execute(SYNC, (thread,)):
            if status != 0 or start >= end:
                raise ValueError("unexpected failed or empty synchronization call")
            sync[api.split("_", 1)[0]].append((start, end))
        for left, right in zip(steps, steps[1:]):
            if not left[3] or not right[3] or left[0][:4] != right[0][:4]:
                continue
            start, end = left[2], right[1]
            if start > end:
                raise ValueError("unexpected overlap between matched decode graphs")
            times["kernel_to_kernel"].append(end - start)
            dma_in_gap = overlap(dma, start, end)
            times["dma_coverage"].append(union_ns(dma_in_gap))
            touched = []
            for api, calls in sync.items():
                in_gap = overlap(calls, start, end)
                touched.extend(in_gap)
                times[api].append(union_ns(in_gap))
                counters[api + "_calls_overlapping"] += len(in_gap)
                counters[api + "_gaps_with_overlap"] += bool(in_gap)
            both = union_ns([*dma_in_gap, *touched])
            times["dma_or_synchronization_coverage"].append(both)
            times["unattributed_to_dma_or_synchronization"].append(end - start - both)
            launch_start, launch_end = right[0][4:]
            launch_in_gap = overlap([(launch_start, launch_end)], start, end)
            times["next_graph_launch_api_coverage"].append(union_ns(launch_in_gap))
            times["next_kernel_minus_launch_return"].append(end - launch_end)
            times["prior_kernel_end_to_next_launch_call"].append(launch_start - start)
            attributed = union_ns([*dma_in_gap, *touched, *launch_in_gap])
            times["unattributed_to_dma_synchronization_or_launch_api"].append(end - start - attributed)

            # The next launch call starts before its first captured kernel.
            # Report any StreamSynchronize that follows EventSynchronize made
            # entirely after the previous graph's last kernel and before launch.
            events = [stop for begin, stop in sync["cudaEventSynchronize"] if start <= begin <= stop <= launch_start]
            if events:
                streams = [(begin, stop) for begin, stop in sync["cudaStreamSynchronize"]
                           if events[-1] <= begin <= stop <= launch_start]
                if streams:
                    counters["gaps_with_stream_sync_after_event_sync"] += 1
                    times["stream_sync_after_event_sync_when_present"].append(union_ns(streams))
    return {
        "sqlite_sha256": digest(path),
        "sqlite_bytes": path.stat().st_size,
        "graph_launches": len(steps),
        "matched_graphs": len(matched),
        "matched_kernel_counts": dict(Counter(step[4] for step in matched)),
        "adjacent_matched_gaps": len(times["kernel_to_kernel"]),
        "intervals": {key: stats(values) for key, values in times.items()},
        "counts": dict(counters),
        "scope": "CPU calls and DMA are clipped to kernel-to-kernel gaps in one profiled graph lane."
                 " Ordering alone does not identify which fence a stream call waited for.",
    }


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("sqlite", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.sqlite), indent=2, allow_nan=False))
