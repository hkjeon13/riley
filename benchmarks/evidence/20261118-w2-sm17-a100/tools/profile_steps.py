"""Summarize Nsight CUDA graph kernel spans bound to 32-row decode geometry."""
from collections import Counter
import itertools
import json
from pathlib import Path
import sqlite3
import statistics
import sys


def summary(values):
    if not values:
        raise ValueError("no matching trace samples")
    return {"count": len(values), "median_us": statistics.median(values) / 1000,
            "p95_us": statistics.quantiles(values, n=20, method="inclusive")[18] / 1000}


def measure(path):
    query = """
        SELECT k.globalPid, k.correlationId, k.start, k.end, k.gridX, s.value
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.correlationId = k.correlationId
        JOIN StringIds launch ON launch.id = r.nameId
        JOIN StringIds s ON s.id = k.shortName
        WHERE launch.value LIKE 'cudaGraphLaunch%'
        ORDER BY k.globalPid, k.correlationId, k.start
    """
    with sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True) as db:
        steps = []
        for (pid, _), records in itertools.groupby(db.execute(query), key=lambda r: r[:2]):
            rows = list(records)
            embed = [r[4] for r in rows if r[5] == "embedding_kernel"]
            argmax = [r[4] for r in rows if r[5] == "bf16_argmax_kernel"]
            end = rows[0][3]
            inner_gaps = 0
            for row in rows[1:]:
                inner_gaps += max(0, row[2] - end)
                end = max(end, row[3])
            steps.append((pid, rows[0][2], end, len(rows), embed == [256] and argmax == [32], inner_gaps))
    steps.sort(key=lambda r: (r[0], r[1]))
    full = [r for r in steps if r[4]]
    gaps = [
        right[1] - left[2]
        for left, right in zip(steps, steps[1:])
        if left[0] == right[0] and left[4] and right[4]
    ]
    return {
        "criteria": "CUDA graph launch, embedding gridX=256 and argmax gridX=32. Gaps only between adjacent matching graph launches. Kernel intervals exclude DMA.",
        "graph_launches": len(steps),
        "matching_kernel_counts": dict(Counter(s[3] for s in full)),
        "kernel_span": summary([s[2] - s[1] for s in full]),
        "sum_intra_step_kernel_gaps": summary([s[5] for s in full]),
        "inter_step_kernel_gap": summary(gaps),
        "overlap_between_matching_graphs": sum(g < 0 for g in gaps),
    }


if __name__ == "__main__":
    files = dict(arg.split("=", 1) for arg in sys.argv[1:])
    print(json.dumps({label: measure(Path(path)) for label, path in files.items()}, indent=2))
