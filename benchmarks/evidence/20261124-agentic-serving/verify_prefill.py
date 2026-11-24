"""Validate every packaged ABBA arm and report the paired workflow changes."""

import json
import tarfile
from argparse import ArgumentParser
from pathlib import Path
from statistics import median


def report(source):
    result = {"gpu": 7, "histories": {}}
    arms = ["control-a", "wide-b", "wide-c", "control-d"]
    with tarfile.open(source) as tar:

        def read(arm, file):
            return json.load(tar.extractfile(f"prefill/{arm}/{file}"))

        leases = [read(arm, "lease-after.json") for arm in arms]
        if not all(
            lease["phase"] == "idle"
            and not lease["other_processes"]
            and lease["pid"] == leases[0]["pid"]
            and lease["allocation_bytes"] == 20 * 2**30
            and lease["start_ticks"] == leases[0]["start_ticks"]
            for lease in leases
        ):
            raise ValueError("GPU lease differed at arm boundaries")
        binaries = [
            tar.extractfile(f"prefill/{arm}/binary.sha256").read() for arm in arms
        ]
        if len(set(binaries)) != 1:
            raise ValueError("binary changed across arms")
        for history in (0, 200):
            workloads = {arm: read(arm, f"workflows-{history}.json") for arm in arms}
            cell = {"arms": {}, "pairs": []}
            for arm, workflows in workloads.items():
                if len(workflows) != 5 or not all(r["ok"] for r in workflows):
                    raise ValueError(f"incomplete workflows {arm}/{history}")
                if [r["fixture_sha256"] for r in workflows] != [
                    r["fixture_sha256"] for r in workloads[arms[0]]
                ]:
                    raise ValueError("different fixture")
                for r, control in zip(workflows, workloads[arms[0]], strict=True):
                    if r["turns"][-1]["answer"] != control["turns"][-1]["answer"] or [
                        t.get("call", {}).get("function") for t in r["turns"]
                    ] != [t.get("call", {}).get("function") for t in control["turns"]]:
                        raise ValueError(
                            f"tool or final response changed in {arm}/{history}"
                        )
                reader, new = workflows[-2:]
                stream = reader["turns"][-1]["content_ns"]
                if not stream[0] <= new["turns"][-1]["sent_ns"] < stream[-1]:
                    raise ValueError(
                        "long request was not submitted during final answer"
                    )
                shut = read(arm, "shutdown.json")
                if any(
                    shut[key]
                    for key in (
                        "active_requests",
                        "waiting_requests",
                        "kv_allocated_blocks",
                    )
                ) or any(shut["allocation"].values()):
                    raise ValueError("unreleased resources")
                before = read(arm, f"before-{history}.json")["batch_shapes"]["buckets"]
                after = read(arm, f"after-{history}.json")["batch_shapes"]["buckets"]
                wide_steps = sum(
                    b["hit_count"] - a["hit_count"]
                    for a, b in zip(before, after, strict=True)
                    if b["dense_rows"] > 256
                )
                cell["arms"][arm] = {
                    "workflow_first_p95_ms": max(
                        r["first_final_ms"] for r in workflows
                    ),
                    "workflow_complete_p95_ms": max(
                        r["last_final_ms"] for r in workflows
                    ),
                    "final_gap_tail_ms": max(r["final_gap_p95_ms"] for r in workflows),
                    "tpot_p50_ms": median(r["final_tpot_ms"] for r in workflows),
                    "wide_steps": wide_steps,
                }
            for base, candidate in ((arms[0], arms[1]), (arms[3], arms[2])):
                a, b = [cell["arms"][name] for name in (base, candidate)]
                cell["pairs"].append(
                    {
                        k: round((b[k] / a[k] - 1) * 100, 2)
                        for k in a
                        if k.endswith("_ms")
                    }
                )
            result["histories"][history] = cell
    return result


if __name__ == "__main__":
    p = ArgumentParser(description=__doc__)
    p.add_argument("raw", type=Path)
    print(json.dumps(report(p.parse_args().raw), indent=2))
