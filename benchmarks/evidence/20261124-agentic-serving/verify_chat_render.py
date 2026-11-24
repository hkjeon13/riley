"""Verify render-only changes on real token IDs, CUDA serving and host timing."""

import argparse
import json
import tarfile
from collections import Counter
from hashlib import sha256
from pathlib import Path
from statistics import median

from agent_workflow import fingerprint, percentile
from chat_render_trial import cases, recorded

HERE = Path(__file__).resolve().parent


def read(p):
    return json.loads(p.read_bytes())


def expect(ok, detail):
    if not ok:
        raise ValueError(detail)


def cpu(file, requests):
    d = read(file)
    synth = cases()
    inputs = [*requests, *((k, q, None) for k, q in synth.items() for _ in range(12))]
    cells = {}
    seen = {}
    orders = {i: [] for i in range(4)}
    for arm in d["samples"]:
        p, a = arm["pair"], arm["arm"]
        orders[p].append(a)
        v = arm["observations"]
        expect(
            sorted(r["index"] for r in v) == list(range(len(inputs))), "CPU inventory"
        )
        times = {}
        for x in v:
            i = x["index"]
            k, q, h = inputs[i]
            expect(
                x["case"] == k and x["input_sha256"] == fingerprint(q),
                "different CPU request",
            )
            key = x["ids_sha256"], x["tokens"]
            expect(
                seen.setdefault(i, key) == key and (not h or h == key[0]),
                "token IDs differ",
            )
            times.setdefault(k, []).append(x)
        for k, rows in times.items():
            block = cells.setdefault(k, {})
            block.setdefault(a, []).append(median(x["normalize_us"] for x in rows))
            block.setdefault(a + "_encode", []).append(
                median(x["encode_us"] for x in rows)
            )
    expect(
        [orders[p] for p in orders]
        == [
            ["control", "candidate"],
            ["candidate", "control"],
            ["candidate", "control"],
            ["control", "candidate"],
        ],
        "wrong CPU run order",
    )
    for x in cells.values():
        ratios = [b / a for a, b in zip(x["control"], x["candidate"], strict=True)]
        x["median_pair_ratio"] = median(ratios)
        x["range_pair_ratio"] = [min(ratios), max(ratios)]
        x["control_median_us"] = median(x["control"])
        x["candidate_median_us"] = median(x["candidate"])
    return {
        "host": d["host"],
        "binaries_sha256": d["probes"],
        "calls": len(d["samples"]) * len(inputs),
        "cells": cells,
    }, seen


def gpu_rows(root, arm, gpu, bins, inputs):
    folder = root / f"gpu{gpu}-{arm}"
    lease = read(root / f"{folder.name}.lease.json")
    expect(
        lease["phase"] == "idle"
        and lease["physical_index"] == gpu
        and not lease["other_processes"]
        and lease["allocation_bytes"] == 20 * 2**30
        and lease["gpu"] in (folder / "device.csv").read_text(),
        "GPU lease",
    )
    shut = read(folder / "shutdown.json")
    expect(
        not any(
            shut[k]
            for k in ("active_requests", "waiting_requests", "kv_allocated_blocks")
        )
        and not any(shut["allocation"].values()),
        "shutdown",
    )
    expect(
        (folder / "binary.sha256").read_text().split()[0] == bins["server-" + arm],
        "served different binary",
    )
    data = read(folder / "requests.json")
    checks = data.pop("invalid")
    expect(
        set(checks) == {"bad-unselected", "unknown-name"}
        and all("error" in v for v in checks.values()),
        "400 inventory",
    )
    metrics = {}
    identities = {}
    begins = []
    ends = []
    all_done = []
    for label, q in inputs.items():
        group = data[label]
        by_phase = {
            "first": [group["cold"]],
            "warm_serial": group["warm_serial"],
            "burst": group["burst"],
        }
        expect([len(v) for v in by_phase.values()] == [1, 5, 8], "workload")
        metrics[label] = {}
        for phase, rows in by_phase.items():
            done = []
            first = []
            caches = []
            ids = []
            prompt = []
            end_types = []
            for row in rows:
                m = row["measurement"]
                t = m["content_ns"]
                calls = row["calls"]
                first_ns = t[0] if t else m.get("last_call_fragment_ns")
                expect(
                    m["input_sha256"] == fingerprint(q)
                    and m["status"] == 200
                    and first_ns
                    and m["sent_ns"] < first_ns <= m["done_ns"],
                    "incomplete HTTP outcome",
                )
                n = m["usage"]["completion_tokens"]
                cached = (m["usage"].get("prompt_tokens_details") or {}).get(
                    "cached_tokens"
                )
                expect(
                    (cached is None or 0 <= cached <= m["usage"]["prompt_tokens"])
                    and bool(m["prompt_ids_sha256"])
                    and n > 0,
                    "usage",
                )
                expect(len(m.get("generated_ids", [])) == n, "missing generated tokens")
                if label == "32-named":
                    expect(
                        len(calls) == 1
                        and calls[0]["function"]["name"] == "inventory_000"
                        and json.loads(calls[0]["function"]["arguments"])
                        == {"bin": "A3"},
                        "wrong named tool",
                    )
                identities[label, phase, len(done)] = (
                    m["prompt_ids_sha256"],
                    m["generated_ids"],
                    n,
                    m["finish"],
                    row["answer"],
                    [c["function"] for c in calls],
                )
                done.append((m["done_ns"] - m["sent_ns"]) / 1e6)
                first.append((first_ns - m["sent_ns"]) / 1e6)
                begins.append(m["sent_ns"])
                ends.append(m["done_ns"])
                caches.append(cached)
                ids.append(fingerprint(m["generated_ids"]))
                prompt.append(m["prompt_ids_sha256"])
                end_types.append(m["finish"])
            metrics[label][phase] = {
                "count": len(done),
                "done_median_ms": median(done),
                "done_p95_ms": percentile(done, 0.95),
                "first_content_or_call_median_ms": median(first),
                "cached_tokens": caches,
                "generated_sha256": ids,
                "prompt_sha256": prompt,
                "finish": end_types,
            }
            all_done.extend(done)
    expect(set(data) == set(inputs), "extra samples")
    return metrics, identities, (min(begins), max(ends)), checks, median(all_done)


def verify(root, darwin):
    expect(
        read(root / "chat-cpu-registration.json")
        == read(HERE / "chat-cpu-registration.json"),
        "changed registration",
    )
    expect(
        (root / "chat-owned-render.patch").read_bytes()
        == (HERE / "chat-owned-render.patch").read_bytes(),
        "changed source delta",
    )
    for f in ["build.exit", "exit-6", "exit-7", "exit-cpu"]:
        expect((root / f).read_text().strip() == "0", f)
    for line in (root / "inputs.sha256").read_text().splitlines():
        h, p = line.split()
        expect(sha256((root / p).read_bytes()).hexdigest() == h, p)
    with tarfile.open(HERE / "final-layout-raw.tar.gz") as ar:
        expect(
            (root / "recorded.json").read_bytes()
            == ar.extractfile("./gpu6-riley-hide/workflows.json").read(),
            "different archived inputs",
        )
    historical = list(recorded(root / "recorded.json"))
    expect(len(historical) == 60, "workflows")
    l, ids = cpu(root / "linux.json", historical)
    d, other = cpu(darwin, historical)
    expect(ids == other, "cross-host prompt mismatch")
    bins = {
        p: h
        for h, p in [
            line.split() for line in (root / "binary.sha256").read_text().splitlines()
        ]
    }
    expect(
        bins["server-control"] != bins["server-candidate"]
        and all(bins["probe-" + k] == v for k, v in l["binaries_sha256"].items()),
        "binary receipts",
    )
    inputs = {
        k: cases()[k]
        for k in ("no-tools", "4-auto", "32-auto", "32-named", "32-none", "long-result")
    }
    by_gpu = {}
    cfg = None
    for gpu in (6, 7):
        runs = {}
        refs = {}
        spans = {}
        errors = {}
        medians = {}
        for arm in ("control", "candidate"):
            argv = (root / f"gpu{gpu}-{arm}/server-arguments.txt").read_bytes()
            cfg = cfg or argv
            expect(argv == cfg, "unmatched server configuration")
            runs[arm], refs[arm], spans[arm], errors[arm], medians[arm] = gpu_rows(
                root, arm, gpu, bins, inputs
            )
        x, y = ("control", "candidate") if gpu == 6 else ("candidate", "control")
        expect(spans[x][1] < spans[y][0], "wrong physical order")
        diff = [
            list(k) for k, a in refs["control"].items() if a != refs["candidate"][k]
        ]
        cache_diff = {
            k: {
                phase: [
                    a["cached_tokens"],
                    runs["candidate"][k][phase]["cached_tokens"],
                ]
                for phase, a in groups.items()
                if Counter(a["cached_tokens"])
                != Counter(runs["candidate"][k][phase]["cached_tokens"])
            }
            for k, groups in runs["control"].items()
        }
        expect(errors["control"] == errors["candidate"], "different 400 payload")
        by_gpu[gpu] = {
            "http_ok_per_arm": len(refs["control"]),
            "invalid_400_per_arm": len(errors["control"]),
            "different_responses": diff,
            "cache_differences": {k: v for k, v in cache_diff.items() if v},
            "all_completion_median_ms": medians,
            "cells": runs,
        }
    return {
        "linux_cpu": l,
        "darwin_cpu": d,
        "physical_gpu": by_gpu,
        "successful_http": 336,
        "http_400": 8,
        "source_sha256": {
            a: (root / f"{a}.source.sha256").read_text().split()[0]
            for a in ("control", "candidate")
        },
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("darwin", type=Path)
    args = p.parse_args()
    print(json.dumps(verify(args.root, args.darwin), indent=2))
