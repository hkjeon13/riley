"""Check recency maintenance costs and token/cache equivalence after a GPU crossover."""

import argparse
import json
from pathlib import Path
from statistics import median

from agent_workflow import fingerprint
from recency_serving import requests


def read(path):
    return json.loads(path.read_text())


def verify_cpu(root):
    shapes = {}
    for index, name in enumerate(("control", "candidate", "candidate", "control")):
        rows = read(root / f"cpu-{index}-{name}.json")
        assert len(rows) == 12
        for row in rows:
            key = (row["tokens_per_path"], row["chunk"], row["paths"])
            assert key[0] in (8192, 32768, 131072)
            assert key[1] in (32, 256) and key[2] in (1, 2)
            samples = row["publish_touch_ns"]
            assert len(samples) == 5 and all(0 < t <= total for total, t in samples)
            shapes.setdefault(key, {}).setdefault(name, []).extend(samples)
    assert len(shapes) == 12
    summary = []
    for key, timings in sorted(shapes.items()):
        entry = dict(zip(("tokens_per_path", "chunk", "paths"), key))
        for arm in ("control", "candidate"):
            data = timings[arm]
            assert len(data) == 10
            entry[arm] = {
                "publication_ms": median(r[0] for r in data) / 1e6,
                "touch_ms": median(r[1] for r in data) / 1e6,
            }
        summary.append(entry)
    return summary


def verify(root):
    artifacts = {
        name: digest
        for digest, name in (
            line.split()
            for line in (root / "artifacts.sha256").read_text().splitlines()
        )
    }
    assert artifacts["server-control"] != artifacts["server-candidate"]
    expected = list(requests())
    reference = {}
    runs = []
    counts = None
    hardware = {}
    for block in (0, 1):
        pair = root / f"prefix-recency-{block}"
        replays = read(pair / "prefix-requests.json")
        assert len(replays) == 2
        for idx, rows in enumerate(replays):
            gpu = idx + 6
            arm = "control" if idx == block else "candidate"
            path = pair / f"gpu{gpu}"
            device = (path / "device.csv").read_text()
            assert hardware.setdefault(gpu, device) == device
            assert (path / "binary.sha256").read_text().split()[0] == artifacts[
                "server-" + arm
            ]
            assert read(pair / f"allocation-{gpu}.json")["returncode"] == 0
            assert read(path / "lease-after.json")["phase"] == "idle"
            metrics = read(path / "metrics.json")["prefix_cache"]
            assert metrics["evicted_blocks"] > 0 and metrics["hit_tokens"] > 0
            assert not metrics["metrics_degraded"]
            counts = counts or metrics
            assert counts == metrics
            drained = read(path / "shutdown.json")
            assert all(
                drained[k] == 0
                for k in ("active_requests", "waiting_requests", "kv_allocated_blocks")
            )
            hits = {}
            prompts = {}
            first = []
            last = []
            for (label, request), row in zip(expected, rows, strict=True):
                assert row["case"] == label and row["input_sha256"] == fingerprint(
                    request
                )
                assert row["status"] == 200 and row["done_ns"] and row["content_ns"]
                assert (
                    row["generated_ids"] and row["prompt_ids_sha256"] and row["answer"]
                )
                identity = [
                    row[k]
                    for k in (
                        "generated_ids",
                        "prompt_ids_sha256",
                        "answer",
                        "finish",
                        "usage",
                    )
                ]
                assert reference.setdefault(label, identity) == identity
                usage = row["usage"]
                hits[label] = (usage.get("prompt_tokens_details") or {}).get(
                    "cached_tokens", 0
                )
                prompts[label] = usage["prompt_tokens"]
                first.append((row["content_ns"][0] - row["sent_ns"]) / 1e6)
                last.append((row["done_ns"] - row["sent_ns"]) / 1e6)
            assert (
                hits["prime"] < hits["repeat"]
                and hits["revisit"] < hits["warm-revisit"]
            )
            assert hits["extend"] > 0 and hits["branch"] > 0
            runs.append(
                {
                    "block": block,
                    "gpu": gpu,
                    "arm": arm,
                    "requests": len(rows),
                    "hit_tokens": hits,
                    "prompt_tokens": prompts,
                    "prefix_cache": metrics,
                    "first_content_median_ms": median(first),
                    "http_done_median_ms": median(last),
                }
            )
    assert len(set(hardware.values())) == 2
    return {"cpu": verify_cpu(root), "gpu": runs}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    print(json.dumps(verify(parser.parse_args().root), indent=2))
