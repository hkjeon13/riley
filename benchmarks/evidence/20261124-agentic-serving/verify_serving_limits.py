"""Check data coverage, load outcomes, recovered capacity and raw streaming receipts."""

import argparse
import collections
import json
import math
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def p95(values):
    return sorted(values)[math.ceil(len(values) * 0.95) - 1] if values else None


def startup(folder):
    ready = folder / "ready.json"
    if ready.is_file() and ready.stat().st_size and read(ready).get("ready"):
        if read(folder / "lease-after.json")["phase"] != "idle":
            raise ValueError(f"GPU was not returned: {folder}")
        return None
    cause = (folder / "server.log").read_text().splitlines()
    if not cause or not cause[0].startswith("riley:"):
        raise ValueError(f"no verified startup failure: {folder}")
    return cause[0]


def shapes(root):
    manifest = read(root / "public-shapes.json")["rows"]
    results = {}
    for name in ("baseline", "extended", "page32"):
        folder = root / name
        failure = startup(folder)
        if failure:
            results[name] = {"startup_error": failure, "submitted": 0}
            continue
        rows = read(folder / "shapes.json")
        if [r["request_sha256"] for r in rows] != [
            r["request_sha256"] for r in manifest
        ]:
            raise ValueError("shape cohort changed")
        result = {}
        for label in sorted({r["set"] for r in rows}):
            group = [r for r in rows if r["set"] == label]
            ok = [r for r in group if r.get("complete") and r["http_status"] == 200]
            result[label] = {
                "submitted": sum(r["attempted"] for r in group),
                "status": dict(
                    collections.Counter(str(r["http_status"]) for r in group)
                ),
                "completed": len(ok),
                "exact_input": sum(r["prompt_exact"] for r in ok),
                "exact_output": sum(r["output_exact"] for r in ok),
                "largest_input_accepted": max(
                    (r["usage"]["prompt_tokens"] for r in ok), default=None
                ),
                "stop_reasons": dict(collections.Counter(r["finish"] for r in ok)),
                "failures": [
                    {"index": r["index"], "error": r.get("error")}
                    for r in group
                    if r not in ok
                ],
                "completion_mismatches": [
                    {
                        "index": r["index"],
                        "requested": r["next_completion_tokens"],
                        "actual": r["usage"]["completion_tokens"],
                    }
                    for r in ok
                    if not r["output_exact"]
                ],
            }
        results[name] = result
    return results


def soak(root):
    if startup(root) is not None:
        raise ValueError("soak did not start")
    rows = [
        json.loads(line) for line in (root / "requests.jsonl").read_text().splitlines()
    ]
    series = read(root / "time-series.json")
    summary = read(root / "summary.json")
    reg = read(root.parent / "soak-registration.json")
    schedule = []
    offset = 0.0
    while offset < reg["seconds"]:
        schedule.append(offset)
        phase = offset % reg["period_s"]
        rate = (
            reg["peak_workflows_per_s"]
            if 120 <= phase < 120 + reg["peak_duration_s"]
            else reg["base_workflows_per_s"]
        )
        offset += 1 / rate
    if sorted((r["index"], r["scheduled_s"]) for r in rows) != list(
        enumerate(schedule)
    ):
        raise ValueError("submitted indices or times do not match the registered load")
    successful = [r for r in rows if r["correct"]]
    good = [
        r
        for r in successful
        if r["workflow_first_ms"] <= 10000
        and r["workflow_last_ms"] <= 18000
        and r["gap_p95_ms"] <= 150
    ]
    drops = sum(r["cancelled_at_client"] for r in rows)
    statuses = collections.Counter(
        str(t["status"]) if t.get("status") is not None else "null"
        for r in rows
        for t in r["requests"]
    )
    if (len(rows), len(successful), len(good), drops, dict(statuses)) != (
        summary["submitted"],
        summary["correct"],
        summary["slo_completed"],
        summary["client_drops"],
        summary["status_counts"],
    ):
        raise ValueError("summary diverges from individual requests")
    by_kind = {}
    for kind in ("normal", "slow", "cancel"):
        group = [r for r in rows if r["kind"] == kind]
        by_kind[kind] = {
            "submitted": len(group),
            "correct": sum(r["correct"] for r in group),
            "slo_completed": sum(r["kind"] == kind for r in good),
        }
    windows = {}
    for window, start, end in [
        ("start", 0, reg["seconds"] / 3),
        ("end", 2 * reg["seconds"] / 3, reg["seconds"]),
    ]:
        group = [r for r in successful if start <= r["scheduled_s"] < end]
        windows[window] = {
            k: p95([r[k] for r in group])
            for k in ("workflow_first_ms", "workflow_last_ms", "gap_p95_ms")
        }
    if windows != summary["windows"] or series[-1]["elapsed_s"] < reg["seconds"] - 2:
        raise ValueError("soak window coverage differs")
    zero = read(root / "shutdown.json")
    return {
        "submitted": len(rows),
        "correct": len(successful),
        "slo_completed": len(good),
        "good_workflows_per_gpu_hour": len(good) * 3600 / summary["measurement_s"],
        "duration_s": summary["measurement_s"],
        "client_drops": drops,
        "http_status": dict(statuses),
        "by_kind": by_kind,
        "failure_reasons": dict(
            collections.Counter(r["reason"] for r in rows if not r["correct"])
        ),
        "windows": windows,
        "dispatch_lag_p95_ms": p95([r["dispatch_lag_ms"] for r in rows]),
        "samples": len(series),
        "device_live_bytes_sampled_range": [
            min(r["allocation"]["device_live_bytes"] for r in series),
            max(r["allocation"]["device_live_bytes"] for r in series),
        ],
        "counter_changes": {
            k: series[-1]["counters"][k] - v for k, v in series[0]["counters"].items()
        },
        "highest_waiting": max(r["waiting_requests"] for r in series),
        "last_metrics": series[-1],
        "shutdown_resources_zero": all(
            zero[k] == 0
            for k in ("active_requests", "waiting_requests", "kv_allocated_blocks")
        )
        and not any(zero["allocation"].values()),
    }


def crossover(root):
    for gpu in (6, 7):
        if startup(root / f"gpu{gpu}") is not None:
            raise ValueError("replica did not start")
    results = []
    for row in read(root / "cache-queue-crossover.json"):
        a, b = row["warm"], row["cold"]
        if a["input_sha256"] != b["input_sha256"] or not all(
            r.get("done_ns") and r["status"] == 200 for r in [a, b, *row["peers"]]
        ):
            raise ValueError("incomplete matched probe")
        view = {}
        for key, r in (("warm", a), ("cold", b)):
            view[key] = {
                "first_ms": (r["content_ns"][0] - r["sent_ns"]) / 1e6,
                "cached": (r["usage"].get("prompt_tokens_details") or {}).get("cached_tokens", 0),
            }
        view.update(
            repeat=row["repeat"],
            streams=row["streams"],
            peers_active_at_offer=sum(
                r["content_ns"][0] <= a["sent_ns"] < r["done_ns"] for r in row["peers"]
            ),
        )
        results.append(view)
    if [(r["repeat"], r["streams"]) for r in results] != [
        (i, n) for i in (0, 1) for n in (0, 4, 8)
    ]:
        raise ValueError("crossover coverage differs")
    waves = []
    for wave in read(root / "overload.json"):
        rows = wave["requests"]
        if sorted(r["index"] for r in rows) != list(range(wave["submitted"])):
            raise ValueError("missing offered arrivals")
        peer = wave["protected"]
        waves.append(
            {
                "submitted": len(rows),
                "statuses": dict(
                    collections.Counter(str(r.get("status")) for r in rows)
                ),
                "sse_complete": sum(bool(r.get("done_ns")) for r in rows),
                "peer_correct": peer["ok"],
                "peer_first_ms": peer.get("first_final_ms"),
                "peer_last_ms": peer.get("last_final_ms"),
                "post_wave_ok": bool(wave["post_wave"].get("done_ns")),
            }
        )
    if [r["submitted"] for r in waves] != [16, 64, 160]:
        raise ValueError("missing saturation burst")
    return {"probes": results, "waves": waves}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["shapes", "soak", "crossover"])
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            {"shapes": shapes, "soak": soak, "crossover": crossover}[args.kind](
                args.root
            ),
            indent=2,
        )
    )
