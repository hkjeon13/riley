"""Reconcile worker queue telemetry and compare matched Coder-7B runs."""

import argparse
import json
from pathlib import Path
from statistics import median

from agent_workflow import answer_format, fixture, percentile
from join_trace import join
from verify_coder_pressure import (
    check_manifest,
    check_server,
    natural,
    qualifies,
    read,
    reconcile,
    work_latency,
)


def delays(log):
    return [
        int(line.split()[1])
        for line in log.splitlines()
        if line.startswith(b"HTTP_DEQUEUED ")
    ]


def verify_wave(raw, log, size, base, *, check_job=None):
    assert (raw["size"], raw["base"]) == (size, base)
    steps = reconcile(raw, log)
    before, after = raw["before"], raw["after"]
    cache_before, cache_after = [m["metrics"]["prefix_cache"] for m in (before, after)]
    assert not cache_before["metrics_degraded"] and not cache_after["metrics_degraded"]
    cache = {
        key: cache_after[key] - cache_before[key]
        for key in ("hit_tokens", "evicted_blocks")
    }
    assert min(cache.values()) >= 0
    lines = log[before["log_offset"] : after["log_offset"]]
    waits = delays(lines)
    a, b = [x["metrics"]["http_worker_queue"] for x in (before, after)]
    assert b["dequeued_connections"] - a["dequeued_connections"] == len(waits)
    assert b["wait_ns_total"] - a["wait_ns_total"] == sum(waits)
    assert b["wait_ns_max"] == max(a["wait_ns_max"], *waits)
    request_wait = {}
    steps["over_eight"] = 0
    for line in lines.decode().splitlines():
        if line.startswith("AGENT_STEP "):
            steps["over_eight"] += len(json.loads(line.split(maxsplit=4)[4])) > 8
        if line.startswith("HTTP_REQUEST_WAIT "):
            _, req, ns = line.split()
            assert req not in request_wait
            request_wait[req] = int(ns)
    rows = []
    for i, job in zip(range(size), raw["jobs"], strict=True):
        reader = i % 4 == 0
        seed = base + i
        work = job["work"]
        assert job["index"] == i and job["role"] == ("reader" if reader else "target")
        t0 = raw["origin_ns"] + i * 30_000_000
        assert job["scheduled_ns"] == t0 and work["end_ns"] <= raw["drained_ns"]
        assert not job["preview"] and job["decision"].get("choice") in ("off", None)
        if check_job is None:
            facts = natural(
                work,
                seed,
                2 if reader else 96,
                0 if reader else 200,
                delay=0 if reader else 0.5,
            )
        else:
            facts = check_job(job, seed, reader)
        latency = work_latency(work, t0)
        if latency.get("completed") and facts["verified_tool_calls"] == 2:
            joined = join(lines.decode(), [work], details=True)
            prefill = []
            for turn, stage in zip(work["turns"], joined, strict=True):
                usage = turn["usage"]
                cached = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
                assert stage["prefill_rows"] == usage["prompt_tokens"] - cached
                prefill.append(
                    {
                        "prompt": usage["prompt_tokens"],
                        "cached": cached,
                        "executed": stage["prefill_rows"],
                    }
                )
            latency["prefill_by_turn"] = prefill
            trace = joined[-1]
            for key in (
                "client_to_render_ms",
                "command_queue_ms",
                "admit_to_first_prefill_ms",
                "prefill_span_ms",
                "prefill_participating_gpu_ms",
                "prefill_between_steps_ms",
            ):
                latency[key] = trace[key]
            latency["http_wait_ms"] = (
                request_wait[work["turns"][-1]["response_id"]] / 1e6
            )
            latency["other_before_render_ms"] = (
                latency["client_to_render_ms"] - latency["http_wait_ms"]
            )
        rows.append(
            {
                "seed": seed,
                "role": job["role"],
                "valid": facts["valid"],
                "explanation_words": answer_format(
                    work["turns"][-1].get("answer", ""),
                    fixture(seed, 2 if reader else 96)[2],
                )["explanation_words"],
                "slo": qualifies(facts["valid"], latency),
                "tool_calls": facts["verified_tool_calls"],
                "error": work.get("error"),
                **latency,
            }
        )
    return {
        "size": size,
        "base": base,
        "duration_s": (raw["drained_ns"] - raw["origin_ns"]) / 1e9,
        "http_connections": len(waits),
        "prefix_cache": cache,
        "gpu": steps,
        "rows": rows,
    }


def summary(waves):
    rows = [r for w in waves for r in w["rows"]]
    targets = [r for r in rows if r["role"] == "target"]
    readers = [r for r in rows if r["role"] == "reader"]

    def quant(group, key):
        return percentile([r[key] for r in group if r.get(key) is not None], 0.95)

    count = sum(r["slo"] for r in rows)
    seconds = sum(w["duration_s"] for w in waves)
    return {
        "jobs": len(rows),
        "slo": count,
        "reader_slo": sum(r["slo"] for r in readers),
        "valid": sum(r["valid"] for r in rows),
        "invalid_seeds": [r["seed"] for r in rows if not r["valid"]],
        "completed": sum(bool(r.get("completed")) for r in rows),
        "strict_90_words": sum(r["explanation_words"] == 90 for r in rows),
        "warm_slo_per_gpu_hour": count * 3600 / seconds,
        "target_http_median_ms": median(
            r["http_wait_ms"] for r in targets if "http_wait_ms" in r
        ),
        "target_final_ttft_p95_ms": quant(targets, "ttft_ms"),
        "target_last_p95_ms": quant(targets, "last_ms"),
        "reader_first_p95_ms": quant(readers, "first_ms"),
        "reader_gap_p95_ms": quant(readers, "gap_p95_ms"),
        "last_p95_ms": quant(rows, "last_ms"),
        "target_medians_ms": {
            key: median(r[key] for r in targets if key in r)
            for key in (
                "before_final_submission_ms",
                "client_to_render_ms",
                "other_before_render_ms",
                "command_queue_ms",
                "admit_to_first_prefill_ms",
                "prefill_span_ms",
                "prefill_participating_gpu_ms",
                "prefill_between_steps_ms",
                "stream_ms",
                "ttft_ms",
            )
        },
        "gpu_seconds": sum(w["gpu"]["gpu_ns"] for w in waves) / 1e9,
        "target_prefill_by_turn": [
            {
                "prompt_tokens": sum(
                    r["prefill_by_turn"][i]["prompt"]
                    for r in targets
                    if "prefill_by_turn" in r
                ),
                "cached_tokens": sum(
                    r["prefill_by_turn"][i]["cached"]
                    for r in targets
                    if "prefill_by_turn" in r
                ),
                "executed_tokens": sum(
                    r["prefill_by_turn"][i]["executed"]
                    for r in targets
                    if "prefill_by_turn" in r
                ),
                "at_most_one_block_cached": sum(
                    r["prefill_by_turn"][i]["cached"] <= 16
                    for r in targets
                    if "prefill_by_turn" in r
                ),
            }
            for i in range(3)
        ],
        "prefill_tokens": sum(w["gpu"]["p"] for w in waves),
        "decode_tokens": sum(w["gpu"]["d"] for w in waves),
        "cached_tokens": sum(w["prefix_cache"]["hit_tokens"] for w in waves),
        "evicted_blocks": sum(w["prefix_cache"]["evicted_blocks"] for w in waves),
        "eight_member_steps": sum(w["gpu"]["full_steps"] for w in waves),
        "over_eight_member_steps": sum(w["gpu"]["over_eight"] for w in waves),
        "peak_members": max(w["gpu"]["peak_members"] for w in waves),
    }


def verify(root):
    check_manifest(root)
    slot_trial = (root / "engine-slots-registration.json").exists()
    conf = read(
        root
        / (
            "engine-slots-registration.json"
            if slot_trial
            else "http-worker-registration.json"
        )
    )
    schedule = read(root / "coder-pressure-registration.json")["waves"]
    sha = (root / "trace-server.sha256").read_text().split()[0]
    assert sha == "5262f4450c62de25c68d979978627ca6715b8cd4d642cb0abcbe6dacbc6a291c"
    assert (root / "build.exit").read_text().strip() == "0"
    result = {}
    common = None
    for gpu, order in conf["orders"].items():
        runs = result[gpu] = {}
        for n in order:
            prefix = "a" if slot_trial else "w"
            d = root / f"gpu{gpu}" / f"{prefix}{n}"
            check_server(d, sha, gpu)
            assert read(d / "warm.json")["ok"]
            cli = (d / "server-arguments.txt").read_text().splitlines()
            if slot_trial:
                assert cli[cli.index("--http-workers") + 1] == "16"
                i = cli.index("--max-active-sequences")
                assert cli[i + 1] == str(n)
                del cli[i : i + 2]
            else:
                if n != 8:
                    i = cli.index("--http-workers")
                    assert cli[i + 1] == str(n)
                    del cli[i : i + 2]
                assert (
                    "--http-workers" not in cli
                    and cli[cli.index("--max-active-sequences") + 1] == "8"
                )
            if common is None:
                common = cli
            assert cli == common
            log = (d / "server.log").read_bytes()
            waits = delays(log)
            lifetime = {
                "dequeued_connections": len(waits),
                "wait_ns_total": sum(waits),
                "wait_ns_max": max(waits),
            }
            assert (
                lifetime
                == read(d / "shutdown.json")["http_worker_queue"]
                == read(d / "metrics.json")["http_worker_queue"]
            )
            waves = [
                verify_wave(w, log, size, base)
                for w, (size, base) in zip(
                    read(d / "waves.json"), schedule, strict=True
                )
            ]
            assert max(w["gpu"]["peak_members"] for w in waves) <= (
                n if slot_trial else 8
            )
            runs[str(n)] = {
                "shutdown_http_queue": lifetime,
                "waves": waves,
                "high": summary(waves[2:]),
                "by_size": [summary([w]) for w in waves],
            }
        rows8, rows16 = [
            [r for wave in runs[n]["waves"][2:] for r in wave["rows"]]
            for n in ("8", "16")
        ]
        assert [r["seed"] for r in rows8] == [r["seed"] for r in rows16]
        matched = [
            (a, b)
            for a, b in zip(rows8, rows16, strict=True)
            if a.get("completed") and b.get("completed")
        ]
        runs["paired_high_outputs"] = {
            "both_completed": len(matched),
            "same_text": sum(a["text_hash"] == b["text_hash"] for a, b in matched),
            "same_completion_length": sum(
                a["tokens"] == b["tokens"] for a, b in matched
            ),
            "completion_tokens_8": sum(a["tokens"] for a, b in matched),
            "completion_tokens_16": sum(b["tokens"] for a, b in matched),
        }
        a, b = [runs[str(n)]["high"] for n in (8, 16)]
        runs["gates"] = {
            "shorter_final_ttft": b["target_final_ttft_p95_ms"]
            < a["target_final_ttft_p95_ms"],
            "preserved_valid": set(b["invalid_seeds"]) <= set(a["invalid_seeds"]),
            "more_slo": b["slo"] > a["slo"],
            "more_goodput": b["warm_slo_per_gpu_hour"] > a["warm_slo_per_gpu_hour"],
            **{
                k: b[k] <= 1.05 * a[k]
                for k in ("reader_first_p95_ms", "reader_gap_p95_ms", "last_p95_ms")
            },
        }
        if slot_trial:
            runs["gates"]["additional_slots_used"] = (
                b["over_eight_member_steps"] > 0 and a["over_eight_member_steps"] == 0
            )
        else:
            runs["gates"]["shorter_http_wait"] = (
                b["target_http_median_ms"] < a["target_http_median_ms"]
            )
    return {"binary_sha256": sha, "gpus": result}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(
        json.dumps(verify(args.root), indent=2, allow_nan=False) + "\n"
    )
