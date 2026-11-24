"""Validate every arrival, user-visible outcome, GPU step and throughput denominator."""

import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
from statistics import median

from join_trace import join

from agent_workflow import answer_format, fingerprint, fixture, percentile
from verify_model_decode import natural, read
from verify_coder_wait import check_preview


def check_manifest(root):
    assert (root / "exits.log").read_text().split() == ["0", "0"]
    for line in (root / "inputs.sha256").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name


def check_server(d, sha, gpu):
    assert (d / "binary.sha256").read_text().split()[0] == sha
    lease = read(d / "lease-after.json")
    assert lease["phase"] == "idle" and lease["physical_index"] == int(gpu)
    for name in ("metrics", "shutdown"):
        m = read(d / f"{name}.json")
        assert m["active_requests"] == m["waiting_requests"] == 0
    shutdown = read(d / "shutdown.json")
    assert shutdown["kv_allocated_blocks"] == 0 and not any(
        shutdown["allocation"].values()
    )


def work_latency(work, scheduled):
    final = work["turns"][-1]
    assert scheduled <= work["start_ns"] <= work["end_ns"]
    if final["turn"] != 2 or not final.get("content_ns"):
        return {}
    events = final["content_ns"]
    gaps = [(b - a) / 1e6 for a, b in itertools.pairwise(events)]
    gap = percentile(gaps, 0.95)
    return {
        "before_final_submission_ms": (final["sent_ns"] - scheduled) / 1e6,
        "stream_ms": (events[-1] - events[0]) / 1e6,
        "first_ms": (events[0] - scheduled) / 1e6,
        "last_ms": (events[-1] - scheduled) / 1e6,
        "gap_p95_ms": gap,
        "ttft_ms": (events[0] - final["sent_ns"]) / 1e6,
        "tokens": final.get("usage", {}).get("completion_tokens"),
        "text_hash": fingerprint(final.get("answer", "")),
        "completed": bool(final.get("done_ns") and final.get("finish") == "stop"),
    }


def qualifies(valid, t):
    return bool(
        valid
        and t.get("completed")
        and t.get("gap_p95_ms") is not None
        and t["first_ms"] <= 5000
        and t["last_ms"] <= 8000
        and t["gap_p95_ms"] <= 80
    )


def reconcile(w, log):
    begin, end = [w[k]["log_offset"] for k in ("before", "after")]
    assert 0 <= begin < end <= len(log)
    counts = Counter()
    for line in log[begin:end].splitlines():
        if not line.startswith(b"AGENT_STEP "):
            continue
        _, _, _, ns, items = line.split(maxsplit=4)
        members = json.loads(items)
        counts["iterations"] += 1
        counts["gpu_ns"] += int(ns)
        counts["peak_members"] = max(counts["peak_members"], len(members))
        counts["full_steps"] += len(members) == 8
        for _, kind, rows, *_ in members:
            counts[kind] += rows
    for field, key in (
        ("iterations", "iterations"),
        ("gpu_execution_ns", "gpu_ns"),
        ("prefill_tokens", "p"),
        ("decode_tokens", "d"),
    ):
        sums = []
        for mark in ("before", "after"):
            m = w[mark]["metrics"]
            assert m["active_requests"] == m["waiting_requests"] == 0
            assert not m["batch_shapes"]["metrics_degraded"]
            sums.append(sum(p[field] for p in m["batch_shapes"]["by_phase"].values()))
        assert sums[1] - sums[0] == counts[key]
    return dict(counts)


def aggregate(waves):
    rows = [r for w in waves for r in w["rows"]]
    duration = sum(w["duration_s"] for w in waves)

    def quant(role, key):
        return percentile(
            [r[key] for r in rows if r["role"] == role and r.get(key) is not None], 0.95
        )

    n = sum(r["slo"] for r in rows)
    return {
        "jobs": len(rows),
        "target_final_stage_medians_ms": {
            key: median(values)
            for key in (
                "before_final_submission_ms",
                "ttft_ms",
                "client_to_render_ms",
                "command_queue_ms",
                "admit_to_first_prefill_ms",
                "prefill_span_ms",
                "prefill_participating_gpu_ms",
                "prefill_between_steps_ms",
                "stream_ms",
            )
            if (values := [r[key] for r in rows if r["role"] == "target" and key in r])
        },
        "valid": sum(r["valid"] for r in rows),
        "slo": n,
        "wave_duration_s": duration,
        "goodput_per_gpu_hour": n * 3600 / duration,
        "completed_per_gpu_hour": sum(bool(r.get("completed")) for r in rows)
        * 3600
        / duration,
        "reader_first_p95_ms": quant("reader", "first_ms"),
        "reader_gap_p95_of_p95_ms": quant("reader", "gap_p95_ms"),
        "target_first_p95_ms": quant("target", "first_ms"),
        "target_last_p95_ms": quant("target", "last_ms"),
        "strict_90": sum(r["explanation_words"] == 90 for r in rows),
        "gpu_s": sum(w["steps"]["gpu_ns"] for w in waves) / 1e9,
        "full_steps": sum(w["steps"]["full_steps"] for w in waves),
        "choices": dict(Counter(r["choice"] for r in rows)),
        "errors": dict(Counter(r["error"] for r in rows if r["error"])),
        "preview_errors": sum(r["preview_error"] is not None for r in rows),
        "invalid_seeds": [r["seed"] for r in rows if not r["valid"]],
    }


def verify(root):
    check_manifest(root)
    conf = read(root / "coder-pressure-registration.json")
    sha = (root / "trace-server.sha256").read_text().split()[0]
    assert sha == "8db3d5c634bab3ef5a4d3178e680db0c3c67beb781ecd042067654f2871bd990"
    result = {}
    reference_cli = None
    for gpu, modes in conf["orders"].items():
        runs = result[gpu] = {}
        for mode in modes:
            d = root / f"gpu{gpu}" / mode
            check_server(d, sha, gpu)
            cli = (d / "server-arguments.txt").read_text()
            if reference_cli is None:
                reference_cli = cli
            assert reference_cli == cli
            assert read(d / "warm.json")["ok"]
            log = (d / "server.log").read_bytes()
            waves = []
            for raw, (size, base) in zip(
                read(d / "waves.json"), conf["waves"], strict=True
            ):
                assert (raw["size"], raw["base"]) == (size, base)
                steps = reconcile(raw, log)
                wave_log = log[
                    raw["before"]["log_offset"] : raw["after"]["log_offset"]
                ].decode()
                rows = []
                spans = []
                for i, job in zip(range(size), raw["jobs"], strict=True):
                    work = job["work"]
                    seed = base + i
                    reader = i % 4 == 0
                    assert (
                        job["role"] == ("reader" if reader else "target")
                        and job["index"] == i
                    )
                    t0 = raw["origin_ns"] + 30_000_000 * i
                    assert (
                        t0 == job["scheduled_ns"]
                        and work["end_ns"] <= raw["drained_ns"]
                    )
                    facts = natural(
                        work,
                        seed,
                        2 if reader else 96,
                        0 if reader else 200,
                        delay=0 if reader else 0.5,
                    )
                    latency = work_latency(work, t0)
                    if latency.get("completed") and facts["verified_tool_calls"] == 2:
                        trace = join(wave_log, [work], details=True)[-1]
                        for key in (
                            "client_to_render_ms",
                            "command_queue_ms",
                            "admit_to_first_prefill_ms",
                            "prefill_span_ms",
                            "prefill_participating_gpu_ms",
                            "prefill_between_steps_ms",
                        ):
                            latency[key] = trace[key]
                    dec = job["decision"]
                    prev = job["preview"]
                    choice = dec.get(
                        "choice", "reader" if reader else "no_tool_decision"
                    )
                    if dec:
                        assert not reader
                        if mode == "off":
                            assert choice == "off"
                        elif mode == "headroom" and dec["foreground"] >= 6:
                            assert choice == "foreground_busy"
                        else:
                            assert choice in (
                                "submit",
                                "prefetch_in_flight",
                                "short_context",
                            )
                    if choice == "submit":
                        check_preview(work, "history", prev)
                        if prev.get("done_ns"):
                            spans.append((prev["sent_ns"], prev["done_ns"]))
                    else:
                        assert not prev
                    final = work["turns"][-1]
                    fmt = answer_format(
                        final.get("answer", ""), fixture(seed, 2 if reader else 96)[2]
                    )
                    rows.append(
                        {
                            "seed": seed,
                            "role": job["role"],
                            "valid": facts["valid"],
                            "tool_calls": facts["verified_tool_calls"],
                            "slo": qualifies(facts["valid"], latency),
                            "error": work.get("error"),
                            "explanation_words": fmt["explanation_words"],
                            "choice": choice,
                            "foreground": dec.get("foreground"),
                            "preview_error": prev.get("error"),
                            "arrival_slip_ms": (work["start_ns"] - t0) / 1e6,
                            **latency,
                        }
                    )
                spans.sort()
                assert all(b <= c for (_, b), (c, _) in itertools.pairwise(spans))
                waves.append(
                    {
                        "size": size,
                        "base": base,
                        "steps": steps,
                        "duration_s": (raw["drained_ns"] - raw["origin_ns"]) / 1e9,
                        "rows": rows,
                    }
                )
            total = aggregate(waves)
            lifetime = (
                int((d.parent / f"{mode}.end").read_text())
                - int((d.parent / f"{mode}.start").read_text())
            ) / 1e9
            assert lifetime > total["wave_duration_s"]
            total["server_lifetime_s"] = lifetime
            total["including_startup_goodput_per_gpu_hour"] = (
                total["slo"] * 3600 / lifetime
            )
            runs[mode] = {"all": total, "high": aggregate(waves[2:]), "waves": waves}
        off = runs["off"]["high"]
        for mode in ("bounded", "headroom"):
            v = runs[mode]["high"]
            runs[mode]["promotion"] = {
                "preserved_valid": set(v["invalid_seeds"]) <= set(off["invalid_seeds"]),
                "preserved_slo_count": v["slo"] >= off["slo"],
                "increased_high_goodput": v["goodput_per_gpu_hour"]
                > off["goodput_per_gpu_hour"],
                "protected_reader_first": v["reader_first_p95_ms"]
                <= 1.05 * off["reader_first_p95_ms"],
                "protected_reader_gap": v["reader_gap_p95_of_p95_ms"]
                <= 1.05 * off["reader_gap_p95_of_p95_ms"],
                "observed_saturation": v["full_steps"] > 0 and off["full_steps"] > 0,
            }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    print(json.dumps(verify(parser.parse_args().directory), indent=2))
