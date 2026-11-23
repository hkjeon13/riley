"""Bind committed prefill steps and execution spans to the original RAG requests."""

import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re
import runpy
import sys
import tarfile
from typing import NamedTuple


HERE = Path(__file__).resolve().parent
ADMISSION = HERE.parent / "20261124-w3-rag-admission"
FROZEN = HERE.parent / "20261124-w3-separated-client"
ARM = "gpu7/coder14-riley"
sys.path.insert(0, str(ADMISSION))
PRIOR = runpy.run_path(str(ADMISSION / "verify.py"))


class Step(NamedTuple):
    id: int
    start: int
    end: int
    commit: int
    gpu: int
    rows: int
    decode: int
    prefill: list[tuple[int, int, int]]


def put_unique(table, key, value):
    if key in table:
        raise ValueError(f"duplicate trace key {key}")
    table[key] = value


def read_trace(path, all_requests):
    admits, visible, steps, snapshots = {}, {}, [], []
    startup_counts = []
    for line in path.read_text().splitlines():
        if match := re.fullmatch(r"riley startup warm-up: (\d+) internal requests in .*", line):
            startup_counts.append(int(match[1]))
        elif line.startswith("W3-RAG-SNAPSHOT "):
            snapshots.append(tuple(map(int, line.split()[1:])))
        elif line.startswith("W3-RAG-ADMIT "):
            engine, scheduler, at = map(int, line.split()[1:])
            put_unique(admits, engine, (scheduler, at))
        elif line.startswith("W3-RAG-VISIBLE "):
            _, request, accepted, first = line.split()
            put_unique(visible, request, (int(accepted), int(first)))
        elif line.startswith("W3-RAG-STEP "):
            fields = line.split()
            items = [tuple(map(int, item.split(":"))) for item in fields[8].split(",")] if len(fields) == 9 else []
            step = Step(*map(int, fields[1:8]), items)
            if (not (0 <= step.start < step.end <= step.commit and 0 < step.gpu <= step.end - step.start + 50_000)
                    or not (0 < step.rows <= 1024 and step.decode >= 0)
                    or step.rows != step.decode + sum(item[1] for item in items)
                    or len({item[0] for item in items}) != len(items)
                    or (steps and (steps[-1].id >= step.id or steps[-1].commit > step.start))):
                raise ValueError(f"invalid or overlapping committed step {step.id}")
            steps.append(step)
    schedulers = {scheduler: at for scheduler, at in admits.values()}
    expected_engine_ids = {int(request[5:], 16) for request in all_requests}
    startup = set(admits) - expected_engine_ids
    if (not steps or len(admits) != len(schedulers) or not expected_engine_ids <= set(admits)
            or visible.keys() != all_requests or snapshots != [(len(admits) + len(visible) + len(steps), 0)]):
        raise ValueError("incomplete step or request timeline")
    first_request = min(admits[engine][1] for engine in expected_engine_ids)
    if (startup_counts != ([len(startup)] if startup else [])
            or (startup and (max(startup) >= min(expected_engine_ids)
                             or max(admits[engine][1] for engine in startup) >= first_request))):
        raise ValueError("unmatched admissions are not the registered startup warmup")
    startup_schedulers = {admits[engine][0] for engine in startup}
    by_scheduler = defaultdict(list)
    for step in steps:
        for scheduler, count, target in step.prefill:
            if (schedulers[scheduler] > step.start or not 0 < count <= target
                    or (scheduler in startup_schedulers and step.commit > first_request)):
                raise ValueError(f"prefill precedes admission or has invalid coverage in step {step.id}")
            by_scheduler[scheduler].append((step, count, target))
    return admits, visible, steps, by_scheduler, schedulers, len(startup), sum(step.commit <= first_request for step in steps)


def overlap(start, end, window_start, window_end):
    return max(0, min(end, window_end) - max(start, window_start))


def phases(row, timeline, observation, *, include_steps=False):
    admits, visible, steps, by_scheduler, schedulers, _, _ = timeline
    request = row["response_id"]
    scheduler, submitted = admits[int(request[5:], 16)]
    accepted, first = visible[request]
    own = by_scheduler[scheduler]
    end_token = 0
    for step, count, target in own:
        end_token += count
        if end_token != target:
            raise ValueError(f"prefill token coverage differs on {request}")
    if (end_token != row["prompt_tokens"] or not own or submitted > accepted
            or first - accepted != observation[1] or first < own[-1][0].commit):
        raise ValueError(f"first content timeline disagrees with the request observation on {request}")
    own_ids = {step.id for step, _, _ in own}
    own_exec = sum(overlap(step.start, step.end, accepted, first) for step, _, _ in own)
    foreign_prefill = [(step, overlap(step.start, step.end, accepted, first)) for step in steps
                       if step.id not in own_ids and step.prefill and step.end > accepted and step.start < first]
    foreign_prefill_ns = sum(span for _, span in foreign_prefill)
    newer_only_ns = sum(span for step, span in foreign_prefill
                        if all(schedulers[scheduler] > submitted for scheduler, _, _ in step.prefill))
    foreign_decode = sum(overlap(step.start, step.end, accepted, first) for step in steps if not step.prefill)
    host_gaps = first - accepted - own_exec - foreign_prefill_ns - foreign_decode
    if host_gaps < 0:
        raise ValueError(f"overlapping prefill intervals for {request}")
    first_wait = max(0, own[0][0].start - accepted)
    between_wait = sum(max(0, right[0].start - max(accepted, left[0].end))
                       for left, right in zip(own, own[1:]))
    last_wait = first - max(accepted, own[-1][0].end)
    if first_wait + own_exec + between_wait + last_wait != first - accepted:
        raise ValueError(f"prefill spans do not partition the request wait on {request}")
    result = {"own_prefill_steps": len(own), "own_prefill_with_decode": sum(step.decode > 0 for step, _, _ in own),
              "own_step_gpu_ms": sum(step.gpu for step, _, _ in own) / 1e6,
              "own_execution_wall_ms": own_exec / 1e6, "foreign_prefill_execution_wall_ms": foreign_prefill_ns / 1e6,
              "newer_only_prefill_wall_ms": newer_only_ns / 1e6,
              "other_decode_only_execution_wall_ms": foreign_decode / 1e6, "outside_execution_ms": host_gaps / 1e6,
              "before_first_prefill_ms": first_wait / 1e6, "between_prefill_steps_ms": between_wait / 1e6,
              "after_last_prefill_ms": last_wait / 1e6}
    if include_steps:
        result["own_steps"] = [{"step_id": step.id, "own_rows": count, "step_rows": step.rows, "decode_rows": step.decode,
                                "gpu_ms": step.gpu / 1e6, "start_from_admission_ms": (step.start - accepted) / 1e6}
                               for step, count, _ in own]
    return result


def read(path):
    return json.loads(path.read_bytes())


def verify(root):
    reg = read(HERE / "registration.json")
    for marker in ("build.exit", "campaign.exit"):
        if (root / marker).read_text().strip() != "0":
            raise ValueError(f"incomplete run: {marker}")
    for name, key in (("build-dispatch.sha256", "build_dispatch_sha256"),
                      ("campaign-dispatch.sha256", "campaign_dispatch_sha256")):
        if sha256((root / name).read_bytes()).hexdigest() != reg[key]:
            raise ValueError(f"registered input differs: {name}")
    arm = root / ARM
    receipt = read(arm / "receipt.json")
    binary, name = (root / "build-result.sha256").read_text().splitlines()[0].split()
    if (name != "riley-diagnostic" or binary != receipt["riley_binary_sha256"]
            or receipt["source_revision"] != reg["base_revision"]):
        raise ValueError("served binary or base revision differs from the build receipt")
    observations = PRIOR["observations"](arm / "server.log")
    timeline = read_trace(arm / "server.log", observations.keys())
    bound = observations.copy()
    old_summary = read(ADMISSION / "verified.json")["cells"]
    results = {}
    with (tarfile.open(ADMISSION / "raw-results.tar.gz", "r:gz") as admission_run,
          tarfile.open(FROZEN / "initial-results.tar.gz", "r:gz") as frozen_run):
        old = lambda name: json.load(admission_run.extractfile(name))
        frozen = lambda name: json.load(frozen_run.extractfile(name))
        ref_receipt = old(f"{ARM}/receipt.json")
        if (binary == ref_receipt["riley_binary_sha256"] or
                any(receipt[k] != ref_receipt[k] for k in set(ref_receipt) - {"riley_binary_sha256", "results"})):
            raise ValueError("step trace server differs outside the binary")
        plan, prior_plan = read(arm / "plan.json"), old(f"{ARM}/plan.json")
        if any(plan[key] != prior_plan[key] for key in ("server", "cells", "server_cpu_affinity")):
            raise ValueError("an arrival schedule or server flag differs")
        before = old("gpu7/after.json")
        for phase in ("before", "after"):
            holder = read(root / f"gpu7/{phase}.json")
            if (holder["physical_index"] != 7 or holder["phase"] != "idle" or holder["other_processes"]
                    or holder["allocation_bytes"] != holder["torch_reserved_bytes"] or holder["allocation_bytes"] != 20 * 2**30
                    or any(holder[key] != before[key] for key in ("pid", "start_ticks", "gpu"))):
                raise ValueError("GPU 7 reservation differs")
        for seed, label in enumerate(PRIOR["LABELS"], start=1):
            path = arm / f"{label}.json"
            if receipt["results"][label] != sha256(path.read_bytes()).hexdigest():
                raise ValueError(f"missing client data for {label}")
            capture = read(path)
            focus = reg["focus_requests_by_seed"][str(seed)]
            cell = PRIOR["analyze_cell"](capture, frozen(f"{ARM}/{label}.json"),
                                         frozen(f"gpu7/coder14-vllm/{label}.json"), focus[0], bound)
            targets = set(focus) | {cell["diagnostic_top_four"][0]["request_index"]}
            for row in cell["all_requests"]:
                row.update(phases(row, timeline, observations[row["response_id"]],
                                  include_steps=row["request_index"] in targets))
            reference = old(f"{ARM}/{label}.json")
            pair_delta = [abs(a["write_error_ms"] - b["write_error_ms"])
                          for a, b in zip(capture["measured"], reference["measured"], strict=True)]
            cell["export_only_write_error_p95_ms"] = PRIOR["p"](pair_delta, .95)
            cell["export_only_ttft_p95_ms"] = old_summary[label]["diagnostic_ttft_p95_ms"]
            cell["focus_requests"] = [row for row in cell["all_requests"] if row["request_index"] in targets]
            results[label] = cell
    if bound:
        raise ValueError("unmatched prior request observations remain")
    return {"base_revision": receipt["source_revision"], "gpu": 7, "completed_observations": len(observations),
            "committed_steps": len(timeline[2]), "startup_backend_requests": timeline[-2], "startup_steps": timeline[-1],
            "diagnostic_binary_sha256": binary, "cells": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", type=Path)
    print(json.dumps(verify(parser.parse_args().stage), indent=2, allow_nan=False))
