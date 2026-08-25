#!/usr/bin/env python3
"""Fail-closed checker for source-bound PR-16 reliability soak evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, NoReturn, Sequence


MANIFEST_VERSION = "rustinfer.reliability-soak-manifest.v1"
RUN_VERSION = "rustinfer.reliability-soak-run.v1"
EVENT_VERSION = "rustinfer.reliability-soak-event.v1"
REPORT_VERSION = "rustinfer.reliability-soak-report.v1"
REQUIRED_KINDS = {
    "steady",
    "burst-idle",
    "mixed",
    "invalid",
    "overload",
    "cancellation-disconnect",
    "near-kv",
    "graceful-restart",
    "rollback",
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_RE = re.compile(r"^[0-9a-f]{40}([0-9a-f]{24})?$")
PYTHON_RE = re.compile(r"(^|/)(python|python[23](?:\.[0-9]+)?)(?:$|\s)", re.IGNORECASE)
MAX_INPUT_BYTES = 512 * 1024 * 1024


class InputError(ValueError):
    """Evidence is malformed, incomplete, or not source-bound."""


def _fail(path: str, message: str) -> NoReturn:
    raise InputError(f"{path}: {message}")


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise InputError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _nonfinite(value: str) -> NoReturn:
    raise InputError(f"non-finite JSON number {value!r} is forbidden")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        if path.stat().st_size > MAX_INPUT_BYTES:
            _fail(str(path), "exceeds evidence size bound")
        with path.open(encoding="utf-8") as handle:
            value = json.load(
                handle,
                object_pairs_hook=_pairs,
                parse_constant=_nonfinite,
            )
    except FileNotFoundError:
        _fail(str(path), "file does not exist")
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        _fail(str(path), f"cannot read strict UTF-8 JSON: {error}")
    if not isinstance(value, dict):
        _fail(str(path), "root must be an object")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        if path.stat().st_size > MAX_INPUT_BYTES:
            _fail(str(path), "exceeds evidence size bound")
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        _fail(str(path), "file does not exist")
    except (OSError, UnicodeDecodeError) as error:
        _fail(str(path), f"cannot read UTF-8 JSONL: {error}")
    if not lines:
        _fail(str(path), "must contain events")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            _fail(f"{path}:{line_number}", "blank JSONL lines are forbidden")
        try:
            value = json.loads(
                line,
                object_pairs_hook=_pairs,
                parse_constant=_nonfinite,
            )
        except (json.JSONDecodeError, InputError) as error:
            _fail(f"{path}:{line_number}", f"invalid JSON: {error}")
        if not isinstance(value, dict):
            _fail(f"{path}:{line_number}", "event must be an object")
        rows.append(value)
    return rows


def _object(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(path, "must be an object")
    return value


def _exact(value: Any, keys: set[str], path: str) -> dict[str, Any]:
    result = _object(value, path)
    missing = sorted(keys - set(result))
    extra = sorted(set(result) - keys)
    if missing or extra:
        _fail(path, f"closed object mismatch; missing={missing}, unexpected={extra}")
    return result


def _string(value: Any, path: str, pattern: re.Pattern[str] | None = None) -> str:
    if not isinstance(value, str) or not value:
        _fail(path, "must be a non-empty string")
    if pattern is not None and pattern.fullmatch(value) is None:
        _fail(path, "has invalid format")
    return value


def _integer(value: Any, path: str, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        _fail(path, f"must be an integer >= {minimum}")
    return value


def _number(value: Any, path: str, minimum: float = 0.0) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        _fail(path, "must be a finite number")
    try:
        result = float(value)
    except (OverflowError, ValueError):
        _fail(path, "must be representable as a finite number")
    if not math.isfinite(result) or result < minimum:
        _fail(path, f"must be finite and >= {minimum}")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while block := handle.read(1024 * 1024):
                digest.update(block)
    except OSError as error:
        _fail(str(path), f"cannot hash manifest: {error}")
    return digest.hexdigest()


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_manifest(value: dict[str, Any], path: str) -> dict[str, Any]:
    manifest = _exact(
        value,
        {"schema_version", "contract_id", "target", "thresholds", "requests", "golden", "scenarios"},
        path,
    )
    if manifest["schema_version"] != MANIFEST_VERSION:
        _fail(f"{path}.schema_version", f"must be {MANIFEST_VERSION}")
    _string(manifest["contract_id"], f"{path}.contract_id")
    target = _exact(
        manifest["target"],
        {"kind", "binary", "model_path", "bind", "completion_path", "health_path", "metrics_path", "shutdown_signal", "launch_arguments"},
        f"{path}.target",
    )
    if target["kind"] not in {"process", "container"}:
        _fail(f"{path}.target.kind", "must be process or container")
    if target["shutdown_signal"] != "TERM":
        _fail(f"{path}.target.shutdown_signal", "release soak requires SIGTERM")
    for key in ("binary", "model_path", "bind", "completion_path", "health_path", "metrics_path"):
        _string(target[key], f"{path}.target.{key}")
    if not isinstance(target["launch_arguments"], list) or not target["launch_arguments"]:
        _fail(f"{path}.target.launch_arguments", "must be a non-empty string array")
    for index, argument in enumerate(target["launch_arguments"]):
        if not isinstance(argument, str):
            _fail(f"{path}.target.launch_arguments[{index}]", "must be a string")
    requests = _object(manifest["requests"], f"{path}.requests")
    golden = _exact(
        manifest["golden"],
        {"request_profile", "digest_domain", "generated_sha256", "provenance_sha256"},
        f"{path}.golden",
    )
    if golden["request_profile"] not in requests:
        _fail(f"{path}.golden.request_profile", "references an absent request")
    if golden["digest_domain"] != "completion-text-utf8":
        _fail(f"{path}.golden.digest_domain", "must be completion-text-utf8")
    for key in ("generated_sha256", "provenance_sha256"):
        digest = _string(golden[key], f"{path}.golden.{key}", SHA256_RE)
        if digest == "0" * 64:
            _fail(f"{path}.golden.{key}", "template placeholder must be materialized")
    thresholds = _exact(
        manifest["thresholds"],
        {
            "sample_interval_ms", "maximum_sample_gap_ms", "minimum_samples_per_scenario",
            "plateau_tail_fraction", "maximum_rss_plateau_growth_bytes",
            "maximum_rss_slope_bytes_per_hour", "maximum_vram_plateau_growth_bytes",
            "maximum_vram_slope_bytes_per_hour", "minimum_cancellations",
            "minimum_disconnects", "minimum_overloads", "graceful_shutdown_deadline_ms",
        },
        f"{path}.thresholds",
    )
    for key in (
        "sample_interval_ms", "maximum_sample_gap_ms", "minimum_samples_per_scenario",
        "maximum_rss_plateau_growth_bytes", "minimum_cancellations", "minimum_disconnects",
        "minimum_overloads", "graceful_shutdown_deadline_ms",
    ):
        _integer(thresholds[key], f"{path}.thresholds.{key}", 1 if key != "maximum_rss_plateau_growth_bytes" else 0)
    for key in (
        "maximum_rss_slope_bytes_per_hour", "maximum_vram_plateau_growth_bytes",
        "maximum_vram_slope_bytes_per_hour",
    ):
        _number(thresholds[key], f"{path}.thresholds.{key}")
    fraction = _number(thresholds["plateau_tail_fraction"], f"{path}.thresholds.plateau_tail_fraction")
    if fraction <= 0 or fraction > 1:
        _fail(f"{path}.thresholds.plateau_tail_fraction", "must be in (0, 1]")
    scenarios = manifest["scenarios"]
    if not isinstance(scenarios, list) or not scenarios:
        _fail(f"{path}.scenarios", "must be a non-empty array")
    seen: set[str] = set()
    kind_counts: Counter[str] = Counter()
    rollback_modes: set[str] = set()
    for index, raw in enumerate(scenarios):
        scenario_path = f"{path}.scenarios[{index}]"
        scenario = _object(raw, scenario_path)
        required = {"id", "kind", "required", "duration_seconds", "concurrency", "cycle_interval_ms", "request_profile", "execution_completion"}
        if scenario.get("kind") == "mixed":
            required.add("secondary_request_profile")
        _exact(scenario, required, scenario_path)
        scenario_id = _string(scenario["id"], f"{scenario_path}.id")
        if scenario_id in seen:
            _fail(f"{scenario_path}.id", "duplicate scenario id")
        seen.add(scenario_id)
        kind = _string(scenario["kind"], f"{scenario_path}.kind")
        if kind not in REQUIRED_KINDS:
            _fail(f"{scenario_path}.kind", "is not a closed v1 scenario kind")
        if scenario["required"] is not True:
            _fail(f"{scenario_path}.required", "all release scenarios must be required")
        kind_counts[kind] += 1
        _integer(scenario["duration_seconds"], f"{scenario_path}.duration_seconds", 1)
        _integer(scenario["concurrency"], f"{scenario_path}.concurrency", 1)
        cycle_interval_ms = _integer(scenario["cycle_interval_ms"], f"{scenario_path}.cycle_interval_ms")
        if kind == "burst-idle" and cycle_interval_ms < 1000:
            _fail(f"{scenario_path}.cycle_interval_ms", "burst-idle requires a visible idle interval")
        profile = _string(scenario["request_profile"], f"{scenario_path}.request_profile")
        if profile not in requests:
            _fail(f"{scenario_path}.request_profile", "references an absent request")
        mode = scenario["execution_completion"]
        if mode not in {"iteration-batch", "per-operation"}:
            _fail(f"{scenario_path}.execution_completion", "must be a stable exact mode")
        if kind == "rollback":
            rollback_modes.add(mode)
    expected_counts = Counter({kind: 1 for kind in REQUIRED_KINDS})
    expected_counts["rollback"] = 2
    if kind_counts != expected_counts:
        _fail(f"{path}.scenarios", f"required kind counts mismatch: {dict(kind_counts)}")
    if rollback_modes != {"iteration-batch", "per-operation"}:
        _fail(f"{path}.scenarios", "rollback must cover both exact completion modes")
    return manifest


def _validate_run(value: dict[str, Any], path: str, manifest_sha: str) -> dict[str, Any]:
    run = _exact(
        value,
        {"schema_version", "run_id", "manifest_sha256", "binding_sha256", "source", "target", "started_at_utc"},
        path,
    )
    if run["schema_version"] != RUN_VERSION:
        _fail(f"{path}.schema_version", f"must be {RUN_VERSION}")
    _string(run["run_id"], f"{path}.run_id")
    if run["manifest_sha256"] != manifest_sha:
        _fail(f"{path}.manifest_sha256", "does not bind the exact manifest bytes")
    source = _exact(
        run["source"],
        {"git_commit", "git_dirty", "source_archive_sha256", "binary_sha256", "image_sha256", "model_sha256", "model_id", "model_revision"},
        f"{path}.source",
    )
    _string(source["git_commit"], f"{path}.source.git_commit", GIT_RE)
    if source["git_dirty"] is not False:
        _fail(f"{path}.source.git_dirty", "release evidence requires a clean source tree")
    for key in ("source_archive_sha256", "binary_sha256", "image_sha256", "model_sha256"):
        _string(source[key], f"{path}.source.{key}", SHA256_RE)
    _string(source["model_id"], f"{path}.source.model_id")
    _string(source["model_revision"], f"{path}.source.model_revision")
    binding_sha = _canonical_sha256(source)
    if run["binding_sha256"] != binding_sha:
        _fail(f"{path}.binding_sha256", "does not bind the canonical source object")
    target = _exact(run["target"], {"kind", "pid", "image_id", "command_sha256"}, f"{path}.target")
    if target["kind"] not in {"process", "container"}:
        _fail(f"{path}.target.kind", "must be process or container")
    _integer(target["pid"], f"{path}.target.pid", 1)
    _string(target["image_id"], f"{path}.target.image_id")
    _string(target["command_sha256"], f"{path}.target.command_sha256", SHA256_RE)
    _string(run["started_at_utc"], f"{path}.started_at_utc")
    return run


def _validate_sample(event: dict[str, Any], path: str) -> None:
    process = _exact(event["process"], {"pid", "rss_bytes", "hwm_bytes", "fd_count", "thread_count", "children"}, f"{path}.process")
    _integer(process["pid"], f"{path}.process.pid", 0 if event["scenario_id"] is None else 1)
    for key in ("rss_bytes", "hwm_bytes", "fd_count", "thread_count"):
        _integer(process[key], f"{path}.process.{key}")
    if not isinstance(process["children"], list):
        _fail(f"{path}.process.children", "must be an array")
    for index, raw in enumerate(process["children"]):
        child = _exact(raw, {"pid", "comm", "executable"}, f"{path}.process.children[{index}]")
        _integer(child["pid"], f"{path}.process.children[{index}].pid", 1)
        _string(child["comm"], f"{path}.process.children[{index}].comm")
        _string(child["executable"], f"{path}.process.children[{index}].executable")
    gpu = _exact(event["gpu"], {"vram_bytes"}, f"{path}.gpu")
    _integer(gpu["vram_bytes"], f"{path}.gpu.vram_bytes")
    metrics = _exact(
        event["metrics"],
        {"active_requests", "waiting_requests", "kv_allocated_blocks", "allocation", "counters"},
        f"{path}.metrics",
    )
    for key in ("active_requests", "waiting_requests", "kv_allocated_blocks"):
        _integer(metrics[key], f"{path}.metrics.{key}")
    allocation = _exact(metrics["allocation"], {"device_live_count", "device_live_bytes", "pinned_live_count", "pinned_live_bytes"}, f"{path}.metrics.allocation")
    for key, value in allocation.items():
        _integer(value, f"{path}.metrics.allocation.{key}")
    counters = _exact(metrics["counters"], {"cancellations", "disconnects", "overloads", "dropped_samples"}, f"{path}.metrics.counters")
    for key, value in counters.items():
        _integer(value, f"{path}.metrics.counters.{key}")
    if not isinstance(event["sample_dropped"], bool):
        _fail(f"{path}.sample_dropped", "must be boolean")


def _validate_events(rows: list[dict[str, Any]], binding_sha: str, scenario_ids: set[str]) -> None:
    common = {"schema_version", "sequence", "monotonic_ns", "kind", "scenario_id", "binding_sha256"}
    extras = {
        "run_start": set(), "scenario_start": {"execution_completion"},
        "sample": {"process", "gpu", "metrics", "sample_dropped"},
        "request": {"request_id", "outcome", "http_status", "latency_ms", "generated_sha256"},
        "restart": {"graceful", "exit_code", "elapsed_ms", "before_generated_sha256", "after_generated_sha256"},
        "scenario_end": {"status"}, "failure": {"stage", "message"}, "run_end": {"status"},
    }
    previous_time = -1
    for index, event in enumerate(rows, 1):
        path = f"events[{index}]"
        kind = event.get("kind")
        if kind not in extras:
            _fail(f"{path}.kind", "is not a closed v1 event kind")
        _exact(event, common | extras[kind], path)
        if event["schema_version"] != EVENT_VERSION:
            _fail(f"{path}.schema_version", f"must be {EVENT_VERSION}")
        if event["sequence"] != index:
            _fail(f"{path}.sequence", f"must be contiguous value {index}")
        monotonic_ns = _integer(event["monotonic_ns"], f"{path}.monotonic_ns")
        if monotonic_ns <= previous_time:
            _fail(f"{path}.monotonic_ns", "must be strictly increasing")
        previous_time = monotonic_ns
        if event["binding_sha256"] != binding_sha:
            _fail(f"{path}.binding_sha256", "does not match run binding")
        scenario_id = event["scenario_id"]
        if scenario_id is not None and scenario_id not in scenario_ids:
            _fail(f"{path}.scenario_id", "is absent from manifest")
        if kind not in {"run_start", "run_end", "sample", "failure"} and scenario_id is None:
            _fail(f"{path}.scenario_id", "must identify a scenario")
        if kind in {"run_start", "run_end"} and scenario_id is not None:
            _fail(f"{path}.scenario_id", "run boundary events must use null")
        if kind == "sample":
            _validate_sample(event, path)
        elif kind == "request":
            _string(event["request_id"], f"{path}.request_id")
            if event["outcome"] not in {"success", "invalid", "overload", "cancelled", "disconnected", "timeout", "failure"}:
                _fail(f"{path}.outcome", "is not a closed outcome")
            status = _integer(event["http_status"], f"{path}.http_status")
            if status > 599:
                _fail(f"{path}.http_status", "must be <= 599")
            _number(event["latency_ms"], f"{path}.latency_ms")
            generated = event["generated_sha256"]
            if generated is not None:
                _string(generated, f"{path}.generated_sha256", SHA256_RE)
            if (event["outcome"] == "success") != (generated is not None):
                _fail(f"{path}.generated_sha256", "must be present exactly for success")
            if event["outcome"] == "success" and not 200 <= status < 300:
                _fail(f"{path}.http_status", "success requires 2xx")
            if event["outcome"] == "invalid" and not (400 <= status < 500 and status != 429):
                _fail(f"{path}.http_status", "invalid requires non-429 4xx")
            if event["outcome"] == "overload" and status != 429:
                _fail(f"{path}.http_status", "overload requires 429")
        elif kind == "restart":
            if not isinstance(event["graceful"], bool):
                _fail(f"{path}.graceful", "must be boolean")
            _integer(event["exit_code"], f"{path}.exit_code")
            _number(event["elapsed_ms"], f"{path}.elapsed_ms")
            _string(event["before_generated_sha256"], f"{path}.before_generated_sha256", SHA256_RE)
            _string(event["after_generated_sha256"], f"{path}.after_generated_sha256", SHA256_RE)
        elif kind in {"scenario_end", "run_end"} and event["status"] not in {"success", "failure"}:
            _fail(f"{path}.status", "must be success or failure")
        elif kind == "failure":
            _string(event["stage"], f"{path}.stage")
            _string(event["message"], f"{path}.message")
    if rows[0]["kind"] != "run_start" or rows[-1]["kind"] != "run_end":
        _fail("events", "must be bracketed by run_start and run_end")


def _slope_per_hour(samples: list[dict[str, Any]], field: str) -> float:
    if len(samples) < 2:
        return math.inf
    x = [(sample["monotonic_ns"] - samples[0]["monotonic_ns"]) / 1_000_000_000 for sample in samples]
    y = [float(sample[field.split(".")[0]][field.split(".")[1]]) for sample in samples]
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    denominator = sum((value - mean_x) ** 2 for value in x)
    if denominator == 0:
        return math.inf
    return sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y)) / denominator * 3600


def _check(name: str, passed: bool, observed: Any, threshold: Any) -> dict[str, Any]:
    return {"name": name, "passed": passed, "observed": observed, "threshold": threshold}


def evaluate(manifest_path: Path | str, run_directory: Path | str) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema_version": REPORT_VERSION, "status": "error", "passed": False,
        "bindings": None, "scenario_summaries": [], "observations": {}, "checks": [], "errors": [],
    }
    try:
        manifest_file = Path(manifest_path)
        directory = Path(run_directory)
        manifest = _validate_manifest(_load_json(manifest_file), str(manifest_file))
        run = _validate_run(_load_json(directory / "run.json"), str(directory / "run.json"), _sha256(manifest_file))
        if run["target"]["kind"] != manifest["target"]["kind"]:
            _fail("run.json.target.kind", "does not match manifest target kind")
        if run["target"]["image_id"] != f"sha256:{run['source']['image_sha256']}":
            _fail("run.json.target.image_id", "does not match bound image SHA-256")
        rows = _load_jsonl(directory / "events.jsonl")
        scenarios = {scenario["id"]: scenario for scenario in manifest["scenarios"]}
        _validate_events(rows, run["binding_sha256"], set(scenarios))
        by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for event in rows:
            if event["scenario_id"] is not None:
                by_scenario[event["scenario_id"]].append(event)
        checks: list[dict[str, Any]] = []
        boundary_counts = Counter(event["kind"] for event in rows)
        checks.append(_check("run_boundaries", boundary_counts["run_start"] == 1 and boundary_counts["run_end"] == 1 and rows[-1]["status"] == "success", {"run_start": boundary_counts["run_start"], "run_end": boundary_counts["run_end"], "status": rows[-1].get("status")}, "one successful pair"))
        thresholds = manifest["thresholds"]
        outcome_counts: Counter[str] = Counter()
        metric_counter_maxima: Counter[str] = Counter()
        rollback_hashes: dict[str, set[str]] = {}
        for scenario_id, scenario in scenarios.items():
            events = by_scenario[scenario_id]
            kinds = Counter(event["kind"] for event in events)
            samples = [event for event in events if event["kind"] == "sample"]
            requests = [event for event in events if event["kind"] == "request"]
            starts = [event for event in events if event["kind"] == "scenario_start"]
            for request in requests:
                outcome_counts[request["outcome"]] += 1
            for sample in samples:
                for name, value in sample["metrics"]["counters"].items():
                    metric_counter_maxima[name] = max(metric_counter_maxima[name], value)
            counters_monotonic = all(
                right["process"]["pid"] != left["process"]["pid"]
                or all(right["metrics"]["counters"][name] >= left["metrics"]["counters"][name] for name in left["metrics"]["counters"])
                for left, right in zip(samples, samples[1:])
            )
            complete = kinds["scenario_start"] == 1 and kinds["scenario_end"] == 1 and events and events[0]["kind"] == "scenario_start" and events[-1].get("status") == "success"
            checks.append(_check(f"{scenario_id}.complete", complete, dict(kinds), "one successful start/end"))
            mode_matches = len(starts) == 1 and starts[0]["execution_completion"] == scenario["execution_completion"]
            checks.append(_check(f"{scenario_id}.execution_completion", mode_matches, None if not starts else starts[0]["execution_completion"], scenario["execution_completion"]))
            checks.append(_check(f"{scenario_id}.service_counters_monotonic", counters_monotonic, counters_monotonic, True))
            checks.append(_check(f"{scenario_id}.samples", len(samples) >= thresholds["minimum_samples_per_scenario"], len(samples), thresholds["minimum_samples_per_scenario"]))
            checks.append(_check(f"{scenario_id}.requests", bool(requests), len(requests), ">= 1"))
            restart_times = [event["monotonic_ns"] for event in events if event["kind"] == "restart"]
            gaps = [
                (right["monotonic_ns"] - left["monotonic_ns"]) / 1_000_000
                for left, right in zip(samples, samples[1:])
                if not any(left["monotonic_ns"] < restart < right["monotonic_ns"] for restart in restart_times)
            ]
            maximum_gap = max(gaps, default=0.0)
            checks.append(_check(f"{scenario_id}.sample_gap_ms", maximum_gap <= thresholds["maximum_sample_gap_ms"], maximum_gap, thresholds["maximum_sample_gap_ms"]))
            tail_count = max(2, math.ceil(len(samples) * thresholds["plateau_tail_fraction"]))
            tail = samples[-tail_count:]
            if len(tail) >= 2:
                rss_growth = max(sample["process"]["rss_bytes"] for sample in tail) - min(sample["process"]["rss_bytes"] for sample in tail)
                vram_growth = max(sample["gpu"]["vram_bytes"] for sample in tail) - min(sample["gpu"]["vram_bytes"] for sample in tail)
                rss_slope = _slope_per_hour(tail, "process.rss_bytes")
                vram_slope = _slope_per_hour(tail, "gpu.vram_bytes")
            else:
                rss_growth = vram_growth = 0
                rss_slope = vram_slope = None
            checks.extend([
                _check(f"{scenario_id}.rss_plateau_growth", rss_growth <= thresholds["maximum_rss_plateau_growth_bytes"], rss_growth, thresholds["maximum_rss_plateau_growth_bytes"]),
                _check(f"{scenario_id}.rss_slope_per_hour", rss_slope is not None and rss_slope <= thresholds["maximum_rss_slope_bytes_per_hour"], rss_slope, thresholds["maximum_rss_slope_bytes_per_hour"]),
                _check(f"{scenario_id}.vram_plateau_growth", vram_growth <= thresholds["maximum_vram_plateau_growth_bytes"], vram_growth, thresholds["maximum_vram_plateau_growth_bytes"]),
                _check(f"{scenario_id}.vram_slope_per_hour", vram_slope is not None and vram_slope <= thresholds["maximum_vram_slope_bytes_per_hour"], vram_slope, thresholds["maximum_vram_slope_bytes_per_hour"]),
            ])
            allowed = {"success"}
            if scenario["kind"] == "invalid":
                allowed = {"invalid"}
            elif scenario["kind"] == "overload":
                allowed = {"success", "overload"}
            elif scenario["kind"] == "cancellation-disconnect":
                allowed = {"success", "cancelled", "disconnected"}
            unexpected = Counter(request["outcome"] for request in requests if request["outcome"] not in allowed)
            checks.append(_check(f"{scenario_id}.request_outcomes", not unexpected, dict(unexpected), sorted(allowed)))
            if scenario["kind"] == "rollback":
                hashes = {request["generated_sha256"] for request in requests if request["outcome"] == "success"}
                rollback_hashes[scenario["execution_completion"]] = hashes
            report["scenario_summaries"].append({
                "scenario_id": scenario_id, "kind": scenario["kind"], "events": len(events),
                "samples": len(samples), "requests": len(requests), "maximum_sample_gap_ms": maximum_gap,
                "rss_slope_bytes_per_hour": rss_slope, "vram_slope_bytes_per_hour": vram_slope,
            })
        final_samples = [event for event in rows if event["kind"] == "sample" and event["scenario_id"] is None]
        final_shape = len(final_samples) == 1 and len(rows) >= 2 and rows[-2] is final_samples[0]
        checks.append(_check("final_sample_position", final_shape, len(final_samples), "exactly one penultimate global sample"))
        first_process_sample = next((event for event in rows if event["kind"] == "sample" and event["scenario_id"] is not None), None)
        checks.append(_check("initial_target_pid_binding", first_process_sample is not None and first_process_sample["process"]["pid"] == run["target"]["pid"], None if first_process_sample is None else first_process_sample["process"]["pid"], run["target"]["pid"]))
        final = final_samples[-1] if final_samples else None
        final_values = None if final is None else {
            "active_requests": final["metrics"]["active_requests"],
            "waiting_requests": final["metrics"]["waiting_requests"],
            "kv_allocated_blocks": final["metrics"]["kv_allocated_blocks"],
            **final["metrics"]["allocation"],
        }
        checks.append(_check("final_quiescence", final_values is not None and all(value == 0 for value in final_values.values()), final_values, "all zero"))
        python_children = []
        for event in rows:
            if event["kind"] == "sample":
                for child in event["process"]["children"]:
                    if PYTHON_RE.search(child["comm"]) or PYTHON_RE.search(child["executable"]):
                        python_children.append(child)
        checks.append(_check("no_python_children", not python_children, python_children, []))
        dropped = any(event["kind"] == "sample" and (event["sample_dropped"] or event["metrics"]["counters"]["dropped_samples"] != 0) for event in rows)
        checks.append(_check("no_dropped_samples", not dropped, dropped, False))
        failures = [event for event in rows if event["kind"] == "failure"]
        checks.append(_check("no_failure_events", not failures, len(failures), 0))
        checks.extend([
            _check("cancellations_observed", outcome_counts["cancelled"] >= thresholds["minimum_cancellations"], outcome_counts["cancelled"], thresholds["minimum_cancellations"]),
            _check("disconnects_observed", outcome_counts["disconnected"] >= thresholds["minimum_disconnects"], outcome_counts["disconnected"], thresholds["minimum_disconnects"]),
            _check("overloads_observed", outcome_counts["overload"] >= thresholds["minimum_overloads"], outcome_counts["overload"], thresholds["minimum_overloads"]),
            _check("service_cancellations_observed", metric_counter_maxima["cancellations"] >= thresholds["minimum_cancellations"], metric_counter_maxima["cancellations"], thresholds["minimum_cancellations"]),
            _check("service_disconnects_observed", metric_counter_maxima["disconnects"] >= thresholds["minimum_disconnects"], metric_counter_maxima["disconnects"], thresholds["minimum_disconnects"]),
            _check("service_overloads_observed", metric_counter_maxima["overloads"] >= thresholds["minimum_overloads"], metric_counter_maxima["overloads"], thresholds["minimum_overloads"]),
        ])
        restarts = [event for event in rows if event["kind"] == "restart"]
        expected_golden = manifest["golden"]["generated_sha256"]
        restart_ok = len(restarts) == 1 and restarts[0]["graceful"] and restarts[0]["exit_code"] == 0 and restarts[0]["elapsed_ms"] <= thresholds["graceful_shutdown_deadline_ms"] and restarts[0]["before_generated_sha256"] == expected_golden and restarts[0]["after_generated_sha256"] == expected_golden
        checks.append(_check("graceful_restart_golden_parity", restart_ok, restarts, "one bounded graceful exact-parity restart"))
        left = rollback_hashes.get("iteration-batch", set())
        right = rollback_hashes.get("per-operation", set())
        rollback_ok = left == {expected_golden} and right == {expected_golden}
        checks.append(_check("rollback_golden_parity", rollback_ok, {"iteration-batch": sorted(left), "per-operation": sorted(right)}, "one identical non-null hash"))
        passed = all(check["passed"] for check in checks)
        report.update({
            "status": "passed" if passed else "failed", "passed": passed,
            "bindings": {"manifest_sha256": run["manifest_sha256"], "binding_sha256": run["binding_sha256"], "source": run["source"]},
            "observations": {"event_count": len(rows), "outcome_counts": dict(sorted(outcome_counts.items())), "service_counter_maxima": dict(sorted(metric_counter_maxima.items())), "final": final_values},
            "checks": checks,
        })
    except InputError as error:
        report["errors"] = [str(error)]
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--run-directory", required=True, type=Path)
    parser.add_argument("--report", type=Path, help="create without overwriting")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = evaluate(args.manifest, args.run_directory)
    encoded = json.dumps(report, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.report is not None:
        try:
            with args.report.open("x", encoding="utf-8") as handle:
                handle.write(encoded)
        except (FileExistsError, OSError) as error:
            print(f"cannot create report {args.report}: {error}", file=sys.stderr)
            return 2
    sys.stdout.write(encoded)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
