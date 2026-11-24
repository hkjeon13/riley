#!/usr/bin/env python3
"""Verify improvement-plan links, evidence digests, and recorded distribution inputs."""

import argparse
from decimal import Decimal
import gzip
from hashlib import file_digest, sha256
import json
from math import isclose, isfinite
from pathlib import Path
import re
from statistics import fmean
import subprocess
import sys
import tomllib
from urllib.parse import unquote, urlsplit


MANIFEST = Path("benchmarks/evidence/20261119-improvement-qualification/manifest.json")
PLANS = Path("docs/improvement-plan")
INLINE_LINK = re.compile(r"\[[^\]\n]+\]\((<[^>]+>|[^\s)]+)\)")
ATX_HEADING = re.compile(r"^ {0,3}#{1,6}[ \t]+(.+?)\s*#*\s*$")
FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
METRIC_PAIR = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*/\s*([0-9]+(?:\.[0-9]+)?)")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate manifest key: {key}")
        result[key] = value
    return result


def evidence_entries(value):
    if isinstance(value, dict):
        if "path" in value:
            yield value
        for child in value.values():
            yield from evidence_entries(child)
    elif isinstance(value, list):
        for child in value:
            yield from evidence_entries(child)


def check_evidence(root, manifest):
    count = 0
    for entry in evidence_entries(manifest):
        name, expected = entry.get("path"), entry.get("sha256")
        if not isinstance(name, str) or not name or Path(name).is_absolute():
            raise ValueError(f"invalid evidence path: {name!r}")
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError(f"invalid evidence SHA-256: {name}")
        artifact = (root / name).resolve()
        if not artifact.is_relative_to(root) or not artifact.is_file():
            raise ValueError(f"evidence file missing from repository: {name}")
        with artifact.open("rb") as stream:
            actual = file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"evidence SHA-256 differs: {name}")
        count += 1
    if count == 0:
        raise ValueError("no checked evidence files in manifest")
    return count


def check_distribution_inputs(root, candidate):
    revisions = (candidate["source_revision"], *candidate.get("follow_up_audited_revisions", []))
    if "audited_main" in candidate:
        revisions += (candidate["audited_main"],)
    if any(not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision)
           for revision in revisions):
        raise ValueError("expected full Git revisions for the recorded distribution comparison")
    inputs = candidate["matching_distribution_input_git_objects"]
    if not isinstance(inputs, dict) or not inputs:
        raise ValueError("no recorded distribution inputs")
    expected = {}
    for path, object_id in inputs.items():
        if not path or Path(path).is_absolute() or ".." in Path(path).parts:
            raise ValueError(f"invalid distribution input path: {path!r}")
        if not isinstance(object_id, str) or not re.fullmatch(r"[0-9a-f]{40}", object_id):
            raise ValueError(f"invalid recorded Git object: {path}")
        for revision in revisions:
            expected[f"{revision}:{path}"] = object_id
    result = subprocess.run(["git", "rev-parse", *expected], cwd=root, capture_output=True, text=True)
    entries = result.stdout.splitlines()
    if result.returncode or len(entries) != len(expected):
        raise ValueError(f"unable to resolve the recorded distribution inputs: {result.stderr.strip()}")
    for (expression, object_id), actual in zip(expected.items(), entries, strict=True):
        if actual != object_id:
            raise ValueError(f"recorded distribution input differs: {expression}")
    return len(inputs)


def check_distribution_revision(root, candidate, revision, historical=()):
    current = subprocess.run(
        ["git", "rev-parse", "--verify", "--end-of-options", f"{revision}^{{commit}}"],
        cwd=root, capture_output=True, text=True,
    )
    if current.returncode:
        raise ValueError(f"unable to resolve distribution comparison revision: {revision}")
    commit = current.stdout.strip()
    mismatches = []
    for record in (candidate, *historical):
        paths = record["matching_distribution_input_git_objects"]
        diff = subprocess.run(
            ["git", "diff", "--no-ext-diff", "--name-only", record["source_revision"], commit, "--", *paths],
            cwd=root, capture_output=True, text=True,
        )
        if diff.returncode:
            raise ValueError(f"unable to compare distribution inputs at {commit}: {diff.stderr.strip()}")
        if not diff.stdout:
            return commit
        mismatches.append(diff.stdout.splitlines())
    raise ValueError(f"distribution inputs changed at {commit}: {', '.join(mismatches[0])}")


def check_package_records(root, candidate):
    evidence = candidate["evidence"]
    hosted, a100, serving = (
        json.loads((root / evidence[name]["path"]).read_bytes(), object_pairs_hook=unique_object)
        for name in ("hosted_cpu", "a100_cpu", "a100_serving")
    )
    revision = candidate["source_revision"]
    if any(report["status"] != "passed" or report["source_revision"] != revision
           for report in (hosted, a100, serving)):
        raise ValueError("follow-up package reports must pass for the same committed source revision")
    binary = "/opt/riley/bin/riley"
    if (serving["image_id"] != a100["image_id"] or serving["riley_binary_sha256"] != a100["binaries"][binary]
            or serving["source_archive_sha256"] != a100["source_archive_sha256"]
            or serving["cpu_report_sha256"] != evidence["a100_cpu"]["sha256"]):
        raise ValueError("follow-up GPU serving is not bound to the A100 package")
    scope_paths = candidate["matching_distribution_input_git_objects"].keys() - {".github/workflows/distribution-image.yml"}
    objects = gzip.decompress((root / evidence["a100_source_objects"]["path"]).read_bytes())
    actual = subprocess.run(["git", "ls-tree", "-r", revision, "--", *sorted(scope_paths)],
                            cwd=root, capture_output=True, check=True).stdout
    inventory = candidate.get("a100_source_inventory", serving)
    if (objects != actual or sha256(objects).hexdigest() != inventory["source_objects_sha256"]
            or len(objects.splitlines()) != inventory["source_object_count"]):
        raise ValueError("follow-up A100 build inputs differ from the committed Git objects")

    runtime = {}
    for report, name in ((hosted, "hosted_runtime"), (a100, "a100_runtime")):
        lines = (root / evidence[name]["path"]).read_text(encoding="utf-8").splitlines()
        files = {path: digest for digest, path in (line.split("  ", 1) for line in lines)}
        if len(files) != len(lines) or any(files.get(path) != digest for path, digest in report["binaries"].items()):
            raise ValueError(f"follow-up runtime file digests differ from {name}")
        runtime[name] = files
    hosted_files = runtime["hosted_runtime"]
    a100_files = runtime["a100_runtime"]
    if hosted_files.keys() != a100_files.keys():
        raise ValueError("follow-up runtime file inventories differ")
    differences = sorted(path for path, digest in a100_files.items() if digest != hosted_files[path])
    if differences != candidate["differing_runtime_files"]:
        raise ValueError("follow-up runtime file comparison drifted")
    return a100, serving, len(a100_files)


def check_package_follow_up(root, candidate):
    evidence = candidate["evidence"]
    a100, serving, _ = check_package_records(root, candidate)
    smoke = serving["smoke"]
    if (serving["api_checks"] != 40 or serving["shutdown"] != "passed"
            or smoke["seconds"] != 45 or smoke["rate"] != 8 or smoke["problems"] != 0
            or smoke["completed"] <= 0 or smoke["client_disconnects"] <= 0
            or smoke["client_disconnects"] != smoke["server_cancellations"]):
        raise ValueError("follow-up package lacks passing A100 inference, cancellation and shutdown evidence")
    soaks = json.loads((root / evidence["a100_soaks"]["path"]).read_bytes(), object_pairs_hook=unique_object)
    check_package_soaks(soaks, serving)
    check_latest_allocator(root, candidate["allocator_follow_up"], evidence["a100_cpu"], a100, serving)


def check_prefix_metrics_package(root, candidate):
    a100, serving, runtime_files = check_package_records(root, candidate)
    if serving["matching_runtime_files"] != runtime_files or runtime_files != 21 or candidate["differing_runtime_files"]:
        raise ValueError("prefix metrics package must reproduce all 21 hosted runtime files")
    check_prefix_metrics_serving(serving)
    if "allocator_experiment" in candidate:
        allocator = check_prefix_metrics_allocator(root, candidate, a100, serving)
        if "recovery_drill" in candidate:
            entry = candidate["recovery_drill"]["report"]
            drill = json.loads((root / entry["path"]).read_bytes(), object_pairs_hook=unique_object)
            check_prefix_metrics_recovery(drill, serving, allocator)
    elif "recovery_drill" in candidate:
        raise ValueError("the recovery drill requires a verified allocator experiment")


def check_structured_serving_package(root, candidate):
    parent, smoke, runtime_files = check_package_records(root, candidate)
    evidence = candidate["evidence"]
    structured, verified = (
        json.loads((root / evidence[name]["path"]).read_bytes(), object_pairs_hook=unique_object)
        for name in ("a100_structured", "a100_verified")
    )
    scopes = ("source_revision", "image_id", "source_archive_sha256", "riley_binary_sha256",
              "cpu_report_sha256", "checkpoint_sha256")
    if (runtime_files != 21 or candidate["differing_runtime_files"]
            or verified["status"] != "verified"
            or verified["matching_runtime_files"] != runtime_files
            or verified["a100_image"] != smoke["image_id"]
            or any(structured[key] != smoke[key] for key in scopes)
            or verified["source_revision"] != smoke["source_revision"]
            or verified["riley_binary_sha256"] != smoke["riley_binary_sha256"]):
        raise ValueError("structured serving checks must use the reproduced package and checkpoint")
    if any(row["status"] != "passed" or row["api_checks"] != 40 or row["shutdown_exit"] != 0
           or row["tool_checks_enabled"] is not True
           for row in (structured, smoke)):
        raise ValueError("structured serving package must pass the two GPU API and shutdown checks")
    mixed = smoke["serving_smoke"]
    if (mixed["status"] != "passed" or (mixed["seconds"], mixed["rate"]) != (45, 8)
            or mixed["failed_checks"] or not mixed["checks"] or not all(mixed["checks"].values())
            or smoke["prefix_cache_metrics"]["status"] != "passed"):
        raise ValueError("structured serving package must pass the 45-second GPU mixed load")
    suites = structured["accuracy_suites"]
    holdout = suites["structured_holdout"]
    counts = {"c1": 16, "c8": 16, "c32": 32}
    if (holdout["status"] != "passed" or holdout["c1_transport_pairs"] != 8
            or holdout["context_rejection"]["http_status"] != 400
            or {name: row["requests"] for name, row in holdout["modes"].items()} != counts
            or any(holdout["modes"][name]["exact_matches"] != counts[name] for name in ("c8", "c32"))):
        raise ValueError("structured serving package lacks the 64-request holdout")
    if len(holdout["cases"]) != 16:
        raise ValueError("structured serving package lacks the 16 structured reference responses")
    reservations = verified["reservations"]
    if set(reservations) != {"6", "7"} or any(row["allocation_bytes"] != 20 * 2**30
                                              for row in reservations.values()):
        raise ValueError("structured serving package lacks the two 20 GiB reservation receipts")
    stock = {
        **{key: smoke[key] for key in ("source_revision", "riley_binary_sha256", "checkpoint_sha256")},
        "reservations": {gpu: {"pid": row["pid"], "start_ticks": row["start_ticks"], "gib": 20}
                         for gpu, row in reservations.items()},
        "accuracy": {"suites": suites},
    }
    return parent, smoke, stock


def check_pattern_fix_package(root, candidate):
    parent, smoke, stock = check_structured_serving_package(root, candidate)
    if candidate["allocator_experiment"]["matched_structured_cases"] != 16:
        raise ValueError("pattern-fix allocator must preserve all 16 structured reference responses")
    allocator = check_prefix_metrics_allocator(root, candidate, parent, stock)
    if "recovery_drill" in candidate:
        entry = candidate["recovery_drill"]["report"]
        drill = json.loads((root / entry["path"]).read_bytes(), object_pairs_hook=unique_object)
        check_prefix_metrics_recovery(drill, smoke, allocator)
    if "stock_soak_comparison" in candidate:
        check_stock_soak_comparison(root, candidate, smoke, allocator)


def check_pattern_class_probes(root, candidate):
    evidence = candidate["evidence"]
    report, summary = (
        json.loads((root / evidence[name]["path"]).read_bytes(), object_pairs_hook=unique_object)
        for name in ("a100_pattern_classes", "a100_verified")
    )
    if report["status"] != "passed" or report["checks"] != summary["pattern_class_checks"]:
        raise ValueError("character class HTTP probes differ from the A100 package summary")

    success = {"quote-range": '"', "backslash-range": "\\"}
    rejected = ("unsupported-class-escape", "unsupported-unicode-escape", "unsupported-negated-space")
    expected = {}
    for parameter, finish in (("response_format", "stop"), ("tools", "tool_calls")):
        for case, value in success.items():
            expected[case, parameter] = (200, {"value": value}, finish, None)
        for case in rejected:
            expected[case, parameter] = (400, None, None, "invalid_schema")
    checks = report["checks"]
    observed = {
        (row["case"], row["parameter"]): (row["status"], row.get("output"), row.get("finish_reason"), row.get("code"))
        for row in checks
    }
    if len(checks) != len(expected) or observed != expected:
        raise ValueError("character class HTTP probes must prove escaped values and named schema rejections")


def check_schema_reference_probes(root, candidate, *, uri_fragments=False):
    evidence = candidate["evidence"]
    report, summary = (
        json.loads((root / evidence[name]["path"]).read_bytes(), object_pairs_hook=unique_object)
        for name in ("a100_schema_references", "a100_verified")
    )
    checks = report["checks"]
    if report["status"] != "passed" or checks != summary["schema_reference_checks"]:
        raise ValueError("schema reference HTTP probes differ from the A100 package summary")
    outputs = {"root-id-local-pointer": {"$id": 3}}
    embedded_id = "unsupported JSON Schema: $ref crossing an embedded $id"
    errors = {"nested-id": "unsupported JSON Schema: embedded $id", "cross-resource-pointer": embedded_id}
    if uri_fragments:
        outputs.update({"encoded-space": {"value": 1}, "encoded-unicode-and-pointer-escapes": {"value": 3}})
        invalid_encoding = "invalid percent encoding in local $ref"
        errors.update({"malformed-percent-escape": invalid_encoding, "invalid-percent-encoded-utf8": invalid_encoding,
                       "encoded-cross-resource-pointer": embedded_id})
    expected = {}
    for parameter in ("response_format", "tools"):
        for case, value in outputs.items():
            expected[case, parameter] = (200, value, None)
        for case, message in errors.items():
            expected[case, parameter] = (400, None, {
                "type": "invalid_request_error", "param": parameter, "code": "invalid_schema",
                "message": message,
            })
    observed = {(row["case"], row["parameter"]): (row["status"], row.get("output"), row.get("error"))
                for row in checks}
    if len(checks) != len(expected) or observed != expected:
        if uri_fragments:
            raise ValueError("schema reference probes must prove URI fragment decoding and named rejections")
        raise ValueError("schema reference probes must preserve root-local output and reject embedded IDs")


def check_pattern_anchor_probes(root, candidate):
    evidence = candidate["evidence"]
    report, summary = (
        json.loads((root / evidence[name]["path"]).read_bytes(), object_pairs_hook=unique_object)
        for name in ("a100_pattern_anchors", "a100_verified")
    )
    checks = report["checks"]
    if report["status"] != "passed" or checks != summary["pattern_anchor_checks"]:
        raise ValueError("pattern anchor HTTP probes differ from the A100 package summary")
    outputs = {
        "start-branch": ("^cat|dog$", "catfish"),
        "end-branch": ("^cat|dog$", "sheepdog"),
        "unanchored-middle-branch": ("^cat$|dog|^eel$", "hotdogs"),
        "escaped-dollar-and-class": (r"^(cat|dog)$|\$[|]", "a$|b"),
    }
    expected = {}
    for parameter in ("response_format", "tools"):
        for case, (pattern, value) in outputs.items():
            expected[case, parameter] = (pattern, 200, {"value": value}, None)
        expected["unsupported-interior-anchor", parameter] = ("ca$t|dog", 400, None, {
            "type": "invalid_request_error", "param": parameter, "code": "invalid_schema",
            "message": "unsupported JSON Schema: pattern operator $ in this position",
        })
    observed = {
        (row["case"], row["parameter"]): (row["pattern"], row["status"], row.get("output"), row.get("error"))
        for row in checks
    }
    if len(checks) != len(expected) or observed != expected:
        raise ValueError("pattern anchor probes must prove matches and named schema rejections")


def check_stock_soak_comparison(root, candidate, smoke, allocator):
    entry = candidate["stock_soak_comparison"]["report"]
    report = json.loads((root / entry["path"]).read_bytes(), object_pairs_hook=unique_object)
    parent_sha = candidate["evidence"]["a100_cpu"]["sha256"]
    allocator_sha = candidate["allocator_experiment"]["evidence"]["serving"]["sha256"]
    if (report["status"] != "verified" or report["parent_cpu_report_sha256"] != parent_sha
            or report["allocator_report_sha256"] != allocator_sha
            or report["reservations"] != allocator["reservations"]):
        raise ValueError("stock soak comparison is not bound to the current package and allocator run")
    stock = report["stock_soaks"]
    check_package_soaks(stock, smoke)
    comparisons = report["comparisons"]
    if comparisons.keys() != stock["modes"].keys():
        raise ValueError("stock soak comparison must cover both cache settings")
    for mode, comparison in comparisons.items():
        baseline = stock["modes"][mode]
        trial = allocator["soaks"]["modes"][mode]
        before, after = (arm["rss"]["rss_growth_kib"] for arm in (baseline, trial))
        delta = after - before if before is not None and after is not None else None
        if comparison["stock_to_allocator_rss_change_kib"] != delta:
            raise ValueError(f"{mode} allocator RSS difference changed")
        for label, arm in (("stock", baseline), ("allocator", trial)):
            counts = comparison[f"{label}_counts"]
            windows = comparison[f"{label}_latency_windows"]
            if (sum(row.get("completed", 0) for row in counts.values()) != arm["completed"]
                    or counts["cancel"].get("disconnected", 0) != arm["client_disconnects"]):
                raise ValueError(f"{mode} {label} comparison outcome counts changed")
            for metric, measured in arm["p95_change_percent"].items():
                early, late = (windows[period][f"{metric}_p95"] for period in ("early", "late"))
                change = (late / early - 1) * 100 if early and late else None
                if change != measured:
                    raise ValueError(f"{mode} {label} comparison latency windows changed")


def check_prefix_metrics_serving(serving):
    accuracy = serving["accuracy"]
    suites = accuracy["suites"]
    structured = suites["structured_holdout"]
    counts = {"c1": 16, "c8": 16, "c32": 32}
    if (accuracy["api_checks"] != 40 or accuracy["shutdown_exit"] != 0
            or suites["baseline"]["verified_requests"] != 144 or suites["holdout"]["verified_requests"] != 88
            or structured["status"] != "passed" or structured["c1_transport_pairs"] != 8
            or {name: row["requests"] for name, row in structured["modes"].items()} != counts
            or any(structured["modes"][name]["exact_matches"] != counts[name] for name in ("c8", "c32"))
            or structured["context_rejection"]["http_status"] != 400):
        raise ValueError("prefix metrics package lacks exact long and structured output evidence")
    check_package_soaks(serving["soaks"], serving)
    for mode, soak in serving["soaks"]["modes"].items():
        prefix = soak["prefix_cache_metrics"]
        capacity = 128 if mode == "cache-on" else 0
        passed = prefix["status"] == "passed"
        if (prefix["status"] not in ("passed", "failed")
                or passed != ("current_prefix_cache_metrics" not in soak["failed_checks"])):
            raise ValueError(f"{mode} cache metric verdict differs from the soak verdict")
        if passed and (prefix["capacity_blocks"] != capacity or prefix["snapshots"] < 3
                       or not 0 <= prefix["peak_sampled_retained_blocks"] <= capacity
                       or not 0 <= prefix["hit_tokens_delta"] <= prefix["shutdown_hit_tokens"]
                       or not 0 <= prefix["evicted_blocks_delta"] <= prefix["shutdown_evicted_blocks"]
                       or prefix["hit_tokens_delta"] % 16 != 0 or prefix["shutdown_hit_tokens"] % 16 != 0
                       or (capacity and prefix["hit_tokens_delta"] == 0)
                       or (not capacity and (prefix["shutdown_hit_tokens"] or prefix["shutdown_evicted_blocks"]))):
            raise ValueError(f"{mode} cache metric evidence exceeds its capacity or workload")


def check_package_soaks(soaks, serving):
    if (any(soaks[key] != serving[key] for key in ("source_revision", "image_id", "checkpoint_sha256"))
            or soaks["binary_sha256"] != serving["riley_binary_sha256"]
            or (soaks["duration_seconds_per_mode"], soaks["requests_per_second_per_mode"]) != (1800, 8)):
        raise ValueError("30-minute soaks do not match the verified follow-up package and workload")
    modes = soaks["modes"]
    if set(modes) != {"cache-off", "cache-on"}:
        raise ValueError("30-minute soaks must cover both cache settings")
    for mode, result in modes.items():
        failures = result["failed_checks"]
        rss = result["rss"]
        growth = rss["rss_growth_kib"]
        baseline, drain = rss["baseline"], rss["drain"]
        measured_growth = drain["rss_kib"] - baseline["rss_kib"] if baseline is not None and drain is not None else None
        windows = result["p95_change_percent"]
        if windows.keys() != {"ttft_ms", "e2e_ms"}:
            raise ValueError(f"{mode} TTFT and end-to-end measurements are required")
        rss_failed = growth is None or growth > 0
        latency_failed = any(value is None or not isfinite(value) or value > 5 for value in windows.values())
        if (growth != measured_growth
                or rss_failed != ("rss_growth_second_half_at_most_zero" in failures)
                or latency_failed != ("p95_drift_at_most_5_percent" in failures)
                or result["status"] != ("failed" if failures else "passed")):
            raise ValueError(f"{mode} soak verdict differs from its measurements")
    expected_gate = "passed" if all(result["status"] == "passed" for result in modes.values()) else "failed"
    if soaks["release_gate"] != expected_gate:
        raise ValueError("30-minute soak gate differs from the two cache outcomes")


def check_allocator_parent(package, parent, parent_report_sha256):
    if (package["status"] != "passed" or package["parent_image_id"] != parent["image_id"]
            or package["parent_cpu_report_sha256"] != parent_report_sha256
            or package["same_parent_runtime_files"] != 21
            or any(package[key] != parent[key] for key in ("source_revision", "source_archive_sha256", "binaries"))):
        raise ValueError("latest allocator package does not match its verified parent")


def check_prefix_metrics_allocator(root, candidate, parent, stock):
    trial = candidate["allocator_experiment"]
    records = trial["evidence"]
    package, serving = (json.loads((root / records[key]["path"]).read_bytes(), object_pairs_hook=unique_object)
                        for key in ("package", "serving"))
    parent_sha = candidate["evidence"]["a100_cpu"]["sha256"]
    check_allocator_parent(package, parent, parent_sha)
    library = package["added_files_sha256"]["/opt/riley/lib/libjemalloc.so.2"]
    if (serving["status"] != "passed" or serving["image_id"] != package["image_id"]
            or serving["parent_cpu_report_sha256"] != parent_sha
            or serving["allocator_cpu_report_sha256"] != records["package"]["sha256"]
            or serving["allocator_library_sha256"] != library or serving["allocator_settings"] != trial["malloc_conf"]
            or serving["parent_image_id"] != parent["image_id"]
            or any(serving[key] != stock[key] for key in ("source_revision", "riley_binary_sha256", "checkpoint_sha256", "reservations"))):
        raise ValueError("prefix metrics allocator experiment used a different parent, settings, or runtime")
    proofs = serving["allocator_process_proof_sha256"]
    if proofs.keys() != {"cache-off", "cache-on", "accuracy"} or any(
            not re.fullmatch(r"[0-9a-f]{64}", value) for value in proofs.values()):
        raise ValueError("prefix metrics allocator experiment lacks live process evidence")
    check_prefix_metrics_serving(serving)
    original = stock["accuracy"]["suites"]["structured_holdout"]["cases"]
    observed = serving["accuracy"]["suites"]["structured_holdout"]["cases"]
    matches = sum(observed.get(name) == case for name, case in original.items())
    if len(original) != 16 or observed.keys() != original.keys() or trial["matched_structured_cases"] != matches:
        raise ValueError("allocator output comparison differs from its structured response receipts")
    return serving


def check_prefix_metrics_recovery(drill, stock, allocator):
    bindings = ("source_revision", "riley_binary_sha256", "checkpoint_sha256")
    if (drill["status"] != "passed" or drill["scope"] != "isolated loopback recovery"
            or drill["arms"].keys() != {"allocator", "stock"}
            or any(drill[key] != stock[key] or drill[key] != allocator[key] for key in bindings)
            or drill["reservations"] != allocator["reservations"]
            or drill["reservations"].keys() != {"6", "7"}
            or any(row["gib"] != 20 for row in drill["reservations"].values())):
        raise ValueError("recovery drill does not match the qualified candidates or GPU reservations")
    routes, streams = [], []
    for label, candidate in (("allocator", allocator), ("stock", stock)):
        arm = drill["arms"][label]
        route, probe, process = (arm[key] for key in ("route", "admission", "process"))
        shutdown, cache = arm["shutdown"], arm["prefix_cache"]
        if (arm["image_id"] != candidate["image_id"] or arm["image_id"] != process["image_id"] or arm["api_checks"] != 40
                or not re.fullmatch(r"[0-9a-f]{64}", arm["gpu_report_sha256"])
                or (label == "allocator" and (process["library_sha256"] != allocator["allocator_library_sha256"]
                                              or process["malloc_conf"] != allocator["allocator_settings"]))):
            raise ValueError(f"{label} recovery arm ran a different image or allocator")
        if (probe["rejected_http_status"] != 503 or probe["active_after_gate"]["active_requests"] != 1
                or probe["active_after_gate"]["waiting_requests"] != 0 or probe["active_after_gate"]["kv_allocated_blocks"] <= 0
                or probe["cached_prompt_tokens"] != probe["cache_hit_delta"] or probe["cache_hit_delta"] <= 0
                or probe["stream"]["completion_tokens"] <= 0
                or not route["closed_before_start_ns"] < route["ready_ns"] < probe["gate_closed_ns"]
                <= probe["active_after_gate_ns"] < probe["sse_completed_ns"] <= probe["drained_ns"]
                < route["stop_requested_ns"] < route["closed_after_cleanup_ns"]):
            raise ValueError(f"{label} recovery arm lacks an active drain after admission was blocked")
        gauges = ("active_requests", "waiting_requests", "kv_allocated_blocks")
        if (probe["drained"] != dict.fromkeys(gauges, 0) or any(shutdown[key] for key in gauges)
                or not shutdown["allocation"] or any(shutdown["allocation"].values())
                or cache["status"] != "passed" or cache["capacity_blocks"] != 128
                or shutdown["prefix_cache"]["retained_blocks"] != 0
                or any(shutdown["prefix_cache"][key] != cache[f"shutdown_{key}"] for key in ("hit_tokens", "evicted_blocks"))):
            raise ValueError(f"{label} recovery arm did not drain cache and allocations")
        routes.append(route)
        streams.append(probe["stream"])
    first, second = routes
    if (streams[0] != streams[1] or first["gateway"] == first["server"]
            or any(not re.fullmatch(r"127\.0\.0\.1:[1-9][0-9]{0,4}", first[key]) for key in ("gateway", "server"))
            or any(first[key] != second[key] for key in ("gateway", "server", "server_arguments"))
            or first["closed_after_cleanup_ns"] >= second["closed_before_start_ns"]
            or drill["stop_to_stock_ready_ms"] != (second["ready_ns"] - first["stop_requested_ns"]) / 1e6):
        raise ValueError("recovery did not restore the same stream and loopback addresses in order")


def check_latest_allocator(root, trial, parent_entry, parent, serving):
    records = trial["evidence"]
    reports = {kind: json.loads((root / entry["path"]).read_bytes(), object_pairs_hook=unique_object)
               for kind, entry in records.items()}
    package = reports["package"]
    soaks = reports["soaks"]
    check_allocator_parent(package, parent, parent_entry["sha256"])
    image = package["image_id"]
    library = package["added_files_sha256"]["/opt/riley/lib/libjemalloc.so.2"]
    if (soaks["parent_image_id"] != parent["image_id"] or soaks["allocator_cpu_sha256"] != records["package"]["sha256"]
            or soaks["allocator_library_sha256"] != library or soaks["allocator_settings"] != trial["malloc_conf"]):
        raise ValueError("latest allocator soaks used different package evidence or settings")
    check_package_soaks(soaks, {**serving, "image_id": image})

    for mode, result in soaks["modes"].items():
        if not re.fullmatch(r"[0-9a-f]{64}", result["allocator_process_proof_sha256"]):
            raise ValueError(f"{mode} lacks live allocator evidence")
    if (soaks["release_gate"] == "passed") != ("accuracy" in reports):
        raise ValueError("latest allocator accuracy must accompany passing soaks")
    if "accuracy" in reports:
        accuracy = reports["accuracy"]
        counts = {name: suite["verified_requests"] for name, suite in accuracy["accuracy_suites"].items()}
        if (accuracy["status"] != "passed" or accuracy["image_id"] != image
                or accuracy["source_revision"] != package["source_revision"]
                or accuracy["riley_binary_sha256"] != package["binaries"]["/opt/riley/bin/riley"]
                or counts != {"baseline": 144, "holdout": 88}):
            raise ValueError("latest allocator accuracy does not cover the passing soak image")
    check_latest_allocator_rag(reports["rag"], parent["image_id"], image, library, trial["malloc_conf"])
    cohorts = reports["rag_cohorts"]
    if (cohorts["source_receipt_sha256"] != records["rag"]["sha256"]
            or cohorts["cells"].keys() != reports["rag"]["cells"].keys()):
        raise ValueError("latest RAG cohort audit is not bound to the matched captures")
    check_latest_allocator_rag_replay(reports["rag_replay"], reports["rag"], records["rag"]["sha256"])


def check_latest_allocator_rag(report, parent, image, library, settings):
    order = ["parent-first", "packaged-first", "packaged-second", "parent-second"]
    arms = report["arms"]
    if report["order"] != order or set(arms) != set(order):
        raise ValueError("latest RAG comparison needs both image orders")
    arguments = arms[order[0]]["server"]["server_arguments"]
    for name, arm in arms.items():
        expected = parent if name.startswith("parent-") else image
        server = arm["server"]
        if (arm["api_checks"] != 40 or arm["image_id"] != expected or server["image_id"] != expected
                or server["server_arguments"] != arguments):
            raise ValueError(f"{name} RAG used different image evidence or server settings")
        allocator = server["allocator"]
        if expected == parent:
            if allocator != "glibc":
                raise ValueError(f"{name} RAG baseline allocator differs")
        elif (allocator["image_id"] != image or allocator["library_sha256"] != library
              or allocator["malloc_conf"] != settings):
            raise ValueError(f"{name} RAG used different allocator settings")

    cells = report["cells"]
    if set(cells) != {f"rag-c8-seed{seed}" for seed in (1, 2, 3)}:
        raise ValueError("latest RAG comparison needs all three seed workloads")
    for label, cell in cells.items():
        metrics = cell["metrics"]
        if (cell["aligned_samples_per_arm"] != 256 or set(metrics) != set(order)
                or set(cell["request_timing"]) != set(order)):
            raise ValueError(f"{label} RAG evidence is incomplete")
        if any(not isfinite(metrics[name][key]) or metrics[name][key] <= 0
               for name in order for key in ("ttft_p95_ms", "tpot_p50_ms")):
            raise ValueError(f"{label} RAG latency samples must be positive and finite")
        for pair in ("first", "second"):
            ttft = [metrics[f"{kind}-{pair}"]["ttft_p95_ms"] for kind in ("parent", "packaged")]
            change = (ttft[1] / ttft[0] - 1) * 100
            if not isclose(change, cell["change_percent_by_order"][pair]["ttft_p95_ms"], abs_tol=1e-9):
                raise ValueError(f"{label} RAG TTFT change differs from the paired measurements")
        tpot = [fmean(metrics[f"{kind}-{pair}"]["tpot_p50_ms"] for pair in ("first", "second"))
                for kind in ("parent", "packaged")]
        expected = (tpot[1] / tpot[0] - 1) * 100 < 1
        if cell["measured_regression_guards"]["decode_tpot_below_one_percent_regression"] is not expected:
            raise ValueError(f"{label} RAG decode guard differs from the measured regression")
    updates = {
        "all_tested_streams_match": all(cell["same_complete_streams"] for cell in cells.values()),
        "all_measured_regression_guards_pass": all(
            all(cell["measured_regression_guards"].values()) for cell in cells.values()),
    }
    if any(report[key] is not value for key, value in updates.items()):
        raise ValueError("latest RAG aggregate verdict differs from its cells")


def check_latest_allocator_rag_replay(report, closed_loop, closed_sha256):
    captures = closed_loop["cells"]["rag-c8-seed2"]["request_timing"]
    order = closed_loop["order"]
    if (report["reference_receipt_sha256"] != closed_sha256 or report["order"] != order
            or set(report["arms"]) != set(order) or set(report["schedules"]) != {"parent-first", "packaged-first"}):
        raise ValueError("fixed-arrival RAG does not match the closed-loop reference and image order")
    for name in order:
        arm = report["arms"][name]
        baseline = closed_loop["arms"][name]
        server = baseline["server"]
        holder = arm["reservations_after"]
        if (arm["image_id"] != baseline["image_id"] or arm["api_checks"] != 40
                or arm["server_arguments"] != server["server_arguments"]
                or holder["gib"] != 20 or holder != report["arms"][order[0]]["reservations_after"]):
            raise ValueError(f"{name} fixed-arrival RAG used different image, workload settings or reservation")
        allocator = arm["allocator"]
        expected = server["allocator"]
        if expected == "glibc":
            allocator_matches = allocator == "glibc"
        else:
            fields = ("image_id", "library_sha256", "malloc_conf", "mounts")
            allocator_matches = all(allocator[key] == expected[key] for key in fields)
        if not allocator_matches:
            raise ValueError(f"{name} fixed-arrival RAG used different allocator settings")

    for origin, schedule in report["schedules"].items():
        metrics = schedule["metrics"]
        if (schedule["source_sha256"] != captures[origin]["records_sha256"]
                or schedule["requests_per_arm"] != 256 or schedule["streams_match_original"] is not True
                or not isfinite(schedule["scheduled_span_seconds"]) or schedule["scheduled_span_seconds"] <= 0
                or set(metrics) != set(order) or set(schedule["paired_deltas"]) != {"first", "second"}):
            raise ValueError(f"{origin} fixed-arrival RAG is missing a matched schedule or response")
        write_errors = [metrics[name]["write_error_abs_p95_ms"] for name in order]
        for name, error in zip(order, write_errors, strict=True):
            values = metrics[name]
            if (not all(isfinite(values[key]) and values[key] > 0 for key in (
                    "ttft_p95_ms", "tpot_p50_ms", "long_prompt_ttft_p95_ms", "write_to_first_text_p95_ms"))
                    or not 0 <= error <= values["write_error_abs_max_ms"] < float("inf")):
                raise ValueError(f"{origin} {name} fixed-arrival RAG timing is invalid")
        if schedule["all_p95_relative_write_errors_at_most_5ms"] is not all(error <= 5 for error in write_errors):
            raise ValueError(f"{origin} fixed-arrival RAG write precision differs from the measurements")
        for pair, changes in schedule["paired_deltas"].items():
            left, right = (metrics[f"{kind}-{pair}"] for kind in ("parent", "packaged"))
            expected = {
                "ttft_p95_change_percent": (right["ttft_p95_ms"] / left["ttft_p95_ms"] - 1) * 100,
                "write_to_first_text_p95_difference_ms":
                    right["write_to_first_text_p95_ms"] - left["write_to_first_text_p95_ms"],
                "long_prompt_ttft_p95_difference_ms":
                    right["long_prompt_ttft_p95_ms"] - left["long_prompt_ttft_p95_ms"],
            }
            if any(not isclose(changes[key], value, abs_tol=1e-9) for key, value in expected.items()):
                raise ValueError(f"{origin} {pair} fixed-arrival RAG latency differs from the paired measurements")
        averages = [fmean(metrics[f"{kind}-{pair}"]["ttft_p95_ms"] for pair in ("first", "second"))
                    for kind in ("parent", "packaged")]
        if not isclose(schedule["ttft_p95_change_percent_by_arm_means"],
                       (averages[1] / averages[0] - 1) * 100, abs_tol=1e-9):
            raise ValueError(f"{origin} fixed-arrival RAG mean TTFT differs from the measurements")


def check_distribution_serving(root, candidate):
    records = candidate["serving_reports"]
    reports = {role: json.loads((root / entry["path"]).read_bytes(), object_pairs_hook=unique_object)
               for role, entry in records.items()}
    image = candidate["a100_rebuild_gpu_inference"]["image_id"]
    for role, report in reports.items():
        binary_key = "riley_binary_sha256" if role in ("mixed_stage", "tp2") else "binary_sha256"
        if report["image_id"] != image or report[binary_key] != candidate["binary_sha256"]:
            raise ValueError(f"distribution serving report {role} belongs to a different image or binary")

    recheck = reports["mha_recheck"]
    mha = "SmolLM2-1.7B-Instruct"
    if (recheck["model"] != mha or recheck["prior_result_sha256"] != records["extended"]["sha256"]
            or recheck["api_checks_passed"] != recheck["api_checks"] or recheck["api_failed_check_names"]
            or recheck["smoke"]["expect_tools_unsupported"] is not True
            or recheck["smoke"]["counts"]["tools_unsupported"]["unsupported"] <= 0):
        raise ValueError("distribution MHA report does not verify the template rejection and original run")

    smoke_runs = {
        **reports["representative"]["results"],
        "Qwen2.5-14B-Instruct": reports["extended"]["results"]["Qwen2.5-14B-Instruct"],
        mha: recheck,
    }
    if smoke_runs.keys() != {"Qwen2.5-3B-Instruct", "Qwen3-4B", "Qwen2.5-14B-Instruct", mha}:
        raise ValueError("distribution smoke does not cover all four representative checkpoints")
    expected = {}
    for model, run in smoke_runs.items():
        smoke = run["smoke"]
        if (run["status"] != "passed" or run["exit"] != 0 or smoke["status"] != "passed"
                or smoke["failed_checks"] or run["api_checks"] != (39 if model == mha else 40)):
            raise ValueError(f"distribution smoke for {model} did not pass the API and mixed load")
        counts = smoke["counts"]
        completed = sum(category.get("completed", 0) for category in counts.values())
        disconnects = counts["cancel"]["disconnected"]
        detail = f'{smoke["seconds"]:g} s mixed load, {completed} completions, {disconnects} disconnects'
        if model == mha:
            detail += f', {counts["tools_unsupported"]["unsupported"]} expected tool rejections'
        expected[model] = ("1", str(run["api_checks"]), detail)

    mixed = reports["mixed_stage"]
    trials = mixed["mixed"]["trials"]
    if (mixed["api_checks"] != 40 or mixed["mixed"]["status"] != "passed"
            or len(trials) != 3 or any(trial["matching_responses"] != 2 or not trial["witnesses"] for trial in trials)):
        raise ValueError("distribution mixed-stage report lacks the matched mixed GPU iterations")
    iterations = sum(len(trial["witnesses"]) for trial in trials)
    matches = sum(trial["matching_responses"] for trial in trials)
    expected["Qwen2.5-7B-Instruct"] = ("1", str(mixed["api_checks"]),
                                       f"{iterations} committed mixed GPU iterations, {matches} exact responses")

    tp = reports["tp2"]
    holdout = tp["tensor_parallel"]["holdout"]
    modes = holdout["modes"]
    c8, c32 = modes["c8"], modes["c32"]
    if (tp["status"] != "passed" or tp["api_checks"] != 40 or holdout["status"] != "passed"
            or len(set(tp["tensor_parallel"]["gpu_uuids"])) != 2 or modes["c1"]["requests"] != 11
            or c8["exact_matches"] != c8["requests"] or c8["requests"] != 11
            or c32["exact_matches"] != c32["requests"] or c32["requests"] != 32
            or not 1 < c8["observed_peak_active"] <= 8 or not 1 < c32["observed_peak_active"] <= 32):
        raise ValueError("distribution TP=2 report lacks exact long-context matches")
    expected["Qwen2.5-32B-Instruct"] = ("2", str(tp["api_checks"]),
                                      f'{c8["exact_matches"] + c32["exact_matches"]} exact concurrent responses, '
                                      f'peak {c32["observed_peak_active"]} active with 32 clients')

    support = root / "docs/model-support.md"
    image_label = f'A100 image `{image.removeprefix("sha256:")[:12]}` (source `{candidate["source_revision"][:9]}`)'
    if image_label not in support.read_text(encoding="utf-8"):
        raise ValueError("model-support.md: serving image and source differ from the recorded A100 candidate")
    documented = markdown_table(support, ("Checkpoint", "A100 GPUs", "API checks", "Serving exercise"),
                                "## Packaged serving checks")
    if documented != expected:
        rows = sorted(name for name in documented.keys() | expected.keys() if documented.get(name) != expected.get(name))
        raise ValueError(f"model-support.md: packaged serving coverage differs for {', '.join(rows)}")
    return len(expected)


def check_allocator_trial(root, trial, distribution):
    records = trial["evidence"]
    reports = {kind: json.loads((root / entry["path"]).read_bytes(), object_pairs_hook=unique_object)
               for kind, entry in records.items()}
    package = reports["package"]
    soak = reports["soak"]
    accuracy = reports["accuracy"]
    performance = reports["performance"]
    confirmation = reports["rag_confirmation"]
    replay = reports["rag_replay"]
    image = trial["image_id"]
    parent = distribution["a100_rebuild_gpu_inference"]["image_id"]
    if (package["status"] != "passed" or package["image_id"] != image or package["parent_image_id"] != parent
            or package["binaries"]["/opt/riley/bin/riley"] != distribution["binary_sha256"]
            or any(package[key] != distribution[key] for key in ("source_revision", "source_archive_sha256"))):
        raise ValueError("allocator package is not linked to the A100 parent and its binary")
    binding = soak["image_binding"]
    if (binding != accuracy["image_binding"] or binding["status"] != "passed"
            or any(binding[key] != package[key] for key in ("image_id", "parent_image_id", "source_revision",
                                                            "source_archive_sha256", "binaries"))
            or binding["experimental_packaged_cpu_attestation_sha256"] != records["package"]["sha256"]
            or soak["gpu_checker_input_sha256"] != accuracy["gpu_checker_input_sha256"]):
        raise ValueError("allocator package, soak and accuracy attestations differ")
    combined = {**soak, "results": {**soak["results"], "accuracy": accuracy["result"]}}
    encoded = (json.dumps(combined, indent=2, allow_nan=False) + "\n").encode("utf-8")
    if sha256(encoded).hexdigest() != accuracy["combined_report_sha256"]:
        raise ValueError("allocator accuracy is not part of the recorded combined GPU report")

    library = package["added_files_sha256"]["/opt/riley/lib/libjemalloc.so.2"]
    for scope, result in combined["results"].items():
        process = result["process_allocator"]
        if (result["status"] != "passed" or result["api_checks"] != 40 or process["image_id"] != image
                or process["library_sha256"] != library or process["malloc_conf"] != trial["malloc_conf"]):
            raise ValueError(f"allocator {scope} did not use the tested image and settings")
    if set(soak["results"]) != {"cache-on", "cache-off"}:
        raise ValueError("allocator trial needs both cache modes")
    for mode, result in soak["results"].items():
        verified = result["verified_soak"]
        rss = result["mixed_soak"]["rss"]
        if (verified["duration_seconds"] != 1800 or verified["problem_count"] != 0
                or verified["checks"]["rss_growth_second_half_at_most_zero"] is not True
                or rss["rss_growth_kib"] != rss["drain"]["rss_kib"] - rss["baseline"]["rss_kib"]
                or rss["rss_growth_kib"] > 0):
            raise ValueError(f"allocator {mode} lacks a passing 30-minute RSS result")
    counts = {name: suite["verified_requests"] for name, suite in accuracy["result"]["accuracy_suites"].items()}
    if counts != trial["accuracy_requests"]:
        raise ValueError("allocator accuracy coverage differs from the recorded qualification scope")

    order = ["parent-first", "packaged-first", "packaged-second", "parent-second"]
    if any(run["order"] != order or set(run["arms"]) != set(order)
           for run in (performance, confirmation, replay)):
        raise ValueError("allocator timing comparison arms differ")
    server_args = performance["arms"][order[0]]["server"]["server_arguments"]
    for name in order:
        closed = [run["arms"][name] for run in (performance, confirmation)]
        servers = [arm["server"] for arm in closed] + [replay["arms"][name]]
        expected = parent if name.startswith("parent-") else image
        if (any(arm["api_checks"] != 40 or arm["image_id"] != expected for arm in closed)
                or any(server["image_id"] != expected or server["server_arguments"] != server_args
                       for server in servers)):
            raise ValueError(f"allocator {name} timing reports use a different image or serving configuration")
        for server in servers:
            allocator = server["allocator"]
            if expected == parent:
                if allocator != "glibc":
                    raise ValueError(f"allocator {name} parent is not running glibc")
            elif (allocator["image_id"] != image or allocator["library_sha256"] != library
                  or allocator["malloc_conf"] != trial["malloc_conf"]):
                raise ValueError(f"allocator {name} timing reports use different allocator settings")

    if confirmation["prior_closed_loop_summary_sha256"] != records["performance"]["sha256"]:
        raise ValueError("allocator RAG confirmation uses a different prior performance report")
    rag_cells = confirmation["cells"]
    if (set(rag_cells) != {f"rag-c8-seed{seed}" for seed in (1, 2, 3)}
            or any(cell["aligned_samples_per_arm"] != 256 or cell["same_complete_streams"] is not True
                   or set(cell["request_timing"]) != set(order) for cell in rag_cells.values())):
        raise ValueError("allocator RAG confirmation does not cover the three matched 256-request cells")

    original = reports["metrics_observer"]
    warmed = reports["metrics_observer_warmed"]
    for record in (original, warmed):
        observation = record["result"]
        execution = record["execution"]
        raw = (json.dumps(observation, indent=2, allow_nan=False) + "\n").encode("utf-8")
        if (execution["campaign_exit"] != 0 or execution["physical_gpu"] != 6
                or execution["remote_verified_json_sha256"] != sha256(raw).hexdigest()
                or observation["api_checks"] != soak["results"]["cache-on"]["api_checks"]
                or observation["image_id"] != image
                or observation["candidate_binary_sha256"] != distribution["binary_sha256"]):
            raise ValueError("allocator metrics observation is not bound to the completed A100 image run")

    replay_result = warmed["result"]
    if (replay_result["matched_prior_report_sha256"] != original["execution"]["remote_verified_json_sha256"]
            or replay_result["checkpoint_sha256"] != original["result"]["checkpoint_sha256"]
            or replay_result["same_counts_per_phase"] != original["result"]["same_counts_per_phase"]):
        raise ValueError("allocator warmed metrics observation does not match the original run")
    return len(reports)


def prose_lines(path):
    fence = None
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        match = FENCE.match(line)
        if match:
            marker = match[1]
            if fence is None:
                fence = marker
            elif marker.startswith(fence) and not line[match.end():].strip():
                fence = None
            continue
        if fence is None:
            yield number, line


def markdown_anchors(path):
    counts = {}
    for _, line in prose_lines(path):
        match = ATX_HEADING.match(line)
        if match:
            text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", match[1])
            slug = re.sub(r"[^\w -]", "", text.lower()).replace(" ", "-")
            ordinal = counts.get(slug, 0)
            counts[slug] = ordinal + 1
            yield f"{slug}-{ordinal}" if ordinal else slug


def check_plan_links(root):
    count = 0
    anchors = {}
    for plan in sorted((root / PLANS).glob("*.md")):
        for number, line in prose_lines(plan):
            for href in INLINE_LINK.findall(line):
                destination = urlsplit(href.strip("<>"))
                if destination.scheme or destination.netloc:
                    continue
                target = (plan.parent / unquote(destination.path)).resolve() if destination.path else plan
                location = f"{plan.relative_to(root)}:{number}"
                if not target.is_relative_to(root) or not target.exists():
                    raise ValueError(f"{location}: unresolved local link {href}")
                if destination.fragment and target.suffix.lower() == ".md":
                    if target not in anchors:
                        anchors[target] = set(markdown_anchors(target))
                    if unquote(destination.fragment) not in anchors[target]:
                        raise ValueError(f"{location}: unresolved heading in {href}")
                count += 1
    if count == 0:
        raise ValueError("no local improvement-plan links checked")
    return count


def markdown_table(path, header, section=None):
    lines = iter(prose_lines(path))
    if section is not None and not any(line == section for _, line in lines):
        raise ValueError(f"{path.name}: missing section {section}")
    for _, line in lines:
        if line.startswith("## ") and section is not None:
            raise ValueError(f"{path.name}: missing table in {section}")
        if line.startswith("|") and tuple(cell.strip() for cell in line.strip("|").split("|")) == header:
            break
    else:
        raise ValueError(f"{path.name}: missing table {header[0]}")

    rows = {}
    for number, line in lines:
        if not line.startswith("|"):
            break
        fields = [field.strip() for field in line.strip("|").split("|")]
        if all(re.fullmatch(r":?-{3,}:?", field) for field in fields):
            continue
        if len(fields) != len(header) or not fields[0] or fields[0] in rows:
            raise ValueError(f"{path.name}:{number}: malformed or duplicate table row")
        rows[fields[0]] = tuple(value.replace("**", "") for value in fields[1:])
    return rows


def metric_pair(value):
    match = METRIC_PAIR.fullmatch(value)
    if match is None:
        raise ValueError(f"invalid October A100 metric pair: {value!r}")
    riley, vllm = map(Decimal, match.groups())
    if min(riley, vllm) <= 0:
        raise ValueError(f"nonpositive October A100 metric pair: {value!r}")
    return riley, vllm


def check_readme(root):
    readme = root / "README.md"
    with (root / "Cargo.toml").open("rb") as cargo:
        version = tomllib.load(cargo)["workspace"]["package"]["version"]
    status = re.findall(r"^> Status: (\S+) \(\d{4}-\d{2}-\d{2}\)\.",
                        readme.read_text(encoding="utf-8"), flags=re.MULTILINE)
    if status != [version]:
        raise ValueError(f"README.md: Status version must match Cargo workspace version {version}")

    public = markdown_table(readme,
                            ("Model", "Throughput tok/s", "Speed-up", "TTFT p50 ms", "TPOT p50 ms"))
    reference = root / "docs/model-coverage-a100-perf-table.md"
    columns = ("모델", "c1-fixed", "c1-natural", "c8-fixed", "c8-natural",
               "c16-fixed", "c16-natural", "c32-fixed", "c32-natural")
    sections = ("## 처리량 (출력 tok/s, 높을수록 좋음)", "## TTFT p50 (ms, 낮을수록 좋음)",
                "## TPOT p50 (ms, 낮을수록 좋음)")
    sources = [markdown_table(reference, columns, section) for section in sections]
    if len(public) != 24 or any(source.keys() != public.keys() for source in sources):
        raise ValueError("README.md: October A100 C32 fixed table must match all 24 reference models")
    for model, (throughput, ratio, ttft, tpot) in public.items():
        for name, value, source in zip(("throughput", "TTFT", "TPOT"), (throughput, ttft, tpot), sources, strict=True):
            expected = source[model][columns.index("c32-fixed") - 1]
            if value != expected:
                raise ValueError(f"README.md: {model} {name} differs from the October A100 C32 fixed table")
        riley, vllm = metric_pair(throughput)
        if ratio != f"{riley / vllm:.2f}×":
            raise ValueError(f"README.md: {model} speed-up differs from the displayed throughput")

    throughput, ttft, tpot = (
        [[metric_pair(value) for value in row] for row in source.values()] for source in sources
    )
    # The first two cells in each row are the fixed and natural single-client trials.
    c1 = [riley / vllm for row in throughput for riley, vllm in row[:2]]
    throughput_ratios = [riley / vllm for row in throughput for riley, vllm in row[2:]]
    ttft_ratios = [vllm / riley for row in ttft for riley, vllm in row]
    tpot_ratios = [vllm / riley for row in tpot for riley, vllm in row[2:]]
    if min(throughput_ratios) <= 1 or min(ttft_ratios) <= 1:
        raise ValueError("README.md: October A100 summary no longer shows a win in every throughput or TTFT cell")
    if not Decimal("0.97") <= min(c1) <= Decimal("1.03"):
        raise ValueError("README.md: October A100 single-client throughput minimum is outside the stated tie band")

    claims = {
        "**Throughput** at 8–32 concurrent clients": (
            f"higher in all {len(throughput_ratios)} cells — {min(throughput_ratios):.2f}× to "
            f"{max(throughput_ratios):.2f}×"
        ),
        "**Time to first token** (p50)": (
            f"lower in all {len(ttft_ratios)} cells — {min(ttft_ratios):.1f}× to {max(ttft_ratios):.1f}× faster"
        ),
        "**Time per output token** (p50) at 8–32 clients": (
            f"lower in {sum(ratio > 1 for ratio in tpot_ratios)} of {len(tpot_ratios)} cells"
        ),
        "One client": f"within ±3 % (tie) to {max(c1):.1f}× faster on throughput",
    }
    summary = markdown_table(readme, ("", "Riley vs vLLM"))
    for label, expected in claims.items():
        if summary.get(label) != (expected,):
            raise ValueError(f"README.md: October A100 summary differs for {label.replace('**', '')}")
    return len(public)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--distribution-revision", metavar="REV",
                        help="also require REV to match the recorded distribution image inputs")
    args = parser.parse_args()
    root = args.repo_root.resolve()
    try:
        manifest = json.loads((root / MANIFEST).read_bytes(), object_pairs_hook=unique_object)
        files = check_evidence(root, manifest)
        inputs = check_distribution_inputs(root, manifest["current_distribution_candidate"])
        historical = manifest["distribution_after_terminal_fix"]
        latest = manifest.get("distribution_with_prefix_metrics", historical)
        earlier = [historical]
        historical_inputs = check_distribution_inputs(root, historical)
        check_package_follow_up(root, historical)
        latest_inputs = historical_inputs
        if latest is not historical:
            latest_inputs = check_distribution_inputs(root, latest)
            check_prefix_metrics_package(root, latest)
        if "distribution_with_pattern_fix" in manifest:
            earlier.append(latest)
            latest = manifest["distribution_with_pattern_fix"]
            latest_inputs = check_distribution_inputs(root, latest)
            check_pattern_fix_package(root, latest)
        character_classes = manifest.get("distribution_with_character_class_fix")
        if character_classes is not None:
            inputs_character_classes = check_distribution_inputs(root, character_classes)
            check_pattern_class_probes(root, character_classes)
            if "allocator_experiment" in character_classes:
                check_pattern_fix_package(root, character_classes)
                if "stock_soak_comparison" in character_classes:
                    earlier.append(latest)
                    latest = character_classes
                    latest_inputs = inputs_character_classes
            else:
                if "stock_soak_comparison" in character_classes or "recovery_drill" in character_classes:
                    raise ValueError("character-class comparison or recovery requires an allocator experiment")
                check_structured_serving_package(root, character_classes)
        schema_references = manifest.get("distribution_with_schema_reference_fix")
        if schema_references is not None:
            schema_inputs = check_distribution_inputs(root, schema_references)
            check_pattern_class_probes(root, schema_references)
            check_schema_reference_probes(root, schema_references)
            if "allocator_experiment" in schema_references:
                check_pattern_fix_package(root, schema_references)
                if {"stock_soak_comparison", "recovery_drill"} <= schema_references.keys():
                    earlier.append(latest)
                    latest = schema_references
                    latest_inputs = schema_inputs
            else:
                if "stock_soak_comparison" in schema_references or "recovery_drill" in schema_references:
                    raise ValueError("schema reference comparison or recovery requires an allocator experiment")
                check_structured_serving_package(root, schema_references)
        parallel_shards = manifest.get("distribution_with_parallel_shard_verification")
        if parallel_shards is not None:
            earlier.append(latest)
            latest = parallel_shards
            latest_inputs = check_distribution_inputs(root, latest)
            if "allocator_experiment" in latest:
                check_pattern_fix_package(root, latest)
            else:
                if "stock_soak_comparison" in latest or "recovery_drill" in latest:
                    raise ValueError("parallel-shard comparison or recovery requires an allocator experiment")
                check_structured_serving_package(root, latest)
        uri_references = manifest.get("distribution_with_uri_reference_fix")
        if uri_references is not None:
            earlier.append(latest)
            latest = uri_references
            latest_inputs = check_distribution_inputs(root, latest)
            if "allocator_experiment" in latest:
                check_pattern_fix_package(root, latest)
            else:
                if "stock_soak_comparison" in latest or "recovery_drill" in latest:
                    raise ValueError("URI reference comparison or recovery requires an allocator experiment")
                check_structured_serving_package(root, latest)
            check_pattern_class_probes(root, latest)
            check_schema_reference_probes(root, latest, uri_fragments=True)
        pattern_anchors = manifest.get("distribution_with_pattern_anchor_fix")
        if pattern_anchors is not None:
            earlier.append(latest)
            latest = pattern_anchors
            latest_inputs = check_distribution_inputs(root, latest)
            if "allocator_experiment" in latest:
                check_pattern_fix_package(root, latest)
            else:
                if "stock_soak_comparison" in latest or "recovery_drill" in latest:
                    raise ValueError("pattern anchor comparison or recovery requires an allocator experiment")
                check_structured_serving_package(root, latest)
            check_pattern_class_probes(root, latest)
            check_schema_reference_probes(root, latest, uri_fragments=True)
            check_pattern_anchor_probes(root, latest)
        allocator_reports = check_allocator_trial(root, manifest["current_allocator_trial"],
                                                  manifest["current_distribution_candidate"])
        serving_models = check_distribution_serving(root, manifest["current_distribution_candidate"])
        links = check_plan_links(root)
        models = check_readme(root)
        revision = None
        if args.distribution_revision is not None:
            revision = check_distribution_revision(root, latest, args.distribution_revision, historical=earlier)
    except (OSError, EOFError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as error:
        print(f"improvement evidence check failed: {error}", file=sys.stderr)
        return 1
    print(f"improvement evidence verified: {files} file hashes, {inputs} baseline and {latest_inputs} latest Git inputs, "
          f"{links} local links, {models} README performance rows and summary, "
          f"{allocator_reports} linked allocator reports, {serving_models} packaged serving rows")
    if revision is not None:
        print(f"distribution source inputs match at {revision} ({latest_inputs} recorded paths)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
