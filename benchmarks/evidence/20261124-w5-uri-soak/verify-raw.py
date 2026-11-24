"""Rebind the completed W5 summary to its raw workload and recovery receipts."""

from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import re
import sys
import tarfile


def require(condition, message):
    if not condition:
        raise ValueError(message)


def member(archive, name):
    return archive.extractfile(name).read()


def read(archive, name):
    return json.loads(member(archive, name))


def digest(archive, name):
    return sha256(member(archive, name)).hexdigest()


def check_manifest(archive, filename):
    base = PurePosixPath(filename).parent
    seen = set()
    for line in member(archive, filename).decode().splitlines():
        expected, relative = line.split("  ", 1)
        path = PurePosixPath(relative)
        require(not path.is_absolute() and ".." not in path.parts and path not in seen,
                f"invalid result path in {filename}: {path}")
        seen.add(path)
        require(digest(archive, str(base / path)) == expected, f"raw W5 digest differs: {base / path}")
    require(bool(seen), f"empty W5 manifest: {filename}")


def check_order(archive, registration):
    text = member(archive, "campaign.log").decode()
    events = re.findall(r"^(START|END) (stock|allocator) (cache-off|cache-on|accuracy) ([67]) "
                        r"(\S+?)(?: exit=(\d+))?$", text, re.MULTILINE)
    for gpu in ("6", "7"):
        expected = [tuple(row[gpu].split("/")) for row in registration["rounds"]]
        if gpu == "6":
            expected.append(("allocator", "accuracy"))
        actual = [(event, kind, mode) for event, kind, mode, index, _, _ in events if index == gpu]
        require(actual == [(event, kind, mode) for kind, mode in expected for event in ("START", "END")],
                f"GPU {gpu} did not follow the registered crossover order")
        for event, kind, mode, index, _, code in events:
            if index == gpu and event == "END":
                require(member(archive, f"{kind}/{mode}.exit").decode().strip() == code,
                        f"GPU {gpu}/{kind}/{mode} exit differs from its log")


def check_recovery(archive, allocator, parent):
    recovery = read(archive, "recovery/verified.json")
    require(recovery["status"] == "passed" and recovery["scope"] == "isolated loopback recovery"
            and recovery["source_revision"] == parent["source_revision"]
            and recovery["reservations"] == allocator["reservations"], "unbound recovery outcome")
    for image, expected_id in (("stock", parent["image_id"]), ("allocator", allocator["image_id"])):
        base = f"recovery/{image}"
        summary = recovery["arms"][image]
        require(summary["image_id"] == expected_id and summary["gpu_report_sha256"] == digest(archive, f"{base}/report.json"),
                f"recovery does not bind the verified GPU report: {image}")
        for key, name in (("route", f"{base}-route.json"), ("admission", f"{base}/admission.json"),
                          ("process", f"{base}/process-start.json"), ("shutdown", f"{base}/shutdown.json")):
            raw = read(archive, name)
            if key == "admission":
                raw = {name: value for name, value in raw.items() if name not in ("initial", "final")}
            require(summary[key] == raw, f"recovery {key} differs from the raw receipt: {image}")
    a, b = (recovery["arms"][label]["route"] for label in ("allocator", "stock"))
    elapsed = (b["ready_ns"] - a["stop_requested_ns"]) / 1e6
    require(elapsed > 0 and recovery["stop_to_stock_ready_ms"] == elapsed, "recovery duration differs from the route")
    return elapsed


def check_precise_rss(load, expected, name):
    first = next((row for row in load["rss_samples"] if row["at_seconds"] >= load["seconds"] / 2), {})
    baseline = first.get("smaps_rollup")
    drain = load["final"].get("smaps_rollup")
    for sample in (baseline, drain):
        if sample is not None:
            require(type(sample["rss_kib"]) is int and type(sample["anonymous_kib"]) is int
                    and 0 <= sample["anonymous_kib"] <= sample["rss_kib"], f"invalid precise RSS sample: {name}")
    delta = anonymous_delta = None
    if baseline is not None and drain is not None:
        delta = drain["rss_kib"] - baseline["rss_kib"]
        anonymous_delta = drain["anonymous_kib"] - baseline["anonymous_kib"]
    precise = {"baseline": baseline, "drain": drain, "rss_growth_kib": delta, "anonymous_growth_kib": anonymous_delta}
    passed = delta is not None and delta <= 0
    require(load["smaps_rollup_second_half"] == precise == expected
            and load["checks"]["rss_growth_second_half_at_most_zero"] is passed,
            f"W5 RSS verdict differs from the first second-half and drain samples: {name}")


def verify(archive, registration, parent_path):
    names = set()
    for entry in archive.getmembers():
        path = PurePosixPath(entry.name)
        require(entry.isfile() and entry.name == path.as_posix() and not path.is_absolute()
                and ".." not in path.parts and entry.name not in names,
                f"invalid or duplicate W5 archive entry: {entry.name}")
        names.add(entry.name)
    require(int(member(archive, "campaign.pid").decode().strip()) == registration["campaign_pid"]
            and member(archive, "campaign.started").decode().strip() == registration["campaign_started_at"],
            "W5 campaign identity differs from registration")
    parent = json.loads(parent_path.read_bytes())
    require(parent["source_revision"] == registration["source_revision"]
            and parent["image_id"] == registration["parent_image_id"]
            and sha256(parent_path.read_bytes()).hexdigest() == registration["parent_cpu_sha256"],
            "W5 parent receipt differs from registration")
    for filename in ("campaign.exit", "recovery/campaign.exit"):
        require(member(archive, filename).decode().strip() == "0", f"incomplete W5 campaign: {filename}")
    require(digest(archive, "dispatch.sha256") == registration["dispatch_sha256"], "W5 dispatch differs from registration")
    check_manifest(archive, "dispatch.sha256")
    check_manifest(archive, "result.sha256")
    check_manifest(archive, "recovery/result.sha256")
    check_order(archive, registration)
    summary, allocator, binding = (read(archive, name) for name in
                                   ("verified.json", "allocator/verified.json", "allocator/gpu-checker-input.json"))
    require(summary["parent_cpu_report_sha256"] == registration["parent_cpu_sha256"]
            and summary["allocator_report_sha256"] == digest(archive, "allocator/verified.json")
            and summary["reservations"] == allocator["reservations"]
            and binding["image_id"] == allocator["image_id"] and binding["binaries"] == parent["binaries"],
            "report does not bind the current parent and allocator")
    outcomes = {}
    for mode in ("cache-off", "cache-on"):
        comparison = summary["comparisons"][mode]
        rows = {}
        for image, expected_id, cpu_hash in (("stock", parent["image_id"], registration["parent_cpu_sha256"]),
                                            ("allocator", allocator["image_id"], digest(archive, "allocator/gpu-checker-input.json"))):
            name = f"{image}/{mode}"
            report, load, shutdown = (read(archive, f"{name}/{file}.json") for file in ("report", "soak", "shutdown"))
            expected = (summary["stock_soaks"] if image == "stock" else allocator["soaks"])["modes"][mode]
            require(report["image_id"] == expected_id and report["source_revision"] == parent["source_revision"]
                    and report["riley_binary_sha256"] == parent["binaries"]["/opt/riley/bin/riley"]
                    and report["cpu_report_sha256"] == cpu_hash and report["api_checks"] == 40
                    and report["shutdown_exit"] == 0 and report["gpu_uuid"] == comparison["gpu_uuid"]
                    and report["checkpoint_sha256"] == allocator["checkpoint_sha256"]
                    and report["soak_server_arguments"] == comparison["server_arguments"],
                    f"W5 arm differs from its registered package or crossover: {name}")
            require(all(report["artifacts_sha256"][f"{file}.json"] == digest(archive, f"{name}/{file}.json")
                        for file in ("api", "soak", "shutdown")), f"unbound W5 artifacts: {name}")
            require((load["seconds"], load["rate"]) == (registration["duration_seconds"], registration["client_rate_per_second"])
                    and load["counts"] == comparison[f"{image}_counts"] == report["mixed_soak"]["counts"]
                    and load["latency_windows"] == comparison[f"{image}_latency_windows"]
                    and load["p95_change_percent"] == expected["p95_change_percent"]
                    and load["client_dispatch_p95_ms"] == comparison[f"{image}_client_dispatch_p95_ms"]
                    and load["problem_count"] == len(load["problems"])
                    and report["mixed_soak"]["failed_checks"] == expected["failed_checks"]
                    and report["mixed_soak"]["rss"] == expected["rss"],
                    f"W5 metrics differ from the retained summary: {name}")
            check_precise_rss(load, expected["rss"], name)
            require(all(shutdown[key] == 0 for key in ("active_requests", "waiting_requests", "kv_allocated_blocks"))
                    and shutdown["allocation"] and not any(shutdown["allocation"].values()), f"live resources after {name}")
            rows[image] = {
                "completed": sum(row.get("completed", 0) for row in load["counts"].values()),
                "client_disconnects": load["counts"]["cancel"]["disconnected"],
                "unexpected_errors": load["problem_count"],
                "rss_growth_kib": expected["rss"]["rss_growth_kib"],
                "status": expected["status"],
            }
        outcomes[mode] = rows
    return {"source_revision": parent["source_revision"], "arms": outcomes,
            "recovery_ms": check_recovery(archive, allocator, parent)}


if __name__ == "__main__":
    bundle, registration_file, cpu_report = map(Path, sys.argv[1:])
    with tarfile.open(bundle, "r:gz") as results:
        print(json.dumps(verify(results, json.loads(registration_file.read_bytes()), cpu_report),
                         indent=2, allow_nan=False))
