"""Rehearse replacing the packaged allocator with its parent on an isolated port."""

from argparse import ArgumentParser
from contextlib import closing
from errno import ECONNREFUSED
import json
from pathlib import Path
import socket
import time
from unittest.mock import patch

import check_distribution_gpu as checker
from compare_allocator_performance import expected_image
from qualify_allocator import check_reservations
import run_allocator_performance as runner
from verify_a100_candidate import digest, gpu_evidence, read, require


ORDER = ("packaged-first", "parent-second")
GAUGES = ("active_requests", "waiting_requests", "kv_allocated_blocks")


def closed(port):
    with closing(socket.socket()) as listener:
        listener.settimeout(2)
        require(listener.connect_ex(("127.0.0.1", port)) == ECONNREFUSED, "the isolated route is still listening")


def choose_port():
    with closing(socket.socket()) as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def write(path, payload):
    with path.open("x", encoding="utf-8") as output:
        json.dump(payload, output, indent=2, allow_nan=False)
        output.write("\n")


def run(package, evidence, stage, name):
    port = int((stage / "port").read_text())
    closed(port)
    start = time.monotonic_ns()
    endpoint = f"127.0.0.1:{port}"
    original_docker, original_ready = checker.docker, checker.wait_ready
    milestone = {}

    def docker(*args):
        if args[0] == "create":
            require(args.count("127.0.0.1::8080") == 1 and args[args.index("--publish") + 1] == "127.0.0.1::8080",
                    "expected one loopback-only publish argument")
            args = tuple(f"{endpoint}:8080" if value == "127.0.0.1::8080" else value for value in args)
        if args[0] == "stop":
            milestone["stop_requested_ns"] = time.monotonic_ns()
        return original_docker(*args)

    def ready(container, actual):
        require(actual == port, "routed port changed")
        result = original_ready(container, actual)
        milestone["ready_ns"] = time.monotonic_ns()
        return result

    with patch.object(checker, "docker", docker), patch.object(checker, "wait_ready", ready):
        rc = runner.run(package, evidence, stage, name)
    require(rc == 0, "image execution failed")
    closed(port)
    write(stage / name / "route.json", {"endpoint": endpoint, "closed_before_start_ns": start, **milestone,
                                         "closed_after_cleanup_ns": time.monotonic_ns()})
    return rc


def verify(package, evidence, stage):
    arms = {}
    for name in ORDER:
        folder = stage / name
        _, expected = expected_image(package, evidence, stage, name)
        receipt = gpu_evidence(folder, expected, int((folder / "exit").read_text()))
        require(receipt["status"] == "passed", "image API or resource check failed")
        process = read(folder / "process-before.json")
        require(process == read(folder / "process-after.json") and process["image_id"] == expected["image_id"]
                and process["server_arguments"] == list(runner.SERVER_ARGS), "unexpected process or cache configuration")
        holders = check_reservations(folder, "after")
        require(holders == check_reservations(folder, "before"), "GPU reservation was not restored")
        route = read(folder / "route.json")
        report = folder / "gpu-evidence/route-probe.json"
        probe = read(report)
        require(receipt["artifacts_sha256"][report.name] == digest(report) and route["endpoint"] == probe["endpoint"]
                and route["closed_before_start_ns"] < route["ready_ns"] < probe["client_gated_ns"]
                <= probe["drained_ns"] < route["stop_requested_ns"] < route["closed_after_cleanup_ns"],
                "unexpected route, admission, or drain chronology")
        require(probe["restored_stream"]["completion_tokens"] > 0 and probe["cached_prompt_tokens"] > 0
                and all(probe["drained"][key] == 0 for key in GAUGES), "cache, SSE or drain evidence did not pass")
        arms[name] = {"image_id": receipt["image_id"], "riley_binary_sha256": receipt["riley_binary_sha256"],
                      "checkpoint_sha256": receipt["checkpoint_sha256"], "api_checks": receipt["api_checks"],
                      "gpu_receipt_sha256": receipt["gpu_receipt_sha256"],
                      "shutdown_metrics": receipt["shutdown_metrics"], "reservations_restored": holders,
                      "server_arguments": process["server_arguments"], "route": route, "probe": probe}
    before, after = (arms[key] for key in ORDER)
    first, second = before["route"], after["route"]
    require(before["image_id"] != after["image_id"] and before["riley_binary_sha256"] == after["riley_binary_sha256"]
            and before["checkpoint_sha256"] == after["checkpoint_sha256"]
            and before["server_arguments"] == after["server_arguments"]
            and before["probe"]["restored_stream"] == after["probe"]["restored_stream"]
            and first["endpoint"] == second["endpoint"]
            and first["closed_after_cleanup_ns"] < second["closed_before_start_ns"],
            "recovery did not switch images with the same checkpoint, route, options and streaming output")
    return {"scope": "isolated loopback recovery drill", "image_order": ORDER, "arms": arms,
            "stop_to_parent_ready_ms": (second["ready_ns"] - first["stop_requested_ns"]) / 1_000_000,
            "same_sse_probe_output": True}


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "verify"))
    for key in ("package", "evidence", "stage"):
        parser.add_argument(key, type=Path)
    parser.add_argument("arm", choices=ORDER, nargs="?")
    args = parser.parse_args()
    if args.action == "run":
        require(args.arm in ORDER, "select one drill arm")
        raise SystemExit(run(args.package, args.evidence, args.stage, args.arm))
    print(json.dumps(verify(args.package, args.evidence, args.stage), indent=2, allow_nan=False))
