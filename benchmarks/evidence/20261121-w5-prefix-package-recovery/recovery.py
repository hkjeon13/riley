"""Replace the current allocator trial with its verified parent on two loopback ports."""

from contextlib import ExitStack
from errno import ECONNREFUSED
import json
from pathlib import Path
import signal
import socket
import sys
import time
from unittest.mock import patch

import check_distribution_gpu as checker
from allocator import bind
from current_binding import process_proof
from drain_probe import GAUGES, HOST, run as probe_route
from verify import MODEL, MODES, RILEY, arm, check_reservations, digest, read, require
from verify_allocator_image import image_config, inspect

ORDER = ("allocator", "stock")
OPTIONS = (*checker.MIXED_SERVER_ARGS, "--kv-blocks", "16384", "--prefix-cache-blocks", "128")


def ports():
    with ExitStack() as stack:
        listeners = [stack.enter_context(socket.socket()) for _ in range(2)]
        for listener in listeners:
            listener.bind((HOST, 0))
        return [listener.getsockname()[1] for listener in listeners]


def closed(*endpoints):
    for port in endpoints:
        with socket.socket() as wire:
            wire.settimeout(2)
            require(wire.connect_ex((HOST, port)) == ECONNREFUSED, "a recovery port remains open")


def record(path, result):
    with path.open("x") as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")


def run(root, variant, parent_file, model, label):
    front, backend = read(root / "ports.json")
    closed(front, backend)
    milestones = {"closed_before_start_ns": time.monotonic_ns()}
    spec, parent = bind(variant, parent_file)
    selected = label == "allocator"
    cpu = variant / "gpu-checker-input.json" if selected else parent_file
    out = root / label
    expected = read(cpu)
    require(expected == (spec if selected else read(parent_file)), "GPU checker input changed")

    original_config, original_ready, original_docker = checker.check_config, checker.wait_ready, checker.docker

    def config(documents, image, revision, archive):
        if selected:
            observed, = documents
            require(observed["Id"] == image == spec["image_id"], "wrong allocator image")
            stock = inspect(parent)
            original_config([stock], parent, revision, archive)
            image_config(stock, observed)
        else:
            original_config(documents, image, revision, archive)

    def process(container):
        if selected:
            return process_proof(container, expected["image_id"])
        info = checker.inspect_container(container)
        require(info["Image"] == parent and info["State"]["Running"] and info["State"]["Pid"] > 0,
                "wrong stock container")
        original_docker("exec", container, "/bin/sh", "-ec",
                        r'''! tr '\0' '\n' </proc/1/environ | grep -Eq '^(LD_PRELOAD|MALLOC_CONF|GLIBC_TUNABLES)='
! grep -Fq '/opt/riley/lib/libjemalloc.so.2' /proc/1/maps''')
        return {"image_id": parent, "container_id": container, "host_pid": info["State"]["Pid"]}

    def ready(container, port):
        require(port == backend, "the container is not serving on the recorded port")
        reply = original_ready(container, port)
        milestones["ready_ns"] = time.monotonic_ns()
        info = checker.inspect_container(container)
        require(tuple(info["Args"][-len(OPTIONS):]) == OPTIONS, "the effective server options changed")
        record(out / "process-start.json", process(container))
        return reply

    def docker(*args):
        if args[0] == "create":
            publish = args.index("--publish") + 1
            require(args[publish] == f"{HOST}::8080" and args[-2:] == ("--model-id", "m"),
                    "unexpected GPU checker command")
            args = (*args[:publish], f"{HOST}:{backend}:8080", *args[publish + 1:], *OPTIONS)
        if args[0] == "stop":
            record(out / "admission.json", probe_route(front, backend))
            require(process(args[-1]) == read(out / "process-start.json"), "process changed while serving")
            milestones["stop_requested_ns"] = time.monotonic_ns()
        return original_docker(*args)

    with (patch.object(checker, "check_config", config), patch.object(checker, "wait_ready", ready),
          patch.object(checker, "docker", docker)):
        status = checker.check_gpu(cpu, MODES["cache-off"], model, MODEL, out, True)
    require(status == 0, "packaged recovery arm failed")
    closed(front, backend)
    record(root / f"{label}-route.json", {**milestones, "closed_after_cleanup_ns": time.monotonic_ns(),
                                          "gateway": f"{HOST}:{front}", "server": f"{HOST}:{backend}",
                                          "server_arguments": OPTIONS})
    return status


def verify(root, variant, parent_file):
    front, backend = read(root / "ports.json")
    spec, _ = bind(variant, parent_file)
    results = {}
    reservations = check_reservations(root, "before")
    for label, cpu_file in (("allocator", variant / "gpu-checker-input.json"), ("stock", parent_file)):
        cpu = read(cpu_file)
        report, shutdown, out = arm(root, label, MODES["cache-off"], cpu, digest(cpu_file))
        require(report["status"] == "passed" and check_reservations(root, f"after-{label}") == reservations,
                "GPU serving failed or the reservation changed")
        route = read(root / f"{label}-route.json")
        obs = read(out / "admission.json")
        shape = obs["active_after_gate"]
        require(route["server_arguments"] == list(OPTIONS)
                and route["gateway"] == f"{HOST}:{front}" and route["server"] == f"{HOST}:{backend}"
                and route["closed_before_start_ns"] < route["ready_ns"] < obs["gate_closed_ns"]
                <= obs["active_after_gate_ns"] < obs["sse_completed_ns"] <= obs["drained_ns"]
                < route["stop_requested_ns"] < route["closed_after_cleanup_ns"], "invalid recovery chronology")
        require(obs["rejected_http_status"] == 503 and shape["active_requests"] == 1
                and shape["waiting_requests"] == 0 and shape["kv_allocated_blocks"] > 0
                and obs["stream"]["completion_tokens"] > 0
                and obs["cached_prompt_tokens"] == obs["cache_hit_delta"] > 0
                and obs["drained"] == dict.fromkeys(GAUGES, 0), "missing live drain or cache evidence")
        cache = checker.prefix_cache_evidence({"initial": obs.pop("initial"), "final": obs.pop("final"),
                                               "rss_samples": []}, shutdown, 128)
        require(cache["status"] == "passed", "shutdown did not preserve the drained cache counters")
        process = read(out / "process-start.json")
        require(process["image_id"] == report["image_id"], "process image differs from the receipt")
        results[label] = {"image_id": report["image_id"], "api_checks": report["api_checks"],
                          "gpu_report_sha256": digest(out / "report.json"), "admission": obs,
                          "prefix_cache": cache, "shutdown": shutdown, "route": route, "process": process}

    trial, stock = (results[label] for label in ORDER)
    require(trial["image_id"] == spec["image_id"] != stock["image_id"]
            and trial["admission"]["stream"] == stock["admission"]["stream"]
            and trial["route"]["closed_after_cleanup_ns"] < stock["route"]["closed_before_start_ns"],
            "the replacement did not restore the same stream on the same route")
    ready_ms = (stock["route"]["ready_ns"] - trial["route"]["stop_requested_ns"]) / 1e6
    return {"status": "passed", "scope": "isolated loopback recovery", "source_revision": spec["source_revision"],
            "remote_evidence": str(root), "riley_binary_sha256": spec["binaries"][RILEY],
            "checkpoint_sha256": MODEL, "stop_to_stock_ready_ms": ready_ms, "reservations": reservations,
            "arms": results}


def interrupted(_number, _frame):
    raise KeyboardInterrupt


if __name__ == "__main__":
    action, *arguments = sys.argv[1:]
    if action == "ports":
        print(json.dumps(ports()))
    elif action == "run":
        signal.signal(signal.SIGTERM, interrupted)
        root, variant, parent, model = map(Path, arguments[:-1])
        label = arguments[-1]
        require(label in ORDER, "unknown recovery arm")
        raise SystemExit(run(root, variant, parent, model, label))
    elif action == "verify":
        print(json.dumps(verify(*map(Path, arguments)), indent=2, allow_nan=False))
    else:
        raise SystemExit("expected a recovery action")
