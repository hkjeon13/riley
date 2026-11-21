#!/usr/bin/env python3
"""Run the current distribution's real-model API and shutdown checks on one GPU."""

from argparse import ArgumentParser
import csv
from hashlib import sha256
from http.client import HTTPConnection
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from uuid import uuid4

from check_distribution_image import IMAGE_ID, check_config, digest, require

RILEY = "/opt/riley/bin/riley"
CHECKPOINT = "riley-checkpoint.json"
SHUTDOWN = "/tmp/riley-distribution-shutdown.json"
API_CHECK = Path(__file__).resolve().parents[1] / "benchmarks/live/api_check.py"
ACCURACY_CHECK = API_CHECK.with_name("long_accuracy.py")
STRUCTURED_HOLDOUT_CHECK = API_CHECK.with_name("structured_holdout_check.py")
SOAK_CHECK = API_CHECK.with_name("soak.py")
MIXED_STAGE_CHECK = API_CHECK.with_name("mixed_stage_check.py")
TP_HOLDOUT_CHECK = API_CHECK.with_name("tp_holdout_check.py")
CACHE_BLOCKS = {"cache-off": None, "cache-on": 128}
SMOKE_SECONDS = 45
SMOKE_KV_BLOCKS = 4096
MIXED_LOAD_RATE = 8
MIXED_LOAD_CHECKS = {"no_unexpected_http_or_stream_error", "completed_schema_and_tool_covered",
                     "disconnects_observed", "idle_after_drain", "native_allocation_plateau", "workload_coverage"}
BATCH_SHAPE_CAPACITY = 24
BATCH_SHAPE_FIELDS = ("dense_rows", "hit_count", "latency_sample_count", "gpu_execution_ns_total",
                      "gpu_execution_ns_average", "gpu_execution_ns_maximum", "gpu_execution_ns_last")
MIXED_SERVER_ARGS = ("--max-active-sequences", "32", "--max-sequence-tokens", "8192", "--max-output-tokens", "512")
GPU_UUID = re.compile(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}")
SHA256 = re.compile(r"[0-9a-f]{64}")


def capture(*argv):
    return subprocess.check_output(argv, text=True, timeout=30)


def docker(*argv):
    return capture("docker", *argv)


def inspect_container(container):
    record, = json.loads(docker("container", "inspect", container))
    require(record["Id"] == container, "container ID changed")
    return record


def holder_pids(gpu, *, peer=False):
    prefix = "RILEY_GPU_PEER_HOLDER" if peer else "RILEY_GPU_HOLDER"
    pid = os.getenv(f"{prefix}_PID")
    ticks = os.getenv(f"{prefix}_START_TICKS")
    if pid is None and ticks is None:
        return set()
    require(pid and pid.isdecimal() and ticks and ticks.isdecimal()
            and os.getenv(f"{prefix}_GPU" if peer else "CUDA_VISIBLE_DEVICES") == gpu,
            "GPU holder identity is inconsistent")
    proc = Path("/proc") / pid
    require(proc.stat().st_uid == os.getuid()
            and (proc / "stat").read_text().rsplit(")", 1)[1].split()[19] == ticks,
            "GPU holder owner or start time changed")
    return {int(pid)}


def gpu_pids(gpu):
    output = capture("nvidia-smi", "-i", gpu, "--query-compute-apps=pid", "--format=csv,noheader,nounits")
    return {int(line) for line in output.splitlines() if line.strip()}


def wait_for_gpu_release(gpu, holders, riley_pid):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        current = gpu_pids(gpu)
        if current == holders:
            return
        require(current == holders | {riley_pid}, "unexpected GPU process after shutdown")
        time.sleep(0.25)
    raise TimeoutError("Riley CUDA context remained after shutdown")


def wait_ready(container, port, *, timeout=480):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        connection = HTTPConnection("127.0.0.1", port, timeout=2)
        try:
            connection.request("GET", "/readyz")
            reply = connection.getresponse()
            if reply.status == 200:
                body = json.loads(reply.read())
                require(body.get("ready") is True and body.get("accepting") is True, "invalid readiness response")
                return body
        except OSError:
            pass
        finally:
            connection.close()
        require(inspect_container(container)["State"]["Running"], "container exited before model readiness")
        time.sleep(1)
    raise TimeoutError("Riley did not load the checkpoint before the readiness deadline")


def log_command(out, file, argv, seconds, *, check=True):
    with (out / file).open("xb") as log:
        return subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, check=check, timeout=seconds)


def check_accuracy(out, address, label, golden, workload=None):
    captured = out / f"accuracy-{label}.json"
    comparison = out / f"accuracy-{label}-comparison.json"
    golden_sha256 = digest(golden)
    workload_sha256 = digest(workload) if workload is not None else None
    workload_args = ["--workload", str(workload)] if workload is not None else []
    runner = [sys.executable, "-B", str(ACCURACY_CHECK)]
    log_command(out, f"accuracy-{label}-capture.log",
                [*runner, "capture", address, "m", str(captured), *workload_args], 1200)
    log_command(out, f"accuracy-{label}-comparison.log",
                [*runner, "compare-captures", str(golden), str(captured), str(comparison), *workload_args], 60)
    result = json.loads(comparison.read_bytes())
    require(result["status"] == "passed" and result["candidate_sha256"] == digest(captured)
            and result["golden_sha256"] == golden_sha256 and result["workload_sha256"] == workload_sha256,
            f"{label} accuracy evidence failed or changed")
    return {"golden_sha256": result["golden_sha256"], "workload_sha256": result["workload_sha256"],
            "verified_requests": sum(mode["matched"] for mode in result["modes"].values())}


def check_structured_holdout(out, address, workload):
    result_file = out / "structured-holdout.json"
    workload_sha256 = digest(workload)
    log_command(out, "structured-holdout.log", [sys.executable, "-B", str(STRUCTURED_HOLDOUT_CHECK), address,
                                               "m", str(workload), "--out", str(result_file)], 900)
    result = json.loads(result_file.read_bytes())
    modes = result["modes"]
    rejection = result["context_rejection"]
    require(result["status"] == "passed" and result["workload_sha256"] == workload_sha256 == digest(workload)
            and result["c1_transport_pairs"] == 8 and len(result["cases"]) == 16
            and rejection["http_status"] == 400 and rejection["error_code"] == "invalid_request"
            and rejection["sequence_tokens_requested"] > rejection["max_sequence_tokens"] == 8192
            and {mode: row["requests"] for mode, row in modes.items()} == {"c1": 16, "c8": 16, "c32": 32}
            and modes["c1"]["observed_peak_active"] == 1
            and all(modes[mode]["requests"] == modes[mode]["exact_matches"]
                    and 2 <= modes[mode]["observed_peak_active"] <= concurrency
                    for mode, concurrency in (("c8", 8), ("c32", 32))), "invalid structured holdout receipt")
    return result


def current_batch_shapes(sample):
    require(isinstance(sample, dict) and isinstance(sample.get("metrics"), dict), "missing metrics sample")
    shape = sample["metrics"].get("batch_shapes")
    require(isinstance(shape, dict) and shape.get("metrics_degraded") is False, "missing or degraded batch shapes")
    count, entries = shape.get("bucket_count"), shape.get("buckets")
    require(type(count) is int and 0 < count <= BATCH_SHAPE_CAPACITY
            and isinstance(entries, list) and len(entries) == BATCH_SHAPE_CAPACITY,
            "expected 24 batch-shape slots with a nonempty configured prefix")
    rows = {}
    previous = 0
    for index, bucket in enumerate(entries):
        require(isinstance(bucket, dict)
                and all(type(bucket.get(key)) is int and bucket[key] >= 0 for key in BATCH_SHAPE_FIELDS),
                f"invalid fields in batch-shape slot {index}")
        dense, hits, samples, total, average, maximum, last = (bucket[key] for key in BATCH_SHAPE_FIELDS)
        if index >= count:
            require(not any((dense, hits, samples, total, average, maximum, last)), "unused batch-shape slot is nonzero")
            continue
        require(dense > previous, "configured batch-shape rows must increase")
        require(samples == hits and last <= maximum <= total <= maximum * samples
                and average == (total // samples if samples else 0), f"inconsistent timing for {dense} dense rows")
        rows[dense] = (hits, total, maximum)
        previous = dense

    observed = shape.get("last")
    if any(hits for hits, _, _ in rows.values()):
        require(isinstance(observed, dict) and all(type(observed.get(key)) is int
                for key in ("active_rows", "selected_dense_rows", "padding_rows")), "missing last batch shape")
        dense, active = observed["selected_dense_rows"], observed["active_rows"]
        require(dense in rows and rows[dense][0] > 0 and 0 < active <= dense
                and dense - active == observed["padding_rows"], "invalid last batch shape or padding")
    else:
        require("last" in shape and observed is None, "batch shape observed before the first hit")
    return rows


def batch_shape_evidence(result):
    try:
        samples = result.get("rss_samples")
        require(isinstance(samples, list), "missing periodic metrics samples")
        first = previous = current_batch_shapes(result.get("initial"))
        for sample in [*samples, result.get("final")]:
            current = current_batch_shapes(sample)
            require(current.keys() == previous.keys(), "configured batch-shape rows changed")
            require(all(all(a <= b for a, b in zip(previous[rows], counters, strict=True))
                        for rows, counters in current.items()), "batch-shape counters went backwards")
            previous = current
        hits = sum(current[rows][0] - start[0] for rows, start in first.items())
        gpu_ns = sum(current[rows][1] - start[1] for rows, start in first.items())
        require(hits > 0 and gpu_ns > 0, "mixed load produced no new committed GPU timing")
    except ValueError as error:
        return {"status": "failed", "detail": str(error)}
    return {"status": "passed", "snapshots": len(samples) + 2, "slot_capacity": BATCH_SHAPE_CAPACITY,
            "configured_dense_rows": list(first), "committed_steps": hits, "gpu_execution_ns": gpu_ns}


def check_mixed_load(out, address, container, pid, *, smoke, expect_tools_unsupported=False):
    label, seconds, option = ("smoke", SMOKE_SECONDS, "--smoke") if smoke else (
        "soak", 1800, "--require-zero-rss-growth")
    path = out / f"{label}.json"
    command = [sys.executable, "-B", str(SOAK_CHECK), address, "m", "--pid", str(pid),
               "--docker-container", container, "--seconds", str(seconds), "--rate", str(MIXED_LOAD_RATE),
               "--shared-prefix", option, "--out", str(path)]
    if expect_tools_unsupported:
        command.append("--expect-tools-unsupported")
    completed = log_command(out, f"{label}.log", command, seconds + 300, check=False)
    result = json.loads(path.read_bytes())
    checks = result["checks"]
    require(MIXED_LOAD_CHECKS <= checks.keys() and all(type(passed) is bool for passed in checks.values()),
            "missing or invalid mixed load checks")
    failed = sorted(name for name, passed in checks.items() if not passed)
    status = "failed" if failed else "passed"
    require(result["status"] == status and completed.returncode == int(bool(failed))
            and result["mode"] == ("smoke" if smoke else "mixed_soak")
            and result["seconds"] == seconds and result["rate"] == MIXED_LOAD_RATE
            and result["shared_prefixes"] == 8
            and result.get("expect_tools_unsupported", False) is expect_tools_unsupported
            and result["rss_source"] == {"container_id": container, "host_pid": pid, "container_pid": 1},
            "mixed load evidence differs from the requested container, workload or exit status")
    batch_shapes = batch_shape_evidence(result)
    if batch_shapes["status"] != "passed":
        failed = sorted([*failed, "current_batch_shape_metrics"])
    return {**result, "batch_shape_metrics": batch_shapes, "checks": {
        **checks, "current_batch_shape_metrics": batch_shapes["status"] == "passed",
    }}, "failed" if failed else "passed", failed


def check_mixed_soak(out, address, mode, container, pid):
    result, status, failed = check_mixed_load(out, address, container, pid, smoke=False)
    rss_gate = "rss_growth_second_half_at_most_zero"
    checks = result["checks"]
    require({rss_gate, "p95_drift_at_most_5_percent"} <= checks.keys(), "missing strict soak checks")
    rss = result["smaps_rollup_second_half"]
    delta = rss["rss_growth_kib"]
    require(checks[rss_gate] == (type(delta) is int and delta <= 0), "strict RSS verdict differs from the measurements")
    return {"mode": mode, "status": status, "failed_checks": failed,
            "rss": rss, "counts": result["counts"], "batch_shape_metrics": result["batch_shape_metrics"]}


def find_interrupted_container(name):
    # The daemon can complete create after the Docker client is interrupted.
    try:
        record, = json.loads(docker("container", "inspect", name))
    except subprocess.CalledProcessError:
        return None
    require(record["Name"] == "/" + name and record["Config"]["Labels"].get("io.riley.gpu-check") == name
            and SHA256.fullmatch(record["Id"]), "cannot bind interrupted container to this run")
    return record["Id"]


def check_gpu(cpu_report, gpu, model_dir, model_sha, out, tools, *, accuracy_baseline=None, accuracy_holdout=None,
              mixed_soak=None, serving_smoke=False, smoke_kv_blocks=None, expect_tools_unsupported=False,
              mixed_stage=False, tp_holdout=None, structured_holdout=None):
    require(GPU_UUID.fullmatch(gpu) and SHA256.fullmatch(model_sha), "pass a GPU UUID and model SHA-256")
    long_accuracy = accuracy_baseline or accuracy_holdout or structured_holdout
    require(mixed_soak is None or (mixed_soak in CACHE_BLOCKS and not long_accuracy),
            "select mixed soak or long accuracy for each container")
    require(not serving_smoke or not (long_accuracy or mixed_soak),
            "select serving smoke separately from mixed soak and long accuracy")
    require(not mixed_stage or not (long_accuracy or mixed_soak or serving_smoke),
            "select the mixed-stage probe separately from other workloads")
    require(structured_holdout is None or tools, "structured holdout requires a tool-compatible checkpoint and --tools")
    if tp_holdout is not None:
        peer, tp_workload = tp_holdout
        require(gpu != peer and GPU_UUID.fullmatch(peer), "TP=2 needs a distinct peer GPU UUID")
        require(not (long_accuracy or mixed_soak or serving_smoke or mixed_stage),
                "select the TP=2 holdout separately from other workloads")
        tp_workload = Path(tp_workload).resolve(strict=True)
    require(not expect_tools_unsupported or serving_smoke and not tools,
            "--expect-tools-unsupported requires serving smoke and excludes --tools")
    if smoke_kv_blocks is not None:
        require(serving_smoke and type(smoke_kv_blocks) is int and smoke_kv_blocks > 0,
                "--smoke-kv-blocks requires serving smoke and a positive block count")
    cpu_report_bytes = cpu_report.read_bytes()
    report = json.loads(cpu_report_bytes)
    require(report["schema_version"] == "riley.distribution-image-check.v1" and report["status"] == "passed",
            "expected a passed distribution CPU report")
    image, revision, archive = (report[key] for key in ("image_id", "source_revision", "source_archive_sha256"))
    binary = report["binaries"][RILEY]
    require(IMAGE_ID.fullmatch(image) and re.fullmatch(r"[0-9a-f]{40}", revision)
            and SHA256.fullmatch(archive) and SHA256.fullmatch(binary), "invalid reviewed image provenance")
    check_config(json.loads(docker("image", "inspect", image)), image, revision, archive)

    model_dir = model_dir.resolve(strict=True)
    require(all(character not in str(model_dir) for character in (",", "\n", "\r")), "invalid bind mount path")
    manifest = model_dir / CHECKPOINT
    require(manifest.is_file() and not manifest.is_symlink(), "expected a regular checkpoint manifest")
    manifest_bytes = manifest.read_bytes()
    require(sha256(manifest_bytes).hexdigest() == model_sha, "checkpoint manifest differs from the supplied SHA-256")
    checkpoint = json.loads(manifest_bytes)
    holders = {gpu: holder_pids(gpu)}
    if tp_holdout is not None:
        holders[peer] = holder_pids(peer, peer=True)
    require(all(gpu_pids(uuid) == pids for uuid, pids in holders.items()),
            "target GPU has an unexpected process before startup")
    out.mkdir()
    accuracy = {}
    soak = None
    smoke = None
    stage = None
    parallel = None
    server_args = ()
    gpu_request = f"device={gpu}"
    gpu_env = ()
    if tp_holdout is not None:
        ordered_devices = ",".join(holders)
        gpu_request = f'"device={ordered_devices}"'
        gpu_env = ("--env", f"CUDA_VISIBLE_DEVICES={ordered_devices}")
        server_args = ("--inference-profile", "native-bf16-varlen-v1", "--tensor-parallel-devices", "0,1",
                       "--max-active-sequences", "32", "--max-sequence-tokens", "8192",
                       "--max-output-tokens", "256", "--kv-blocks", "2048")
    elif long_accuracy:
        server_args = ("--max-active-sequences", "32", "--max-sequence-tokens", "8192", "--max-output-tokens", "256")
    elif mixed_soak is not None:
        server_args = (*MIXED_SERVER_ARGS, "--kv-blocks", "16384")
        cache_blocks = CACHE_BLOCKS[mixed_soak]
        if cache_blocks is not None:
            server_args += ("--prefix-cache-blocks", str(cache_blocks))
    elif serving_smoke:
        blocks = SMOKE_KV_BLOCKS if smoke_kv_blocks is None else smoke_kv_blocks
        server_args = (*MIXED_SERVER_ARGS, "--kv-blocks", str(blocks), "--prefix-cache-blocks", "128")
    elif mixed_stage:
        server_args = ("--inference-profile", "native-bf16-varlen-v1", "--max-active-sequences", "2",
                       "--max-sequence-tokens", "8192", "--max-output-tokens", "256", "--kv-blocks", "4096",
                       "--batch-token-budget", "512", "--prefill-chunk-tokens", "256", "--batch-shape-buckets",
                       "1,2,4,8,16,24,32,48,64,80,96,112,128,144,160,176,192,208,224,240,256,512")
    mount = f"type=bind,source={model_dir},target=/models,readonly"
    name = f"riley-gpu-check-{uuid4().hex}"
    container = None
    try:
        created = docker(
            "create", "--pull=never", "--name", name, "--label", f"io.riley.gpu-check={name}",
            "--gpus", gpu_request, "--env", "NVIDIA_DRIVER_CAPABILITIES=compute,utility", *gpu_env,
            "--env", f"RILEY_SHUTDOWN_METRICS_PATH={SHUTDOWN}", "--publish", "127.0.0.1::8080",
            "--mount", mount, image, "serve", "--model", "/models", "--model-id", "m", *server_args,
        ).strip()
        require(SHA256.fullmatch(created), "Docker did not return a full container ID")
        container = created
        docker("start", container)
        runtime = inspect_container(container)
        require(runtime["Image"] == image and runtime["State"]["Running"], "unexpected running image")
        (binding,) = runtime["NetworkSettings"]["Ports"]["8080/tcp"]
        port = int(binding["HostPort"])
        require(binding["HostIp"] == "127.0.0.1" and 0 < port <= 65535, "invalid loopback test binding")
        ready = wait_ready(container, port, timeout=1200) if tp_holdout is not None else wait_ready(container, port)
        (out / "ready.json").write_text(json.dumps(ready, indent=2) + "\n")

        devices = docker("exec", container, "nvidia-smi", "--query-gpu=uuid,name,driver_version",
                         "--format=csv,noheader")
        device_rows = list(csv.reader(devices.splitlines(), skipinitialspace=True))
        require(len(device_rows) == len(holders) and all(len(row) == 3 for row in device_rows),
                "unexpected GPU count visible in the container")
        devices_by_uuid = {row[0]: row for row in device_rows}
        require(devices_by_uuid.keys() == holders.keys(), "unexpected GPU visible in the container")
        device = devices_by_uuid[gpu]
        riley_pid = runtime["State"]["Pid"]
        require(type(riley_pid) is int and riley_pid > 0
                and all(gpu_pids(uuid) == pids | {riley_pid} for uuid, pids in holders.items()),
                "Riley does not own the expected CUDA context")
        observed, _ = docker("exec", container, "/bin/sh", "-ec",
                             'test "$(id -u):$(id -g)" = 65532:65532; '
                             f'test "$(readlink /proc/1/exe)" = {RILEY}; sha256sum /proc/1/exe').split()
        require(observed == binary, "running Riley differs from the reviewed distribution binary")
        log_command(out, "checkpoint-verify.log", ["docker", "exec", container, "/opt/riley/bin/riley-checkpoint",
                                                   "verify", "--checkpoint", "/models"],
                    1200 if tp_holdout is not None else 600)

        api = out / "api.json"
        address = f"127.0.0.1:{port}"
        command = [sys.executable, str(API_CHECK), address, "m", "--out", str(api)]
        if tools:
            command.append("--tools")
        if expect_tools_unsupported:
            command.append("--expect-tools-unsupported")
        log_command(out, "api.log", command, 600)
        results = json.loads(api.read_bytes())["results"]
        require(results and all(record["passed"] is True for record in results), "the live API suite failed")
        if accuracy_baseline is not None:
            accuracy["baseline"] = check_accuracy(out, address, "baseline", accuracy_baseline)
        if accuracy_holdout is not None:
            workload, golden = accuracy_holdout
            accuracy["holdout"] = check_accuracy(out, address, "holdout", golden, workload)
        if structured_holdout is not None:
            accuracy["structured_holdout"] = check_structured_holdout(out, address, structured_holdout)
        if mixed_soak is not None:
            soak = check_mixed_soak(out, address, mixed_soak, container, riley_pid)
        if serving_smoke:
            observed_smoke, smoke_status, failed = check_mixed_load(
                out, address, container, riley_pid, smoke=True, expect_tools_unsupported=expect_tools_unsupported)
            smoke = {key: observed_smoke[key] for key in ("seconds", "rate", "counts", "counter_changes", "checks",
                                                        "batch_shape_metrics")}
            smoke.update(status=smoke_status, failed_checks=failed, server_arguments=list(server_args))
            if expect_tools_unsupported:
                smoke["expect_tools_unsupported"] = True
        if mixed_stage:
            path = out / "mixed-stage.json"
            log_command(out, "mixed-stage.log",
                        [sys.executable, "-B", str(MIXED_STAGE_CHECK), address, "m", "--out", str(path)], 600)
            stage = json.loads(path.read_bytes())
            require(stage["status"] == "passed" and stage["prefill_chunk_tokens"] == 256 and len(stage["trials"]) == 3
                    and all(trial["matching_responses"] == 2 and trial["witnesses"] for trial in stage["trials"]),
                    "mixed-stage evidence does not cover three matched trials")
            stage = {"matching_responses": 6, "observed_mixed_iterations": sum(
                len(trial["witnesses"]) for trial in stage["trials"]), "server_arguments": list(server_args)}
        if tp_holdout is not None:
            result_file = out / "tp-holdout.json"
            log_command(out, "tp-holdout.log",
                        [sys.executable, "-B", str(TP_HOLDOUT_CHECK), address, "m", str(tp_workload),
                         "--out", str(result_file)], 900)
            parity = json.loads(result_file.read_bytes())
            require(parity["status"] == "passed" and parity["workload_sha256"] == digest(tp_workload)
                    and {mode: data["requests"] for mode, data in parity["modes"].items()}
                    == {"c1": 11, "c8": 11, "c32": 32}
                    and all(parity["modes"][mode]["exact_matches"] == parity["modes"][mode]["requests"]
                            for mode in ("c8", "c32")), "invalid TP=2 holdout receipt")
            parallel = {"gpu_uuids": list(holders), "server_arguments": list(server_args), "holdout": parity}
        require(all(gpu_pids(uuid) == pids | {riley_pid} for uuid, pids in holders.items()),
                "GPU process ownership changed during inference checks")
        require(digest(manifest) == model_sha, "checkpoint manifest changed during inference checks")

        docker("stop", "--time", "20", container)
        state = inspect_container(container)["State"]
        require(not state["Running"] and state["ExitCode"] == 0, "Riley did not stop gracefully")
        subprocess.run(["docker", "cp", f"{container}:{SHUTDOWN}", str(out / "shutdown.json")],
                       check=True, timeout=30)
        metrics = json.loads((out / "shutdown.json").read_bytes())
        require(all(metrics[key] == 0 for key in ("active_requests", "waiting_requests", "kv_allocated_blocks"))
                and metrics["allocation"] and all(value == 0 for value in metrics["allocation"].values()),
                "Riley retained allocation or KV state at shutdown")
        for uuid, pids in holders.items():
            wait_for_gpu_release(uuid, pids, riley_pid)
    finally:
        container = container or find_interrupted_container(name)
        if container is not None:
            try:
                log_command(out, "server.log", ["docker", "logs", container], 15, check=False)
            finally:
                docker("rm", "--force", "--volumes", container)

    measured_load = smoke if serving_smoke else soak
    status = measured_load["status"] if measured_load is not None else "passed"
    result = {"schema_version": "riley.distribution-gpu-check.v1", "status": status, "image_id": image,
              "source_revision": revision, "source_archive_sha256": archive, "riley_binary_sha256": observed,
              "cpu_report_sha256": sha256(cpu_report_bytes).hexdigest(), "checkpoint_sha256": model_sha,
              "checkpoint_source_model": checkpoint["source_model"], "checkpoint_source_revision": checkpoint["source_revision"],
              "gpu_uuid": device[0], "gpu_name": device[1], "driver_version": device[2], "api_checks": len(results),
              "tool_checks_enabled": tools, "shutdown_exit": state["ExitCode"],
              "artifacts_sha256": {path.name: digest(path) for path in sorted(out.iterdir())}}
    if accuracy:
        result["accuracy_suites"] = accuracy
        result["accuracy_server_arguments"] = list(server_args)
    if soak is not None:
        result["mixed_soak"] = soak
        result["soak_server_arguments"] = list(server_args)
    if smoke is not None:
        result["serving_smoke"] = smoke
    if stage is not None:
        result["mixed_stage"] = stage
    if parallel is not None:
        result["tensor_parallel"] = parallel
    with (out / "report.json").open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    return int(status != "passed")


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("cpu_report", type=Path)
    parser.add_argument("gpu_uuid")
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("model_manifest_sha256")
    parser.add_argument("output", type=Path)
    tool_mode = parser.add_mutually_exclusive_group()
    tool_mode.add_argument("--tools", action="store_true", help="also validate tool calls for a compatible checkpoint")
    tool_mode.add_argument("--expect-tools-unsupported", action="store_true",
                           help="validate the tool rejection contract for a checkpoint without tool support")
    parser.add_argument("--accuracy-baseline", type=Path, metavar="GOLDEN",
                        help="compare the historical long-context workload against a Riley capture")
    parser.add_argument("--accuracy-holdout", type=Path, nargs=2, metavar=("WORKLOAD", "GOLDEN"),
                        help="compare the pinned holdout against a Riley capture")
    parser.add_argument("--structured-holdout", type=Path, metavar="WORKLOAD",
                        help="validate long-context JSON and forced tools with a pinned holdout; requires --tools and jsonschema")
    parser.add_argument("--mixed-soak", choices=CACHE_BLOCKS, help="run the strict 30-minute W5 load in this cache mode")
    parser.add_argument("--serving-smoke", action="store_true",
                        help="combine API checks with 45 seconds of mixed traffic and cancellation")
    parser.add_argument("--mixed-stage", action="store_true",
                        help="compare singleton streams and witness committed mixed prefill/decode iterations")
    parser.add_argument("--tp-holdout", nargs=2, metavar=("PEER_GPU_UUID", "WORKLOAD"),
                        help="verify the two-device distribution and compare C1, C8, and C32 holdout responses")
    parser.add_argument("--smoke-kv-blocks", type=int,
                        help=f"set the serving smoke KV pool (default: {SMOKE_KV_BLOCKS} blocks)")
    args = parser.parse_args()

    def interrupted(_signal, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    return check_gpu(args.cpu_report, args.gpu_uuid, args.model_dir, args.model_manifest_sha256, args.output, args.tools,
                     accuracy_baseline=args.accuracy_baseline, accuracy_holdout=args.accuracy_holdout,
                     mixed_soak=args.mixed_soak, serving_smoke=args.serving_smoke, smoke_kv_blocks=args.smoke_kv_blocks,
                     expect_tools_unsupported=args.expect_tools_unsupported, mixed_stage=args.mixed_stage,
                     tp_holdout=args.tp_holdout, structured_holdout=args.structured_holdout)


if __name__ == "__main__":
    raise SystemExit(main())
