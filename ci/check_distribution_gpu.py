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
SOAK_CHECK = API_CHECK.with_name("soak.py")
MIXED_STAGE_CHECK = API_CHECK.with_name("mixed_stage_check.py")
CACHE_BLOCKS = {"cache-off": None, "cache-on": 128}
SMOKE_SECONDS = 45
SMOKE_KV_BLOCKS = 4096
MIXED_LOAD_RATE = 8
MIXED_LOAD_CHECKS = {"no_unexpected_http_or_stream_error", "completed_schema_and_tool_covered",
                     "disconnects_observed", "idle_after_drain", "native_allocation_plateau", "workload_coverage"}
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


def holder_pids(gpu):
    pid = os.getenv("RILEY_GPU_HOLDER_PID")
    ticks = os.getenv("RILEY_GPU_HOLDER_START_TICKS")
    if pid is None and ticks is None:
        return set()
    require(pid and pid.isdecimal() and ticks and ticks.isdecimal()
            and os.getenv("CUDA_VISIBLE_DEVICES") == gpu, "GPU holder identity is inconsistent")
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


def wait_ready(container, port):
    deadline = time.monotonic() + 480
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
    return result, status, failed


def check_mixed_soak(out, address, mode, container, pid):
    result, status, failed = check_mixed_load(out, address, container, pid, smoke=False)
    rss_gate = "rss_growth_second_half_at_most_zero"
    checks = result["checks"]
    require({rss_gate, "p95_drift_at_most_5_percent"} <= checks.keys(), "missing strict soak checks")
    rss = result["smaps_rollup_second_half"]
    delta = rss["rss_growth_kib"]
    require(checks[rss_gate] == (type(delta) is int and delta <= 0), "strict RSS verdict differs from the measurements")
    return {"mode": mode, "status": status, "failed_checks": failed,
            "rss": rss, "counts": result["counts"]}


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
              mixed_stage=False):
    require(GPU_UUID.fullmatch(gpu) and SHA256.fullmatch(model_sha), "pass a GPU UUID and model SHA-256")
    require(mixed_soak is None or (mixed_soak in CACHE_BLOCKS and not (accuracy_baseline or accuracy_holdout)),
            "select mixed soak or long accuracy for each container")
    require(not serving_smoke or not (accuracy_baseline or accuracy_holdout or mixed_soak),
            "select serving smoke separately from mixed soak and long accuracy")
    require(not mixed_stage or not (accuracy_baseline or accuracy_holdout or mixed_soak or serving_smoke),
            "select the mixed-stage probe separately from other workloads")
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
    holders = holder_pids(gpu)
    require(gpu_pids(gpu) == holders, "target GPU has an unexpected process before startup")
    out.mkdir()
    accuracy = {}
    soak = None
    smoke = None
    stage = None
    server_args = ()
    if accuracy_baseline or accuracy_holdout:
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
            "--gpus", f"device={gpu}", "--env", "NVIDIA_DRIVER_CAPABILITIES=compute,utility",
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
        ready = wait_ready(container, port)
        (out / "ready.json").write_text(json.dumps(ready, indent=2) + "\n")

        devices = docker("exec", container, "nvidia-smi", "--query-gpu=uuid,name,driver_version",
                         "--format=csv,noheader")
        (device,) = list(csv.reader(devices.splitlines(), skipinitialspace=True))
        require(len(device) == 3 and device[0] == gpu, "unexpected GPU visible in the container")
        riley_pid = runtime["State"]["Pid"]
        require(type(riley_pid) is int and riley_pid > 0 and gpu_pids(gpu) == holders | {riley_pid},
                "Riley does not own the expected CUDA context")
        observed, _ = docker("exec", container, "/bin/sh", "-ec",
                             'test "$(id -u):$(id -g)" = 65532:65532; '
                             f'test "$(readlink /proc/1/exe)" = {RILEY}; sha256sum /proc/1/exe').split()
        require(observed == binary, "running Riley differs from the reviewed distribution binary")
        log_command(out, "checkpoint-verify.log", ["docker", "exec", container, "/opt/riley/bin/riley-checkpoint",
                                                   "verify", "--checkpoint", "/models"], 600)

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
        if mixed_soak is not None:
            soak = check_mixed_soak(out, address, mixed_soak, container, riley_pid)
        if serving_smoke:
            observed_smoke, smoke_status, failed = check_mixed_load(
                out, address, container, riley_pid, smoke=True, expect_tools_unsupported=expect_tools_unsupported)
            smoke = {key: observed_smoke[key] for key in ("seconds", "rate", "counts", "counter_changes", "checks")}
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
        require(gpu_pids(gpu) == holders | {riley_pid}, "GPU process ownership changed during inference checks")
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
        wait_for_gpu_release(gpu, holders, riley_pid)
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
    parser.add_argument("--mixed-soak", choices=CACHE_BLOCKS, help="run the strict 30-minute W5 load in this cache mode")
    parser.add_argument("--serving-smoke", action="store_true",
                        help="combine API checks with 45 seconds of mixed traffic and cancellation")
    parser.add_argument("--mixed-stage", action="store_true",
                        help="compare singleton streams and witness committed mixed prefill/decode iterations")
    parser.add_argument("--smoke-kv-blocks", type=int,
                        help=f"set the serving smoke KV pool (default: {SMOKE_KV_BLOCKS} blocks)")
    args = parser.parse_args()

    def interrupted(_signal, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    return check_gpu(args.cpu_report, args.gpu_uuid, args.model_dir, args.model_manifest_sha256, args.output, args.tools,
                     accuracy_baseline=args.accuracy_baseline, accuracy_holdout=args.accuracy_holdout,
                     mixed_soak=args.mixed_soak, serving_smoke=args.serving_smoke, smoke_kv_blocks=args.smoke_kv_blocks,
                     expect_tools_unsupported=args.expect_tools_unsupported, mixed_stage=args.mixed_stage)


if __name__ == "__main__":
    raise SystemExit(main())
