#!/usr/bin/env python3
"""Build the HTTP-only batch from frozen batch2; never copy the local worktree.

Run outside an active measurement. This script builds binaries and runs the
server's CPU-only library tests. It does not run CUDA tests or HTTP requests.
All new directories, logs and receipts are exclusive, preserving prior runs.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path


BATCH2_COMMIT = "1bfb23053fb15a3528eb6413abe6e1bbdb0db004"
SERVICE = "crates/riley-server/src/service.rs"
SERVICE_SHA = "b6cfe46e9b6ad3efdea6505408d955389ff5b114e090eed1a2a8e358e029eca6"
FROZEN_FILES = {
    "crates/riley-cuda/src/ffi.rs",
    "crates/riley-cuda/src/graph_resources.rs",
    "crates/riley-runtime/src/llama/graph_decode_full.rs",
    "crates/riley-runtime/src/llama/graph_decode_prefill_parity_gpu.rs",
    "crates/riley-runtime/src/llama/executor/config.rs",
    "crates/riley-runtime/src/llama/executor/owner.rs",
    "crates/riley-runtime/src/llama/batch_executor.rs",
    "crates/riley-server/src/engine.rs",
    "crates/riley-server/src/benchmark.rs",
    "crates/riley-server/src/main.rs",
    "crates/riley-server/src/bin/riley-profile.rs",
    "kernels/src/graph_resources.cu",
    "kernels/src/primitives.cu",
    "kernels/src/graph_numerics.cu",
    "kernels/src/graph_numerics_precise.cu",
    "kernels/src/ffi_internal.hpp",
    "kernels/include/riley_cuda.h",
}
CPU_COMMAND = ["cargo", "test", "--release", "-p", "riley-server", "--features", "server", "--lib"]
CPU_TESTS = (
    "accepted_connection_waits_for_delayed_request_bytes",
    "shutdown_wakes_idle_listener_and_all_workers_without_tcp_connections",
    "listener_shutdown_wakeup_persists_and_takes_priority_over_backlog",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def evidence(path):
    return {"path": str(path), "sha256": sha(path)}


def git(source, *args):
    return subprocess.check_output(["git", "-C", str(source), *args], text=True).strip()


def validate_frozen(root, receipt):
    source = root / "batch2-source"
    require(receipt["source_commit"] == BATCH2_COMMIT, "unexpected frozen batch2 commit")
    require(git(source, "rev-parse", "HEAD") == BATCH2_COMMIT, "batch2 checkout moved")
    require(not git(source, "status", "--porcelain", "--untracked-files=all"), "batch2 source is dirty")
    require(set(receipt["source_files"]) == FROZEN_FILES, "expected exactly 17 batch2 source pins")
    for relative, expected in receipt["source_files"].items():
        require(sha(source / relative) == expected, "frozen file changed: " + relative)
    expected_binaries = {str(root / "batch2-target/release" / name):
                         sha(root / "batch2-target/release" / name)
                         for name in ("riley", "riley-profile")}
    require(receipt["binaries"] == expected_binaries, "batch2 binaries changed")


def component_parity(root, receipt):
    source = root / "batch3-source"
    require(not git(source, "status", "--porcelain", "--untracked-files=all"), "batch3 source is dirty")
    require(git(source, "rev-parse", "HEAD^") == BATCH2_COMMIT, "batch3 is not a direct child of batch2")
    changed = git(source, "diff", "--name-only", BATCH2_COMMIT, "HEAD").splitlines()
    require(changed == [SERVICE], "batch3 must change only service.rs: " + repr(changed))
    require(sha(source / SERVICE) == SERVICE_SHA, "service overlay identity differs")
    require(set(receipt["source_files"]) == FROZEN_FILES, "expected exactly 17 frozen files")
    parity = {}
    for relative, expected in sorted(receipt["source_files"].items()):
        prior, current = sha(root / "batch2-source" / relative), sha(source / relative)
        require(prior == current == expected, "runtime/native component differs: " + relative)
        parity[relative] = {"batch2_sha256": prior, "batch3_sha256": current, "equal": True}
    return parity


def validate_cpu_log(path):
    text = Path(path).read_text(encoding="utf-8")
    result = re.search(r"test result: ok\. (\d+) passed; 0 failed; (\d+) ignored;", text)
    require(result is not None and int(result[1]) > 0, "CPU suite did not pass")
    for name in CPU_TESTS:
        require(re.search(r"test service::tests::" + name + r" \.\.\. ok(?:\n|$)", text),
                "new HTTP lifecycle test did not pass: " + name)
    return {"passed": int(result[1]), "failed": 0, "ignored": int(result[2])}


def build(root):
    root = root.resolve()
    source, target = root / "batch3-source", root / "batch3-target"
    cpu_target = root / "batch3-cpu-target"
    overlay = root / "service-batch3.rs"
    baseline_path = root / "batch2-build.json"
    baseline = read(baseline_path)
    validate_frozen(root, baseline)
    require(sha(overlay) == SERVICE_SHA, "transported service.rs has the wrong SHA256")
    for path in (source, target, cpu_target, root / "batch3-build.log",
                 root / "batch3-cpu-tests.log", root / "batch3-cpu-tests.json", root / "batch3-build.json"):
        require(not path.exists(), "refusing to overwrite: " + str(path))

    subprocess.run(["git", "clone", "--no-hardlinks", "--quiet", str(root / "batch2-source"), str(source)], check=True)
    shutil.copyfile(overlay, source / SERVICE)
    require(git(source, "diff", "--name-only").splitlines() == [SERVICE], "overlay changed another file")
    subprocess.run(["git", "add", "--", SERVICE], cwd=source, check=True)
    subprocess.run(["git", "-c", "user.name=Codex", "-c", "user.email=codex@local", "commit", "--quiet",
                    "-m", "snapshot: HTTP readiness and worker wakeup batch"], cwd=source, check=True)
    parity = component_parity(root, baseline)
    commit = git(source, "rev-parse", "HEAD")

    (target / "release").mkdir(parents=True)
    subprocess.run(["rsync", "-a", "--exclude=riley-cuda-*", "--exclude=libriley_cuda-*",
                    str(root / "batch2-target/release") + "/", str(target / "release") + "/"], check=True)
    env = os.environ.copy()
    fixed_environment = {
        "CUDA_HOME": "/data/riley-g04-cuda13", "CUDAToolkit_ROOT": "/data/riley-g04-cuda13",
        "CMAKE": "/data/cmake-3.31.12/bin/cmake", "CMAKE_BUILD_PARALLEL_LEVEL": "4",
        "CARGO_BUILD_JOBS": "4", "LD_LIBRARY_PATH": "/data/riley-g04-cuda13/lib",
    }
    env.update(fixed_environment)
    env["PATH"] = "/home/psyche/.cargo/bin:/data/riley-g04-cuda13/bin:" + env["PATH"]
    cpu_env = dict(env, CARGO_TARGET_DIR=str(cpu_target))
    cpu_log = root / "batch3-cpu-tests.log"
    with cpu_log.open("x") as log:
        subprocess.run(CPU_COMMAND, cwd=source, env=cpu_env, stdout=log, stderr=log, check=True)
    cpu_counts = validate_cpu_log(cpu_log)
    cpu_receipt = {
        "passed": True, "source_commit": commit, "source_clean": True, "service_sha256": SERVICE_SHA,
        "argv": CPU_COMMAND, "features": ["server"], "target_dir": str(cpu_target),
        "log": evidence(cpu_log), "test_counts": cpu_counts, "required_tests": list(CPU_TESTS),
        "gpu_executed": False, "transport_timing_executed": False,
    }
    env["CARGO_TARGET_DIR"] = str(target)
    command = ["cargo", "build", "--release", "-p", "riley-server", "--features", "server,bench,cuda",
               "--bin", "riley", "--bin", "riley-profile"]
    build_log = root / "batch3-build.log"
    with build_log.open("x") as log:
        subprocess.run(command, cwd=source, env=env, stdout=log, stderr=log, check=True)
    validate_frozen(root, baseline)
    require(component_parity(root, baseline) == parity, "source changed during build")
    require(git(source, "rev-parse", "HEAD") == commit, "batch3 commit changed during build")
    cpu_path = root / "batch3-cpu-tests.json"
    write_new(cpu_path, cpu_receipt)
    receipt = {
        "schema_version": "riley.http-wakeup-build.v1", "source_root": str(source),
        "source_commit": commit, "source_clean": True, "parent_source_commit": BATCH2_COMMIT,
        "only_changed_source_files": [SERVICE], "service_sha256": SERVICE_SHA,
        "overlay": evidence(overlay), "batch2_build": evidence(baseline_path),
        "source_files": dict(baseline["source_files"], **{SERVICE: SERVICE_SHA}),
        "batch2_component_parity": {"count": len(parity), "all_equal": True, "files": parity},
        "binaries": {str(target / "release" / name): sha(target / "release" / name)
                     for name in ("riley", "riley-profile")},
        "build_argv": command, "build_environment": fixed_environment, "build_log": evidence(build_log),
        "cpu_tests": evidence(cpu_path), "gpu_tests_executed": False, "performance_measured": False,
    }
    write_new(root / "batch3-build.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/tmp/riley-opt-260912"))
    print(json.dumps(build(parser.parse_args().root), indent=2))


if __name__ == "__main__":
    main()
