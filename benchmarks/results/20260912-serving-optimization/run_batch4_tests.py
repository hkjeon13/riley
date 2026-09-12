#!/usr/bin/env python3
"""Correctness-only checks for the frozen batch4 snapshot; no performance run.

The desktop/Blender session is left untouched. Every log and receipt is created
exclusively. Run this only after batch4-build.json exists; the source and release
binaries are checked before and after every test and the HTTP helper.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


DEFAULT_ROOT = Path("/tmp/riley-opt-260912")
DEFAULT_BASE = Path("/tmp/riley-g04-vllm-profile-260911")
MODEL = Path("/data/riley-benchmark/20260827T051948Z-d7ad713a/model")
HTTP_PYTHON = Path("/data/riley-vllm-interim.CfrT9T/venv/bin/python")
GPU_TESTS = (
    "batched_prefill_retained_reuse_cancel_and_rejected_output_invalidation",
    "batched_prefill_exact_logits_status_and_every_decode_kv",
    "owned_graph_smol_p128_o32_reuses_scheduler_block_mappings",
    "owned_graph_vllm_smol_p128_o32_reuses_scheduler_block_mappings",
)
PARITY_MARKERS = {
    GPU_TESTS[0]: "P128_PREFILL_REUSE requests=6 retained_slow_replays=899 "
    "retained_fast_replays=137 cancelled_after_prefill=1 cancelled_after_decode=1 "
    "invalid_shape_stage_cases=42 raw_outputs_exact=true full_initialized_final_kv_exact=true "
    "zero_allocations=true performance_claim=false",
    GPU_TESTS[1]: "P128_PREFILL_PARITY prompts=3 full_logits_exact=true "
    "argmax_status_exact=true full_initialized_kv_snapshots=96 every_decode_kv_exact=true "
    "snapshot_method=close_and_reopen zero_allocations=true performance_claim=false",
}


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


def verify_snapshot(root, build, build_sha256):
    source = root / "batch4-source"
    require(sha(root / "batch4-build.json") == build_sha256, "build receipt changed")
    require(build["source_root"] == str(source), "build source root differs")
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=all"], text=True
    )
    require(commit == build["source_commit"] and not dirty, "source is not the frozen clean build")
    require(bool(build["source_files"]), "build source receipts are empty")
    for name, expected in build["source_files"].items():
        path = source / name
        require(not Path(name).is_absolute() and path.resolve().is_relative_to(source.resolve()),
                "source receipt path escapes snapshot")
        require(sha(path) == expected, "source file changed: " + name)
    binaries = {str(root / "batch4-target/release" / name): sha(root / "batch4-target/release" / name)
                for name in ("riley", "riley-profile")}
    require(binaries == build["binaries"], "release binaries differ from frozen build")
    return {"source_commit": commit, "source_clean": True, "binaries": binaries}


def environment(root):
    env = dict(os.environ)
    env.update(CUDA_HOME="/data/riley-g04-cuda13", CUDAToolkit_ROOT="/data/riley-g04-cuda13",
               CMAKE="/data/cmake-3.31.12/bin/cmake", CMAKE_BUILD_PARALLEL_LEVEL="4",
               CARGO_BUILD_JOBS="4", CARGO_TARGET_DIR=str(root / "batch4-target"),
               LD_LIBRARY_PATH="/data/riley-g04-cuda13/lib", RILEY_REAL_CHECKPOINT=str(MODEL))
    env["PATH"] = "/home/psyche/.cargo/bin:/data/riley-g04-cuda13/bin:" + env["PATH"]
    return env


def validate_test_log(path, name=None):
    content = path.read_text(encoding="utf-8")
    results = re.findall(r"test result: ok\. (\d+) passed; (\d+) failed; (\d+) ignored;", content)
    require(len(results) == 1, "expected exactly one successful Rust test summary: " + str(path))
    passed, failed, ignored = map(int, results[0])
    require(passed > 0 and failed == 0, "Rust test did not execute a nonzero passing sample")
    if name is not None:
        require(passed == 1 and ignored == 0, "named GPU check must execute exactly one test")
        require(re.search(r"(?:^|::)" + re.escape(name) + r"(?:\s|\.)", content, re.MULTILINE),
                "GPU log lacks the requested test name: " + name)
        if name in PARITY_MARKERS:
            require(PARITY_MARKERS[name] in content, "GPU log lacks exact parity evidence: " + name)
    return {"passed_tests": passed, "failed_tests": failed, "ignored_tests": ignored}


def run_check(root, build, build_sha256, env, name, argv, *, gpu):
    before = verify_snapshot(root, build, build_sha256)
    path = root / ("batch4-" + name + ".log")
    with path.open("x", encoding="utf-8") as log:
        subprocess.run(argv, cwd=root / "batch4-source", env=env,
                       stdout=log, stderr=log, check=True)
    after = verify_snapshot(root, build, build_sha256)
    require(before == after, "snapshot changed during test")
    counts = validate_test_log(path, name if gpu else None)
    check = {"name": name, "passed": True, "path": str(path), "sha256": sha(path),
             "argv": argv, **counts}
    print(json.dumps(check), flush=True)
    return check


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--http-python", type=Path, default=HTTP_PYTHON)
    parser.add_argument("--http-script", type=Path)
    args = parser.parse_args()
    root, base = args.root.resolve(), args.base.resolve()
    helper = (args.http_script or root / "batch4_http_check.py").resolve()
    build_path = root / "batch4-build.json"
    build, build_sha256 = read(build_path), sha(build_path)
    reference = base / "native-binding.json"
    reference_sha256, helper_sha256 = sha(reference), sha(helper)
    runner_sha256 = sha(Path(__file__))
    initial = verify_snapshot(root, build, build_sha256)
    env = environment(root)
    checks = []
    for name in GPU_TESTS:
        argv = ["cargo", "test", "--release", "-p", "riley-runtime", "--features", "cuda", "--lib",
                name, "--", "--ignored", "--nocapture", "--test-threads=1"]
        checks.append(run_check(root, build, build_sha256, env, name, argv, gpu=True))
    profile = run_check(
        root, build, build_sha256, env, "profile-unit-tests",
        ["cargo", "test", "--release", "-p", "riley-server", "--features", "server,bench,cuda",
         "--bin", "riley-profile", "--", "--test-threads=1"], gpu=False,
    )
    final = verify_snapshot(root, build, build_sha256)
    require(initial == final and sha(reference) == reference_sha256
            and sha(Path(__file__)) == runner_sha256, "qualification identity changed")
    result = {"schema_version": "riley.batch4-gpu-correctness.v1", "passed": True,
              **final, "build_sha256": build_sha256, "checks": checks, "profile_unit_tests": profile,
              "reference_binding_sha256": reference_sha256,
              "gpu_tests_executed": True,
              "vllm_reference_tokens_exact": True, "full_logits_and_kv_exact": True,
              "runner_sha256": runner_sha256, "http_helper_sha256": helper_sha256,
              "desktop_session_policy": "left untouched; no performance qualification",
              "performance_claim_eligible": False, "performance_measured": False}
    write_new(root / "batch4-gpu-tests.json", result)
    require(sha(helper) == helper_sha256, "HTTP helper changed before execution")
    subprocess.run([str(args.http_python), str(helper), "--root", str(root), "--base", str(base)],
                   cwd=root / "batch4-source", env=env, check=True)
    require(verify_snapshot(root, build, build_sha256) == final
            and sha(reference) == reference_sha256 and sha(helper) == helper_sha256
            and sha(Path(__file__)) == runner_sha256,
            "snapshot or reference changed during HTTP qualification")
    print(json.dumps({"gpu_tests": str(root / "batch4-gpu-tests.json"),
                      "http_validation": str(root / "batch4-http-correctness/http-validation.json"),
                      "passed": True, "performance_measured": False}), flush=True)


if __name__ == "__main__":
    main()
