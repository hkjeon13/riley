#!/usr/bin/env python3
"""Qualify actual batch4 GPU/HTTP checks and prepare exclusive paired plans.

The g04 numerical contract stays unchanged. GPU tests must have executed on
the new clean batch4 snapshot and binaries; historical component equivalence
is not accepted. This script executes neither GPU work nor measurements.
"""

import argparse
import copy
import hashlib
import json
import subprocess
from pathlib import Path

import prepare_batch2_plan as prior
import run_batch4_tests as checks


PROFILE = "vllm-smol-p128-v1"
GATE = "g04-vllm-smol-p128-v1"
IMPLEMENTATION = "g08-packed-decode-v1"
PARENT_COMMIT = "83cda03e183e29f48a82f7e7be3282c2e83ee186"
CHANGED_FILES = {
    "crates/riley-cuda/src/ffi.rs",
    "crates/riley-cuda/src/graph_resources.rs",
    "crates/riley-runtime/src/llama/graph_decode_full.rs",
    "crates/riley-runtime/src/llama/graph_decode_packed.rs",
    "crates/riley-runtime/src/llama/graph_decode_prefill_parity_gpu.rs",
    "kernels/src/graph_resources.cu",
    "kernels/src/graph_numerics_precise.cu",
    "kernels/src/ffi_internal.hpp",
    "kernels/src/gemm.cu",
    "kernels/include/riley_cuda.h",
}
NEW_MODULE = "crates/riley-runtime/src/llama/graph_decode_packed.rs"
# 30 layers of packed QKV/gate-up BF16 weights plus two M=1 output buffers.
# This is a source-derived size, not an observed GPU allocation or timing.
PACKED_BUFFER_BYTES = 2 * (576 + 2 * 192 + 2 * 1536) * (30 * 576 + 1)

read, require, sha, evidence = prior.read, prior.require, prior.sha, prior.evidence


def git(source, *args):
    return subprocess.check_output(["git", "-C", str(source), *args], text=True).strip()


def source_files_match(source, files):
    require(bool(files), "source receipt is empty")
    for name, expected in files.items():
        path = source / name
        require(not Path(name).is_absolute() and prior.below(path, source), "source path escapes snapshot")
        require(sha(path) == expected, "source file changed: " + str(path))


def validate_build(root, build, parent, overlay):
    source, parent_source = root / "batch4-source", root / "batch3-source"
    require(parent["source_commit"] == PARENT_COMMIT
            and git(parent_source, "rev-parse", "HEAD") == PARENT_COMMIT,
            "batch3 parent identity differs")
    require(not git(parent_source, "status", "--porcelain", "--untracked-files=all"), "batch3 parent is dirty")
    source_files_match(parent_source, parent["source_files"])
    parent_binaries = {str(root / "batch3-target/release" / name): sha(root / "batch3-target/release" / name)
                       for name in ("riley", "riley-profile")}
    require(parent["binaries"] == parent_binaries, "batch3 parent binaries changed")
    require(build["parent_source_commit"] == PARENT_COMMIT
            and build["parent_build_sha256"] == sha(root / "batch3-build.json"), "parent build receipt differs")
    require(build["source_root"] == str(source), "batch4 source root differs")
    commit = git(source, "rev-parse", "HEAD")
    require(commit == build["source_commit"] and git(source, "rev-parse", "HEAD^") == PARENT_COMMIT,
            "batch4 is not the built direct child of batch3")
    require(not git(source, "status", "--porcelain", "--untracked-files=all"), "batch4 source is dirty")
    # A full tracked-tree diff proves every file outside this exact set stayed
    # unchanged, not merely the smaller list present in the build receipt.
    changed = git(source, "diff", "--name-status", "--no-renames", PARENT_COMMIT, "HEAD")
    actual = {}
    for line in changed.splitlines():
        status, name = line.split("\t", 1)
        require(name not in actual, "duplicate changed path")
        actual[name] = status
    expected = {name: "A" if name == NEW_MODULE else "M" for name in CHANGED_FILES}
    require(actual == expected, "batch4 must change exactly the 10 declared files with one added module")
    require(build["changed_files"] == sorted(CHANGED_FILES) and set(overlay) == CHANGED_FILES,
            "build or overlay changed-file manifest differs")
    require(build["source_files"] == {**parent["source_files"], **overlay},
            "batch4 source pins must contain all parent pins plus the exact overlay")
    source_files_match(source, build["source_files"])
    expected_commands = [
        ["cargo", "test", "--release", "-p", "riley-runtime", "--features", "cuda", "--lib", "--no-run"],
        ["cargo", "build", "--release", "-p", "riley-server", "--features", "server,bench,cuda",
         "--bin", "riley", "--bin", "riley-profile"],
    ]
    require(build["build_argv"] == expected_commands, "batch4 build command differs")
    require(build["build_environment"] == {
        "CUDA_HOME": "/data/riley-g04-cuda13", "CUDAToolkit_ROOT": "/data/riley-g04-cuda13",
        "CMAKE": "/data/cmake-3.31.12/bin/cmake", "CMAKE_BUILD_PARALLEL_LEVEL": "4",
        "CARGO_BUILD_JOBS": "4", "CARGO_TARGET_DIR": str(root / "batch4-target"),
        "LD_LIBRARY_PATH": "/data/riley-g04-cuda13/lib",
    }, "batch4 build environment differs")
    require(build["build_log_sha256"] == sha(root / "batch4-build.log"), "batch4 build log changed")
    require(build["gpu_tests_executed"] is False and build["performance_measured"] is False,
            "compile-only build receipt contains execution claims")
    binaries = {str(root / "batch4-target/release" / name): sha(root / "batch4-target/release" / name)
                for name in ("riley", "riley-profile")}
    require(build["binaries"] == binaries, "batch4 release binaries changed")
    return source, commit, binaries


def validate_gpu(root, receipt, commit, binaries, reference_binding):
    prior.validate_gpu(receipt, commit, binaries, reference_binding)
    require(receipt["source_clean"] is True and receipt["gpu_tests_executed"] is True,
            "GPU tests did not execute on the clean current snapshot")
    require(receipt["build_sha256"] == sha(root / "batch4-build.json"), "GPU build identity differs")
    require(receipt["performance_measured"] is False, "GPU correctness contains performance trials")
    require(receipt["runner_sha256"] == sha(Path(checks.__file__)), "GPU test runner changed")
    require(receipt["http_helper_sha256"] == sha(root / "batch4_http_check.py"), "HTTP helper changed")
    require({item["name"] for item in receipt["checks"]} == set(checks.GPU_TESTS),
            "all four current-source GPU checks are required")
    for item in receipt["checks"]:
        name, path = item["name"], root / ("batch4-" + item["name"] + ".log")
        require(item["path"] == str(path), "GPU log is not the batch4 check log")
        require(item["argv"] == ["cargo", "test", "--release", "-p", "riley-runtime", "--features", "cuda",
                                  "--lib", name, "--", "--ignored", "--nocapture", "--test-threads=1"],
                "GPU test argv differs: " + name)
        counts = checks.validate_test_log(path, name)
        require(all(item[key] == value for key, value in counts.items()), "GPU test counts differ")
    profile, path = receipt["profile_unit_tests"], root / "batch4-profile-unit-tests.log"
    require(profile["name"] == "profile-unit-tests" and profile["passed"] is True,
            "current profile unit tests did not pass")
    require(profile["path"] == str(path) and profile["sha256"] == sha(path), "profile unit log changed")
    require(profile["argv"] == ["cargo", "test", "--release", "-p", "riley-server", "--features",
                                 "server,bench,cuda", "--bin", "riley-profile", "--", "--test-threads=1"],
            "profile unit test argv differs")
    counts = checks.validate_test_log(path)
    require(all(profile[key] == value for key, value in counts.items()), "profile unit test counts differ")


def validate_http(root, receipt, commit, binaries, binding, base, model):
    prior.validate_http(receipt, binaries, binding, model)
    require(receipt["source_commit"] == commit and receipt["source_clean"] is True,
            "HTTP source differs from the current clean snapshot")
    require(receipt["build_sha256"] == sha(root / "batch4-build.json"), "HTTP build receipt differs")
    require(receipt["reference_binding_sha256"] == sha(base / "native-binding.json"), "HTTP reference differs")
    require(receipt["references"] == {str(base / name): sha(base / name)
                                      for name in ("native-binding.json", "request.json")}, "HTTP reference pins differ")
    expected_model = {str(Path(model) / name): binding["workload"][field]
                      for name, field in (("model.safetensors", "weights_sha256"),
                                          ("tokenizer.json", "tokenizer_sha256"))}
    require(receipt["model_files"] == expected_model, "HTTP model pins differ")
    require(receipt["runner_sha256"] == sha(root / "batch4_http_check.py")
            and receipt["shared_runner_sha256"] == sha(Path(checks.__file__)), "HTTP checker identity changed")
    for result in receipt["results"]:
        path = root / "batch4-http-correctness" / ("http-" + result["sampling"] + ".log")
        require(result["graceful_exit_code"] == 0 and result["log"] == evidence(path),
                "HTTP log or graceful shutdown evidence differs")
        require(result["disconnect_points"] == ["after_headers_before_reading_text", "after_first_nonempty_text"],
                "HTTP disconnect coverage differs")


def prepare(root, base, template_path, http_runner, engine_runner):
    root, base, template_path = root.resolve(), base.resolve(), template_path.resolve()
    outputs = {name: root / ("batch4-" + name + ".json")
               for name in ("qualification", "binding", "http-plan", "engine-plan")}
    for path in outputs.values():
        require(not path.exists(), "refusing to replace existing output: " + str(path))
    template, binding = read(template_path), read(base / "native-binding.json")
    request = read(base / "request.json")
    require(template["source_commit"] == PARENT_COMMIT
            and template["source_root"] == str(root / "batch3-source")
            and template["implementation_id"] == "g07-http-wakeup-v1", "template must identify frozen batch3")
    require(template["numerical_profile"] == PROFILE and binding["source"]["correctness_gate_id"] == GATE,
            "numerical profile or reference gate differs")
    require(prior.option(template["engine_lanes"]["riley"]["argv"], "--correctness-gate-id") == GATE,
            "template engine gate differs")
    workload = binding["workload"]
    for key, value in {"concurrency": 1, "prompt_tokens": 128, "output_tokens": 32,
                       "warmups": 5, "measured_iterations": 30, "sampling_id": "greedy"}.items():
        require(workload[key] == value, "fixed workload differs: " + key)
    require(len(binding["input_token_ids"]) == 128 and len(binding["generated_token_ids"]) == 32
            and request["prompt_token_ids"] == binding["input_token_ids"], "fixed token reference differs")
    model = prior.option(template["http_lanes"]["riley"]["argv"], "--model")
    require(prior.option(template["engine_lanes"]["riley"]["argv"], "--model") == model,
            "HTTP and engine model paths differ")
    for flag in ("--batch-token-budget", "--prefill-chunk-tokens"):
        require(prior.option(template["http_lanes"]["riley"]["argv"], flag) == "128", "template P128 budget differs")

    build_path, parent_path, overlay_path = (root / name for name in
                                           ("batch4-build.json", "batch3-build.json", "batch4-source-overlay.json"))
    gpu_path, http_path = root / "batch4-gpu-tests.json", root / "batch4-http-correctness/http-validation.json"
    parent_qualification_path = root / "batch3-qualification.json"
    build, parent, overlay, gpu, http = map(read, (build_path, parent_path, overlay_path, gpu_path, http_path))
    source, commit, binaries = validate_build(root, build, parent, overlay)
    validate_gpu(root, gpu, commit, binaries, base / "native-binding.json")
    validate_http(root, http, commit, binaries, binding, base, model)
    parent_qualification = read(parent_qualification_path)
    require(parent_qualification["passed"] is True and parent_qualification["source_commit"] == PARENT_COMMIT
            and parent_qualification["binaries"] == parent["binaries"]
            and parent_qualification["build"] == evidence(parent_path), "batch3 parent qualification differs")
    require(template["immutable_files"].get(str(parent_qualification_path)) == sha(parent_qualification_path),
            "template does not pin the batch3 parent qualification")

    # Preserve reference/model/environment pins. Prior candidate identities are
    # retained below only as explicitly named lineage, never as new test proof.
    immutable = {path: digest for path, digest in template["immutable_files"].items()
                 if not prior.below(path, root) and not prior.below(path, template["source_root"])}
    for path, expected in immutable.items():
        require(sha(path) == expected, "retained template artifact changed: " + path)
    for name in ("request.json", "native-binding.json", "verification.json", "prompts.jsonl",
                 "vllm_lane.py", "environment.json", "candidate.json"):
        require(str(base / name) in immutable, "template lacks reference artifact pin: " + name)
    for name, expected in (("model.safetensors", workload["weights_sha256"]),
                           ("tokenizer.json", workload["tokenizer_sha256"])):
        require(immutable.get(str(Path(model) / name)) == expected, "model data pin differs: " + name)

    profile_binary, http_binary = (str(root / "batch4-target/release" / name) for name in ("riley-profile", "riley"))
    qualification = {
        "schema_version": "riley.packed-decode-qualification.v1", "passed": True,
        "source_root": str(source), "source_commit": commit, "source_clean": True,
        "implementation_id": IMPLEMENTATION, "numerical_profile": PROFILE, "correctness_gate_id": GATE,
        "scope": "SmolLM2-135M BF16 c1 P128 O32; current packed decode projections and fused RoPE/KV publication. "
                 "Actual current-source GPU full logits/status/KV parity, retained reuse, cancellation, rejection, "
                 "scheduler block mapping tests, and release HTTP reference text/SSE/disconnect checks.",
        "build": evidence(build_path), "gpu_tests": evidence(gpu_path), "gpu_test_logs": gpu["checks"],
        "profile_unit_tests": gpu["profile_unit_tests"], "release_http": evidence(http_path), "binaries": binaries,
        "gpu_tests_executed_on_current_snapshot": True, "vllm_reference_tokens_exact": True,
        "full_logits_and_kv_exact": True,
        "source_lineage": {"parent_commit": PARENT_COMMIT, "parent_build": evidence(parent_path),
                           "parent_qualification": evidence(parent_qualification_path),
                           "changed_files": sorted(CHANGED_FILES), "other_parent_tracked_files_unchanged": True,
                           "overlay": evidence(overlay_path)},
        "source_derived_packed_device_buffer_bytes": PACKED_BUFFER_BYTES,
        "packed_buffer_size_basis": "30 layers of QKV/gate-up BF16 weight copies plus two M1 output buffers; source arithmetic only",
        "packed_device_buffer_bytes_measured": False,
        "reference_only": {"binding": evidence(base / "native-binding.json"),
                           "verification": evidence(base / "verification.json")},
        "canonical_e0_claim": False, "performance_measured": False,
    }
    qualification_bytes = prior.encoded(qualification)
    qualification_sha = hashlib.sha256(qualification_bytes).hexdigest()
    binding["source"].update(git_commit=commit, git_dirty=False, implementation_id=IMPLEMENTATION,
                             correctness_gate_id=GATE, correctness_report_sha256=qualification_sha,
                             executable_sha256=binaries[profile_binary])
    binding_bytes = prior.encoded(binding)
    plan = copy.deepcopy(template)
    plan.pop("live_preflight_last_failure", None)
    plan.update(source_root=str(source), source_commit=commit, implementation_id=IMPLEMENTATION,
                measurement_started=False, comparison_status="Packed decode correctness qualified on current snapshot; performance pending",
                qualification_blockers=[], measurement_environment_blockers=[], prepare_only_argv=[],
                qualification_path=str(outputs["qualification"]), binding_path=str(outputs["binding"]),
                request_path=str(base / "request.json"), canonical_e0_claim=False,
                binding_note="The g04 numerical contract is unchanged. The g08 implementation, clean source and binaries "
                             "have new actual GPU and HTTP correctness receipts. No performance result is inferred.")
    http_argv, engine_argv = plan["http_lanes"]["riley"]["argv"], plan["engine_lanes"]["riley"]["argv"]
    http_argv[0], engine_argv[0] = http_binary, profile_binary
    for flag, value in {"--git-commit": commit, "--git-dirty": "false", "--implementation-id": IMPLEMENTATION,
                        "--correctness-gate-id": GATE, "--correctness-report-sha256": qualification_sha,
                        "--executable-sha256": binaries[profile_binary], "--run-id": "g08-riley-{index}"}.items():
        prior.replace(engine_argv, flag, value)

    new_pins = [build_path, parent_path, parent_qualification_path, overlay_path, gpu_path, http_path, template_path,
                Path(__file__).resolve(), Path(prior.__file__).resolve(), Path(checks.__file__).resolve(),
                root / "build_batch4.py", root / "batch4_http_check.py", root / "batch4-build.log",
                http_runner, engine_runner, source / "benchmarks/scripts/check_vllm_profile_run.py",
                source / "benchmarks/scripts/preflight.sh", Path(gpu["profile_unit_tests"]["path"])]
    new_pins.extend(Path(item["path"]) for item in gpu["checks"])
    new_pins.extend(Path(item["log"]["path"]) for item in http["results"])
    new_pins.extend(source / name for name in sorted(build["source_files"]))
    reference_argv = plan["engine_lanes"]["vllm"]["argv"]
    new_pins.extend([Path(reference_argv[0]), Path(reference_argv[1]),
                     Path(prior.option(reference_argv, "--matrix")), Path(plan["reference_checker_python"])])
    reference_source = Path(read(base / "candidate.json")["source_root"])
    new_pins.extend(sorted((reference_source / "benchmarks/lanes/vllm").rglob("*.py")))
    for path in new_pins:
        path = path.absolute()
        current = sha(path)
        require(str(path) not in immutable or immutable[str(path)] == current, "pinned artifact changed: " + str(path))
        immutable[str(path)] = current
    immutable.update(binaries)
    immutable[str(outputs["qualification"])] = qualification_sha
    immutable[str(outputs["binding"])] = hashlib.sha256(binding_bytes).hexdigest()
    plan["immutable_files"] = immutable
    http_plan, engine_plan = copy.deepcopy(plan), copy.deepcopy(plan)
    http_plan.update(measurement_mode="http", runner=str(http_runner))
    engine_plan.update(measurement_mode="engine", runner=str(engine_runner))
    payloads = {"qualification": qualification_bytes, "binding": binding_bytes,
                "http-plan": prior.encoded(http_plan), "engine-plan": prior.encoded(engine_plan)}
    for name, payload in payloads.items():
        with outputs[name].open("xb") as stream:
            stream.write(payload)
    return {"source_commit": commit, "implementation_id": IMPLEMENTATION,
            "correctness_report_sha256": qualification_sha,
            "outputs": {name: str(path) for name, path in outputs.items()}, "measurement_started": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/tmp/riley-opt-260912"))
    parser.add_argument("--base", type=Path, default=Path("/tmp/riley-g04-vllm-profile-260911"))
    parser.add_argument("--template", type=Path)
    parser.add_argument("--http-runner", type=Path)
    parser.add_argument("--engine-runner", type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(
        args.root, args.base, args.template or args.root / "batch3-engine-plan.json",
        (args.http_runner or args.root / "run_serving_optimization.py").absolute(),
        (args.engine_runner or args.root / "run_engine_optimization.py").absolute(),
    ), indent=2))


if __name__ == "__main__":
    main()
