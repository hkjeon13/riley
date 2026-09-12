#!/usr/bin/env python3
"""Prepare batch3 plans from new HTTP/CPU checks and unchanged GPU components.

No measurement or GPU test is executed. Historical GPU logs remain bound to
batch2's commit and binaries. Their reuse is explicitly based on a one-file
HTTP-only commit delta and exact equality of all 17 batch2 component hashes.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import prepare_batch2_plan as prior
from build_batch3 import (BATCH2_COMMIT, CPU_COMMAND, CPU_TESTS, FROZEN_FILES,
                          SERVICE, SERVICE_SHA, component_parity, evidence, git,
                          read, require, sha, validate_cpu_log, validate_frozen)


PROFILE = "vllm-smol-p128-v1"
GATE = "g04-vllm-smol-p128-v1"
IMPLEMENTATION = "g07-http-wakeup-v1"


def validate_build(root, build, baseline):
    validate_frozen(root, baseline)
    source = root / "batch3-source"
    parity = component_parity(root, baseline)
    commit = git(source, "rev-parse", "HEAD")
    require(build["source_commit"] == commit and build["source_clean"] is True,
            "new build source identity differs")
    require(build["source_root"] == str(source), "new build source root differs")
    require(build["parent_source_commit"] == BATCH2_COMMIT, "new build parent differs")
    require(build["only_changed_source_files"] == [SERVICE], "build includes another source change")
    require(build["service_sha256"] == SERVICE_SHA, "new HTTP source identity differs")
    require(build["source_files"] == {**baseline["source_files"], SERVICE: SERVICE_SHA},
            "new source receipt differs from the 17 frozen files plus service.rs")
    require(build["batch2_build"] == evidence(root / "batch2-build.json"), "prior build receipt changed")
    require(build["batch2_component_parity"] == {"count": 17, "all_equal": True, "files": parity},
            "17-file component equivalence receipt differs")
    require(build["gpu_tests_executed"] is False and build["performance_measured"] is False,
            "build receipt unexpectedly claims GPU or performance tests")
    require(build["build_argv"] == ["cargo", "build", "--release", "-p", "riley-server", "--features",
                                    "server,bench,cuda", "--bin", "riley", "--bin", "riley-profile"],
            "new CUDA build command differs")
    require(build["build_environment"] == {
        "CUDA_HOME": "/data/riley-g04-cuda13", "CUDAToolkit_ROOT": "/data/riley-g04-cuda13",
        "CMAKE": "/data/cmake-3.31.12/bin/cmake", "CMAKE_BUILD_PARALLEL_LEVEL": "4",
        "CARGO_BUILD_JOBS": "4", "LD_LIBRARY_PATH": "/data/riley-g04-cuda13/lib",
    }, "new CUDA build environment differs from the frozen campaign")
    binaries = {str(root / "batch3-target/release" / name): sha(root / "batch3-target/release" / name)
                for name in ("riley", "riley-profile")}
    require(build["binaries"] == binaries, "batch3 binary hashes differ from build")
    require(build["build_log"] == evidence(root / "batch3-build.log"), "build log changed")
    require(build["cpu_tests"] == evidence(root / "batch3-cpu-tests.json"), "CPU receipt changed")
    return source, commit, binaries, parity


def validate_cpu(root, receipt, commit):
    require(receipt["passed"] is True and receipt["source_clean"] is True, "CPU tests did not pass cleanly")
    require(receipt["source_commit"] == commit and receipt["service_sha256"] == SERVICE_SHA,
            "CPU tests are not bound to the new HTTP source")
    require(receipt["argv"] == CPU_COMMAND and receipt["features"] == ["server"], "CPU test command differs")
    require(receipt["target_dir"] == str(root / "batch3-cpu-target"), "CPU target isolation differs")
    require(receipt["gpu_executed"] is False and receipt["transport_timing_executed"] is False,
            "CPU correctness receipt contains another test mode")
    require(receipt["required_tests"] == list(CPU_TESTS), "required HTTP lifecycle tests differ")
    log = root / "batch3-cpu-tests.log"
    require(receipt["log"] == evidence(log), "CPU test log changed")
    require(receipt["test_counts"] == validate_cpu_log(log), "CPU result counts differ")


def validate_new_http(root, receipt, build, binaries, binding, reference_binding, model):
    prior.validate_http(receipt, binaries, binding, model)
    require(receipt["passed"] is True, "new HTTP correctness did not pass")
    require(receipt["source_commit"] == build["source_commit"] and receipt["service_sha256"] == SERVICE_SHA,
            "HTTP receipt is not bound to the new source")
    require(receipt["build"] == evidence(root / "batch3-build.json"), "HTTP build identity differs")
    require(receipt["reference_binding"] == evidence(reference_binding), "HTTP reference binding differs")
    require(receipt["gpu_component_tests_rerun"] is False, "HTTP receipt mislabels component tests")
    for result in receipt["results"]:
        log = root / "batch3-http-correctness" / ("http-" + result["sampling"] + ".log")
        require(result["server_exit_code"] == 0, "new HTTP server did not exit cleanly")
        require(result["server_log"] == evidence(log), "HTTP server log changed")


def prepare(root, base, template_path, http_runner, engine_runner):
    root, base = root.resolve(), base.resolve()
    template_path = template_path.resolve()
    outputs = {name: root / ("batch3-" + name + ".json")
               for name in ("qualification", "binding", "http-plan", "engine-plan")}
    for path in outputs.values():
        require(not path.exists(), "refusing to replace output: " + str(path))
    template = read(template_path)
    reference_binding = base / "native-binding.json"
    binding, request = read(reference_binding), read(base / "request.json")
    require(template["source_commit"] == BATCH2_COMMIT
            and template["implementation_id"] == "g06-p128-prefill-v1",
            "template must identify frozen batch2")
    require(template["source_root"] == str(root / "batch2-source"), "template source root differs")
    require(template["numerical_profile"] == PROFILE, "template numerical profile differs")
    require(binding["source"]["correctness_gate_id"] == GATE, "reference numerical gate differs")
    require(prior.option(template["engine_lanes"]["riley"]["argv"], "--correctness-gate-id") == GATE,
            "template engine numerical gate differs")
    workload = binding["workload"]
    for key, value in {"concurrency": 1, "prompt_tokens": 128, "output_tokens": 32,
                       "warmups": 5, "measured_iterations": 30, "sampling_id": "greedy"}.items():
        require(workload[key] == value, "fixed workload differs: " + key)
    require(len(binding["input_token_ids"]) == 128 and len(binding["generated_token_ids"]) == 32,
            "fixed reference token lengths differ")
    require(request["prompt_token_ids"] == binding["input_token_ids"], "request reference differs")
    model = prior.option(template["http_lanes"]["riley"]["argv"], "--model")
    require(prior.option(template["engine_lanes"]["riley"]["argv"], "--model") == model,
            "engine and HTTP model paths differ")

    build_path, baseline_path = root / "batch3-build.json", root / "batch2-build.json"
    gpu_path, previous_qualification = root / "batch2-gpu-tests.json", root / "batch2-qualification.json"
    cpu_path, http_path = root / "batch3-cpu-tests.json", root / "batch3-http-correctness/http-validation.json"
    build, baseline, gpu, cpu, http = map(read, (build_path, baseline_path, gpu_path, cpu_path, http_path))
    source, commit, binaries, parity = validate_build(root, build, baseline)
    validate_cpu(root, cpu, commit)
    # Validate historical GPU evidence against its actual old source/binaries.
    # Never rewrite those identities to make it appear the tests ran on batch3.
    prior.validate_gpu(gpu, BATCH2_COMMIT, baseline["binaries"], reference_binding)
    previous = read(previous_qualification)
    require(previous["passed"] is True and previous["source_commit"] == BATCH2_COMMIT,
            "batch2 qualification did not pass for the frozen source")
    require(previous["build"] == evidence(baseline_path) and previous["gpu_tests"] == evidence(gpu_path),
            "prior qualification evidence changed")
    require(previous["binaries"] == baseline["binaries"], "prior qualified binaries differ")
    require(previous["correctness_gate_id"] == GATE and previous["numerical_profile"] == PROFILE,
            "prior qualification numerical contract differs")
    require(template["immutable_files"].get(str(previous_qualification)) == sha(previous_qualification),
            "template does not pin the prior qualification")
    validate_new_http(root, http, build, binaries, binding, reference_binding, model)

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

    profile_binary, http_binary = (str(root / "batch3-target/release" / name)
                                  for name in ("riley-profile", "riley"))
    qualification = {
        "schema_version": "riley.http-wakeup-qualification.v1", "passed": True,
        "source_root": str(source), "source_commit": commit, "source_clean": True,
        "implementation_id": IMPLEMENTATION, "numerical_profile": PROFILE, "correctness_gate_id": GATE,
        "scope": "HTTP-only readiness/wakeup change; SmolLM2-135M BF16 c1 P128 O32. "
                 "New server CPU lifecycle tests and release HTTP reference text, SSE, rejection, "
                 "disconnect/reuse and shutdown checks. GPU component correctness is reused by exact source equivalence.",
        "build": evidence(build_path), "cpu_tests": evidence(cpu_path), "release_http": evidence(http_path),
        "binaries": binaries, "new_binary_http_reference_text_exact": True,
        "gpu_component_tests_rerun_on_batch3": False,
        "gpu_component_evidence": {
            "basis": "direct child commit changes only service.rs; all 17 frozen runtime/native source hashes are equal",
            "source_commit": BATCH2_COMMIT, "source_root": str(root / "batch2-source"),
            "binaries": baseline["binaries"], "qualification": evidence(previous_qualification),
            "gpu_tests": evidence(gpu_path), "gpu_test_logs": gpu["checks"],
            "prior_vllm_reference_tokens_exact": True, "prior_full_logits_and_kv_exact": True,
            "only_changed_source_files": [SERVICE], "file_count": 17, "source_file_parity": parity,
            "note": "These GPU tests ran on batch2, not the new batch3 source or binaries.",
        },
        "reference_only": {"binding": evidence(reference_binding),
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
    plan.update(
        source_root=str(source), source_commit=commit, measurement_started=False,
        comparison_status="HTTP wakeup qualified; GPU component evidence reused by exact source equivalence; performance pending",
        qualification_blockers=[], measurement_environment_blockers=[], implementation_id=IMPLEMENTATION,
        qualification_path=str(outputs["qualification"]), binding_path=str(outputs["binding"]),
        request_path=str(base / "request.json"), prepare_only_argv=[], canonical_e0_claim=False,
        pairs=[{"index": i, "order": ["riley", "vllm"] if i % 2 else ["vllm", "riley"]} for i in range(1, 6)],
        warmups_per_process=5, measured_requests_per_process=30, fresh_process_per_lane_per_pair=True,
        binding_note="The g04 numerical gate is unchanged. New g07 source/binaries have HTTP and CPU correctness receipts; "
                     "batch2 GPU tests retain their old identities and are reused only through explicit source component equivalence.",
    )
    http_argv, engine_argv = plan["http_lanes"]["riley"]["argv"], plan["engine_lanes"]["riley"]["argv"]
    http_argv[0], engine_argv[0] = http_binary, profile_binary
    for flag in ("--batch-token-budget", "--prefill-chunk-tokens"):
        prior.replace(http_argv, flag, "128")
    for flag, value in {"--git-commit": commit, "--git-dirty": "false", "--implementation-id": IMPLEMENTATION,
                        "--correctness-gate-id": GATE, "--correctness-report-sha256": qualification_sha,
                        "--executable-sha256": binaries[profile_binary], "--run-id": "g07-riley-{index}"}.items():
        prior.replace(engine_argv, flag, value)

    scripts = Path(__file__).resolve().parent
    new_pins = [build_path, baseline_path, cpu_path, http_path, gpu_path, previous_qualification,
                template_path, Path(__file__).resolve(), scripts / "build_batch3.py",
                scripts / "batch3_http_check.py", Path(prior.__file__).resolve(),
                http_runner, engine_runner, root / "batch3-build.log", root / "batch3-cpu-tests.log",
                source / "benchmarks/scripts/check_vllm_profile_run.py", source / "benchmarks/scripts/preflight.sh"]
    new_pins.extend(Path(check["path"]) for check in gpu["checks"])
    new_pins.extend(Path(result["server_log"]["path"]) for result in http["results"])
    new_pins.extend(root / "batch2-source" / relative for relative in sorted(FROZEN_FILES))
    new_pins.extend(source / relative for relative in sorted(build["source_files"]))
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
    immutable.update(baseline["binaries"])
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
            "outputs": {name: str(path) for name, path in outputs.items()},
            "gpu_component_tests_rerun_on_batch3": False, "measurement_started": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/tmp/riley-opt-260912"))
    parser.add_argument("--base", type=Path, default=Path("/tmp/riley-g04-vllm-profile-260911"))
    parser.add_argument("--template", type=Path)
    parser.add_argument("--http-runner", type=Path)
    parser.add_argument("--engine-runner", type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(
        args.root, args.base, args.template or args.root / "batch2-engine-plan.json",
        (args.http_runner or args.root / "run_serving_optimization.py").absolute(),
        (args.engine_runner or args.root / "run_engine_optimization.py").absolute(),
    ), indent=2))


if __name__ == "__main__":
    main()
