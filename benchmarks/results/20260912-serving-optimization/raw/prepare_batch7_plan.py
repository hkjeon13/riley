#!/usr/bin/env python3
"""Qualify actual batch7 GPU/HTTP checks and prepare exclusive paired plans.

The g04 numerical contract stays unchanged. Full-model GPU tests and the
separate 8,640-case synthetic attention probe must have executed on the new
clean batch7 snapshot; historical projection proof is not new attention proof.
This script executes neither GPU work nor measurements.
"""

import argparse
import copy
import hashlib
import json
import subprocess
from pathlib import Path

import prepare_batch2_plan as prior
import run_batch7_tests as checks


PROFILE = "vllm-smol-p128-v1"
GATE = "g04-vllm-smol-p128-v1"
IMPLEMENTATION = "g11-two-warp-attention-v1"
PARENT_COMMIT = "67a4197b007d4ec6c9f7ad473bdf1c4f2a636111"
CHANGED_FILES = {
    "kernels/src/graph_numerics.cu",
    "kernels/src/ffi_internal.hpp",
    "kernels/src/graph_resources.cu",
    "crates/riley-runtime/src/llama/graph_decode_full.rs",
}
GRAPH_IMPLEMENTATION_SIGNATURE = "0xF107"
ATTENTION_ENTRY = "enqueue_compiled_packed_decode_attention_two_warp"

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
    source, parent_source = root / "batch7-source", root / "batch6-source"
    require(parent["source_commit"] == PARENT_COMMIT
            and git(parent_source, "rev-parse", "HEAD") == PARENT_COMMIT,
            "batch6 parent identity differs")
    require(not git(parent_source, "status", "--porcelain", "--untracked-files=all"), "batch6 parent is dirty")
    source_files_match(parent_source, parent["source_files"])
    parent_binaries = {str(root / "batch6-target/release" / name): sha(root / "batch6-target/release" / name)
                       for name in ("riley", "riley-profile")}
    require(parent["binaries"] == parent_binaries, "batch6 parent binaries changed")
    require(build["parent_source_commit"] == PARENT_COMMIT
            and build["parent_build_sha256"] == sha(root / "batch6-build.json"), "parent build receipt differs")
    require(build["source_root"] == str(source), "batch7 source root differs")
    commit = git(source, "rev-parse", "HEAD")
    require(commit == build["source_commit"] and git(source, "rev-parse", "HEAD^") == PARENT_COMMIT,
            "batch7 is not the built direct child of batch6")
    require(not git(source, "status", "--porcelain", "--untracked-files=all"), "batch7 source is dirty")
    # A full tracked-tree diff proves every file outside this exact set stayed
    # unchanged, not merely the smaller list present in the build receipt.
    changed = git(source, "diff", "--name-status", "--no-renames", PARENT_COMMIT, "HEAD")
    actual = {}
    for line in changed.splitlines():
        status, name = line.split("\t", 1)
        require(name not in actual, "duplicate changed path")
        actual[name] = status
    expected = {name: "M" for name in CHANGED_FILES}
    require(actual == expected, "batch7 must change exactly the four declared modified files; no additions or removals")
    require(build["changed_files"] == sorted(CHANGED_FILES) and set(overlay) == CHANGED_FILES,
            "build or overlay changed-file manifest differs")
    require(build["source_files"] == {**parent["source_files"], **overlay},
            "batch7 source pins must contain all parent pins plus the exact overlay")
    source_files_match(source, build["source_files"])
    signature = (source / "crates/riley-runtime/src/llama/graph_decode_full.rs").read_text()
    signature_anchor = "let implementation = if packed.is_some() {\n            " + GRAPH_IMPLEMENTATION_SIGNATURE
    require(signature.count(signature_anchor) == 1, "packed decode graph implementation must be F107")
    expected_commands = [
        ["cargo", "test", "--release", "-p", "riley-runtime", "--features", "cuda", "--lib", "--no-run"],
        ["cargo", "build", "--release", "-p", "riley-server", "--features", "server,bench,cuda",
         "--bin", "riley", "--bin", "riley-profile"],
    ]
    require(build["build_argv"] == expected_commands, "batch7 build command differs")
    require(build["build_environment"] == {
        "CUDA_HOME": "/data/riley-g04-cuda13", "CUDAToolkit_ROOT": "/data/riley-g04-cuda13",
        "CMAKE": "/data/cmake-3.31.12/bin/cmake", "CMAKE_BUILD_PARALLEL_LEVEL": "4",
        "CARGO_BUILD_JOBS": "4", "CARGO_TARGET_DIR": str(root / "batch7-target"),
        "LD_LIBRARY_PATH": "/data/riley-g04-cuda13/lib",
    }, "batch7 build environment differs")
    require(build["build_log_sha256"] == sha(root / "batch7-build.log"), "batch7 build log changed")
    require(build["gpu_tests_executed"] is False and build["performance_measured"] is False,
            "compile-only build receipt contains execution claims")
    binaries = {str(root / "batch7-target/release" / name): sha(root / "batch7-target/release" / name)
                for name in ("riley", "riley-profile")}
    require(build["binaries"] == binaries, "batch7 release binaries changed")
    return source, commit, binaries


def validate_gpu(root, receipt, commit, binaries, reference_binding):
    require(receipt["schema_version"] == "riley.batch7-gpu-correctness.v1", "GPU receipt schema differs")
    prior.validate_gpu(receipt, commit, binaries, reference_binding)
    require(receipt["source_clean"] is True and receipt["gpu_tests_executed"] is True,
            "GPU tests did not execute on the clean current snapshot")
    require(receipt["build_sha256"] == sha(root / "batch7-build.json"), "GPU build identity differs")
    require(receipt["performance_measured"] is False, "GPU correctness contains performance trials")
    require(receipt["runner_sha256"] == sha(Path(checks.__file__)), "GPU test runner changed")
    require(receipt["http_helper_sha256"] == sha(root / "batch7_http_check.py"), "HTTP helper changed")
    require({item["name"] for item in receipt["checks"]} == set(checks.GPU_TESTS),
            "all four current-source GPU checks are required")
    for item in receipt["checks"]:
        name, path = item["name"], root / ("batch7-" + item["name"] + ".log")
        require(item["path"] == str(path), "GPU log is not the batch7 check log")
        require(item["argv"] == ["cargo", "test", "--release", "-p", "riley-runtime", "--features", "cuda",
                                  "--lib", name, "--", "--ignored", "--nocapture", "--test-threads=1"],
                "GPU test argv differs: " + name)
        counts = checks.validate_test_log(path, name)
        require(all(item[key] == value for key, value in counts.items()), "GPU test counts differ")
    profile, path = receipt["profile_unit_tests"], root / "batch7-profile-unit-tests.log"
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
    require(receipt["build_sha256"] == sha(root / "batch7-build.json"), "HTTP build receipt differs")
    require(receipt["reference_binding_sha256"] == sha(base / "native-binding.json"), "HTTP reference differs")
    require(receipt["references"] == {str(base / name): sha(base / name)
                                      for name in ("native-binding.json", "request.json")}, "HTTP reference pins differ")
    expected_model = {str(Path(model) / name): binding["workload"][field]
                      for name, field in (("model.safetensors", "weights_sha256"),
                                          ("tokenizer.json", "tokenizer_sha256"))}
    require(receipt["model_files"] == expected_model, "HTTP model pins differ")
    require(receipt["runner_sha256"] == sha(root / "batch7_http_check.py")
            and receipt["shared_runner_sha256"] == sha(Path(checks.__file__)), "HTTP checker identity changed")
    for result in receipt["results"]:
        path = root / "batch7-http-correctness" / ("http-" + result["sampling"] + ".log")
        require(result["graceful_exit_code"] == 0 and result["log"] == evidence(path),
                "HTTP log or graceful shutdown evidence differs")
        require(result["disconnect_points"] == ["after_headers_before_reading_text", "after_first_nonempty_text"],
                "HTTP disconnect coverage differs")


def validate_attention(root, build, commit, binaries):
    # The probe owns its detailed raw-record, fixture and compilation validation.
    # Import only here so test/HTTP helpers do not require the separate probe.
    import batch7_attention_probe as probe

    path = root / "batch7-attention-probe/receipt.json"
    receipt = probe.validate_receipt(path, root / "batch7-build.json")
    require(receipt["schema_version"] == "riley.batch7-attention-correctness.v1"
            and receipt["passed"] is True and receipt["gpu_tests_executed"] is True,
            "current attention probe did not execute and pass")
    require(receipt["source_commit"] == commit and receipt["source_root"] == build["source_root"]
            and receipt["source_files"] == build["source_files"] and receipt["binaries"] == binaries,
            "attention probe source or binary identity differs")
    require(receipt["source_build"] == evidence(root / "batch7-build.json"),
            "attention probe build identity differs")
    require(receipt["runner"] == evidence(root / "batch7_attention_probe.py")
            and sha(Path(probe.__file__)) == receipt["runner"]["sha256"],
            "attention probe validator or runner differs")
    require(receipt["candidate_entry"] == ATTENTION_ENTRY
            and receipt["candidate_source_sha256"] == build["source_files"]["kernels/src/graph_numerics.cu"],
            "attention candidate entry or source binding differs")
    require(receipt["oracle_source"] == evidence(root / "batch6-source/kernels/src/graph_numerics.cu"),
            "attention oracle must be the preserved accepted batch6 source")
    for field, value in {"cases": 8640, "layers": 30, "positions": 32, "patterns": 3, "mappings": 3}.items():
        require(receipt[field] == value, "attention probe coverage differs: " + field)
    for field in ("exact_outputs", "finite_outputs", "guards_intact", "inputs_unchanged",
                  "oracle_outputs_unchanged", "mapping_invariant", "all_allocations_freed"):
        require(receipt[field] is True, "attention probe check failed: " + field)
    require(receipt["performance_measured"] is False and receipt["performance_claim_eligible"] is False,
            "attention correctness receipt contains performance claims")
    return path, receipt


def prepare(root, base, template_path, http_runner, engine_runner):
    root, base, template_path = root.resolve(), base.resolve(), template_path.resolve()
    outputs = {name: root / ("batch7-" + name + ".json")
               for name in ("qualification", "binding", "http-plan", "engine-plan")}
    for path in outputs.values():
        require(not path.exists(), "refusing to replace existing output: " + str(path))
    template, binding = read(template_path), read(base / "native-binding.json")
    request = read(base / "request.json")
    require(template["source_commit"] == PARENT_COMMIT
            and template["source_root"] == str(root / "batch6-source")
            and template["implementation_id"] == "g10-prefill-m16-v1", "template must identify frozen batch6")
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
                                           ("batch7-build.json", "batch6-build.json", "batch7-source-overlay.json"))
    gpu_path, http_path = root / "batch7-gpu-tests.json", root / "batch7-http-correctness/http-validation.json"
    parent_qualification_path = root / "batch6-qualification.json"
    build, parent, overlay, gpu, http = map(read, (build_path, parent_path, overlay_path, gpu_path, http_path))
    source, commit, binaries = validate_build(root, build, parent, overlay)
    validate_gpu(root, gpu, commit, binaries, base / "native-binding.json")
    validate_http(root, http, commit, binaries, binding, base, model)
    attention_path, attention = validate_attention(root, build, commit, binaries)
    parent_qualification = read(parent_qualification_path)
    require(parent_qualification["passed"] is True and parent_qualification["source_commit"] == PARENT_COMMIT
            and parent_qualification["binaries"] == parent["binaries"]
            and parent_qualification["build"] == evidence(parent_path), "batch6 parent qualification differs")
    require(template["immutable_files"].get(str(parent_qualification_path)) == sha(parent_qualification_path),
            "template does not pin the batch6 parent qualification")

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

    profile_binary, http_binary = (str(root / "batch7-target/release" / name) for name in ("riley-profile", "riley"))
    qualification = {
        "schema_version": "riley.two-warp-attention-qualification.v1", "passed": True,
        "source_root": str(source), "source_commit": commit, "source_clean": True,
        "implementation_id": IMPLEMENTATION, "numerical_profile": PROFILE, "correctness_gate_id": GATE,
        "scope": "SmolLM2-135M BF16 c1 P128 O32; packed decode uses independent QK, two-warp PV and exact softmax publication by warp 0. "
                 "Actual current-source 8640-case synthetic attention parity and mapping invariance, GPU full logits/status/KV parity, retained reuse, "
                 "cancellation, rejection, scheduler block mapping tests, and release HTTP reference text/SSE/disconnect checks.",
        "build": evidence(build_path), "gpu_tests": evidence(gpu_path), "gpu_test_logs": gpu["checks"],
        "profile_unit_tests": gpu["profile_unit_tests"], "release_http": evidence(http_path), "binaries": binaries,
        "gpu_tests_executed_on_current_snapshot": True, "vllm_reference_tokens_exact": True,
        "full_logits_and_kv_exact": True,
        "attention_probe": evidence(attention_path), "attention_cases_exact": 8640,
        "attention_probe_scope": {
            "inputs": "synthetic BF16; 30 layer indices are independent seeds, not checkpoint activations",
            "positions_inclusive": [128, 159], "patterns": 3, "physical_block_mappings": 3,
            "implementations_compared": ["original rows1", "accepted packed decode", "candidate two-warp"],
            "output_bf16_words_per_implementation": 576, "comparison_pairs": 2,
            "mapping_invariance_exact": True,
        },
        "source_lineage": {"parent_commit": PARENT_COMMIT, "parent_build": evidence(parent_path),
                           "parent_qualification": evidence(parent_qualification_path),
                           "changed_files": sorted(CHANGED_FILES), "other_parent_tracked_files_unchanged": True,
                           "overlay": evidence(overlay_path)},
        "graph_implementation_signature": GRAPH_IMPLEMENTATION_SIGNATURE,
        "candidate_scope": "Packed decode attention dispatch only; original rows1 and accepted packed attention remain unchanged oracles; P128 prefill remains unchanged",
        "performance_metrics": {"throughput": None, "ttft": None, "tpot": None,
                                "p95_latency": None, "p99_latency": None,
                                "gpu_span": None, "memory_efficiency": None},
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
                measurement_started=False, comparison_status="Two-warp attention correctness qualified on current snapshot; performance pending",
                qualification_blockers=[], measurement_environment_blockers=[], prepare_only_argv=[],
                qualification_path=str(outputs["qualification"]), binding_path=str(outputs["binding"]),
                request_path=str(base / "request.json"), canonical_e0_claim=False,
                binding_note="The g04 numerical contract is unchanged. The g11 implementation, clean source and binaries "
                             "have new actual attention, full-model GPU and HTTP correctness receipts. No performance result is inferred.")
    http_argv, engine_argv = plan["http_lanes"]["riley"]["argv"], plan["engine_lanes"]["riley"]["argv"]
    http_argv[0], engine_argv[0] = http_binary, profile_binary
    for flag, value in {"--git-commit": commit, "--git-dirty": "false", "--implementation-id": IMPLEMENTATION,
                        "--correctness-gate-id": GATE, "--correctness-report-sha256": qualification_sha,
                        "--executable-sha256": binaries[profile_binary], "--run-id": "g11-riley-{index}"}.items():
        prior.replace(engine_argv, flag, value)

    new_pins = [build_path, parent_path, parent_qualification_path, overlay_path, gpu_path, http_path, template_path,
                Path(__file__).resolve(), Path(prior.__file__).resolve(), Path(checks.__file__).resolve(),
                root / "build_batch7.py", root / "batch7_http_check.py", root / "batch7-build.log",
                http_runner, engine_runner, source / "benchmarks/scripts/check_vllm_profile_run.py",
                source / "benchmarks/scripts/preflight.sh", Path(gpu["profile_unit_tests"]["path"])]
    new_pins.extend(Path(item["path"]) for item in gpu["checks"])
    new_pins.extend(Path(item["log"]["path"]) for item in http["results"])
    new_pins.append(attention_path)
    new_pins.extend(Path(attention[field]["path"]) for field in
                    ("probe_binary", "runner", "probe_source", "fixture_manifest", "oracle_source",
                     "compile_receipt", "raw_results", "stderr"))
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
        args.root, args.base, args.template or args.root / "batch6-engine-plan.json",
        (args.http_runner or args.root / "run_serving_optimization.py").absolute(),
        (args.engine_runner or args.root / "run_engine_optimization.py").absolute(),
    ), indent=2))


if __name__ == "__main__":
    main()
