#!/usr/bin/env python3
"""Qualify the immutable P128 batch and prepare, never execute, paired plans.

Run after run_batch2_tests.py and batch2_http_check.py have completed. Every
output is created exclusively; existing qualifications and plans are preserved.
The g04 gate names the unchanged numerical contract, while g06 names this
implementation. This is a GUI-retained c1 qualification, not a performance or
canonical E0 claim.
"""

import argparse
import copy
import hashlib
import json
import re
import subprocess
from pathlib import Path


PROFILE = "vllm-smol-p128-v1"
GATE = "g04-vllm-smol-p128-v1"
IMPLEMENTATION = "g06-p128-prefill-v1"
REQUIRED_TESTS = {
    "batched_prefill_exact_logits_status_and_every_decode_kv": (
        "P128_PREFILL_PARITY prompts=3 full_logits_exact=true "
        "argmax_status_exact=true full_initialized_kv_snapshots=96 "
        "every_decode_kv_exact=true snapshot_method=close_and_reopen "
        "zero_allocations=true performance_claim=false"
    ),
    "batched_prefill_retained_reuse_cancel_and_rejected_output_invalidation": (
        "P128_PREFILL_REUSE requests=6 retained_slow_replays=899 "
        "retained_fast_replays=137 cancelled_after_prefill=1 "
        "cancelled_after_decode=1 invalid_shape_stage_cases=42 "
        "raw_outputs_exact=true full_initialized_final_kv_exact=true "
        "zero_allocations=true performance_claim=false"
    ),
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


def encoded(value):
    return (json.dumps(value, indent=2) + "\n").encode("utf-8")


def evidence(path):
    return {"path": str(path), "sha256": sha(path)}


def option(argv, flag):
    require(argv.count(flag) == 1, "expected one argv option: " + flag)
    index = argv.index(flag)
    require(index + 1 < len(argv), "missing argv value: " + flag)
    return argv[index + 1]


def replace(argv, flag, value):
    option(argv, flag)
    argv[argv.index(flag) + 1] = value


def below(path, directory):
    return Path(path).resolve().is_relative_to(Path(directory).resolve())


def validate_build(root, build):
    source = root / "batch2-source"
    commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain"], text=True
    )
    require(commit == build["source_commit"] and not dirty,
            "batch2 source differs from the clean build snapshot")
    require(build.get("source_clean", True) is True,
            "build explicitly reports an unclean source")
    require(bool(build["source_files"]), "build has no source file receipts")
    for relative, expected in build["source_files"].items():
        path = source / relative
        require(not Path(relative).is_absolute() and below(path, source),
                "build source path escapes snapshot: " + relative)
        require(sha(path) == expected, "built source changed: " + str(path))
    binaries = {
        str(root / "batch2-target/release" / name):
            sha(root / "batch2-target/release" / name)
        for name in ("riley", "riley-profile")
    }
    require(build["binaries"] == binaries, "batch2 binaries differ from build receipt")
    return source, commit, binaries


def validate_gpu(receipt, commit, binaries, reference_binding):
    require(receipt["passed"] is True, "GPU qualification did not pass")
    require(receipt["source_commit"] == commit, "GPU source identity differs")
    require(receipt["binaries"] == binaries, "GPU binary identity differs")
    require(receipt["reference_binding_sha256"] == sha(reference_binding),
            "GPU reference binding identity differs")
    for flag in ("vllm_reference_tokens_exact", "full_logits_and_kv_exact"):
        require(receipt[flag] is True, "missing GPU correctness evidence: " + flag)
    require(receipt["performance_claim_eligible"] is False,
            "correctness receipt unexpectedly claims performance eligibility")
    seen = set()
    for check in receipt["checks"]:
        name, path = check["name"], Path(check["path"])
        require(name not in seen, "duplicate GPU check: " + name)
        seen.add(name)
        require(check["passed"] is True, "GPU check failed: " + name)
        require(sha(path) == check["sha256"], "GPU log changed: " + str(path))
        text = path.read_text(encoding="utf-8")
        require("test result: ok. 1 passed; 0 failed" in text,
                "GPU log lacks a single passing test: " + name)
        require(re.search(r"(?:^|::)" + re.escape(name) + r"(?:\s|\.)", text,
                          flags=re.MULTILINE), "GPU log lacks named test: " + name)
        if name in REQUIRED_TESTS:
            require(REQUIRED_TESTS[name] in text, "GPU log lacks exact parity marker: " + name)
    require(set(REQUIRED_TESTS).issubset(seen), "both P128 GPU tests are required")


def validate_http(receipt, binaries, binding, model):
    binary = next(path for path in binaries if Path(path).name == "riley")
    require(receipt["binary_sha256"] == binaries[binary], "HTTP binary identity differs")
    require(receipt["profile"] == PROFILE, "HTTP numerical profile differs")
    require(receipt["reference_tokens"] == binding["generated_token_ids"],
            "HTTP reference token IDs differ from the fixed binding")
    require(receipt["performance_trials"] == 0, "HTTP qualification contains timing trials")
    seen = set()
    for result in receipt["results"]:
        sampling, argv = result["sampling"], result["argv"]
        require(sampling not in seen, "duplicate HTTP sampling receipt")
        seen.add(sampling)
        for flag in ("full_text_exact", "streaming_exact", "post_disconnect_reuse_exact"):
            require(result[flag] is True, "HTTP qualification failed: " + flag)
        require(result["invalid_request_statuses"] == [400, 400, 400],
                "HTTP unsupported shape rejection differs")
        require(argv[:2] == [binary, "serve"], "HTTP command uses another binary")
        expected = {
            "--model": model, "--model-id": "g04-smol", "--max-active-sequences": "1",
            "--batch-token-budget": "128", "--prefill-chunk-tokens": "128",
            "--max-sequence-tokens": "160", "--max-output-tokens": "32",
            "--kv-blocks": "10", "--residual-rmsnorm": "separate",
            "--execution-completion": "iteration-batch", "--metadata-transport": "packed-async",
            "--execution-graph-policy": "require", "--graph-numerics": PROFILE,
            "--sampling-backend": sampling,
        }
        for flag, value in expected.items():
            require(option(argv, flag) == value, "HTTP qualification argv differs: " + flag)
    require(seen == {"cpu", "gpu-greedy"}, "both HTTP sampling paths are required")


def prepare(root, base, template_path, http_runner, engine_runner):
    root, base = root.resolve(), base.resolve()
    outputs = {name: root / ("batch2-" + name + ".json")
               for name in ("qualification", "binding", "http-plan", "engine-plan")}
    for path in outputs.values():
        require(not path.exists(), "refusing to replace existing output: " + str(path))
    template = read(template_path)
    reference_binding_path = base / "native-binding.json"
    binding = read(reference_binding_path)
    request_path = base / "request.json"
    request = read(request_path)
    require(template["numerical_profile"] == PROFILE, "template numerical profile differs")
    require(binding["source"]["correctness_gate_id"] == GATE, "reference numerical gate differs")
    require(option(template["engine_lanes"]["riley"]["argv"], "--correctness-gate-id") == GATE,
            "use the corrected candidate-engine-v2-plan template")
    workload = binding["workload"]
    for key, value in {"concurrency": 1, "prompt_tokens": 128, "output_tokens": 32,
                       "warmups": 5, "measured_iterations": 30, "sampling_id": "greedy"}.items():
        require(workload[key] == value, "fixed qualification workload differs: " + key)
    require(len(binding["input_token_ids"]) == 128 and len(binding["generated_token_ids"]) == 32,
            "fixed token reference lengths differ")
    require(request["prompt_token_ids"] == binding["input_token_ids"], "request reference differs")
    model = option(template["http_lanes"]["riley"]["argv"], "--model")
    require(option(template["engine_lanes"]["riley"]["argv"], "--model") == model,
            "engine and HTTP model paths differ")

    build_path, gpu_path = root / "batch2-build.json", root / "batch2-gpu-tests.json"
    http_path = root / "batch2-http-correctness/http-validation.json"
    build, gpu, http = read(build_path), read(gpu_path), read(http_path)
    source, commit, binaries = validate_build(root, build)
    validate_gpu(gpu, commit, binaries, reference_binding_path)
    validate_http(http, binaries, binding, model)

    # Preserve verified data/reference pins, dropping every former candidate's
    # source, binary, binding and qualification from the new candidate identity.
    immutable = {path: digest for path, digest in template["immutable_files"].items()
                 if not below(path, root) and not below(path, template["source_root"])}
    for path, expected in immutable.items():
        require(sha(path) == expected, "retained template artifact changed: " + path)
    for name in ("request.json", "native-binding.json", "verification.json", "prompts.jsonl",
                 "vllm_lane.py", "environment.json", "candidate.json"):
        path = str(base / name)
        require(path in immutable, "template lacks reference artifact pin: " + path)
    for name, expected in (("model.safetensors", workload["weights_sha256"]),
                           ("tokenizer.json", workload["tokenizer_sha256"])):
        require(immutable.get(str(Path(model) / name)) == expected,
                "model data pin differs from workload binding: " + name)

    profile_binary = str(root / "batch2-target/release/riley-profile")
    http_binary = str(root / "batch2-target/release/riley")
    qualification = {
        "schema_version": "riley.p128-prefill-qualification.v1", "passed": True,
        "source_root": str(source), "source_commit": commit, "source_clean": True,
        "implementation_id": IMPLEMENTATION, "numerical_profile": PROFILE,
        "correctness_gate_id": GATE,
        "scope": "SmolLM2-135M BF16 c1 P128 O32; fixed prompt exact vLLM reference; "
                 "three-prompt full-logit/KV parity against the prior arithmetic; "
                 "retained reuse, cancellation and unsupported shape rejection",
        "build": evidence(build_path), "gpu_tests": evidence(gpu_path),
        "gpu_test_logs": gpu["checks"], "release_http": evidence(http_path),
        "binaries": binaries,
        "reference_only": {"binding": evidence(reference_binding_path),
                           "verification": evidence(base / "verification.json"),
                           "note": "Historical g04 source/binary identity is reference evidence only."},
        "vllm_reference_tokens_exact": True, "full_logits_and_kv_exact": True,
        "canonical_e0_claim": False, "performance_measured": False,
    }
    qualification_bytes = encoded(qualification)
    qualification_sha = hashlib.sha256(qualification_bytes).hexdigest()
    binding["source"].update(git_commit=commit, git_dirty=False,
                             implementation_id=IMPLEMENTATION, correctness_gate_id=GATE,
                             correctness_report_sha256=qualification_sha,
                             executable_sha256=binaries[profile_binary])
    binding_bytes = encoded(binding)

    plan = copy.deepcopy(template)
    for old_key in ("live_preflight_last_failure", "binding_note"):
        plan.pop(old_key, None)
    plan.update(source_root=str(source), source_commit=commit, measurement_started=False,
                comparison_status="P128 prefill correctness qualified; performance pending",
                qualification_blockers=[], measurement_environment_blockers=[],
                numerical_profile=PROFILE, implementation_id=IMPLEMENTATION,
                qualification_path=str(outputs["qualification"]),
                binding_path=str(outputs["binding"]), request_path=str(request_path),
                prepare_only_argv=[], canonical_e0_claim=False,
                pairs=[{"index": i, "order": ["riley", "vllm"] if i % 2 else ["vllm", "riley"]}
                       for i in range(1, 6)],
                warmups_per_process=5, measured_requests_per_process=30,
                fresh_process_per_lane_per_pair=True,
                binding_note="The unchanged g04 gate names the numerical contract. "
                "The g06 implementation, clean source, new binaries and qualification are separately pinned.",
                actual_condition={"name": "gui-retained-cool48-512mib-v1",
                                  "max_start_temperature_c": 48, "max_idle_gpu_memory_mib": 512,
                                  "foreign_cuda_compute_allowed": False, "canonical_e0": False,
                                  "legacy_vllm_raw_environment": "Keep immutable raw identity; "
                                  "the runner envelope records current preflight conditions."})
    http_argv = plan["http_lanes"]["riley"]["argv"]
    http_argv[0] = http_binary
    for flag in ("--batch-token-budget", "--prefill-chunk-tokens"):
        replace(http_argv, flag, "128")
    argv = plan["engine_lanes"]["riley"]["argv"]
    argv[0] = profile_binary
    for flag, value in {"--git-commit": commit, "--git-dirty": "false",
                        "--implementation-id": IMPLEMENTATION, "--correctness-gate-id": GATE,
                        "--correctness-report-sha256": qualification_sha,
                        "--executable-sha256": binaries[profile_binary],
                        "--run-id": "g06-riley-{index}"}.items():
        replace(argv, flag, value)

    new_pins = [build_path, gpu_path, http_path, template_path, Path(__file__).resolve(),
                http_runner, engine_runner, source / "benchmarks/scripts/check_vllm_profile_run.py",
                source / "benchmarks/scripts/preflight.sh"]
    new_pins.extend(Path(check["path"]) for check in gpu["checks"])
    reference_argv = plan["engine_lanes"]["vllm"]["argv"]
    new_pins.extend([Path(reference_argv[0]), Path(reference_argv[1]),
                     Path(option(reference_argv, "--matrix")), Path(plan["reference_checker_python"])])
    # Pin the untouched legacy adapter source used by vllm_lane.py separately;
    # it must never be reported as this batch's new Riley source snapshot.
    reference_source = Path(read(base / "candidate.json")["source_root"])
    new_pins.extend(sorted((reference_source / "benchmarks/lanes/vllm").rglob("*.py")))
    for path in new_pins:
        path = path.absolute()
        current = sha(path)
        require(str(path) not in immutable or immutable[str(path)] == current,
                "reference executable or checker changed: " + str(path))
        immutable[str(path)] = current
    immutable.update(binaries)
    immutable[str(outputs["qualification"])] = qualification_sha
    immutable[str(outputs["binding"])] = hashlib.sha256(binding_bytes).hexdigest()
    plan["immutable_files"] = immutable
    http_plan, engine_plan = copy.deepcopy(plan), copy.deepcopy(plan)
    http_plan.update(measurement_mode="http", runner=str(http_runner))
    engine_plan.update(measurement_mode="engine", runner=str(engine_runner))
    payloads = {"qualification": qualification_bytes, "binding": binding_bytes,
                "http-plan": encoded(http_plan), "engine-plan": encoded(engine_plan)}
    # All qualification checks precede the first write. Exclusive creation also
    # rejects a competing preparer; it never truncates an earlier campaign.
    for name, payload in payloads.items():
        with outputs[name].open("xb") as stream:
            stream.write(payload)
    return {"source_commit": commit, "implementation_id": IMPLEMENTATION,
            "correctness_report_sha256": qualification_sha,
            "outputs": {name: str(path) for name, path in outputs.items()},
            "measurement_started": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/tmp/riley-opt-260912"))
    parser.add_argument("--base", type=Path, default=Path("/tmp/riley-g04-vllm-profile-260911"))
    parser.add_argument("--template", type=Path)
    parser.add_argument("--http-runner", type=Path)
    parser.add_argument("--engine-runner", type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(
        args.root, args.base, args.template or args.root / "candidate-engine-v2-plan.json",
        (args.http_runner or args.root / "run_serving_optimization.py").absolute(),
        (args.engine_runner or args.root / "run_engine_optimization.py").absolute(),
    ), indent=2))


if __name__ == "__main__":
    main()
