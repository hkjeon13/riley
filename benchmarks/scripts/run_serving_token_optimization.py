#!/usr/bin/env python3
"""Strict paired token-serving measurements. No remote execution is performed.

Usage: --plan PLAN --output NEW_DIRECTORY [--measure]. Preparation is default
and reads all proofs/runtime/session identities before creating preparation.json;
it does not query the GPU, start a server, or stop Blender.

Manifest schema riley.serving-token-plan.v1 (all paths absolute):
  immutable_files: {path: sha256}, including this/client/legacy/shared module,
    session helper + dependencies, contract, Python, source/build/proof/reference
    files, model files, vLLM executable/package files and all validator imports.
  contract: {path,sha256}; reference: {parent_c1_plan,request,binding,verification}
    (each a {path,sha256} reference). The parent supplies expected_output_text;
    its c1 workload is reference-only and is never reinterpreted or executed.
  workload: {id,purpose: initial-measurement|screening,pairs:1..5,
    retained_requests_per_process:2..100000,warmups_per_worker_per_transport:5,
    arrival_policy:closed-loop-refill}. Initial measurement requires 5x>=1000.
  settings: [{id,offered_concurrency:1|2,vllm_token_budget:128|256}]
  comparisons: [{id,left:LANE_ID,right:LANE_ID}]
  base_environment: explicit HOME/PATH/locale etc; no ambient server inheritance.
  session: {helper:{path,sha256},python:{path,sha256},root:ROUND13_ROOT}
  nvidia_smi: {path,sha256}; startup_timeout_seconds<=1800,
    request_timeout_seconds<=120, cooldown_timeout_seconds<=1800.
  lanes: {LANE_ID:{kind:riley|vllm,argv,env,cwd,port,model_path,
    model_files:{absolute_path:sha256},model_id:g04-smol,
    [Riley] build:{path,sha256},model_proof:{path,sha256},
      raw_token_proof:{path,sha256}, [optional] default_http_proof:{path,sha256},
      raw_validator:{path,sha256},validator_dependencies:[{path,sha256}],
      [fusion candidates] model_qualification:{path,sha256} whose model_tests
        equals model_proof, and model_validator:{path,sha256} exporting the pinned
        qualify_batch8 read-only validate_build/validate_model/validate_fusion,
    [vLLM] output_diagnostics:{SETTING_ID:{path,sha256}}, package_root:PATH,
      runtime_files:{path:sha256} for all package files except __pycache__/.pyc
      plus venv/bin/python. All model-directory JSON files must also be pinned. }}

argv supports {port}, {concurrency}, {vllm_budget}, {lane_output}. Riley's
qualified gpu-greedy HTTP argv may change only bind/wait64; C02 is forbidden
by the qualified graph profile and is never enabled here. vLLM argv may change only port/capacity/budget
from the pinned reference parent. Its actual startup configuration is checked.
Riley raw_validator exports read-only validate_completion(path); a candidate
requires its OWN model/default/raw proof, all bound to its exact build/binary.

All warmups and retained requests use strict one-ID SSE frames/usage validation.
No partial campaign is promoted: completion.json exists only after every pair,
post-run identities, owned-process cleanup AND verified Blender restoration.
Metrics are client delivery observations, not GPU commit times or a winner claim.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time

import run_serving_concurrency as legacy
import serving_token_client as tokens

shared = legacy.shared
read, evidence, require = legacy.read, legacy.evidence, legacy.require
SCHEMA = "riley.serving-token-plan.v1"
RESULT = "riley.serving-token-measurement.v1"
VLLM_RUNTIME = {"engine_version": "0.27.1", "enforce_eager": False,
                "enable_chunked_prefill": True, "compilation_mode": "VLLM_COMPILE",
                "cudagraph_mode": "FULL_AND_PIECEWISE"}
CONDITION = {"scope": "private driver runtime; GUI retained but authorized Blender sessions paused",
             "canonical_headless": False, "idle_memory_limit_mib": 512,
             "start_temperature_limit_c": 48, "foreign_cuda_compute_allowed": False,
             "riley_active_capacity": 1, "riley_waiting_capacity": 64, "riley_http_workers": 8}


def write(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def check_ref(ref, pins=None):
    require(isinstance(ref, dict) and set(ref) == {"path", "sha256"}, "invalid artifact reference")
    path = Path(ref["path"])
    require(path.is_absolute() and evidence(path) == ref, "artifact changed or path is not canonical: " + str(path))
    if pins is not None:
        require(pins.get(str(path)) == ref["sha256"], "artifact absent from immutable_files: " + str(path))
    return path


def check_refs(value):
    """Check nested evidence and absolute path->hash mappings, without executing artifacts."""
    if isinstance(value, dict):
        if {"path", "sha256"} <= set(value):
            check_ref({key: value[key] for key in ("path", "sha256")})
        if set(value) != {"path", "sha256"}:
            for key, child in value.items():
                if isinstance(key, str) and key.startswith("/") and isinstance(child, str) and re.fullmatch(r"[a-f0-9]{64}", child):
                    require(shared.digest(key) == child, "nested artifact changed: " + key)
                else:
                    check_refs(child)
    elif isinstance(value, list):
        for child in value:
            check_refs(child)


def freeze_receipt(ref, inventory, seen=None):
    """Freeze transitive JSON evidence before pause, without trusting flags alone."""
    seen = set() if seen is None else seen
    path = Path(ref["path"])
    require(path.is_absolute() and shared.digest(path) == ref["sha256"], "transitive artifact changed: " + str(path))
    name = str(path)
    require(name not in inventory or inventory[name] == ref["sha256"], "conflicting transitive pin")
    inventory[name] = ref["sha256"]
    if name in seen or path.suffix != ".json":
        return
    seen.add(name)
    def walk(value):
        if isinstance(value, dict):
            if {"path", "sha256"} <= set(value):
                freeze_receipt({key: value[key] for key in ("path", "sha256")}, inventory, seen)
            for key, child in value.items():
                if isinstance(key, str) and key.startswith("/") and isinstance(child, str) and re.fullmatch(r"[a-f0-9]{64}", child):
                    freeze_receipt({"path": key, "sha256": child}, inventory, seen)
                elif key not in ("path", "sha256"):
                    walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)
    walk(read(path))


def load_module(path, name):
    # Some pinned validators inspect __import__(__name__); register before exec.
    path = Path(path)
    sys.path.insert(0, str(path.parent))
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def validate_pins(plan):
    pins = plan["immutable_files"]
    require(isinstance(pins, dict) and pins, "immutable inventory is required")
    for path, sha in pins.items():
        require(Path(path).is_absolute() and shared.digest(path) == sha, "immutable artifact changed: " + path)
    for module in (Path(__file__), Path(tokens.__file__), Path(legacy.__file__), Path(shared.__file__)):
        require(pins.get(str(module.resolve())) == shared.digest(module), "controller/client/helper is not pinned: " + str(module))
    check_ref(plan["contract"], pins)
    return pins


def source_snapshot(build):
    source = Path(build["source_root"])
    require(source.is_absolute() and re.fullmatch(r"[a-f0-9]{40}", build["source_commit"]), "invalid source identity")
    def git(*args):
        return subprocess.check_output(["git", "-C", str(source), *args], text=True).strip()
    require(git("rev-parse", "HEAD") == build["source_commit"] and not git("status", "--porcelain", "--untracked-files=all"),
            "qualified source is dirty or commit changed")
    for relative, sha in build["source_files"].items():
        path = source / relative
        require(not Path(relative).is_absolute() and ".." not in Path(relative).parts
                and path.resolve().is_relative_to(source.resolve()) and shared.digest(path) == sha,
                "qualified source content changed: " + relative)
    require(build["source_files"] and build["binaries"], "empty source/binary inventory")
    for path, sha in build["binaries"].items():
        require(Path(path).is_absolute() and shared.digest(path) == sha, "qualified binary changed")
    return {"source_commit": build["source_commit"], "source_clean": True, "binaries": build["binaries"]}


def reference_data(plan, pins):
    refs = plan["reference"]
    require(set(refs) == {"parent_c1_plan", "request", "binding", "verification"}, "reference contract fields differ")
    values = {key: read(check_ref(ref, pins)) for key, ref in refs.items()}
    parent, request, binding, verification = (values[key] for key in ("parent_c1_plan", "request", "binding", "verification"))
    require(parent["measurement_mode"] == "http" and binding["workload"]["concurrency"] == 1
            and binding["workload"]["prompt_tokens"] == 128 and binding["workload"]["output_tokens"] == 32
            and binding["workload"]["sampling_id"] == "greedy"
            and binding["source"]["correctness_gate_id"] == "g04-vllm-smol-p128-v1", "not the unchanged native c1 reference")
    require(binding["source"]["correctness_report_sha256"] == refs["verification"]["sha256"]
            and verification["status"] == "correctness-qualified"
            and verification["unequal_tensor_elements"] == 0 and verification["generated_tokens_exact"] == 32
            and verification["native_prefill_and_decode"] is True, "native reference qualification differs")
    for key in ("request", "binding", "verification"):
        require(parent["immutable_files"].get(refs[key]["path"]) == refs[key]["sha256"], "native input differs from original parent")
    require(request["prompt_token_ids"] == binding["input_token_ids"] and len(binding["input_token_ids"]) == 128
            and len(binding["generated_token_ids"]) == 32 and request["requested_output_tokens"] == 32
            and request["temperature"] == 0 and request["top_p"] == 1, "native request differs")
    text = parent["http_lanes"]["riley"]["expected_output_text"]
    require(text and text == parent["http_lanes"]["vllm"]["expected_output_text"], "reference text differs")
    reference = tokens.TokenReference("g04-smol", tuple(binding["input_token_ids"]), tuple(binding["generated_token_ids"]), text)
    return values, reference


def validate_workload(plan):
    require(plan["schema_version"] == SCHEMA, "old c1/concurrency plans cannot be relabelled")
    workload = plan["workload"]
    require(set(workload) == {"id", "purpose", "pairs", "retained_requests_per_process",
                             "warmups_per_worker_per_transport", "arrival_policy"}, "workload fields differ")
    require(re.fullmatch(r"[A-Za-z0-9_.-]+", workload["id"] or "") and workload["id"] != "g04-c1-p128-o32",
            "new workload identity required")
    legacy.positive(workload["pairs"], "pairs", 5)
    legacy.positive(workload["retained_requests_per_process"], "retained requests", 100000)
    require(workload["warmups_per_worker_per_transport"] == 5 and workload["arrival_policy"] == "closed-loop-refill",
            "five concurrent warmups per transport/worker and closed-loop refill required")
    require(workload["purpose"] in ("screening", "initial-measurement"), "measurement purpose differs")
    if workload["purpose"] == "initial-measurement":
        require(workload["pairs"] == 5 and workload["retained_requests_per_process"] >= 1000,
                "initial measurement needs five pairs and >=1000 retained requests/process")
    seen, capacities = set(), set()
    require(plan["settings"], "settings are required")
    for setting in plan["settings"]:
        require(set(setting) == {"id", "offered_concurrency", "vllm_token_budget"}, "setting fields differ")
        c = setting["offered_concurrency"]
        require(type(c) is int and c in (1, 2) and setting["vllm_token_budget"] == 128*c,
                "initial strict reference supports only C1/b128 and C2/b256")
        require(re.fullmatch(r"[A-Za-z0-9_.-]+", setting["id"] or "") and setting["id"] not in seen
                and c not in capacities and workload["retained_requests_per_process"] >= c, "duplicate/invalid setting")
        seen.add(setting["id"]); capacities.add(c)
    for key, upper in (("startup_timeout_seconds", 1800), ("request_timeout_seconds", 120), ("cooldown_timeout_seconds", 1800)):
        value = plan[key]
        require(type(value) in (int, float) and math.isfinite(value) and 0 < value <= upper, "invalid timeout: " + key)
    ids = set()
    require(plan["comparisons"], "comparisons are required")
    for comparison in plan["comparisons"]:
        require(set(comparison) == {"id", "left", "right"} and comparison["left"] != comparison["right"]
                and comparison["left"] in plan["lanes"] and comparison["right"] in plan["lanes"]
                and re.fullmatch(r"[A-Za-z0-9_.-]+", comparison["id"] or "")
                and comparison["id"] not in ids, "invalid comparison")
        ids.add(comparison["id"])


def lane_argv(lane, setting, directory):
    return [item.format(port=lane["port"], concurrency=setting["offered_concurrency"],
                        vllm_budget=setting["vllm_token_budget"], lane_output=str(directory)) for item in lane["argv"]]


def lane_environment(plan, lane, runtime):
    base_keys = {"HOME", "PATH", "LANG", "LC_ALL", "LC_CTYPE", "USER", "LOGNAME", "TMPDIR"}
    require(set(plan["base_environment"]) <= base_keys, "base environment is not the declared minimal whitelist")
    env = {**plan["base_environment"], **runtime["child_only_overrides"], **lane["env"]}
    require(env.get("HOME") and env.get("PATH"), "explicit HOME/PATH required")
    require(all(isinstance(k, str) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", k)
                and isinstance(v, str) and "\0" not in v for k, v in env.items()), "invalid explicit environment")
    require(not any(k in ("LD_PRELOAD", "LD_AUDIT", "PYTHONPATH", "VLLM_BATCH_INVARIANT") or k.startswith("RILEY_")
                    for k in env), "instrumentation or unqualified numerical environment")
    overrides = runtime["child_only_overrides"]
    for key, value in overrides.items():
        if key == "LD_LIBRARY_PATH":
            require(env[key] == value or env[key].startswith(value + ":"), "private driver library paths must precede CUDA libraries")
        else:
            require(env[key] == value, "private runtime override changed: " + key)
    require(env.get("CUDA_VISIBLE_DEVICES", "0") == "0", "only qualified CUDA device 0 is supported")
    return env


def validate_riley(lane, plan, values, pins):
    build_ref = lane["build"]
    build = read(check_ref(build_ref, pins))
    identity = source_snapshot(build)
    binary = lane["argv"][0]
    require(binary in build["binaries"] and Path(lane["cwd"]).resolve() == Path(build["source_root"]).resolve(),
            "Riley launch does not use qualified source/binary")
    proof = read(check_ref(lane["model_proof"], pins))
    check_refs(proof)
    require(proof["passed"] is True and proof["source_clean"] is True and proof["source_commit"] == build["source_commit"]
            and proof["binaries"] == build["binaries"] and proof["full_logits_and_kv_exact"] is True,
            "current source model/logit/KV proof required")
    require(proof.get("build_sha256") == build_ref["sha256"] or proof.get("build") == build_ref, "model proof build differs")
    require(proof.get("gpu_tests_executed", proof.get("gpu_tests_executed_on_current_snapshot")) is True,
            "current source GPU tests were not executed")
    if "model_qualification" in lane:
        qualification = read(check_ref(lane["model_qualification"], pins))
        require(qualification["model_tests"] == lane["model_proof"] and qualification["build"] == build_ref
                and qualification["passed"] is True and qualification["source_clean"] is True
                and qualification["source_commit"] == build["source_commit"] and qualification["binaries"] == build["binaries"]
                and qualification["source_files"] == build["source_files"]
                and qualification["full_logits_and_kv_exact"] is True
                and qualification["gpu_tests_executed_on_current_snapshot"] is True
                and qualification["correctness_gate_id"] == "g04-vllm-smol-p128-v1", "model qualification differs from current model proof/build")
        proof = qualification
    elif proof.get("schema_version") == "riley.batch8-model-correctness.v1":
        raise ValueError("batch8 requires separate full model/fusion qualification")
    if "model_tests" in proof:
        model = read(proof["model_tests"]["path"])
        require(model["source_commit"] == build["source_commit"] and model["binaries"] == build["binaries"]
                and model["build"] == build_ref, "nested model tests are for another build")
        validator_ref = lane["model_validator"]
        validator = load_module(check_ref(validator_ref, pins), "token_model_validator_" + validator_ref["sha256"][:16])
        root, base = Path(build_ref["path"]).parent, Path(plan["reference"]["binding"]["path"]).parent
        require(validator.validate_build(root) == build and validator.validate_model(root, base, build) == model,
                "model qualification detailed validation differs")
        fusion = validator.validate_fusion(root, build, values["binding"])
        require(fusion["cases"] == proof["fusion_cases_exact"], "fusion qualification count differs")
    else:
        model = proof
    names = ("batched_prefill_retained_reuse_cancel_and_rejected_output_invalidation",
             "batched_prefill_exact_logits_status_and_every_decode_kv",
             "owned_graph_smol_p128_o32_reuses_scheduler_block_mappings",
             "owned_graph_vllm_smol_p128_o32_reuses_scheduler_block_mappings")
    require([row["name"] for row in model["checks"]] == list(names), "four current GPU checks required")
    for row in model["checks"]:
        text = check_ref({key: row[key] for key in ("path", "sha256")}).read_text()
        require(row["passed"] is True and row["passed_tests"] == 1 and row["failed_tests"] == row["ignored_tests"] == 0
                and re.search(r"test result: ok\. 1 passed; 0 failed; 0 ignored;", text), "GPU test receipt/log does not prove one pass")
    first, second = (Path(row["path"]).read_text() for row in model["checks"][:2])
    require("P128_PREFILL_REUSE requests=6" in first and "invalid_shape_stage_cases=42" in first
            and "raw_outputs_exact=true full_initialized_final_kv_exact=true zero_allocations=true" in first
            and "P128_PREFILL_PARITY prompts=3 full_logits_exact=true" in second
            and "full_initialized_kv_snapshots=96 every_decode_kv_exact=true" in second, "GPU parity/reuse marker missing")
    for key in ("profile_unit_tests", "server_lib_tests" if "server_lib_tests" in model else "server_unit_tests"):
        row = model[key]
        text = check_ref({key: row[key] for key in ("path", "sha256")}).read_text()
        require(row["passed"] is True and row["passed_tests"] > 0 and row["failed_tests"] == 0
                and f"test result: ok. {row['passed_tests']} passed; 0 failed;" in text, "server/profile CPU check missing")

    validator_path = check_ref(lane["raw_validator"], pins)
    for dep in lane["validator_dependencies"]:
        check_ref(dep, pins)
    raw_path = check_ref(lane["raw_token_proof"], pins)
    validator = load_module(validator_path, "token_raw_validator_" + lane["raw_validator"]["sha256"][:16])
    raw = validator.validate_completion(raw_path)
    require(raw["source_build"] == build_ref and raw["source_commit"] == build["source_commit"]
            and raw["binary"] == {"path": binary, "sha256": build["binaries"][binary]}
            and raw["reference_binding"] == plan["reference"]["binding"] and raw["request"] == plan["reference"]["request"]
            and raw["client_module"] == evidence(tokens.__file__), "raw API proof is for a different build/reference/client")
    for ref in [lane["model_proof"]] + ([lane["default_http_proof"]] if "default_http_proof" in lane else []):
        require(ref in raw["prerequisites"], "raw API proof lacks current prerequisite")
    require(raw["model_files"] == lane["model_files"], "raw API/model pin differs")
    if "default_http_proof" in lane:
        default = read(check_ref(lane["default_http_proof"], pins))
        check_refs(default)
        require(default["source_commit"] == build["source_commit"] and default["source_clean"] is True
                and default["build_sha256"] == build_ref["sha256"] and default["binary_sha256"] == build["binaries"][binary]
                and default["reference_binding_sha256"] == plan["reference"]["binding"]["sha256"], "default HTTP proof differs")
    # The raw validator qualifies BOTH default and opt-in wire paths. Its actual
    # gpu-greedy launch is the source for the new serving configuration.
    require([item["sampling_backend"] for item in raw["per_sampler"]] == ["cpu", "gpu-greedy"], "both wire samplers required")
    original = read(raw["per_sampler"][1]["preparation"]["path"])["argv"]
    allowed = {"--bind", "--max-waiting-requests"}
    for setting in plan["settings"]:
        argv = lane_argv(lane, setting, Path("/tmp/token-plan-validation"))
        require(legacy.without_options(argv, allowed) == legacy.without_options(original, allowed), "Riley qualified argv changed")
        require(legacy.option(argv, "--bind") == f"127.0.0.1:{lane['port']}"
                and legacy.option(argv, "--max-waiting-requests") == "64"
                and legacy.option(argv, "--sampling-backend") == "gpu-greedy"
                and legacy.option(argv, "--max-active-sequences") == "1"
                and not any(item.startswith("--c02-") for item in argv), "Riley queue/sampler/profile differs")
    return identity


def validate_vllm(lane, plan, values, pins):
    original = values["parent_c1_plan"]["http_lanes"]["vllm"]
    require(lane["env"].items() >= original["env"].items(), "vLLM reference environment changed")
    require(set(lane["env"]) - set(original["env"]) <= {"LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES", "CUDA_HOME"},
            "additional unqualified vLLM environment settings")
    require(pins.get(lane["argv"][0]) == values["parent_c1_plan"]["immutable_files"].get(original["argv"][0]),
            "vLLM executable differs from reference")
    package = Path(lane["package_root"])
    require(package.is_absolute() and package.name == "vllm" and (package / "__init__.py").is_file(), "installed vLLM package root required")
    expected = {str(path) for path in package.rglob("*") if path.is_file()
                and "__pycache__" not in path.parts and path.suffix != ".pyc"}
    interpreter = str(Path(original["argv"][0]).with_name("python"))
    expected.add(interpreter)
    require(set(lane["runtime_files"]) == expected and expected, "vLLM installed runtime inventory differs")
    require(lane["runtime_files"][interpreter] == values["parent_c1_plan"]["immutable_files"].get(interpreter),
            "vLLM interpreter differs from reference")
    for path, sha in lane["runtime_files"].items():
        require(pins.get(path) == sha and shared.digest(path) == sha, "vLLM runtime file is not pinned: " + path)
    allowed = {"--port", "--max-num-seqs", "--max-num-batched-tokens"}
    for setting in plan["settings"]:
        argv = lane_argv(lane, setting, Path("/tmp/token-plan-validation"))
        require(legacy.without_options(argv, allowed) == legacy.without_options(original["argv"], allowed)
                and legacy.option(argv, "--port") == str(lane["port"])
                and legacy.option(argv, "--max-num-seqs") == str(setting["offered_concurrency"])
                and legacy.option(argv, "--max-num-batched-tokens") == str(setting["vllm_token_budget"]),
                "vLLM qualified argv/capacity/budget differs")
        diagnostic = read(check_ref(lane["output_diagnostics"][setting["id"]], pins))
        check_refs(diagnostic)
        require(diagnostic["completed"] is True and diagnostic["diagnostic_only"] is True
                and diagnostic["performance_claim"] is False
                and diagnostic["label"] == f"c{setting['offered_concurrency']}-vllm-budget{setting['vllm_token_budget']}",
                "vLLM setting provenance differs")
        require(diagnostic["phases"] and all(phase["responses_complete"] == phase["requested"]
                and phase["reference_token_ids_equal"] == phase["requested"]
                and phase["reference_text_equal"] == phase["requested"]
                and phase["counts_prompt_usage_and_finish_valid"] == phase["requested"]
                and not phase["parse_errors"] and not phase["request_errors"] and not phase["worker_errors"]
                for phase in diagnostic["phases"]), "selected vLLM output diagnostic was not exact")


def prepare(plan_path):
    plan_path = Path(plan_path).resolve(strict=True)
    plan = read(plan_path)
    validate_workload(plan)
    pins = validate_pins(plan)
    values, reference = reference_data(plan, pins)
    session_spec = plan["session"]
    helper = check_ref(session_spec["helper"], pins)
    require(helper.name == "remote_session_round13_v2.py", "only authorized round13 v2 restoration helper supported")
    check_ref(session_spec["python"], pins)
    session = load_module(helper, "token_round13_session")
    require(str(session.ROOT) == session_spec["root"] and not session.ROOT.exists(), "restoration round is wrong or already used")
    runtime = session.validate_runtime()
    sessions = session.check(runtime)
    require(len(sessions) == 3, "three authorized Blender sessions required")
    smi = check_ref(plan["nvidia_smi"], pins)
    require(runtime["files"].get(str(smi)) == plan["nvidia_smi"]["sha256"], "GPU query executable not in private runtime")
    check_refs(runtime)
    identities, transitive = {}, {}
    require(plan["lanes"], "no lanes")
    for name, lane in plan["lanes"].items():
        require(re.fullmatch(r"[A-Za-z0-9_.-]+", name) and lane["kind"] in ("riley", "vllm"), "invalid lane identity")
        legacy.positive(lane["port"], "port", 65535)
        shared.check_port(lane["port"])
        require(lane["model_id"] == "g04-smol" and Path(lane["cwd"]).is_absolute(), "lane model/cwd differs")
        lane_environment(plan, lane, runtime)
        for suffix, key in (("model.safetensors", "weights_sha256"), ("tokenizer.json", "tokenizer_sha256")):
            path = str(Path(lane["model_path"]) / suffix)
            require(lane["model_files"].get(path) == values["binding"]["workload"][key] == pins.get(path), "lane model pin differs")
        check_refs(lane["model_files"])
        for path in Path(lane["model_path"]).rglob("*.json"):
            require(pins.get(str(path)) == shared.digest(path), "model metadata is not pinned: " + str(path))
        if lane["kind"] == "riley":
            identities[name] = validate_riley(lane, plan, values, pins)
            for key in ("build", "model_proof", "model_qualification", "raw_token_proof", "default_http_proof"):
                if key in lane:
                    freeze_receipt(lane[key], transitive)
        else:
            validate_vllm(lane, plan, values, pins)
            for ref in lane["output_diagnostics"].values():
                freeze_receipt(ref, transitive)
    require(any(lane["kind"] == "riley" for lane in plan["lanes"].values()), "Riley lane is required")
    # Validators/imports are finished. Pin every campaign-local imported module,
    # including transitive helpers, before an authorized process can be stopped.
    roots = {helper.parent, *(Path(lane["raw_validator"]["path"]).parent for lane in plan["lanes"].values() if lane["kind"] == "riley")}
    for module in list(sys.modules.values()):
        file = getattr(module, "__file__", None)
        if file and Path(file).suffix == ".py" and Path(file).resolve().parent in roots:
            require(pins.get(str(Path(file).resolve())) == shared.digest(file), "imported campaign helper is not pinned: " + file)
    validate_pins(plan)
    resolved_targets = {path: str(Path(path).resolve(strict=True)) for path in set(pins) | set(transitive)}
    result = {"schema_version": RESULT, "measurement_started": False, "plan": evidence(plan_path),
              "condition": CONDITION, "runtime": runtime, "sessions": sessions, "source_identities": identities,
              "transitive_evidence": transitive, "resolved_targets": resolved_targets, "reference_sha256": reference.sha256, "historical_c1_scope": "reference only; not reinterpreted",
              "workload": plan["workload"], "settings": plan["settings"], "performance_claim": False}
    return plan, result, session, reference, values["request"], values["binding"]


def unchanged(plan_path, plan, prepared):
    require(evidence(plan_path) == prepared["plan"], "exact plan bytes changed")
    validate_pins(plan)
    for path, target in prepared["resolved_targets"].items():
        require(str(Path(path).resolve(strict=True)) == target, "immutable symlink target changed: " + path)
    for path, sha in prepared["transitive_evidence"].items():
        require(shared.digest(path) == sha, "transitive qualification artifact changed: " + path)
    for name, lane in plan["lanes"].items():
        if lane["kind"] == "riley":
            require(source_snapshot(read(lane["build"]["path"])) == prepared["source_identities"][name], "source identity changed")


def session_action(plan, action, directory):
    argv = [plan["session"]["python"]["path"], plan["session"]["helper"]["path"], action]
    result = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=600, text=True)
    path = directory / ("session-" + action + "-" + str(time.time_ns()) + ".log")
    path.write_text(result.stdout)
    require(result.returncode == 0, "session " + action + " failed; see " + str(path))
    return evidence(path)


def gpu_snapshot(plan, env, binding):
    base = [plan["nvidia_smi"]["path"], "-i", str(binding["environment"]["gpu"]["device_index"])]
    def query(flag):
        return subprocess.check_output([*base, flag, "--format=csv,noheader,nounits"], env=env, text=True, timeout=15).strip()
    fields = [part.strip() for part in query("--query-gpu=uuid,driver_version,temperature.gpu,memory.used,utilization.gpu").split(",")]
    require(len(fields) == 5 and fields[0] == binding["environment"]["gpu"]["uuid"]
            and fields[1] == "580.173.02", "GPU UUID/private driver differs")
    raw = query("--query-compute-apps=pid,used_memory")
    processes = [{"pid": int(row.split(",")[0]), "used_memory_mib": row.split(",")[1].strip()}
                 for row in raw.splitlines() if row.strip()]
    return {"timestamp_ns": time.perf_counter_ns(), "gpu_uuid": fields[0], "driver_version": fields[1],
            "temperature_c": int(fields[2]), "memory_used_mib": int(fields[3]), "utilization_percent": int(fields[4]),
            "compute_processes": processes}


def cooldown(plan, env, binding, path):
    samples, deadline = [], time.monotonic() + plan["cooldown_timeout_seconds"]
    try:
        while True:
            row = gpu_snapshot(plan, env, binding); samples.append(row)
            require(not row["compute_processes"], "foreign CUDA compute present; no process will be killed")
            if row["temperature_c"] <= 48 and row["memory_used_mib"] <= 512:
                return row
            require(time.monotonic() < deadline, "GPU did not meet common cooldown/memory condition")
            time.sleep(2)
    finally:
        write(path, samples)


def owned(pid, leader):
    try:
        return os.getsid(pid) == leader
    except ProcessLookupError:
        return False


def compute_maps(plan, env, binding, session, runtime, process):
    sample = gpu_snapshot(plan, env, binding)
    require(sample["compute_processes"], "owned CUDA process is missing")
    result = {}
    for row in sample["compute_processes"]:
        require(owned(row["pid"], process.pid), "foreign CUDA process present")
        maps = session.verify_private_maps(row["pid"], runtime)
        require(maps["compute_loaded"] and maps["all_observed_vendor_mappings_pinned"], "private CUDA mapping missing")
        result[str(row["pid"])] = maps
    return {"gpu": sample, "process_maps": result}


class Watchdog:
    """Campaign-local checks supplement the independent restoration watchdog."""
    def __init__(self, plan, directory):
        self.plan, self.directory = plan, directory
        self.stop_event, self.failed = threading.Event(), threading.Event()
        self.errors, self.actions = [], []
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        while not self.stop_event.wait(300):
            try:
                self.actions.append(session_action(self.plan, "extend", self.directory))
            except Exception as error:
                self.errors.append(str(error)); self.failed.set(); return

    def check(self):
        require(not self.failed.is_set(), "external restoration watchdog extension failed")
        root = Path(self.plan["session"]["root"])
        require(not (root / "verified.json").exists() and time.time() < read(root / "deadline.json"),
                "restoration started/completed or watchdog deadline expired during measurement")

    def __enter__(self):
        self.thread.start(); return self

    def __exit__(self, *_):
        self.stop_event.set(); self.thread.join(timeout=610)
        write(self.directory / "watchdog-extensions.json", {"actions": self.actions, "errors": self.errors})


def phase_check(rows, accounting, count, concurrency):
    require(accounting["completed"] is True and accounting["requested"] == count
            and accounting["succeeded"] == count and accounting["failed"] == 0
            and accounting["observed_max_request_in_flight"] == concurrency and len(rows) == count,
            "token phase incomplete; no retries or partial qualification")
    require(all(row["status"] == "success" and row["mode"] == "strict" and row["protocol_valid"]
                and row["transport_complete"] and row["reference_match"] for row in rows), "token phase correctness failed")


def run_lane(plan, setting, name, directory, session, runtime, reference, request, binding, watchdog):
    directory.mkdir(mode=0o700)
    lane = plan["lanes"][name]
    env = lane_environment(plan, lane, runtime)
    argv = lane_argv(lane, setting, directory)
    shared.check_port(lane["port"])
    start_gpu = cooldown(plan, env, binding, directory / "cooldown.json")
    watchdog.check()
    write(directory / "launch.json", {"argv": argv, "environment": env, "cwd": lane["cwd"], "fresh_process": True,
                                      "lane": name, "setting": setting, "condition": CONDITION, "start_gpu": start_gpu})
    process = None
    summary, error, cleanup = None, None, None
    samples, monitor_errors = [], []
    monitor_stop = threading.Event()
    monitor = None
    try:
        with (directory / "server.log").open("x") as log:
            process = subprocess.Popen(argv, cwd=lane["cwd"], env=env, stdout=log, stderr=log, start_new_session=True)
            write(directory / "process.json", {"pid": process.pid, "session_id": os.getsid(process.pid)})
            shared.wait_ready(process, lane["port"], plan["startup_timeout_seconds"])
            if lane["kind"] == "vllm":
                snapshot = directory / "startup-log-snapshot.log"
                with snapshot.open("xb") as handle:
                    handle.write((directory / "server.log").read_bytes())
                actual = legacy.validate_vllm_startup(snapshot,
                    {"active_capacity": setting["offered_concurrency"], "token_budget": setting["vllm_token_budget"]}, VLLM_RUNTIME)
                write(directory / "vllm-startup.json", actual)
            write(directory / "driver-maps-before.json", compute_maps(plan, env, binding, session, runtime, process))
            with tokens.TokenHttpClient() as client:
                def inspect():
                    while not monitor_stop.wait(5):
                        try:
                            watchdog.check()
                            require(process.poll() is None, "owned server exited during retained process")
                            row = gpu_snapshot(plan, env, binding); samples.append(row)
                            require(all(owned(item["pid"], process.pid) for item in row["compute_processes"]), "foreign CUDA compute contaminated lane")
                        except Exception as failure:
                            monitor_errors.append(str(failure)); client.abort_pending(str(failure)); return
                monitor = threading.Thread(target=inspect, daemon=True); monitor.start()
                payload = {"model": reference.model, "prompt": request["prompt"], "max_tokens": 32, "temperature": 0, "top_p": 1}
                c = setting["offered_concurrency"]
                for phase, streaming, count, per_worker in (
                    ("warmup-nonstream", False, c*5, True), ("warmup-stream", True, c*5, True),
                    ("retained", True, plan["workload"]["retained_requests_per_process"], False)):
                    watchdog.check()
                    def one():
                        require(not monitor_errors, "GPU/session monitor failed")
                        return client.request(lane["port"], payload, reference, streaming=streaming,
                                              mode="strict", timeout_seconds=plan["request_timeout_seconds"])
                    rows, accounting = tokens.run_phase(client, one, concurrency=c, count=count, phase=phase, per_worker=per_worker)
                    with (directory / (phase + ".jsonl")).open("x") as handle:
                        for row in rows:
                            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    write(directory / (phase + "-accounting.json"), accounting)
                    described = tokens.summarize_phase(rows, accounting)
                    write(directory / (phase + "-descriptive.json"), described)
                    phase_check(rows, accounting, count, c)
                    require(not monitor_errors and process.poll() is None, "lane contaminated or server exited")
                    watchdog.check()
                    if phase == "retained":
                        summary = described
                write(directory / "driver-maps-after.json", compute_maps(plan, env, binding, session, runtime, process))
    except BaseException as failure:
        error = {"type": type(failure).__name__, "message": str(failure)}
    finally:
        monitor_stop.set()
        if monitor is not None:
            monitor.join(timeout=35)
            if monitor.is_alive():
                monitor_errors.append("GPU monitor did not stop")
        if process is not None:
            try:
                shared.stop_owned_process(process)
                # Confirm no live member remains in the owned session. Zombies
                # hold no GPU/socket resources and are separately reported.
                members, zombies = [], []
                for entry in Path("/proc").iterdir():
                    if entry.name.isdigit() and owned(int(entry.name), process.pid):
                        try:
                            state = (entry / "stat").read_text().rsplit(") ", 1)[1].split()[0]
                            (zombies if state == "Z" else members).append(int(entry.name))
                        except FileNotFoundError:
                            pass
                cleanup = {"returncode": process.returncode, "remaining_owned_pids": members, "zombie_pids": zombies,
                           "cleanup_verified": not members, "log": evidence(directory / "server.log")}
                require(not members and process.returncode in ((0,) if lane["kind"] == "riley" else (0, -signal.SIGTERM)), "owned server cleanup was not graceful")
            except BaseException as failure:
                error = error or {"type": type(failure).__name__, "message": str(failure)}
        write(directory / "process-exit.json", {"cleanup": cleanup, "failure": error})
        write(directory / "gpu-monitor.json", {"samples": samples, "errors": monitor_errors})
    require(error is None and not monitor_errors and summary is not None, "lane incomplete: " + str(error or monitor_errors))
    result = {"lane": name, "kind": lane["kind"], "completed": True, "setting": setting, "summary": summary,
              "cleanup": cleanup, "performance_claim": False, "raw": evidence(directory / "retained.jsonl")}
    write(directory / "summary.json", result)
    return result


def pair_result(comparison, index, order, rows):
    require(set(rows) == {comparison["left"], comparison["right"]} and all(row["completed"] for row in rows.values()),
            "incomplete pair cannot be summarized")
    left, right = (rows[comparison[key]]["summary"] for key in ("left", "right"))
    metric = "successful_output_tokens_per_wall_second"
    ratios = {"throughput_right_over_left": right[metric] / left[metric]}
    for name in ("token_ttft_ms", "token_tpot_ms", "token_itl_ms", "e2e_ms", "first_text_ms"):
        ratios[name + "_right_over_left"] = {key: right[name][key] / left[name][key] if left[name][key] > 0 else None
                                             for key in ("median", "p95", "p99")}
    return {"comparison": comparison, "index": index, "order": order, "completed": True,
            "processes": rows, "ratios": ratios, "performance_claim": False}


def verify_restoration(session, prepared):
    receipt = read(session.ROOT / "verified.json")
    require(receipt["alive_and_listening"] and receipt["commands_and_gui_environment_match"]
            and receipt["all_relaunched_processes_have_pinned_vendor_maps"] and len(receipt["processes"]) == 3,
            "three-session restoration receipt incomplete")
    runtime = session.bound_runtime()
    require(runtime == prepared["runtime"], "restored runtime identity differs")
    originals = {row["pid"]: row for row in prepared["sessions"]}
    require({row["original_pid"] for row in receipt["processes"]} == set(originals)
            and len({row["new_pid"] for row in receipt["processes"]}) == 3, "restoration scope differs")
    for row in receipt["processes"]:
        original, pid = originals[row["original_pid"]], row["new_pid"]
        require(row["port"] == original["port"] and session.live(pid) and session.listening(row["port"]),
                "restored session is not alive/listening on its original port")
        current = session.identity(pid)
        require(current["start"] == row["start"] and session.same_command(current, original), "restored command/PID birth differs")
        expected_tag = original.get("restore_tag") if pid == original["pid"] else row["tag"]
        require(session.process_tag(pid) == expected_tag, "restored tag differs")
        if pid != original["pid"]:
            session.process_runtime_environment(pid, runtime)
        if pid != original["pid"]:
            require(session.verify_private_maps(pid, runtime)["ready"], "restored GL vendor runtime is not ready")
        require(current == session.identity(pid), "restored session changed during inspection")
    return evidence(session.ROOT / "verified.json")


def measure(plan_path, output, plan, prepared, session, reference, request, binding):
    completed, failure, restored = [], None, None
    stopped = False
    try:
        unchanged(plan_path, plan, prepared)
        require(session.validate_runtime() == prepared["runtime"] and session.check(prepared["runtime"]) == prepared["sessions"],
                "private runtime/session identity changed before stop")
        # Set before invoking stop: its own partial-stop journal must be restored
        # even if the subprocess exits unsuccessfully or the parent is interrupted.
        for lane in plan["lanes"].values():
            shared.check_port(lane["port"])
        stopped = True
        session_action(plan, "stop", output)
        require(session.bound_runtime() == prepared["runtime"], "stopped runtime differs")
        with Watchdog(plan, output) as watchdog:
            for setting in plan["settings"]:
                for comparison in plan["comparisons"]:
                    for index in range(1, plan["workload"]["pairs"] + 1):
                        order = [comparison["left"], comparison["right"]]
                        if index % 2 == 0:
                            order.reverse()
                        pair_dir = output / f"{setting['id']}-{comparison['id']}-pair{index:02d}"
                        pair_dir.mkdir(mode=0o700)
                        rows = {}
                        for name in order:
                            unchanged(plan_path, plan, prepared)
                            watchdog.check()
                            rows[name] = run_lane(plan, setting, name, pair_dir / name, session, prepared["runtime"],
                                                  reference, request, binding, watchdog)
                            unchanged(plan_path, plan, prepared)
                        result = pair_result(comparison, index, order, rows)
                        write(pair_dir / "pair.json", result)
                        completed.append({"setting": setting["id"], "pair": evidence(pair_dir / "pair.json"), "ratios": result["ratios"]})
            watchdog.check()
        unchanged(plan_path, plan, prepared)
    except BaseException as error:
        failure = {"type": type(error).__name__, "message": str(error)}
    finally:
        if stopped and session.SNAPSHOT.exists():
            try:
                session_action(plan, "restore", output)
                restored = verify_restoration(session, prepared)
            except BaseException as error:
                failure = {"primary": failure, "restore": {"type": type(error).__name__, "message": str(error)},
                           "independent_watchdog_remains_responsible": True}
        write(output / "finalization.json", {"completed_pairs": completed, "failure": failure, "restoration": restored})
    require(failure is None and restored is not None, "measurement incomplete; see finalization.json")
    require(len(completed) == len(plan["settings"]) * len(plan["comparisons"]) * plan["workload"]["pairs"], "campaign pair inventory incomplete")
    result = {"schema_version": RESULT, "completed": True, "plan": prepared["plan"], "workload": plan["workload"],
              "condition": CONDITION, "pairs": completed, "restoration": restored,
              "reference_policy": "strict exact prompt/output IDs, text, length, usage for every warmup and retained response",
              "timing_scope": "client SSE delivery; one ID/frame; shared read timestamps may yield zero ITL",
              "performance_claim": False, "high_concurrency_qualified": False, "p99_stability_qualified": False,
              "historical_c1_receipts_reinterpreted": False}
    write(output / "completion.json", result)
    return result



class MeasurementInterrupted(BaseException):
    """Not an OSError: HTTP readiness must not mistake termination for I/O retry."""


@contextmanager
def termination_handlers():
    """Normal termination enters the same owned cleanup/finally path as Ctrl-C."""
    previous = {}
    interrupted_once = False
    def interrupted(signum, _frame):
        nonlocal interrupted_once
        if interrupted_once:
            return  # Let the first interruption finish owned cleanup/restoration.
        interrupted_once = True
        raise MeasurementInterrupted("measurement interrupted by signal " + str(signum))
    try:
        for signum in (signal.SIGTERM, signal.SIGHUP):
            previous[signum] = signal.signal(signum, interrupted)
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--measure", action="store_true")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    require(not output.exists(), "output already exists; refusing to replace evidence")
    plan, prepared, session, reference, request, binding = prepare(args.plan)
    output.mkdir(parents=True, mode=0o700)
    write(output / "preparation.json", prepared)
    if args.measure:
        with termination_handlers():
            measure(args.plan.resolve(), output, plan, prepared, session, reference, request, binding)
    else:
        print(json.dumps({"prepared": True, "measurement_started": False, "preparation": evidence(output / "preparation.json")}))


if __name__ == "__main__":
    main()
