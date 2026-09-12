#!/usr/bin/env python3
"""Read-only grouped-token campaign analyzer for the frozen V4 controller.

--campaign DIRECTORY [--path-map ORIGINAL_ROOT=LOCAL_ROOT ...] [--output NEW_JSON]
The frozen V1 analyzer supplies accounting, lifecycle and pair verification in a
private module instance. This adapter binds the old API proofs to their original
V1 client, checks the separate V2 CPU/replay qualification, and reparses new raw
rows with V2. Grouped IDs keep their shared observed arrival time. No old stream
is relabelled, no finish is invented, and no process, network or GPU work runs.
"""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import re
import sys
import uuid

import run_serving_token_optimization_v4 as controller
import serving_token_client as qualified_tokens
import serving_token_client_v2 as tokens
import validate_grouped_token_client as client_validator

SCHEMA = "riley.serving-token-analysis.v2"
BASE_SHA = "36567e3bbeadcb636002587c598648d85fe70c6c149a7573259359da7578d56d"
VALIDATOR_SHA = "314bfa0e9e5f8c76c5c0c0d2ed870a06b148b9debd7ffaf042b62ff4b72bc983"
require = controller.require
PROOFS = {"published_ids_match_reference", "default_and_raw_text_usage_match", "input_ids_exact",
          "blank_generated_token_events_counted", "disconnect_reuse_exact", "source_commit_path_reviewed_and_unit_tested",
          "existing_full_gpu_logits_kv_exact", "owned_process_cleanup_verified"}


def pinned(ref, plan, preparation):
    require(isinstance(ref, dict) and set(ref) == {"path", "sha256"}
            and Path(ref["path"]).is_absolute() and re.fullmatch(r"[a-f0-9]{64}", ref["sha256"]),
            "invalid grouped-client evidence reference")
    require(plan["immutable_files"].get(ref["path"]) == ref["sha256"]
            and preparation["transitive_evidence"].get(ref["path"]) == ref["sha256"],
            "grouped-client artifact is not pinned in both plan and preparation")


def measurement_client_proof(plan, preparation, reference, evidence):
    """Verify copied proof files without requiring the remote Python executable.

    The saved Python binary hash is bound to the immutable preparation; analysis
    does not rerun that interpreter or tests. Every other explicit input, all raw
    rows and the actual CPU log are read and checked. Historical metadata JSON is
    parsed only for its original reference/failure identity, never recursively as
    a description of current remote source, model or runtime dependencies.
    """
    for key in ("token_client_validation", "token_client_validator"):
        require(preparation[key] == plan[key], "prepared grouped-client proof differs")
        pinned(plan[key], plan, preparation)
    proof = evidence.json_ref(plan["token_client_validation"])
    require(plan["token_client_validator"]["sha256"] == VALIDATOR_SHA
            == controller.shared.digest(client_validator.__file__), "grouped-client validator source differs")
    evidence.ref(plan["token_client_validator"])
    require(proof["schema_version"] == client_validator.SCHEMA and proof["completed"] is True
            and proof["cpu_only"] is True and proof["gpu_executed"] is False
            and proof["performance_claim"] is False and proof["measurement_client_qualified"] is True
            and proof["original_campaign_complete"] is False
            and proof["production_or_server_configuration_changed"] is False,
            "grouped-client qualification scope is invalid")
    refs = proof["inputs"]
    require(set(refs) == {"runner", "python", "tests", "client_v1", "client_v2", "preparation", "plan",
                          "finalization", "binding", "reference_parent"}, "grouped-client input inventory differs")
    for ref in (*refs.values(), *proof["raw_files"].values(), proof["tests"]["log"]):
        pinned(ref, plan, preparation)
    # All explicit provenance except the execution-host Python is available in a
    # copied campaign. Its immutable binary pin is recorded, not silently mapped
    # to the analyzer host's Python.
    for key, ref in refs.items():
        if key != "python":
            evidence.ref(ref)
    require(refs["runner"] == plan["token_client_validator"]
            and refs["binding"] == plan["reference"]["binding"]
            and refs["reference_parent"] == plan["reference"]["parent_c1_plan"],
            "grouped-client qualification uses another validator or native reference")
    require(proof["client_v1"] == refs["client_v1"] == preparation["qualified_api_client"]
            and proof["client_v2"] == refs["client_v2"] == preparation["measurement_client"]
            and refs["client_v1"]["sha256"] == client_validator.V1_SHA == controller.shared.digest(qualified_tokens.__file__)
            and refs["client_v2"]["sha256"] == client_validator.V2_SHA == controller.shared.digest(tokens.__file__)
            and refs["tests"]["sha256"] == client_validator.TESTS_SHA, "qualified API/client/test identities differ")
    tests = Path(refs["tests"]["path"])
    require(all(tests.parent.parent / filename == Path(refs[key]["path"]) for key, filename in
                (("client_v1", "serving_token_client.py"), ("client_v2", "serving_token_client_v2.py"))),
            "recorded CPU suite imported a different client location")
    test = proof["tests"]
    log = evidence.ref(test["log"]).read_text()
    matches = re.findall(r"^Ran ([1-9][0-9]*) tests? in [0-9.]+s$", log, re.M)
    require(test["argv"] == [refs["python"]["path"], refs["tests"]["path"], "-v"]
            and test["returncode"] == 0 and test["passed_tests"] == 26 and matches == ["26"]
            and "\nOK\n" in log and not re.search(r"^(FAILED|ERROR:|FAIL:)", log, re.M),
            "grouped-client CPU test log is not the qualified 26-test run")
    prior_preparation = evidence.json_ref(refs["preparation"])
    prior_plan = evidence.json_ref(refs["plan"])
    prior_final = evidence.json_ref(refs["finalization"])
    require(prior_preparation["plan"] == refs["plan"]
            and prior_plan["reference"]["binding"] == refs["binding"]
            and prior_plan["reference"]["parent_c1_plan"] == refs["reference_parent"]
            and prior_preparation["reference_sha256"] == reference.sha256
            and prior_final["failure"] is not None and prior_final["completed_pairs"] == [],
            "original incomplete campaign identity or outcome was relabelled")
    require(not (evidence.locate(refs["preparation"]["path"]).parent / "completion.json").exists(),
            "original failed campaign acquired a completion claim")
    expected_reference = {"model": reference.model, "prompt_token_ids": list(reference.prompt_token_ids),
                          "output_token_ids": list(reference.output_token_ids), "text": reference.text,
                          "finish_reason": reference.finish_reason}
    require(proof["reference"] == expected_reference, "grouped-client replay reference differs")
    mapped_raw = {phase: {"path": str(evidence.ref(ref)), "sha256": ref["sha256"]}
                  for phase, ref in proof["raw_files"].items()}
    replayed = client_validator.replay_files(qualified_tokens, tokens, expected_reference, mapped_raw)
    require(replayed == proof["replay"], "grouped-client offline replay differs from saved qualification")
    return {"validated": True, "receipt": plan["token_client_validation"],
            "successful_requests_replayed": replayed["successful_requests_replayed"],
            "incomplete_prefix_tokens": replayed["partial"][0]["replayed_prefix_tokens"],
            "original_campaign_still_incomplete": True, "qualification_python": refs["python"],
            "qualification_python_scope": "saved plan/preparation binary pin; not executed or reread on analysis host",
            "historical_metadata_scope": "explicit file bytes and original identity only; nested manifests remain historical",
            "cpu_tests_rerun": False, "gpu_executed": False}


def grouped_observations(row):
    if not row["streaming"]:
        return {}
    groups = row["token_delivery_groups"]
    # The V2 parser replay already checks every field against raw frames. Count
    # frame sizes separately so aggregate maxima are not accidentally summed.
    result = {"generated_frames": groups["generated_frame_count"],
              "multi_token_frames": groups["multi_token_frame_count"],
              "tokens_in_multi_token_frames": groups["tokens_in_multi_token_frames"],
              "within_frame_zero_itls": groups["within_frame_zero_itl_count"],
              "requests_with_multi_token_frames": int(groups["multi_token_frame_count"] > 0)}
    result.update({"frame_size_" + str(size): count for size, count in Counter(groups["frame_token_counts"]).items()})
    result["between_frame_zero_itls"] = sum(value == 0 for value in row["metrics"]["token_itl_ns"]) - groups["within_frame_zero_itl_count"]
    require(result["between_frame_zero_itls"] >= 0, "within-frame zeros exceed observed zero delivery ITLs")
    result["tokens_in_blank_generated_frames"] = sum(len(choice.get("token_ids", []))
        for frame in row["frames"] if frame["data"] != "[DONE]"
        for choice in tokens.parse_json(frame["data"])["choices"] if choice.get("text") == "")
    return result


def format_observations(value):
    histogram = {key.removeprefix("frame_size_"): value.pop(key) for key in list(value) if key.startswith("frame_size_")}
    value["frame_token_count_histogram"] = dict(sorted(histogram.items(), key=lambda pair: int(pair[0])))
    value["max_frame_token_count"] = max(map(int, histogram), default=0)
    return value


def private_core():
    """No global mutation of the frozen analyzer or existing imported modules."""
    source = Path(__file__).with_name("analyze_serving_token_optimization.py")
    require(controller.shared.digest(source) == BASE_SHA, "frozen V1 analyzer changed")
    name = "_token_analysis_v2_core_" + uuid.uuid4().hex
    spec = importlib.util.spec_from_file_location(name, source)
    core = importlib.util.module_from_spec(spec)
    # This module defines no dataclasses and need not be globally registered.
    spec.loader.exec_module(core)
    core.controller, core.tokens, core.SCHEMA = controller, tokens, SCHEMA
    original_reference = core.reference_and_proofs
    original_row = core.reparse_row
    original_phase = core.verify_phase
    proof_summary = {}

    def reference_and_proofs(plan, preparation, evidence):
        for module in (controller.legacy, controller.shared):
            require(controller.shared.digest(module.__file__) in plan["immutable_files"].values(),
                    "shared measurement validation source is not pinned")
        core.tokens = qualified_tokens
        try:
            reference, request, binding = original_reference(plan, preparation, evidence)
        finally:
            core.tokens = tokens
        reference = tokens.TokenReference(reference.model, reference.prompt_token_ids, reference.output_token_ids,
                                          reference.text, reference.finish_reason)
        proof_summary.update(measurement_client_proof(plan, preparation, reference, evidence))
        for module in (sys.modules[__name__], core, client_validator, tokens, qualified_tokens, controller,
                       controller.legacy, controller.shared):
            evidence.capture(module.__file__)
        return reference, request, binding

    def reparse_row(*args, **kwargs):
        result = original_row(*args, **kwargs)
        result.update(grouped_observations(args[0] if args else kwargs["row"]))
        return result

    def verify_phase(*args, **kwargs):
        result, rows = original_phase(*args, **kwargs)
        if "token_delivery_observations" in result:
            format_observations(result["token_delivery_observations"])
        return result, rows

    core.reference_and_proofs = reference_and_proofs
    core.reparse_row = reparse_row
    core.verify_phase = verify_phase
    return core, proof_summary


def analyze(campaign, mappings=()):
    core, proof_summary = private_core()
    result = core.analyze(campaign, mappings)
    result.update(measurement_client_qualification=proof_summary or None,
                  metrics_scope="client token delivery: every ID in a frame shares its observed arrival timestamp; first visible text and engine times remain distinct",
                  zero_itl_scope="within-frame zeros and same-time arrivals across separate frames are reported separately; neither proves simultaneous engine generation",
                  ratio_scope="right / left; a zero left latency statistic has no defined ratio and remains null",
                  blank_text_scope="blank generated frames and IDs carried by blank frames; text inside a nonempty grouped frame is not attributed to individual IDs",
                  historical_v1_measurements_reinterpreted=False)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--path-map", action="append", default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    mappings = [item.split("=", 1) for item in args.path_map]
    require(all(len(item) == 2 for item in mappings), "path maps use ORIGINAL=LOCAL")
    result = analyze(args.campaign, mappings)
    if args.output:
        controller.write(args.output, result)
    else:
        print(json.dumps(result, indent=2, allow_nan=False))
    return 2 if result["status"] == "invalid" else 0


if __name__ == "__main__":
    sys.exit(main())
