#!/usr/bin/env python3
"""Read-only token campaign verification and paired descriptive statistics.

--campaign DIRECTORY [--path-map ORIGINAL_ROOT=LOCAL_ROOT ...] [--output NEW_JSON]
Path maps locate copied evidence without changing original paths or hashes.
No process/GPU/runtime hook is executed. The analyzer binds saved qualification
receipts; it does not rerun their GPU tests or require live source/model files.
Every successful response is reparsed from saved raw frame payloads/timestamps,
including both warmup transports. Accounting and metrics are independently
recomputed before accepting stored lane/pair summaries. Missing/failed lanes
remain diagnostics; partial campaigns never receive comparable aggregate ratios.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import sys

import run_serving_token_optimization_v3 as controller
import serving_token_client as tokens

SCHEMA = "riley.serving-token-analysis.v1"
PROOFS = {"published_ids_match_reference", "default_and_raw_text_usage_match", "input_ids_exact",
          "blank_generated_token_events_counted", "disconnect_reuse_exact", "source_commit_path_reviewed_and_unit_tested",
          "existing_full_gpu_logits_kv_exact", "owned_process_cleanup_verified"}
METRICS = ("token_ttft_ms", "token_tpot_ms", "token_itl_ms", "e2e_ms", "first_text_ms")
require = controller.require


class Evidence:
    def __init__(self, mappings=()):
        self.mappings = sorted([(Path(a), Path(b).resolve()) for a, b in mappings], key=lambda pair: len(pair[0].parts), reverse=True)
        require(all(a.is_absolute() and b.is_absolute() for a, b in self.mappings), "path maps must be absolute")
        require(len({str(a) for a, _ in self.mappings}) == len(self.mappings), "duplicate path-map source")
        self.inputs = {}

    def locate(self, original):
        path = Path(original)
        for old, new in self.mappings:
            if path.is_relative_to(old):
                return new / path.relative_to(old)
        return path

    def capture(self, path, original=None, expected=None):
        path = Path(path).resolve(strict=True)
        sha = controller.shared.digest(path)
        require(expected is None or expected == sha, "artifact hash changed: " + str(original or path))
        record = {"original_path": str(original or path), "local_path": str(path), "sha256": sha}
        key = str(original or path)
        require(key not in self.inputs or self.inputs[key] == record, "artifact changed during analysis")
        self.inputs[key] = record
        return path

    def ref(self, reference):
        require(isinstance(reference, dict) and set(reference) == {"path", "sha256"}
                and Path(reference["path"]).is_absolute(), "invalid evidence reference")
        return self.capture(self.locate(reference["path"]), reference["path"], reference["sha256"])

    def json(self, path):
        return tokens.parse_json(self.capture(path).read_bytes())

    def json_ref(self, reference):
        return tokens.parse_json(self.ref(reference).read_bytes())

    def rows(self, path):
        return [tokens.parse_json(line) for line in self.capture(path).read_text().splitlines() if line.strip()]

    def finish(self):
        for record in self.inputs.values():
            require(controller.shared.digest(record["local_path"]) == record["sha256"], "evidence changed during analysis")
        return list(self.inputs.values())


def reference_and_proofs(plan, preparation, evidence):
    controller.validate_workload(plan)
    refs = plan["reference"]
    for ref in refs.values():
        require(plan["immutable_files"].get(ref["path"]) == ref["sha256"], "reference is not pinned in manifest")
    parent, request, binding, native = (evidence.json_ref(refs[key]) for key in ("parent_c1_plan", "request", "binding", "verification"))
    require(binding["source"]["correctness_gate_id"] == "g04-vllm-smol-p128-v1"
            and binding["source"]["correctness_report_sha256"] == refs["verification"]["sha256"]
            and binding["workload"]["concurrency"] == 1 and native["status"] == "correctness-qualified"
            and native["unequal_tensor_elements"] == 0 and native["generated_tokens_exact"] == 32,
            "native reference is not qualified")
    require(all(parent["immutable_files"].get(refs[key]["path"]) == refs[key]["sha256"] for key in ("request", "binding", "verification")),
            "native reference differs from historical parent")
    require(request["prompt_token_ids"] == binding["input_token_ids"] and len(binding["input_token_ids"]) == 128
            and len(binding["generated_token_ids"]) == 32, "reference P128/O32 differs")
    text = parent["http_lanes"]["riley"]["expected_output_text"]
    require(text == parent["http_lanes"]["vllm"]["expected_output_text"], "reference text differs")
    ref = tokens.TokenReference("g04-smol", tuple(binding["input_token_ids"]), tuple(binding["generated_token_ids"]), text)
    require(preparation["schema_version"] == controller.RESULT and preparation["measurement_started"] is False
            and preparation["reference_sha256"] == ref.sha256 and preparation["workload"] == plan["workload"]
            and preparation["settings"] == plan["settings"] and preparation["condition"] == controller.CONDITION,
            "preparation identity/workload differs")
    for module in (controller, tokens):
        require(controller.shared.digest(module.__file__) in plan["immutable_files"].values(), "measurement/parser source is not pinned")
    for name, lane in plan["lanes"].items():
        if lane["kind"] != "riley":
            continue
        for key in ("build", "model_proof", "raw_token_proof"):
            require(plan["immutable_files"].get(lane[key]["path"]) == lane[key]["sha256"], "qualification absent from manifest pins")
            require(preparation["transitive_evidence"].get(lane[key]["path"]) == lane[key]["sha256"], "qualification absent from prepared evidence")
        build, model, raw = (evidence.json_ref(lane[key]) for key in ("build", "model_proof", "raw_token_proof"))
        identity = preparation["source_identities"][name]
        require(identity["source_clean"] is True and identity["source_commit"] == build["source_commit"]
                and identity["binaries"] == build["binaries"] and model["source_commit"] == build["source_commit"]
                and model["binaries"] == build["binaries"] and model["passed"] is True and model["source_clean"] is True
                and model["full_logits_and_kv_exact"] is True and model.get("gpu_tests_executed", model.get("gpu_tests_executed_on_current_snapshot")) is True,
                "current source model qualification missing")
        require(model.get("build", lane["build"]) == lane["build"]
                and model.get("build_sha256", lane["build"]["sha256"]) == lane["build"]["sha256"], "model proof build differs")
        require(raw["schema_version"] in ("riley.http-token-observation-correctness.v3", "riley.batch8-http-token-observation-correctness.v1")
                and raw["completed"] is True and raw["gpu_tests_executed"] is True and raw["performance_claim"] is False
                and raw["source_build"] == lane["build"] and raw["source_commit"] == build["source_commit"]
                and raw["binary"] == {"path": lane["argv"][0], "sha256": build["binaries"][lane["argv"][0]]}
                and raw["reference_binding"] == refs["binding"] and raw["request"] == refs["request"]
                and raw["client_module"]["sha256"] == controller.shared.digest(tokens.__file__)
                and lane["model_proof"] in raw["prerequisites"] and raw["model_files"] == lane["model_files"]
                and len(raw["checks"]) == 92 and len(raw["per_sampler"]) == 2
                and set(raw["proofs"]) == PROOFS and all(value is True for value in raw["proofs"].values()), "raw/default wire proof is incomplete or belongs to another build")
        require(len(model["checks"]) == 4 and all(item["passed"] is True and item["passed_tests"] == 1
                and item["failed_tests"] == item["ignored_tests"] == 0 for item in model["checks"]), "current model test counts are invalid")
        require([item["sampling_backend"] for item in raw["per_sampler"]] == ["cpu", "gpu-greedy"]
                and all(item["completed"] is True and item["completion_checks"] == 44 and item["disconnect_checks"] == 2
                        and len(item["checks"]) == 46 and all(check["completed"] is True for check in item["checks"])
                        for item in raw["per_sampler"])
                and raw["checks"] == [item for sampler in raw["per_sampler"] for item in sampler["checks"]], "raw qualification check inventory incomplete")
        if "model_qualification" in lane:
            qualification = evidence.json_ref(lane["model_qualification"])
            require(qualification["passed"] is True and qualification["source_commit"] == build["source_commit"]
                    and qualification["binaries"] == build["binaries"] and qualification["build"] == lane["build"]
                    and qualification["model_tests"] == lane["model_proof"] and qualification["full_logits_and_kv_exact"] is True
                    and lane["model_qualification"] in raw["prerequisites"], "candidate fusion qualification differs")
            evidence.ref(qualification["fusion_probe"])
    return ref, request, binding


def reparse_row(row, reference, request, streaming, phase, deadline):
    require(row["status"] == "success" and row["mode"] == "strict" and row["streaming"] is streaming
            and row["protocol_valid"] is True and row["parser_protocol_valid"] is True
            and row["transport_complete"] is True and row["reference_match"] is True
            and row["observation_only"] is False and row["error"] is None and row["cleanup_errors"] == []
            and row["owned_connection_closed"] is True and row["http_status"] == 200,
            "unsuccessful/non-strict raw response cannot qualify")
    require(row["phase"] == phase and row["warmup"] is (phase != "retained"), "raw phase label differs")
    require(all(type(row[key]) is int for key in ("call_started_ns", "started_ns", "finished_ns", "call_finished_ns", "headers_received_ns"))
            and row["call_started_ns"] <= row["started_ns"] <= row["headers_received_ns"] <= row["finished_ns"] <= row["call_finished_ns"],
            "request timestamps are invalid")
    require(row["total_deadline_seconds"] == deadline, "request deadline differs from plan")
    parser = tokens.TokenResponseParser(reference, streaming=streaming, started_ns=row["started_ns"], mode="strict")
    for frame in row["frames"]:
        require(row["headers_received_ns"] <= frame["arrived_ns"] <= row["finished_ns"], "frame timestamp precedes headers or exceeds completion")
        payload = frame["data"].encode()
        if streaming:
            parser.feed_sse(payload, frame["arrived_ns"])
        else:
            parser.feed_nonstream(payload, frame["arrived_ns"])
    replayed = parser.finish()
    require(all(row.get(key) == value for key, value in replayed.items()), "saved raw payload/timestamps disagree with parsed fields/metrics")
    payload = {"model": reference.model, "prompt": request["prompt"], "max_tokens": 32, "temperature": 0, "top_p": 1,
               "stream": streaming, "return_token_ids": True}
    if streaming:
        payload["stream_options"] = {"include_usage": True}
    require(row["request"] == payload and row["request_body_sha256"] == hashlib.sha256(
        json.dumps(row["request"], ensure_ascii=False, allow_nan=False).encode()).hexdigest(), "saved request body differs")
    blank = 0
    for frame in row["frames"]:
        if frame["data"] != "[DONE]":
            value = tokens.parse_json(frame["data"])
            blank += sum(bool(choice.get("token_ids")) and choice.get("text") == "" for choice in value["choices"])
    itls = row["metrics"]["token_itl_ns"]
    return {"blank_generated_frames": blank if streaming else 0,
            "zero_observed_itls": sum(value == 0 for value in itls), "observed_itls": len(itls),
            "requests_with_coalesced_token_arrivals": int(any(value == 0 for value in itls))}


def verify_phase(directory, phase, count, c, reference, request, deadline, evidence):
    streaming = phase != "warmup-nonstream"
    rows = evidence.rows(directory / (phase + ".jsonl"))
    accounting = evidence.json(directory / (phase + "-accounting.json"))
    result = {"phase": phase, "requested": count, "rows": len(rows), "accounting": accounting, "validated": False}
    if not accounting.get("completed") or len(rows) != count or any(row.get("status") != "success" for row in rows):
        result["failure"] = "phase is incomplete"
        observations, good, failures = Counter(), 0, []
        for row in rows:
            if row.get("status") == "success":
                try:
                    observations.update(reparse_row(row, reference, request, streaming, phase, deadline))
                    good += 1
                except Exception as error:
                    failures.append({"index": row.get("index"), "type": type(error).__name__, "message": str(error)})
        result.update(successful_rows_validated=good, successful_row_errors=failures,
                      token_delivery_observations=dict(observations), comparison_eligible=False)
        description = directory / (phase + "-descriptive.json")
        if description.exists():
            result["saved_partial_description"] = evidence.json(description)
        result["failed_rows"] = [{key: row.get(key) for key in ("index", "worker_id", "status", "error", "token_ids")} for row in rows if row.get("status") != "success"]
        return result, rows
    require([row["index"] for row in rows] == list(range(count)), "request index inventory differs")
    observations = Counter()
    for row in rows:
        require(type(row["worker_id"]) is int and 0 <= row["worker_id"] < c, "invalid worker identity")
        observations.update(reparse_row(row, reference, request, streaming, phase, deadline))
    ids = [row["response_identity"]["id"] for row in rows]
    require(len(set(ids)) == count, "response IDs repeat within a phase")
    for worker in range(c):
        worker_rows = sorted((row for row in rows if row["worker_id"] == worker), key=lambda row: row["call_started_ns"])
        require(worker_rows and all(left["call_finished_ns"] <= right["call_started_ns"] for left, right in zip(worker_rows, worker_rows[1:])),
                "worker requests overlap; not a closed-loop refill")
        if phase != "retained":
            require(len(worker_rows) == 5, "each warmup worker must complete five requests")
    expected = {"schema_version": tokens.PHASE_SCHEMA, "phase": phase, "warmup": phase != "retained",
                "offered_concurrency": c, "requested": count, "attempted": count, "succeeded": count, "failed": 0,
                "unresolved_attempts": 0, "not_started": 0, "reference_matches": count, "protocol_valid_responses": count,
                "observed_max_request_in_flight": tokens.overlap_peak(rows),
                "observed_max_call_in_flight": tokens.overlap_peak(rows, "call_started_ns", "call_finished_ns"),
                "observed_max_successful_request_in_flight": tokens.overlap_peak(rows), "completed": True,
                "duplicate_response_ids": [], "reference_sha256s": [reference.sha256], "modes": ["strict"],
                "streaming_transports": [streaming], "worker_errors": [], "interrupted": None,
                "correctness_qualified": False, "performance_qualified": False}
    require(all(accounting.get(key) == value for key, value in expected.items())
            and expected["observed_max_request_in_flight"] == c, "phase accounting differs from raw request spans")
    require(type(accounting["phase_started_ns"]) is int and type(accounting["phase_finished_ns"]) is int
            and accounting["phase_started_ns"] <= min(row["call_started_ns"] for row in rows)
            and max(row["call_finished_ns"] for row in rows) <= accounting["phase_finished_ns"],
            "phase does not enclose complete client calls and EOF cleanup")
    computed = tokens.summarize_phase(rows, accounting)
    require(evidence.json(directory / (phase + "-descriptive.json")) == computed, "stored descriptive metrics differ from raw rows")
    result.update(validated=True, metrics=computed, token_delivery_observations=dict(observations))
    return result, rows


def verify_lane(directory, name, setting, plan, prepared, reference, request, binding, evidence):
    lane = plan["lanes"][name]
    result = {"lane": name, "kind": lane["kind"], "status": "incomplete", "phases": [], "errors": []}
    if not directory.exists():
        result["errors"].append("lane was not started")
        return result, None
    rows_all = []
    try:
        launch = evidence.json(directory / "launch.json")
        expected_argv = controller.lane_argv(lane, setting, Path(launch["argv"][0]).parent)  # No output placeholder on these qualified profiles.
        require(launch["argv"] == expected_argv and launch["environment"] == controller.lane_environment(plan, lane, prepared["runtime"])
                and launch["cwd"] == lane["cwd"] and launch["lane"] == name and launch["setting"] == setting
                and launch["fresh_process"] is True and launch["condition"] == controller.CONDITION, "lane launch identity differs")
        gpu = launch["start_gpu"]
        require(gpu["gpu_uuid"] == binding["environment"]["gpu"]["uuid"] and gpu["driver_version"] == "580.173.02"
                and gpu["temperature_c"] <= 48 and gpu["memory_used_mib"] <= 512 and not gpu["compute_processes"], "lane start condition failed")
        require(evidence.json(directory / "cooldown.json")[-1] == gpu, "cooldown evidence differs")
        process = evidence.json(directory / "process.json")
        require(type(process["pid"]) is int and process["pid"] > 0 and process["pid"] == process["session_id"], "fresh process/session identity differs")
        if lane["kind"] == "vllm":
            actual = evidence.json(directory / "vllm-startup.json")
            log = evidence.ref(actual["log"])
            expected = controller.legacy.validate_vllm_startup(log, {"active_capacity": setting["offered_concurrency"],
                        "token_budget": setting["vllm_token_budget"]}, controller.VLLM_RUNTIME)
            require({k:v for k,v in actual.items() if k != "log"} == {k:v for k,v in expected.items() if k != "log"}, "actual vLLM startup contract differs")
        c = setting["offered_concurrency"]
        previous_end = None
        for phase, count in (("warmup-nonstream", c*5), ("warmup-stream", c*5), ("retained", plan["workload"]["retained_requests_per_process"])):
            parsed, rows = verify_phase(directory, phase, count, c, reference, request, plan["request_timeout_seconds"], evidence)
            result["phases"].append(parsed)
            if not parsed["validated"]:
                return result, None
            if previous_end is not None:
                require(previous_end <= parsed["accounting"]["phase_started_ns"], "warmup/retained phases overlap")
            previous_end = parsed["accounting"]["phase_finished_ns"]
            rows_all.extend(rows)
        require(len({row["response_identity"]["id"] for row in rows_all}) == len(rows_all), "response IDs repeat across process phases")
        monitor = evidence.json(directory / "gpu-monitor.json")
        require(not monitor["errors"], "GPU/watchdog monitoring failed")
        for boundary in ("before", "after"):
            maps = evidence.json(directory / ("driver-maps-" + boundary + ".json"))
            require(maps["process_maps"], "owned CUDA mappings missing")
            for item in maps["process_maps"].values():
                require(item["compute_loaded"] is True and item["all_observed_vendor_mappings_pinned"] is True, "private CUDA mapping proof failed")
                for path, identity in item["private_vendor_files"].items():
                    require(prepared["runtime"]["files"].get(path) == identity["sha256"], "private driver mapping hash differs")
        exit_record = evidence.json(directory / "process-exit.json")
        cleanup = exit_record["cleanup"]
        require(exit_record["failure"] is None and cleanup["cleanup_verified"] is True and cleanup["remaining_owned_pids"] == []
                and cleanup["returncode"] in ((0,) if lane["kind"] == "riley" else (0, -15)), "owned process did not clean up gracefully")
        evidence.ref(cleanup["log"])
        saved = evidence.json(directory / "summary.json")
        require(saved["completed"] is True and saved["lane"] == name and saved["kind"] == lane["kind"] and saved["setting"] == setting
                and saved["summary"] == result["phases"][-1]["metrics"] and saved["cleanup"] == cleanup
                and saved["performance_claim"] is False, "lane summary is not the validated retained phase")
        require(evidence.ref(saved["raw"]) == (directory / "retained.jsonl").resolve(), "lane summary references different raw data")
        retained = result["phases"][-1]
        result.update(status="validated", process_id=process["pid"], retained=retained["metrics"],
                      token_delivery_observations=retained["token_delivery_observations"],
                      raw_response_samples=len(rows_all), retained_response_samples=len(rows), arrival_policy=plan["workload"]["arrival_policy"],
                      observed_calls_started_ns=result["phases"][0]["accounting"]["phase_started_ns"],
                      observed_calls_finished_ns=result["phases"][-1]["accounting"]["phase_finished_ns"])
        return result, saved
    except (Exception,) as error:
        result["errors"].append({"type": type(error).__name__, "message": str(error)})
        result["status"] = "invalid" if (directory / "summary.json").exists() else "incomplete"
        return result, None
    finally:
        result["diagnostics"] = {}
        for filename in ("process-exit.json", "gpu-monitor.json"):
            path = directory / filename
            if path.exists():
                try:
                    value = evidence.json(path)
                    if filename == "process-exit.json":
                        result["diagnostics"]["process_exit"] = value
                        if value.get("cleanup") and value["cleanup"].get("log"):
                            evidence.ref(value["cleanup"]["log"])
                    else:
                        result["diagnostics"]["monitor"] = {"sample_count": len(value.get("samples", [])), "errors": value.get("errors")}
                except Exception as error:
                    result["errors"].append({"artifact": filename, "type": type(error).__name__, "message": str(error)})


def distribution(values):
    return {"samples": len(values), "median": statistics.median(values), "min": min(values), "max": max(values), "values": values} if values else None


def aggregate_pairs(pairs):
    output = {"throughput_right_over_left": distribution([pair["ratios"]["throughput_right_over_left"] for pair in pairs])}
    for name in METRICS:
        key = name + "_right_over_left"
        output[key] = {stat: distribution([pair["ratios"][key][stat] for pair in pairs if pair["ratios"][key][stat] is not None])
                       for stat in ("median", "p95", "p99")}
    return output


def analyze(campaign, mappings=()):
    campaign = Path(campaign).resolve(strict=True)
    evidence = Evidence(mappings)
    prepared = evidence.json(campaign / "preparation.json")
    plan = evidence.json_ref(prepared["plan"])
    errors = []
    try:
        reference, request, binding = reference_and_proofs(plan, prepared, evidence)
        qualified = True
    except Exception as error:
        errors.append({"scope": "qualification", "type": type(error).__name__, "message": str(error)})
        qualified = False
        reference = request = binding = None
    completion = evidence.json(campaign / "completion.json") if (campaign / "completion.json").exists() else None
    final = evidence.json(campaign / "finalization.json") if (campaign / "finalization.json").exists() else None
    groups, expected_entries, process_ids = [], [], []
    previous_lane_end = None
    for setting in plan["settings"]:
        for comparison in plan["comparisons"]:
            group = {"setting": setting, "comparison": comparison, "ratio_direction": "right / left",
                     "left_lane": comparison["left"], "right_lane": comparison["right"], "pairs": []}
            complete_pairs = []
            for index in range(1, plan["workload"]["pairs"] + 1):
                directory = campaign / f"{setting['id']}-{comparison['id']}-pair{index:02d}"
                order = [comparison["left"], comparison["right"]]
                if index % 2 == 0:
                    order.reverse()
                pair = {"index": index, "order": order, "validated": False, "lanes": [], "errors": []}
                saved_lanes = {}
                if qualified:
                    for name in order:
                        lane, saved = verify_lane(directory / name, name, setting, plan, prepared, reference, request, binding, evidence)
                        pair["lanes"].append(lane)
                        if saved is not None:
                            start, end = lane["observed_calls_started_ns"], lane["observed_calls_finished_ns"]
                            if previous_lane_end is not None and start < previous_lane_end:
                                lane["status"] = "invalid"
                                lane["errors"].append({"scope": "serial_order", "message": "observed lane calls overlap or precede the previous lane in declared execution order"})
                            else:
                                saved_lanes[name] = saved
                                process_ids.append(lane["process_id"])
                            previous_lane_end = max(previous_lane_end or end, end)
                try:
                    require(qualified and len(saved_lanes) == 2, "pair lacks two qualified complete lanes")
                    stored = evidence.json(directory / "pair.json")
                    recomputed = controller.pair_result(comparison, index, order, saved_lanes)
                    require(stored == recomputed, "paired ratios/roles disagree with raw process metrics")
                    # Locate the original reference from finalization; keep it
                    # unchanged when artifacts have been copied to another host.
                    matches = [item for item in (final or {}).get("completed_pairs", [])
                               if evidence.locate(item["pair"]["path"]).resolve() == (directory / "pair.json").resolve()]
                    require(len(matches) == 1 and matches[0]["setting"] == setting["id"] and matches[0]["ratios"] == stored["ratios"], "pair missing/duplicated in finalization")
                    evidence.ref(matches[0]["pair"])
                    expected_entries.append(matches[0])
                    pair.update(validated=True, ratios=stored["ratios"])
                    complete_pairs.append(stored)
                except Exception as error:
                    pair["errors"].append({"type": type(error).__name__, "message": str(error)})
                group["pairs"].append(pair)
            group["validated_pairs"] = len(complete_pairs)
            group["planned_pairs"] = plan["workload"]["pairs"]
            group["paired_ratios"] = None
            group["_complete_pairs"] = complete_pairs
            groups.append(group)
    complete = qualified and completion is not None and final is not None and all(group["validated_pairs"] == group["planned_pairs"] for group in groups)
    if complete:
        try:
            require(len(set(process_ids)) == len(process_ids), "fresh server process IDs repeat")
            require(completion["schema_version"] == controller.RESULT and completion["completed"] is True
                    and completion["plan"] == prepared["plan"] and completion["workload"] == plan["workload"]
                    and completion["condition"] == prepared["condition"] and completion["pairs"] == final["completed_pairs"] == expected_entries
                    and final["failure"] is None and completion["restoration"] == final["restoration"]
                    and completion["performance_claim"] is False and completion["historical_c1_receipts_reinterpreted"] is False,
                    "completion/finalization inventory or identity differs")
            restoration = evidence.json_ref(completion["restoration"])
            require(restoration["alive_and_listening"] is True and restoration["commands_and_gui_environment_match"] is True
                    and restoration["all_relaunched_processes_have_pinned_vendor_maps"] is True
                    and len(restoration["processes"]) == 3, "restoration receipt incomplete")
            originals = {row["pid"]: row for row in prepared["sessions"]}
            require({row["original_pid"] for row in restoration["processes"]} == set(originals)
                    and len({row["new_pid"] for row in restoration["processes"]}) == 3
                    and all(row["port"] == originals[row["original_pid"]]["port"] for row in restoration["processes"]), "restored session scope differs")
            require(evidence.json_ref(restoration["runtime_manifest"]) == prepared["runtime"], "restored runtime snapshot differs")
        except Exception as error:
            errors.append({"scope": "completion", "type": type(error).__name__, "message": str(error)})
            complete = False
    for group in groups:
        group["comparison_eligible"] = complete
        if complete:
            group["paired_ratios"] = aggregate_pairs(group["_complete_pairs"])
        del group["_complete_pairs"]
    return {"schema_version": SCHEMA, "status": "validated-complete" if complete else "invalid" if completion else "incomplete",
            "validated_complete": complete, "campaign": str(campaign), "plan": prepared["plan"], "workload": plan["workload"],
            "groups": groups, "errors": errors, "finalization_failure": (final or {}).get("failure"),
            "inputs": evidence.finish(), "path_maps": [{"original": str(a), "local": str(b)} for a,b in evidence.mappings],
            "numerical_qualification_scope": "saved current-source GPU/default/raw receipts bound; GPU tests not rerun by analyzer",
            "metrics_scope": "client token delivery; first visible text and engine TTFT/TPOT are distinct",
            "serial_order_scope": "observed request-phase windows; process cleanup has no saved timestamp",
            "tail_scope": "finite per-process samples; initial 1000-request tails do not qualify P99 stability",
            "concurrency_scope": "offered C<=2; Riley active1; no broad high-concurrency qualification",
            "zero_itl_scope": "coalesced client observations; does not imply simultaneous engine token generation",
            "winner": None, "performance_claim": False, "p99_stability_qualified": False, "high_concurrency_qualified": False}


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
