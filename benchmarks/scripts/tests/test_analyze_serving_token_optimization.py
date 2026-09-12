"""Synthetic persisted campaign tests; no remote, GPU or process execution."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import analyze_serving_token_optimization as analyzer

c, t = analyzer.controller, analyzer.tokens


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False))
    return c.evidence(path)


def response(ref, *, phase, index, concurrency, base, speed, prefix):
    streaming = phase != "warmup-nonstream"
    worker = index % concurrency
    start = base + (index // concurrency)*1000 + worker*3
    parser = t.TokenResponseParser(ref, streaming=streaming, started_ns=start)
    identity = {"id": f"{prefix}-{phase}-{index}", "model": ref.model, "object": "text_completion", "created": 1}
    usage = {"prompt_tokens": 128, "completion_tokens": 32, "total_tokens": 160}
    if streaming:
        for i, token in enumerate(ref.output_token_ids):
            choice = {"index": 0, "text": "x" if i == 1 else "", "token_ids": [token], "finish_reason": None}
            if i == 0:
                choice["prompt_token_ids"] = list(ref.prompt_token_ids)
            parser.feed_sse(json.dumps({**identity, "choices": [choice]}).encode(), start + (2+i//2)*speed)
        parser.feed_sse(json.dumps({**identity, "choices": [{"index": 0, "text": "", "finish_reason": "length"}]}).encode(), start+18*speed)
        parser.feed_sse(json.dumps({**identity, "choices": [], "usage": usage}).encode(), start+19*speed)
        parser.feed_sse(b"[DONE]", start+20*speed)
    else:
        parser.feed_nonstream(json.dumps({**identity, "choices": [{"index": 0, "text": "x", "token_ids": list(ref.output_token_ids),
            "prompt_token_ids": list(ref.prompt_token_ids), "finish_reason": "length"}], "usage": usage}).encode(), start+20*speed)
    row = parser.finish()
    payload = {"model": ref.model, "prompt": "Hello", "max_tokens": 32, "temperature": 0, "top_p": 1,
               "stream": streaming, "return_token_ids": True}
    if streaming:
        payload["stream_options"] = {"include_usage": True}
    row.update(status="success", parser_protocol_valid=True, transport_complete=True, error=None, cleanup_errors=[],
               owned_connection_closed=True, http_status=200, call_started_ns=start-1, call_finished_ns=row["finished_ns"]+1,
               headers_received_ns=start+1, total_deadline_seconds=30, request=payload,
               request_body_sha256=hashlib.sha256(json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()).hexdigest(),
               index=index, worker_id=worker, phase=phase, warmup=phase != "retained")
    return row


def phase_fixture(directory, ref, phase, count, concurrency, base, speed, prefix):
    rows = [response(ref, phase=phase, index=i, concurrency=concurrency, base=base, speed=speed, prefix=prefix) for i in range(count)]
    accounting = {"schema_version": t.PHASE_SCHEMA, "phase": phase, "warmup": phase != "retained", "offered_concurrency": concurrency,
        "requested": count, "attempted": count, "succeeded": count, "failed": 0, "unresolved_attempts": 0, "not_started": 0,
        "reference_matches": count, "protocol_valid_responses": count, "observed_max_request_in_flight": concurrency,
        "observed_max_call_in_flight": concurrency, "observed_max_successful_request_in_flight": concurrency,
        "phase_started_ns": base-2, "phase_finished_ns": max(row["call_finished_ns"] for row in rows)+1,
        "completed": True, "duplicate_response_ids": [], "reference_sha256s": [ref.sha256], "modes": ["strict"],
        "streaming_transports": [phase != "warmup-nonstream"], "worker_errors": [], "interrupted": None,
        "correctness_qualified": False, "performance_qualified": False,
        "failure_policy": "stop refill; drain owned in-flight calls with total deadlines; no retries"}
    directory.mkdir(parents=True, exist_ok=True)
    (directory / (phase+".jsonl")).write_text("".join(json.dumps(row)+"\n" for row in rows))
    put(directory/(phase+"-accounting.json"), accounting)
    summary = t.summarize_phase(rows, accounting)
    put(directory/(phase+"-descriptive.json"), summary)
    return summary


def campaign_fixture(root):
    refs = root / "refs"
    reference = t.TokenReference("g04-smol", tuple(range(128)), tuple(range(32)), "x")
    verification = put(refs/"native.json", {"status": "correctness-qualified", "unequal_tensor_elements": 0, "generated_tokens_exact": 32})
    binding = {"source": {"correctness_gate_id": "g04-vllm-smol-p128-v1", "correctness_report_sha256": verification["sha256"]},
               "workload": {"concurrency": 1}, "input_token_ids": list(range(128)), "generated_token_ids": list(range(32)),
               "environment": {"gpu": {"uuid": "GPU-test"}}}
    binding_ref = put(refs/"binding.json", binding)
    request_ref = put(refs/"request.json", {"prompt": "Hello", "prompt_token_ids": list(range(128))})
    parent = {"http_lanes": {"riley": {"expected_output_text": "x"}, "vllm": {"expected_output_text": "x"}},
              "immutable_files": {item["path"]: item["sha256"] for item in (verification, binding_ref, request_ref)}}
    parent_ref = put(refs/"parent.json", parent)
    reference_refs = dict(parent_c1_plan=parent_ref, verification=verification, binding=binding_ref, request=request_ref)
    plan = {"schema_version": c.SCHEMA, "reference": reference_refs,
            "workload": {"id": "synthetic-token-screen", "purpose": "screening", "pairs": 5, "retained_requests_per_process": 4,
                         "warmups_per_worker_per_transport": 5, "arrival_policy": "closed-loop-refill"},
            "settings": [{"id": "c2", "offered_concurrency": 2, "vllm_token_budget": 256}],
            "comparisons": [{"id": "candidate-baseline", "left": "baseline", "right": "candidate"}],
            "startup_timeout_seconds": 120, "request_timeout_seconds": 30, "cooldown_timeout_seconds": 120,
            "base_environment": {"HOME": "/explicit", "PATH": "/usr/bin"},
            "immutable_files": {item["path"]: item["sha256"] for item in reference_refs.values()}, "lanes": {}}
    for module in (c, t):
        plan["immutable_files"][str(Path(module.__file__).resolve())] = c.shared.digest(module.__file__)
    prepared = {"schema_version": c.RESULT, "measurement_started": False, "reference_sha256": reference.sha256,
                "workload": plan["workload"], "settings": plan["settings"], "condition": c.CONDITION,
                "transitive_evidence": {}, "source_identities": {},
                "sessions": [{"pid": i, "port": 9870+i} for i in range(1,4)],
                "runtime": {"child_only_overrides": {"PATH": "/private/bin:/usr/bin", "LD_LIBRARY_PATH": "/private/lib"},
                            "files": {"/private/libcuda": "c"*64}}}
    for i, name in enumerate(("baseline", "candidate")):
        binary = str(root/(name+"-binary")); sha = str(i)*64; commit = str(i)*40
        build = {"source_commit": commit, "binaries": {binary: sha}}
        build_ref = put(refs/(name+"-build.json"), build)
        model = {"passed": True, "source_clean": True, "source_commit": commit, "binaries": build["binaries"], "build": build_ref,
                 "full_logits_and_kv_exact": True, "gpu_tests_executed": True,
                 "checks": [{"passed": True, "passed_tests": 1, "failed_tests": 0, "ignored_tests": 0} for _ in range(4)]}
        model_ref = put(refs/(name+"-model.json"), model)
        per = [{"sampling_backend": sampler, "completed": True, "completion_checks": 44, "disconnect_checks": 2,
                "checks": [{"completed": True} for _ in range(46)]} for sampler in ("cpu", "gpu-greedy")]
        raw = {"schema_version": "riley.http-token-observation-correctness.v3", "completed": True, "gpu_tests_executed": True,
               "performance_claim": False, "source_build": build_ref, "source_commit": commit, "binary": {"path": binary, "sha256": sha},
               "reference_binding": binding_ref, "request": request_ref, "client_module": c.evidence(t.__file__), "prerequisites": [model_ref],
               "model_files": {}, "checks": [item for sampler in per for item in sampler["checks"]], "per_sampler": per,
               "proofs": dict.fromkeys(analyzer.PROOFS, True)}
        raw_ref = put(refs/(name+"-raw.json"), raw)
        plan["lanes"][name] = {"kind": "riley", "argv": [binary, "serve", "--bind", "127.0.0.1:{port}"], "port": 19341+i,
                               "cwd": "/qualified/source", "env": {}, "model_files": {}, "build": build_ref,
                               "model_proof": model_ref, "raw_token_proof": raw_ref}
        for item in (build_ref, model_ref, raw_ref):
            plan["immutable_files"][item["path"]] = prepared["transitive_evidence"][item["path"]] = item["sha256"]
        prepared["source_identities"][name] = {"source_clean": True, "source_commit": commit, "binaries": build["binaries"]}
    plan_ref = put(root/"plan.json", plan); prepared["plan"] = plan_ref
    campaign = root/"campaign"; put(campaign/"preparation.json", prepared)
    entries = []
    for index in range(1,6):
        setting, comparison = plan["settings"][0], plan["comparisons"][0]
        pair_dir = campaign/f"c2-candidate-baseline-pair{index:02d}"
        order = ["baseline", "candidate"] if index%2 else ["candidate", "baseline"]
        saved = {}
        for position, name in enumerate(order):
            directory = pair_dir/name; directory.mkdir(parents=True)
            lane = plan["lanes"][name]
            gpu = {"gpu_uuid": "GPU-test", "driver_version": "580.173.02", "temperature_c": 40, "memory_used_mib": 200, "compute_processes": []}
            put(directory/"launch.json", {"argv": c.lane_argv(lane,setting,directory), "environment": c.lane_environment(plan,lane,prepared["runtime"]),
                "cwd": lane["cwd"], "lane": name, "setting": setting, "fresh_process": True, "condition": c.CONDITION, "start_gpu": gpu})
            put(directory/"cooldown.json", [gpu]); pid=1000+index*10+position
            put(directory/"process.json", {"pid": pid, "session_id": pid})
            speed = 2 if name == "baseline" else 1
            for number, phase in enumerate(("warmup-nonstream", "warmup-stream", "retained")):
                summary = phase_fixture(directory, reference, phase, 4 if phase == "retained" else 10, 2,
                                        index*100000+position*40000+number*10000, speed, str(pid))
            put(directory/"gpu-monitor.json", {"samples": [], "errors": []})
            for boundary in ("before", "after"):
                put(directory/f"driver-maps-{boundary}.json", {"process_maps": {str(pid): {"compute_loaded": True,
                    "all_observed_vendor_mappings_pinned": True, "private_vendor_files": {"/private/libcuda": {"sha256": "c"*64}}}}})
            log = directory/"server.log"; log.write_text("graceful exit")
            cleanup = {"returncode": 0, "remaining_owned_pids": [], "zombie_pids": [], "cleanup_verified": True, "log": c.evidence(log)}
            put(directory/"process-exit.json", {"cleanup": cleanup, "failure": None})
            saved[name] = {"lane": name, "kind": "riley", "completed": True, "setting": setting, "summary": summary,
                           "cleanup": cleanup, "performance_claim": False, "raw": c.evidence(directory/"retained.jsonl")}
            put(directory/"summary.json", saved[name])
        pair = c.pair_result(comparison,index,order,saved)
        pair_ref = put(pair_dir/"pair.json", pair)
        entries.append({"setting": "c2", "pair": pair_ref, "ratios": pair["ratios"]})
    runtime_ref = put(root/"restored-runtime.json", prepared["runtime"])
    restoration = put(root/"restored.json", {"alive_and_listening": True, "commands_and_gui_environment_match": True,
          "all_relaunched_processes_have_pinned_vendor_maps": True, "runtime_manifest": runtime_ref,
          "processes": [{"original_pid": i, "new_pid": i+10, "port": 9870+i} for i in range(1,4)]})
    put(campaign/"finalization.json", {"completed_pairs": entries, "failure": None, "restoration": restoration})
    put(campaign/"completion.json", {"schema_version": c.RESULT, "completed": True, "plan": plan_ref, "workload": plan["workload"],
        "condition": c.CONDITION, "pairs": entries, "restoration": restoration, "performance_claim": False,
        "historical_c1_receipts_reinterpreted": False})
    return campaign


class AnalysisTests(unittest.TestCase):
    def test_complete_five_pairs_replay_raw_and_keep_delivery_semantics(self):
        with tempfile.TemporaryDirectory() as temporary:
            campaign = campaign_fixture(Path(temporary).resolve())
            result = analyzer.analyze(campaign)
            self.assertTrue(result["validated_complete"], result)
            group = result["groups"][0]
            self.assertEqual(group["validated_pairs"],5)
            self.assertEqual(group["paired_ratios"]["token_tpot_ms_right_over_left"]["median"]["median"], .5)
            lane = group["pairs"][0]["lanes"][0]
            self.assertEqual(lane["retained_response_samples"],4)
            self.assertEqual(lane["raw_response_samples"],24)
            self.assertEqual(lane["token_delivery_observations"]["blank_generated_frames"],124)
            self.assertEqual(lane["token_delivery_observations"]["observed_itls"],124)
            self.assertGreater(lane["token_delivery_observations"]["zero_observed_itls"],0)
            self.assertNotEqual(lane["retained"]["common_wall_ns"],lane["retained"]["phase_wall_ns"])
            self.assertIsNone(result["winner"])
            self.assertFalse(result["p99_stability_qualified"])

    def test_failed_and_missing_lanes_are_not_zero_or_promoted(self):
        with tempfile.TemporaryDirectory() as temporary:
            campaign = campaign_fixture(Path(temporary).resolve())
            (campaign/"completion.json").unlink()
            last = campaign/"c2-candidate-baseline-pair05"
            shutil.rmtree(last/"candidate")
            final = c.read(campaign/"finalization.json")
            final["completed_pairs"] = final["completed_pairs"][:-1];final["failure"]={"message":"request failed"}
            put(campaign/"finalization.json",final)
            result=analyzer.analyze(campaign)
            self.assertEqual(result["status"],"incomplete")
            self.assertEqual(result["groups"][0]["validated_pairs"],4)
            self.assertIsNone(result["groups"][0]["paired_ratios"])
            self.assertFalse(result["groups"][0]["comparison_eligible"])

    def test_mutated_raw_metrics_and_accounting_reject_claimed_completion(self):
        for artifact in ("metrics", "accounting", "warmup"):
            with self.subTest(artifact=artifact),tempfile.TemporaryDirectory() as temporary:
                campaign=campaign_fixture(Path(temporary).resolve())
                lane=campaign/"c2-candidate-baseline-pair01/baseline"
                if artifact=="accounting":
                    path=lane/"retained-accounting.json";value=c.read(path);value["requested"]=3;put(path,value)
                else:
                    path=lane/("warmup-stream.jsonl" if artifact=="warmup" else "retained.jsonl")
                    rows=[json.loads(line) for line in path.read_text().splitlines()]
                    if artifact=="metrics": rows[0]["metrics"]["token_ttft_ns"]+=1
                    else: rows[0]["frames"][0]["data"]=rows[0]["frames"][0]["data"].replace('"token_ids": [0]','"token_ids": [999]')
                    path.write_text("".join(json.dumps(row)+"\n" for row in rows))
                result=analyzer.analyze(campaign)
                self.assertFalse(result["validated_complete"])
                self.assertEqual(result["status"],"invalid")
                self.assertIsNone(result["groups"][0]["paired_ratios"])

    def test_unqualified_or_changed_proof_blocks_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            campaign=campaign_fixture(Path(temporary).resolve())
            root=campaign.parent;path=root/"refs/baseline-model.json"
            value=c.read(path);value["passed"]=False;put(path,value)
            result=analyzer.analyze(campaign)
            self.assertEqual(result["status"],"invalid")
            self.assertEqual(result["errors"][0]["scope"],"qualification")


    def test_headers_and_eof_cleanup_boundaries_are_checked(self):
        for change in ("headers", "phase-end"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temporary:
                campaign=campaign_fixture(Path(temporary).resolve())
                lane=campaign/"c2-candidate-baseline-pair01/baseline"
                if change=="headers":
                    path=lane/"retained.jsonl";rows=[json.loads(line) for line in path.read_text().splitlines()]
                    rows[0]["headers_received_ns"]=rows[0]["token_arrival_ns"][0]+1
                    path.write_text("".join(json.dumps(row)+"\n" for row in rows))
                else:
                    path=lane/"retained-accounting.json";value=c.read(path)
                    value["phase_finished_ns"]-=2;put(path,value)
                result=analyzer.analyze(campaign)
                self.assertFalse(result["validated_complete"])
                self.assertIsNone(result["groups"][0]["paired_ratios"])

    def test_partial_phase_reparses_successes_and_preserves_cleanup_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            campaign=campaign_fixture(Path(temporary).resolve())
            (campaign/"completion.json").unlink()
            lane=campaign/"c2-candidate-baseline-pair01/baseline"
            path=lane/"retained.jsonl";rows=[json.loads(line) for line in path.read_text().splitlines()]
            rows[-1]["status"]="failed";rows[-1]["error"]={"message":"timeout"}
            path.write_text("".join(json.dumps(row)+"\n" for row in rows))
            path=lane/"retained-accounting.json";value=c.read(path);value.update(completed=False,failed=1,succeeded=3);put(path,value)
            path=lane/"process-exit.json";value=c.read(path);value["failure"]={"message":"token timeout"};put(path,value)
            result=analyzer.analyze(campaign)
            observed=result["groups"][0]["pairs"][0]["lanes"][0]
            self.assertEqual(observed["phases"][-1]["successful_rows_validated"],3)
            self.assertEqual(observed["diagnostics"]["process_exit"]["failure"],{"message":"token timeout"})
            self.assertFalse(result["validated_complete"])


    def test_consistently_rehashed_overlapping_lanes_fail_serial_pair_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            campaign=campaign_fixture(Path(temporary).resolve())
            lane=campaign/"c2-candidate-baseline-pair01/candidate"
            shift=40000
            for phase in ("warmup-nonstream","warmup-stream","retained"):
                raw=lane/(phase+".jsonl");rows=[json.loads(line) for line in raw.read_text().splitlines()]
                for row in rows:
                    for key in ("call_started_ns","started_ns","finished_ns","call_finished_ns","headers_received_ns","done_ns"):
                        if row[key] is not None:row[key]-=shift
                    for key in ("token_arrival_ns","text_arrival_ns"):
                        row[key]=[value-shift for value in row[key]]
                    for frame in row["frames"]:frame["arrived_ns"]-=shift
                raw.write_text("".join(json.dumps(row)+"\n" for row in rows))
                path=lane/(phase+"-accounting.json");accounting=c.read(path)
                accounting["phase_started_ns"]-=shift;accounting["phase_finished_ns"]-=shift;put(path,accounting)
                summary=t.summarize_phase(rows,accounting);put(lane/(phase+"-descriptive.json"),summary)
            saved=c.read(lane/"summary.json");saved["summary"]=summary;saved["raw"]=c.evidence(lane/"retained.jsonl");put(lane/"summary.json",saved)
            pair_path=lane.parent/"pair.json";pair=c.read(pair_path);pair["processes"]["candidate"]=saved
            pair=c.pair_result(pair["comparison"],pair["index"],pair["order"],pair["processes"]);pair_ref=put(pair_path,pair)
            for filename,key in (("completion.json","pairs"),("finalization.json","completed_pairs")):
                path=campaign/filename;value=c.read(path);value[key][0]["pair"]=pair_ref;put(path,value)
            result=analyzer.analyze(campaign)
            self.assertEqual(result["status"],"invalid")
            target=result["groups"][0]["pairs"][0]["lanes"][1]
            self.assertEqual(target["errors"][-1]["scope"],"serial_order")
            self.assertIsNone(result["groups"][0]["paired_ratios"])

    def test_copied_evidence_maps_paths_without_rewriting_receipts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();old=root/"original";old.mkdir()
            campaign_fixture(old)
            new=root/"copy";shutil.copytree(old,new);shutil.rmtree(old)
            result=analyzer.analyze(new/"campaign",[(str(old),str(new))])
            self.assertTrue(result["validated_complete"],result)
            self.assertTrue(any(item["original_path"].startswith(str(old)) and item["local_path"].startswith(str(new)) for item in result["inputs"]))


if __name__=="__main__":unittest.main()
