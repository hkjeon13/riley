#!/usr/bin/env python3
"""CPU-only qualification of grouped token observations against frozen round13.

run --campaign DIR --client-v1 FILE --client-v2 FILE --tests FILE --output NEW_DIR
validate --receipt FILE

The read-only validate_completion(path) rechecks every pinned file, the CPU test
receipt and offline replay. It never runs tests, HTTP requests, a server or GPU.
The original failed V1 response remains incomplete; replay adds no finish/usage.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys

import serving_token_client as frozen

SCHEMA = "riley.grouped-token-client-qualification.v1"
V1_SHA = "a2a4a35569d6b892542097c119e2be8a6402beb60ab1565aa95658794c364766"
V2_SHA = "2bc9238265666b99654f4f4e456ca0bb10c45b8bed8f28493452b89c1f450fcf"
TESTS_SHA = "0f963a9cb4a37d031f2c12ef94788b97ddc3598022be527d66743ebb6f0aa353"
RAW_PHASES = {"warmup-nonstream": (5, 5), "warmup-stream": (5, 5), "retained": (124, 123)}


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    import hashlib
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            result.update(block)
    return result.hexdigest()


def evidence(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": digest(path)}


def checked(ref):
    require(set(ref) == {"path", "sha256"} and Path(ref["path"]).is_absolute(), "invalid pinned evidence")
    require(digest(ref["path"]) == ref["sha256"], "pinned evidence changed: " + ref["path"])
    return Path(ref["path"])


def read(path):
    return frozen.parse_json(Path(path).read_bytes())


def write(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def module(ref, name):
    path = checked(ref)
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def validate_client_inputs(refs):
    require(refs["client_v1"]["sha256"] == V1_SHA and refs["client_v2"]["sha256"] == V2_SHA
            and refs["tests"]["sha256"] == TESTS_SHA, "not the frozen V1/V2 client and 26-test qualification suite")
    tests = checked(refs["tests"])
    for key, name in (("client_v1", "serving_token_client.py"), ("client_v2", "serving_token_client_v2.py")):
        require((tests.parent.parent / name).resolve(strict=True) == checked(refs[key]).resolve(strict=True),
                "test import target differs from declared client: " + key)


def replay(module, reference, row, *, incomplete=False):
    parser = module.TokenResponseParser(reference, streaming=row["streaming"], started_ns=row["started_ns"], mode="strict")
    failure = None
    for frame in row["frames"]:
        try:
            payload = frame["data"].encode()
            if row["streaming"]:
                parser.feed_sse(payload, frame["arrived_ns"])
            else:
                parser.feed_nonstream(payload, frame["arrived_ns"])
        except module.ProtocolError as error:
            failure = str(error)
            break
    if incomplete:
        return parser.snapshot(), failure
    require(failure is None, "successful original row failed replay")
    return parser.finish(), None


def group_summary(module, snapshot):
    counts = []
    if snapshot["streaming"]:
        for frame in snapshot["frames"]:
            if frame["data"] != "[DONE]":
                for choice in module.parse_json(frame["data"])["choices"]:
                    if choice.get("token_ids"):
                        counts.append(len(choice["token_ids"]))
        expected = {"frame_token_counts": counts, "generated_frame_count": len(counts),
                    "multi_token_frame_count": sum(n > 1 for n in counts),
                    "tokens_in_multi_token_frames": sum(n for n in counts if n > 1),
                    "within_frame_zero_itl_count": sum(n-1 for n in counts),
                    "max_frame_token_count": max(counts, default=0)}
    else:
        expected = None
    require(snapshot["token_delivery_groups"] == expected, "group metadata differs from saved raw payloads")
    return expected


def replay_files(client_v1, client_v2, reference_values, raw_files):
    old_ref = client_v1.TokenReference(**reference_values)
    new_ref = client_v2.TokenReference(**reference_values)
    require(old_ref.sha256 == new_ref.sha256, "reference hashing changed")
    successful, partial = [], []
    require(set(raw_files) == set(RAW_PHASES), "exact round13 raw phase inventory required")
    for phase in RAW_PHASES:
        path = checked(raw_files[phase])
        rows = [client_v1.parse_json(line) for line in path.read_bytes().splitlines() if line.strip()]
        total, passed = RAW_PHASES[phase]
        require(len(rows) == total and sum(row["status"] == "success" for row in rows) == passed,
                "actual round13 request inventory differs")
        for index, row in enumerate(rows):
            require(row["schema_version"] == client_v1.SCHEMA and row["phase"] == phase and row["index"] == index
                    and row["mode"] == "strict" and row["reference_sha256"] == old_ref.sha256,
                    "original request schema/reference/index differs")
            if row["status"] == "success":
                old, _ = replay(client_v1, old_ref, row)
                new, _ = replay(client_v2, new_ref, row)
                require(all(row.get(key) == value for key, value in old.items()), "original saved fields disagree with raw V1 replay")
                require(all(new.get(key) == value for key, value in old.items() if key not in ("schema_version", "timing_scope")),
                        "group support changed successful IDs/text/usage/finish/timestamps/metrics")
                require(row["transport_complete"] is True and row["protocol_valid"] is True and row["reference_match"] is True,
                        "original successful request lacked complete exact transport")
                groups = group_summary(client_v2, new)
                require(groups is None or groups["max_frame_token_count"] == 1, "V1 successful stream unexpectedly contained grouped IDs")
                successful.append({"phase": phase, "index": index, "reference_sha256": old_ref.sha256,
                                   "old_new_common_fields_exact": True, "group_metadata_exact": True,
                                   "transport_complete_in_original_record": True})
            else:
                require(phase == "retained" and index == 123 and row["error"]["type"] == "ProtocolError"
                        and row["error"]["message"] == "multiple IDs share one SSE timestamp; token timing is ambiguous",
                        "not the recorded grouped-frame failure")
                old, failure = replay(client_v1, old_ref, row, incomplete=True)
                new, new_failure = replay(client_v2, new_ref, row, incomplete=True)
                last = client_v1.parse_json(row["frames"][-1]["data"])["choices"]
                require(len(last) == 1 and last[0]["token_ids"] == [314, 338] and last[0]["text"] == " is that"
                        and failure == row["error"]["message"] and new_failure is None,
                        "actual multi-token failure frame differs")
                require(old["token_ids"] == row["token_ids"] == list(old_ref.output_token_ids[:21])
                        and new["token_ids"] == list(new_ref.output_token_ids[:23])
                        and new["prompt_token_ids"] == list(new_ref.prompt_token_ids)
                        and new_ref.text.startswith(new["text"]), "partial token/text prefix differs")
                require(new["protocol_valid"] is False and new["reference_match"] is False
                        and new["finished_ns"] is None and new["done_ns"] is None and new["usage"] is None
                        and new["finish_reason"] is None and all(new["metrics"][key] is None for key in
                        ("e2e_ns", "token_ttft_ns", "token_tpot_ns", "first_text_ns")), "partial replay fabricated completion/metrics")
                groups = group_summary(client_v2, new)
                require(groups["multi_token_frame_count"] == 1 and groups["max_frame_token_count"] == 2
                        and groups["within_frame_zero_itl_count"] == 1
                        and new["token_arrival_ns"][-1] == new["token_arrival_ns"][-2] == row["frames"][-1]["arrived_ns"],
                        "grouped IDs did not share actual delivery timestamp")
                partial.append({"phase": phase, "index": index, "original_prefix_tokens": 21, "replayed_prefix_tokens": 23,
                                "observed_group_ids": [314,338], "observed_group_text": " is that",
                                "prefix_exact": True, "group_metadata": groups, "completion_recovered": False,
                                "finish_usage_done_added": False, "full_request_correctness_qualified": False,
                                "performance_qualified": False})
    require(len(successful) == 133 and len(partial) == 1, "replay totals differ")
    return {"successful_requests_replayed": 133, "retained_successes_replayed": 123,
            "successful_old_new_common_fields_exact": True, "successful": successful,
            "partial_requests_replayed": 1, "partial": partial, "performance_claim": False}


def validate_completion(path):
    result = read(path)
    require(result["schema_version"] == SCHEMA and result["completed"] is True
            and result["cpu_only"] is True and result["gpu_executed"] is False
            and result["performance_claim"] is False and result["measurement_client_qualified"] is True,
            "client qualification scope is invalid")
    refs = result["inputs"]
    for ref in refs.values():
        checked(ref)
    validate_client_inputs(refs)
    require(refs["runner"] == evidence(__file__) and refs["client_v1"]["sha256"] == V1_SHA,
            "validator/V1 identity differs")
    require(refs["python"]["sha256"] == digest(sys.executable), "qualification Python differs")
    old = module(refs["client_v1"], "qualified_token_v1")
    new = module(refs["client_v2"], "qualified_token_v2")
    require(old.SCHEMA == "riley.http-token-observation.v1" and new.SCHEMA == "riley.http-token-observation.v2"
            and new.PHASE_SCHEMA == "riley.http-token-phase.v2", "client observation schemas differ")
    require(result["client_v1"] == refs["client_v1"] and result["client_v2"] == refs["client_v2"], "client evidence aliases differ")
    test = result["tests"]
    log = checked(test["log"]).read_text()
    match = re.search(r"^Ran ([1-9][0-9]*) tests? in [0-9.]+s$", log, re.M)
    require(test["argv"] == [refs["python"]["path"], refs["tests"]["path"], "-v"] and test["returncode"] == 0
            and match and int(match.group(1)) == test["passed_tests"] == 26 and "\nOK\n" in log
            and not re.search(r"^(FAILED|ERROR:|FAIL:)", log, re.M), "actual CPU test receipt failed")
    preparation, plan = read(refs["preparation"]["path"]), read(refs["plan"]["path"])
    require(preparation["plan"] == refs["plan"] and plan["reference"]["binding"] == refs["binding"]
            and plan["reference"]["parent_c1_plan"] == refs["reference_parent"], "actual round13 reference provenance differs")
    require(not (Path(refs["preparation"]["path"]).parent / "completion.json").exists(), "original failed campaign acquired a completion claim")
    final = read(refs["finalization"]["path"])
    require(final["failure"] is not None and not final["completed_pairs"], "original round13 failure was relabelled")
    binding, parent = read(refs["binding"]["path"]), read(refs["reference_parent"]["path"])
    reference = {"model": "g04-smol", "prompt_token_ids": binding["input_token_ids"],
                 "output_token_ids": binding["generated_token_ids"], "text": parent["http_lanes"]["riley"]["expected_output_text"],
                 "finish_reason": "length"}
    require(reference == result["reference"] and preparation["reference_sha256"] == old.TokenReference(**reference).sha256,
            "replay reference differs from original measurement")
    require(result["replay"] == replay_files(old, new, reference, result["raw_files"]), "stored replay result differs")
    require(result["original_campaign_complete"] is False and result["production_or_server_configuration_changed"] is False,
            "incorrect recovered-campaign/source claim")
    return result


def run(args):
    output = args.output.resolve()
    require(not output.exists(), "output already exists")
    campaign = args.campaign.resolve(strict=True)
    preparation = campaign/"preparation.json"
    prep = read(preparation);plan_ref=prep["plan"];plan=read(checked(plan_ref))
    require(not (campaign/"completion.json").exists(), "requires original failed round13 campaign")
    inputs = {"runner": evidence(__file__), "python": evidence(sys.executable), "tests": evidence(args.tests),
              "client_v1": evidence(args.client_v1), "client_v2": evidence(args.client_v2),
              "preparation": evidence(preparation), "plan": plan_ref, "finalization": evidence(campaign/"finalization.json"),
              "binding": plan["reference"]["binding"], "reference_parent": plan["reference"]["parent_c1_plan"]}
    validate_client_inputs(inputs)
    raw_files = {}
    for phase in RAW_PHASES:
        found = list(campaign.rglob(phase+".jsonl"))
        require(len(found)==1, "expected one attempted round13 process per raw phase")
        raw_files[phase] = evidence(found[0])
    old=module(inputs["client_v1"],"qualified_token_v1");new=module(inputs["client_v2"],"qualified_token_v2")
    binding=read(checked(inputs["binding"]));parent=read(checked(inputs["reference_parent"]))
    reference={"model":"g04-smol","prompt_token_ids":binding["input_token_ids"],"output_token_ids":binding["generated_token_ids"],
               "text":parent["http_lanes"]["riley"]["expected_output_text"],"finish_reason":"length"}
    replayed=replay_files(old,new,reference,raw_files)
    output.mkdir(mode=0o700)
    argv=[inputs["python"]["path"],inputs["tests"]["path"],"-v"]
    with (output/"cpu-tests.log").open("x") as log:
        completed=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT,timeout=180,cwd=args.tests.resolve().parent)
    log=(output/"cpu-tests.log").read_text()
    matched=re.search(r"^Ran ([1-9][0-9]*) tests? in [0-9.]+s$",log,re.M)
    require(completed.returncode==0 and matched and "\nOK\n" in log,"CPU client tests failed")
    result={"schema_version":SCHEMA,"completed":True,"cpu_only":True,"gpu_executed":False,"performance_claim":False,
            "measurement_client_qualified":True,"inputs":inputs,"client_v1":inputs["client_v1"],"client_v2":inputs["client_v2"],
            "raw_files":raw_files,"reference":reference,"replay":replayed,
            "tests":{"argv":argv,"returncode":completed.returncode,"passed_tests":int(matched.group(1)),"log":evidence(output/"cpu-tests.log")},
            "original_campaign_complete":False,"production_or_server_configuration_changed":False,
            "scope":"grouped client delivery observations only; source/API proofs stay bound to their original V1 client"}
    # Validate the exact candidate result before publishing its completion name.
    temporary=output/"candidate.json";write(temporary,result)
    validate_completion(temporary)
    write(output/"completion.json",result)
    print(json.dumps(evidence(output/"completion.json")))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest="command",required=True)
    run_parser=commands.add_parser("run")
    for name in ("campaign","client-v1","client-v2","tests","output"):
        run_parser.add_argument("--"+name,required=True,type=Path)
    validate=commands.add_parser("validate");validate.add_argument("--receipt",required=True,type=Path)
    args=parser.parse_args()
    if args.command=="run":run(args)
    else:print(json.dumps({"validated":True,"receipt":evidence(args.receipt),"replay":validate_completion(args.receipt)["replay"]["successful_requests_replayed"]}))


if __name__=="__main__":main()
