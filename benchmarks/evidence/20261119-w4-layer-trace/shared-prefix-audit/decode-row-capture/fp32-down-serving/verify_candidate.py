"""Validate same-binary A/B serving and compare both pinned corpora without exporting raw responses."""
from hashlib import file_digest
import json
from pathlib import Path
import sys

from long_accuracy import common_prefix, read_capture, read_workload, terminal_normalized


FIELDS = ("tokens", "text", "finish", "usage")
BUCKETS = (1, 2, 4, 8, 16, 24, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256)
GROUPS = ("c1", "diagnostic", "c8", "c32")


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def modes(doc, size):
    c1 = doc["baseline"]
    groups = {
        "c1": list(zip(range(size), c1, strict=True)),
        "diagnostic": list(zip(range(size), doc["diagnostic"], strict=True)),
    }
    for n in (8, 32):
        rows = doc["repeats"][str(n)]
        assert [row["case"] for row in rows] == list(range(size)) * 3
        groups[f"c{n}"] = [(row["case"], row) for row in rows]
    return groups


def compared(before, after, workload, match_prompt=False):
    fields = (*FIELDS, "prompt_tokens") if match_prompt else FIELDS
    differences = []
    for pos, ((i, left), (j, right)) in enumerate(zip(before, after, strict=True)):
        assert i == j
        changed = [key for key in fields if left[key] != right[key]]
        if changed:
            differences.append({"ordinal": pos, "case": workload.cases[i].label, "changed": changed,
                                "token_common_prefix": common_prefix(left["tokens"], right["tokens"])})
    return {"requests": len(before), "equal": len(before) - len(differences), "differences": differences}


def compare_groups(reference, trial, workload):
    return {name: compared(reference[name], trial[name], workload, name == "c1") for name in GROUPS}


def within(groups, workload):
    return {key: compared([(i, groups["c1"][i][1]) for i, _ in trial], trial, workload)
            for key, trial in groups.items() if key != "c1"}


def hf_results(reference, workloads, arm_docs, historical):
    lines = reference["cases"]
    eos = reference["eos_token_ids"]
    assert len(lines) == len(workloads.cases) and all(type(i) is int for i in eos)
    output = []
    for idx, (item, case) in enumerate(zip(lines, workloads.cases, strict=True)):
        assert item["case"] == idx
        if "label" in item:
            assert str(item["label"]) == str(case.label)
        expected_prompt = historical["baseline"][idx]["prompt_tokens"]
        target = terminal_normalized(item["hf_tokens"], item["hf_finish"], eos)
        counts = {}
        for arm, rows in (("historical", historical), *arm_docs.items()):
            response = rows["baseline"][idx]
            assert response["prompt_tokens"] == expected_prompt and item["prompt_tokens"] == len(expected_prompt)
            tokens = terminal_normalized(response["tokens"], response["finish"], eos)
            counts[arm] = {"length": len(tokens), "exact": tokens == target,
                           "finish_matches": response["finish"] == item["hf_finish"],
                           "common_prefix": common_prefix(tokens, target)}
        assert counts["historical"]["exact"] == item["exact_token_match"]
        output.append({"case": case.label, "input_tokens": len(expected_prompt), "hf_tokens": len(target), **counts})
    return {
        "cases": output,
        "exact_matches": {arm: sum(x[arm]["exact"] for x in output) for arm in arm_docs},
        "gained": [x["case"] for x in output if not x["baseline"]["exact"] and x["candidate"]["exact"]],
        "lost": [x["case"] for x in output if x["baseline"]["exact"] and not x["candidate"]["exact"]],
    }


def verify_receipts(root):
    binary = (root / "binary.sha256").read_text().split()[0]
    assert digest(root / "target/release/riley") == binary
    template = None
    candidate_m = set()
    for arm in ("baseline", "candidate"):
        receipt = read(root / arm / "receipt.json")
        assert (receipt["mode"], receipt["server_exit"], receipt["status"], receipt["observed_active"]) == (arm, 0, "passed", True)
        assert receipt["gpu_uuid"] == read(root / f"gpu7-{arm}-before.json")["gpu"]
        assert receipt["sha256"][str(root / "target/release/riley")] == binary
        argv = receipt["server_argv"]
        assert argv[argv.index("--inference-profile") + 1] == "native-bf16-varlen-v1"
        assert argv[argv.index("--max-active-sequences") + 1] == "32"
        assert argv[argv.index("--max-sequence-tokens") + 1] == "8192"
        stripped = argv[:]
        stripped[stripped.index("--bind") + 1] = "loopback"
        if template is None:
            template = stripped
        else:
            assert template == stripped
        log = (root / arm / "server.log").read_text()
        receipt_lines = [line for line in log.splitlines() if line.startswith("W4_DOWN_FP32_PLAN ")]
        assert bool(receipt_lines) == (arm == "candidate")
        for line in receipt_lines:
            data = dict(word.split("=", 1) for word in line.split()[1:])
            rows = int(data["m"])
            expected = dict(id="21", tile="18", stages="18", split_k="3", reduction="2",
                            workspace=str(43008 * rows), flags="131586")
            assert all(data[key] == val for key, val in expected.items())
            candidate_m.add(rows)
    assert 1 in candidate_m and candidate_m.issubset(BUCKETS)
    for arm in ("baseline", "candidate"):
        for point in ("before", "after"):
            for gpu, pid, ticks in ((6, 2431228, "6339570869"), (7, 2431237, "6339570875")):
                state = read(root / f"gpu{gpu}-{arm}-{point}.json")
                assert (state["physical_index"], state["pid"], state["start_ticks"], state["phase"]) == (gpu, pid, ticks, "idle")
                assert state["allocation_bytes"] == state["torch_reserved_bytes"] == 20 * 1024**3
                assert not state["other_processes"]
    return binary, sorted(candidate_m), template


def summarize(root):
    binary, prepared_rows, argv = verify_receipts(root)
    suites = {}
    for suite in ("long", "holdout"):
        fixture = root / "fixtures" / f"20261119-w4-{suite}"
        workload = read_workload(fixture / "holdout.json.gz" if suite == "holdout" else None)
        historical = read_capture(fixture / "riley/capture.json.gz", workload)
        docs = {arm: read_capture(root / arm / f"rows/{suite}.json", workload) for arm in ("baseline", "candidate")}
        grouped = {arm: modes(doc, len(workload.cases)) for arm, doc in docs.items()}
        suite_result = {
            "historical_to_baseline": compare_groups(modes(historical, len(workload.cases)), grouped["baseline"], workload),
            "baseline_to_candidate": compare_groups(grouped["baseline"], grouped["candidate"], workload),
            "within_arm": {arm: within(data, workload) for arm, data in grouped.items()},
            "hf": hf_results(read(fixture / "hf/reference.json"), workload, docs, historical),
            "capture_sha256": {arm: digest(root / arm / f"rows/{suite}.json") for arm in docs},
        }
        suites[suite] = suite_result
    return {
        "remote_stage": str(root), "binary_sha256": binary, "server_options": argv, "prepared_m": prepared_rows,
        "result_manifest_sha256": digest(root / "result.sha256"), "gpu_idle_reservation_gib": {"6": 20, "7": 20},
        "suites": suites,
    }


if __name__ == "__main__":
    print(json.dumps(summarize(Path(sys.argv[1])), indent=2, allow_nan=False))
