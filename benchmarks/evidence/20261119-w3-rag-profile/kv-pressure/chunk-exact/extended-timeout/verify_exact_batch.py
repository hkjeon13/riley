"""Validate the four GPU arms and export counts, digests and divergence locations."""
from argparse import ArgumentParser
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
import re

from exact_batch import ARMS
from long_accuracy import common_prefix, read_workload
from verify_kv_pressure import DRAIN_GAUGES, digest, read, request_trace


WORKLOAD_SHA = "220f13a7306518f996480c608733623475471ff3721ea3aeb895a2d5f91abd11"
TOKENIZER_SHA = "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"
BINARY_SHA = "a73745bd007a7fe8d075af5b1823a365edead470088e1797af31b574cae908ae"
LEVELS = {"c1": 1, "c8": 3, "c32": 3}
FIELDS = ("prompt_tokens", "tokens", "text", "finish", "usage")


def compare(before, after):
    changes = []
    for request, (old, new) in enumerate(zip(before, after, strict=True)):
        assert old["case"] == new["case"]
        changed = [key for key in FIELDS if old[key] != new[key]]
        if changed:
            changes.append({
                "request": request,
                "case": old["case"],
                "changed_fields": changed,
                "common_generation_token_prefix": common_prefix(old["tokens"], new["tokens"]),
                "before_generated": len(old["tokens"]),
                "after_generated": len(new["tokens"]),
            })
    return {"compared": len(before), "exact": len(before) - len(changes), "differences": changes}


def trace_summary(path, capacity, chunk, rows):
    log = path.read_text()
    mapping, traces = request_trace(log, capacity)
    sections = re.split(r"(?m)^riley startup warm-up: ([0-9]+) internal requests in .* s$", log)
    startup, warmup_count, serving = sections
    assert len(mapping) == len(traces) and set(mapping.values()) == set(traces)

    def submissions(section):
        return [line.split()[1] for line in section.splitlines()
                if line.startswith("W3-PHASE ") and " front_submit " in line]

    internal = submissions(startup)
    measured = submissions(serving)
    assert len(internal) == int(warmup_count) == 48
    assert len(measured) == sum(map(len, rows.values()))
    assert len(set(internal + measured)) == len(mapping) and set(internal + measured) == set(mapping)
    prefill = defaultdict(list)
    for line in serving.splitlines():
        if line.startswith("W3-PHASE ") and " prefill_start " in line:
            _, request, phase, stamp, *fields = line.split()
            prefill[request].append(tuple(map(int, fields)))
    assert set(prefill) == set(measured)
    chunk_counts = Counter()
    final_lengths = {}
    for request, parts in prefill.items():
        previous = 0
        for idx, (start, end, final, batch, decodes) in enumerate(parts):
            assert start == previous and 0 < end - start <= chunk
            assert final == int(idx == len(parts) - 1)
            assert 0 <= decodes < batch <= 1024 and end - start <= batch - decodes
            chunk_counts[end - start] += 1
            previous = end
        final_lengths[request] = previous
    assert max(chunk_counts) == chunk

    level_results = {}
    offset = 0
    for level, group in rows.items():
        clients = int(level[1:])
        for start in range(0, len(group), clients):
            jobs = group[start:start + clients]
            frontend_ids = measured[offset + start:offset + start + len(jobs)]
            assert Counter(final_lengths[key] for key in frontend_ids) == Counter(
                len(row["prompt_tokens"]) for row in jobs
            )
        event_groups = [traces[mapping[key]] for key in measured[offset:offset + len(group)]]
        offset += len(group)
        events = [{event[0] for event in trace} for trace in event_groups]
        assert all("released" in seen for seen in events)
        assert all(("admitted" in seen) == ("submit_queued" in seen) for seen in events)
        level_results[level] = {
            "requests_queued": sum("submit_queued" in seen for seen in events),
            "requests_kv_blocked": sum("blocked" in seen for seen in events),
            "peak_reserved_blocks": max(event[3] for trace in event_groups for event in trace),
        }
    assert level_results["c1"]["requests_queued"] == 0
    if capacity == 768:
        assert level_results["c8"]["requests_kv_blocked"] > 0
        assert level_results["c32"]["requests_kv_blocked"] > 0
    return {
        "startup_warmup_requests": len(internal),
        "peak_prefill_chunk_tokens": max(chunk_counts),
        "full_size_chunks": chunk_counts[chunk],
        "requests_split_for_prefill": sum(len(parts) > 1 for parts in prefill.values()),
        "by_client_concurrency": level_results,
    }


def verify(root):
    workload = read_workload(root / "holdout.json.gz")
    assert workload.sha256 == WORKLOAD_SHA
    arms = {}
    configuration = None
    bindings = None
    all_rows = {}
    for mode, (chunk, capacity) in ARMS.items():
        stage = root / f"qwen14-{mode}"
        plan, receipt = (read(stage / f"{name}.json") for name in ("plan", "receipt"))
        assert receipt["model"] == "Qwen2.5-14B-Instruct" and receipt["mode"] == mode
        assert receipt["status"] == "passed" and receipt["common_kv_blocks"] == capacity
        assert receipt["holder_pid"] == 2431237
        assert receipt["holder_start_ticks"] == "6339570875" and receipt["nvml_samples"] > 0
        assert read(stage / "server-exit.json") == {"exit": 0}
        assert plan["profile"] is False and plan["check_token_ids"] is True and not plan["cells"]
        assert plan["env"] == {"RILEY_W3_REQUEST_TRACE": "1", "RILEY_W3_KV_TRACE": "1"}
        argv = plan["server"]
        assert argv[argv.index("--admission-timeout-ms") + 1] == "90000"
        varied = {"--bind": plan["address"], "--kv-blocks": str(capacity),
                  "--prefill-chunk-tokens": str(chunk)}
        assert all(argv[argv.index(flag) + 1] == value for flag, value in varied.items())
        omit = {argv.index(flag) + 1 for flag in varied}
        fixed = [value for i, value in enumerate(argv) if i not in omit]
        bound = {Path(name).name: value for name, value in receipt["sha256"].items()}
        assert bound.pop("plan.json") == digest(stage / "plan.json")
        assert bound["riley"] == BINARY_SHA and bound["tokenizer.json"] == TOKENIZER_SHA
        if configuration is None:
            configuration, bindings = fixed, bound
        else:
            assert fixed == configuration and bound == bindings
        before, after = (read(stage / f"{name}.json") for name in ("idle-before", "drain"))
        assert all(value[field] == 0 for value in (before, after) for field in DRAIN_GAUGES)
        assert all(before["counters"][key] == after["counters"][key]
                   for key in ("cancellations", "disconnects", "overloads"))
        for point in ("before", "after"):
            for gpu, pid, ticks in ((6, 2431228, "6339570869"), (7, 2431237, "6339570875")):
                reserved = read(root / f"gpu{gpu}-{point}-{mode}.json")
                assert (reserved["physical_index"], reserved["pid"], reserved["start_ticks"],
                        reserved["phase"]) == (gpu, pid, ticks, "idle")
                assert reserved["allocation_bytes"] == reserved["torch_reserved_bytes"] == 20 * 1024**3
                assert not reserved["other_processes"]
                if gpu == 7:
                    assert reserved["gpu"] == receipt["gpu"]

        probe_path = stage / "token-id-probe.json"
        assert receipt["results"]["token-id-probe"] == digest(probe_path)
        probe = read(probe_path)
        assert probe["workload_sha256"] == workload.sha256
        rows = probe["rows"]
        assert tuple(rows) == tuple(LEVELS)
        for level, repeat in LEVELS.items():
            for case, row in zip(workload.cases * repeat, rows[level], strict=True):
                assert row["case"] == case.label and row["prompt_tokens"] == case.expected_ids
                assert row["tokens"] and 0 < len(row["tokens"]) <= case.limit
        all_rows[mode] = rows
        traces = trace_summary(stage / "server.log", capacity, chunk, rows)
        arms[mode] = {
            "within_arm": {level: compare(rows["c1"] * repeat, rows[level])
                           for level, repeat in LEVELS.items() if level != "c1"},
            "scheduler": traces,
            "sha256": {filename: digest(stage / filename) for filename in (
                "plan.json", "receipt.json", "token-id-probe.json", "server.log", "drain.json",
            )},
        }

    pairwise = {}
    for left, right in combinations(all_rows, 2):
        pairwise[f"{left}_vs_{right}"] = {
            level: compare(all_rows[left][level], all_rows[right][level]) for level in LEVELS
        }
    return {
        "remote_stage": str(root),
        "model": "Qwen2.5-14B-Instruct",
        "inputs_sha256": {"holdout.json.gz": workload.sha256, **bindings},
        "cases": [{"case": case.label, "prompt_tokens": len(case.expected_ids),
                   "max_generated_tokens": case.limit} for case in workload.cases],
        "arms": arms,
        "cross_arm": pairwise,
        "result_manifest_sha256": digest(root / "result.sha256"),
    }


if __name__ == "__main__":
    import json

    parser = ArgumentParser(description=__doc__)
    parser.add_argument("stage", type=Path)
    args = parser.parse_args()
    assert (args.stage / "campaign.exit").read_text() == "0\n"
    print(json.dumps(verify(args.stage), indent=2, allow_nan=False))
