"""Partition the RAG p95 requests' prefill timelines in the preserved A100 trace."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import runpy
import tarfile
from tempfile import TemporaryDirectory

TRACE = "20261124-w3-c32-prefill-tail"
NATIVE = "20261124-w3-c32-native-repeat"
HISTORY = "20261124-w3-c32-rag-history"
REFERENCE = "20261124-w3-separated-client"
LABELS = ("rag-c8-s1-vllm-arrivals", "rag-c8-s2-vllm-arrivals", "c32-riley-arrivals")
IDENTITY = ("request_index", "scheduled_write_ns", "input_sha256", "prompt_tokens")

def read(archive, name):
    return json.load(archive.extractfile(name))

def digest(path):
    return sha256(path.read_bytes()).hexdigest()

def account(request, admissions, executions, prefill):
    rid = int(request["response_id"].removeprefix("cmpl-"), 16)
    ack = admissions[rid][-1]
    own = [(span, prefill[start][rid])
           for start, (span, _, _) in executions.items() if rid in prefill.get(start, {})]
    if not own or own[0][0].start < ack or len([work for _, work in own if work[2]]) != 1 or not own[-1][1][2]:
        raise ValueError("missing or inconsistent terminal prefill")
    ends = [position for _, (position, _, _) in own]
    if ends[-1] != request["prompt_tokens"] or any(
        pos - tokens != former for former, (_, (pos, tokens, _)) in zip([0, *ends[:-1]], own, strict=True)
    ):
        raise ValueError("prefill position continuity differs")
    final = own[-1][0].end
    own_starts = {span.start for span, _ in own}
    own_ns, other_prefill_ns, decode_only_ns = 0, 0, 0
    other_steps = []
    for start, (span, rows, decode) in executions.items():
        if span.end <= ack or span.start >= final:
            continue
        during = min(span.end, final) - max(span.start, ack)
        if start in own_starts:
            own_ns += during
        else:
            other_steps.append(during)
            if rows == decode:
                decode_only_ns += during
            else:
                other_prefill_ns += during
    outside = final - ack - own_ns - other_prefill_ns - decode_only_ns
    if outside < 0:
        raise ValueError("overlapping attribution intervals")
    return ({
        "prefill_chunks": len(own), "first_prefill_start_after_ack_ms": (own[0][0].start - ack) / 1e6,
        "own_prefill_execution_ms": own_ns / 1e6,
        "other_prefill_execution_ms": other_prefill_ns / 1e6,
        "decode_only_execution_ms": decode_only_ns / 1e6,
        "outside_execution_ms": outside / 1e6,
        "largest_other_iteration_ms": max(other_steps, default=0) / 1e6,
    }, sorted(other_steps, reverse=True))

def summarize(evidence):
    trace_path = evidence / TRACE
    trace = runpy.run_path(str(trace_path / "verify.py"))
    stages = runpy.run_path(str(evidence / trace["STEP"] / "stages.py"))
    observe = runpy.run_path(str(evidence / "20261124-w3-aged-c32-observation/verify.py"))
    history = runpy.run_path(str(evidence / HISTORY / "verify.py"))
    baseline = runpy.run_path(str(evidence / history["BASELINE"] / "verify.py"))
    repeat = runpy.run_path(str(evidence / NATIVE / "verify.py"))
    reg = json.loads((trace_path / "registration.json").read_text())
    p = baseline["percentile"]
    refs = evidence / REFERENCE / "initial-results.tar.gz"
    output = {}
    with (tarfile.open(trace_path / "raw-results.tar.gz") as archive,
          tarfile.open(evidence / HISTORY / "raw-results.tar.gz") as golden,
          tarfile.open(refs) as vllm):
        for gpu in ("6", "7"):
            arms = {}
            for mode in ("control", "reserve"):
                path = f"gpu{gpu}/rag/{mode}"
                captures = {label: read(archive, f"{path}/{label}.json") for label in LABELS}
                for label, capture in captures.items():
                    prior = read(golden, f"gpu{gpu}/rag/control/{label}.json")["measured"]
                    if any(any(a[key] != b[key] for key in (*IDENTITY, "output_sha256"))
                           for a,b in zip(prior,capture["measured"],strict=True)):
                        raise ValueError(f"{path}/{label} differs from the registered workload")
                log = archive.extractfile(f"{path}/server.log").read().decode()
                _, observations = observe["bind_observations"](log, captures, reg["holdout_requests_per_arm"])
                normalized, admissions, executions, prefill = trace["parse_trace"](log, stages)
                with TemporaryDirectory() as scratch:
                    receipt = Path(scratch) / "history.json"
                    receipt.write_bytes(archive.extractfile(f"{path}/history.json").read())
                    on_schedule = history["check_history"](receipt, LABELS, reg)["release_timed"]
                refs_by_label = (repeat["compare_vllm"](
                    {name: (captures[name]["measured"], captures[name]["statistics"]) for name in LABELS[:2]},
                    LABELS[:2], vllm, reg, p, on_schedule,
                ) if gpu == "7" else {})
                rows = {}
                for label in LABELS[:2]:
                    ordered = sorted(captures[label]["measured"], key=lambda x: x["ttft"])
                    tail = ordered[int(.95 * len(ordered))]
                    if tail["ttft"] * 1000 != captures[label]["statistics"]["ttft_p95_ms"]:
                        raise ValueError("RAG p95 disagrees with the measured responses")
                    idx = tail["request_index"]
                    timing = next(row for row in observations[label] if row["request_index"] == idx)
                    queues = stages["reconcile"](normalized, observations[label])
                    queue = next(row for row in queues if row["request_index"] == idx)
                    record, durations = account(tail, admissions, executions, prefill)
                    record.update(request_index=idx, prompt_tokens=tail["prompt_tokens"], ttft_ms=timing["ttft_ms"],
                                  command_queue_ms=queue["command_queue_ms"],
                                  admitted_to_first_content_ms=timing["admitted_to_first_visible_ms"])
                    ref = refs_by_label.get(label)
                    if ref and ref["ttft_p95_ratio"] is not None:
                        target = 1.15 * (timing["ttft_ms"] / ref["ttft_p95_ratio"])
                        deficit, best, count = record["ttft_ms"] - target, 0.0, 0
                        while best < deficit and count < len(durations):
                            best += durations[count] / 1e6
                            count += 1
                        record["frozen_vllm"] = {
                            **ref, "target_ms": target, "shortfall_ms": deficit,
                            "optimistic_minimum_other_iterations": count if best >= deficit else None,
                        }
                    rows[label] = record
                arms[mode] = rows
            output[gpu] = arms
    return {
        "trace_archive_sha256": digest(trace_path / "raw-results.tar.gz"),
        "vllm_archive_sha256": digest(refs),
        "gpus": output,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, default=Path("benchmarks/evidence"))
    args = parser.parse_args()
    print(json.dumps(summarize(args.evidence_root), indent=2, allow_nan=False))
