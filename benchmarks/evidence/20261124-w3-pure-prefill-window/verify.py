"""Measure the exposure of RAG tail requests to steps without decode rows."""

import argparse
from collections import Counter
from hashlib import file_digest
import json
from pathlib import Path
import runpy
import tarfile

TRACE_NAME = "20261124-w3-c32-prefill-tail"
ACCOUNTING_NAME = "20261124-w3-rag-tail-accounting"


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def count_steps(executions, start, end):
    buckets = Counter()
    elapsed = Counter()
    for span, rows, decode in executions.values():
        overlap = min(span.end, end) - max(span.start, start)
        if overlap <= 0:
            continue
        kind = "mixed" if 0 < decode < rows else "decode_only" if decode else "prefill_only"
        buckets[kind] += 1
        elapsed[kind] += overlap
    return {
        kind: {"steps": buckets[kind], "ms": elapsed[kind] / 1e6}
        for kind in ("prefill_only", "mixed", "decode_only")
    }


def summarize(root):
    trace_root = root / TRACE_NAME
    trace_path = trace_root / "raw-results.tar.gz"
    accounting_path = root / ACCOUNTING_NAME / "verified.json"
    accounting = json.loads(accounting_path.read_bytes())
    trace_hash = digest(trace_path)
    if accounting["trace_archive_sha256"] != trace_hash:
        raise ValueError("the previously verified RAG accounting uses a different trace")
    trace = runpy.run_path(str(trace_root / "verify.py"))
    stages = runpy.run_path(str(root / trace["STEP"] / "stages.py"))
    result = {}
    with tarfile.open(trace_path, "r:gz") as archive:
        for gpu, arms in accounting["gpus"].items():
            summaries = {}
            for mode, cells in arms.items():
                prefix = f"gpu{gpu}/rag/{mode}"
                log = archive.extractfile(f"{prefix}/server.log").read().decode()
                _, admissions, executions, prefill = trace["parse_trace"](log, stages)
                finish = {}
                for start, group in prefill.items():
                    for request, (_, _, terminal) in group.items():
                        if terminal:
                            if request in finish:
                                raise ValueError("duplicate request completion in trace")
                            finish[request] = executions[start][0].end
                summaries[mode] = {}
                for label, baseline in cells.items():
                    capture = json.load(archive.extractfile(f"{prefix}/{label}.json"))
                    rows = sorted(capture["measured"], key=lambda row: row["ttft"])
                    if len(rows) != 64:
                        raise ValueError("expected the frozen 64-request RAG replay")
                    p95 = rows[int(.95 * len(rows))]
                    if (p95["request_index"] != baseline["request_index"]
                            or p95["ttft"] * 1000 != capture["statistics"]["ttft_p95_ms"]):
                        raise ValueError("p95 request disagrees with prior accounting")
                    tail = []
                    p95_post_admission = None
                    for row in rows[int(.90 * len(rows)):]:
                        rid = int(row["response_id"].removeprefix("cmpl-"), 16)
                        submit, _, _, admission = admissions[rid]
                        terminal = finish[rid]
                        if not submit <= admission < terminal:
                            raise ValueError("request escaped the registered timeline")
                        full = count_steps(executions, submit, terminal)
                        tail.append({"request_index": row["request_index"], "submit_to_prefill_end": full})
                        if row is p95:
                            p95_post_admission = count_steps(executions, admission, terminal)
                    if p95_post_admission is None:
                        raise ValueError("missing p95 request in the inspected tail")
                    known_time = sum(baseline[key] for key in (
                        "own_prefill_execution_ms", "other_prefill_execution_ms", "decode_only_execution_ms",
                    ))
                    measured_time = sum(kind["ms"] for kind in p95_post_admission.values())
                    if abs(known_time - measured_time) > 1e-6:
                        raise ValueError("RAG step classes do not cover the previously audited interval")
                    summaries[mode][label] = {
                        "p95_request_index": p95["request_index"],
                        "p95_post_admission": p95_post_admission,
                        "p95_target_gap_ms": baseline.get("frozen_vllm", {}).get("shortfall_ms"),
                        "tail_requests": tail,
                        "tail_requests_with_prefill_only": sum(
                            bool(row["submit_to_prefill_end"]["prefill_only"]["steps"]) for row in tail
                        ),
                    }
            result[gpu] = summaries
    return {
        "trace_sha256": trace_hash,
        "accounting_sha256": digest(accounting_path),
        "gpus": result,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, default=Path("benchmarks/evidence"))
    args = parser.parse_args()
    print(json.dumps(summarize(args.evidence_root), indent=2, allow_nan=False))
