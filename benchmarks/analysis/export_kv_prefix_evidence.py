"""Validate the small COW GPU gate and emit a reproducible evidence receipt."""
import argparse
import hashlib
import json
import re
import sqlite3
from collections import Counter
from pathlib import Path


def export(repo: Path, evidence: Path, output: Path) -> None:
    assert (evidence / "completion.txt").read_text().strip() == "complete"
    assert not (evidence / "compute-after.txt").read_text().strip()
    assert not list(evidence.glob("*.nsys-rep")), "original profiler reports must not be published"
    source_hashes = {}
    for line in (evidence / "source.sha256").read_text().splitlines():
        digest, relative = line.split(maxsplit=1)
        source = (repo / relative).resolve()
        assert source.is_relative_to(repo.resolve()), relative
        assert hashlib.sha256(source.read_bytes()).hexdigest() == digest, relative
        source_hashes[relative] = digest
    gpu = (evidence / "gpu.log").read_text()
    assert "test result: ok. 4 passed; 0 failed; 0 ignored" in gpu
    assert "ambiguous_completion=true retained_host_pages=2 copy_use_releases=0 cleanup=process_exit" in gpu
    cases = re.findall(r"kv-copy mode=(\w+) tokens=(\d+) bytes_verified=(\d+) host_pages=0 cuda_allocations=0", gpu)
    expected = Counter([("publish", str(n)) for n in [1, 15, 17, 31, 33]]
                       + [(mode, "17") for mode in ["cancel", "orphan", "partial"]])
    assert Counter((mode, length) for mode, length, _ in cases) == expected
    assert all(int(size) == 24576 for _, _, size in cases)
    memcheck = (evidence / "memcheck.log").read_text()
    assert "ERROR SUMMARY: 0 errors" in memcheck
    assert "LEAK SUMMARY: 0 bytes leaked in 0 allocations" in memcheck
    assert "test result: ok. 3 passed; 0 failed; 0 ignored" in memcheck
    assert "test result: ok. 7 passed; 0 failed; 0 ignored" in (evidence / "memory-regression.log").read_text()
    with sqlite3.connect(f"file:{evidence / 'copy.sqlite'}?mode=ro", uri=True) as database:
        tables = {row[0] for row in database.execute("select name from sqlite_master where type='table'")}
        assert tables == {"CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_RUNTIME",
                          "ENUM_CUDA_MEMCPY_OPER", "StringIds", "RILEY_TRACE_REDACTION"}, tables
        assert database.execute("select version from RILEY_TRACE_REDACTION").fetchone() == (1,)
        copies = {label: {"operations": count, "bytes": size} for label, count, size in database.execute(
            "select e.label,count(*),sum(m.bytes) from CUPTI_ACTIVITY_KIND_MEMCPY m "
            "join ENUM_CUDA_MEMCPY_OPER e on e.id=m.copyKind group by e.label")}
        events = {name.split("_v")[0]: count for name, count in database.execute(
            "select s.value,count(*) from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s "
            "on s.id=r.nameId where s.value like '%Event%' group by s.value")}
    assert copies["Device-to-Device"] == {"operations": 85, "bytes": 6736}, copies
    for operation in ["cudaEventCreateWithFlags", "cudaEventRecord", "cudaEventDestroy"]:
        assert events[operation] == 8, events
    assert events["cudaEventQuery"] == 5 and 3 <= events["cudaEventSynchronize"] <= 8, events
    receipt = {
        "scope": "local COW transfer correctness and lifetime; not model or serving qualification",
        "gpu_tests_passed": 4,
        "normal_cancel_orphan_partial_cases": len(cases),
        "whole_buffer_bytes_verified": sum(int(size) for _, _, size in cases),
        "ambiguous_completion": "isolated process retained both host pages and all native copy uses until exit",
        "memcheck_errors": 0,
        "memcheck_leaked_bytes": 0,
        "memcheck_scope": "8 cases; deliberate ambiguity retention tested separately",
        "existing_memory_regression_tests_passed": 7,
        "trace_copies": copies,
        "trace_event_calls": events,
        "source_hashes": source_hashes,
        "evidence_files_sha256": {str(path.relative_to(evidence)): hashlib.sha256(path.read_bytes()).hexdigest()
                             for path in sorted(evidence.rglob("*")) if path.is_file()},
    }
    with output.open("x") as destination:
        json.dump(receipt, destination, indent=2)
        destination.write("\n")
    print(json.dumps({key: value for key, value in receipt.items() if not key.endswith("hashes") and key != "evidence_files_sha256"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("repo", type=Path)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    export(args.repo, args.evidence, args.output)
