"""Verify the four image runs and recompute latency from closed-loop request records."""

from argparse import ArgumentParser
import json
from math import isclose
from pathlib import Path

from audit_allocator_ttft import percentile, split_timing
from compare_allocator_performance import ORDER, verify
from performance_client import REQUESTS
from verify_a100_candidate import digest, read, require


def raw_result(root, label, expected_p95):
    path = root / f"{label}.records.json"
    rows = read(path)
    require(len(rows) == REQUESTS, "incomplete request timings")
    timing = [split_timing(row) for row in rows]
    p95 = {key: percentile([item[key] for item in timing], .95) for key in timing[0]}
    require(isclose(p95["ttft_ms"], expected_p95, rel_tol=0, abs_tol=1e-9), "TTFT p95 differs")
    return {"records_sha256": digest(path), "p95_ms": p95}


def confirmation(package, evidence, stage, prior):
    report = verify(package, evidence, stage)
    labels = list(report["cells"])
    require(len(labels) == 3, "need three seed cells")
    for name in ORDER:
        root = stage / name / "gpu-evidence"
        recorded = [line.split()[1] for line in (root / "api.log").read_text().splitlines()
                    if line.startswith("RAG_CELL_COMPLETED ")]
        expected = labels if name.endswith("-first") else list(reversed(labels))
        require(recorded == expected, f"{name}: workload order differs")
        require(report["arms"][name]["api_checks"] == 40, f"{name}: expected 40 API checks")
        for label, cell in report["cells"].items():
            timing = raw_result(root, label, cell["metrics"][name]["ttft_p95_ms"])
            cell.setdefault("request_timing", {})[name] = timing
    report["scope"] = ("Synthetic closed-loop RAG C8 on the two pinned images, Qwen2.5-7B-Instruct, A100 6. "
                       "256 requests per seed and arm, two reversed image pairs.")
    report["remote_stage"] = str(stage)
    report["prior_closed_loop_summary_sha256"] = digest(prior / "verified-confirmation.json")
    return report


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for key in ("package", "evidence", "stage", "prior"):
        parser.add_argument(key, type=Path)
    args = parser.parse_args()
    print(json.dumps(confirmation(args.package, args.evidence, args.stage, args.prior), indent=2, allow_nan=False))
