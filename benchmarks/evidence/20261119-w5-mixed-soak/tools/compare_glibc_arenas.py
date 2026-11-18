"""Compare two exact RSS runs that use different glibc arena limits."""
from argparse import ArgumentParser
import json
from pathlib import Path

from compare_anon_mappings import compare, read_bound_trace
from compare_cache_off_repeat import require, require_matched_soaks


def evidence(root, evidence_root):
    require((root / "campaign.exit").read_text().strip() == "0", "paired allocator run did not finish")
    mapping_inputs = evidence_root / "anon-mappings"
    reference = evidence_root / "smaps-rollup/result"
    runner = evidence_root / "glibc-arena-control/runner.sha256"
    results, receipts, soaks = {}, {}, {}
    for name, maximum in (("control", None), ("arena2", "2")):
        arm = root / name
        results[name] = compare(reference, arm, mapping_inputs, runner_digest_file=runner)
        receipt, soak, _, _ = read_bound_trace(arm)
        require(receipt["allocator_arm"] == name and receipt["server_malloc_arena_max"] == maximum,
                f"{name}: server allocator environment differs")
        receipts[name], soaks[name] = receipt, soak
    require_matched_soaks(receipts["control"], soaks["control"], receipts["arena2"], soaks["arena2"])
    libc = receipts["control"]["libc"]
    require(isinstance(libc, str) and libc.startswith("glibc ") and libc == receipts["arena2"]["libc"],
            "allocator runtime differs between arms")
    before, after = (soaks[name]["latency_windows"]["late"] for name in ("control", "arena2"))
    return {
        "libc": libc,
        "control": results["control"],
        "arena2": results["arena2"],
        "late_p95_change_percent": {key: (after[key] / before[key] - 1) * 100
                                    for key in ("ttft_ms_p95", "e2e_ms_p95")},
    }


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("evidence_root", type=Path)
    args = parser.parse_args()
    print(json.dumps(evidence(args.root, args.evidence_root), indent=2, allow_nan=False))
