"""Verify the aged-prefill completion experiment and the A100 reservation."""
from argparse import ArgumentParser
import gzip
import json
from pathlib import Path

from compare_prefill_continuation import compare, read, require

PROMOTION = "RILEY_W3_PREFILL_FINISH applied=true"


def evidence(root):
    result = compare(root, candidate_mode="finishing", env_name="RILEY_W3_PREFILL_FINISH", physical_gpu=6)
    for arm, count in (("riley", 0), ("finishing", 1)):
        run = root / f"qwen14-{arm}"
        before = read(root / f"idle-before-{arm}.json")
        after = read(run / "idle-after.json")
        receipt = read(run / "receipt.json")
        require(all(before[key] == after[key] for key in before if key != "updated_ns"),
                f"{arm}: hold changed during the experiment")
        require(after["phase"] == "idle" and after["physical_index"] == 6 and after["status"] == "running"
                and after["allocation_bytes"] == after["torch_reserved_bytes"] == 20 * 1024**3
                and not after["other_processes"] and after["pid"] == receipt["holder_pid"]
                and after["start_ticks"] == receipt["holder_start_ticks"] and after["gpu"] == receipt["gpu"],
                f"{arm}: reserved A100 was not restored")
        path = run / "server.log"
        server_log = path.read_text() if path.exists() else gzip.decompress(path.with_suffix(".log.gz").read_bytes()).decode()
        require(sum(line == PROMOTION for line in server_log.splitlines()) == count,
                f"{arm}: missing or unexpected prefill reorder")
    result["candidate_promoted_a_prompt"] = True
    result["gpu_holder_restored"] = True
    return result


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(evidence(args.root), handle, indent=2, allow_nan=False)
        handle.write("\n")
