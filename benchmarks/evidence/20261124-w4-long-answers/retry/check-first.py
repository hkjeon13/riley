"""Verify the Riley receipts from the W4 attempt that stopped at vLLM startup."""

import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import sys

FIXTURE = Path(__file__).resolve().parents[1] / "facts.json.gz"
sys.path.insert(0, str(FIXTURE.parents[2] / "live"))
from long_answer_check import load_fixture, same_completion, score

CAPTURE_SHA = "fd1daa0ad4945c225cad455283221329fce1b9a11f7e46a6f856c93ce0b15082"
RECEIPT_SHA = "43529941faaafff0350e1a622d4dad47276cea7d99fd20a5ec4295061212ad79"
FIXTURE_SHA = "e68ed38a73caad88ae8de0a0000a439f732331a9b118849de28d92d4366efda3"
ERROR = "No such file or directory: 'ninja'"


def summarize_modes(cases, capture):
    modes = {}
    for mode, rows in capture["modes"].items():
        totals = defaultdict(lambda: dict.fromkeys(("requests", "complete_exact_answers", "first_line_matches", "length"), 0))
        for case, row in zip(cases, rows, strict=True):
            result = score(case, row)
            first_line = next((line.strip() for line in row["text"].splitlines() if line.strip()), "")
            reached = totals[case["task"]]
            reached["requests"] += 1
            reached["complete_exact_answers"] += result["complete_exact_answer"]
            reached["first_line_matches"] += first_line == case["answer"]
            reached["length"] += result["finish"] == "length"
        modes[mode] = dict(totals)
    return modes


def verify(root):
    capture_bytes = (root / "riley/capture.json").read_bytes()
    receipt_bytes = (root / "riley/receipt.json").read_bytes()
    if sha256(capture_bytes).hexdigest() != CAPTURE_SHA or sha256(receipt_bytes).hexdigest() != RECEIPT_SHA:
        raise ValueError("W4 Riley evidence differs")
    cases = load_fixture(FIXTURE, FIXTURE_SHA)
    capture, receipt = json.loads(capture_bytes), json.loads(receipt_bytes)
    gpu = json.loads((root / "gpu7-after-vllm.json").read_bytes())
    if (receipt["capture_sha256"] != CAPTURE_SHA
            or receipt["fixture_sha256"] != capture["fixture_sha256"] or capture["fixture_sha256"] != FIXTURE_SHA
            or capture["runtime"] != "riley" or receipt["shutdown_exit"] != 0
            or capture["modes"].keys() != {"c1", "c8"}
            or gpu["pid"] != receipt["holder_pid"] or gpu["gpu"] != receipt["gpu"]
            or gpu["phase"] != "idle" or gpu["allocation_bytes"] != 20 * 2**30 or gpu["other_processes"]):
        raise ValueError("W4 Riley result is not bound to its fixture and restored GPU reservation")
    if ((root / "campaign.exit").read_text().strip() != "1"
            or ERROR not in (root / "vllm/server.log").read_text()):
        raise ValueError("The recorded vLLM startup failure differs")
    c1, c8 = (capture["modes"][mode] for mode in ("c1", "c8"))
    preserved = sum(same_completion(a, b) and a["usage"] == b["usage"] for a, b in zip(c1, c8, strict=True))
    return {"riley_capture_sha256": CAPTURE_SHA, "riley_receipt_sha256": RECEIPT_SHA, "fixture_sha256": FIXTURE_SHA,
            "riley_image_id": receipt["image_id"], "source_revision": receipt["source_revision"],
            "gpu": receipt["gpu"], "riley_batch_invariant": preserved, "riley_by_mode": summarize_modes(cases, capture),
            "vllm_startup_error": ERROR}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", type=Path)
    print(json.dumps(verify(parser.parse_args().stage), indent=2, allow_nan=False))
