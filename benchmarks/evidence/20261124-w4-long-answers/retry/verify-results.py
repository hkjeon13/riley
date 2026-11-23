"""Recompute W4's paired outputs and report the first-line diagnostic separately."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import runpy

FIRST = runpy.run_path(str(Path(__file__).with_name("check-first.py")))
from long_answer_check import load_fixture, verify


def compare(root):
    if (root / "campaign.exit").read_text().strip() != "0":
        raise ValueError("W4 comparison is incomplete")
    fixture = FIRST["FIXTURE_SHA"]
    cases = load_fixture(root / "facts.json.gz", fixture)
    paths = {engine: root / engine / "capture.json" for engine in ("riley", "vllm")}
    counts = verify(cases, fixture, paths["riley"], paths["vllm"])
    if counts != json.loads((root / "report.json").read_bytes()):
        raise ValueError("W4 score report differs from the paired captures")
    receipts = {engine: json.loads((path.parent / "receipt.json").read_bytes()) for engine, path in paths.items()}
    riley, vllm = receipts["riley"], receipts["vllm"]
    identities = ("fixture_sha256", "checkpoint_sha256", "gpu", "holder_pid")
    if (any(riley[key] != vllm[key] for key in identities) or riley["fixture_sha256"] != fixture
            or (riley["shutdown_exit"], vllm["shutdown_exit"]) not in ((0, 0), (0, -15))
            or sha256((root / "riley/receipt.json").read_bytes()).hexdigest() != FIRST["RECEIPT_SHA"]
            or riley["capture_sha256"] != FIRST["CAPTURE_SHA"] or vllm["vllm"] != "0.27.1+cu129"
            or any(counts["runtimes"][engine]["capture_sha256"] != receipt["capture_sha256"]
                   for engine, receipt in receipts.items())):
        raise ValueError("W4 comparison is not bound to the preserved Riley run and retried vLLM")
    reservations = {}
    for gpu in (6, 7):
        before, after = (json.loads((root / f"gpu{gpu}-{phase}.json").read_bytes()) for phase in ("before", "after"))
        if (any(row["physical_index"] != gpu or row["phase"] != "idle" or row["allocation_bytes"] != 20 * 2**30
                or row["other_processes"] for row in (before, after))
                or any(before[key] != after[key] for key in ("pid", "start_ticks", "gpu"))):
            raise ValueError(f"GPU {gpu} reservation did not return")
        reservations[gpu] = after
    if (riley["gpu"], riley["holder_pid"]) != (reservations[7]["gpu"], reservations[7]["pid"]):
        raise ValueError("W4 comparison does not match the restored GPU 7 holder")
    diagnostic = {engine: FIRST["summarize_modes"](cases, json.loads(path.read_bytes()))
                  for engine, path in paths.items()}
    return {"fixture_sha256": fixture, "gpu": riley["gpu"], "riley_image_id": riley["image_id"],
            "source_revision": riley["source_revision"], "vllm": vllm["vllm"], "runtimes": counts["runtimes"],
            "first_line_diagnostic": diagnostic}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", type=Path)
    print(json.dumps(compare(parser.parse_args().stage), indent=2, allow_nan=False))
