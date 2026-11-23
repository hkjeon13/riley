"""Independently verify the API controls, identity and captured arithmetic results."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import sys
import tarfile


HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / "20261124-w4-arithmetic-controls"
sys.path[0:0] = [str(PRIOR), str(HERE.parent / "20261124-w4-chat-answers")]
from structured_arithmetic import report, workload


def read(path):
    return json.loads(path.read_bytes())


def verify(root):
    registration = read(HERE / "registration.json")
    if (root / "campaign.exit").read_text().strip() != "0":
        raise ValueError("structured-arithmetic campaign did not complete")
    for name, key in (("dispatch.sha256", "dispatch_sha256"), ("inputs.json", "inputs_sha256")):
        if sha256((root / name).read_bytes()).hexdigest() != registration[key]:
            raise ValueError(f"registered {name} changed")
    log = (root / "campaign.log").read_text()
    events = re.findall(r"^(START|END) riley (\S+?)(?: exit=(\d+))?$", log, re.MULTILINE)
    if (len(events) != 2 or [(name, code) for name, _, code in events] != [("START", ""), ("END", "0")]
            or not registration["campaign_started_at"] <= events[0][1] <= events[-1][1]):
        raise ValueError("Riley lifecycle differs")
    _, cases = workload(root)
    if len(cases) != registration["heldout_pairs"] or read(root / "preflight.json")["input_sha256"] != registration["inputs_sha256"]:
        raise ValueError("held-out coverage differs")
    computed = report(root)
    if computed != read(root / "report.json"):
        raise ValueError("raw captures do not reproduce the arithmetic score")
    with tarfile.open(PRIOR / "raw-results.tar.gz", "r:gz") as archive:
        anchors = {gpu: json.load(archive.extractfile(f"gpu{gpu}-after-vllm.json")) for gpu in (6, 7)}
        receipt = json.load(archive.extractfile("riley/receipt.json"))
        source = archive.extractfile("arm.py").read().decode()
    expected = source.replace('str(STAGE / "arithmetic_capture.py")', 'str(STAGE / "structured_arithmetic.py")')
    if (root / "arm.py").read_text() != expected:
        raise ValueError("Riley runner changed outside the diagnostic client")
    actual_receipt = read(root / "riley/receipt.json")
    if (any(actual_receipt[key] != receipt[key] for key in receipt if key not in ("server", "capture_sha256"))
            or actual_receipt["capture_sha256"] != computed["capture_sha256"] or actual_receipt["shutdown_exit"] != 0
            or actual_receipt["source_revision"] != registration["source_revision"]):
        raise ValueError("Riley checkpoint, image, exit or capture differs")
    for gpu, anchor in anchors.items():
        for phase in ("before", "after"):
            row = read(root / f"gpu{gpu}-{phase}.json")
            if (row["physical_index"] != gpu or row["phase"] != "idle" or row["other_processes"]
                    or row["allocation_bytes"] != row["torch_reserved_bytes"] or row["allocation_bytes"] != 20 * 2**30
                    or any(row[key] != anchor[key] for key in ("pid", "start_ticks", "gpu"))):
                raise ValueError(f"GPU {gpu} reservation differs")
    return {**computed, "source_revision": actual_receipt["source_revision"], "image_id": actual_receipt["image_id"],
            "gpu": registration["gpu"], "preflight": read(root / "preflight.json")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", type=Path)
    print(json.dumps(verify(parser.parse_args().stage), indent=2, ensure_ascii=False, allow_nan=False))
