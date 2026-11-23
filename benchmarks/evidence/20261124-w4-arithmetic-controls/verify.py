"""Verify the arithmetic controls and the unchanged engine configuration."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import sys
import tarfile


HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / "20261124-w4-chat-answers"
sys.path.insert(0, str(PRIOR))
from arithmetic_capture import report, workload
from long_answer_check import score


def read(path):
    return json.loads(path.read_bytes())


def compute_from_extraction(cases, engine_report):
    expected = next(case["answer"] for case in cases if case["control"] == "fact-only")
    totals = {mode: {"requests": 0, "parsed": 0, "correct": 0} for mode in ("c1", "c8")}
    for case in cases:
        if case["control"] != "extract-operands":
            continue
        for mode in totals:
            row = engine_report["cases"][case["label"]][mode]
            totals[mode]["requests"] += 1
            operands = re.fullmatch(r"\s*([0-9]{1,8})\s*,\s*([0-9]{1,8})\s*,\s*([0-9]{1,8})\s*", row["output"])
            if operands and row["finish"] == "stop":
                initial, per_tray, trays = map(int, operands.groups())
                totals[mode]["parsed"] += 1
                totals[mode]["correct"] += str(initial - per_tray * trays) == expected
    return totals


def verify(root):
    reg = read(HERE / "registration.json")
    if (root / "campaign.exit").read_text().strip() != "0":
        raise ValueError("arithmetic campaign did not complete")
    for name, key in (("inputs.json", "inputs_sha256"), ("dispatch.sha256", "dispatch_sha256")):
        if sha256((root / name).read_bytes()).hexdigest() != reg[key]:
            raise ValueError(f"registered input differs: {name}")
    log = (root / "campaign.log").read_text()
    events = re.findall(r"^(START|END) (riley|vllm) (\S+?)(?: exit=(\d+))?$", log, re.MULTILINE)
    if ([(event, engine, exit_code) for event, engine, _, exit_code in events]
            != [(event, engine, exit_code) for engine in ("riley", "vllm") for event, exit_code in (("START", ""), ("END", "0"))]
            or "W3_TERMINAL exit=0 " not in log):
        raise ValueError("run order or dependency differs")
    recomputed = report(root)
    if recomputed != read(root / "report.json"):
        raise ValueError("arithmetic control scores differ from the raw captures")
    cases = workload(root)
    with tarfile.open(PRIOR / "raw-results.tar.gz", "r:gz") as archive:
        anchors = {gpu: json.load(archive.extractfile(f"gpu{gpu}-after-vllm.json")) for gpu in (6, 7)}
        receipts = {engine: json.load(archive.extractfile(f"{engine}/receipt.json")) for engine in ("riley", "vllm")}
        source = archive.extractfile("arm.py").read().decode()
    expected = source.replace('str(STAGE / "chat_answers.py")', 'str(STAGE / "arithmetic_capture.py")')
    if (root / "arm.py").read_text() != expected:
        raise ValueError("server runner changed outside the client path")
    for gpu, anchor in anchors.items():
        for phase in ("before", "after-riley", "after-vllm"):
            holder = read(root / f"gpu{gpu}-{phase}.json")
            if (holder["physical_index"] != gpu or holder["phase"] != "idle" or holder["other_processes"]
                    or holder["allocation_bytes"] != holder["torch_reserved_bytes"] or holder["allocation_bytes"] != 20 * 2**30
                    or any(holder[key] != anchor[key] for key in ("pid", "start_ticks", "gpu"))):
                raise ValueError(f"GPU {gpu} reservation differs")
    for engine, keys in (("riley", ("image_id", "source_revision", "binary_sha256", "cpu_report_sha256")),
                         ("vllm", ("vllm", "launcher_sha256"))):
        receipt = read(root / engine / "receipt.json")
        if (any(receipt[key] != receipts[engine][key] for key in (*keys, "checkpoint_sha256", "fixture_sha256", "gpu", "holder_pid"))
                or receipt["capture_sha256"] != recomputed["engines"][engine]["capture_sha256"]
                or receipt["shutdown_exit"] not in ((0,) if engine == "riley" else (0, -15))):
            raise ValueError(f"engine identity or capture changed for {engine}")
        for mode in read(root / engine / "capture.json")["modes"].values():
            for case, row in zip(cases, mode, strict=True):
                score(case, row)
    computed = {engine: compute_from_extraction(cases, report) for engine, report in recomputed["engines"].items()}
    return {**recomputed, "computed_from_captured_operands": computed,
            "source_revision": receipts["riley"]["source_revision"],
            "riley_image_id": receipts["riley"]["image_id"], "vllm": receipts["vllm"]["vllm"], "gpu": reg["gpu"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", type=Path)
    print(json.dumps(verify(parser.parse_args().stage), ensure_ascii=False, indent=2))
