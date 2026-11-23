"""Verify the paired chat captures against the registered inputs and server identities."""

import argparse
import gzip
import json
from pathlib import Path
import re
import tarfile

from chat_answers import CONFIG_SHA, FIXTURE_SHA, VARIANTS, digest, report
from long_answer_check import same_completion


HERE = Path(__file__).resolve().parent
BASELINE = HERE.parent / "20261124-w4-long-answers"


def read(path):
    return json.loads(path.read_bytes())


def compare_engines(root):
    captures = [read(root / engine / "capture.json")["variants"] for engine in ("riley", "vllm")]
    results = {}
    for variant in VARIANTS:
        results[variant] = {}
        for mode in ("c1", "c8"):
            riley, vllm = (capture[variant][mode] for capture in captures)
            pairs = list(zip(riley, vllm, strict=True))
            usage = ("prompt_tokens", "completion_tokens", "total_tokens")
            results[variant][mode] = {
                "same_text": sum(left["text"] == right["text"] for left, right in pairs),
                "same_completion_and_usage_counts": sum(same_completion(left, right) and
                                                        all(left["usage"][k] == right["usage"][k] for k in usage)
                                                        for left, right in pairs),
                "arithmetic_integer_outputs": {engine: sum(row["text"].strip().isdecimal()
                                                           for row in rows if row["label"].startswith("arithmetic-ko-"))
                                               for engine, rows in (("riley", riley), ("vllm", vllm))},
            }
    return results


def verify(root):
    registration = read(HERE / "registration.json")
    if ((root / "campaign.exit").read_text().strip() != "0"
            or digest((root / "dispatch.sha256").read_bytes()) != registration["dispatch_sha256"]
            or digest((root / "runtime.sha256").read_bytes()) != registration["runtime_sha256"]):
        raise ValueError("campaign is incomplete or differs from the registration")
    events = re.findall(r"^(START|END) (riley|vllm) (\S+?)(?: exit=(\d+))?$",
                        (root / "campaign.log").read_text(), re.MULTILINE)
    expected = [("START", "riley", ""), ("END", "riley", "0"), ("START", "vllm", ""), ("END", "vllm", "0")]
    if [(phase, engine, code) for phase, engine, _, code in events] != expected or events[0][2] < registration["started_at"]:
        raise ValueError("engine order differs from the registration")
    source = (BASELINE / "arm.py").read_text()
    if (root / "arm.py").read_text() != source.replace('str(STAGE / "long_answer_check.py")', 'str(STAGE / "chat_answers.py")'):
        raise ValueError("server runner differs outside the benchmark client entry point")
    if (root / "facts.json.gz").read_bytes() != (BASELINE / "facts.json.gz").read_bytes():
        raise ValueError("source facts differ from the raw-completion comparison")
    recomputed = report(root)
    if recomputed != read(root / "report.json"):
        raise ValueError("captured scores disagree with the independent score calculation")
    preflight = read(root / "preflight.json")
    inputs = json.loads(gzip.decompress((root / "inputs.json.gz").read_bytes()))
    if (recomputed["inputs_sha256"] != registration["inputs_sha256"]
            or preflight["inputs_sha256"] != registration["inputs_sha256"]
            or preflight["tokenizer_config_sha256"] != CONFIG_SHA
            or preflight["chat_template_sha256"] != inputs["chat_template_sha256"]
            or inputs["chat_template_sha256"] != registration["chat_template_sha256"]):
        raise ValueError("the rendered input binding differs")
    with tarfile.open(BASELINE / "retry/raw-results.tar.gz", "r:gz") as archive:
        anchors = {index: json.load(archive.extractfile(f"gpu{index}-after.json")) for index in (6, 7)}
        baseline_receipts = {engine: json.load(archive.extractfile(f"{engine}/receipt.json")) for engine in ("riley", "vllm")}
    for index, anchor in anchors.items():
        for phase in ("before", "after-riley", "after-vllm"):
            row = read(root / f"gpu{index}-{phase}.json")
            if (row["physical_index"] != index or row["phase"] != "idle" or row["other_processes"]
                    or row["allocation_bytes"] != 20 * 2**30 or row["torch_reserved_bytes"] != row["allocation_bytes"]
                    or any(row[key] != anchor[key] for key in ("gpu", "pid", "start_ticks"))):
                raise ValueError(f"GPU {index} holder changed or did not recover after {phase}")
    for engine, keys in (("riley", ("image_id", "source_revision", "binary_sha256", "cpu_report_sha256")),
                         ("vllm", ("vllm", "launcher_sha256"))):
        receipt = read(root / engine / "receipt.json")
        if (receipt["fixture_sha256"] != FIXTURE_SHA or receipt["gpu"] != anchors[7]["gpu"]
                or receipt["holder_pid"] != anchors[7]["pid"] or receipt["capture_sha256"] != recomputed["engines"][engine]["capture_sha256"]
                or receipt["shutdown_exit"] not in ((0,) if engine == "riley" else (0, -15))
                or any(receipt[key] != baseline_receipts[engine][key] for key in (*keys, "checkpoint_sha256"))):
            raise ValueError(f"{engine} did not match the raw-completion engine, model or run")
    return {**recomputed, "paired_engines": compare_engines(root),
            "source_revision": baseline_receipts["riley"]["source_revision"],
            "riley_image_id": baseline_receipts["riley"]["image_id"], "vllm": baseline_receipts["vllm"]["vllm"],
            "gpu": anchors[7]["gpu"], "preflight": preflight}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", type=Path)
    print(json.dumps(verify(parser.parse_args().stage), indent=2, allow_nan=False))
