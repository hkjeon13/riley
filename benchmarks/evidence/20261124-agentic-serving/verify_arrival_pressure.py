"""Check the live-cohort crossover against original protected cells and vLLM arrivals."""

import argparse
import hashlib
import json
import re
import runpy
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / "20261124-w3-prefill-cohort"
REFERENCE = HERE.parent / "20261124-w3-separated-client/initial-results.tar.gz"


def read(path):
    return json.loads(path.read_text())


def inspect(root):
    if (root / "exit-status").read_text().strip() != "0" or not (
        root / "released"
    ).exists():
        raise ValueError("incomplete experiment")
    for file in ("arrival-pressure.patch", "arrival-pressure-registration.json"):
        if (root / file).read_bytes() != (HERE / file).read_bytes():
            raise ValueError("registered inputs have changed")
    sched = (root / "binaries.sha256").read_text().splitlines()
    binary = sched[0].split()[0]
    if (
        sched[1].split()[0]
        != "9bd616c000033f95cf4e644c04e8898ff8778104b449085429c9db57edd0e2d5"
    ):
        raise ValueError("scheduler differs from independently patched source")
    tests = (root / "cpu-tests.log").read_text()
    passed = re.findall(
        r"^test result: ok\. (\d+) passed; 0 failed;", tests, re.MULTILINE
    )
    if sum(map(int, passed)) != 148 or "test result: FAILED" in tests:
        raise ValueError("incomplete CPU scheduler tests")
    if (
        hashlib.sha256((root / "schedules.json").read_bytes()).hexdigest()
        != read(PRIOR / "registration.json")["schedules_sha256"]
    ):
        raise ValueError("arrival calendar differs from protected reference")
    native = runpy.run_path(
        str(HERE.parent / "20261124-w3-aged-decode-native/verify.py")
    )
    base = runpy.run_path(str(HERE.parent / "20261124-w3-rag-prefill-order/verify.py"))
    reg = read(PRIOR / "registration.json")
    reg.update(stage=(root / "stage-path.txt").read_text().strip(), mode_flag="")
    outcomes = {}
    with (
        tarfile.open(PRIOR / "raw-results.tar.gz") as previous,
        tarfile.open(REFERENCE) as vllm,
    ):

        def get(gpu, filename):
            return json.load(previous.extractfile(f"gpu{gpu}/control/{filename}.json"))

        for gpu in (6, 7):
            modes = {}
            holds = {}
            for mode in ("control", "candidate"):
                d = root / f"gpu{gpu}" / mode
                plan = read(d / "plan.json")
                switch = (
                    f"RILEY_EXPERIMENT_W3_ADMISSION_FIFO={int(mode == 'candidate')}"
                )
                if plan["docker"].count(switch) != 1:
                    raise ValueError("incorrect scheduling policy")
                state = read(d / "lease-after.json")
                if (
                    state["phase"] != "idle"
                    or state["physical_index"] != gpu
                    or state["other_processes"]
                ):
                    raise ValueError("GPU not released")
                cap, tokens, batch = native["verify_arm"](
                    d,
                    get(gpu, "plan"),
                    get(gpu, "receipt"),
                    reg,
                    binary,
                    "control",
                    gpu + 38,
                    base,
                )
                if (
                    tokens != get(gpu, "token-id-holdout")
                    or tokens["c1"] != tokens["c8"]
                ):
                    raise ValueError("held-out output tokens changed")
                modes[mode] = cap
                holds[mode] = batch
            order = ("control", "candidate") if gpu == 6 else ("candidate", "control")
            first_rows, second_rows = (
                [r for rows, _stats in modes[mode].values() for r in rows]
                for mode in order
            )
            if max(r["start"] + r["e2e"] for r in first_rows) >= min(
                r["start"] for r in second_rows
            ):
                raise ValueError("observed crossover order differs from registration")
            cells = {}
            for name, left in modes["control"].items():
                right = modes["candidate"][name]
                out = base["compare_cells"](left, right, name, reg)
                out["candidate"] = out.pop("fifo")
                errors = out["abs_write_error_p95_ms"]
                errors["candidate"] = errors.pop("fifo")
                past = get(gpu, name)["measured"]
                for arm in (left, right):
                    if any(
                        r["output_sha256"] != b["output_sha256"]
                        for r, b in zip(arm[0], past, strict=True)
                    ):
                        raise ValueError(
                            f"output changed from original reference: {gpu}/{name}"
                        )
                p = base["percentile"]
                out["last_p95_ms"] = {
                    m: 1000 * p([r["e2e"] for r in a[0]], 0.95)
                    for m, a in zip(
                        ("control", "candidate"), (left, right), strict=True
                    )
                }
                if gpu == 7 and name.startswith("rag-"):
                    frozen = json.load(
                        vllm.extractfile(f"gpu7/coder14-vllm/{name}.json")
                    )
                    pairs = list(zip(right[0], frozen["measured"], strict=True))
                    if any(
                        any(
                            a[k] != b[k]
                            for k in (
                                "input_sha256",
                                "request_index",
                                "scheduled_write_ns",
                                "prompt_tokens",
                            )
                        )
                        for a, b in pairs
                    ):
                        raise ValueError("vLLM comparison schedule differs")
                    err = p(
                        [
                            abs(a["write_error_ms"] - b["write_error_ms"])
                            for a, b in pairs
                        ],
                        0.95,
                    )
                    ratio = (
                        out["candidate"]["ttft_p95_ms"]
                        / frozen["statistics"]["ttft_p95_ms"]
                    )
                    control_error = p(
                        [
                            abs(a["write_error_ms"] - b["write_error_ms"])
                            for a, b in zip(left[0], frozen["measured"], strict=True)
                        ],
                        0.95,
                    )
                    out["vllm"] = {
                        "ttft_p95_ms": frozen["statistics"]["ttft_p95_ms"],
                        "control_ratio": out["control"]["ttft_p95_ms"]
                        / frozen["statistics"]["ttft_p95_ms"],
                        "control_paired_write_error_ms": control_error,
                        "candidate_ratio": ratio,
                        "paired_write_error_ms": err,
                        "pass": ratio <= 1.15 and err <= 3 and out["gates"]["timing"],
                    }
                cells[name] = out
            outcomes[str(gpu)] = {
                "batches": holds,
                "order": order,
                "heldout_ids_equal": True,
                "cells": cells,
            }
    cells = [cell for g in outcomes.values() for cell in g["cells"].values()]
    return {
        "binary": binary,
        "scheduler_tests_passed": sum(map(int, passed)),
        "gpus": outcomes,
        "measured_requests": 2 * sum(c["requests"] for c in cells),
        "all_guards_pass": all(
            all(c["gates"].values()) and c.get("vllm", {}).get("pass", True)
            for c in cells
        ),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    print(json.dumps(inspect(p.parse_args().root), indent=2))
