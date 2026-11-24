"""Verify the current distribution's short-prefill continuation replay."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import runpy
import sys
import tarfile
from tempfile import TemporaryDirectory


TAIL = "20261124-w3-c32-prefill-tail"
NATIVE = "20261124-w3-c32-native-repeat"
STEP = "20261124-w3-c32-step-observation"
HISTORY = "20261124-w3-c32-rag-history"
PACKAGE = "20261124-w4-pattern-anchors/a100/cpu-evidence/report.json"
ORDERS = {"6": ["control", "candidate", "control"], "7": ["candidate", "control", "candidate"]}
FLAG = "RILEY_W3_SHORT_PREFILL_CONTINUATION"


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def check_arm(path, old_plan, old_receipt, reg, binary, mode, cpu, baseline, native, repeat):
    plan, receipt = (read(path / f"{name}.json") for name in ("plan", "receipt"))
    if any(plan[key] != old_plan[key] for key in ("server", "cells", "server_cpu_affinity")):
        raise ValueError("server profile or workload differs from the frozen arms")
    args = plan["docker"]
    mount = f"type=bind,source={reg['stage']}/riley-diagnostic,target=/opt/riley/bin/riley,readonly"
    candidate_env = f"{FLAG}={int(mode == 'candidate')}"
    if (args[-(len(plan["server"]) + 1):] != [reg["package_image_id"], *plan["server"]]
            or [arg for arg in args if arg.startswith("RILEY_W3_")] != [candidate_env]
            or args.count(mount) != 1 or "127.0.0.1::8080" not in args):
        raise ValueError("execution environment or packaged binary differs")
    changed = {
        "riley_binary_sha256": binary, "source_revision": reg["source_revision"],
        "image_id": reg["package_image_id"], "cpu_report_sha256": reg["package_cpu_report_sha256"],
    }
    fixed = set(old_receipt) - set(changed) - {"holdout_sha256", "results"}
    if (set(receipt) != set(old_receipt) or any(receipt[key] != value for key, value in changed.items())
            or any(receipt[key] != old_receipt[key] for key in fixed)
            or set(receipt["results"]) != set(plan["cells"])):
        raise ValueError("image, process or checkpoint receipt differs")
    workload = {
        label: baseline["check_cell"](path / f"{label}.json", receipt, cell, native["MATCH_KEYS"], cpu)
        for label, cell in plan["cells"].items()
    }
    repeat["verify_metrics"](path, workload, baseline["percentile"])
    tokens_path = path / "token-id-holdout.json"
    tokens = read(tokens_path)
    if (digest(tokens_path) != receipt["holdout_sha256"] or set(tokens) != {"c1", "c8"}
            or any(len(rows) * 2 != reg["holdout_requests_per_arm"] for rows in tokens.values())):
        raise ValueError("token-ID holdout is incomplete")
    if any(line.startswith(("W3-OBS ", "W3-ADMISSION ", "W3-EXECUTION ", "W3-PREFILL "))
           for line in (path / "server.log").read_text().splitlines()):
        raise ValueError("expected ordinary serving with request tracing off")
    return workload, tokens, baseline["batch_summary"](path / "shutdown.json")


def verify(archive, registration, evidence, scratch):
    reg = read(registration)
    package = read(evidence / PACKAGE)
    prior = read(evidence / NATIVE / "registration.json")
    if (reg["orders"] != ORDERS or reg["mode_env"] != FLAG or package["status"] != "passed"
            or reg["source_revision"] != package["source_revision"] or reg["package_image_id"] != package["image_id"]
            or reg["package_cpu_report_sha256"] != digest(evidence / PACKAGE)
            or reg["previous_diagnostic_binary"] != (evidence / TAIL / "build-result.sha256").read_text().split()[0]
            or reg["native_repeat_archive_sha256"] != digest(evidence / NATIVE / "raw-results.tar.gz")
            or any(reg[key] != prior[key] for key in (
                "schedules_sha256", "holdout_requests_per_arm", "write_error_p95_limit_ms",
                "decode_tpot_p50_max_regression_pct_exclusive", "short_ttft_p50_max_regression_pct_exclusive",
                "short_throughput_max_drop_pct_exclusive", "short_start_target_s", "rag_start_target_s",
                "history_release_tolerance_s",
            ))):
        raise ValueError("current package, frozen schedule or gates differ")
    sys.path.insert(0, str(evidence / STEP))
    parent = runpy.run_path(str(evidence / STEP / "verify.py"))
    history = runpy.run_path(str(evidence / HISTORY / "verify.py"))
    native = runpy.run_path(str(evidence / history["NATIVE"] / "verify.py"))
    baseline = runpy.run_path(str(evidence / history["BASELINE"] / "verify.py"))
    repeat = runpy.run_path(str(evidence / NATIVE / "verify.py"))
    binary = parent["check_archive"](archive, scratch, reg, require_runtime_binary=True)
    if digest(scratch / "candidate.patch") != reg["patch_sha256"]:
        raise ValueError("scheduler candidate patch differs from the launch")
    events = re.findall(r"^(START|END) gpu([67]) ([123]) (control|candidate) \S+(?: exit=(\d+))?$",
                        (scratch / "campaign.log").read_text(), re.MULTILINE)
    results = {}
    with (tarfile.open(evidence / HISTORY / "raw-results.tar.gz") as gold,
          tarfile.open(evidence / repeat["VLLM"]) as vllm):
        for filename in ("client.py", "holdout.py", "replay.py", "schedules.json", "cpu-placement.json"):
            if (scratch / filename).read_bytes() != gold.extractfile(filename).read():
                raise ValueError(f"frozen runner differs at {filename}")
        for gpu, order in ORDERS.items():
            expected = [(kind, str(index), mode, code) for index, mode in enumerate(order, 1)
                        for kind, code in (("START", ""), ("END", "0"))]
            if [(k, i, m, c) for k, device, i, m, c in events if device == gpu] != expected:
                raise ValueError(f"run order or terminal code differs on GPU {gpu}")
            old_holder = json.load(gold.extractfile(f"gpu{gpu}/after.json"))
            states = [scratch / f"gpu{gpu}-staged-start.json", *(
                scratch / f"gpu{gpu}/{phase}.json" for phase in ("before", "between-1", "between-2", "after")
            )]
            for path in states:
                holder = read(path)
                if (holder["phase"] != "idle" or holder["other_processes"]
                        or holder["allocation_bytes"] != 20 * 2**30 or holder["torch_reserved_bytes"] != 20 * 2**30
                        or any(holder[k] != old_holder[k] for k in ("physical_index", "gpu", "pid", "start_ticks"))):
                    raise ValueError(f"original reservation did not recover on GPU {gpu}")
            def previous(name):
                return json.load(gold.extractfile(f"gpu{gpu}/rag/control/{name}.json"))
            plan, receipt, tokens = [previous(name) for name in ("plan", "receipt", "token-id-holdout")]
            old_rows = {label: previous(label)["measured"] for label in plan["cells"]}
            arms = []
            for index, mode in enumerate(order, 1):
                path = scratch / f"gpu{gpu}/rag/{index}/{mode}"
                measured, current_tokens, batches = check_arm(
                    path, plan, receipt, reg, binary, mode, int(gpu) + 38, baseline, native, repeat,
                )
                unchanged = {
                    label: [b["request_index"] for a, b in zip(old_rows[label], rows, strict=True)
                            if a["output_sha256"] != b["output_sha256"]]
                    for label, (rows, _) in measured.items()
                }
                timing = history["check_history"](path / "history.json", tuple(plan["cells"]), reg)
                arms.append({
                    "mode": mode, "workload": measured, "history": timing, "batch_shapes": batches,
                    "changed_outputs_vs_reference": unchanged,
                    "token_ids_match_reference": current_tokens == tokens,
                    "token_ids_batch_invariant": current_tokens["c1"] == current_tokens["c8"],
                })
            alias = tuple("reserve" if mode == "candidate" else mode for mode in order)
            comparisons = repeat["compare"](arms, alias, reg, baseline)
            for pair in comparisons.values():
                for key in ("first_mode", "second_mode"):
                    if pair[key] == "reserve":
                        pair[key] = "candidate"
            result = {
                "order": order, "comparisons": comparisons,
                "arms": [{k: v for k, v in arm.items() if k != "workload"} for arm in arms],
            }
            if gpu == "7":
                result["frozen_vllm"] = [repeat["compare_vllm"](
                    arm["workload"], history["RAG"], vllm, reg, baseline["percentile"], arm["history"]["release_timed"],
                ) for arm in arms]
            results[gpu] = result
    return {"binary_sha256": binary, "source_revision": reg["source_revision"], "gpus": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("registration", type=Path)
    parser.add_argument("--evidence-root", type=Path, default=Path(__file__).resolve().parent.parent)
    args = parser.parse_args()
    with TemporaryDirectory() as scratch:
        print(json.dumps(verify(args.archive, args.registration, args.evidence_root, Path(scratch)),
                         indent=2, allow_nan=False))
