"""Verify repeated ordinary CUDA runs and report every valid cross-setting pair."""
import argparse
from hashlib import sha256
import json
from math import isclose, isfinite
from pathlib import Path
import re
import runpy
import tarfile
from tempfile import TemporaryDirectory


PARENT = "20261124-w3-c32-rag-history"
VLLM = "20261124-w3-separated-client/initial-results.tar.gz"
ORDERS = {"6": ("control", "reserve", "control"), "7": ("reserve", "control", "reserve")}
MODES = ("control", "reserve")


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def verify_metrics(path, workload, percentile):
    for name, (rows, reported) in workload.items():
        raw = read(path / f"{name}.json")
        wall = raw["statistics"]["wall_s"]
        epoch = raw["client_timing"]["epoch"]
        if not (isfinite(wall) and wall > 0 and isfinite(epoch)
                and all(epoch <= row["start"] and row["start"] - epoch + row["e2e"] <= wall + 1e-8 for row in rows)):
            raise ValueError(f"invalid measured interval in {path}/{name}")
        ttft = []
        tpot = []
        gaps = []
        output_tokens = 0
        for row in rows:
            completion = row["completion_tokens"]
            content_gaps = row["content_gaps_s"]
            if (type(completion) is not int or completion <= 0
                    or any(not isfinite(gap) or gap < 0 for gap in content_gaps)):
                raise ValueError(f"invalid completion or content gaps in {path}/{name}")
            output_tokens += completion
            ttft.append(row["ttft"] * 1000)
            tpot.append(sum(content_gaps) / max(completion - 1, 1) * 1000)
            gaps.extend(content_gaps)
        recomputed = {
            "ttft_p50_ms": percentile(ttft, .5),
            "ttft_p95_ms": percentile(ttft, .95),
            "tpot_p50_ms": percentile(tpot, .5),
            "itl_p99_ms": (percentile(gaps, .99) or 0) * 1000,
            "output_tok_s": output_tokens / wall,
        }
        if any(not isclose(value, reported[key], rel_tol=0, abs_tol=1e-8) for key, value in recomputed.items()):
            raise ValueError(f"reported performance metrics disagree with raw requests in {path}/{name}")


def client_phases(rows, percentile):
    before = [(row["sent_at"] - row["start"]) * 1000 for row in rows]
    after = [row["ttft"] * 1000 - elapsed for row, elapsed in zip(rows, before, strict=True)]
    if any(elapsed < 0 for elapsed in after):
        raise ValueError("first content preceded completion of the request write")
    return {
        f"{phase}_p{label}_ms": percentile(values, q)
        for phase, values in (("before_write", before), ("after_write", after))
        for label, q in ((50, .5), (95, .95))
    }


def compare(arms, order, reg, baseline):
    pairs = {}
    for label, indexes in (("first", (0, 1)), ("last", (1, 2)), ("repeat", (0, 2))):
        first, second = indexes
        if label != "repeat" and order[first] == "reserve":
            first, second = second, first
        cells = {}
        for name, measured in arms[first]["workload"].items():
            compared = arms[second]["workload"][name]
            current = baseline["compare_cells"](measured, compared, name, reg)
            if name.startswith("c32-"):
                current["client_phases"] = {
                    "first": client_phases(measured[0], baseline["percentile"]),
                    "second": client_phases(compared[0], baseline["percentile"]),
                }
            current["second"] = current.pop("fifo")
            current["first"] = current.pop("control")
            for old, new in (("fifo", "second"), ("control", "first")):
                current["abs_write_error_p95_ms"][new] = current["abs_write_error_p95_ms"].pop(old)
            if label == "repeat":
                current["gates"] = {key: current["gates"][key] for key in ("timing", "output")}
            cells[name] = current
        pairs[label] = {
            "first_pass": first + 1, "second_pass": second + 1, "first_mode": order[first], "second_mode": order[second],
            "server_age_matched": all(arms[i]["history"]["release_timed"] for i in (first, second)), "cells": cells,
        }
    return pairs


def compare_vllm(workload, labels, reference, reg, percentile, release_timed):
    results = {}
    identity = ("request_index", "input_sha256", "prompt_tokens", "scheduled_write_ns")
    for name in labels:
        rows, current = workload[name]
        old = json.load(reference.extractfile(f"gpu7/coder14-vllm/{name}.json"))
        stats = old["statistics"]
        prior = old["measured"]
        if (stats["ok"] != len(prior) or stats["requests"] != len(prior) or stats["errors"]
                or any(any(a[key] != b[key] for key in identity) for a, b in zip(rows, prior, strict=True))):
            raise ValueError(f"saved vLLM input, schedule or status differs in {name}")
        ttft_p95 = percentile([row["ttft"] * 1000 for row in prior], .95)
        if ttft_p95 != stats["ttft_p95_ms"]:
            raise ValueError(f"saved vLLM latency differs from raw requests in {name}")
        errors = {
            "riley": percentile([abs(row["write_error_ms"]) for row in rows], .95),
            "vllm": percentile([abs(row["write_error_ms"]) for row in prior], .95),
            "between": percentile([
                abs(a["write_error_ms"] - b["write_error_ms"]) for a, b in zip(rows, prior, strict=True)
            ], .95),
        }
        eligible = release_timed and max(errors.values()) <= reg["write_error_p95_limit_ms"]
        ratio = current["ttft_p95_ms"] / ttft_p95 if eligible else None
        results[name] = {
            "abs_write_error_p95_ms": errors, "ttft_p95_ratio": ratio,
            "phase_1_target_at_most_1_4": ratio <= 1.4 if eligible else None,
            "final_target_at_most_1_15": ratio <= 1.15 if eligible else None,
        }
    return results


def verify(root, archive, registration, evidence):
    reg = read(registration)
    parent = evidence / PARENT
    prior = read(parent / "registration.json")
    history = runpy.run_path(str(parent / "verify.py"))
    native_dir = evidence / history["NATIVE"]
    native = runpy.run_path(str(native_dir / "verify.py"))
    baseline = runpy.run_path(str(evidence / history["BASELINE"] / "verify.py"))
    binary = (native_dir / "build-result.sha256").read_text().split()[0]
    if (reg["parent_archive_sha256"] != digest(parent / "raw-results.tar.gz")
            or reg["diagnostic_binary_sha256"] != binary or
            any(reg[name] != prior[name] for name in (
                "source_revision", "package_image_id", "package_cpu_report_sha256", "mode_flag",
                "schedules_sha256", "holdout_requests_per_arm", "write_error_p95_limit_ms",
                "decode_tpot_p50_max_regression_pct_exclusive", "short_ttft_p50_max_regression_pct_exclusive",
                "short_throughput_max_drop_pct_exclusive", "short_start_target_s", "rag_start_target_s",
                "history_release_tolerance_s",
            ))):
        raise ValueError("native binary, schedule or gates differ from their recorded experiment")
    history["check_archive"](archive, root, reg)
    if ((root / "campaign.exit").read_text().strip() != "0"
            or int((root / "campaign.pid").read_text()) != reg["campaign_pid"]
            or (root / "campaign.started").read_text().strip() != reg["campaign_started_at"]
            or (root / "native-build-result.sha256").read_bytes() != (native_dir / "build-result.sha256").read_bytes()):
        raise ValueError("campaign identity or archived ordinary binary differs")
    observed = re.findall(r"^(START|END) gpu([67]) ([123]) (control|reserve) \S+(?: exit=(\d+))?$",
                          (root / "campaign.log").read_text(), re.MULTILINE)
    results = {}
    with (tarfile.open(parent / "raw-results.tar.gz", "r:gz") as gold,
          tarfile.open(evidence / VLLM, "r:gz") as vllm):
        for filename in ("client.py", "holdout.py", "replay.py", "schedules.json", "cpu-placement.json"):
            if (root / filename).read_bytes() != gold.extractfile(filename).read():
                raise ValueError(f"frozen runner differs at {filename}")
        for gpu, order in ORDERS.items():
            expected = [(kind, str(index), mode, code) for index, mode in enumerate(order, 1)
                        for kind, code in (("START", ""), ("END", "0"))]
            if [(kind, index, mode, code) for kind, physical, index, mode, code in observed if physical == gpu] != expected:
                raise ValueError(f"invalid run order or exit on GPU {gpu}")
            old_holder = json.load(gold.extractfile(f"gpu{gpu}/after.json"))
            for phase in ("before", "between-1", "between-2", "after"):
                state = read(root / f"gpu{gpu}/{phase}.json")
                if (state["phase"] != "idle" or state["other_processes"]
                        or state["allocation_bytes"] != state["torch_reserved_bytes"] or state["allocation_bytes"] != 20 * 2**30
                        or any(state[k] != old_holder[k] for k in ("pid", "gpu", "physical_index", "start_ticks"))):
                    raise ValueError(f"original reservation did not recover on GPU {gpu}")
            def previous(name):
                return json.load(gold.extractfile(f"gpu{gpu}/rag/control/{name}.json"))
            expected_plan, expected_receipt, expected_tokens = [
                previous(name) for name in ("plan", "receipt", "token-id-holdout")
            ]
            original = {
                name: [(row["request_index"], row["output_sha256"]) for row in previous(name)["measured"]]
                for name in expected_plan["cells"]
            }
            arms = []
            for index, mode in enumerate(order, 1):
                path = root / f"gpu{gpu}/rag/{index}/{mode}"
                workload, tokens, shapes = native["verify_arm"](
                    path, expected_plan, expected_receipt, reg, binary, mode, int(gpu) + 38, baseline,
                )
                verify_metrics(path, workload, baseline["percentile"])
                if (tokens != expected_tokens or tokens["c1"] != tokens["c8"] or any(
                    [(row["request_index"], row["output_sha256"]) for row in rows] != original[name]
                    for name, (rows, _) in workload.items()
                )):
                    raise ValueError(f"ordinary replay output changed in {path}")
                log = (path / "server.log").read_text()
                if any(line.startswith(("W3-OBS ", "W3-ADMISSION ", "W3-EXECUTION ")) for line in log.splitlines()):
                    raise ValueError("an ordinary runtime unexpectedly exported diagnostics")
                arms.append({
                    "workload": workload, "history": history["check_history"](
                        path / "history.json", tuple(expected_plan["cells"]), reg,
                    ), "batch_shapes": shapes,
                })
            results[gpu] = {
                "order": order, "comparisons": compare(arms, order, reg, baseline),
                "arms": [{k: v for k, v in arm.items() if k != "workload"} for arm in arms],
            }
            if gpu == "7":
                results[gpu]["frozen_vllm"] = [compare_vllm(
                    arm["workload"], history["RAG"], vllm, reg, baseline["percentile"],
                    arm["history"]["release_timed"],
                ) for arm in arms]
    return {
        "binary_sha256": binary,
        "frozen_vllm_archive_sha256": digest(evidence / VLLM),
        "gpus": results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("registration", type=Path)
    parser.add_argument("--evidence-root", type=Path, default=Path(__file__).resolve().parent.parent)
    args = parser.parse_args()
    with TemporaryDirectory() as destination:
        print(json.dumps(verify(Path(destination), args.archive, args.registration, args.evidence_root),
                         indent=2, allow_nan=False))
