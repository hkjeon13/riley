"""Validate the repeated short cell, its raw timings, and both device-local comparisons."""

from pathlib import Path
from unittest.mock import patch
import json
import sys

import comparison_base
import verify_probe
from run_warm import GPUS
from verify_a100_candidate import digest, read, require
from workload import REQUEST_COUNTS, make_workloads

CASES = {"7": ("off-gpu7", "filtered-gpu7"), "6": ("filtered-gpu6", "off-gpu6")}
KEYS = ("ttft_p50_ms", "ttft_p95_ms", "tpot_p50_ms", "output_tok_s")


def checked_reservation(path, index):
    state = read(path)
    require(state["phase"] == "idle" and state["physical_index"] == int(index)
            and state["allocation_bytes"] == state["torch_reserved_bytes"] == 20 * 2**30
            and not state["other_processes"], "standing GPU reservation was not restored")
    return {index: {"pid": state["pid"], "start_ticks": state["start_ticks"], "gib": 20}}


def same_outputs(baseline, candidate):
    return sum(all(left[key] == right[key] for key in (*comparison_base.OUTPUT, "input_sha256"))
               for left, right in zip(baseline["rows"], candidate["rows"], strict=True))


def buckets(sample):
    return verify_probe.current_batch_shapes(sample)


def prior_192(evidence):
    earlier = evidence / "w3-prefix-min-gpu-20261121.8gBDZBBM"
    summary = {}
    for label in ("off-first", "filtered-first", "filtered-second", "off-second"):
        path = earlier / label
        receipt = read(path / "receipt.json")
        work = path / "workload.json"
        require(digest(work) == receipt["artifacts_sha256"]["workload.json"], "archived request evidence differs")
        raw = read(work)
        before, after = (buckets(raw[cell])[192] for cell in ("rag", "short"))
        summary[label] = {"previous_steps": before[0], "short_steps": after[0] - before[0],
                          "short_gpu_ns": after[1] - before[1], "workload_sha256": digest(work)}
    return summary


def verify(stage):
    evidence = stage.parent
    baseline = evidence / "w3-coder-rag-cache-20261121.kXvzFrJJ/off-first"
    build = evidence / "w3-prefix-min-build-20261121.vdX2vOQA"
    parent = evidence / "w5-current-package-20261121.f6SEbjvA/cpu-evidence/report.json"
    original = read(baseline / "workload.json")
    reference_hash = digest(baseline / "workload.json")
    require(reference_hash == "a77aa2092bd229a40038622d1a2de6768d2a8a6fdca159fbff42c20d9d6cfb7b"
            == read(baseline / "receipt.json")["artifacts_sha256"]["workload.json"], "stock outputs changed")
    chronology = [row.split() for row in (stage / "campaign.log").read_text().splitlines()
                  if row.startswith(("START ", "END "))]
    require(len(chronology) == 8 and all(row[-1] == "exit=0" for row in chronology if row[0] == "END"),
            "unexpected campaign exit")
    inputs = make_workloads()
    results, comparisons, reservations = {}, {}, {}
    for index, labels in CASES.items():
        events = [row for row in chronology if row[1].endswith(f"-gpu{index}")]
        require([row[:2] for row in events] == [[phase, name] for name in labels for phase in ("START", "END")]
                and [row[2] for row in events] == sorted(row[2] for row in events), "comparison sequence changed")
        expected_holder = checked_reservation(stage / f"gpu{index}-before.json", index)
        require(expected_holder == checked_reservation(stage / f"gpu{index}-after.json", index),
                "standing GPU holder changed across the campaign")
        reservations.update(expected_holder)
        for name in labels:
            def reservation(path, phase):
                return checked_reservation(path / f"reservation-{phase}.json", index)

            with (patch.object(verify_probe, "GPU", GPUS[index]),
                  patch.object(verify_probe, "check_reservations", reservation)):
                raw, arm = verify_probe.probe_arm(stage, name, build, inputs, parent)
            require(arm["reservation"] == expected_holder, "arm changed the GPU holder")
            require(raw.keys() == {"prime", "rag", "short", "warm_short"}, "unexpected measurements")
            arm["cells"]["warm_short"] = comparison_base.check_cell(raw["warm_short"], inputs["short"], "short")
            arm["warm_short_gpu_shapes"] = verify_probe.short_gpu_shapes({"rag": raw["short"], "short": raw["warm_short"]})
            arm["warm_short_shapes_unused_before"] = [
                int(dense) for dense in arm["warm_short_gpu_shapes"]["by_dense_rows"] if not buckets(raw["short"])[int(dense)][0]]
            arm["stock_matches"] = {name: same_outputs(original[name], raw[name]) for name in REQUEST_COUNTS}
            arm["stock_matches"]["warm_short"] = same_outputs(original["short"], raw["warm_short"])
            require(arm["stock_matches"] == {**REQUEST_COUNTS, "warm_short": 96}, "outputs differ from the stock image")
            if name.startswith("off-"):
                require(all(not cell["cached_requests"] for cell in arm["cells"].values()), "disabled cache returned hits")
            else:
                require(arm["cells"]["rag"]["cached_requests"] == 64 and arm["cells"]["rag"]["cached_min"] >= 1536
                        and all(arm["cells"][key]["cached_requests"] == 0 and arm["cells"][key]["prompt_tokens_max"] < 128
                                for key in ("short", "warm_short")), "filtered cache reuse changed")
            results[name] = arm

        old, new = (results[f"{mode}-gpu{index}"]["cells"] for mode in ("off", "filtered"))
        comparisons[index] = {name: {key: 100 * (new[name][key] / old[name][key] - 1) for key in KEYS}
                              for name in ("rag", "short", "warm_short")}

    def passes(cell):
        return (cell["ttft_p50_ms"] < 2 and cell["ttft_p95_ms"] < 2 and cell["tpot_p50_ms"] < 1
                and cell["output_tok_s"] > -2)

    return {"status": "passed", "source_revision": read(parent)["source_revision"], "image_id": verify_probe.STOCK_IMAGE,
            "binary_sha256": digest(build / "target/release/riley"), "model_sha256": verify_probe.MODEL_SHA,
            "remote_evidence": str(stage), "stock_workload_sha256": reference_hash, "prior_192_rows": prior_192(evidence),
            "order": chronology, "reservations": reservations, "runs": results, "change_percent": comparisons,
            "guards": {"short_both_devices": all(passes(row["short"]) for row in comparisons.values()),
                       "warm_short_both_devices": all(passes(row["warm_short"]) for row in comparisons.values()),
                       "rag_decode_tpot_below_1_percent": all(row["rag"]["tpot_p50_ms"] < 1 for row in comparisons.values()),
                       "rag_ttft_p95_improved_in_both_orders": all(row["rag"]["ttft_p95_ms"] < 0 for row in comparisons.values())}}


if __name__ == "__main__":
    print(json.dumps(verify(Path(sys.argv[1])), indent=2, allow_nan=False))
