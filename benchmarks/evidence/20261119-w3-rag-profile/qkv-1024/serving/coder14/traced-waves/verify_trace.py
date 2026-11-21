"""Match archived Coder waves to the trace and bound overlap with prefill."""

from argparse import ArgumentParser
import json
from math import isclose
from pathlib import Path
from statistics import median

from analyze_coder_wave_decode import FIELDS, REFERENCE, decode_pair, send_span
from compare_coder import ORDER, EXPECTED_CELLS
from compare_profile_arms import arm, digest, ratio_percent, read, require
from summarize_profile_arm import stable_server
from summarize_request_stages import overlap, prefill_union, read_phases, request_phases


CELLS = ("wave-sync", "wave-50ms")
BINARY = "917caae4ff7467c58650ff5cf67004c17893aa862fd38371d2a940952c2ca398"
MODEL = "Qwen2.5-Coder-14B-Instruct"


def client_bounds(row, phases):
    front, = phases["front_submit"]
    first, = phases["first_delta"]
    start_ns = round(row["start"] * 1e9)
    first_ns = round((row["start"] + row["ttft"]) * 1e9)
    return start_ns - front[0], first_ns - first[0]


def window_overlap(row, spans, clock):
    low, high = clock
    first_ns = round((row["start"] + row["ttft"]) * 1e9)
    last_ns = round((row["start"] + row["e2e"]) * 1e9)
    require(first_ns + high - low < last_ns, "empty observed decode window")
    return [overlap(first_ns - low, last_ns - high, spans) / 1e6,
            overlap(first_ns - high, last_ns - low, spans) / 1e6]


def checked_arm(root, archived, mode):
    control, original, originals = arm(archived, "coder14", mode, expected_cells=EXPECTED_CELLS)
    receipt, plan, workloads = arm(root, "coder14", mode, expected_cells=CELLS)
    folder = root / f"coder14-{mode}"
    active = mode.startswith("qkv-")
    env = dict(original["env"], RILEY_W3_REQUEST_TRACE="1")
    require(receipt["model"] == control["model"] == MODEL and receipt["common_kv_blocks"] == 2576
            and all(receipt[k] == control[k] for k in ("gpu", "kv_cache_bytes"))
            and plan["env"] == env and plan["cells"] == {name: original["cells"][name] for name in CELLS}
            and stable_server(plan["server"])[1:] == stable_server(original["server"])[1:],
            f"{mode}: traced inputs differ from archived serving arm")
    old = {Path(k).name: v for k, v in control["sha256"].items()}
    new = {Path(k).name: v for k, v in receipt["sha256"].items()}
    for name in ("config.json", "riley-checkpoint.json", "tokenizer.json", "load.py"):
        require(old[name] == new[name], f"{mode}: {name} differs")
    require(new["riley"] == BINARY and new["plan.json"] == digest(folder / "plan.json")
            and read(folder / "server-exit.json")["exit"] == 0, "trace binary, plan or shutdown differs")
    markers = [line for line in (folder / "server.log").read_text().splitlines()
               if line.startswith("RILEY_W3_QKV1024")]
    require(bool(markers) == active and set(markers) <= {"RILEY_W3_QKV1024 recorded=1 cuda=120405"},
            f"{mode}: exact-row candidate selection differs")
    phases = read_phases(folder / "server.log")
    current = [row for _, rows in workloads.values() for row in rows]
    require(len({row["request_id"] for row in current}) == len(current) == 80,
            f"{mode}: request trace identities differ")
    bounds = [client_bounds(row, phases[row["request_id"]]) for row in current]
    clock = max(a for a, _ in bounds) - 2, min(b for _, b in bounds) + 2
    require(0 <= clock[1] - clock[0] <= 2_000_000, f"{mode}: cross-process clock interval too wide: {clock}")
    spans = prefill_union(phases)

    cells = {}
    for name, (stats, rows) in workloads.items():
        old_stats, expected = originals[name]
        require(all(all(row[field] == base[field] for field in FIELDS)
                    for row, base in zip(rows, expected, strict=True)), f"{mode}/{name}: traced outputs differ")
        require(isclose(decode_pair(rows, rows)["p50_ms"]["control"], stats["tpot_p50_ms"], rel_tol=1e-12),
                "trace TPOT aggregate differs from raw intervals")
        per_request = {}
        for row in rows:
            traced = request_phases(phases[row["request_id"]], row["prompt_tokens"], spans)
            per_request[row["request_index"]] = {
                "overlap_ms": window_overlap(row, spans, clock),
                "prefill_between_steps_ms": traced["stages_ms"]["prefill_between_steps"],
            }
        cells[name] = {"tpot_p50_ms": stats["tpot_p50_ms"], "ttft_p95_ms": stats["ttft_p95_ms"],
                       "tpot_p50_change_vs_archived_percent":
                           ratio_percent(stats["tpot_p50_ms"], old_stats["tpot_p50_ms"]),
                       "trace_source_sha256": receipt["results"][name],
                       "archived_source_sha256": control["results"][name],
                       "per_request": per_request, "rows": rows}
    return cells, {"server_log_sha256": digest(folder / "server.log"), "clock_interval_ns": clock[1] - clock[0]}


def summarize_pair(old, new):
    left, right = old["rows"], new["rows"]
    first, last = (left[8]["request_index"], left[15]["request_index"])
    bounds = [(old["per_request"][index]["overlap_ms"], new["per_request"][index]["overlap_ms"])
              for index in range(first, last + 1)]
    second = decode_pair(left[8:16], right[8:16])
    second["paired_overlap_change_median_ms_interval"] = [
        median(after[0] - before[1] for before, after in bounds),
        median(after[1] - before[0] for before, after in bounds),
    ]
    second["socket_write_span_ms"] = [send_span(left[8:16]), send_span(right[8:16])]
    focal = {label: cell["per_request"][41] for label, cell in (("control", old), ("candidate", new))}
    focal["ttft_change_ms"] = 1000 * (right[33]["ttft"] - left[33]["ttft"])
    return {"matched_responses": len(left), "overall_decode": decode_pair(left, right),
            "second_wave_decode": second, "request_41": focal,
            "ttft_p95_change_percent": ratio_percent(new["ttft_p95_ms"], old["ttft_p95_ms"])}


def verify(root, archived):
    require(digest(archived / "verified.json") == REFERENCE and (root / "campaign.exit").read_text().strip() == "0",
            "trace campaign did not complete or historical source differs")
    chronology = [line.split() for line in (root / "campaign.log").read_text().splitlines()
                  if line.startswith(("START ", "END "))]
    require([tuple(row[:2]) for row in chronology] == [(event, mode) for mode in ORDER for event in ("START", "END")]
            and all(row[3:] == ["exit=0"] for row in chronology[1::2])
            and [row[2] for row in chronology] == sorted(row[2] for row in chronology), "campaign arms overlap or failed")
    runs = {mode: checked_arm(root, archived, mode) for mode in ORDER}
    arms = {}
    for mode, (cells, run) in runs.items():
        reservations = {}
        for index in (6, 7):
            a, b = (read(root / f"gpu{index}-{phase}-{mode}.json") for phase in ("before", "after"))
            require(all(a[k] == b[k] for k in ("pid", "start_ticks", "gpu")) and
                    all(row["physical_index"] == index and row["phase"] == "idle" and not row["other_processes"]
                        and row["allocation_bytes"] == row["torch_reserved_bytes"] == 20 * 2**30 for row in (a, b)),
                    "reservation changed")
            reservations[str(index)] = {"pid": b["pid"], "start_ticks": b["start_ticks"], "gib": 20}
        arms[mode] = {**run, "reservations": reservations,
                      "cells": {cell: {k: v for k, v in item.items() if k not in ("rows", "per_request")}
                                for cell, item in cells.items()}}
    require(all(arm["reservations"] == arms[ORDER[0]]["reservations"] for arm in arms.values()),
            "reservation owner changed between arms")
    pairs = {name: {order: summarize_pair(runs[f"riley-{order}"][0][name], runs[f"qkv-{order}"][0][name])
                    for order in ("first", "second")} for name in CELLS}
    return {"remote_stage": root.name, "archived_report_sha256": REFERENCE, "binary_sha256": BINARY,
            "physical_gpu": 7, "order": ORDER, "chronology": chronology,
            "arms": arms, "cells": pairs}


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("archived", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.root, args.archived), indent=2, allow_nan=False))
