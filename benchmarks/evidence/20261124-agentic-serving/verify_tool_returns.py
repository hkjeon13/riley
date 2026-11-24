"""Rebuild tool-return latency, exact answers and stream progress from raw events."""

import argparse
import json
import statistics
import sys
import tarfile
from hashlib import sha256
from itertools import pairwise
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "live"))
from agent_workflow import fingerprint, percentile
from long_tool_records import extract, fixture
from tool_return_client import schedule


def read(p):
    return json.loads(p.read_bytes())


def metric(row):
    events = row["content_ns"]
    usage = row["usage"]
    if (
        not events
        or sorted(events) != events
        or not row["sent_ns"]
        <= row["request_written_ns"]
        <= events[0]
        <= events[-1]
        <= row["done_ns"]
        or row["status"] != 200
        or not row["prompt_ids_sha256"]
        or not row["generated_ids"]
        or usage["completion_tokens"] < 2
        or row["finish"] not in ("stop", "length")
    ):
        raise ValueError("stream omitted timing, token identities or terminal state")
    return {
        "first_ms": (events[0] - row.get("scheduled_ns", row["sent_ns"])) / 1e6,
        "last_ms": (events[-1] - row.get("scheduled_ns", row["sent_ns"])) / 1e6,
        "tpot_ms": (events[-1] - events[0]) / 1e6 / (usage["completion_tokens"] - 1),
        "gaps_ms": [(b - a) / 1e6 for a, b in pairwise(events)],
    }


def inspect(root):
    if (root / "exit-status").read_text().strip() != "0" or not (
        root / "released"
    ).is_file():
        raise ValueError("incomplete experiment")
    for p in [
        "tool-return-registration.json",
        "tool_return_client.py",
        "long_tool_records.py",
        "arrival-pressure.patch",
    ]:
        if (root / p).read_bytes() != (HERE / p).read_bytes():
            raise ValueError(f"inputs changed: {p}")
    if (root / "agent_workflow.py").read_bytes() != (
        HERE.parents[1] / "live/agent_workflow.py"
    ).read_bytes():
        raise ValueError("stream capture changed")
    reg = read(root / "tool-return-registration.json")
    schedules = schedule(reg)["schedules"]
    if read(root / "schedules.json")["schedules"] != schedules:
        raise ValueError("workload changed")
    protected = read(HERE / "arrival-pressure-verified.json")
    all_results = {}
    with tarfile.open(HERE / "arrival-pressure-raw.tar.gz") as source:
        for gpu, order in reg["order"].items():
            identities = {}
            arms = {}
            previous_done = 0
            for mode in order:
                work = root / f"gpu{gpu}" / mode

                receipt, plan, lease = (
                    read(work / f"{n}.json") for n in ("receipt", "plan", "lease-after")
                )
                baseline_receipt, baseline_plan = (
                    json.load(source.extractfile(f"./gpu{gpu}/{mode}/{name}.json"))
                    for name in ("receipt", "plan")
                )
                if (
                    receipt["riley_binary_sha256"] != protected["binary"]
                    or any(
                        receipt[k] != v
                        for k, v in baseline_receipt.items()
                        if k not in ("results",)
                    )
                    or plan["server"] != baseline_plan["server"]
                    or plan["server_cpu_affinity"]
                    != baseline_plan["server_cpu_affinity"]
                    or plan["cells"] != schedules[gpu]
                    or set(receipt["results"]) != set(schedules[gpu])
                    or f"RILEY_EXPERIMENT_W3_ADMISSION_FIFO={int(mode == 'candidate')}"
                    not in plan["docker"]
                    or lease["phase"] != "idle"
                    or lease["other_processes"]
                    or (lease["gpu"], lease["physical_index"])
                    != (receipt["gpu"], int(gpu))
                    or sha256((work / "token-id-holdout.json").read_bytes()).hexdigest()
                    != receipt["holdout_sha256"]
                ):
                    raise ValueError(
                        "device, policy, configuration or held-out tokens changed"
                    )
                firsts, lasts, tpots, gaps, delays = [], [], [], [], []
                waves = {}
                for name, spec in schedules[gpu].items():
                    p = work / f"{name}.json"
                    if sha256(p.read_bytes()).hexdigest() != receipt["results"][name]:
                        raise ValueError("receipt does not bind wave")
                    data = read(p)
                    reader, rows = data["reader"], data["measured"]
                    if (
                        len(rows) != spec["requests"]
                        or reader["sent_ns"] <= previous_done
                    ):
                        raise ValueError(
                            "unexpected wave size or physical execution order"
                        )
                    previous_done = max(row["done_ns"] for row in [reader, *rows])
                    reader_metrics = metric(reader)
                    tpots.append(reader_metrics["tpot_ms"])
                    gaps.extend(reader_metrics["gaps_ms"])
                    wave = []
                    exact = 0
                    for i, row in enumerate(rows):
                        req, expected = fixture(
                            spec["seed"] + i, spec["counts"][i % len(spec["counts"])]
                        )
                        req["return_token_ids"] = True
                        if (
                            fingerprint(req) != row["input_sha256"]
                            or expected != row["expected"]
                            or row["scheduled_ns"]
                            != reader["content_ns"][0]
                            + i * spec["spacing_ms"] * 1_000_000
                            or row["usage"]["prompt_tokens"] < 512
                        ):
                            raise ValueError("unmatched request or trigger")
                        exact += bool(
                            row["correct"] and extract(row["answer"]) == expected
                        )
                        delays.append(
                            abs(row["request_written_ns"] - row["scheduled_ns"]) / 1e6
                        )
                        m = metric(row)
                        firsts.append(m["first_ms"])
                        lasts.append(m["last_ms"])
                        wave.append(m["last_ms"])
                    ids = [
                        (
                            r["input_sha256"],
                            r["prompt_ids_sha256"],
                            r["generated_ids"],
                            r["finish"],
                            r["usage"],
                        )
                        for r in [reader, *rows]
                    ]
                    if mode == order[0]:
                        identities[name] = ids
                    waves[name] = {
                        "correct": exact,
                        "total": len(rows),
                        "same_tokens": ids == identities[name],
                        "overlap": sum(
                            r["request_written_ns"] < reader["content_ns"][-1]
                            for r in rows
                        ),
                        "last_max_ms": max(wave),
                        "reader_tpot_ms": tpots[-1],
                    }
                arms[mode] = {
                    "waves": waves,
                    "first_p95_ms": percentile(firsts, 0.95),
                    "last_p95_ms": percentile(lasts, 0.95),
                    "reader_tpot_p50_ms": statistics.median(tpots),
                    "reader_content_gap_p95_ms": percentile(gaps, 0.95),
                    "max_dispatch_error_ms": max(delays),
                    "prompt_tokens": sorted(
                        {
                            r[4]["prompt_tokens"]
                            for sel in identities.values()
                            for r in sel[1:]
                        }
                    ),
                }
            c, t = arms["control"], arms["candidate"]
            gates = {
                "quality": all(
                    w["correct"] == w["total"] and w["same_tokens"]
                    for a in arms.values()
                    for w in a["waves"].values()
                ),
                "overlap": all(
                    w["overlap"] >= 1
                    for a in arms.values()
                    for w in a["waves"].values()
                ),
                "dispatch": max(a["max_dispatch_error_ms"] for a in arms.values())
                <= 15,
                "first": t["first_p95_ms"] < c["first_p95_ms"],
                "last": t["last_p95_ms"] < c["last_p95_ms"],
                "reader_tpot": t["reader_tpot_p50_ms"]
                <= c["reader_tpot_p50_ms"] * 1.05,
                "reader_gap": t["reader_content_gap_p95_ms"]
                <= c["reader_content_gap_p95_ms"] * 1.05,
            }
            all_results[gpu] = {"order": order, "arms": arms, "gates": gates}
    return {
        "gpus": all_results,
        "all_gates_pass": all(all(a["gates"].values()) for a in all_results.values()),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    print(json.dumps(inspect(p.parse_args().root), indent=2))
