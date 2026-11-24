"""Verify whole-record inputs, generated answers and contention during the return waves."""

import argparse
import json
import statistics
import sys
import tarfile
from hashlib import sha256
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "live"))
from agent_workflow import fingerprint, percentile
from long_tool_records import extract, fixture
from tool_return_projection import plan
from verify_tool_returns import metric, read


def identity(row):
    return (
        row["input_sha256"],
        row["prompt_ids_sha256"],
        row["generated_ids"],
        row["finish"],
        row["usage"],
    )


def inspect(root):
    if (root / "exit-status").read_text().strip() != "0" or not (
        root / "released"
    ).is_file():
        raise ValueError("incomplete run")
    for name in (
        "tool-return-projection-registration.json",
        "tool-return-registration.json",
        "tool_return_client.py",
        "tool_return_projection.py",
        "long_tool_records.py",
        "validate_projection.py",
    ):
        if (root / name).read_bytes() != (HERE / name).read_bytes():
            raise ValueError("tool or workload changed")
    for name in ("agent_tool_projection.py", "agent_workflow.py"):
        if (root / name).read_bytes() != (HERE.parents[1] / "live" / name).read_bytes():
            raise ValueError("source mismatch")
    schedules = plan(read(root / "tool-return-registration.json"))["schedules"]
    if schedules != read(root / "schedules.json")["schedules"]:
        raise ValueError("registered schedule changed")
    result, shared = {}, {}
    with tarfile.open(HERE / "tool-return-raw.tar.gz") as ref:
        for gpu, specs in schedules.items():
            work = root / f"gpu{gpu}/control"
            receipt, run_plan, lease, shut = (
                read(work / f"{n}.json")
                for n in ("receipt", "plan", "lease-after", "shutdown")
            )
            previous_receipt, previous_plan = (
                json.load(ref.extractfile(f"./gpu{gpu}/control/{n}.json"))
                for n in ("receipt", "plan")
            )
            if (
                any(
                    receipt[k] != v
                    for k, v in previous_receipt.items()
                    if k != "results"
                )
                or set(receipt["results"]) != set(specs)
                or run_plan["cells"] != specs
                or previous_plan["server"] != run_plan["server"]
                or previous_plan["server_cpu_affinity"]
                != run_plan["server_cpu_affinity"]
                or "RILEY_EXPERIMENT_W3_ADMISSION_FIFO=0" not in run_plan["docker"]
                or int(gpu) != lease["physical_index"]
                or lease["phase"] != "idle"
                or lease["other_processes"]
                or any(
                    shut[k]
                    for k in (
                        "active_requests",
                        "waiting_requests",
                        "kv_allocated_blocks",
                    )
                )
                or any(shut["allocation"].values())
            ):
                raise ValueError("hardware, scheduler, holdout or shutdown mismatch")
            events = {"full": [], "selected": []}
            details = {}
            previous_done = 0
            for name, spec in specs.items():
                policy = spec["policy"]
                case = name.removesuffix("-" + policy)
                path = work / f"{name}.json"
                if sha256(path.read_bytes()).hexdigest() != receipt["results"][name]:
                    raise ValueError("unbound result")
                data = read(path)
                original = json.load(ref.extractfile(f"./gpu{gpu}/control/{case}.json"))
                reader, rows = data["reader"], data["measured"]
                reader_metrics = metric(reader)
                if reader["sent_ns"] <= previous_done or len(rows) != spec["requests"]:
                    raise ValueError("incorrect measurement order")
                previous_done = max(r["done_ns"] for r in [reader, *rows])
                samples = []
                matches = 0
                full_reference_match = True
                same_generated = 0
                for i, (r, base) in enumerate(
                    zip(rows, original["measured"], strict=True)
                ):
                    seed = spec["seed"] + i
                    count = spec["counts"][i % len(spec["counts"])]
                    request, expected = fixture(seed, count)
                    request["return_token_ids"] = True
                    tool = request["messages"][-1]
                    if (
                        r["seed"] != seed
                        or r["expected"] != expected
                        or r["keys"] != list(expected)
                        or r["records_full"] != count
                        or r["full_result_hash"] != fingerprint(tool)
                    ):
                        raise ValueError("original source or requested keys changed")
                    if policy == "selected":
                        records = [
                            x
                            for x in json.loads(tool["content"])
                            if x["source"] in expected
                        ]
                        if len(records) != 2:
                            raise ValueError("not a two-record lookup")
                        tool["content"] = json.dumps(records, separators=(",", ":"))
                    if (
                        fingerprint(request) != r["input_sha256"]
                        or r["scheduled_ns"]
                        != reader["content_ns"][0] + i * spec["spacing_ms"] * 1_000_000
                    ):
                        raise ValueError("wire request or release time changed")
                    sig = (r["input_sha256"], r["prompt_ids_sha256"])
                    if sig != shared.setdefault((name, i), sig):
                        raise ValueError("the two GPUs received different tokens")
                    m = metric(r)
                    samples.append(m)
                    matches += bool(r["correct"] and extract(r["answer"]) == expected)
                    same_generated += r["generated_ids"] == base["generated_ids"]
                    if policy == "full":
                        full_reference_match &= identity(r) == identity(base)
                info = {
                    "correct": matches,
                    "full_reference_match": full_reference_match
                    if policy == "full"
                    else None,
                    "same_generated_as_full": same_generated,
                    "total": len(rows),
                    "worst_last_ms": max(m["last_ms"] for m in samples),
                    "overlap": sum(
                        r["request_written_ns"] < reader["content_ns"][-1] for r in rows
                    ),
                    "reader_match": identity(reader) == identity(original["reader"]),
                    "max_write_error_ms": max(
                        abs(r["request_written_ns"] - r["scheduled_ns"]) / 1e6
                        for r in rows
                    ),
                }
                details[name] = info
                events[policy].append((samples, reader_metrics, rows))
            arms = {}
            for policy, runs in events.items():
                samples = [s for sample, _, _ in runs for s in sample]
                rows = [r for _, _, rows in runs for r in rows]
                arms[policy] = {
                    "first_p95_ms": percentile([s["first_ms"] for s in samples], 0.95),
                    "last_p95_ms": percentile([s["last_ms"] for s in samples], 0.95),
                    "reader_tpot_p50_ms": statistics.median(
                        m["tpot_ms"] for _, m, _ in runs
                    ),
                    "reader_gap_p95_ms": percentile(
                        [g for _, m, _ in runs for g in m["gaps_ms"]], 0.95
                    ),
                    "prompt_tokens": sorted(
                        {r["usage"]["prompt_tokens"] for r in rows}
                    ),
                    "completion_tokens": sorted(
                        {r["usage"]["completion_tokens"] for r in rows}
                    ),
                }
            full, selected = arms["full"], arms["selected"]
            gates = {
                "accuracy": all(
                    w["correct"] == w["total"]
                    and w["reader_match"]
                    and w["full_reference_match"] is not False
                    for w in details.values()
                ),
                "dispatch": all(
                    w["max_write_error_ms"] <= 15 and w["overlap"] > 0
                    for w in details.values()
                ),
                "first": selected["first_p95_ms"] <= 0.8 * full["first_p95_ms"],
                "last": selected["last_p95_ms"] <= 0.8 * full["last_p95_ms"],
                "reader_tpot": selected["reader_tpot_p50_ms"]
                <= 1.05 * full["reader_tpot_p50_ms"],
                "reader_gap": selected["reader_gap_p95_ms"]
                <= 1.05 * full["reader_gap_p95_ms"],
            }
            result[gpu] = {"arms": arms, "gates": gates, "waves": details}
    return {
        "gpus": result,
        "all_pass": all(all(g["gates"].values()) for g in result.values()),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    print(json.dumps(inspect(p.parse_args().root), indent=2))
