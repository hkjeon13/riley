"""Validate paired full-register contents, exact answers and first/last content times."""

import argparse
import json
import sys
from hashlib import sha256
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "live"))
from agent_workflow import fingerprint
from full_register import body, case, parse
from verify_long_tile import read, receipt


def inspect(root):
    if (root / "exit-status").read_text().strip() != "0" or not (
        root / "released"
    ).exists():
        raise ValueError("incomplete crossover")
    reg = read(root / "full-register-registration.json")
    if reg != read(HERE / "full-register-registration.json"):
        raise ValueError("registration mismatch")
    for line in (root / "inputs.sha256").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        if sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"input drift: {name}")
    results = {}
    sources = {}
    checkpoints = read(HERE / "long-tool-records-verified.json")
    for gpu, order in reg["order"].items():
        modes = {}
        previous_done = 0
        reference_args = None
        for mode in order:
            path = root / f"gpu{gpu}-{mode}"
            args, device, binary = receipt(
                path, "page-v1" if mode == "page" else "fixed-tile-unsplit-v1"
            )
            lease = read(path / "lease-after.json")
            if (
                binary != checkpoints["binary_sha256"]
                or lease["physical_index"] != int(gpu)
                or device
                != (path / "device.csv").read_text().splitlines()[1].split(",")[0]
                or lease["other_processes"]
                or "--prefix-cache-blocks" in args
            ):
                raise ValueError("incorrect binary, GPU or retention")
            if reference_args is not None and reference_args != args:
                raise ValueError("runner configuration differs")
            reference_args = args
            captured = iter(read(path / "full-register.json"))
            outcomes = {}
            for kind, specs in reg["cases"].items():
                for seed, count in specs:
                    combos = [
                        (False, False),
                        (False, True),
                        (True, False),
                        (True, True),
                    ]
                    if gpu == "7":
                        combos.reverse()
                    for table, strict in combos:
                        row = next(captured)
                        req, records, expected, schema = case(kind, seed, count)
                        wire = body(req, records, schema, table, strict)
                        sent = json.loads(wire["messages"][-1]["content"])
                        if table:
                            sent = [
                                dict(zip(sent["columns"], r, strict=True))
                                for r in sent["rows"]
                            ]
                        if sent != records or (
                            row["seed"],
                            row["count"],
                            row["kind"],
                            row["table"],
                            row["strict"],
                        ) != (seed, count, kind, table, strict):
                            raise ValueError("incomplete source table or wrong order")
                        m = row["measurement"]
                        if (
                            row["result_hash"] != fingerprint(records)
                            or row["expected"] != expected
                            or m["input_sha256"] != fingerprint(wire)
                            or row["start_ns"] <= previous_done
                        ):
                            raise ValueError(
                                "wrong wire body or physical execution order"
                            )
                        events = m.get("content_ns", [])
                        complete = bool(
                            events
                            and m.get("done_ns")
                            and m.get("finish")
                            and m.get("status") == 200
                            and not row.get("error")
                        )
                        previous_done = m.get("done_ns", m["sent_ns"])
                        usage = m.get("usage", {})
                        if (
                            complete
                            and not row["start_ns"]
                            <= m["sent_ns"]
                            <= m["request_written_ns"]
                            <= events[0]
                            <= events[-1]
                            <= m["done_ns"]
                        ) or usage.get("prompt_tokens_details", {}).get(
                            "cached_tokens", 0
                        ):
                            raise ValueError("invalid event times or cached tokens")
                        key = f"{kind}-{seed}-{'table' if table else 'records'}-{'schema' if strict else 'free'}"
                        correct = (
                            complete
                            and m["finish"] == "stop"
                            and not row["calls"]
                            and fingerprint(parse(row["answer"]))
                            == fingerprint(expected)
                        )
                        outcomes[key] = {
                            "correct": correct,
                            "complete": complete,
                            "error": row.get("error"),
                            "finish": m.get("finish"),
                            "prompt": usage.get("prompt_tokens"),
                            "output": usage.get("completion_tokens"),
                            "first_ms": (events[0] - row["start_ns"]) / 1e6
                            if events
                            else None,
                            "last_ms": (events[-1] - row["start_ns"]) / 1e6
                            if events
                            else None,
                            "parsed": parse(row.get("answer", "")),
                        }
                        signature = (m["input_sha256"], usage.get("prompt_tokens"))
                        if complete and signature != sources.setdefault(key, signature):
                            raise ValueError("unmatched HTTP body or token length")
            if next(captured, None) is not None:
                raise ValueError("extra workload")
            modes[mode] = outcomes
        results[gpu] = modes
    summaries = {}
    for gpu, modes in results.items():
        stats = {}
        for mode, rows in modes.items():
            for form in ("records", "table"):
                for constraint in ("free", "schema"):
                    tag = f"{mode}-{form}-{constraint}"
                    stats[tag] = {}
                    for kind, specs in reg["cases"].items():
                        sampled = [
                            rows[f"{kind}-{seed}-{form}-{constraint}"]
                            for seed, _ in specs
                        ]
                        stats[tag][kind] = sum(r["correct"] for r in sampled)
                    changes = [
                        (
                            rows[f"{kind}-{seed}-{form}-{constraint}"],
                            modes["page"][f"{kind}-{seed}-records-free"],
                        )
                        for kind, specs in reg["cases"].items()
                        for seed, _ in specs
                    ]
                    stats[tag]["faster_first_and_last_all"] = all(
                        a["complete"]
                        and b["complete"]
                        and a["first_ms"] < b["first_ms"]
                        and a["last_ms"] < b["last_ms"]
                        for a, b in changes
                    )
                    stats[tag]["original_correct_lookups_preserved"] = all(
                        not b["correct"] or a["correct"]
                        for a, b in changes[: len(reg["cases"]["lookup"])]
                    )
                    stats[tag]["all_scan_max_correct"] = stats[tag]["scan"] == len(
                        reg["cases"]["scan"]
                    ) and stats[tag]["max"] == len(reg["cases"]["max"])
                    stats[tag]["smaller_all"] = all(
                        a["complete"] and b["complete"] and a["prompt"] < b["prompt"]
                        for a, b in changes
                    )
        summaries[gpu] = stats
    return {
        "gpus": results,
        "summary_against_page_records_free": summaries,
        "measured": sum(
            len(rows) for modes in results.values() for rows in modes.values()
        ),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    print(json.dumps(inspect(p.parse_args().root), indent=2))
