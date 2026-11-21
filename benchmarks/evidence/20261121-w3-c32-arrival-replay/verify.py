"""Recheck the fixed C32 schedule against the actual client writes and pair the outputs."""

import json
from pathlib import Path
import sys

from comparison_base import p
import private_reference
from replay import make_schedules
from verify_a100_candidate import digest, read, require


def check_replay(raw, plan):
    rows = raw["rows"]
    target = [value / 1e9 for value in plan["offsets_ns"]]
    require(len(rows) == len(target) == 96, "C32 record count changed")
    expected = sorted(range(len(target)), key=target.__getitem__)
    observed = sorted(expected, key=lambda index: rows[index]["sent_at"])
    require(observed == expected, "client write order differs from the reference")
    first = rows[observed[0]]["sent_at"]
    errors = [1000 * (row["sent_at"] - first - offset) for row, offset in zip(rows, target, strict=True)]
    require(all(abs(row["relative_write_error_ms"] - error) < 1e-6
                for row, error in zip(rows, errors, strict=True)), "client write offsets differ from the receipt")
    stats = raw["statistics"]
    error_p95 = p([abs(num) for num in errors], .95)
    burst_p95 = p([abs(errors[i]) for i in expected[:32]], .95)
    require(stats["write_order_matches"] is True and stats["config"]["reference_case"] == plan["reference_case"]
            and abs(stats["write_error_p95_ms"] - error_p95) < 1e-6
            and abs(stats["first_32_write_error_p95_ms"] - burst_p95) < 1e-6
            and error_p95 <= 5 and burst_p95 <= 5, "replay fidelity failed")
    write_ttft = [1000 * (row["ttft"] - (row["sent_at"] - row["start"])) for row in rows]
    require(all(ms > 0 for ms in write_ttft), "a first token preceded its client write")
    return {"reference_case": plan["reference_case"], "requests": len(rows),
            "write_error_p95_ms": error_p95, "first_32_write_error_p95_ms": burst_p95,
            "write_error_max_ms": max(abs(num) for num in errors),
            "reference_first_32_write_span_ms": 1000 * target[expected[31]],
            "actual_first_32_write_span_ms": 1000 * (rows[expected[31]]["sent_at"] - first),
            "first_32_indices": expected[:32], "write_to_first_token_p95_ms": p(write_ttft, .95),
            "first_32_write_to_first_token_p95_ms": p([write_ttft[i] for i in expected[:32]], .95),
            "remaining_64_write_to_first_token_p95_ms": p([write_ttft[i] for i in expected[32:]], .95)}


def verify(stage):
    prior = stage.parent / "w3-private-warm-gpu-20261121.Zdv0ggSI"
    plan = make_schedules(prior)
    checked = read(stage / "schedules.json")
    require(checked == plan, "actual schedule differs from the preserved source")
    result = private_reference.verify(stage)
    fidelity, changes, paired_errors = {}, {}, {}
    for index, order in private_reference.ORDERS.items():
        outcomes, actual = {}, {}
        for case in order:
            tag = f"{case}-gpu{index}"
            dataset = read(stage / tag / "workload.json")["short"]
            fidelity[tag] = check_replay(dataset, plan[index])
            outcomes[case] = fidelity[tag]
            actual[case] = [row["relative_write_error_ms"] for row in dataset["rows"]]
        changes[index], paired_errors[index] = {}, {}
        for name, (before, after) in private_reference.CONTRASTS.items():
            base, candidate = outcomes[before], outcomes[after]
            keys = ("write_to_first_token_p95_ms", "first_32_write_to_first_token_p95_ms",
                    "remaining_64_write_to_first_token_p95_ms")
            changes[index][name] = {key: 100 * (candidate[key] / base[key] - 1) for key in keys}
            errors = [abs(b - c) for b, c in zip(actual[before], actual[after], strict=True)]
            pair = {"all": p(errors, .95), "first_32": p([errors[i] for i in base["first_32_indices"]], .95)}
            require(all(value <= 5 for value in pair.values()), "client timing diverged between the paired arms")
            paired_errors[index][name] = pair
    result.update(schedules_sha256=digest(stage / "schedules.json"), write_fidelity=fidelity,
                  write_to_token_p95_change_percent=changes, paired_write_error_p95_ms=paired_errors)
    return result


if __name__ == "__main__":
    print(json.dumps(verify(Path(sys.argv[1])), indent=2, allow_nan=False))
