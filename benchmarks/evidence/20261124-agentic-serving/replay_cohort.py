"""Causal replay of four final answers sharing a GPU, using measured step costs.

Tool turns have already completed at the barrier. The three followers may be
released only after the reader's simulated first output. Prefix hits and output
lengths are taken from their recorded requests; results are diagnostic.
"""

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median


@dataclass
class Steps:
    begin: int
    end: int
    gpu: int
    items: list


def read_steps(path):
    steps = []
    for line in path.read_text().splitlines():
        if line.startswith("AGENT_STEP "):
            _, start, end, gpu, members = line.split(maxsplit=4)
            steps.append(Steps(int(start), int(end), int(gpu), json.loads(members)))
    return steps


def bucket(value, size):
    return (value + size - 1) // size


def features(items):
    p = [i for i in items if i[1] == "p"]
    d = [i for i in items if i[1] == "d"]
    return (
        sum(i[2] for i in p),
        len(d),
        len(p),
        bucket(sum(i[3] for i in d), 512),
        bucket(sum(i[2] * i[3] for i in p), 65536),
    )


def example_costs(log, workflows):
    ids = {int(r["turns"][-1]["response_id"][5:], 16) for r in workflows}
    result = []
    for i, step in enumerate(log[:-1]):
        if all(item[0] in ids for item in step.items):
            # Include observed handoff before the next GPU step only if this
            # cohort already has more work. Client/tool idle is not step cost.
            gap = (
                log[i + 1].begin - step.end
                if any(x[0] in ids for x in log[i + 1].items)
                else 0
            )
            result.append(
                (
                    features(step.items),
                    (step.end - step.begin + max(0, min(gap, 2_000_000))) / 1e6,
                )
            )
    return result


class UnsupportedShape(ValueError):
    """No calibrated cost close enough to support a policy comparison."""


class StepCost:
    def __init__(self, training):
        groups = defaultdict(list)
        for key, value in training:
            groups[key].append(value)
        self.groups = {key: median(values) for key, values in groups.items()}

    def predict(self, key):
        rows, decodes, prefills, dkv, qwork = key
        valid = [
            k
            for k in self.groups
            if bool(k[0]) == bool(rows) and bool(k[1]) == bool(decodes)
        ]
        if not valid or rows > max(k[0] for k in valid) + 3:
            raise UnsupportedShape(f"uncalibrated row count for {key}")

        def distance(k):
            return (
                abs(k[0] - rows) / max(16, rows),
                abs(k[1] - decodes),
                abs(k[2] - prefills),
                abs(k[4] - qwork) + abs(k[3] - dkv),
            )

        best = min(distance(k) for k in valid)
        return median(self.groups[k] for k in valid if distance(k) == best)


def scenario(log, records):
    jobs = []
    for record in records:
        last = record["turns"][-1]
        rid = int(last["response_id"][5:], 16)
        hits = last["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0)
        selected = [(s, it) for s in log for it in s.items if it[0] == rid]
        pref = [s for s, it in selected if it[1] == "p"]
        dec = [s for s, it in selected if it[1] == "d"]
        if (
            not pref
            or not dec
            or sum(it[2] for _, it in selected if it[1] == "p") + hits
            != last["usage"]["prompt_tokens"]
        ):
            raise ValueError("missing final-turn trace")
        jobs.append(
            {
                "idx": len(jobs),
                "cached": hits,
                "prompt": last["usage"]["prompt_tokens"] - hits,
                "decode": len(dec),
                "first": pref[-1].end,
                "start": pref[0].begin,
                "end": dec[-1].end,
                "submitted": last["sent_ns"] + record["epoch_offset_ns"],
            }
        )
    reader = jobs[0]
    if any(job["submitted"] < reader["first"] - 1_000_000 for job in jobs[1:]):
        raise ValueError("follower released before reader first output")
    return jobs


def simulate(jobs, cost, width, policy):
    state = [
        dict(
            j,
            pre=j["prompt"],
            left=j["decode"],
            first_ms=None,
            end_ms=None,
            ready=0 if i == 0 else None,
            last_scheduled=-1,
        )
        for i, j in enumerate(jobs)
    ]
    now = 0
    gaps = []
    while any(j["end_ms"] is None for j in state):
        active = [
            j
            for j in state
            if j["ready"] is not None and j["ready"] <= now and j["end_ms"] is None
        ]
        if not active:
            now = min(
                j["ready"]
                for j in state
                if j["end_ms"] is None and j["ready"] is not None
            )
            continue
        decoding = [j for j in active if j["pre"] == 0]
        prefill = [j for j in active if j["pre"]]
        candidates = []
        if policy == "rotate":
            prefill.sort(key=lambda j: (j["last_scheduled"], j["ready"], j["idx"]))
        else:
            prefill.sort(key=lambda j: (j["ready"], j["idx"]))
        for size in (
            [128, 256, 512, 1024]
            if policy == "deadline" and decoding and prefill
            else [width]
        ):
            remaining = size - len(decoding)
            items = [
                (
                    j["idx"],
                    "d",
                    1,
                    j["cached"] + j["prompt"] + j["decode"] - j["left"] + 1,
                    True,
                )
                for j in decoding
            ]
            for j in prefill:
                n = min(remaining, j["pre"])
                if n > 0:
                    items.append(
                        (
                            j["idx"],
                            "p",
                            n,
                            j["cached"] + j["prompt"] - j["pre"] + n,
                            n == j["pre"],
                        )
                    )
                    remaining -= n
            try:
                duration = cost.predict(features(items))
            except UnsupportedShape:
                if policy == "deadline" and len(items) and decoding and prefill:
                    continue
                raise
            candidates.append((items, duration))
        if not candidates:
            raise UnsupportedShape("no feasible calibrated shape for current streams")
        if len(candidates) > 1:
            # 30ms exploratory service budget; the public SLO gate stays separate.
            feasible = [
                item
                for item in candidates
                if all(now + item[1] - j["last_scheduled"] <= 30 for j in decoding)
            ]
            items, duration = (
                feasible[-1] if feasible else min(candidates, key=lambda pair: pair[1])
            )
        else:
            items, duration = candidates[0]
        now += duration
        for idx, kind, n, _, _ in items:
            j = state[idx]
            if kind == "p":
                j["pre"] -= n
                if j["pre"] == 0:
                    j["first_ms"] = now
                    if idx == 0:
                        for follower in state[1:]:
                            follower["ready"] = now + max(
                                0, (follower["submitted"] - jobs[0]["first"]) / 1e6
                            )
            else:
                gaps.append(now - j["last_scheduled"])
                j["left"] -= 1
                if j["left"] == 0:
                    j["end_ms"] = now
            j["last_scheduled"] = now
    return {
        "last_first_ms": max(j["first_ms"] for j in state),
        "last_end_ms": now,
        "max_decode_gap_ms": max(gaps),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    training = []
    cases = []
    directories = [*args.directory.glob("width-*"), *args.directory.glob("retry-*")]
    for directory in sorted(directories):
        if not (directory / "cohort-761.json").is_file():
            continue
        log = read_steps(directory / "server.log")
        for seed in (101, 111, 761):
            records = json.loads((directory / f"cohort-{seed}.json").read_text())
            if seed == 761:
                cases.append(
                    (
                        int(directory.name.split("-")[1]),
                        scenario(log, records),
                        sum(r["ok"] for r in records),
                    )
                )
            else:
                training.extend(example_costs(log, records))
    model = StepCost(training)
    report = []
    for width, jobs, correct in cases:
        base = jobs[0]["start"]
        predictions = {}
        for policy in ("fifo", "rotate", "deadline"):
            try:
                predictions[policy] = simulate(jobs, model, width, policy)
            except UnsupportedShape as error:
                predictions[policy] = {"unsupported": str(error)}
        report.append(
            {
                "width": width,
                "correct_workflows": correct,
                "observed_last_first_ms": (max(j["first"] for j in jobs) - base) / 1e6,
                "observed_last_end_ms": (max(j["end"] for j in jobs) - base) / 1e6,
                "scenarios": predictions,
            }
        )
    print(
        json.dumps(
            {
                "training_steps": len(training),
                "distinct_cost_cells": len(model.groups),
                "heldout": report,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
