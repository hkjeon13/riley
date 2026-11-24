"""Compare two tool result representations with fixed GPU scheduling."""

import argparse
import json
import os
from pathlib import Path
from unittest.mock import patch

from agent_tool_projection import project_exact_records
from agent_workflow import fingerprint, stream

import tool_return_client as replay
from validate_projection import check_fallback


def plan(reg):
    result = replay.schedule(reg)
    for gpu, cells in result["schedules"].items():
        result["schedules"][gpu] = {}
        for i, (name, spec) in enumerate(cells.items()):
            order = (
                ("full", "selected")
                if (int(gpu) + i) % 2 == 0
                else ("selected", "full")
            )
            for policy in order:
                result["schedules"][gpu][f"{name}-{policy}"] = {
                    **spec,
                    "policy": policy,
                }
    return result


def prepare(req, seed, count, selected):
    # The fixture's typed lookup declares these two keys before the tool runs.
    keys = [f"F{seed}-{i:04d}" for i in (count // 5, count * 4 // 5)]
    tool = req["messages"][-1]
    meta = {"keys": keys, "full_result_hash": fingerprint(tool), "records_full": count}
    if selected:
        data = json.loads(tool["content"])
        retained = project_exact_records(data, key="source", requested_ids=keys)
        tool["content"] = json.dumps(retained, separators=(",", ":"))
    return meta


def run(address, spec):
    def dispatch(endpoint, req, row, **kwargs):
        if "seed" in row:
            i = row["seed"] - spec["seed"]
            row.update(
                prepare(
                    req,
                    row["seed"],
                    spec["counts"][i % len(spec["counts"])],
                    spec["policy"] == "selected",
                )
            )
        return stream(endpoint, req, row, **kwargs)

    # Replay's worker threads use this transport instrumentation only for this wave.
    with patch.object(replay, "stream", dispatch):
        return replay.wave(address, spec)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("plan", type=Path)
    p.add_argument("address", nargs="?")
    a = p.parse_args()
    check_fallback()
    if a.address:
        os.sched_setaffinity(0, {int(os.environ["W3_CLIENT_CPU"])})
        for name, spec in json.loads(a.plan.read_text())["cells"].items():
            out = run(a.address, spec)
            (a.plan.parent / f"{name}.json").write_text(json.dumps(out) + "\n")
            print(name, len(out["measured"]), flush=True)
    else:
        reg = json.loads(
            Path(__file__).with_name("tool-return-registration.json").read_bytes()
        )
        a.plan.write_text(json.dumps(plan(reg), indent=2) + "\n")
