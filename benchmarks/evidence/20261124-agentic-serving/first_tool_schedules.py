"""Compare full and cache-budgeted first-tool preparation with native prompt/scheduler probes."""

import argparse
import json
import subprocess
from pathlib import Path

from agent_workflow import fingerprint
from first_tool_capacity import common
from tool_wait_trial import prepare_prefix
from verify_coder_named_steps import capture_final


def render(prompts, executable, tokenizer):
    proc = subprocess.run(
        [executable, tokenizer],
        input="".join(json.dumps(p) + "\n" for p in prompts),
        text=True,
        capture_output=True,
        check=True,
    )
    output = [json.loads(s)["ids"] for s in proc.stdout.splitlines()]
    assert len(output) == len(prompts)
    return output


def partials(request):
    # This registered fixture starts with a single long user audit log.
    # Stop at that message: all later conversation and tool results are absent.
    msg = request["messages"][0]
    assert msg["role"] == "user"
    return [
        {
            **request,
            "messages": [{**msg, "content": msg["content"][:cut]}],
            "add_generation_prompt": False,
        }
        for cut in range(512, len(msg["content"]), 512)
    ]


def interleaved(scheduler, groups, readers):
    results = []
    for ticks in (0, 4, 8, 16):
        for action in ("keep", "cancel"):
            proc = subprocess.run(
                [scheduler, str(ticks), action],
                input="".join(json.dumps(g) + "\n" for g in groups),
                text=True,
                capture_output=True,
                check=True,
            )
            for mode, g, line in zip(
                ("none", "full", "budget"),
                groups,
                proc.stdout.splitlines(),
                strict=True,
            ):
                prep, final = json.loads(line)
                assert sorted(j["seed"] for j in g[0]) == sorted(
                    [r[0] for r in prep["rows"]] + [r[0] for r in prep["cancelled"]]
                )
                assert sorted((j["seed"], len(j["ids"])) for j in g[1]) == sorted(
                    (k, n) for k, n, _ in final["rows"]
                )
                assert prep["computed"] == sum(n - h for _, n, h in prep["rows"]) + sum(
                    n for _, n in prep["cancelled"]
                )
                assert final["computed"] == sum(n - h for _, n, h in final["rows"])
                assert len(prep["cancelled"]) == (
                    prep["pending_at_boundary"] if action == "cancel" else 0
                )
                results.append(
                    {
                        "ticks": ticks,
                        "action": action,
                        "mode": mode,
                        "preview": prep,
                        "final": final,
                        "total_rows": prep["computed"] + final["computed"],
                        "last_final_tick": max(t for _, t in final["completion_ticks"]),
                        "last_reader_tick": max(
                            t
                            for seed, t in final["completion_ticks"]
                            if seed in readers
                        ),
                    }
                )
    return results


def run(root, executable, tokenizer, scheduler, cap, *, overlap=False):
    output = []
    for wave, bounds in zip(
        json.loads((root / "gpu6/selected/waves.json").read_text()),
        cap["devices"]["gpu6"],
        strict=True,
    ):
        assert wave["base"] == bounds["base"]
        targets = sum(j["role"] == "target" for j in wave["jobs"])
        share = 1024 // targets * 16
        raw = []
        for j in wave["jobs"]:
            snapshots = []

            def capture(index, request, row, snapshots=snapshots):
                if index == 0:
                    snapshots.append(
                        prepare_prefix(
                            request, row["call"], include_completed_call=True
                        )
                    )

            final, _, _ = capture_final(
                j["work"], j["role"] == "reader", on_tool_start=capture
            )
            [preview] = snapshots
            options = partials(preview) if j["role"] == "target" else []
            raw.append([final, preview, *options])
        flat = iter(
            render([p for variants in raw for p in variants], executable, tokenizer)
        )
        cases = []
        for job, variants, checked in zip(
            wave["jobs"], raw, bounds["jobs"], strict=True
        ):
            ids = [next(flat) for _ in variants]
            final, full, *short = ids
            seed = job["work"]["seed"]
            assert (
                seed == checked["seed"]
                and len(final) == checked["final_tokens"]
                and min(common(full, final), len(final) - 1) // 16 * 16
                == checked["prefix_tokens"][0]
            )
            budget = (
                full
                if len(full) <= share
                else next((s for s in reversed(short) if len(s) <= share), [])
            )
            cases.append(
                {
                    "seed": seed,
                    "target": job["role"] == "target",
                    "final": final,
                    "full": full,
                    "budget": budget,
                }
            )
        assert next(flat, None) is None
        trials = []
        groups = []
        for order in ("forward", "reverse"):
            final_order = cases if order == "forward" else list(reversed(cases))
            for mode in ("none", "full", "budget"):
                previews = [
                    {"seed": j["seed"], "ids": j[mode]}
                    for j in cases
                    if j["target"] and mode != "none" and j[mode]
                ]
                if mode == "budget":
                    assert all(len(p["ids"]) <= share for p in previews)
                finals = [{"seed": j["seed"], "ids": j["final"]} for j in final_order]
                groups.append([previews, finals])
                trials.append(
                    {
                        "order": order,
                        "mode": mode,
                        "inputs_sha256": fingerprint([previews, finals]),
                    }
                )
        proc = subprocess.run(
            [scheduler],
            input="".join(json.dumps(g) + "\n" for g in groups),
            capture_output=True,
            text=True,
            check=True,
        )
        readers = {j["seed"] for j in cases if not j["target"]}
        for t, g, line in zip(trials, groups, proc.stdout.splitlines(), strict=True):
            prep, final = json.loads(line)
            for expected, observed in zip(g, (prep, final), strict=True):
                assert sorted((j["seed"], len(j["ids"])) for j in expected) == sorted(
                    (k, n) for k, n, h in observed["rows"]
                )
                assert (
                    sum(n - h for _, n, h in observed["rows"]) == observed["computed"]
                )
                assert observed["retained"] <= 1536
            missing = [n - h for k, n, h in final["rows"] if k not in readers]
            t.update(
                preview=prep,
                final=final,
                total_rows=prep["computed"] + final["computed"],
                worst_target_rows=max(missing),
                reader_rows=sum(n - h for k, n, h in final["rows"] if k in readers),
            )
        output.append(
            {
                "base": wave["base"],
                "size": wave["size"],
                "share_tokens": share,
                "trials": trials,
                **(
                    {"interleaved": interleaved(scheduler, groups[:3], readers)}
                    if overlap and wave["size"] == 16
                    else {}
                ),
            }
        )
    return output


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("root", "profiler", "tokenizer", "scheduler", "capacity", "out"):
        p.add_argument(n, type=Path)
    p.add_argument("--overlap", action="store_true")
    a = p.parse_args()
    a.out.write_text(
        json.dumps(
            run(
                a.root,
                a.profiler,
                a.tokenizer,
                a.scheduler,
                json.loads(a.capacity.read_text()),
                overlap=a.overlap,
            ),
            indent=2,
        )
        + "\n"
    )
