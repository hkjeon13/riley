"""Replay dependent workflows in two clients with private routing state and a shared GPU pair."""

import argparse
import concurrent.futures
import json
import multiprocessing
import os
import time
from contextlib import nullcontext
from pathlib import Path

from agent_fleet_replay import inventory
from agent_workflow import replay
from replica_load import ReplicaLoad
from replica_workflow import SAMPLED_POLICIES, Router

LAYOUT_INSTRUCTION = (
    "Keep the three opening sentences exactly in the format requested, with no citations in them. "
    "Then write one separate 90-word paragraph. In that paragraph repeat each measured value "
    "followed by its unit and its exact report id in parentheses. Explain the 95th percentile. "
    "Limit the interpretation to the reported percentile; avoid claims about causes or how "
    "individual requests changed."
)


def with_layout(request):
    return {
        **request,
        "messages": [
            *request["messages"],
            {"role": "user", "content": LAYOUT_INSTRUCTION},
        ],
    }


def client(
    endpoints,
    model,
    policy,
    client_id,
    wave,
    barrier,
    seed_stride=10,
    seed_offset=0,
    final_transform=None,
):
    ctx = ReplicaLoad(endpoints) if policy in SAMPLED_POLICIES else nullcontext()
    with ctx as sampler:
        router = Router(endpoints, policy, [8, 8], sampler)
        barrier.wait(30)
        origin = time.monotonic() + 0.05

        def submit(index):
            time.sleep(max(0, origin + index * 0.025 - time.monotonic()))
            seed = 761 + seed_offset + seed_stride * wave + 20 * client_id + index
            route = router.sender(index)

            def send(address, request, row, first):
                if final_transform and row["turn"] == 2:
                    request = final_transform(request)
                return route(address, request, row, first)

            return replay(
                "",
                model,
                seed,
                96 if index % 2 else 2,
                0.1,
                f"session-{seed}",
                history=100,
                send=send,
            )

        with concurrent.futures.ThreadPoolExecutor(4) as workers:
            records = list(workers.map(submit, range(4)))
    return {"pid": os.getpid(), "scheduled_start_s": origin, "records": records}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inventory", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("block", type=int, choices=[0, 1, 2])
    p.add_argument("--defer-mixed", action="store_true")
    p.add_argument("--citation-mixed", action="store_true")
    p.add_argument("--citation-layout", action="store_true")
    args = p.parse_args()
    if args.citation_layout and not args.citation_mixed:
        p.error("--citation-layout requires --citation-mixed")
    manifest = json.loads(args.inventory.read_text())
    endpoints, _ = inventory(manifest)
    ctx = multiprocessing.get_context("spawn")
    order = ["capacity", "observed-capacity"]
    final_transform = None
    if args.citation_mixed:
        if args.block == 2 or args.defer_mixed:
            p.error("citation crossover accepts blocks 0 and 1")
        if args.citation_layout:
            final_transform = with_layout
        else:
            from verify_agent_citations import with_citations

            final_transform = with_citations
        order = ["round-robin"] * 8
    elif args.defer_mixed:
        policies = ["round-robin", "observed-capacity", "defer-cold-prefill"]
        order = [policies[(args.block + offset) % 3] for offset in (0, 1, 2, 2, 1, 0)]
    elif args.block == 1:
        order.reverse()
    elif args.block:
        p.error("block 2 requires --defer-mixed")
    result = []
    for wave, policy in enumerate(order):
        with (
            ctx.Manager() as manager,
            concurrent.futures.ProcessPoolExecutor(2, mp_context=ctx) as pool,
        ):
            gate = manager.Barrier(2)
            peers = [
                pool.submit(
                    client,
                    endpoints,
                    manifest["model"],
                    policy,
                    i,
                    wave,
                    gate,
                    40 if args.defer_mixed or args.citation_mixed else 10,
                    23760 if args.citation_mixed and wave >= 6 else 0,
                    final_transform if (args.block + wave) % 2 == 1 else None,
                )
                for i in (0, 1)
            ]
            result.append(
                {
                    "block": args.block,
                    "wave": wave,
                    "policy": policy,
                    "clients": [p.result() for p in peers],
                }
            )
        args.output.write_text(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
