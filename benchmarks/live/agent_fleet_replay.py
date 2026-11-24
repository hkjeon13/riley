"""Measure agent workflow latency and capacity on a supplied inventory of Riley endpoints."""

import argparse
import json
import os
from contextlib import nullcontext
from pathlib import Path
from urllib.parse import urlsplit

from replica_workflow import run


def inventory(source):
    endpoints = []
    members = []
    addresses = set()
    for item in source["replicas"]:
        url = item["url"].rstrip("/")
        parts = urlsplit(url)
        if (
            parts.scheme not in ("http", "https")
            or not parts.hostname
            or parts.username is not None
            or parts.password is not None
            or parts.query
            or parts.fragment
            or url in addresses
        ):
            raise ValueError(
                "expected distinct HTTP(S) endpoint URLs without inline credentials or query parameters"
            )
        addresses.add(url)
        bearer = os.environ[item["bearer_env"]] if item.get("bearer_env") else None
        if item.get("bearer_env") and not bearer:
            raise ValueError("required bearer environment variable is empty")
        gpus = item["gpu_count"]
        if (
            type(gpus) is not int
            or gpus < 1
            or not isinstance(item["node"], str)
            or not item["node"].strip()
        ):
            raise ValueError(
                "each replica needs a node label and positive GPU allocation"
            )
        endpoints.append((url, bearer))
        members.append({"node": item["node"], "gpu_count": gpus})
    if not members or not source["model"]:
        raise ValueError("expected a model and at least one replica")
    return endpoints, members


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inventory", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument(
        "--policy",
        choices=(
            "round-robin",
            "session",
            "cost",
            "capacity",
            "observed-capacity",
            "fresh-capacity",
            "defer-cold-prefill",
        ),
        default=None,
    )
    p.add_argument("--waves", nargs="+", type=int, default=[1, 2, 4, 8])
    p.add_argument(
        "--dispatch-fixed-tools",
        action="store_true",
        help="let the application execute uniquely specified named tools on all workflows",
    )
    p.add_argument(
        "--preview-budget-blocks",
        type=int,
        nargs=2,
        metavar=("PER_REPLICA", "PER_WORKFLOW"),
        help="reserve 16-token blocks for speculative prompts on each session-pinned replica",
    )
    p.add_argument("--prompt-profiler", type=Path)
    p.add_argument("--tokenizer-dir", type=Path)
    p.add_argument("--llama3", action="store_true")
    args = p.parse_args()
    bounded = args.preview_budget_blocks is not None
    if bounded != (args.prompt_profiler is not None) or bounded != (
        args.tokenizer_dir is not None
    ):
        p.error(
            "bounded fleet previews require --preview-budget-blocks, --prompt-profiler and --tokenizer-dir together"
        )
    if (args.llama3 and not bounded) or (
        bounded and min(args.preview_budget_blocks) < 1
    ):
        p.error(
            "supply positive preview block counts and a checkpoint for the native profiler"
        )
    args.policy = args.policy or ("session" if bounded else "round-robin")

    if any(w < 1 for w in args.waves):
        p.error("wave sizes must be positive")
    source = json.loads(args.inventory.read_text())
    endpoints, members = inventory(source)
    session = nullcontext()
    if bounded:
        from budgeted_tool_prefix import FleetPreviews, PromptProfiler

        if len(endpoints) > 1 and args.policy != "session":
            p.error(
                "multiple replicas need session routing so real finals can reuse their prepared KV"
            )
        session = PromptProfiler(
            args.prompt_profiler, args.tokenizer_dir, llama3=args.llama3
        )
    with session as profiler, args.output.open("x") as file:
        previews = (
            FleetPreviews(profiler, len(endpoints), *args.preview_budget_blocks)
            if bounded
            else None
        )
        results = run(
            endpoints,
            args.policy,
            model=source["model"],
            waves=args.waves,
            active_limits=[r.get("max_active_sequences") for r in source["replicas"]],
            gpu_count=sum(m["gpu_count"] for m in members),
            dispatch_fixed_tools=args.dispatch_fixed_tools,
            previews=previews,
        )
        json.dump(
            {
                "model": source["model"],
                "reported_topology": members,
                "policy": args.policy,
                "tool_dispatch": "application-when-fixed"
                if args.dispatch_fixed_tools
                else "model",
                "waves": results,
                **({"preview_budget": previews.summary()} if previews else {}),
            },
            file,
            indent=2,
        )
        file.write("\n")
    if not any(
        t.get("status")
        for wave in results
        for work in wave["records"]
        for t in work["turns"]
    ):
        raise SystemExit(
            "no endpoint returned an HTTP response; see individual request errors"
        )


if __name__ == "__main__":
    main()
