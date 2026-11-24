"""Acquire each owned lease through its controller and join both on every exit."""

import argparse
import concurrent.futures
import json
import os
import secrets
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from replica_process import serve


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("holder", type=Path)
    p.add_argument("directory", type=Path)
    p.add_argument(
        "policy",
        choices=(
            "round-robin",
            "session",
            "cost",
            "crossover",
            "inventory",
            "prefix-recency-0",
            "prefix-recency-1",
            "cross-router",
            "early-rejection",
            "natural-burst-0",
            "natural-burst-1",
            "citation-layout-0",
            "citation-layout-1",
            "citation-mixed-0",
            "citation-mixed-1",
            "defer-mixed-0",
            "defer-mixed-1",
            "defer-mixed-2",
            "multi-router-0",
            "multi-router-1",
            "readiness-burst",
            "protected-burst",
        ),
    )
    args = p.parse_args()
    root = Path(__file__).resolve().parent
    pair = args.directory.resolve()
    pair.mkdir()
    holder = args.holder.resolve()

    with (
        tempfile.TemporaryDirectory() as tmp,
        concurrent.futures.ThreadPoolExecutor(2) as pool,
    ):
        keydir = Path(tmp)
        for i in (6, 7):
            (keydir / str(i)).write_text(secrets.token_hex(32))
        multi_router = args.policy.startswith(
            ("multi-router-", "defer-mixed-", "citation-mixed-", "citation-layout-")
        )
        recency = args.policy.startswith("prefix-recency-")
        natural_burst = args.policy.startswith("natural-burst-")
        burst = args.policy in ("readiness-burst", "protected-burst")
        native = (
            args.policy in ("cross-router", "early-rejection")
            or recency
            or burst
            or multi_router
            or natural_burst
        )
        if recency and pair.name != args.policy:
            raise ValueError(
                "prefix-recency crossover requires matching directory and policy"
            )
        launcher = "run-cross-router-native.sh" if native else "run-replica.sh"
        futures = [
            pool.submit(serve, root, holder, pair, i, keydir, launcher=launcher)
            for i in (6, 7)
        ]
        try:
            deadline = time.monotonic() + 340
            while not all((pair / f"gpu{i}" / "address").is_file() for i in (6, 7)):
                for future in futures:
                    if future.done():
                        future.result()
                        raise RuntimeError("replica exited before both ready")
                if time.monotonic() >= deadline:
                    raise TimeoutError("pair readiness timed out")
                time.sleep(1)
            if native or args.policy == "inventory":
                manifest = pair / "inventory.json"
                members = []
                env = os.environ.copy()
                for idx in (6, 7):
                    name = f"AGENT_BENCH_{idx}"
                    address = (pair / f"gpu{idx}" / "address").read_text().strip()
                    members.append(
                        {
                            "node": "aws-a100",
                            "max_active_sequences": 8,
                            "gpu_count": 1,
                            "url": "http://" + address,
                            "bearer_env": name,
                        }
                    )
                    env[name] = (keydir / str(idx)).read_text()
                manifest.write_text(
                    json.dumps({"model": "m", "replicas": members}) + "\n"
                )

                def run_client(driver, output, *options):
                    subprocess.run(
                        [
                            sys.executable,
                            "-B",
                            root / driver,
                            manifest,
                            pair / output,
                            *options,
                        ],
                        env=env,
                        check=True,
                    )

                if args.policy == "early-rejection":
                    run_client("early_rejection_trial.py", "arrivals.json")
                elif recency:
                    run_client("recency_serving.py", "prefix-requests.json")
                elif burst:
                    flags = (
                        ["--protect-reader"] if args.policy == "protected-burst" else []
                    )
                    run_client("readiness_burst.py", "waves.json", *flags)
                elif natural_burst:
                    run_client("natural_burst.py", "natural.json", args.policy[-1])
                elif multi_router:
                    run_client(
                        "multi_router_trial.py",
                        "multi-router.json",
                        args.policy[-1],
                        *(
                            ["--citation-layout"]
                            if args.policy.startswith("citation-layout-")
                            else []
                        ),
                        *(
                            ["--citation-mixed"]
                            if args.policy.startswith(
                                ("citation-mixed-", "citation-layout-")
                            )
                            else []
                        ),
                        *(
                            ["--defer-mixed"]
                            if args.policy.startswith("defer-mixed-")
                            else []
                        ),
                    )
                else:
                    flags = []
                    if args.policy == "cross-router":
                        run_client("cross_router_trial.py", "cross-router.json")
                        flags = ["--policy", "observed-capacity"]
                    run_client(
                        "agent_fleet_replay.py",
                        "inventory-replay.json",
                        *flags,
                        "--waves",
                        "1",
                        "2",
                        "4",
                    )
                    if args.policy == "inventory":
                        run_client("probe_capacity_route.py", "capacity-route.json")
            else:
                driver = (
                    "probe_replica_wait.py"
                    if args.policy == "crossover"
                    else "replica_workflow.py"
                )
                extra = [] if args.policy == "crossover" else [args.policy]
                subprocess.run(
                    [
                        sys.executable,
                        "-B",
                        str(root / driver),
                        str(pair),
                        str(keydir),
                        *extra,
                    ],
                    check=True,
                )
        finally:
            (pair / "finished").touch()
        for future in futures:
            future.result()


if __name__ == "__main__":
    main()
