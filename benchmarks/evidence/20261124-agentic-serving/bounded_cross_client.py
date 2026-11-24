"""Submit a real single-workflow HTTP trial through a loopback SSH forward into a separate test host."""

import argparse
import json
import socket
import subprocess
from functools import partial
from pathlib import Path

from budgeted_tool_prefix import PromptProfiler, budgeted_preview
from coder_named_cpu_live import record_tool
from tool_wait_trial import measure


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("host")
    p.add_argument("remote_port", type=int)
    for n in ("profiler", "tokenizer", "registration", "out"):
        p.add_argument(n, type=Path)
    a = p.parse_args()
    cfg = json.loads(a.registration.read_text())
    [(seed, history)] = cfg["jobs"]
    receipts = []
    invocations = []
    with socket.socket() as addr:
        addr.bind(("127.0.0.1", 0))
        port = addr.getsockname()[1]
    forward = f"127.0.0.1:{port}:127.0.0.1:{a.remote_port}"
    command = [
        "ssh",
        "-o",
        "ExitOnForwardFailure=yes",
        "-o",
        "ConnectTimeout=10",
        "-L",
        forward,
        a.host,
        "echo ready && cat >/dev/null",
    ]
    with subprocess.Popen(
        command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
    ) as tunnel:
        try:
            assert tunnel.stdout.readline().strip() == "ready", (
                "SSH forward did not become ready"
            )
            with PromptProfiler(a.profiler, a.tokenizer) as native:
                trial = measure(
                    f"127.0.0.1:{port}",
                    True,
                    "cross-host",
                    seed,
                    cfg["delay_s"],
                    include_completed_call=True,
                    first_fixed_tool=True,
                    cancel_on_final=True,
                    document_pairs=2,
                    history=history,
                    execute_tool=partial(record_tool, seed, receipts, invocations),
                    preview_planner=partial(
                        budgeted_preview, budget=cfg["http_budget"], encode=native.ids
                    ),
                )
            (a.out / "client.json").write_text(
                json.dumps(
                    [
                        {
                            "work": trial["target"],
                            "receipts": receipts,
                            "invocations": invocations,
                        }
                    ]
                )
                + "\n"
            )
            (a.out / "external-client-preview.json").write_text(
                json.dumps(trial["prefix"]) + "\n"
            )
        finally:
            tunnel.stdin.close()
        if tunnel.wait(timeout=10) != 0:
            raise RuntimeError("SSH forward failed")


if __name__ == "__main__":
    main()
