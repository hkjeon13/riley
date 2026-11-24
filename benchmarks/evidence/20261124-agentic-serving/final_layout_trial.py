"""Replay the same agent tasks with tool eligibility disabled on every final turn."""

import argparse
import concurrent.futures
import json
import os
import threading
from pathlib import Path

from agent_workflow import replay, stream


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("address")
    p.add_argument("out", type=Path)
    p.add_argument("engine", choices=("riley", "vllm"))
    args = p.parse_args()
    keep = os.environ["FINAL_LAYOUT"] == "keep"
    output = []

    def send(address, request, row, first_visible):
        req = {**request, "return_token_ids": True}
        if mode := os.environ.get("FINAL_REPORT_MODE"):
            from final_report_presentation import present

            req = present(req, mode)
        elif args.engine == "riley" and keep and req["tool_choice"] == "none":
            if os.environ.get("FINAL_LAYOUT_NAMED"):
                req["include_tools_when_tool_choice_none"] = True
            else:
                req["chat_template_kwargs"] = {"tools": req["tools"]}
        return stream(address, req, row, first_visible)

    for history, seed in [(0, 761), (200, 761), (200, 18081), (200, 19959)]:
        group = []

        def run(
            name,
            offset,
            pairs,
            delay=0,
            event=None,
            gate=None,
            history=history,
            seed=seed,
        ):
            return replay(
                args.address,
                "m",
                seed + offset,
                pairs,
                delay,
                name,
                event,
                gate,
                history,
                send=send,
            )

        for name, delay in (("cold", 0), ("immediate", 0), ("delayed", 0.1)):
            group.append(run(name, 0, 2, delay))
        visible = threading.Event()
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            reader = pool.submit(run, "reader", 1, 2, 0, visible)
            writer = pool.submit(run, "long-tool-result", 2, 96, 0, None, visible)
            group.extend([reader.result(), writer.result()])
        output.append({"history": history, "seed": seed, "records": group})
        args.out.write_text(json.dumps(output) + "\n")
        print(
            history,
            seed,
            [(r["workflow_id"], r["ok"], r.get("error")) for r in group],
            flush=True,
        )


if __name__ == "__main__":
    main()
