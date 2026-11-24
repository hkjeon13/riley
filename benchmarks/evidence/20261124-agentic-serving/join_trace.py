"""Join an isolated diagnostic server's request and iteration records to its SSE client."""

import argparse
import json
from pathlib import Path


def join(log, records, *, details=False):
    event = {}
    steps = []
    for line in log.splitlines():
        if not line.startswith("AGENT_"):
            continue
        if line.startswith("AGENT_STEP "):
            _, start, end, gpu_ns, items = line.split(maxsplit=4)
            steps.append((int(start), int(end), int(gpu_ns), json.loads(items)))
        else:
            name, req, start, end, *flags = line.split()
            key = int(req[5:], 16) if req.startswith("cmpl-") else int(req)
            request = event.setdefault(key, {})
            if name in request:
                raise ValueError(f"duplicate {key} {name}")
            request[name] = (int(start), int(end), flags)
    rows = []
    for r in records:
        if abs(r["end_epoch_offset_ns"] - r["epoch_offset_ns"]) > 1_000_000:
            raise ValueError("clock anchor drift exceeded 1ms")
        for turn in r["turns"]:
            req = int(turn["response_id"].removeprefix("cmpl-"), 16)
            stages = event[req]
            render, rendered, _ = stages["AGENT_RENDER"]
            submit, queued, flags = stages["AGENT_SUBMIT"]
            receive, admitted, _ = stages["AGENT_ADMIT"]
            worker_encode = stages.get("AGENT_ENCODE")
            encode = (
                queued - submit
                if flags == ["true"]
                else worker_encode[1] - worker_encode[0]
            )
            prefill = []
            decode = []
            for start, end, gpu_ns, items in steps:
                for engine_id, kind, count, position, output in items:
                    if engine_id == req:
                        (prefill if kind == "p" else decode).append(
                            (start, end, gpu_ns, count, position, output, len(items))
                        )
            cached = (
                turn["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0)
            )
            if not prefill or prefill[0][4] - prefill[0][3] != cached:
                raise ValueError(
                    "first computed position differs from HTTP cached tokens"
                )
            if (
                sum(step[3] for step in prefill) + cached
                != turn["usage"]["prompt_tokens"]
                or not prefill[-1][5]
                or not decode
            ):
                raise ValueError("incomplete GPU item coverage")
            if (
                not render
                <= rendered
                <= submit
                <= queued
                <= receive
                <= admitted
                <= prefill[0][0]
            ):
                raise ValueError("trace endpoints out of order")
            arrived = (turn["content_ns"] or [turn["last_call_fragment_ns"]])[0] + r[
                "epoch_offset_ns"
            ]
            rows.append(
                {
                    "workflow": r["workflow_id"],
                    "turn": turn["turn"],
                    "request_id": turn["response_id"],
                    "prefill_rows": sum(s[3] for s in prefill),
                    "decode_steps": len(decode),
                    "prefill_steps": len(prefill),
                    "shared_prefill_steps": sum(s[-1] > 1 for s in prefill),
                    "normalize_ms": (rendered - render) / 1e6,
                    "tokenizer_path_ms": encode / 1e6,
                    "command_queue_ms": (receive - queued) / 1e6,
                    "admit_to_first_prefill_ms": (prefill[0][0] - admitted) / 1e6,
                    "prefill_participating_gpu_ms": sum(s[2] for s in prefill) / 1e6,
                    "decode_participating_gpu_ms": sum(s[2] for s in decode) / 1e6,
                    "prefill_end_to_first_client_event_ms": (arrived - prefill[-1][1])
                    / 1e6,
                }
            )
            if details:
                sent = turn["sent_ns"] + r["epoch_offset_ns"]
                span = prefill[-1][1] - prefill[0][0]
                execute = sum(end - start for start, end, *_ in prefill)
                if execute > span:
                    raise ValueError("overlapping prefill step times")
                rows[-1].update(
                    client_to_render_ms=(render - sent) / 1e6,
                    render_to_submit_after_normalize_ms=(submit - rendered) / 1e6,
                    submit_to_queued_ms=(queued - submit) / 1e6,
                    admission_ms=(admitted - receive) / 1e6,
                    tokenizer_on_submit=flags == ["true"],
                    prefill_span_ms=span / 1e6,
                    prefill_execute_wall_ms=execute / 1e6,
                    prefill_between_steps_ms=(span - execute) / 1e6,
                    chunks=[
                        {
                            "rows": s[3],
                            "end_position": s[4],
                            "members": s[-1],
                            "gpu_ms": s[2] / 1e6,
                            "execute_ms": (s[1] - s[0]) / 1e6,
                        }
                        for s in prefill
                    ],
                )
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("records", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            join(args.log.read_text(), json.loads(args.records.read_text())), indent=2
        )
    )
