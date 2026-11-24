"""Submit final-format prefix work during a real tool wait and measure the full chain."""

import argparse
import concurrent.futures
import http.client
import json
import threading
import time
from contextlib import nullcontext
from functools import partial
from pathlib import Path

from agent_workflow import StreamCancellation, replay, stream


def prepare_prefix(request, completed_call, *, include_completed_call=False):
    """Snapshot the final-turn prefix while the actual tool result is pending."""
    messages = list(request["messages"])
    if include_completed_call:
        messages.extend(
            [
                {"role": "assistant", "content": None, "tool_calls": [completed_call]},
                {"role": "tool", "tool_call_id": completed_call["id"], "content": ""},
            ]
        )
    return {**request, "messages": messages, "tool_choice": "none", "max_tokens": 1}


def measure(
    address,
    enabled,
    label,
    seed,
    delay,
    peer_seed=None,
    *,
    policy=None,
    estimated_wait_ms=None,
    include_completed_call=False,
    release_on_preview=False,
    first_fixed_tool=False,
    cancel_on_final=False,
    preview_planner=None,
    preview_pool=None,
    document_pairs=96,
    history=200,
    model="m",
    bearer=None,
    target_sender=None,
    execute_tool=None,
):
    if preview_pool is not None and preview_planner is None:
        raise ValueError("replica preview reservations require an exact token planner")
    grant = 0
    visible = threading.Event()
    prefix_row = {}
    preview = None
    cancellation = StreamCancellation() if cancel_on_final else None
    decision = "forced_on" if enabled else "control_off"
    context_key = (model, seed, history)
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        peer = None
        if peer_seed is not None:
            # Keep the peer at its final call until the target's second tool starts.
            release = threading.Event()
            peer = pool.submit(
                replay,
                address,
                model,
                peer_seed,
                2,
                0,
                "reader",
                visible,
                release,
                200,
                send=partial(stream, bearer=bearer),
            )

        def start(index, request, row):
            nonlocal preview, decision, grant
            if index != (0 if first_fixed_tool else 1):
                return
            if peer is not None:
                release.set()
                if not visible.wait(45):
                    raise ValueError("reader did not start streaming")
            row["tool_wait_begin_ns"] = time.monotonic_ns()
            if enabled and policy is not None:
                usage = row.get("usage")
                decision = (
                    policy.reserve(
                        context_key, estimated_wait_ms, usage["prompt_tokens"]
                    )
                    if usage
                    else "unknown_prompt_length"
                )
            if enabled and decision in ("forced_on", "submit"):
                if preview_pool is not None:
                    grant, reason = preview_pool.reserve()
                    prefix_row["reserved_blocks"] = grant
                    if grant == 0:
                        decision = reason
                        if policy is not None:
                            policy.finish(context_key, False)
                        return
                prepared = prepare_prefix(
                    request, row["call"], include_completed_call=include_completed_call
                )

                def run_preview():
                    success = False
                    try:
                        selected = prepared
                        if preview_planner is not None:
                            prefix_row["plan_started_ns"] = time.monotonic_ns()
                            selected, receipt = preview_planner(
                                prepared,
                                **(
                                    {"budget": 16 * grant}
                                    if preview_pool is not None
                                    else {}
                                ),
                                cancelled=lambda: (
                                    cancellation is not None and cancellation.cancelled
                                ),
                            )
                            prefix_row["plan_finished_ns"] = time.monotonic_ns()
                            prefix_row["plan"] = receipt
                            if selected is None:
                                return
                        stream(
                            address,
                            selected,
                            prefix_row,
                            bearer=bearer,
                            cancellation=cancellation,
                        )
                        if preview_planner is not None:
                            observed = prefix_row["usage"]["prompt_tokens"]
                            ids_hash = prefix_row.get("prompt_ids_sha256")
                            if observed != receipt["planned_tokens"] or (
                                ids_hash is not None
                                and ids_hash != receipt["ids_sha256"]
                            ):
                                prefix_row["tokenizer_mismatch"] = True
                                if preview_pool is not None:
                                    preview_pool.tokenizer_mismatch()
                                raise ValueError(
                                    "preview input disagrees with serving checkpoint"
                                )
                        success = "done_ns" in prefix_row
                    except (OSError, ValueError, http.client.HTTPException) as error:
                        prefix_row["error"] = str(error)
                    finally:
                        if policy is not None and release_on_preview:
                            policy.finish(context_key, success)

                try:
                    preview = pool.submit(run_preview)
                except RuntimeError:
                    if policy is not None:
                        policy.finish(context_key, False)
                    raise

        def stop_preview(reason):
            if (
                cancellation is not None
                and preview is not None
                and not preview.done()
                and "cancel_requested_ns" not in prefix_row
            ):
                prefix_row["cancel_requested_ns"] = time.monotonic_ns()
                prefix_row["cancel_reason"] = reason
                cancellation.cancel()

        def send_target(address, request, row, event):
            if row["turn"] == 2:
                stop_preview("final_submission")
            if target_sender is not None:
                return target_sender(address, request, row, event)
            return stream(address, request, row, event, bearer=bearer)

        try:
            target = replay(
                address,
                model,
                seed,
                document_pairs,
                delay,
                label,
                history=history,
                on_tool_start=start,
                execute_tool=execute_tool,
                send=send_target,
                dispatch_fixed_tools=first_fixed_tool,
            )
        finally:
            stop_preview("workflow_ended")
            if peer is not None:
                release.set()
            try:
                if preview is not None:
                    try:
                        preview.result()
                    finally:
                        if policy is not None and not release_on_preview:
                            policy.finish(
                                context_key,
                                "done_ns" in prefix_row and "error" not in prefix_row,
                            )
            finally:
                if grant:
                    preview_pool.release(grant)
        result = {
            "case": label,
            "decision": decision,
            "estimated_wait_ms": estimated_wait_ms,
            "enabled": enabled,
            "target": target,
            "prefix": prefix_row,
        }
        if peer is not None:
            result["peer"] = peer.result()
        return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("address")
    p.add_argument("mode", choices=["on", "off"])
    p.add_argument("out", type=Path)
    p.add_argument(
        "--include-completed-call",
        action="store_true",
        help="Include the validated tool call and an empty tool result in the preview.",
    )
    p.add_argument(
        "--first-fixed-tool",
        action="store_true",
        help="execute both forced tools in the application and begin optional preparation at the first one",
    )
    p.add_argument(
        "--cancel-on-final",
        action="store_true",
        help="interrupt an unfinished preview before sending the final request",
    )
    p.add_argument(
        "--preview-token-budget",
        type=int,
        help="maximum input tokens for an optional prefix request",
    )
    p.add_argument(
        "--prompt-profiler",
        type=Path,
        help="Riley qwen_prompt_profile release executable",
    )
    p.add_argument(
        "--tokenizer-dir", type=Path, help="serving checkpoint tokenizer and template"
    )
    p.add_argument(
        "--llama3",
        action="store_true",
        help="use the native Llama 3 tokenizer for planning",
    )
    args = p.parse_args()
    options = (args.preview_token_budget, args.prompt_profiler, args.tokenizer_dir)
    if any(v is not None for v in options) and not all(v is not None for v in options):
        p.error(
            "a bounded preview requires --preview-token-budget, --prompt-profiler and --tokenizer-dir"
        )
    if args.preview_token_budget is not None and args.preview_token_budget < 1:
        p.error("--preview-token-budget must be positive")
    if args.llama3 and args.tokenizer_dir is None:
        p.error("--llama3 requires a checkpoint for bounded preview planning")

    # Seeds repeat only for the deliberate warm exact-history control.
    cases = [
        ("zero", 1901, 0, None),
        ("short", 1918, 0.1, None),
        ("long", 1935, 0.5, None),
        ("warm", 1935, 0.5, None),
        ("overlap", 1952, 0.5, 1969),
    ]
    session = nullcontext()
    if args.preview_token_budget is not None and args.mode == "on":
        from budgeted_tool_prefix import PromptProfiler

        session = PromptProfiler(
            args.prompt_profiler, args.tokenizer_dir, llama3=args.llama3
        )
    with session as profiler:
        planner = None
        if profiler is not None:
            from budgeted_tool_prefix import budgeted_preview

            planner = partial(
                budgeted_preview, budget=args.preview_token_budget, encode=profiler.ids
            )
        data = []
        for spec in cases:
            data.append(
                measure(
                    args.address,
                    args.mode == "on",
                    *spec,
                    include_completed_call=args.include_completed_call,
                    first_fixed_tool=args.first_fixed_tool,
                    cancel_on_final=args.cancel_on_final,
                    preview_planner=planner,
                )
            )
            args.out.write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    main()
