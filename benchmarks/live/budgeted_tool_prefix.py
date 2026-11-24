"""Plan an optional chat preview inside a prompt-token budget using Riley's exact renderer."""

import json
import subprocess
import threading
from contextlib import AbstractContextManager

from agent_workflow import fingerprint


class PromptProfiler(AbstractContextManager):
    """Keep the checkpoint tokenizer warm across tool waits; serialize one request/response pair."""

    def __init__(self, executable, tokenizer, *, llama3=False):
        self.command = [executable, tokenizer, "--interactive"]
        if llama3:
            self.command.append("--llama3")
        self.lock = threading.Lock()

    def __enter__(self):
        self.proc = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )
        try:
            if json.loads(self.proc.stdout.readline()) != {"ready": True}:
                raise ValueError("unexpected native chat profiler handshake")
        except (OSError, ValueError):
            self.__exit__(None, None, None)
            raise
        return self

    def ids(self, req):
        with self.lock:
            self.proc.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
            self.proc.stdin.flush()
            line = self.proc.stdout.readline()
            if not line:
                raise ValueError(
                    "native chat profiler exited before returning token IDs"
                )
            result = json.loads(line)
            if "error" in result:
                raise ValueError(result["error"])
            return result["ids"]

    def __exit__(self, *exc):
        try:
            self.proc.stdin.close()
        except BrokenPipeError:
            pass
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
        finally:
            self.proc.stdout.close()


def budgeted_preview(request, budget, encode, *, cancelled=lambda: False):
    """Select a fitting known message prefix. A skip still permits the complete final call."""
    if type(budget) is not int or budget < 1:
        raise ValueError("preview budget must be a positive number of prompt tokens")
    receipt = {"budget_tokens": budget, "renders": 0}
    if cancelled():
        return None, {**receipt, "decision": "cancelled"}
    full = encode(request)
    receipt.update(renders=1, full_tokens=len(full))
    selection = None
    selected_ids = []
    shared = 0
    if len(full) <= budget:
        selection, selected_ids, shared = request, full, len(full)
    else:
        # The first user message can follow fixed system/developer instructions.
        # Preserve those verbatim. Avoid editing any tool data or structured content.
        messages = request["messages"]
        index = next(
            (
                i
                for i, m in enumerate(messages)
                if m["role"] not in ("system", "developer")
            ),
            len(messages),
        )
        if (
            index < len(messages)
            and messages[index]["role"] == "user"
            and isinstance(messages[index].get("content"), str)
        ):
            message = messages[index]
            content = message["content"]
            # Match the registered 512-character slices. Validate every accepted
            # candidate independently: BPE length need not be strictly monotonic.
            lo, hi = 1, (len(content) - 1) // 512
            while lo <= hi:
                if cancelled():
                    return None, {**receipt, "decision": "cancelled"}
                middle = (lo + hi) // 2
                partial = {
                    **request,
                    "messages": [
                        *messages[:index],
                        {**message, "content": content[: middle * 512]},
                    ],
                    "add_generation_prompt": False,
                }
                ids = encode(partial)
                receipt["renders"] += 1
                if len(ids) > budget:
                    hi = middle - 1
                else:
                    same = next(
                        (i for i, (x, y) in enumerate(zip(ids, full)) if x != y),
                        min(len(ids), len(full)),
                    )
                    if same > shared:
                        selection, selected_ids, shared = partial, ids, same
                    lo = middle + 1
    if cancelled():
        return None, {**receipt, "decision": "cancelled"}
    if shared < 16:
        return None, {**receipt, "decision": "no_matching_block"}
    receipt.update(
        decision="full" if selection is request else "partial",
        planned_tokens=len(selected_ids),
        known_shared_blocks=shared // 16,
        ids_sha256=fingerprint(selected_ids),
    )
    return selection, receipt


class ReplicaPreviewBudget:
    """Bound outstanding preview reservations for clients sharing one replica and trust domain."""

    def __init__(self, blocks, per_request_blocks):
        if not all(type(v) is int and v > 0 for v in (blocks, per_request_blocks)):
            raise ValueError(
                "preview pool and per-request limits must be positive block counts"
            )
        self.blocks = blocks
        self.per_request_blocks = per_request_blocks
        self.remaining = blocks
        self.peak = 0
        self.mismatch = False
        self.lock = threading.Lock()

    def reserve(self):
        with self.lock:
            if self.mismatch:
                return 0, "replica_preview_tokenizer_mismatch"
            grant = min(self.remaining, self.per_request_blocks)
            self.remaining -= grant
            self.peak = max(self.peak, self.blocks - self.remaining)
            return grant, ("submit" if grant else "replica_preview_budget_exhausted")

    def tokenizer_mismatch(self):
        with self.lock:
            self.mismatch = True

    def release(self, blocks):
        with self.lock:
            if not 0 <= blocks <= self.blocks - self.remaining:
                raise ValueError("invalid preview reservation release")
            self.remaining += blocks


class FleetPreviews:
    def __init__(self, profiler, replicas, blocks, per_request_blocks):
        self.profiler = profiler
        self.pools = [
            ReplicaPreviewBudget(blocks, per_request_blocks) for _ in range(replicas)
        ]

    def workflow(
        self, router, session, model, seed, pairs, label, dispatch_fixed_tools
    ):
        from functools import partial

        from tool_wait_trial import measure

        replica = session % len(self.pools)
        address, bearer = router.endpoints[replica]
        trial = measure(
            address,
            True,
            label,
            seed,
            0.1,
            model=model,
            document_pairs=pairs,
            history=100,
            first_fixed_tool=dispatch_fixed_tools,
            include_completed_call=True,
            cancel_on_final=True,
            bearer=bearer,
            target_sender=router.sender(session),
            preview_pool=self.pools[replica],
            preview_planner=partial(budgeted_preview, encode=self.profiler.ids),
        )
        work = trial["target"]
        work["preview"] = {
            "replica": replica,
            "decision": trial["decision"],
            **trial["prefix"],
        }
        return work

    def summary(self):
        return [
            {
                "blocks": p.blocks,
                "per_workflow_blocks": p.per_request_blocks,
                "peak_reserved": p.peak,
                "remaining_after_drain": p.remaining,
                "tokenizer_mismatch": p.mismatch,
            }
            for p in self.pools
        ]
