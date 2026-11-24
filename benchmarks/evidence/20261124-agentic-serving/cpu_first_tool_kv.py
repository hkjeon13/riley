"""CPU checkpoint experiment. Own one bounded, disposable prefix per sequential session."""

import subprocess
import threading
import time

from agent_workflow import fingerprint
from tool_wait_trial import prepare_prefix


class FirstToolKV:
    def __init__(self, model, encode, config):
        self.model = model
        self.encode = encode
        self.config = config
        self.worker = None
        self.stop = threading.Event()
        self.records = []

    def start(self, index, request, row):
        if index != 0:
            return
        assert self.worker is None and row["dispatch"] == "application"
        self.stop.clear()
        self.ids = []
        self.kv = None
        meta = {"decision_ns": time.monotonic_ns(), "commits": []}
        self.records.append(meta)
        history_bytes = sum(
            len(m["content"].encode())
            for m in request["messages"]
            if isinstance(m.get("content"), str)
        )
        meta["history_bytes"] = history_bytes
        if history_bytes < self.config["history_bytes"]:
            meta["decision"] = "short_context"
            return
        partial = prepare_prefix(request, row["call"], include_completed_call=True)
        meta.update(
            decision="prepare", request=partial, input_sha256=fingerprint(partial)
        )
        self.worker = threading.Thread(target=self._prepare, args=(partial, meta))
        self.worker.start()

    def _prepare(self, partial, meta):
        import torch

        try:
            self.ids = self.encode(partial)
            meta["preview_ids_sha256"] = fingerprint(self.ids)
            meta["preview_tokens"] = len(self.ids)
            chunk = self.config["chunk"]
            limit = min(len(self.ids), self.config["maximum_tokens"])
            with torch.inference_mode():
                for pos in range(0, limit - chunk + 1, chunk):
                    if self.stop.is_set():
                        break
                    started = time.monotonic_ns()
                    hint = self.config.get("wait_hint_ms")
                    if hint is not None:
                        deadline = meta["decision_ns"] + hint * 1_000_000
                        if (
                            started + self.config["chunk_budget_ms"] * 1_000_000
                            > deadline
                        ):
                            meta["stop_reason"] = "insufficient_predicted_time"
                            break
                    result = self.model.model(
                        input_ids=torch.tensor([self.ids[pos : pos + chunk]]),
                        past_key_values=self.kv,
                        use_cache=True,
                    )
                    self.kv = result.past_key_values
                    end = time.monotonic_ns()
                    assert self.kv.get_seq_length() == pos + chunk
                    meta["commits"].append([started, end, pos + chunk])
        except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
            meta["error"] = repr(error)
            self.kv = None

    def stop_preparation(self):
        self.stop.set()

    def take(self, ids):
        meta = self.records[-1]
        begin = time.monotonic_ns()
        self.stop_preparation()
        if self.worker is not None:
            self.worker.join()
            self.worker = None
        meta["joined_ns"] = time.monotonic_ns()
        meta["join_started_ns"] = begin
        common = next(
            (i for i, (a, b) in enumerate(zip(self.ids, ids)) if a != b),
            min(len(self.ids), len(ids)),
        )
        count = self.kv.get_seq_length() if self.kv is not None else 0
        block = self.config["block"]
        reused = min(count, common, len(ids) - 1) // block * block
        meta.update(
            common_tokens=common, computed=count, reused=reused, final_tokens=len(ids)
        )
        cache, self.kv = self.kv, None
        if reused:
            if reused < count:
                cache.crop(reused - count)
            assert cache.get_seq_length() == reused
            return cache
        return None
