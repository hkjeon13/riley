"""Bound speculative final-prefix work using hints available before a tool returns."""

import math
import threading
import time
from collections import OrderedDict


class ToolWaitPolicy:
    """One instance per replica and tenant trust domain; no prompt or KV leaves the server."""

    def __init__(
        self, *, minimum_wait_ms=350, minimum_prompt_tokens=1024, ttl_s=60, capacity=256
    ):
        if not all(
            math.isfinite(n) and n > 0
            for n in (minimum_wait_ms, minimum_prompt_tokens, ttl_s, capacity)
        ):
            raise ValueError(
                "policy thresholds and capacity must be finite and positive"
            )
        self.minimum_wait_ms = minimum_wait_ms
        self.minimum_prompt_tokens = minimum_prompt_tokens
        self.ttl_s = ttl_s
        self.capacity = capacity
        self.recent = OrderedDict()
        self.lock = threading.Lock()
        self.in_flight = None

    def reserve(self, context_key, estimated_wait_ms, prompt_tokens):
        """context_key identifies the application's immutable conversation prefix revision."""
        if context_key is None:
            raise ValueError("expected a context revision identifier")
        if (
            estimated_wait_ms is None
            or not math.isfinite(estimated_wait_ms)
            or estimated_wait_ms < self.minimum_wait_ms
        ):
            return "short_or_unknown_wait"
        if prompt_tokens < self.minimum_prompt_tokens:
            return "short_context"
        with self.lock:
            if self.recent.get(context_key, 0) > time.monotonic():
                return "recent_prefix"
            if self.in_flight is not None:
                return "prefetch_in_flight"
            self.in_flight = context_key
            return "submit"

    def finish(self, context_key, success):
        with self.lock:
            if self.in_flight is None or context_key != self.in_flight:
                raise ValueError("no matching outstanding prefix reservation")
            self.in_flight = None
            if success:
                self.recent[context_key] = time.monotonic() + self.ttl_s
                self.recent.move_to_end(context_key)
                if len(self.recent) > self.capacity:
                    self.recent.popitem(last=False)
