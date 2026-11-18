"""Closed-stream and latency window contracts for the live mixed soak."""
import asyncio
import json
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch

with patch.object(sys, "path", [str(Path(__file__).resolve().parents[2] / "live"), *sys.path]):
    import soak


def frame(delta, finish=None):
    return {"choices": [{"delta": delta, "finish_reason": finish}]}


class MixedSoakTests(unittest.IsolatedAsyncioTestCase):
    async def read(self, frames, *, kind="plain", done=True, status=200):
        data = f"HTTP/1.1 {status} Status\r\n\r\n".encode()
        data += b"".join(b"data: " + json.dumps(chunk).encode() + b"\n\n" for chunk in frames)
        if done:
            data += b"data: [DONE]\n\n"
        reader = asyncio.StreamReader()
        reader.feed_data(data)
        reader.feed_eof()
        return await soak.read_completion(reader, kind, 64, time.perf_counter())

    def frames(self, text="hi", finish="stop"):
        return [frame({"content": text}), frame({}, finish),
                {"usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11}}]

    async def test_success_requires_usage_finish_and_done(self):
        frames = self.frames()
        self.assertEqual((await self.read(frames))["outcome"], "completed")
        with self.assertRaisesRegex(ValueError, "without .*DONE"):
            await self.read(frames, done=False)
        with self.assertRaisesRegex(ValueError, "missing or invalid SSE finish"):
            await self.read(self.frames(finish=None))
        with self.assertRaisesRegex(ValueError, "invalid SSE usage"):
            await self.read(frames[:-1])

    async def test_real_disconnect_is_distinct_from_early_completion_and_eof(self):
        events = [frame({"content": str(index)}) for index in range(4)]
        self.assertEqual((await self.read(events, kind="cancel", done=False))["outcome"], "disconnected")
        self.assertEqual((await self.read(self.frames(), kind="cancel"))["outcome"], "completed")
        with self.assertRaisesRegex(ValueError, "without .*DONE"):
            await self.read(events[:-1], kind="cancel", done=False)

    async def test_overload_is_visible_to_the_gate(self):
        self.assertEqual(await self.read([], status=429), {"outcome": "rejected", "http_status": 429})

    async def test_schema_and_tool_terminals_require_valid_payloads(self):
        self.assertEqual((await self.read(self.frames('{"title":"T","year":2000}'), kind="schema"))["finish"], "stop")
        with self.assertRaisesRegex(ValueError, "invalid completed schema"):
            await self.read(self.frames('{"title":"T","year":true}'), kind="schema")
        tool_call = {"index": 0, "function": {"name": "get_weather", "arguments": '{"city":"Seoul"}'}}
        frames = [frame({"tool_calls": [tool_call]}, "tool_calls"), self.frames()[-1]]
        self.assertEqual((await self.read(frames, kind="tools"))["finish"], "tool_calls")
        with self.assertRaisesRegex(ValueError, "required tool call is missing"):
            await self.read(self.frames(), kind="tools")

    def test_latency_uses_scheduled_windows_and_counts_missing_visible_tokens(self):
        rows = [{"outcome": "completed", "scheduled": 30, "ttft_ms": 20, "e2e_ms": 100},
                {"outcome": "completed", "scheduled": 80, "ttft_ms": None, "e2e_ms": 140},
                {"outcome": "disconnected", "scheduled": 85}]
        windows, drift = soak.latency_windows(rows, 0, 100)
        self.assertEqual(windows["late"]["completed"], 1)
        self.assertEqual(windows["late"]["ttft_ms_count"], 0)
        self.assertIsNone(drift["ttft_ms"])
        self.assertAlmostEqual(drift["e2e_ms"], 40)
