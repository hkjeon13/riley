"""A truncated stream must not produce a successful benchmark request."""
import asyncio
import hashlib
import importlib.util
import json
import time
import unittest
from pathlib import Path
from unittest import mock

SOURCE = Path(__file__).resolve().parents[2] / "live/load.py"
SPEC = importlib.util.spec_from_file_location("serving_load", SOURCE)
loader = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(loader)


class StreamingLoadTests(unittest.IsolatedAsyncioTestCase):
    async def request(self, *, done=True, finish="stop", usage=None, content=("a",), capture_hashes=False):
        reader = asyncio.StreamReader()
        chunks = [{"choices": [{"delta": {"content": text}, "finish_reason": None}]} for text in content]
        chunks.append({"choices": [{"delta": {}, "finish_reason": finish}],
                       "usage": usage if usage is not None else {"prompt_tokens": 10, "completion_tokens": 1}})
        stream = b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n"
        stream += b"".join(b"data: " + json.dumps(chunk).encode() + b"\n\n" for chunk in chunks)
        if done:
            stream += b"data: [DONE]\n\n"
        reader.feed_data(stream)
        reader.feed_eof()
        writer = mock.Mock()
        writer.drain = mock.AsyncMock()
        writer.wait_closed = mock.AsyncMock()
        row = {}
        with mock.patch.object(loader.asyncio, "open_connection", new=mock.AsyncMock(return_value=(reader, writer))):
            await loader.one("localhost", 1, {}, time.perf_counter(), row, capture_hashes=capture_hashes)
        writer.wait_closed.assert_awaited_once()
        return row

    async def test_complete_stream_is_measured(self):
        self.assertIs((await self.request())["error"], False)

    async def test_measured_output_digest_preserves_utf8_content_across_sse_chunks(self):
        intact = await self.request(content=("한글",), capture_hashes=True)
        split = await self.request(content=("한", "글"), capture_hashes=True)
        self.assertEqual(intact["output_sha256"], hashlib.sha256("한글".encode()).hexdigest())
        self.assertEqual(intact["output_sha256"], split["output_sha256"])
        self.assertEqual(intact["input_sha256"], split["input_sha256"])
        self.assertNotIn("output_sha256", await self.request(done=False, capture_hashes=True))

    async def test_eof_after_usage_is_not_a_completed_stream(self):
        self.assertIs((await self.request(done=False))["error"], True)

    async def test_missing_finish_or_invalid_usage_is_rejected(self):
        self.assertIs((await self.request(finish=None))["error"], True)
        for usage in ({}, {"prompt_tokens": 10, "completion_tokens": True},
                      {"prompt_tokens": -1, "completion_tokens": 1}):
            with self.subTest(usage=usage):
                self.assertIs((await self.request(usage=usage))["error"], True)

    def test_unfilled_request_record_is_a_failure(self):
        result = loader.summarize([{}], 1, {})
        self.assertEqual(result["ok"], 0)
        self.assertEqual(result["errors"], 1)
