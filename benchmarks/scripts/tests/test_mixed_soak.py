"""Closed-stream and latency window contracts for the live mixed soak."""
import asyncio
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, mock_open, patch

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

    def test_rss_growth_uses_first_second_half_sample_and_drain(self):
        samples = [{"at_seconds": 899.9, "rss_mib": 10},
                   {"at_seconds": 900, "rss_mib": 14},
                   {"at_seconds": 1200, "rss_mib": 14.5}]
        self.assertEqual(soak.rss_growth_second_half(samples, {"rss_mib": 14.75}, 1800), 0.75)
        self.assertEqual(soak.rss_growth_second_half(samples, {"rss_mib": 14}, 1800), 0)
        self.assertIsNone(soak.rss_growth_second_half(samples[:1], {"rss_mib": 14}, 1800))

    def test_smaps_growth_is_visible_with_an_unchanged_status_counter(self):
        start = {"rss_kib": 15_544_012, "anonymous_kib": 15_358_440}
        end = {"rss_kib": 15_544_712, "anonymous_kib": 15_359_140}
        samples = [{"at_seconds": 899, "rss_mib": 15_161.46},
                   {"at_seconds": 900, "rss_mib": 15_161.46, "smaps_rollup": start}]
        final = {"rss_mib": 15_161.46, "smaps_rollup": end}
        self.assertEqual(soak.rss_growth_second_half(samples, final, 1800), 0)
        precise = soak.smaps_growth_second_half(samples, final, 1800)
        self.assertEqual(precise["rss_growth_kib"], 700)
        self.assertEqual(precise["anonymous_growth_kib"], 700)
        self.assertIsNone(soak.smaps_growth_second_half(samples[:1], final, 1800)["rss_growth_kib"])
        self.assertIsNone(soak.smaps_growth_second_half(samples, {}, 1800)["rss_growth_kib"])

    def test_smaps_reader_handles_unavailable_or_malformed_measurements(self):
        with patch("builtins.open", mock_open(read_data="Rss: 1024 kB\nAnonymous: 1000 kB\n")):
            sample = soak.read_smaps_rollup(42)
        self.assertEqual((sample["rss_kib"], sample["anonymous_kib"]), (1024, 1000))
        for contents in ("Anonymous: 3 kB\n", "Rss: 5 kB\nAnonymous: 6 kB\n", "Rss: 5 MB\n"):
            with self.subTest(contents=contents), patch("builtins.open", mock_open(read_data=contents)):
                sample = {}
                soak.record_smaps_rollup(sample, 42)
                self.assertIsNone(sample["smaps_rollup"])
                self.assertIn("smaps_rollup_error", sample)
        with patch("builtins.open", side_effect=PermissionError("smaps not readable")):
            sample = {}
            soak.record_smaps_rollup(sample, 42)
            self.assertIsNone(sample["smaps_rollup"])
            self.assertEqual(sample["smaps_rollup_error"], "smaps not readable")

    async def test_precise_growth_fails_the_soak_despite_flat_status_rss(self):
        args = SimpleNamespace(address="localhost:1", model="test", pid=42, seconds=120, rate=1,
                               shared_prefix=False, smoke=True, require_zero_rss_growth=True)
        metrics = dict.fromkeys(soak.IDLE_GAUGES, 0)
        metrics["allocation"] = {}
        metrics["counters"] = {"overloads": 0, "disconnects": 0, "cancellations": 0}
        final_metrics = dict(metrics, counters={"overloads": 0, "disconnects": 1, "cancellations": 1})
        baseline = {"rss_kib": 15_544_012, "anonymous_kib": 15_358_440}
        final = {"rss_kib": 15_544_712, "anonymous_kib": 15_359_140}
        snapshots = [
            {"at_seconds": 0, "rss_mib": 15_161.46, "metrics": metrics},
            {"at_seconds": 10, "rss_mib": 15_161.46, "metrics": metrics},
            {"at_seconds": 60, "rss_mib": 15_161.46, "metrics": metrics, "smaps_rollup": baseline},
            {"at_seconds": 120, "rss_mib": 15_161.46, "metrics": final_metrics},
        ]

        async def response(_host, _port, kind, _body, scheduled):
            return {"kind": kind, "scheduled": scheduled, "dispatch_ms": 0, "ttft_ms": 1, "e2e_ms": 1,
                    "outcome": "disconnected" if kind == "cancel" else "completed",
                    "finish": "tool_calls" if kind == "tools" else "stop"}

        requests = [(kind, {}) for kind in (*soak.KINDS, *soak.KINDS[:4])]
        with (patch.object(soak.asyncio, "sleep", new_callable=AsyncMock),
              patch.object(soak.random.Random, "expovariate", return_value=10),
              patch.object(soak, "request", side_effect=requests),
              patch.object(soak, "run_request", side_effect=response),
              patch.object(soak, "snapshot", side_effect=snapshots),
              patch.object(soak, "read_smaps_rollup", return_value=final)):
            result = await soak.soak(args)
        self.assertEqual(result["rss_growth_second_half_mib"], 0)
        self.assertEqual(result["smaps_rollup_second_half"]["rss_growth_kib"], 700)
        self.assertEqual([name for name, passed in result["checks"].items() if not passed],
                         ["rss_growth_second_half_at_most_zero"])
        self.assertEqual(result["status"], "failed")

    def test_container_rss_uses_pid_one_and_rejects_a_changed_host_process(self):
        container = "a" * 64
        binding = f"{container} 42 0\n"
        smaps = "Rss: 1500 kB\nAnonymous: 800 kB\n"
        with patch.object(soak.subprocess, "check_output", side_effect=[binding, smaps, binding]) as run:
            sample = soak.read_smaps_rollup(42, open_proc=soak.partial(soak.open_container_proc, container))
            self.assertEqual(sample["rss_kib"], 1500)
            self.assertEqual(sample["anonymous_kib"], 800)
            self.assertEqual(run.call_args_list[1].args[0],
                             ["docker", "exec", container, "cat", "/proc/1/smaps_rollup"])

        with patch.object(soak.subprocess, "check_output", side_effect=[binding, smaps, f"{container} 43 1\n"]):
            failed = {}
            soak.record_smaps_rollup(failed, 42, open_proc=soak.partial(soak.open_container_proc, container))
            self.assertIsNone(failed["smaps_rollup"])
            self.assertIn("identity changed", failed["smaps_rollup_error"])
