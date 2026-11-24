"""Exercise the public client's reservation while the original final answer is still pending."""

import http.client
import threading
import unittest
from unittest.mock import patch
from concurrent.futures import ThreadPoolExecutor

import tool_wait_trial
from tool_wait_policy import ToolWaitPolicy


class ReleaseTest(unittest.TestCase):
    def case(self, failure, *, release_on_preview=True):
        available = threading.Event()
        final_in_flight = threading.Event()
        drain = threading.Event()
        end_preview = threading.Event()
        preview_returning = threading.Event()

        class Policy(ToolWaitPolicy):
            def finish(self, key, success):
                super().finish(key, success)
                available.set()

        def fake_stream(address, request, row, *, bearer=None, cancellation=None):
            if not end_preview.wait(3):
                raise TimeoutError("test preview timeout")
            preview_returning.set()
            if failure:
                raise http.client.RemoteDisconnected("interrupted preview")
            row["done_ns"] = 1

        def fake_replay(*args, on_tool_start, **kwargs):
            on_tool_start(
                1,
                {"messages": []},
                {"call": {"id": "c"}, "usage": {"prompt_tokens": 2000}},
            )
            final_in_flight.set()
            if not drain.wait(3):
                raise AssertionError("test final timeout")
            return {"ok": True}

        policy = Policy()
        with patch.object(tool_wait_trial, "stream", fake_stream), patch.object(
            tool_wait_trial, "replay", fake_replay
        ), ThreadPoolExecutor(1) as executor:
            result = executor.submit(
                tool_wait_trial.measure,
                "local",
                True,
                "test",
                1,
                0.5,
                policy=policy,
                estimated_wait_ms=500,
                **({"release_on_preview": True} if release_on_preview else {}),
            )
            try:
                self.assertTrue(final_in_flight.wait(2))
                self.assertEqual(
                    policy.reserve(("m", 2, 200), 500, 2000), "prefetch_in_flight"
                )
                end_preview.set()
                self.assertTrue(preview_returning.wait(2))
                if release_on_preview:
                    self.assertTrue(available.wait(2))
                    key = ("m", 1, 200)
                    self.assertEqual(
                        policy.reserve(key, 500, 2000),
                        "submit" if failure else "recent_prefix",
                    )
                    if failure:
                        policy.finish(key, False)
                    self.assertEqual(policy.reserve(("m", 2, 200), 500, 2000), "submit")
                    policy.finish(("m", 2, 200), False)
                else:
                    self.assertFalse(available.is_set())
                    self.assertEqual(
                        policy.reserve(("m", 2, 200), 500, 2000), "prefetch_in_flight"
                    )
                self.assertFalse(result.done())
            finally:
                end_preview.set()
                drain.set()
            self.assertEqual("error" in result.result()["prefix"], failure)

    def test_default_retains_reservation_through_final(self):
        self.case(False, release_on_preview=False)

    def test_completed_preview_releases_during_final(self):
        self.case(False)

    def test_disconnected_preview_releases_without_caching(self):
        self.case(True)


if __name__ == "__main__":
    unittest.main()
