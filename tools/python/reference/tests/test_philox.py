from __future__ import annotations

import unittest

from riley_reference.constants import RNG_ALGORITHM, UINT64_MAX
from riley_reference.philox import (
    Philox4x32,
    derive,
    philox4x32_10,
    uniform_open01,
)


class PhiloxContractTests(unittest.TestCase):
    def test_raw_known_answer_vectors(self) -> None:
        self.assertEqual(
            philox4x32_10((0, 0, 0, 0), (0, 0)),
            (0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8),
        )
        self.assertEqual(
            philox4x32_10((0xFFFFFFFF,) * 4, (0xFFFFFFFF,) * 2),
            (0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD),
        )

    def test_derivation_draw_snapshot_restore_and_fork(self) -> None:
        rng = derive(42, "request-0001", "token-sampling")
        self.assertEqual(
            rng.digest.hex(),
            "c9826cff0d3267e8597dd2f16928ca083e76df3b8fc4af9810c747236deb517a",
        )
        self.assertEqual(rng.key, (0xFF6C82C9, 0xE867320D))
        self.assertEqual(rng.nonce, (0xF1D27D59, 0x08CA2869))
        self.assertEqual(rng.next_u32(), 0xC875248D)
        self.assertEqual(rng.next_u32(), 0x7D889EA1)
        snapshot = rng.snapshot()
        self.assertEqual(snapshot["algorithm_id"], RNG_ALGORITHM)
        self.assertEqual(snapshot["block"], "0")
        self.assertEqual(snapshot["word_offset"], 2)
        restored = Philox4x32.restore(snapshot)
        self.assertEqual(restored.next_u32(), 0xD6887282)
        self.assertEqual(restored.next_u32(), 0x6DAF5198)
        self.assertEqual(restored.snapshot()["block"], "1")
        self.assertEqual(restored.snapshot()["word_offset"], 0)

        child = rng.fork("draft")
        self.assertEqual(
            child.digest.hex(),
            "c266fe1f4647900fd7c88a368c5b9dde897d4d0c49bc5378dbb66dda9c1c84f1",
        )
        self.assertEqual(
            tuple(child.next_u32() for _ in range(4)),
            (0xF58F7119, 0x0CBF7DC4, 0x8531E25F, 0x50267C0D),
        )

    def test_uniform_open_interval_endpoints(self) -> None:
        self.assertEqual(uniform_open01(0), 2.0**-33)
        self.assertEqual(uniform_open01(0xFFFFFFFF), 1.0 - 2.0**-33)

    def test_terminal_counter_never_wraps(self) -> None:
        root = derive(0, b"request", b"token-sampling")
        terminal = Philox4x32(root.digest, UINT64_MAX, 3, False)
        terminal.next_u32()
        self.assertTrue(terminal.exhausted)
        self.assertEqual(terminal.snapshot()["block"], str(UINT64_MAX))
        with self.assertRaises(OverflowError):
            terminal.next_u32()

    def test_restore_rejects_noncanonical_state(self) -> None:
        snapshot = derive(0, "r", "token-sampling").snapshot()
        snapshot["block"] = "00"
        with self.assertRaises(ValueError):
            Philox4x32.restore(snapshot)

    def test_request_streams_are_isolated_under_interleaved_draws(self) -> None:
        isolated_a = derive(7, "request-a", "token-sampling")
        isolated_b = derive(7, "request-b", "token-sampling")
        expected_a = tuple(isolated_a.next_u32() for _ in range(9))
        expected_b = tuple(isolated_b.next_u32() for _ in range(9))

        interleaved_a = derive(7, "request-a", "token-sampling")
        interleaved_b = derive(7, "request-b", "token-sampling")
        actual_a: list[int] = []
        actual_b: list[int] = []
        schedule = "aababbbaaabbbababa"
        for owner in schedule:
            target = interleaved_a if owner == "a" else interleaved_b
            (actual_a if owner == "a" else actual_b).append(target.next_u32())
        self.assertEqual(tuple(actual_a), expected_a)
        self.assertEqual(tuple(actual_b), expected_b)

    def test_fork_identity_is_independent_of_parent_draw_position(self) -> None:
        parent = derive(99, "request-fork", "token-sampling")
        child_before = parent.fork("draft")
        for _ in range(17):
            parent.next_u32()
        child_after = parent.fork("draft")
        self.assertEqual(child_before.digest, child_after.digest)
        self.assertEqual(
            tuple(child_before.next_u32() for _ in range(12)),
            tuple(child_after.next_u32() for _ in range(12)),
        )
        self.assertNotEqual(parent.fork("draft").digest, parent.fork("target").digest)


if __name__ == "__main__":
    unittest.main()
