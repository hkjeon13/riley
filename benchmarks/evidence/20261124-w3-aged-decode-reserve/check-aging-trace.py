"""Reproduce valid and invalid scheduler aging receipts on the CPU."""

from pathlib import Path
import runpy
from typing import NamedTuple
import unittest


CHECK = runpy.run_path(str(Path(__file__).with_name("verify.py")))["aging_steps"]


class Step(NamedTuple):
    id: int
    gpu: int
    decode: int
    prefill: list[tuple[int, int, int]]


def trace(enabled, *, ready=3, row_count=None):
    allocated = 1021 if enabled else 1023
    expected_decode = 3 if enabled else 1
    rows = [
        Step(2, 132_000_000, expected_decode, [(4, allocated, allocated)]),
        Step(3, 132_000_000, 0, [(4, 1024, allocated + 1024)]),
        Step(4, 132_000_000, 0, [(4, 1024, allocated + 2048)]),
        Step(5, 45_000_000, 0, [(4, 3500 - allocated - 2048, 3500)]),
    ]
    markers = ["W3-AGED-TURNS 1", f"W3-AGED-TURN 2 4 3500 {ready} {row_count or allocated} {int(enabled)}"]
    return rows, markers


class AgingTests(unittest.TestCase):
    def test_both_modes_have_expected_committed_decode_coverage(self):
        for enabled, deferred in ((False, 2), (True, 0)):
            with self.subTest(enabled=enabled):
                (row,) = CHECK(*trace(enabled), enabled)
                self.assertEqual(row["deferred_decode_rows"], deferred)
                self.assertEqual(row["prefill_rows"], 1021 if enabled else 1023)

    def test_rejects_claims_not_matched_by_commits(self):
        for args in ({"ready": 33}, {"row_count": 1023}):
            with self.subTest(args=args):
                with self.assertRaisesRegex(ValueError, "disagrees with committed plan"):
                    CHECK(*trace(True, **args), True)

    def test_rejects_duplicate_uncommitted_or_incomplete_aging_trace(self):
        rows, markers = trace(False)
        for observed, problem in (
            (markers + markers[1:], "duplicate aging turn"),
            ([markers[0], markers[1].replace("TURN 2 ", "TURN 1 ")], "a committed step"),
            ([markers[0]], "missing or incomplete"),
        ):
            with self.subTest(problem=problem):
                with self.assertRaisesRegex(ValueError, problem):
                    CHECK(rows, observed, False)
        with self.assertRaisesRegex(ValueError, "complete prefill"):
            CHECK(rows[:-1], markers, False)


if __name__ == "__main__":
    unittest.main()
