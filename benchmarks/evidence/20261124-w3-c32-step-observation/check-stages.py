import unittest

from stages import c32_execution_spans, reconcile


TRACE = """W3-ADMISSION-TRACE 5 0
W3-EXECUTION 105 180 256 15
W3-WORKER-STEP 100 194
W3-EXECUTION 205 220 128 10
W3-WORKER-STEP 200 230
W3-ADMISSION 42 99 110 237 247
"""
OBSERVATIONS = [{"request_index": 32, "response_id": "cmpl-000000000000002a", "backend_admission_ms": .000160}]


class StageTests(unittest.TestCase):
    def test_matches_two_executions_and_uncovered_queue_time(self):
        (row,) = reconcile(TRACE, OBSERVATIONS)
        self.assertEqual(row["worker_step_remaining_ms"], .000084)
        self.assertEqual(row["later_worker_steps_ms"], .000030)
        self.assertEqual(row["cuda_span_overlap_ms"], .000085)
        self.assertEqual(row["queue_outside_steps_ms"], .000013)
        self.assertEqual(row["incumbent_shape"], (256, 15))
        self.assertAlmostEqual(row["admission_remainder_ms"], .000012)

    def test_rejects_missing_or_overlapping_spans(self):
        for altered in (TRACE.replace("5 0", "5 1"), TRACE.replace("200 230", "193 230"),
                        TRACE.replace("105 180", "105 195"), TRACE.replace(" 237 247", " 250 247")):
            with self.assertRaises(ValueError):
                reconcile(altered, OBSERVATIONS)

    def test_requires_observed_admission(self):
        with self.assertRaises(KeyError):
            reconcile(TRACE, [{**OBSERVATIONS[0], "response_id": "cmpl-000000000000002b"}])

    def test_bounds_the_c32_sequence_by_warmup_and_holdout(self):
        log = TRACE.replace("5 0", "7 0").replace("42 99", "42 109") + (
            "W3-ADMISSION 41 89 90 92 93\nW3-ADMISSION 43 248 249 250 251\n"
        )
        capture = {"warmup_response_ids": ["cmpl-0000000000000029"], "measured": OBSERVATIONS}
        steps, first_measured = c32_execution_spans(log, capture, 1)
        self.assertEqual(first_measured, 109)
        self.assertEqual([step[1:] for step in steps], [(256, 15), (128, 10)])
        with self.assertRaises(ValueError):
            c32_execution_spans(log, capture, 2)


if __name__ == "__main__":
    unittest.main()
