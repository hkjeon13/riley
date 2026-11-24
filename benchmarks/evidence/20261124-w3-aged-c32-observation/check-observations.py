"""Exercise observation coverage, token counts, and median rank attribution."""

from pathlib import Path
import runpy
import unittest

TOOL = runpy.run_path(str(Path(__file__).with_name("verify.py")))
FIRST = "cmpl-0000000000000001"
SECOND = "cmpl-0000000000000002"
HOLDOUT = "cmpl-0000000000000003"


def capture():
    return {"short": {"warmup_response_ids": [FIRST], "measured": [{
        "request_index": 1, "response_id": SECOND, "start": 100.,
        "sent_at": 100.001, "ttft": .075, "completion_tokens": 10,
    }]}}


def log():
    return "\n".join((
        "W3-OBS-SNAPSHOT 3 0", f"W3-OBS {FIRST} 10000000 40000000 10 Completed",
        f"W3-OBS {SECOND} 20000000 40000000 10 Completed",
        f"W3-OBS {HOLDOUT} 1000000 8000000 64 Completed",
    ))


class ObservationTests(unittest.TestCase):
    def test_null_drop_and_complete_response_binding(self):
        count, by_workload = TOOL["bind_observations"](log(), capture(), 1)
        self.assertEqual(count, 3)
        row, = by_workload["short"]
        self.assertEqual(row["backend_admission_ms"], 20.)
        self.assertEqual(row["admitted_to_first_visible_ms"], 40.)
        self.assertAlmostEqual(row["other_after_write_ms"], 14.)

    def test_rejects_lost_ambiguous_or_mismatched_request_observations(self):
        for source in (
            log().replace("SNAPSHOT 3 0", "SNAPSHOT 3 1"),
            log().replace(HOLDOUT, SECOND),
            log().replace("40000000 10", "40000000 11"),
        ):
            with self.subTest(source=source), self.assertRaises(ValueError):
                TOOL["bind_observations"](source, capture(), 1)

    def test_separates_rank_shift_from_same_request_latency(self):
        def row(index, queue):
            return {"request_index":index, "ttft_ms":55.+queue, "client_before_write_ms":1.,
                    "backend_admission_ms":queue, "admitted_to_first_visible_ms":50.,
                    "other_after_write_ms":4.}
        a=[row(1, 10.), row(2, 5.)]
        b=[row(1, 1.), row(2, 16.)]
        report=TOOL["phase_effect"](a,b, lambda items,q: sorted(items)[int(len(items)*q)])
        self.assertEqual(report["reserve_p50_request"]["request_index"], 2)
        self.assertEqual(report["p50_request_phase_delta_ms"]["backend_admission_ms"], 11.)
        self.assertEqual(report["control_rank_shift_ms"], -5.)
        self.assertEqual(report["p50_replacing_one_measured_phase_ms"]["backend_admission_ms"], 65.)


if __name__ == "__main__":
    unittest.main()
