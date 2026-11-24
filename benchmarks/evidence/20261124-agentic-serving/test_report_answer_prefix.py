"""The opt-in report adapter can use only the two measurements returned by the requested tool."""

import copy
import json
import unittest

from agent_workflow import append_tool_result, fixture, turn_request
from report_answer_prefix import report_prefix


class ReportPrefixTest(unittest.TestCase):
    def setUp(self):
        self.tools, _, self.docs = fixture(17101, 2)
        self.call = {
            "id": "observed",
            "type": "function",
            "function": {
                "name": "read_reports",
                "arguments": json.dumps({"first": "report-A", "second": "report-B"}),
            },
        }

    def request(self, rows):
        messages = []
        append_tool_result(messages, [self.call], rows, 1)
        return turn_request("m", messages, self.tools, None)

    def test_uses_real_rows_in_declared_roles(self):
        rows = copy.deepcopy(self.docs)
        rows[0]["p95_ms"] = 10
        rows[1]["p95_ms"] = 15
        expected = (
            "Before: 10 ms. After: 15 ms. Reduction: -5 ms.\n\n"
            "The measured p95 changed from 10 ms (report-A) to 15 ms (report-B)."
        )
        self.assertEqual(report_prefix(self.request(rows[::-1])), expected)

    def test_ambiguous_or_incomplete_results_use_normal_generation(self):
        a, b = self.docs
        for rows in (
            [a],
            [a, a],
            [a, a, b],
            [a, {**b, "id": "unrequested"}],
            [a, {**b, "p95_ms": False}],
            [a, {**b, "p95_ms": -1}],
            [a, {**b, "statistic": "unrelated metric"}],
        ):
            self.assertEqual(report_prefix(self.request(rows)), "")
        req = self.request(self.docs)
        req["messages"][-1]["tool_call_id"] = "different"
        self.assertEqual(report_prefix(req), "")
        req = self.request(self.docs)
        req["tool_choice"] = "auto"
        self.assertEqual(report_prefix(req), "")
