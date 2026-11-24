"""Exercise application dispatch, model fallback and dependency failure at the replay entry point."""

import copy
import json
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import agent_workflow as workflow
from audit_agent_prefix import audit
from summarize_agent_workflow import summarize


class NamedSteps(unittest.TestCase):
    def setUp(self):
        self.sent = []
        self.executed = []
        self.tools, self.prompt, self.docs = workflow.fixture(501, 2)
        a, b = [d["p95_ms"] for d in self.docs]
        self.answer = f"Before: {a} ms. After: {b} ms. Reduction: {a - b} ms. report-A and report-B."

    def send(self, address, request, row, first_visible):
        self.sent.append(copy.deepcopy(request))
        row["sent_ns"] = time.monotonic_ns()
        row["response_id"] = "model-response"
        row.update(
            content_ns=[time.monotonic_ns(), time.monotonic_ns()],
            usage={"completion_tokens": 64},
        )
        if request["tool_choice"] == "none":
            row["finish"] = "stop"
            return self.answer, []
        name = request["tool_choice"]["function"]["name"]
        args = (
            {}
            if name == "list_reports"
            else {"first": "report-A", "second": "report-B"}
        )
        row["finish"] = "tool_calls"
        return "", [
            {
                "id": f"model-{name}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            }
        ]

    def execute(self, index, call, docs):
        self.executed.append((index, call["id"]))
        return workflow.fixture_tool_result(index, docs)

    def run_workflow(self, enabled):
        return workflow.replay(
            "unused",
            "m",
            501,
            2,
            0,
            "test",
            send=self.send,
            execute_tool=self.execute,
            dispatch_fixed_tools=enabled,
        )

    def test_owned_steps_execute_in_order_and_only_final_calls_model(self):
        result = self.run_workflow(True)
        self.assertTrue(result["ok"], result.get("error"))
        self.assertEqual(len(self.sent), 1)
        self.assertEqual([i for i, _ in self.executed], [0, 1])
        ids = [cid for _, cid in self.executed]
        self.assertNotEqual(ids[0], ids[1])
        messages = self.sent[0]["messages"]
        self.assertEqual(
            [m["tool_call_id"] for m in messages if m["role"] == "tool"], ids
        )
        self.assertEqual(json.loads(messages[-1]["content"]), self.docs)
        for row in result["turns"][:2]:
            self.assertEqual(row["dispatch"], "application")
            self.assertTrue(row["planned_request_sha256"])
            self.assertTrue(
                {
                    "response_id",
                    "usage",
                    "content_ns",
                    "sent_ns",
                    "finish",
                    "observed_calls",
                }.isdisjoint(row)
            )

    def test_summary_includes_real_tools_without_counting_model_turns_for_them(self):
        r = self.run_workflow(True)
        limits = {
            "workflow_first_final": 5000,
            "workflow_last_final": 8000,
            "final_content_gap_p95": 80,
        }
        summary = summarize([r], limits)
        self.assertEqual(summary["complete"], 1)
        self.assertEqual(summary["completed_latency_breakdowns"], 1)
        r["turns"][1]["dispatch_ns"] = r["turns"][0]["tool_ready_ns"] - 1
        with self.assertRaisesRegex(ValueError, "prerequisite"):
            summarize([r], limits)

    def test_prefix_audit_encodes_only_sent_requests(self):
        record = self.run_workflow(True)
        final = record["turns"][-1]
        final.update(
            input_sha256=workflow.fingerprint(self.sent[0]), done_ns=time.monotonic_ns()
        )
        final["usage"]["prompt_tokens"] = 2
        encoded = {"ids": [17, 23], "normalize_ns": 1, "encode_ns": 1}
        with patch(
            "audit_agent_prefix.subprocess.run",
            return_value=SimpleNamespace(stdout=json.dumps(encoded)),
        ) as invoke:
            rows = audit([record], "m", "profiler", "tokenizer")
        self.assertEqual(len(invoke.call_args.kwargs["input"].splitlines()), 1)
        self.assertEqual(
            [(row["turn"], row["prior_completed_eligible_tokens"]) for row in rows],
            [(2, 0)],
        )

    def test_ambiguous_second_step_goes_to_model(self):
        self.tools[1]["function"]["parameters"]["properties"]["first"]["enum"].append(
            "report-C"
        )
        with patch.object(
            workflow, "fixture", return_value=(self.tools, self.prompt, self.docs)
        ):
            result = self.run_workflow(True)
        self.assertTrue(result["ok"], result.get("error"))
        self.assertEqual(len(self.sent), 2)
        self.assertEqual(self.executed[1], (1, "model-read_reports"))
        self.assertIn("observed_calls", result["turns"][1])

    def test_dependency_failure_stops_before_second_tool(self):
        self.execute = lambda index, call, docs: (
            self.executed.append(index) or ["report-A"]
        )
        result = self.run_workflow(True)
        self.assertFalse(result["ok"])
        self.assertIn("missing", result["error"])
        self.assertEqual(self.executed, [0])
        self.assertEqual(self.sent, [])

    def test_default_goes_through_model_for_all_three_steps(self):
        result = self.run_workflow(False)
        self.assertTrue(result["ok"], result.get("error"))
        self.assertEqual(len(self.sent), 3)
        self.assertEqual(
            [cid for _, cid in self.executed],
            ["model-list_reports", "model-read_reports"],
        )


if __name__ == "__main__":
    unittest.main()
