#!/usr/bin/env python3
"""Contract tests for the source-owned C02 shutdown v2 schema documents."""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path


SCHEMA_DIRECTORY = Path(__file__).resolve().parents[2] / "benchmarks" / "release" / "candidates"
DRAFT = "https://json-schema.org/draft/2020-12/schema"


def _no_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    document: dict[str, object] = {}
    for key, value in pairs:
        if key in document:
            raise ValueError(f"duplicate JSON key {key!r}")
        document[key] = value
    return document


def _read(name: str) -> dict[str, object]:
    document = json.loads(
        (SCHEMA_DIRECTORY / name).read_text(encoding="utf-8"),
        object_pairs_hook=_no_duplicate_keys,
    )
    if not isinstance(document, dict):
        raise AssertionError(f"{name} root must be an object")
    return document


class C02ShutdownV2SchemaTests(unittest.TestCase):
    def assert_exact_object(self, schema: object, fields: set[str]) -> dict[str, object]:
        self.assertIsInstance(schema, dict)
        assert isinstance(schema, dict)
        self.assertEqual(schema.get("type"), "object")
        self.assertIs(schema.get("additionalProperties"), False)
        self.assertEqual(set(schema.get("required", [])), fields)
        self.assertEqual(set(schema.get("properties", {})), fields)
        return schema

    def test_metrics_v2_has_the_exact_nested_raw_shape(self) -> None:
        schema = _read("c02-capture-metrics-v2.schema.json")
        self.assertEqual(schema["$schema"], DRAFT)
        self.assertEqual(schema["$id"], "https://riley.invalid/schemas/c02-capture-metrics-v2.schema.json")
        metrics = self.assert_exact_object(
            schema,
            {"schema_version", "request_states", "kv_blocks", "allocation", "quiescence"},
        )
        self.assertEqual(metrics["properties"]["schema_version"]["const"], "riley.c02-capture-metrics.v2")

        definitions = metrics["$defs"]
        self.assertEqual(definitions["nonnegativeInteger"]["type"], "integer")
        self.assertEqual(definitions["nonnegativeInteger"]["minimum"], 0)
        self.assert_exact_object(
            definitions["requestStates"],
            {"active", "pending_requests", "completed", "failed", "cancelled", "capacity_rejections"},
        )
        self.assert_exact_object(definitions["kvBlocks"], {"free", "reserved", "active"})
        self.assert_exact_object(
            definitions["allocation"],
            {"device_live_count", "device_live_bytes", "pinned_live_count", "pinned_live_bytes"},
        )
        quiescence = self.assert_exact_object(
            definitions["quiescence"],
            {
                "completion_outbox",
                "outstanding_iterations",
                "riley_owned_live_allocations",
                "worker_accepting",
                "scheduler_accepting",
            },
        )
        self.assertEqual(quiescence["properties"]["worker_accepting"]["type"], "boolean")
        self.assertEqual(quiescence["properties"]["scheduler_accepting"]["type"], "boolean")

    def test_shutdown_artifact_is_exactly_the_raw_v2_leaf(self) -> None:
        schema = _read("c02-shutdown-quiescence-v2.schema.json")
        self.assertEqual(schema["$schema"], DRAFT)
        self.assertEqual(schema["$id"], "https://riley.invalid/schemas/c02-shutdown-quiescence-v2.schema.json")
        artifact = self.assert_exact_object(
            schema,
            {
                "schema_version",
                "capture_status",
                "qualification_status",
                "server_pid",
                "server_start_ticks",
                "worker_ready",
                "final_metrics",
            },
        )
        properties = artifact["properties"]
        self.assertEqual(properties["schema_version"]["const"], "riley.c02-shutdown-quiescence.v2")
        self.assertEqual(properties["capture_status"]["const"], "captured")
        self.assertEqual(properties["qualification_status"]["const"], "not-run")
        self.assertEqual(properties["worker_ready"]["const"], False)
        self.assertEqual(properties["server_pid"]["minimum"], 1)
        self.assertEqual(properties["server_start_ticks"]["minimum"], 1)
        self.assertEqual(properties["final_metrics"]["$ref"], "c02-capture-metrics-v2.schema.json")

    def test_completion_marker_is_nonhidden_and_binds_a_sha256(self) -> None:
        schema = _read("c02-shutdown-quiescence-completion-v2.schema.json")
        self.assertEqual(schema["$schema"], DRAFT)
        self.assertEqual(
            schema["$id"],
            "https://riley.invalid/schemas/c02-shutdown-quiescence-completion-v2.schema.json",
        )
        marker = self.assert_exact_object(
            schema,
            {"schema_version", "artifact_filename", "artifact_sha256"},
        )
        properties = marker["properties"]
        self.assertEqual(
            properties["schema_version"]["const"],
            "riley.c02-shutdown-quiescence-complete.v2",
        )
        filename_pattern = properties["artifact_filename"]["pattern"]
        self.assertIsNotNone(re.fullmatch(filename_pattern, "shutdown.json"))
        self.assertIsNone(re.fullmatch(filename_pattern, ".shutdown.json"))
        self.assertIsNone(re.fullmatch(filename_pattern, "nested/shutdown.json"))
        self.assertIsNone(re.fullmatch(filename_pattern, "shutdown.json.complete"))
        self.assertIn("<artifact_filename>.complete", schema["description"])
        sha256 = properties["artifact_sha256"]
        self.assertEqual(sha256["allOf"][0]["pattern"], "^[0-9a-f]{64}$")
        self.assertEqual(sha256["allOf"][1]["not"]["const"], "0" * 64)


if __name__ == "__main__":
    unittest.main()
