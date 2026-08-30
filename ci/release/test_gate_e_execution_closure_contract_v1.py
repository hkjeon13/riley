#!/usr/bin/env python3
"""CPU-only hostile-path tests for the Gate E execution-closure declaration."""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import sys
import unittest
from pathlib import Path


RELEASE_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_DIRECTORY = RELEASE_DIRECTORY.parents[1]
SCHEMA_PATH = (
    REPOSITORY_DIRECTORY
    / "benchmarks"
    / "release"
    / "candidates"
    / "gate-e-execution-closure-manifest-v1.schema.json"
)
sys.path.insert(0, str(RELEASE_DIRECTORY))

import gate_e_execution_closure_contract_v1 as contract  # noqa: E402


def digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def manifest_document() -> dict[str, object]:
    return {
        "schema_version": contract.SCHEMA_VERSION,
        "interpreter": {
            "audit_path": "/usr/bin/python3.10",
            "byte_length": 5_917_224,
            "sha256": digest("python3.10"),
        },
        "dynamic_loader": {
            "audit_path": "/lib64/ld-linux-x86-64.so.2",
            "byte_length": 210_968,
            "sha256": digest("ld-linux"),
        },
        "runtime_leaves": [
            {
                "audit_path": "/lib/x86_64-linux-gnu/libc.so.6",
                "byte_length": 2_022_344,
                "sha256": digest("libc"),
            },
            {
                "audit_path": "/usr/lib/x86_64-linux-gnu/libpython3.10.so.1.0",
                "byte_length": 5_776_912,
                "sha256": digest("libpython"),
            },
        ],
    }


def canonical(value: object) -> bytes:
    return contract.canonical_execution_closure_manifest_bytes(value)


class ExecutionClosureContractTests(unittest.TestCase):
    maxDiff = None

    def assert_reason(self, expected: str, raw: bytes) -> None:
        with self.assertRaises(contract.ExecutionClosureContractError) as raised:
            contract.parse_execution_closure_manifest_v1(raw)
        self.assertEqual(getattr(raised.exception, "reason_code", None), expected)

    def test_schema_is_parseable_and_declares_the_exact_v1_shape(self) -> None:
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

        self.assertEqual(schema["$schema"], "https://json-schema.org/draft/2020-12/schema")
        self.assertEqual(
            schema["properties"]["schema_version"]["const"], contract.SCHEMA_VERSION
        )
        self.assertEqual(
            schema["required"],
            ["dynamic_loader", "interpreter", "runtime_leaves", "schema_version"],
        )
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(schema["properties"]["runtime_leaves"]["minItems"], 1)
        self.assertEqual(
            schema["properties"]["runtime_leaves"]["maxItems"], contract.MAX_RUNTIME_LEAVES
        )

    def test_contract_has_no_filesystem_process_or_operational_surface(self) -> None:
        source = inspect.getsource(contract)
        for forbidden in (
            "import os",
            "import pathlib",
            "import subprocess",
            "import socket",
            "open(",
            "__main__",
            "exec(",
            "eval(",
        ):
            self.assertNotIn(forbidden, source)

    def test_canonical_manifest_returns_opaque_runtime_closure_digest(self) -> None:
        document = manifest_document()
        raw = canonical(document)

        parsed = contract.parse_execution_closure_manifest_v1(raw)

        self.assertEqual(parsed.interpreter.audit_path, "/usr/bin/python3.10")
        self.assertEqual(parsed.dynamic_loader.audit_path, "/lib64/ld-linux-x86-64.so.2")
        self.assertEqual(
            [leaf.audit_path for leaf in parsed.runtime_leaves],
            [
                "/lib/x86_64-linux-gnu/libc.so.6",
                "/usr/lib/x86_64-linux-gnu/libpython3.10.so.1.0",
            ],
        )
        self.assertEqual(parsed.runtime_closure_sha256, hashlib.sha256(raw).hexdigest())
        self.assertEqual(raw, canonical(json.loads(raw[:-1].decode("utf-8"))))

    def test_exact_canonical_bytes_duplicate_keys_and_terminal_newline_are_required(self) -> None:
        raw = canonical(manifest_document())

        self.assert_reason("invalid-terminal-newline", raw[:-1])
        self.assert_reason("invalid-terminal-newline", raw + b"\n")
        self.assert_reason("noncanonical-json", raw.replace(b":", b": ", 1))
        self.assert_reason(
            "duplicate-json-key",
            b'{"schema_version":"one","schema_version":"two"}\n',
        )
        self.assert_reason("non-finite-json-number", b'{"schema_version":NaN}\n')
        self.assert_reason(
            "integer-literal-too-large",
            b'{"byte_length":' + (b"9" * 5_000) + b"}\n",
        )
        self.assert_reason("invalid-json-number", b'{"byte_length":1.0}\n')
        self.assert_reason("invalid-json-number", b'{"byte_length":1e3}\n')
        self.assert_reason("invalid-json", b"\xff\n")

    def test_field_sets_schema_version_sha_and_length_are_closed(self) -> None:
        cases: list[tuple[str, dict[str, object]]] = []

        extra = manifest_document()
        extra["unexpected"] = True
        cases.append(("unexpected-field-set", extra))

        missing = manifest_document()
        del missing["dynamic_loader"]
        cases.append(("unexpected-field-set", missing))

        unsupported = manifest_document()
        unsupported["schema_version"] = "riley.rc3-gate-e-execution-closure-manifest.v2"
        cases.append(("unsupported-schema-version", unsupported))

        zero_digest = manifest_document()
        zero_digest["interpreter"] = copy.deepcopy(zero_digest["interpreter"])
        zero_digest["interpreter"]["sha256"] = "0" * 64  # type: ignore[index]
        cases.append(("zero-sha256", zero_digest))

        upper_digest = manifest_document()
        upper_digest["dynamic_loader"] = copy.deepcopy(upper_digest["dynamic_loader"])
        upper_digest["dynamic_loader"]["sha256"] = digest("loader").upper()  # type: ignore[index]
        cases.append(("invalid-sha256", upper_digest))

        boolean_length = manifest_document()
        boolean_length["interpreter"] = copy.deepcopy(boolean_length["interpreter"])
        boolean_length["interpreter"]["byte_length"] = True  # type: ignore[index]
        cases.append(("invalid-byte-length", boolean_length))

        oversized_length = manifest_document()
        oversized_length["dynamic_loader"] = copy.deepcopy(oversized_length["dynamic_loader"])
        oversized_length["dynamic_loader"]["byte_length"] = contract.MAX_LEAF_BYTES + 1  # type: ignore[index]
        cases.append(("invalid-byte-length", oversized_length))

        for expected, document in cases:
            with self.subTest(expected=expected):
                self.assert_reason(expected, canonical(document))

    def test_audit_path_grammar_is_ascii_absolute_and_canonical(self) -> None:
        invalid_paths = (
            "relative/python3.10",
            "/usr//bin/python3.10",
            "/usr/./bin/python3.10",
            "/usr/../bin/python3.10",
            "/usr/bin/",
            "/usr/.hidden/python3.10",
            "/usr/bin/python 3.10",
            "/usr/bin/python3.10\u2603",
        )
        for path in invalid_paths:
            with self.subTest(path=path):
                document = manifest_document()
                document["interpreter"] = copy.deepcopy(document["interpreter"])
                document["interpreter"]["audit_path"] = path  # type: ignore[index]
                self.assert_reason("invalid-audit-path", canonical(document))

    def test_runtime_leaves_are_nonempty_sorted_unique_and_separate_from_special_leaves(self) -> None:
        empty = manifest_document()
        empty["runtime_leaves"] = []
        self.assert_reason("invalid-runtime-leaves", canonical(empty))

        unsorted = manifest_document()
        unsorted["runtime_leaves"] = list(reversed(unsorted["runtime_leaves"]))  # type: ignore[arg-type]
        self.assert_reason("runtime-leaves-not-strictly-sorted", canonical(unsorted))

        duplicate = manifest_document()
        duplicate["runtime_leaves"] = copy.deepcopy(duplicate["runtime_leaves"])  # type: ignore[index]
        duplicate["runtime_leaves"].append(copy.deepcopy(duplicate["runtime_leaves"][1]))  # type: ignore[index]
        self.assert_reason("runtime-leaves-not-strictly-sorted", canonical(duplicate))

        overlaps_interpreter = manifest_document()
        overlaps_interpreter["runtime_leaves"] = copy.deepcopy(overlaps_interpreter["runtime_leaves"])
        overlaps_interpreter["runtime_leaves"][0]["audit_path"] = "/usr/bin/python3.10"  # type: ignore[index]
        self.assert_reason("duplicate-audit-path", canonical(overlaps_interpreter))

        loader_overlap = manifest_document()
        loader_overlap["runtime_leaves"] = copy.deepcopy(loader_overlap["runtime_leaves"])
        loader_overlap["runtime_leaves"][0]["audit_path"] = "/lib64/ld-linux-x86-64.so.2"  # type: ignore[index]
        self.assert_reason("duplicate-audit-path", canonical(loader_overlap))

        same_special_path = manifest_document()
        same_special_path["dynamic_loader"] = copy.deepcopy(same_special_path["dynamic_loader"])
        same_special_path["dynamic_loader"]["audit_path"] = "/usr/bin/python3.10"  # type: ignore[index]
        self.assert_reason("duplicate-audit-path", canonical(same_special_path))

    def test_runtime_entry_and_total_byte_budgets_are_closed(self) -> None:
        too_many = manifest_document()
        too_many["runtime_leaves"] = [
            {
                "audit_path": f"/runtime/lib{index:03d}.so",
                "byte_length": 1,
                "sha256": digest(f"runtime-{index}"),
            }
            for index in range(contract.MAX_RUNTIME_LEAVES + 1)
        ]
        self.assert_reason("runtime-leaf-budget-exceeded", canonical(too_many))

        over_total = manifest_document()
        over_total["runtime_leaves"] = [
            {
                "audit_path": f"/runtime/lib{index}.so",
                "byte_length": contract.MAX_LEAF_BYTES,
                "sha256": digest(f"large-runtime-{index}"),
            }
            for index in range(4)
        ]
        self.assert_reason("closure-byte-budget-exceeded", canonical(over_total))

    def test_json_depth_node_and_manifest_byte_budgets_fail_closed(self) -> None:
        nested: object = {"schema_version": contract.SCHEMA_VERSION}
        for _ in range(contract.MAX_JSON_NESTING + 1):
            nested = {"nested": nested}
        self.assert_reason("json-nesting-too-deep", canonical(nested))

        node_heavy = {"schema_version": contract.SCHEMA_VERSION, "nodes": [0] * contract.MAX_JSON_NODES}
        self.assert_reason("json-node-budget-exceeded", canonical(node_heavy))
        self.assert_reason("manifest-byte-budget-exceeded", b"x" * (contract.MAX_MANIFEST_BYTES + 1))


if __name__ == "__main__":
    unittest.main()
