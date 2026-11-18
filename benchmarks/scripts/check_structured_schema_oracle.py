#!/usr/bin/env python3
"""Verify the structured output test fixtures against Draft 2020-12.

Run with jsonschema==4.23.0 in the validation environment.
"""

import json
from importlib.metadata import version
from pathlib import Path

from jsonschema import Draft202012Validator


FIXTURE = (Path(__file__).resolve().parents[2]
           / "crates/riley-server/tests/fixtures/structured_schema_oracle.json")
ORACLE_VERSION = "4.23.0"


def main():
    installed = version("jsonschema")
    if installed != ORACLE_VERSION:
        raise SystemExit(f"Expected jsonschema=={ORACLE_VERSION}, found {installed}")
    cases = json.loads(FIXTURE.read_text(encoding="utf-8"))
    samples = 0
    for case in cases:
        schema = case["schema"]
        Draft202012Validator.check_schema(schema)
        validator = Draft202012Validator(schema)
        for sample in case["samples"]:
            actual = validator.is_valid(json.loads(sample["text"]))
            if actual is not sample["valid"]:
                raise SystemExit(f"Draft 2020-12 disagreed for {schema!r}, instance {sample['text']!r}")
            samples += 1
    print(f"Draft 2020-12 (jsonschema {installed}): {samples} instances across {len(cases)} schemas matched")


if __name__ == "__main__":
    main()
