#!/usr/bin/env python3
"""Write one create-only C02 startup artifact from a canonical /v1/config capture.

This is a reference evidence writer for remote qualification orchestration; it
does not start Riley or claim that a local file is a live server response.  A
production server implementation must emit the same closed JSON artifact with
an ``O_EXCL``/create-new operation at startup.  The verifier rejects a
non-canonical capture or a replacement artifact either way.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Sequence

import check_effective_runtime_config_receipt as receipt
import check_rc3_qualification as qualification


def write_startup_artifact(
    endpoint_payload_path: Path,
    output_path: Path,
    *,
    created_at_utc: str,
) -> receipt.StartupArtifactWriteResult:
    """Wrap a canonical endpoint capture in an O_EXCL startup artifact."""

    if not qualification.UTC_RE.fullmatch(created_at_utc):
        raise receipt.ConfigReceiptError("--created-at-utc must be UTC second precision")
    endpoint_raw = qualification._read_regular_path(endpoint_payload_path, "endpoint payload")
    endpoint_document, endpoint = receipt._validate_endpoint_bytes(endpoint_raw, "endpoint payload")
    artifact = {
        "schema_version": receipt.STARTUP_ARTIFACT_VERSION,
        "created_at_utc": created_at_utc,
        "candidate_id": endpoint.candidate_id,
        "endpoint_path": "/v1/config",
        # Freeze/Gate-E hashes are post-capture semantic decision bindings, not
        # startup facts.  Gate E is still a future output at this point.
        "runtime_identity": endpoint.runtime_identity,
        "endpoint_payload_sha256": hashlib.sha256(endpoint_raw).hexdigest(),
        "endpoint_payload": endpoint_document,
    }
    # Validate before publishing so a caller cannot create an invalid immutable
    # artifact and accidentally consume the path forever.
    receipt.validate_startup_artifact(artifact)
    encoded = qualification.canonical_json_bytes(artifact)
    qualification._write_create_only_bytes(output_path, encoded)
    return receipt.StartupArtifactWriteResult(
        candidate_id=endpoint.candidate_id,
        configuration_profile=endpoint.runtime_identity["configuration_profile"],
        endpoint_payload_sha256=hashlib.sha256(endpoint_raw).hexdigest(),
        startup_artifact_sha256=hashlib.sha256(encoded).hexdigest(),
        path=output_path,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint-payload", required=True, type=Path)
    parser.add_argument("--created-at-utc", required=True)
    parser.add_argument("--output", required=True, type=Path, help="new create-only artifact path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = write_startup_artifact(
            args.endpoint_payload,
            args.output,
            created_at_utc=args.created_at_utc,
        )
    except (OSError, qualification.QualificationError) as error:
        print(str(error), file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "candidate_id": result.candidate_id,
                "configuration_profile": result.configuration_profile,
                "endpoint_payload_sha256": result.endpoint_payload_sha256,
                "path": str(result.path),
                "startup_artifact_sha256": result.startup_artifact_sha256,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
