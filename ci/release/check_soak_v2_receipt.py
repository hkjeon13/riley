#!/usr/bin/env python3
"""Read-only C02 soak semantic-replay input precheck.

This narrow dispatcher accepts only a completed raw v4 serial manifest or a
completed raw v5 native-fallback manifest.  It replays that manifest through
one held private evidence-root FD and returns a structural ``bound/not-run``
precheck.  It does not issue a semantic receipt, interpret a visible terminal
marker as producer success, or validate a frozen candidate, Gate E, a workload
campaign, thresholds, or lifecycle authority.
"""

from __future__ import annotations

import argparse
import fcntl
import os
import re
import sys

sys.dont_write_bytecode = True

from pathlib import Path
from typing import Any, Callable, NoReturn, Sequence, TypeVar

import check_c02_provenance_v2 as raw
import provenance_v2_common as common

PRECHECK_REPORT_VERSION = "riley.soak-v2-semantic-replay-precheck.v1"
RAW_STRUCTURAL_ONLY_AUTHORITY = "raw-structural-only"
_DIRECT_RAW_MANIFEST_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,240}\.json$")
_RAW_REPORT_FIELDS = {
    "schema_version",
    "status",
    "qualification_status",
    "candidate_id",
    "bindings",
    "raw_manifest",
    "targets",
    "checks",
    "reason_codes",
}
_HISTORICAL_MANIFEST_REASONS = {
    "riley.soak-v2-receipt.v1": "historical-soak-v1-rejected",
    "riley.soak-v2-raw-provenance.v1": "historical-soak-v1-rejected",
    "riley.soak-v2-raw-provenance.v2": "historical-soak-v2-rejected",
    "riley.soak-v2-raw-provenance.v3": "historical-soak-v3-rejected",
}


class SoakV2ReceiptPrecheckError(ValueError):
    """A raw soak input cannot enter the later semantic replay boundary."""


T = TypeVar("T")


def _fail(code: str, message: str) -> NoReturn:
    error = SoakV2ReceiptPrecheckError(message)
    error.reason_code = code  # type: ignore[attr-defined]
    raise error


def _common(call: Callable[[], T]) -> T:
    try:
        return call()
    except common.ProvenanceV2Error as error:
        _fail(getattr(error, "reason_code", "unsafe-evidence"), str(error))


def _raw(call: Callable[[], T]) -> T:
    try:
        return call()
    except raw.C02ProvenanceError as error:
        _fail(getattr(error, "reason_code", "invalid-raw-provenance"), str(error))


def _evidence_root(value: Path) -> Path:
    try:
        raw_path = os.fspath(value)
    except TypeError as error:
        _fail("invalid-evidence-root", f"--evidence-root is not a path: {error}")
    if (
        type(raw_path) is not str
        or not raw_path
        or "\x00" in raw_path
        or not os.path.isabs(raw_path)
        or raw_path.startswith("//")
        or raw_path != os.path.normpath(raw_path)
    ):
        _fail(
            "invalid-evidence-root",
            "--evidence-root must be a normalized absolute path",
        )
    root = Path(raw_path)
    source_root = Path(__file__).resolve().parents[2]
    try:
        root.relative_to(source_root)
    except ValueError:
        return root
    _fail(
        "evidence-root-inside-source-checkout",
        "--evidence-root must be outside the source checkout",
    )


def _raw_manifest_name(value: str) -> str:
    if type(value) is not str or _DIRECT_RAW_MANIFEST_RE.fullmatch(value) is None:
        _fail(
            "raw-manifest-must-be-direct-root-leaf",
            "--raw-manifest must be one direct nonhidden root JSON leaf",
        )
    return value


def _shared_lock(root_fd: int) -> None:
    try:
        fcntl.flock(root_fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
    except (AttributeError, OSError) as error:
        _fail(
            "evidence-root-lock-unavailable",
            f"cannot acquire shared evidence-root lock: {error}",
        )


def _unlock_quietly(root_fd: int | None) -> None:
    if root_fd is not None:
        try:
            fcntl.flock(root_fd, fcntl.LOCK_UN)
        except OSError:
            pass


def _close_quietly(root_fd: int | None) -> None:
    if root_fd is not None:
        try:
            os.close(root_fd)
        except OSError:
            pass


def _manifest_header(
    root_fd: int,
    manifest_name: str,
) -> tuple[common.EvidenceDescriptor, str]:
    manifest_raw = _common(
        lambda: common.read_bounded_regular_relative(
            root_fd,
            manifest_name,
            "soak raw manifest dispatch header",
            maximum_bytes=raw.MAX_RAW_BYTES,
        )
    )
    descriptor = _common(
        lambda: common.descriptor_for_bytes(
            manifest_name,
            manifest_raw,
            "soak raw manifest dispatch header",
        )
    )
    document = _common(
        lambda: common.parse_canonical_json(
            manifest_raw,
            "soak raw manifest dispatch header",
            maximum_bytes=raw.MAX_RAW_BYTES,
        )
    )
    if type(document) is not dict:
        _fail(
            "unsupported-soak-raw-manifest-version",
            "soak raw manifest dispatch header must be a JSON object",
        )
    schema_version = document.get("schema_version")
    if type(schema_version) is not str:
        _fail(
            "unsupported-soak-raw-manifest-version",
            "soak raw manifest must contain a text schema_version",
        )
    return descriptor, schema_version


def _reject_or_select_manifest_version(schema_version: str) -> tuple[str, str]:
    if schema_version == raw.SOAK_V4_MANIFEST_VERSION:
        return raw.SOAK_V4_REPORT_VERSION, "completed-v4-raw-provenance"
    if schema_version == raw.SOAK_V5_MANIFEST_VERSION:
        return raw.SOAK_V5_REPORT_VERSION, "completed-v5-raw-provenance"
    historical_reason = _HISTORICAL_MANIFEST_REASONS.get(schema_version)
    if historical_reason is not None:
        _fail(
            historical_reason,
            f"historical soak raw manifest version {schema_version!r} is not a semantic input",
        )
    _fail(
        "unsupported-soak-raw-manifest-version",
        f"unsupported soak raw manifest version {schema_version!r}",
    )


def _replay_completed_manifest_fd(
    root_fd: int,
    manifest_name: str,
    schema_version: str,
) -> tuple[dict[str, Any], str, str]:
    expected_report_version, primary_check = _reject_or_select_manifest_version(
        schema_version
    )
    if schema_version == raw.SOAK_V4_MANIFEST_VERSION:
        report = _raw(
            lambda: raw.verify_completed_soak_provenance_v4_fd(root_fd, manifest_name)
        )
    else:
        report = _raw(
            lambda: raw.verify_completed_soak_provenance_v5_fd(root_fd, manifest_name)
        )
    if (
        not isinstance(report, dict)
        or set(report) != _RAW_REPORT_FIELDS
        or report.get("schema_version") != expected_report_version
        or report.get("status") != "bound"
        or report.get("qualification_status") != "not-run"
        or report.get("reason_codes") != []
    ):
        _fail(
            "invalid-completed-raw-provenance-report",
            "completed raw provenance replay returned an invalid report shape",
        )
    return report, expected_report_version, primary_check


def _precheck_report(
    raw_report: dict[str, Any],
    *,
    raw_manifest_version: str,
    primary_check: str,
) -> dict[str, Any]:
    return {
        "schema_version": PRECHECK_REPORT_VERSION,
        "status": "bound",
        "qualification_status": "not-run",
        "authority": RAW_STRUCTURAL_ONLY_AUTHORITY,
        "candidate_id": raw_report["candidate_id"],
        "bindings": raw_report["bindings"],
        "raw_manifest": raw_report["raw_manifest"],
        "raw_manifest_version": raw_manifest_version,
        "targets": raw_report["targets"],
        "checks": [
            {"name": primary_check, "bound": True},
            {
                "name": "header-to-completed-replay-descriptor-binding",
                "bound": True,
            },
        ],
        "reason_codes": [],
    }


def check_soak_v2_receipt(
    evidence_root: Path,
    raw_manifest: str,
) -> dict[str, Any]:
    """Replay one completed raw v4/v5 manifest without issuing a receipt."""

    root = _evidence_root(evidence_root)
    manifest_name = _raw_manifest_name(raw_manifest)
    root_fd = _common(
        lambda: common.open_private_evidence_directory(root, "--evidence-root")
    )
    try:
        _shared_lock(root_fd)
        header_descriptor, manifest_version = _manifest_header(root_fd, manifest_name)
        raw_report, report_version, primary_check = _replay_completed_manifest_fd(
            root_fd,
            manifest_name,
            manifest_version,
        )
        returned_descriptor = _common(
            lambda: common.parse_descriptor(
                raw_report["raw_manifest"],
                "completed raw provenance report.raw_manifest",
            )
        )
        if returned_descriptor != header_descriptor:
            _fail(
                "raw-manifest-changed-during-version-dispatch",
                "raw manifest differs between version dispatch and completed replay",
            )
        if raw_report["schema_version"] != report_version:
            _fail(
                "invalid-completed-raw-provenance-report",
                "completed raw provenance report version changed during replay",
            )
        return _precheck_report(
            raw_report,
            raw_manifest_version=manifest_version,
            primary_check=primary_check,
        )
    finally:
        _unlock_quietly(root_fd)
        _close_quietly(root_fd)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--raw-manifest", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = check_soak_v2_receipt(args.evidence_root, args.raw_manifest)
    except SoakV2ReceiptPrecheckError as error:
        print(f"C02 soak semantic-replay precheck failed: {error}", file=sys.stderr)
        return 2
    sys.stdout.buffer.write(common.canonical_json_bytes(report) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
