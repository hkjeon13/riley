#!/usr/bin/env python3
"""Read-only RC3 rollback raw-structural admission precheck.

This narrow dispatcher accepts one completed terminal rollback provenance v4
manifest through held private evidence-root and switch-directory file
descriptors.  It returns a structural ``bound/not-run`` diagnostic only.  It
does not establish producer success, host rollback, lifecycle success, a
semantic receipt, candidate freeze, Gate E, or qualification.
"""

from __future__ import annotations

import argparse
import fcntl
import os
import re
import sys

_BYTECODE_DISABLED_AT_STARTUP = bool(sys.flags.dont_write_bytecode)
_BYTECODE_DISABLED_ON_MODULE_ENTRY = sys.dont_write_bytecode
sys.dont_write_bytecode = True

from pathlib import Path
from typing import Any, Callable, NoReturn, Sequence, TypeVar

import check_rc3_rollback_provenance_v4 as raw
import prepare_rc3_rollback_artifacts_v1 as prepare
import provenance_v2_common as common


PRECHECK_REPORT_VERSION = "riley.rc3-rollback-raw-structural-precheck.v1"
RAW_STRUCTURAL_ONLY_AUTHORITY = "raw-structural-only"
_DIRECT_RAW_MANIFEST_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,122}\.json$")
_RAW_REPORT_FIELDS = {
    "schema_version",
    "status",
    "qualification_status",
    "candidate_id",
    "bindings",
    "raw_manifest",
    "rollback_v3_manifest",
    "atomic_transaction_session",
    "checks",
    "reason_codes",
}
_HISTORICAL_MANIFEST_REASONS = {
    "riley.rc3-rollback-receipt.v1": "historical-rollback-v1-rejected",
    "riley.rc3-rollback-raw-provenance.v1": "historical-rollback-v1-rejected",
    "riley.rc3-rollback-raw-provenance.v2": "historical-rollback-v2-rejected",
    "riley.rc3-rollback-raw-provenance.v3": "nonterminal-rollback-v3-not-admissible",
}


class RollbackStructuralPrecheckError(ValueError):
    """A raw rollback input cannot enter the later semantic replay boundary."""


T = TypeVar("T")


def _fail(code: str, message: str) -> NoReturn:
    error = RollbackStructuralPrecheckError(message)
    error.reason_code = code  # type: ignore[attr-defined]
    raise error


def _require_bytecode_cache_disabled() -> None:
    if not (
        _BYTECODE_DISABLED_AT_STARTUP and _BYTECODE_DISABLED_ON_MODULE_ENTRY
    ):
        _fail(
            "bytecode-cache-write-not-permitted",
            "invoke this precheck with python3 -B or PYTHONDONTWRITEBYTECODE=1",
        )


def _common(call: Callable[[], T]) -> T:
    try:
        return call()
    except common.ProvenanceV2Error as error:
        _fail(getattr(error, "reason_code", "unsafe-evidence"), str(error))


def _raw(call: Callable[[], T]) -> T:
    try:
        return call()
    except raw.RollbackV4ProvenanceError as error:
        _fail(getattr(error, "reason_code", "invalid-rollback-provenance"), str(error))


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


def _shared_lock(descriptor: int, reason_code: str, label: str) -> None:
    try:
        fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
    except (AttributeError, OSError) as error:
        _fail(reason_code, f"cannot acquire shared {label} lock: {error}")


def _unlock_quietly(descriptor: int | None) -> None:
    if descriptor is not None:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        except OSError:
            pass


def _close_quietly(descriptor: int | None) -> None:
    if descriptor is not None:
        try:
            os.close(descriptor)
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
            "rollback raw manifest dispatch header",
            maximum_bytes=raw.MAX_MANIFEST_BYTES,
        )
    )
    descriptor = _common(
        lambda: common.descriptor_for_bytes(
            manifest_name,
            manifest_raw,
            "rollback raw manifest dispatch header",
        )
    )
    document = _common(
        lambda: common.parse_canonical_json(
            manifest_raw,
            "rollback raw manifest dispatch header",
            maximum_bytes=raw.MAX_MANIFEST_BYTES,
        )
    )
    if type(document) is not dict:
        _fail(
            "unsupported-rollback-raw-manifest-version",
            "rollback raw manifest dispatch header must be a JSON object",
        )
    schema_version = document.get("schema_version")
    if type(schema_version) is not str:
        _fail(
            "unsupported-rollback-raw-manifest-version",
            "rollback raw manifest must contain a text schema_version",
        )
    return descriptor, schema_version


def _select_manifest_version(schema_version: str) -> tuple[str, str]:
    if schema_version == raw.ROLLBACK_V4_MANIFEST_VERSION:
        return raw.ROLLBACK_V4_REPORT_VERSION, "completed-v4-rollback-raw-provenance"
    historical_reason = _HISTORICAL_MANIFEST_REASONS.get(schema_version)
    if historical_reason is not None:
        _fail(
            historical_reason,
            f"rollback raw manifest version {schema_version!r} is not an admissible completed input",
        )
    _fail(
        "unsupported-rollback-raw-manifest-version",
        f"unsupported rollback raw manifest version {schema_version!r}",
    )


def _replay_completed_manifest_on_held_fds(
    root_fd: int,
    switch_fd: int,
    manifest_name: str,
    expected_report_version: str,
    primary_check: str,
) -> tuple[dict[str, Any], str, str]:
    report = _raw(
        lambda: raw.verify_completed_rollback_provenance_v4_on_held_switch_fd(
            root_fd,
            switch_fd,
            manifest_name,
        )
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
            "invalid-completed-rollback-provenance-report",
            "completed rollback raw provenance replay returned an invalid report shape",
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
        "rollback_v3_manifest": raw_report["rollback_v3_manifest"],
        "atomic_transaction_session": raw_report["atomic_transaction_session"],
        "checks": [
            {"name": primary_check, "bound": True},
            {
                "name": "header-to-completed-replay-descriptor-binding",
                "bound": True,
            },
        ],
        "reason_codes": [],
    }


def check_rc3_rollback_structural_precheck(
    evidence_root: Path,
    raw_manifest: str,
) -> dict[str, Any]:
    """Replay one completed rollback v4 manifest without semantic authority."""

    _require_bytecode_cache_disabled()
    root = _evidence_root(evidence_root)
    manifest_name = _raw_manifest_name(raw_manifest)
    root_fd = _common(
        lambda: common.open_private_evidence_directory(root, "--evidence-root")
    )
    switch_fd: int | None = None
    try:
        _shared_lock(root_fd, "evidence-root-lock-unavailable", "evidence-root")
        header_descriptor, manifest_version = _manifest_header(root_fd, manifest_name)
        expected_report_version, primary_check = _select_manifest_version(
            manifest_version
        )
        switch_fd = _common(
            lambda: common.open_private_child_directory(
                root_fd,
                prepare.SWITCH_DIRECTORY_NAME,
                "rollback switch directory",
            )
        )
        _shared_lock(switch_fd, "rollback-switch-lock-unavailable", "rollback switch")
        raw_report, report_version, primary_check = _replay_completed_manifest_on_held_fds(
            root_fd,
            switch_fd,
            manifest_name,
            expected_report_version,
            primary_check,
        )
        returned_descriptor = _common(
            lambda: common.parse_descriptor(
                raw_report["raw_manifest"],
                "completed rollback provenance report.raw_manifest",
            )
        )
        if returned_descriptor != header_descriptor:
            _fail(
                "raw-manifest-changed-during-version-dispatch",
                "raw manifest differs between version dispatch and completed replay",
            )
        if raw_report["schema_version"] != report_version:
            _fail(
                "invalid-completed-rollback-provenance-report",
                "completed rollback provenance report version changed during replay",
            )
        return _precheck_report(
            raw_report,
            raw_manifest_version=manifest_version,
            primary_check=primary_check,
        )
    finally:
        _unlock_quietly(switch_fd)
        _close_quietly(switch_fd)
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
        report = check_rc3_rollback_structural_precheck(
            args.evidence_root,
            args.raw_manifest,
        )
    except RollbackStructuralPrecheckError as error:
        print(f"RC3 rollback raw-structural precheck failed: {error}", file=sys.stderr)
        return 2
    sys.stdout.buffer.write(common.canonical_json_bytes(report) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
