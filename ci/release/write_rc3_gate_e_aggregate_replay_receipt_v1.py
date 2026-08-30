#!/usr/bin/env python3
"""Publish one private RC3 Gate E aggregate-replay terminal record.

This is a deliberately private held-FD continuation for a *future*
authenticated Gate E producer.  It is not a public path wrapper, resume
surface, producer finalizer, or semantic qualification receipt.  The caller
must already retain its ordered locks (freeze-input SH, frozen-candidate SH,
Gate E EX, and this fresh receipt root EX) for the full lexical call.

The helper replays the aggregate semantic core twice over the caller-held
evidence/source descriptors.  It records only a closed projection of the
matching canonical aggregate bytes in a separate, initially empty private
root.  In particular, it never writes into the exact Gate E input-inventory
root and does not establish actual capture, an authenticated producer normal
return, an actual candidate Gate E pass, a semantic receipt, or qualification.
"""

from __future__ import annotations

import hashlib
import os
import stat
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, NoReturn, TypeVar


_BYTECODE_DISABLED_AT_STARTUP = bool(sys.flags.dont_write_bytecode)
_BYTECODE_DISABLED_ON_MODULE_ENTRY = sys.dont_write_bytecode
sys.dont_write_bytecode = True

import provenance_v2_common as common
import rc3_frozen_candidate_common as frozen
import rc3_frozen_candidate_topology as topology
import replay_rc3_gate_e_aggregate_v1 as aggregate


RECEIPT_NAME = "rc3-gate-e-aggregate-replay-record-v1.json"
RECEIPT_VERSION = "riley.rc3-gate-e-aggregate-replay-record.v1"
RECEIPT_COMPLETION_VERSION = "riley.rc3-gate-e-aggregate-replay-record-complete.v1"
SCOPE = "gate-e-aggregate-semantic-replay-terminal-record-only"
AUTHORITY = aggregate.AUTHORITY
MAX_RECEIPT_BYTES = common.DEFAULT_MAX_JSON_BYTES

CHECK_NAMES = (
    "aggregate-semantic-replay-ran-twice-on-one-held-fd-stack",
    "aggregate-semantic-replay-canonical-bytes-match",
    "receipt-root-was-empty-before-fixed-publication",
    "terminal-record-is-aggregate-semantic-replay-only",
)

NOT_ESTABLISHED = {
    **aggregate.NOT_ESTABLISHED,
    "authenticated_gate_e_producer_normal_return": "not-established",
    "actual_gate_e_producer_normal_return": "not-established",
}


class GateEAggregateReplayReceiptError(ValueError):
    """The private Gate E aggregate-replay record cannot be published safely."""


def _fail(code: str, message: str) -> NoReturn:
    if code == "ambiguous-terminal-publication":
        message = f"{code}: {message}"
    error = GateEAggregateReplayReceiptError(message)
    error.reason_code = code  # type: ignore[attr-defined]
    raise error


T = TypeVar("T")


def _common(call: Callable[[], T]) -> T:
    try:
        return call()
    except common.ProvenanceV2Error as error:
        _fail(getattr(error, "reason_code", "unsafe-evidence"), str(error))


def _frozen(call: Callable[[], T]) -> T:
    try:
        return call()
    except frozen.FrozenCandidateError as error:
        _fail(getattr(error, "reason_code", "invalid-frozen-candidate"), str(error))


def _topology(call: Callable[[], T]) -> T:
    try:
        return call()
    except topology.FrozenCandidateTopologyError as error:
        _fail(getattr(error, "reason_code", "unsafe-gate-e-topology"), str(error))


def _aggregate(call: Callable[[], T]) -> T:
    try:
        return call()
    except aggregate.AggregateReplayError as error:
        _fail(getattr(error, "reason_code", "aggregate-replay-failed"), str(error))


def _require_bytecode_cache_disabled() -> None:
    if not (
        _BYTECODE_DISABLED_AT_STARTUP and _BYTECODE_DISABLED_ON_MODULE_ENTRY
    ):
        _fail(
            "bytecode-cache-write-not-permitted",
            "invoke this private helper with python3 -B or PYTHONDONTWRITEBYTECODE=1",
        )


def _receipt_output_names() -> tuple[str, str, str]:
    return RECEIPT_NAME, f"{RECEIPT_NAME}.intent", f"{RECEIPT_NAME}.complete"


def _assert_empty_receipt_root(receipt_root_fd: int) -> None:
    try:
        entries = os.listdir(receipt_root_fd)
    except OSError as error:
        _fail("receipt-root-preflight-failed", f"cannot list private receipt root: {error}")
    if entries:
        _fail(
            "receipt-root-not-empty",
            "private aggregate-replay receipt root must be completely empty before publication",
        )


def _normalized_visible_roots(
    receipt_root: Path,
    gate_e_evidence_root: Path,
    frozen_candidate_root: Path,
    input_evidence_root: Path,
    repository_root: Path,
    scratch_parent: Path,
) -> dict[str, Path]:
    roots = {
        "Gate E aggregate-replay receipt root": _frozen(
            lambda: frozen.normalized_absolute_path(
                receipt_root,
                "Gate E aggregate-replay receipt root",
            )
        ),
        "Gate E evidence root": _frozen(
            lambda: frozen.normalized_absolute_path(
                gate_e_evidence_root,
                "Gate E evidence root",
            )
        ),
        "frozen candidate root": _frozen(
            lambda: frozen.normalized_absolute_path(
                frozen_candidate_root,
                "frozen candidate root",
            )
        ),
        "freeze-input evidence root": _frozen(
            lambda: frozen.normalized_absolute_path(
                input_evidence_root,
                "freeze-input evidence root",
            )
        ),
        "source checkout": _frozen(
            lambda: frozen.normalized_absolute_path(repository_root, "source checkout")
        ),
        "aggregate external scratch parent": _frozen(
            lambda: frozen.normalized_absolute_path(
                scratch_parent,
                "aggregate external scratch parent",
            )
        ),
    }
    _frozen(lambda: frozen.require_disjoint_paths(roots))
    if roots["aggregate external scratch parent"] != aggregate.EXTERNAL_SCRATCH_PARENT:
        _fail(
            "invalid-scratch-parent",
            "aggregate-replay terminal record accepts only the fixed external scratch parent",
        )
    return roots


def _assert_held_topology(
    receipt_root: Path,
    receipt_root_fd: int,
    gate_e_evidence_root: Path,
    gate_e_evidence_root_fd: int,
    frozen_candidate_root: Path,
    frozen_candidate_root_fd: int,
    input_evidence_root: Path,
    input_evidence_root_fd: int,
    repository_root: Path,
    repository_root_fd: int,
    scratch_parent: Path,
) -> None:
    """Bind caller-held roots before every aggregate replay or publication."""

    _common(
        lambda: common.require_private_evidence_directory_fd(
            receipt_root_fd,
            "Gate E aggregate-replay receipt root",
        )
    )
    _common(
        lambda: common.require_private_evidence_directory_fd(
            gate_e_evidence_root_fd,
            "Gate E evidence root",
        )
    )
    _common(
        lambda: common.require_private_evidence_directory_fd(
            frozen_candidate_root_fd,
            "frozen candidate root",
        )
    )
    _common(
        lambda: common.require_private_evidence_directory_fd(
            input_evidence_root_fd,
            "freeze-input evidence root",
        )
    )
    paths = _normalized_visible_roots(
        receipt_root,
        gate_e_evidence_root,
        frozen_candidate_root,
        input_evidence_root,
        repository_root,
        scratch_parent,
    )
    scratch_parent_fd: int | None = None
    try:
        scratch_parent_fd = _common(
            lambda: common.open_absolute_directory(
                paths["aggregate external scratch parent"],
                "aggregate external scratch parent",
            )
        )
        held_fds = {
            "Gate E aggregate-replay receipt root": receipt_root_fd,
            "Gate E evidence root": gate_e_evidence_root_fd,
            "frozen candidate root": frozen_candidate_root_fd,
            "freeze-input evidence root": input_evidence_root_fd,
            "source checkout": repository_root_fd,
            "aggregate external scratch parent": scratch_parent_fd,
        }
        _frozen(lambda: frozen.require_distinct_root_fds(held_fds))
        _topology(
            lambda: topology.assert_existing_roots_disjoint(
                {
                    label: (paths[label], descriptor)
                    for label, descriptor in held_fds.items()
                }
            )
        )
    finally:
        if scratch_parent_fd is not None:
            try:
                os.close(scratch_parent_fd)
            except OSError:
                pass


def _descriptor_json(value: Any, label: str) -> dict[str, Any]:
    descriptor = _common(lambda: common.parse_descriptor(value, label))
    return descriptor.as_json()


def _canonical_aggregate_projection(value: Any) -> tuple[dict[str, Any], bytes]:
    """Validate and reduce one freshly returned aggregate core report."""

    if type(value) is not dict:
        _fail("invalid-aggregate-result", "aggregate held-FD core returned no typed object")
    expected_fields = {
        "schema_version",
        "scope",
        "status",
        "authority",
        "candidate_status",
        "qualification_status",
        "gate_e_status",
        "candidate_id",
        "source_revision",
        "expected_release_image_id",
        "expected_optimizer_build_image_id",
        "expected_correctness_golden_sha256",
        "aggregate_policy_version",
        "aggregate_policy_sha256",
        "gate_e_input_inventory",
        "frozen_candidate_manifest",
        "components",
        "shared_bindings",
        "checks",
        "not_established",
        "reason_codes",
    }
    if set(value) != expected_fields:
        _fail(
            "invalid-aggregate-result",
            "aggregate held-FD core returned an unsupported result shape",
        )
    fixed = {
        "schema_version": aggregate.REPLAY_VERSION,
        "scope": aggregate.SCOPE,
        "status": "bound",
        "authority": aggregate.AUTHORITY,
        "candidate_status": "frozen",
        "qualification_status": "not-run",
        "gate_e_status": "passed",
        "aggregate_policy_version": aggregate.AGGREGATE_POLICY_VERSION,
        "aggregate_policy_sha256": aggregate.AGGREGATE_POLICY_SHA256,
        "checks": [
            {"name": check_name, "satisfied": True}
            for check_name in aggregate.CHECK_NAMES
        ],
        "not_established": dict(aggregate.NOT_ESTABLISHED),
        "reason_codes": [],
    }
    for field, expected in fixed.items():
        if value[field] != expected:
            _fail("invalid-aggregate-result", f"aggregate held-FD core has unexpected {field}")
    candidate_id = value["candidate_id"]
    if (
        type(candidate_id) is not str
        or frozen.CANDIDATE_ID_RE.fullmatch(candidate_id) is None
    ):
        _fail("invalid-aggregate-result", "aggregate candidate ID is invalid")
    source_revision = value["source_revision"]
    if (
        type(source_revision) is not str
        or source_revision == "0" * 40
        or aggregate.GIT_REVISION_RE.fullmatch(source_revision) is None
    ):
        _fail("invalid-aggregate-result", "aggregate source revision is invalid")
    for field, expression, label in (
        (
            "expected_release_image_id",
            aggregate.IMAGE_ID_RE,
            "aggregate expected release image ID",
        ),
        (
            "expected_optimizer_build_image_id",
            aggregate.IMAGE_ID_RE,
            "aggregate expected optimizer image ID",
        ),
        (
            "expected_correctness_golden_sha256",
            aggregate.SHA256_RE,
            "aggregate expected correctness golden SHA-256",
        ),
    ):
        text = value[field]
        if type(text) is not str or expression.fullmatch(text) is None:
            _fail("invalid-aggregate-result", f"{label} is invalid")
    if value["expected_release_image_id"] == "sha256:" + "0" * 64:
        _fail("invalid-aggregate-result", "aggregate expected release image ID is zero")
    if value["expected_optimizer_build_image_id"] == "sha256:" + "0" * 64:
        _fail("invalid-aggregate-result", "aggregate expected optimizer image ID is zero")
    if value["expected_correctness_golden_sha256"] == "0" * 64:
        _fail("invalid-aggregate-result", "aggregate expected correctness golden SHA-256 is zero")
    if type(value["components"]) is not dict or set(value["components"]) != set(
        aggregate._COMPONENT_EXPECTATIONS  # noqa: SLF001
    ):
        _fail("invalid-aggregate-result", "aggregate component projection is invalid")
    if type(value["shared_bindings"]) is not dict:
        _fail("invalid-aggregate-result", "aggregate shared bindings are invalid")

    raw = _common(lambda: common.canonical_json_bytes(value))
    if not raw or len(raw) > MAX_RECEIPT_BYTES:
        _fail("aggregate-report-too-large", "canonical aggregate report exceeds receipt limits")
    parsed = _common(
        lambda: common.parse_canonical_json(
            raw,
            "canonical aggregate semantic replay report",
            maximum_bytes=MAX_RECEIPT_BYTES,
        )
    )
    if type(parsed) is not dict or parsed != value:
        _fail(
            "invalid-aggregate-result",
            "aggregate report cannot round-trip as canonical JSON",
        )
    projection = {
        "schema_version": value["schema_version"],
        "scope": value["scope"],
        "authority": value["authority"],
        "status": value["status"],
        "candidate_status": value["candidate_status"],
        "qualification_status": value["qualification_status"],
        "candidate_id": candidate_id,
        "source_revision": source_revision,
        "expected_release_image_id": value["expected_release_image_id"],
        "expected_optimizer_build_image_id": value["expected_optimizer_build_image_id"],
        "expected_correctness_golden_sha256": value[
            "expected_correctness_golden_sha256"
        ],
        "aggregate_policy_version": value["aggregate_policy_version"],
        "aggregate_policy_sha256": value["aggregate_policy_sha256"],
        "gate_e_input_inventory": _descriptor_json(
            value["gate_e_input_inventory"],
            "aggregate Gate E input inventory",
        ),
        "frozen_candidate_manifest": _descriptor_json(
            value["frozen_candidate_manifest"],
            "aggregate frozen candidate manifest",
        ),
        "report_sha256": hashlib.sha256(raw).hexdigest(),
        "report_byte_length": len(raw),
    }
    return projection, raw


def _record_document(projection: Mapping[str, Any]) -> dict[str, Any]:
    if type(projection) is not dict:
        _fail("invalid-aggregate-projection", "aggregate projection must be a typed object")
    expected_projection = {
        "schema_version",
        "scope",
        "authority",
        "status",
        "candidate_status",
        "qualification_status",
        "candidate_id",
        "source_revision",
        "expected_release_image_id",
        "expected_optimizer_build_image_id",
        "expected_correctness_golden_sha256",
        "aggregate_policy_version",
        "aggregate_policy_sha256",
        "gate_e_input_inventory",
        "frozen_candidate_manifest",
        "report_sha256",
        "report_byte_length",
    }
    if set(projection) != expected_projection:
        _fail("invalid-aggregate-projection", "aggregate projection has an unsupported shape")
    return {
        "schema_version": RECEIPT_VERSION,
        "scope": SCOPE,
        "status": "bound",
        "authority": AUTHORITY,
        "candidate_status": "frozen",
        "qualification_status": "not-run",
        "candidate_id": projection["candidate_id"],
        "source_revision": projection["source_revision"],
        "expected_release_image_id": projection["expected_release_image_id"],
        "expected_optimizer_build_image_id": projection[
            "expected_optimizer_build_image_id"
        ],
        "expected_correctness_golden_sha256": projection[
            "expected_correctness_golden_sha256"
        ],
        "aggregate_policy_version": projection["aggregate_policy_version"],
        "aggregate_policy_sha256": projection["aggregate_policy_sha256"],
        "gate_e_input_inventory": projection["gate_e_input_inventory"],
        "frozen_candidate_manifest": projection["frozen_candidate_manifest"],
        "aggregate_replay": {
            "report_sha256": projection["report_sha256"],
            "report_byte_length": projection["report_byte_length"],
        },
        "checks": [
            {"name": check_name, "satisfied": True}
            for check_name in CHECK_NAMES
        ],
        "not_established": dict(NOT_ESTABLISHED),
        "reason_codes": [],
    }


def _completion_pair_is_visible(receipt_root_fd: int) -> bool:
    try:
        final = os.lstat(f"{RECEIPT_NAME}.complete", dir_fd=receipt_root_fd)
        intent = os.lstat(f"{RECEIPT_NAME}.intent", dir_fd=receipt_root_fd)
    except OSError:
        return False
    return (
        stat.S_ISREG(final.st_mode)
        and stat.S_ISREG(intent.st_mode)
        and stat.S_IMODE(final.st_mode) == 0o600
        and stat.S_IMODE(intent.st_mode) == 0o600
        and final.st_uid == os.geteuid()
        and intent.st_uid == os.geteuid()
        and final.st_nlink == 2
        and intent.st_nlink == 2
        and (final.st_dev, final.st_ino) == (intent.st_dev, intent.st_ino)
    )


def _replay_and_publish_gate_e_aggregate_receipt_on_held_fds(
    receipt_root: Path,
    receipt_root_fd: int,
    gate_e_evidence_root: Path,
    gate_e_evidence_root_fd: int,
    frozen_candidate_root: Path,
    frozen_candidate_root_fd: int,
    input_evidence_root: Path,
    input_evidence_root_fd: int,
    repository_root: Path,
    repository_root_fd: int,
    scratch_parent: Path,
    expected_release_image_id: str,
    expected_optimizer_build_image_id: str,
    expected_correctness_golden_sha256: str,
) -> dict[str, Any]:
    """Publish one terminal replay-only record from two matching held-FD replays.

    There is intentionally no CLI, path-resume API, caller-selected receipt
    name, callback, producer closure, or raw capture claim.  The caller owns
    every lock for this lexical call.  Once the final hard link succeeds this
    helper immediately returns; a visible completion pair after a failed
    durability check is ambiguous and cannot be resumed.
    """

    _require_bytecode_cache_disabled()
    paths = _normalized_visible_roots(
        receipt_root,
        gate_e_evidence_root,
        frozen_candidate_root,
        input_evidence_root,
        repository_root,
        scratch_parent,
    )
    _assert_held_topology(
        paths["Gate E aggregate-replay receipt root"],
        receipt_root_fd,
        paths["Gate E evidence root"],
        gate_e_evidence_root_fd,
        paths["frozen candidate root"],
        frozen_candidate_root_fd,
        paths["freeze-input evidence root"],
        input_evidence_root_fd,
        paths["source checkout"],
        repository_root_fd,
        paths["aggregate external scratch parent"],
    )
    _assert_empty_receipt_root(receipt_root_fd)
    first = _aggregate(
        lambda: aggregate._replay_rc3_gate_e_aggregate_v1_on_held_fds(  # noqa: SLF001
            gate_e_evidence_root_fd,
            frozen_candidate_root_fd,
            input_evidence_root_fd,
            paths["source checkout"],
            repository_root_fd,
            paths["aggregate external scratch parent"],
            expected_release_image_id,
            expected_optimizer_build_image_id,
            expected_correctness_golden_sha256,
        )
    )
    first_projection, first_raw = _canonical_aggregate_projection(first)
    _assert_held_topology(
        paths["Gate E aggregate-replay receipt root"],
        receipt_root_fd,
        paths["Gate E evidence root"],
        gate_e_evidence_root_fd,
        paths["frozen candidate root"],
        frozen_candidate_root_fd,
        paths["freeze-input evidence root"],
        input_evidence_root_fd,
        paths["source checkout"],
        repository_root_fd,
        paths["aggregate external scratch parent"],
    )
    _assert_empty_receipt_root(receipt_root_fd)
    second = _aggregate(
        lambda: aggregate._replay_rc3_gate_e_aggregate_v1_on_held_fds(  # noqa: SLF001
            gate_e_evidence_root_fd,
            frozen_candidate_root_fd,
            input_evidence_root_fd,
            paths["source checkout"],
            repository_root_fd,
            paths["aggregate external scratch parent"],
            expected_release_image_id,
            expected_optimizer_build_image_id,
            expected_correctness_golden_sha256,
        )
    )
    second_projection, second_raw = _canonical_aggregate_projection(second)
    if second_raw != first_raw or second_projection != first_projection:
        _fail(
            "aggregate-replay-drift",
            "two held-FD aggregate semantic replays did not produce identical canonical bytes",
        )
    document = _record_document(first_projection)
    raw_document = _common(lambda: common.canonical_json_bytes(document))
    if not raw_document or len(raw_document) > MAX_RECEIPT_BYTES:
        _fail("receipt-too-large", "aggregate-replay terminal record exceeds receipt limits")
    draft_descriptor = _common(
        lambda: common.descriptor_for_bytes(
            RECEIPT_NAME,
            raw_document,
            "Gate E aggregate-replay record draft",
        )
    )
    _assert_held_topology(
        paths["Gate E aggregate-replay receipt root"],
        receipt_root_fd,
        paths["Gate E evidence root"],
        gate_e_evidence_root_fd,
        paths["frozen candidate root"],
        frozen_candidate_root_fd,
        paths["freeze-input evidence root"],
        input_evidence_root_fd,
        paths["source checkout"],
        repository_root_fd,
        paths["aggregate external scratch parent"],
    )
    _assert_empty_receipt_root(receipt_root_fd)
    final_document = _record_document(second_projection)
    if final_document != document:
        _fail(
            "receipt-closure-drift",
            "aggregate-replay terminal record changed during its final held-FD replay",
        )
    created = _common(
        lambda: common.write_create_only_json(
            receipt_root_fd,
            RECEIPT_NAME,
            document,
            "Gate E aggregate-replay record",
        )
    )
    created_descriptor = created.descriptor(
        RECEIPT_NAME,
        "Gate E aggregate-replay record",
    )
    if created_descriptor != draft_descriptor:
        _fail(
            "published-receipt-descriptor-mismatch",
            "published aggregate-replay record differs from its held-FD draft",
        )
    _common(
        lambda: common.write_create_only_json(
            receipt_root_fd,
            f"{RECEIPT_NAME}.intent",
            {
                "schema_version": RECEIPT_COMPLETION_VERSION,
                "artifact_filename": RECEIPT_NAME,
                "artifact_sha256": created_descriptor.sha256,
            },
            "Gate E aggregate-replay record completion marker intent",
        )
    )
    try:
        _common(
            lambda: common.publish_create_only_hardlink(
                receipt_root_fd,
                f"{RECEIPT_NAME}.intent",
                f"{RECEIPT_NAME}.complete",
                "Gate E aggregate-replay record completion marker",
            )
        )
        return document
    except GateEAggregateReplayReceiptError:
        if _completion_pair_is_visible(receipt_root_fd):
            _fail(
                "ambiguous-terminal-publication",
                "completion marker became visible but final directory sync failed; "
                "no later invocation may treat the replay-only record as normal-return authority",
            )
        raise
