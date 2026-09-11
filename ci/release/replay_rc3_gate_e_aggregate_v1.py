#!/usr/bin/env python3
"""Replay all RC3 Gate E semantic components through one held-FD boundary.

This adapter is an aggregate *semantic replay*, not a release qualification.
It opens and locks the four evidence/source roots once, holds those file
descriptors while it invokes the five component-private replay cores, and
replays the closed Gate E inventory before and after the entire sequence.
The adapter never consumes a previously serialized component result as
authority: every result below is produced by a private held-FD core in this
process.

The only positive conclusion is that the five component replays agree about
the same frozen candidate and shared evidence descriptors.  It does not
establish capture execution, any human review, a semantic receipt,
qualification, deployment, or rollback success.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, NoReturn, Sequence, TypeVar


_BYTECODE_DISABLED_AT_STARTUP = bool(sys.flags.dont_write_bytecode)
_BYTECODE_DISABLED_ON_MODULE_ENTRY = sys.dont_write_bytecode
sys.dont_write_bytecode = True

import provenance_v2_common as common
import rc3_frozen_candidate_common as frozen
import rc3_frozen_candidate_topology as topology
import replay_rc3_gate_e_input_inventory_v1 as gate_inputs
import replay_rc3_gate_e_native_e0_v1 as native_e0
import replay_rc3_gate_e_optimizer_e0_v1 as optimizer_e0
import replay_rc3_gate_e_performance_v1 as performance
import replay_rc3_gate_e_python_free_v1 as python_free
import replay_rc3_gate_e_soak_v1 as soak


REPLAY_VERSION = "riley.rc3-gate-e-aggregate-semantic-replay.v1"
SCOPE = "gate-e-aggregate-semantic-replay-only"
AUTHORITY = "gate-e-aggregate-semantic-replay-only"
AGGREGATE_POLICY_VERSION = "riley.rc3-gate-e-aggregate-policy.v1"
# This is the SHA-256 of ``_AGGREGATE_POLICY_PROJECTION`` below.  It pins the
# aggregate's child-contract projection independently of any saved reports.
AGGREGATE_POLICY_SHA256 = "71ca2c4413ce3939e072b7109a17f36873446eaf34a2cc6d7e076803551f49a8"
EXTERNAL_SCRATCH_PARENT = Path("/var/tmp")

IMAGE_ID_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")

_COMPONENT_EXPECTATIONS = {
    "native_e0": {
        "schema_version": native_e0.REPLAY_VERSION,
        "scope": native_e0.SCOPE,
        "authority": native_e0.AUTHORITY,
        "status_field": "native_e0_status",
        "payload_field": "native_e0",
        "check_names": native_e0.CHECK_NAMES,
        "not_established": native_e0.NOT_ESTABLISHED,
        "anchor_fields": (),
    },
    "optimizer_e0": {
        "schema_version": optimizer_e0.REPLAY_VERSION,
        "scope": optimizer_e0.SCOPE,
        "authority": optimizer_e0.AUTHORITY,
        "status_field": "optimizer_e0_status",
        "payload_field": "optimizer_e0",
        "check_names": optimizer_e0.CHECK_NAMES,
        "not_established": optimizer_e0.NOT_ESTABLISHED,
        "anchor_fields": ("expected_optimizer_build_image_id",),
    },
    "python_free": {
        "schema_version": python_free.REPLAY_VERSION,
        "scope": python_free.SCOPE,
        "authority": python_free.AUTHORITY,
        "status_field": "python_free_status",
        "payload_field": "python_free",
        "check_names": python_free.CHECK_NAMES,
        "not_established": python_free.NOT_ESTABLISHED,
        "anchor_fields": (
            "expected_release_image_id",
            "expected_correctness_golden_sha256",
        ),
    },
    "performance": {
        "schema_version": performance.REPLAY_VERSION,
        "scope": performance.SCOPE,
        "authority": performance.AUTHORITY,
        "status_field": "performance_status",
        "payload_field": "performance",
        "check_names": performance.CHECK_NAMES,
        "not_established": performance.NOT_ESTABLISHED,
        "anchor_fields": (
            "expected_release_image_id",
            "expected_optimizer_build_image_id",
        ),
    },
    "soak": {
        "schema_version": soak.REPLAY_VERSION,
        "scope": soak.SCOPE,
        "authority": soak.AUTHORITY,
        "status_field": "soak_status",
        "payload_field": "soak",
        "check_names": soak.CHECK_NAMES,
        "not_established": soak.NOT_ESTABLISHED,
        "anchor_fields": (
            "expected_release_image_id",
            "expected_correctness_golden_sha256",
        ),
    },
}

CHECK_NAMES = (
    "closed-gate-e-input-inventory-replayed-before-aggregate",
    "five-private-held-fd-semantic-components-replayed",
    "component-candidate-inventory-and-frozen-manifest-bindings-match",
    "shared-source-release-e0-and-golden-bindings-match",
    "externally-supplied-release-optimizer-and-golden-anchors-bound",
    "closed-gate-e-input-inventory-replayed-after-aggregate",
    "aggregate-is-semantic-replay-only",
)

NOT_ESTABLISHED = {
    "actual_gpu_capture": "not-established",
    "actual_capture": "not-established",
    "actual_candidate_gate_e_pass": "not-established",
    "frozen_candidate_writer_normal_return": "not-established",
    "evidence_root_immutability": "not-established",
    "release_image_review": "not-established",
    "optimizer_build_image_review": "not-established",
    "correctness_golden_review": "not-established",
    "host_gpu_test_layer_review": "not-established",
    "release_binary_provenance": "not-established",
    "release_container_content": "not-established",
    "model_content": "not-established",
    "model_mount_provenance": "not-established",
    "producer_sidecar_equality": "not-established",
    "source_archive_content": "not-established",
    "startup_configuration": "not-established",
    "qwen_regression": "not-established",
    "routing": "not-established",
    "fault_extension": "not-established",
    "reproducible_build": "not-established",
    "dependency_manifest": "not-established",
    "semantic_receipt": "not-established",
    "qualification": "not-established",
    "deployment": "not-established",
    "rollback_execution": "not-established",
    "rollback_success": "not-established",
    "rollback_semantic_receipt": "not-established",
    "rollback_result": "not-established",
}

_AGGREGATE_POLICY_PROJECTION = {
    "policy_version": AGGREGATE_POLICY_VERSION,
    "inventory_contract": {
        "schema_version": gate_inputs.REPLAY_VERSION,
        "scope": gate_inputs.SCOPE,
        "authority": gate_inputs.AUTHORITY,
        "status": "bound",
        "candidate_status": "frozen",
        "qualification_status": "not-run",
        "checks": list(gate_inputs.CHECK_NAMES),
        "not_established": gate_inputs.NOT_ESTABLISHED,
    },
    "aggregate_contract": {
        "schema_version": REPLAY_VERSION,
        "scope": SCOPE,
        "authority": AUTHORITY,
        "status": "bound",
        "candidate_status": "frozen",
        "qualification_status": "not-run",
        "gate_e_status": "passed",
        "checks": list(CHECK_NAMES),
        "not_established": NOT_ESTABLISHED,
    },
    "child_contracts": {
        name: {
            "schema_version": expectation["schema_version"],
            "scope": expectation["scope"],
            "authority": expectation["authority"],
            "status": "bound",
            "candidate_status": "frozen",
            "qualification_status": "not-run",
            "status_field": expectation["status_field"],
            "status_value": "passed",
            "anchor_fields": list(expectation["anchor_fields"]),
            "checks": list(expectation["check_names"]),
            "not_established": expectation["not_established"],
        }
        for name, expectation in _COMPONENT_EXPECTATIONS.items()
    },
    "root_stack": [
        "source checkout",
        "freeze-input evidence root",
        "frozen candidate root",
        "Gate E evidence root",
        "aggregate external scratch parent",
    ],
    "external_anchors": [
        "expected_release_image_id",
        "expected_optimizer_build_image_id",
        "expected_correctness_golden_sha256",
    ],
    "dependency_policy_pins": {
        "optimizer_final_report_contract_sha256": (
            optimizer_e0.EXPECTED_FINAL_REPORT_CONTRACT_POLICY_SHA256
        ),
        "performance_bound": {
            "version": performance.PERFORMANCE_POLICY_VERSION,
            "sha256": performance.EXPECTED_PERFORMANCE_POLICY_SHA256,
        },
        "performance_optimizer_contract_sha256": (
            performance.EXPECTED_OPTIMIZER_CONTRACT_POLICY_SHA256
        ),
        "soak_bound": {
            "version": soak.SOAK_POLICY_VERSION,
            "sha256": soak.EXPECTED_SOAK_POLICY_SHA256,
        },
    },
    "shared_bindings": [
        "source_archive",
        "release_elf",
        "profile_binary",
        "canonical_native_report",
        "canonical_optimizer_report",
        "correctness_golden",
    ],
}

T = TypeVar("T")


class AggregateReplayError(ValueError):
    """The aggregate Gate E semantic replay cannot be performed safely."""


def _fail(code: str, message: str) -> NoReturn:
    error = AggregateReplayError(f"{code}: {message}")
    error.reason_code = code  # type: ignore[attr-defined]
    raise error


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


def _gate(call: Callable[[], T]) -> T:
    try:
        return call()
    except gate_inputs.GateEInventoryReplayError as error:
        _fail(getattr(error, "reason_code", "invalid-gate-e-inventory"), str(error))


def _require_bytecode_cache_disabled() -> None:
    if not (
        _BYTECODE_DISABLED_AT_STARTUP and _BYTECODE_DISABLED_ON_MODULE_ENTRY
    ):
        _fail(
            "bytecode-cache-write-not-permitted",
            "invoke this replayer with python3 -B or PYTHONDONTWRITEBYTECODE=1",
        )


def _require_aggregate_policy() -> None:
    try:
        actual = hashlib.sha256(
            common.canonical_json_bytes(_AGGREGATE_POLICY_PROJECTION)
        ).hexdigest()
    except common.ProvenanceV2Error as error:
        _fail(getattr(error, "reason_code", "aggregate-policy-invalid"), str(error))
    if actual != AGGREGATE_POLICY_SHA256:
        _fail("aggregate-policy-drift", "aggregate policy projection SHA-256 differs from its pin")


def _shared_lock(directory_fd: int, label: str) -> None:
    try:
        fcntl.flock(directory_fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
    except (AttributeError, OSError):
        _fail("evidence-root-lock-unavailable", f"cannot acquire shared {label} lock")


def _unlock_quietly(directory_fd: int | None) -> None:
    if directory_fd is not None:
        try:
            fcntl.flock(directory_fd, fcntl.LOCK_UN)
        except OSError:
            pass


def _close_quietly(directory_fd: int | None) -> None:
    if directory_fd is not None:
        try:
            os.close(directory_fd)
        except OSError:
            pass


def _expected_image_id(value: str, label: str) -> str:
    if type(value) is not str or IMAGE_ID_RE.fullmatch(value) is None:
        _fail(f"invalid-{label}", f"{label.replace('-', ' ')} must be sha256:<64 lowercase hex>")
    if value == "sha256:" + "0" * 64:
        _fail(f"invalid-{label}", f"{label.replace('-', ' ')} must not be the zero digest")
    return value


def _expected_sha256(value: str, label: str) -> str:
    if type(value) is not str or SHA256_RE.fullmatch(value) is None or value == "0" * 64:
        _fail(f"invalid-{label}", f"{label.replace('-', ' ')} must be nonzero lowercase SHA-256")
    return value


def _typed_mapping(value: Any, label: str, *, code: str) -> dict[str, Any]:
    if type(value) is not dict:
        _fail(code, f"{label} returned no typed object")
    return value


def _typed_string(value: Any, label: str, *, code: str) -> str:
    if type(value) is not str or not value:
        _fail(code, f"{label} returned no nonempty string")
    return value


def _descriptor(value: Any, label: str, *, code: str) -> dict[str, Any]:
    try:
        descriptor = common.parse_descriptor(value, label)
    except common.ProvenanceV2Error:
        _fail(code, f"{label} is not a valid evidence descriptor")
    return descriptor.as_json()


def _same(left: Any, right: Any, *, code: str, message: str) -> None:
    if left != right:
        _fail(code, message)


def _same_descriptor_identity(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    code: str,
    message: str,
) -> None:
    if (
        left.get("sha256") != right.get("sha256")
        or left.get("byte_length") != right.get("byte_length")
    ):
        _fail(code, message)


def _component_call(name: str, call: Callable[[], Any]) -> dict[str, Any]:
    try:
        value = call()
    except (
        native_e0.NativeE0ReplayError,
        optimizer_e0.OptimizerE0ReplayError,
        python_free.PythonFreeReplayError,
        performance.PerformanceReplayError,
        soak.SoakReplayError,
    ) as error:
        reason = getattr(error, "reason_code", "component-replay-failed")
        if type(reason) is not str or not reason:
            reason = "component-replay-failed"
        _fail(f"{name}-component-{reason}", f"{name} semantic component replay failed")
    return _typed_mapping(value, name, code="invalid-component-result")


def _validate_structural_result(value: Any, *, phase: str) -> dict[str, Any]:
    result = _typed_mapping(value, f"Gate E inventory {phase}", code="invalid-gate-e-inventory")
    expected = {
        "schema_version": gate_inputs.REPLAY_VERSION,
        "scope": gate_inputs.SCOPE,
        "status": "bound",
        "authority": gate_inputs.AUTHORITY,
        "candidate_status": "frozen",
        "qualification_status": "not-run",
        "checks": [
            {"name": check_name, "satisfied": True}
            for check_name in gate_inputs.CHECK_NAMES
        ],
        "not_established": dict(gate_inputs.NOT_ESTABLISHED),
        "reason_codes": [],
    }
    for field, expected_value in expected.items():
        if result.get(field) != expected_value:
            _fail("invalid-gate-e-inventory", f"Gate E inventory {phase} has unexpected {field}")
    candidate_id = _typed_string(
        result.get("candidate_id"),
        f"Gate E inventory {phase} candidate ID",
        code="invalid-gate-e-inventory",
    )
    if frozen.CANDIDATE_ID_RE.fullmatch(candidate_id) is None:
        _fail("invalid-gate-e-inventory", f"Gate E inventory {phase} candidate ID is invalid")
    source_revision = _typed_string(
        result.get("source_revision"),
        f"Gate E inventory {phase} source revision",
        code="invalid-gate-e-inventory",
    )
    if source_revision == "0" * 40 or GIT_REVISION_RE.fullmatch(source_revision) is None:
        _fail("invalid-gate-e-inventory", f"Gate E inventory {phase} source revision is invalid")
    return {
        "candidate_id": candidate_id,
        "source_revision": source_revision,
        "gate_e_input_inventory": _descriptor(
            result.get("gate_e_input_inventory"),
            f"Gate E inventory {phase} descriptor",
            code="invalid-gate-e-inventory",
        ),
        "frozen_candidate_manifest": _descriptor(
            result.get("frozen_candidate_manifest"),
            f"Gate E inventory {phase} frozen manifest descriptor",
            code="invalid-gate-e-inventory",
        ),
    }


def _validate_component_result(name: str, value: Mapping[str, Any]) -> dict[str, Any]:
    expectation = _COMPONENT_EXPECTATIONS[name]
    expected = {
        "schema_version": expectation["schema_version"],
        "scope": expectation["scope"],
        "status": "bound",
        "authority": expectation["authority"],
        "candidate_status": "frozen",
        "qualification_status": "not-run",
        expectation["status_field"]: "passed",
        "checks": [
            {"name": check_name, "satisfied": True}
            for check_name in expectation["check_names"]
        ],
        "not_established": dict(expectation["not_established"]),
        "reason_codes": [],
    }
    for field, expected_value in expected.items():
        if value.get(field) != expected_value:
            _fail("invalid-component-result", f"{name} component has unexpected {field}")
    candidate_id = _typed_string(
        value.get("candidate_id"),
        f"{name} component candidate ID",
        code="invalid-component-result",
    )
    if frozen.CANDIDATE_ID_RE.fullmatch(candidate_id) is None:
        _fail("invalid-component-result", f"{name} component candidate ID is invalid")
    source_revision = _typed_string(
        value.get("source_revision"),
        f"{name} component source revision",
        code="invalid-component-result",
    )
    if source_revision == "0" * 40 or GIT_REVISION_RE.fullmatch(source_revision) is None:
        _fail("invalid-component-result", f"{name} component source revision is invalid")
    payload_field = expectation["payload_field"]
    anchors: dict[str, str] = {}
    for field in expectation["anchor_fields"]:
        anchor = _typed_string(
            value.get(field),
            f"{name} component {field}",
            code="invalid-component-result",
        )
        if field.endswith("sha256"):
            anchors[field] = _expected_sha256(anchor, field.replace("_", "-"))
        else:
            anchors[field] = _expected_image_id(anchor, field.replace("_", "-"))
    return {
        "schema_version": value["schema_version"],
        "scope": value["scope"],
        "authority": value["authority"],
        "status": value["status"],
        "status_field": expectation["status_field"],
        "component_status": value[expectation["status_field"]],
        "candidate_id": candidate_id,
        "source_revision": source_revision,
        "gate_e_input_inventory": _descriptor(
            value.get("gate_e_input_inventory"),
            f"{name} component Gate E inventory descriptor",
            code="invalid-component-result",
        ),
        "frozen_candidate_manifest": _descriptor(
            value.get("frozen_candidate_manifest"),
            f"{name} component frozen manifest descriptor",
            code="invalid-component-result",
        ),
        "payload": _typed_mapping(
            value.get(payload_field),
            f"{name} component payload",
            code="invalid-component-result",
        ),
        "anchors": anchors,
    }


def _payload_descriptor(component: str, payload: Mapping[str, Any], field: str) -> dict[str, Any]:
    return _descriptor(
        payload.get(field),
        f"{component} component {field}",
        code="invalid-component-result",
    )


def _require_common_component_bindings(
    structural: Mapping[str, Any],
    components: Mapping[str, Mapping[str, Any]],
) -> None:
    for name in _COMPONENT_EXPECTATIONS:
        component = components[name]
        _same(
            component["candidate_id"],
            structural["candidate_id"],
            code="component-candidate-identity-mismatch",
            message=f"{name} component candidate ID differs from the held Gate E inventory",
        )
        _same(
            component["source_revision"],
            structural["source_revision"],
            code="component-source-identity-mismatch",
            message=f"{name} component source revision differs from the held Gate E inventory",
        )
        _same(
            component["gate_e_input_inventory"],
            structural["gate_e_input_inventory"],
            code="component-inventory-binding-mismatch",
            message=f"{name} component inventory descriptor differs from the held Gate E inventory",
        )
        _same(
            component["frozen_candidate_manifest"],
            structural["frozen_candidate_manifest"],
            code="component-frozen-manifest-mismatch",
            message=f"{name} component frozen manifest differs from the held Gate E inventory",
        )


def _require_cross_component_bindings(
    components: Mapping[str, Mapping[str, Any]],
    *,
    expected_release_image_id: str,
    expected_optimizer_build_image_id: str,
    expected_correctness_golden_sha256: str,
) -> dict[str, Any]:
    native_payload = components["native_e0"]["payload"]
    optimizer_payload = components["optimizer_e0"]["payload"]
    python_payload = components["python_free"]["payload"]
    performance_payload = components["performance"]["payload"]
    soak_payload = components["soak"]["payload"]

    native_report = _payload_descriptor("native_e0", native_payload, "report")
    native_raw_evidence = _payload_descriptor("native_e0", native_payload, "raw_evidence")
    native_executable = _payload_descriptor("native_e0", native_payload, "candidate_executable")
    native_source_archive = _payload_descriptor("native_e0", native_payload, "source_archive")

    optimizer_report = _payload_descriptor("optimizer_e0", optimizer_payload, "report")
    optimizer_raw_evidence = _payload_descriptor("optimizer_e0", optimizer_payload, "raw_evidence")
    optimizer_profile_binary = _payload_descriptor("optimizer_e0", optimizer_payload, "profile_binary")
    optimizer_source_archive = _payload_descriptor("optimizer_e0", optimizer_payload, "source_archive")

    python_report = _payload_descriptor("python_free", python_payload, "report")
    python_raw_evidence = _payload_descriptor("python_free", python_payload, "raw_evidence")
    correctness_golden = _payload_descriptor("python_free", python_payload, "correctness_golden")
    release_bundle = _payload_descriptor("python_free", python_payload, "release_bundle")
    python_native_report = _payload_descriptor("python_free", python_payload, "native_report")
    python_release_elf = _payload_descriptor("python_free", python_payload, "release_elf")
    python_source_archive = _payload_descriptor("python_free", python_payload, "source_archive")

    performance_report = _payload_descriptor("performance", performance_payload, "report")
    performance_raw_evidence = _payload_descriptor("performance", performance_payload, "raw_evidence")
    performance_optimizer_report = _payload_descriptor("performance", performance_payload, "optimizer_report")
    performance_profile_binary = _payload_descriptor("performance", performance_payload, "profile_binary")
    performance_executable = _payload_descriptor(
        "performance", performance_payload, "native_candidate_executable"
    )
    performance_release_elf = _payload_descriptor("performance", performance_payload, "release_elf")
    performance_source_archive = _payload_descriptor("performance", performance_payload, "source_archive")

    soak_report = _payload_descriptor("soak", soak_payload, "report")
    soak_raw_evidence = _payload_descriptor("soak", soak_payload, "raw_evidence")
    soak_golden = _payload_descriptor("soak", soak_payload, "correctness_golden")
    soak_native_report = _payload_descriptor("soak", soak_payload, "native_report")
    soak_release_elf = _payload_descriptor("soak", soak_payload, "release_elf")
    soak_source_archive = _payload_descriptor("soak", soak_payload, "source_archive")

    for label, descriptor in (
        ("optimizer source archive", optimizer_source_archive),
        ("Python-free source archive", python_source_archive),
        ("performance source archive", performance_source_archive),
        ("soak source archive", soak_source_archive),
    ):
        _same(
            descriptor,
            native_source_archive,
            code="source-archive-binding-mismatch",
            message=f"{label} differs from the native E0 source archive",
        )
    for label, descriptor in (
        ("Python-free native report", python_native_report),
        ("soak native report", soak_native_report),
    ):
        _same(
            descriptor,
            native_report,
            code="native-report-binding-mismatch",
            message=f"{label} differs from the native E0 report",
        )
    _same(
        performance_optimizer_report,
        optimizer_report,
        code="optimizer-report-binding-mismatch",
        message="performance optimizer report differs from the optimizer E0 report",
    )
    _same(
        performance_profile_binary,
        optimizer_profile_binary,
        code="profile-binary-binding-mismatch",
        message="performance profile binary differs from the optimizer E0 profile binary",
    )
    _same(
        performance_executable,
        native_executable,
        code="native-executable-binding-mismatch",
        message="performance native candidate executable differs from the native E0 executable",
    )
    for label, descriptor in (
        ("performance release ELF", performance_release_elf),
        ("soak release ELF", soak_release_elf),
    ):
        _same(
            descriptor,
            python_release_elf,
            code="release-elf-binding-mismatch",
            message=f"{label} differs from the Python-free frozen release ELF",
        )
    _same_descriptor_identity(
        native_executable,
        python_release_elf,
        code="release-executable-identity-mismatch",
        message="native E0 executable digest/length differs from the frozen release ELF",
    )
    _same(
        soak_golden,
        correctness_golden,
        code="correctness-golden-binding-mismatch",
        message="soak correctness golden differs from the Python-free correctness golden",
    )
    _same(
        correctness_golden.get("sha256"),
        expected_correctness_golden_sha256,
        code="correctness-golden-anchor-mismatch",
        message="shared correctness golden differs from the externally supplied anchor",
    )

    for name, payload in (
        ("python_free", python_payload),
        ("performance", performance_payload),
        ("soak", soak_payload),
    ):
        _same(
            payload.get("frozen_release_image_digest"),
            expected_release_image_id,
            code="release-image-binding-mismatch",
            message=f"{name} frozen release image differs from the externally supplied anchor",
        )
    _same(
        components["python_free"]["anchors"]["expected_release_image_id"],
        expected_release_image_id,
        code="release-image-anchor-mismatch",
        message="Python-free release image anchor differs from the external input",
    )
    _same(
        components["performance"]["anchors"]["expected_release_image_id"],
        expected_release_image_id,
        code="release-image-anchor-mismatch",
        message="performance release image anchor differs from the external input",
    )
    _same(
        components["soak"]["anchors"]["expected_release_image_id"],
        expected_release_image_id,
        code="release-image-anchor-mismatch",
        message="soak release image anchor differs from the external input",
    )
    _same(
        components["optimizer_e0"]["anchors"]["expected_optimizer_build_image_id"],
        expected_optimizer_build_image_id,
        code="optimizer-build-image-anchor-mismatch",
        message="optimizer E0 build image anchor differs from the external input",
    )
    _same(
        components["performance"]["anchors"]["expected_optimizer_build_image_id"],
        expected_optimizer_build_image_id,
        code="optimizer-build-image-anchor-mismatch",
        message="performance optimizer build image anchor differs from the external input",
    )
    _same(
        components["python_free"]["anchors"]["expected_correctness_golden_sha256"],
        expected_correctness_golden_sha256,
        code="correctness-golden-anchor-mismatch",
        message="Python-free correctness golden anchor differs from the external input",
    )
    _same(
        components["soak"]["anchors"]["expected_correctness_golden_sha256"],
        expected_correctness_golden_sha256,
        code="correctness-golden-anchor-mismatch",
        message="soak correctness golden anchor differs from the external input",
    )

    return {
        "native_e0": {
            "report": native_report,
            "raw_evidence": native_raw_evidence,
            "candidate_executable": native_executable,
            "source_archive": native_source_archive,
        },
        "optimizer_e0": {
            "report": optimizer_report,
            "raw_evidence": optimizer_raw_evidence,
            "profile_binary": optimizer_profile_binary,
            "source_archive": optimizer_source_archive,
        },
        "python_free": {
            "report": python_report,
            "raw_evidence": python_raw_evidence,
            "correctness_golden": correctness_golden,
            "release_bundle": release_bundle,
            "native_report": python_native_report,
            "release_elf": python_release_elf,
            "source_archive": python_source_archive,
        },
        "performance": {
            "report": performance_report,
            "raw_evidence": performance_raw_evidence,
            "optimizer_report": performance_optimizer_report,
            "profile_binary": performance_profile_binary,
            "native_candidate_executable": performance_executable,
            "release_elf": performance_release_elf,
            "source_archive": performance_source_archive,
            "reviewed_baseline_sha256": _expected_sha256(
                _typed_string(
                    performance_payload.get("reviewed_baseline_sha256"),
                    "performance component reviewed baseline SHA-256",
                    code="invalid-component-result",
                ),
                "performance-reviewed-baseline-sha256",
            ),
        },
        "soak": {
            "report": soak_report,
            "raw_evidence": soak_raw_evidence,
            "correctness_golden": soak_golden,
            "native_report": soak_native_report,
            "release_elf": soak_release_elf,
            "source_archive": soak_source_archive,
            "model_tree_sha256": _expected_sha256(
                _typed_string(
                    soak_payload.get("model_tree_sha256"),
                    "soak component model tree SHA-256",
                    code="invalid-component-result",
                ),
                "soak-model-tree-sha256",
            ),
        },
        "shared_bindings": {
            "source_archive": native_source_archive,
            "release_elf": python_release_elf,
            "release_bundle": release_bundle,
            "profile_binary": optimizer_profile_binary,
            "canonical_native_report": native_report,
            "canonical_optimizer_report": optimizer_report,
            "correctness_golden": correctness_golden,
            "frozen_release_image_digest": expected_release_image_id,
            "optimizer_build_image_id": expected_optimizer_build_image_id,
        },
    }


def _compact_component_results(
    components: Mapping[str, Mapping[str, Any]],
    artifacts: Mapping[str, Any],
) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for name in _COMPONENT_EXPECTATIONS:
        component = components[name]
        row = {
            "schema_version": component["schema_version"],
            "scope": component["scope"],
            "authority": component["authority"],
            "status": component["status"],
            component["status_field"]: component["component_status"],
            "anchors": dict(component["anchors"]),
        }
        row.update(artifacts[name])
        compact[name] = row
    return compact


def _replay_rc3_gate_e_aggregate_v1_on_held_fds(
    gate_e_evidence_root_fd: int,
    frozen_candidate_root_fd: int,
    input_evidence_root_fd: int,
    repository_root: Path,
    repository_root_fd: int,
    scratch_parent: Path,
    expected_release_image_id: str,
    expected_optimizer_build_image_id: str,
    expected_correctness_golden_sha256: str,
) -> dict[str, Any]:
    """Run all component-private replays against one already-locked FD stack."""

    _require_bytecode_cache_disabled()
    _require_aggregate_policy()
    expected_release_image = _expected_image_id(
        expected_release_image_id,
        "expected-release-image-id",
    )
    expected_optimizer_image = _expected_image_id(
        expected_optimizer_build_image_id,
        "expected-optimizer-build-image-id",
    )
    expected_golden = _expected_sha256(
        expected_correctness_golden_sha256,
        "expected-correctness-golden-sha256",
    )
    structural_start = _validate_structural_result(
        _gate(
            lambda: gate_inputs._replay_rc3_gate_e_input_inventory_v1_on_held_fds(
                gate_e_evidence_root_fd,
                frozen_candidate_root_fd,
                input_evidence_root_fd,
                repository_root,
                repository_root_fd,
            )
        ),
        phase="before aggregate",
    )
    components = {
        "native_e0": _validate_component_result(
            "native_e0",
            _component_call(
                "native-e0",
                lambda: native_e0._replay_rc3_gate_e_native_e0_v1_on_held_fds(
                    gate_e_evidence_root_fd,
                    frozen_candidate_root_fd,
                    input_evidence_root_fd,
                    repository_root,
                    repository_root_fd,
                ),
            ),
        ),
        "optimizer_e0": _validate_component_result(
            "optimizer_e0",
            _component_call(
                "optimizer-e0",
                lambda: optimizer_e0._replay_rc3_gate_e_optimizer_e0_v1_on_held_fds(
                    gate_e_evidence_root_fd,
                    frozen_candidate_root_fd,
                    input_evidence_root_fd,
                    repository_root,
                    repository_root_fd,
                    scratch_parent,
                    expected_optimizer_image,
                ),
            ),
        ),
        "python_free": _validate_component_result(
            "python_free",
            _component_call(
                "python-free",
                lambda: python_free._replay_rc3_gate_e_python_free_v1_on_held_fds(
                    gate_e_evidence_root_fd,
                    frozen_candidate_root_fd,
                    input_evidence_root_fd,
                    repository_root,
                    repository_root_fd,
                    scratch_parent,
                    expected_release_image,
                    expected_golden,
                ),
            ),
        ),
        "performance": _validate_component_result(
            "performance",
            _component_call(
                "performance",
                lambda: performance._replay_rc3_gate_e_performance_v1_on_held_fds(
                    gate_e_evidence_root_fd,
                    frozen_candidate_root_fd,
                    input_evidence_root_fd,
                    repository_root,
                    repository_root_fd,
                    scratch_parent,
                    expected_release_image,
                    expected_optimizer_image,
                ),
            ),
        ),
        "soak": _validate_component_result(
            "soak",
            _component_call(
                "soak",
                lambda: soak._replay_rc3_gate_e_soak_v1_on_held_fds(
                    gate_e_evidence_root_fd,
                    frozen_candidate_root_fd,
                    input_evidence_root_fd,
                    repository_root,
                    repository_root_fd,
                    scratch_parent,
                    expected_release_image,
                    expected_golden,
                ),
            ),
        ),
    }
    _require_common_component_bindings(structural_start, components)
    artifacts = _require_cross_component_bindings(
        components,
        expected_release_image_id=expected_release_image,
        expected_optimizer_build_image_id=expected_optimizer_image,
        expected_correctness_golden_sha256=expected_golden,
    )
    structural_end = _validate_structural_result(
        _gate(
            lambda: gate_inputs._replay_rc3_gate_e_input_inventory_v1_on_held_fds(
                gate_e_evidence_root_fd,
                frozen_candidate_root_fd,
                input_evidence_root_fd,
                repository_root,
                repository_root_fd,
            )
        ),
        phase="after aggregate",
    )
    _same(
        structural_end,
        structural_start,
        code="gate-e-input-replay-drift",
        message="Gate E structural inputs changed during aggregate semantic replay",
    )
    return {
        "schema_version": REPLAY_VERSION,
        "scope": SCOPE,
        "status": "bound",
        "authority": AUTHORITY,
        "candidate_status": "frozen",
        "qualification_status": "not-run",
        "gate_e_status": "passed",
        "candidate_id": structural_start["candidate_id"],
        "source_revision": structural_start["source_revision"],
        "expected_release_image_id": expected_release_image,
        "expected_optimizer_build_image_id": expected_optimizer_image,
        "expected_correctness_golden_sha256": expected_golden,
        "aggregate_policy_version": AGGREGATE_POLICY_VERSION,
        "aggregate_policy_sha256": AGGREGATE_POLICY_SHA256,
        "gate_e_input_inventory": structural_start["gate_e_input_inventory"],
        "frozen_candidate_manifest": structural_start["frozen_candidate_manifest"],
        "components": _compact_component_results(components, artifacts),
        "shared_bindings": artifacts["shared_bindings"],
        "checks": [{"name": name, "satisfied": True} for name in CHECK_NAMES],
        "not_established": dict(NOT_ESTABLISHED),
        "reason_codes": [],
    }


def replay_rc3_gate_e_aggregate_v1(
    gate_e_evidence_root: Path,
    *,
    frozen_candidate_root: Path,
    input_evidence_root: Path,
    repository_root: Path,
    expected_release_image_id: str,
    expected_optimizer_build_image_id: str,
    expected_correctness_golden_sha256: str,
) -> dict[str, Any]:
    """Open, lock, and retain one root stack for the aggregate semantic replay."""

    _require_bytecode_cache_disabled()
    _expected_image_id(expected_release_image_id, "expected-release-image-id")
    _expected_image_id(expected_optimizer_build_image_id, "expected-optimizer-build-image-id")
    _expected_sha256(expected_correctness_golden_sha256, "expected-correctness-golden-sha256")
    gate_root = _frozen(
        lambda: frozen.normalized_absolute_path(gate_e_evidence_root, "--gate-e-evidence-root")
    )
    frozen_root = _frozen(
        lambda: frozen.normalized_absolute_path(frozen_candidate_root, "--frozen-candidate-root")
    )
    input_root = _frozen(
        lambda: frozen.normalized_absolute_path(input_evidence_root, "--input-evidence-root")
    )
    source_root = _frozen(
        lambda: frozen.normalized_absolute_path(repository_root, "--repository-root")
    )
    scratch_parent = _frozen(
        lambda: frozen.normalized_absolute_path(
            EXTERNAL_SCRATCH_PARENT,
            "aggregate external scratch parent",
        )
    )
    _frozen(
        lambda: frozen.require_disjoint_paths(
            {
                "Gate E evidence root": gate_root,
                "frozen candidate root": frozen_root,
                "freeze-input evidence root": input_root,
                "source checkout": source_root,
                "aggregate external scratch parent": scratch_parent,
            }
        )
    )
    source_root_fd: int | None = None
    input_root_fd: int | None = None
    frozen_root_fd: int | None = None
    gate_root_fd: int | None = None
    scratch_parent_fd: int | None = None
    try:
        source_root_fd = _common(lambda: common.open_absolute_directory(source_root, "source checkout"))
        input_root_fd = _common(
            lambda: common.open_private_evidence_directory(input_root, "freeze-input evidence root")
        )
        frozen_root_fd = _common(
            lambda: common.open_private_evidence_directory(frozen_root, "frozen candidate root")
        )
        gate_root_fd = _common(
            lambda: common.open_private_evidence_directory(gate_root, "Gate E evidence root")
        )
        scratch_parent_fd = _common(
            lambda: common.open_absolute_directory(scratch_parent, "aggregate external scratch parent")
        )
        roots = {
            "source checkout": (source_root, source_root_fd),
            "freeze-input evidence root": (input_root, input_root_fd),
            "frozen candidate root": (frozen_root, frozen_root_fd),
            "Gate E evidence root": (gate_root, gate_root_fd),
            "aggregate external scratch parent": (scratch_parent, scratch_parent_fd),
        }
        _topology(lambda: topology.assert_existing_roots_disjoint(roots))
        _shared_lock(input_root_fd, "freeze-input evidence root")
        _shared_lock(frozen_root_fd, "frozen candidate root")
        _shared_lock(gate_root_fd, "Gate E evidence root")
        result = _replay_rc3_gate_e_aggregate_v1_on_held_fds(
            gate_root_fd,
            frozen_root_fd,
            input_root_fd,
            source_root,
            source_root_fd,
            scratch_parent,
            expected_release_image_id,
            expected_optimizer_build_image_id,
            expected_correctness_golden_sha256,
        )
        _topology(lambda: topology.assert_existing_roots_disjoint(roots))
        return result
    finally:
        _unlock_quietly(gate_root_fd)
        _unlock_quietly(frozen_root_fd)
        _unlock_quietly(input_root_fd)
        _close_quietly(gate_root_fd)
        _close_quietly(frozen_root_fd)
        _close_quietly(input_root_fd)
        _close_quietly(source_root_fd)
        _close_quietly(scratch_parent_fd)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-e-evidence-root", required=True, type=Path)
    parser.add_argument("--frozen-candidate-root", required=True, type=Path)
    parser.add_argument("--input-evidence-root", required=True, type=Path)
    parser.add_argument("--repository-root", required=True, type=Path)
    parser.add_argument("--expected-release-image-id", required=True)
    parser.add_argument("--expected-optimizer-build-image-id", required=True)
    parser.add_argument("--expected-correctness-golden-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = replay_rc3_gate_e_aggregate_v1(
            args.gate_e_evidence_root,
            frozen_candidate_root=args.frozen_candidate_root,
            input_evidence_root=args.input_evidence_root,
            repository_root=args.repository_root,
            expected_release_image_id=args.expected_release_image_id,
            expected_optimizer_build_image_id=args.expected_optimizer_build_image_id,
            expected_correctness_golden_sha256=args.expected_correctness_golden_sha256,
        )
    except AggregateReplayError as error:
        print(f"RC3 Gate E aggregate replay failed: {error}", file=sys.stderr)
        return 2
    sys.stdout.buffer.write(common.canonical_json_bytes(result) + b"\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
