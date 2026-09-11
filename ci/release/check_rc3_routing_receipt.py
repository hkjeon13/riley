#!/usr/bin/env python3
"""Fail-closed C02 verifier for fixed release-binary routing evidence.

This module verifies a *reviewed, finite* C02 routing corpus.  It is not the
C03 property/fuzz gate: every required C=1/C=5/C=8, cancellation, malformed
plan, commit-failure, KV-ownership, and shutdown trace has a source-controlled
expected body.  A raw ``passed`` field or a bag of hashes is consequently not
enough evidence.  The checker replays Gate E, pins the release executable and
all source/image/model bindings to the frozen candidate, then parses the full
slot -> request -> downloaded-token -> publication/terminal chain.

It is deliberately CPU-only.  It does not start Riley, CUDA, a container, SSH,
or a network request.  A trusted remote producer must capture the raw traces
under an external create-only evidence root before this verifier is invoked.

Outer C02 integration API::

    parsed = validate_check_report(submitted_report)
    replayed = evaluate(freeze, evidence_root, parsed.receipt.path,
                        expected_freeze_sha256=trusted_freeze_sha)
    assert submitted_report == replayed
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn, Sequence

import check_rc3_qualification as qualification


RECEIPT_VERSION = "riley.rc3-routing-receipt.v1"
TRACE_MANIFEST_VERSION = "riley.rc3-routing-trace-manifest.v1"
TRACE_VERSION = "riley.rc3-routing-trace.v1"
CORPUS_VERSION = "riley.rc3-routing-corpus.v1"
CHECK_REPORT_VERSION = "riley.rc3-routing-check.v1"
CORPUS_ID = "rc3-routing-fixed-release-v1"
CORPUS_RELATIVE_PATH = "benchmarks/release/candidates/rc3-routing-corpus-v1.json"
CORPUS_SHA256 = "6ec9f610a7ffc2a2e75d39c6b8894d94901a2dab25cf16c5f2e405df1290c585"
STABLE_DEFAULT_PROFILE = "stable-default"
MAX_EXECUTABLE_BYTES = 1024 * 1024 * 1024

CASE_ID_RE = re.compile(r"^routing-[a-z0-9][a-z0-9-]{0,63}$")
REQUEST_ID_RE = re.compile(r"^[a-z][a-z0-9-]{0,63}$")

CASE_COVERAGE: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("routing-c1-basic", ("c1", "shutdown-quiescence")),
    ("routing-c5-permuted-mixed", ("c5", "dense-permutation", "mixed-prefill-decode")),
    ("routing-c8-cancel-precommit", ("c8", "cancellation", "precommit-cancellation")),
    ("routing-malformed-pre-dispatch", ("malformed-plan-pre-dispatch-rejection",)),
    ("routing-commit-failure-contained", ("commit-failure-containment", "kv-ownership")),
)
CASE_IDS = tuple(case_id for case_id, _ in CASE_COVERAGE)
CASE_COVERAGE_BY_ID = {case_id: coverage for case_id, coverage in CASE_COVERAGE}
REQUIRED_COVERAGE = frozenset(
    item for _, coverage in CASE_COVERAGE for item in coverage
)
CHECK_NAMES = (
    "gate-e-replay",
    "frozen-stable-release-binding",
    "fixed-corpus-integrity",
    "release-executable-binding",
    "slot-request-token-routing",
    "cancellation-and-failure-containment",
    "terminal-kv-shutdown-quiescence",
)


class RoutingReceiptError(qualification.QualificationError):
    """Routing evidence is malformed or fails the fixed C02 contract."""


class RoutingReceiptIncomparable(qualification.IncomparableError):
    """Routing evidence belongs to a different frozen candidate or arm."""


@dataclass(frozen=True)
class Descriptor:
    path: str
    sha256: str


@dataclass(frozen=True)
class RoutingReceipt:
    candidate_id: str
    bindings: dict[str, str]
    model: dict[str, str]
    execution: dict[str, Any]
    corpus: Descriptor
    trace_manifest: Descriptor


@dataclass(frozen=True)
class RoutingCheckReport:
    """Closed report surface an outer C02 finalizer may rerun and compare."""

    candidate_id: str
    freeze_sha256: str
    base_release_candidate_report: Descriptor
    bindings: dict[str, str]
    corpus: Descriptor
    receipt: Descriptor
    trace_manifest: Descriptor
    model: dict[str, str]
    execution: dict[str, Any]
    traces: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class RoutingCorpus:
    descriptor: Descriptor
    cases: tuple[dict[str, Any], ...]


def _raise(error_type: type[qualification.QualificationError], code: str, message: str) -> NoReturn:
    error = error_type(message)
    error.reason_code = code  # type: ignore[attr-defined]
    raise error


def _fail(code: str, message: str) -> NoReturn:
    _raise(RoutingReceiptError, code, message)


def _incomparable(message: str) -> NoReturn:
    _raise(RoutingReceiptIncomparable, "incomparable-binding", message)


def _exact(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    return qualification._exact(value, fields, label)


def _sha256(value: Any, label: str) -> str:
    return qualification._sha256(value, label)


def _candidate_id(value: Any, label: str) -> str:
    candidate_id = qualification._string(value, label)
    if not qualification.release_candidate.CANDIDATE_ID_RE.fullmatch(candidate_id):
        _fail("invalid-candidate-id", f"{label} is not a valid RC candidate")
    return candidate_id


def _case_id(value: Any, label: str) -> str:
    value = qualification._string(value, label)
    if not CASE_ID_RE.fullmatch(value):
        _fail("invalid-case-id", f"{label} is not a routing case ID")
    return value


def _request_id(value: Any, label: str) -> str:
    value = qualification._string(value, label)
    if not REQUEST_ID_RE.fullmatch(value):
        _fail("invalid-request-id", f"{label} is not a normalized request ID")
    return value


def _bounded_int(value: Any, label: str, *, minimum: int = 0, maximum: int = 2**32 - 1) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        _fail("invalid-integer", f"{label} must be an integer in [{minimum}, {maximum}]")
    return value


def _bindings(value: Any, label: str) -> dict[str, str]:
    row = _exact(
        value,
        {
            "freeze_sha256",
            "base_release_candidate_report_sha256",
            "configuration_profile",
            "configuration_sha256",
        },
        label,
    )
    if row["configuration_profile"] != STABLE_DEFAULT_PROFILE:
        _incomparable(f"{label}.configuration_profile is not {STABLE_DEFAULT_PROFILE}")
    return {
        "freeze_sha256": _sha256(row["freeze_sha256"], f"{label}.freeze_sha256"),
        "base_release_candidate_report_sha256": _sha256(
            row["base_release_candidate_report_sha256"],
            f"{label}.base_release_candidate_report_sha256",
        ),
        "configuration_profile": STABLE_DEFAULT_PROFILE,
        "configuration_sha256": _sha256(
            row["configuration_sha256"], f"{label}.configuration_sha256"
        ),
    }


def _model(value: Any, label: str) -> dict[str, str]:
    return qualification._model(value, label)


def _descriptor(value: Any, label: str) -> Descriptor:
    row = _exact(value, {"path", "sha256"}, label)
    return Descriptor(
        path=qualification._relative_path(row["path"], f"{label}.path"),
        sha256=_sha256(row["sha256"], f"{label}.sha256"),
    )


def _canonical_document(raw: bytes, label: str) -> dict[str, Any]:
    document = qualification._parse_document(raw, label)
    if raw != qualification.canonical_json_bytes(document):
        _fail("noncanonical-evidence", f"{label} must be exact canonical JSON bytes")
    return document


def _read_relative_json(
    evidence_root: Path,
    relative_path: Path | str,
    label: str,
) -> tuple[str, bytes, dict[str, Any]]:
    relative = qualification._relative_path(str(relative_path), f"{label}.path")
    raw = qualification._read_relative(evidence_root, relative, label)
    return relative, raw, _canonical_document(raw, label)


def _reserve_descriptor(
    descriptor: Descriptor,
    *,
    label: str,
    used_paths: set[str],
    reserved_paths: set[str],
) -> None:
    if descriptor.path in reserved_paths:
        _fail(
            "reserved-output-path-collision",
            f"{label} reuses a freeze-declared final report or semantic receipt path",
        )
    if descriptor.path in used_paths:
        _fail("duplicate-evidence-path", f"{label} reuses evidence path {descriptor.path!r}")
    used_paths.add(descriptor.path)


def _read_described_json(
    evidence_root: Path,
    descriptor: Descriptor,
    label: str,
    *,
    used_paths: set[str],
    reserved_paths: set[str],
) -> tuple[bytes, dict[str, Any]]:
    _reserve_descriptor(
        descriptor, label=label, used_paths=used_paths, reserved_paths=reserved_paths
    )
    raw = qualification._read_relative(evidence_root, descriptor.path, label)
    if hashlib.sha256(raw).hexdigest() != descriptor.sha256:
        _fail("evidence-hash-mismatch", f"{label} digest mismatch")
    return raw, _canonical_document(raw, label)


def _sha256_relative_executable(
    evidence_root: Path,
    descriptor: Descriptor,
    label: str,
    *,
    used_paths: set[str],
    reserved_paths: set[str],
) -> None:
    """Digest a no-follow release executable without assuming it is small JSON."""

    _reserve_descriptor(
        descriptor, label=label, used_paths=used_paths, reserved_paths=reserved_paths
    )
    try:
        root_metadata = evidence_root.lstat()
    except OSError as error:
        _fail("missing-evidence-root", f"evidence root cannot be inspected: {error}")
    if not stat.S_ISDIR(root_metadata.st_mode):
        _fail("unsafe-evidence-root", "evidence root must be a real directory")
    common = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    directory = common | getattr(os, "O_DIRECTORY", 0)
    try:
        current_fd = os.open(evidence_root, directory)
    except OSError as error:
        _fail("unsafe-evidence-root", f"evidence root cannot be opened safely: {error}")
    try:
        for component in PurePosixPath(descriptor.path).parts[:-1]:
            try:
                next_fd = os.open(component, directory, dir_fd=current_fd)
            except OSError as error:
                _fail("unsafe-evidence-path", f"{label}: unsafe directory component: {error}")
            os.close(current_fd)
            current_fd = next_fd
        try:
            executable_fd = os.open(
                PurePosixPath(descriptor.path).parts[-1], common, dir_fd=current_fd
            )
        except OSError as error:
            _fail("missing-input", f"{label} cannot be opened safely: {error}")
        try:
            before = os.fstat(executable_fd)
            if not stat.S_ISREG(before.st_mode):
                _fail("unsafe-evidence-path", f"{label} must be a regular non-link file")
            if before.st_size < 5 or before.st_size > MAX_EXECUTABLE_BYTES:
                _fail("invalid-release-executable", f"{label} has an unsafe executable size")
            digest = hashlib.sha256()
            remaining = before.st_size
            first = b""
            while remaining:
                chunk = os.read(executable_fd, min(1024 * 1024, remaining))
                if not chunk:
                    _fail("truncated-input", f"{label} changed while it was read")
                if len(first) < 5:
                    first += chunk[: 5 - len(first)]
                digest.update(chunk)
                remaining -= len(chunk)
            if os.read(executable_fd, 1):
                _fail("mutated-input", f"{label} grew while it was read")
            after = os.fstat(executable_fd)
            if (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            ) != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ):
                _fail("mutated-input", f"{label} changed while it was read")
            if first[:4] != b"\x7fELF":
                _fail("invalid-release-executable", f"{label} is not an ELF executable")
            if digest.hexdigest() != descriptor.sha256:
                _fail("evidence-hash-mismatch", f"{label} digest mismatch")
        finally:
            os.close(executable_fd)
    finally:
        os.close(current_fd)


def _expected_bindings(
    frozen: qualification.FrozenCandidate,
    freeze_sha256: str,
    base_report_sha256: str,
) -> dict[str, str]:
    return {
        "freeze_sha256": freeze_sha256,
        "base_release_candidate_report_sha256": base_report_sha256,
        "configuration_profile": STABLE_DEFAULT_PROFILE,
        "configuration_sha256": frozen.arms["stable_default"]["configuration_sha256"],
    }


def _validate_bound_header(
    *,
    candidate_id: str,
    bindings: dict[str, str],
    model: dict[str, str],
    frozen: qualification.FrozenCandidate,
    freeze_sha256: str,
    base_report_sha256: str,
    label: str,
) -> None:
    if candidate_id != frozen.candidate_id:
        _incomparable(f"{label} belongs to another candidate")
    if bindings != _expected_bindings(frozen, freeze_sha256, base_report_sha256):
        _incomparable(f"{label} immutable bindings drifted from frozen stable-default")
    if model != frozen.models["smollm2"]:
        _incomparable(f"{label} model identity drifted from frozen SmolLM2")


def _execution_common(value: Any, label: str, *, executable_descriptor: bool) -> dict[str, Any]:
    executable_field = "test_executable" if executable_descriptor else "test_executable_sha256"
    row = _exact(
        value,
        {
            "source_revision",
            "source_archive_sha256",
            "release_binary_sha256",
            "release_bundle_sha256",
            "release_image_id",
            "test_image_id",
            executable_field,
        },
        label,
    )
    result: dict[str, Any] = {
        "source_revision": qualification._string(row["source_revision"], f"{label}.source_revision"),
        "source_archive_sha256": _sha256(row["source_archive_sha256"], f"{label}.source_archive_sha256"),
        "release_binary_sha256": _sha256(row["release_binary_sha256"], f"{label}.release_binary_sha256"),
        "release_bundle_sha256": _sha256(row["release_bundle_sha256"], f"{label}.release_bundle_sha256"),
        "release_image_id": qualification._image(row["release_image_id"], f"{label}.release_image_id"),
        "test_image_id": qualification._image(row["test_image_id"], f"{label}.test_image_id"),
    }
    if executable_descriptor:
        result["test_executable"] = _descriptor(row["test_executable"], f"{label}.test_executable")
    else:
        result["test_executable_sha256"] = _sha256(
            row["test_executable_sha256"], f"{label}.test_executable_sha256"
        )
    return result


def _validate_execution_for_frozen(
    execution: dict[str, Any],
    frozen: qualification.FrozenCandidate,
    label: str,
) -> None:
    expected = {
        "source_revision": frozen.source["git_revision"],
        "source_archive_sha256": frozen.source["archive_sha256"],
        "release_binary_sha256": frozen.release["binary_sha256"],
        "release_bundle_sha256": frozen.release["bundle_sha256"],
        "release_image_id": frozen.release["image_id"],
        "test_image_id": frozen.images["cuda"],
    }
    for name, expected_value in expected.items():
        if execution[name] != expected_value:
            _incomparable(f"{label}.{name} drifts from the frozen release")
    executable_sha256 = (
        execution["test_executable"].sha256
        if "test_executable" in execution
        else execution["test_executable_sha256"]
    )
    if executable_sha256 != frozen.release["binary_sha256"]:
        _incomparable(f"{label} does not execute the frozen release binary")


def _validate_corpus_descriptor(value: Any, label: str) -> Descriptor:
    descriptor = _descriptor(value, label)
    expected = Descriptor(CORPUS_RELATIVE_PATH, CORPUS_SHA256)
    if descriptor != expected:
        _fail("routing-corpus-binding-mismatch", f"{label} does not bind the reviewed fixed routing corpus")
    return descriptor


def _validate_iteration(
    value: Any,
    label: str,
    *,
    published_by_request: dict[str, list[int]],
    cancelled_requests: set[str],
) -> tuple[set[str], int, int, int]:
    row = _exact(
        value,
        {"phase", "slot_routes", "downloaded_tokens", "published_tokens", "precommit_cancellation"},
        label,
    )
    phase = row["phase"]
    if phase not in {"prefill", "decode", "mixed"}:
        _fail("invalid-routing-phase", f"{label}.phase is unsupported")
    routes_value = row["slot_routes"]
    if not isinstance(routes_value, list) or not routes_value or len(routes_value) > 256:
        _fail("invalid-slot-routing", f"{label}.slot_routes must be a bounded non-empty array")
    route_by_slot: dict[int, tuple[str, str]] = {}
    request_ids: set[str] = set()
    kinds: set[str] = set()
    for index, route_value in enumerate(routes_value):
        route = _exact(route_value, {"slot", "request_id", "work_kind"}, f"{label}.slot_routes[{index}]")
        slot = _bounded_int(route["slot"], f"{label}.slot_routes[{index}].slot", maximum=255)
        request_id = _request_id(route["request_id"], f"{label}.slot_routes[{index}].request_id")
        work_kind = route["work_kind"]
        if work_kind not in {"prefill", "decode"}:
            _fail("invalid-work-kind", f"{label}.slot_routes[{index}].work_kind is unsupported")
        if slot in route_by_slot or request_id in request_ids:
            _fail("duplicate-slot-or-request", f"{label}.slot_routes must be one-to-one")
        route_by_slot[slot] = (request_id, work_kind)
        request_ids.add(request_id)
        kinds.add(work_kind)
    if sorted(route_by_slot) != list(range(len(route_by_slot))):
        _fail("sparse-slot-routing", f"{label}.slot_routes must use dense slots 0..N-1")
    if (phase == "prefill" and kinds != {"prefill"}) or (phase == "decode" and kinds != {"decode"}) or (
        phase == "mixed" and kinds != {"prefill", "decode"}
    ):
        _fail("routing-phase-kind-mismatch", f"{label}.phase does not match route work kinds")

    downloaded_value = row["downloaded_tokens"]
    if not isinstance(downloaded_value, list) or len(downloaded_value) != len(route_by_slot):
        _fail("invalid-downloaded-token-map", f"{label}.downloaded_tokens must map every slot once")
    token_by_slot: dict[int, int] = {}
    for index, downloaded_value_item in enumerate(downloaded_value):
        downloaded = _exact(
            downloaded_value_item, {"slot", "token_id"}, f"{label}.downloaded_tokens[{index}]"
        )
        slot = _bounded_int(downloaded["slot"], f"{label}.downloaded_tokens[{index}].slot", maximum=255)
        token_id = _bounded_int(downloaded["token_id"], f"{label}.downloaded_tokens[{index}].token_id")
        if slot in token_by_slot:
            _fail("duplicate-downloaded-slot", f"{label}.downloaded_tokens reuses a slot")
        token_by_slot[slot] = token_id
    if sorted(token_by_slot) != list(range(len(route_by_slot))):
        _fail("downloaded-slot-routing-mismatch", f"{label}.downloaded_tokens does not cover dense route slots")

    cancellation = row["precommit_cancellation"]
    cancelled_slot: int | None = None
    if cancellation is not None:
        cancellation_row = _exact(cancellation, {"request_id", "slot"}, f"{label}.precommit_cancellation")
        cancelled_request = _request_id(
            cancellation_row["request_id"], f"{label}.precommit_cancellation.request_id"
        )
        cancelled_slot = _bounded_int(
            cancellation_row["slot"], f"{label}.precommit_cancellation.slot", maximum=255
        )
        if route_by_slot.get(cancelled_slot, (None, None))[0] != cancelled_request:
            _fail("precommit-cancellation-route-mismatch", f"{label} cancellation does not name its routed slot")
        if cancelled_request in cancelled_requests:
            _fail("duplicate-precommit-cancellation", f"{label} cancels one request more than once")
        cancelled_requests.add(cancelled_request)

    published_value = row["published_tokens"]
    if not isinstance(published_value, list) or len(published_value) > len(route_by_slot):
        _fail("invalid-published-token-map", f"{label}.published_tokens is invalid")
    published_slots: set[int] = set()
    for index, published_value_item in enumerate(published_value):
        published = _exact(
            published_value_item,
            {"sequence", "slot", "request_id", "token_id"},
            f"{label}.published_tokens[{index}]",
        )
        sequence = _bounded_int(published["sequence"], f"{label}.published_tokens[{index}].sequence")
        slot = _bounded_int(published["slot"], f"{label}.published_tokens[{index}].slot", maximum=255)
        request_id = _request_id(published["request_id"], f"{label}.published_tokens[{index}].request_id")
        token_id = _bounded_int(published["token_id"], f"{label}.published_tokens[{index}].token_id")
        if slot in published_slots:
            _fail("duplicate-published-slot", f"{label}.published_tokens reuses a slot")
        if slot not in route_by_slot or route_by_slot[slot][0] != request_id:
            _fail("slot-request-routing-mismatch", f"{label}.published_tokens does not retain slot ownership")
        if token_by_slot[slot] != token_id:
            _fail("slot-token-routing-mismatch", f"{label}.published_tokens does not retain downloaded token ownership")
        if cancelled_slot == slot:
            _fail("cancelled-token-published", f"{label} publishes a token after precommit cancellation")
        published_slots.add(slot)
        published_by_request.setdefault(request_id, []).append(sequence)
    if cancellation is None and published_slots != set(route_by_slot):
        # Failure containment can intentionally publish nothing; the trace-level
        # failure validator below is the only allowed exception.
        return request_ids, len(token_by_slot), len(published_slots), len(route_by_slot)
    if cancellation is not None and published_slots | {cancelled_slot} != set(route_by_slot):
        _fail("unaccounted-routed-slot", f"{label} leaves a routed slot neither published nor cancelled")
    return request_ids, len(token_by_slot), len(published_slots), len(route_by_slot)


def _validate_trace_body(value: Any, label: str) -> None:
    row = _exact(
        value,
        {
            "concurrency",
            "dispatch_count",
            "failure",
            "iterations",
            "terminal_events",
            "kv_final",
            "shutdown",
        },
        label,
    )
    concurrency = _bounded_int(row["concurrency"], f"{label}.concurrency", minimum=1, maximum=256)
    dispatch_count = _bounded_int(row["dispatch_count"], f"{label}.dispatch_count", maximum=256)
    iterations = row["iterations"]
    if not isinstance(iterations, list) or len(iterations) > 256:
        _fail("invalid-routing-iterations", f"{label}.iterations must be a bounded array")
    published_by_request: dict[str, list[int]] = {}
    cancelled_requests: set[str] = set()
    routed_requests: set[str] = set()
    downloaded_count = 0
    published_count = 0
    maximum_routed_slots = 0
    for index, iteration in enumerate(iterations):
        iteration_requests, iteration_downloaded, iteration_published, iteration_slots = _validate_iteration(
            iteration,
            f"{label}.iterations[{index}]",
            published_by_request=published_by_request,
            cancelled_requests=cancelled_requests,
        )
        routed_requests.update(iteration_requests)
        downloaded_count += iteration_downloaded
        published_count += iteration_published
        maximum_routed_slots = max(maximum_routed_slots, iteration_slots)

    terminal_events = row["terminal_events"]
    if not isinstance(terminal_events, list) or not terminal_events or len(terminal_events) > 1024:
        _fail("invalid-terminal-events", f"{label}.terminal_events must be a bounded non-empty array")
    terminal_by_request: dict[str, tuple[int, str]] = {}
    all_sequences: list[int] = []
    for sequences in published_by_request.values():
        all_sequences.extend(sequences)
    for index, terminal_value in enumerate(terminal_events):
        terminal = _exact(
            terminal_value, {"sequence", "request_id", "reason"}, f"{label}.terminal_events[{index}]"
        )
        sequence = _bounded_int(terminal["sequence"], f"{label}.terminal_events[{index}].sequence")
        request_id = _request_id(terminal["request_id"], f"{label}.terminal_events[{index}].request_id")
        reason = terminal["reason"]
        if reason not in {"length", "stop", "cancelled", "rejected", "executor-failure"}:
            _fail("invalid-terminal-reason", f"{label}.terminal_events[{index}].reason is unsupported")
        if request_id in terminal_by_request:
            _fail("duplicate-terminal-event", f"{label} emits more than one terminal event per request")
        terminal_by_request[request_id] = (sequence, reason)
        all_sequences.append(sequence)
    if sorted(all_sequences) != list(range(len(all_sequences))) or len(set(all_sequences)) != len(all_sequences):
        _fail("terminal-publication-sequence-mismatch", f"{label} does not have one contiguous global event sequence")
    for request_id, sequences in published_by_request.items():
        if request_id not in terminal_by_request or any(
            sequence >= terminal_by_request[request_id][0] for sequence in sequences
        ):
            _fail("token-after-terminal", f"{label} publishes a token without/beyond a terminal event")
    for request_id in cancelled_requests:
        terminal = terminal_by_request.get(request_id)
        if terminal is None or terminal[1] != "cancelled" or published_by_request.get(request_id):
            _fail("precommit-cancellation-not-contained", f"{label} cancellation is not terminal and token-free")

    failure = row["failure"]
    if failure is None:
        if dispatch_count != len(iterations) or not iterations or downloaded_count == 0:
            _fail("invalid-dispatch-accounting", f"{label} successful trace dispatch accounting is invalid")
        if routed_requests != set(terminal_by_request):
            _fail("terminal-routing-ownership-mismatch", f"{label} terminals do not exactly close routed requests")
        if any(reason not in {"length", "stop", "cancelled"} for _, reason in terminal_by_request.values()):
            _fail("invalid-success-terminal", f"{label} successful trace has a failure terminal")
        if published_count + len(cancelled_requests) != downloaded_count:
            _fail("successful-routing-not-fully-accounted", f"{label} successful trace has unaccounted downloaded tokens")
    else:
        failure_row = _exact(failure, {"stage", "reason"}, f"{label}.failure")
        stage = failure_row["stage"]
        reason = failure_row["reason"]
        if stage == "pre-dispatch":
            if reason != "malformed-plan-rejected" or dispatch_count != 0 or iterations or routed_requests or downloaded_count or published_count:
                _fail("malformed-plan-dispatched", f"{label} malformed plan was not rejected before dispatch")
            if any(reason != "rejected" for _, reason in terminal_by_request.values()):
                _fail("invalid-malformed-plan-terminal", f"{label} malformed plan did not produce only rejection terminals")
        elif stage == "post-execute-pre-publication":
            if reason != "commit-reservation-failed" or dispatch_count != len(iterations) or not iterations or not routed_requests or not downloaded_count or published_count:
                _fail("commit-failure-not-contained", f"{label} commit failure leaked a publication or has invalid dispatch accounting")
            if routed_requests != set(terminal_by_request) or any(
                reason != "executor-failure" for _, reason in terminal_by_request.values()
            ):
                _fail("commit-failure-terminal-mismatch", f"{label} commit failure does not terminate every routed request")
        else:
            _fail("invalid-failure-stage", f"{label}.failure.stage is unsupported")
        if cancelled_requests:
            _fail("cancellation-during-failure-trace", f"{label} combines cancellation with a failure containment trace")

    kv_final = _exact(
        row["kv_final"],
        {"total_blocks", "free_blocks", "reserved_blocks", "active_blocks", "promised_blocks"},
        f"{label}.kv_final",
    )
    total_blocks = _bounded_int(kv_final["total_blocks"], f"{label}.kv_final.total_blocks", minimum=1)
    if _bounded_int(kv_final["free_blocks"], f"{label}.kv_final.free_blocks") != total_blocks:
        _fail("kv-ownership-leak", f"{label}.kv_final.free_blocks does not restore the full pool")
    for name in ("reserved_blocks", "active_blocks", "promised_blocks"):
        if _bounded_int(kv_final[name], f"{label}.kv_final.{name}") != 0:
            _fail("kv-ownership-leak", f"{label}.kv_final.{name} is not quiescent")
    shutdown = _exact(
        row["shutdown"],
        {"pending_requests", "completion_outbox", "terminal_outbox", "live_allocations"},
        f"{label}.shutdown",
    )
    for name in ("pending_requests", "completion_outbox", "terminal_outbox", "live_allocations"):
        if _bounded_int(shutdown[name], f"{label}.shutdown.{name}") != 0:
            _fail("shutdown-not-quiescent", f"{label}.shutdown.{name} is not zero")
    if concurrency < maximum_routed_slots:
        _fail("concurrency-slot-mismatch", f"{label}.concurrency is smaller than its routed slot count")


def _load_corpus() -> RoutingCorpus:
    corpus_path = Path(__file__).resolve().parents[2] / CORPUS_RELATIVE_PATH
    raw = qualification._read_regular_path(corpus_path, "source-controlled routing corpus")
    if hashlib.sha256(raw).hexdigest() != CORPUS_SHA256:
        _fail("routing-corpus-sha-mismatch", "source-controlled routing corpus bytes differ from reviewed SHA-256")
    document = qualification._parse_document(raw, "source-controlled routing corpus")
    row = _exact(document, {"schema_version", "corpus_id", "cases"}, "source-controlled routing corpus")
    if row["schema_version"] != CORPUS_VERSION or row["corpus_id"] != CORPUS_ID:
        _fail("unsupported-routing-corpus", "source-controlled routing corpus identity is unsupported")
    cases = row["cases"]
    if not isinstance(cases, list) or len(cases) != len(CASE_COVERAGE):
        _fail("invalid-routing-corpus", "source-controlled routing corpus has the wrong case count")
    normalized: list[dict[str, Any]] = []
    observed_coverage: set[str] = set()
    for index, case_value in enumerate(cases):
        case = _exact(case_value, {"case_id", "coverage", "trace"}, f"routing corpus.cases[{index}]")
        case_id = _case_id(case["case_id"], f"routing corpus.cases[{index}].case_id")
        if case_id != CASE_IDS[index]:
            _fail("routing-corpus-case-order", "source-controlled routing corpus case order drifted")
        coverage = case["coverage"]
        if not isinstance(coverage, list) or tuple(coverage) != CASE_COVERAGE_BY_ID[case_id]:
            _fail("routing-corpus-coverage", f"routing corpus case {case_id} coverage drifted")
        _validate_trace_body(case["trace"], f"routing corpus.cases[{index}].trace")
        observed_coverage.update(coverage)
        normalized.append({"case_id": case_id, "coverage": list(coverage), "trace": case["trace"]})
    if observed_coverage != REQUIRED_COVERAGE:
        _fail("routing-corpus-coverage", "source-controlled routing corpus does not cover the C02 routing inventory")
    return RoutingCorpus(Descriptor(CORPUS_RELATIVE_PATH, CORPUS_SHA256), tuple(normalized))


def _validate_receipt(document: dict[str, Any], label: str) -> RoutingReceipt:
    row = _exact(
        document,
        {
            "schema_version",
            "status",
            "passed",
            "candidate_id",
            "bindings",
            "model",
            "execution",
            "corpus",
            "trace_manifest",
        },
        label,
    )
    if row["schema_version"] != RECEIPT_VERSION:
        _fail("unsupported-routing-receipt-version", f"{label}.schema_version is unsupported")
    if row["status"] != "passed" or row["passed"] is not True:
        _fail("routing-receipt-not-passed", f"{label} must be passed")
    return RoutingReceipt(
        candidate_id=_candidate_id(row["candidate_id"], f"{label}.candidate_id"),
        bindings=_bindings(row["bindings"], f"{label}.bindings"),
        model=_model(row["model"], f"{label}.model"),
        execution=_execution_common(row["execution"], f"{label}.execution", executable_descriptor=True),
        corpus=_validate_corpus_descriptor(row["corpus"], f"{label}.corpus"),
        trace_manifest=_descriptor(row["trace_manifest"], f"{label}.trace_manifest"),
    )


def _validate_trace_manifest(document: dict[str, Any], label: str) -> tuple[str, dict[str, str], dict[str, str], dict[str, Any], Descriptor, list[tuple[str, Descriptor]]]:
    row = _exact(
        document,
        {"schema_version", "candidate_id", "bindings", "model", "execution", "corpus", "traces"},
        label,
    )
    if row["schema_version"] != TRACE_MANIFEST_VERSION:
        _fail("unsupported-routing-trace-manifest-version", f"{label}.schema_version is unsupported")
    traces = row["traces"]
    if not isinstance(traces, list) or len(traces) != len(CASE_IDS):
        _fail("invalid-routing-trace-manifest", f"{label}.traces has the wrong case count")
    parsed: list[tuple[str, Descriptor]] = []
    for index, trace_value in enumerate(traces):
        trace = _exact(trace_value, {"case_id", "trace"}, f"{label}.traces[{index}]")
        case_id = _case_id(trace["case_id"], f"{label}.traces[{index}].case_id")
        if case_id != CASE_IDS[index]:
            _fail("routing-trace-case-order", f"{label} trace case order drifted from fixed corpus")
        parsed.append((case_id, _descriptor(trace["trace"], f"{label}.traces[{index}].trace")))
    return (
        _candidate_id(row["candidate_id"], f"{label}.candidate_id"),
        _bindings(row["bindings"], f"{label}.bindings"),
        _model(row["model"], f"{label}.model"),
        _execution_common(row["execution"], f"{label}.execution", executable_descriptor=False),
        _validate_corpus_descriptor(row["corpus"], f"{label}.corpus"),
        parsed,
    )


def _validate_trace_document(document: dict[str, Any], label: str) -> tuple[str, dict[str, str], dict[str, str], dict[str, Any], Descriptor, str, Any]:
    row = _exact(
        document,
        {"schema_version", "candidate_id", "bindings", "model", "execution", "corpus", "case_id", "trace"},
        label,
    )
    if row["schema_version"] != TRACE_VERSION:
        _fail("unsupported-routing-trace-version", f"{label}.schema_version is unsupported")
    _validate_trace_body(row["trace"], f"{label}.trace")
    return (
        _candidate_id(row["candidate_id"], f"{label}.candidate_id"),
        _bindings(row["bindings"], f"{label}.bindings"),
        _model(row["model"], f"{label}.model"),
        _execution_common(row["execution"], f"{label}.execution", executable_descriptor=False),
        _validate_corpus_descriptor(row["corpus"], f"{label}.corpus"),
        _case_id(row["case_id"], f"{label}.case_id"),
        row["trace"],
    )


def _empty_report() -> dict[str, Any]:
    return {
        "schema_version": CHECK_REPORT_VERSION,
        "status": "failed",
        "passed": False,
        "candidate_id": None,
        "freeze_sha256": None,
        "base_release_candidate_report": None,
        "bindings": None,
        "corpus": None,
        "receipt": None,
        "trace_manifest": None,
        "model": None,
        "execution": None,
        "traces": [],
        "checks": [],
        "reason_codes": [],
    }


def _check_inventory(value: Any, label: str) -> None:
    if not isinstance(value, list) or len(value) != len(CHECK_NAMES):
        _fail("invalid-routing-check-report", f"{label} has an invalid check inventory")
    names: list[str] = []
    for index, check_value in enumerate(value):
        check = _exact(check_value, {"name", "passed"}, f"{label}[{index}]")
        if check["passed"] is not True:
            _fail("routing-check-not-passed", f"{label}[{index}] did not pass")
        names.append(qualification._string(check["name"], f"{label}[{index}].name"))
    if tuple(names) != CHECK_NAMES:
        _fail("invalid-routing-check-report", f"{label} check inventory drifted")


def validate_check_report(document: dict[str, Any]) -> RoutingCheckReport:
    """Strictly parse a passed report before the outer checker reruns it."""

    corpus = _load_corpus()
    row = _exact(
        document,
        {
            "schema_version",
            "status",
            "passed",
            "candidate_id",
            "freeze_sha256",
            "base_release_candidate_report",
            "bindings",
            "corpus",
            "receipt",
            "trace_manifest",
            "model",
            "execution",
            "traces",
            "checks",
            "reason_codes",
        },
        "routing check report",
    )
    if row["schema_version"] != CHECK_REPORT_VERSION:
        _fail("unsupported-routing-check-report-version", "routing check report schema_version is unsupported")
    if row["status"] != "passed" or row["passed"] is not True or row["reason_codes"] != []:
        _fail("routing-check-not-passed", "routing check report must be a clean passed result")
    if _descriptor(row["corpus"], "routing check report.corpus") != corpus.descriptor:
        _fail("routing-corpus-binding-mismatch", "routing check report does not bind the reviewed corpus")
    _check_inventory(row["checks"], "routing check report.checks")
    execution = _execution_common(row["execution"], "routing check report.execution", executable_descriptor=True)
    traces_value = row["traces"]
    if not isinstance(traces_value, list) or len(traces_value) != len(corpus.cases):
        _fail("invalid-routing-check-report", "routing check report has the wrong trace count")
    traces: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for index, trace_value in enumerate(traces_value):
        trace = _exact(trace_value, {"case_id", "coverage", "trace"}, f"routing check report.traces[{index}]")
        case = corpus.cases[index]
        case_id = _case_id(trace["case_id"], f"routing check report.traces[{index}].case_id")
        coverage = trace["coverage"]
        if case_id != case["case_id"] or coverage != case["coverage"]:
            _fail("invalid-routing-check-report", "routing check report trace inventory drifted from corpus")
        descriptor = _descriptor(trace["trace"], f"routing check report.traces[{index}].trace")
        if descriptor.path in seen_paths:
            _fail("duplicate-evidence-path", "routing check report aliases trace descriptors")
        seen_paths.add(descriptor.path)
        traces.append({"case_id": case_id, "coverage": coverage, "trace": {"path": descriptor.path, "sha256": descriptor.sha256}})
    return RoutingCheckReport(
        candidate_id=_candidate_id(row["candidate_id"], "routing check report.candidate_id"),
        freeze_sha256=_sha256(row["freeze_sha256"], "routing check report.freeze_sha256"),
        base_release_candidate_report=_descriptor(
            row["base_release_candidate_report"], "routing check report.base_release_candidate_report"
        ),
        bindings=_bindings(row["bindings"], "routing check report.bindings"),
        corpus=corpus.descriptor,
        receipt=_descriptor(row["receipt"], "routing check report.receipt"),
        trace_manifest=_descriptor(row["trace_manifest"], "routing check report.trace_manifest"),
        model=_model(row["model"], "routing check report.model"),
        execution={
            **{key: value for key, value in execution.items() if key != "test_executable"},
            "test_executable": {
                "path": execution["test_executable"].path,
                "sha256": execution["test_executable"].sha256,
            },
        },
        traces=tuple(traces),
    )


def evaluate(
    freeze_path: Path,
    evidence_root: Path,
    receipt_path: Path | str,
    *,
    expected_freeze_sha256: str,
) -> dict[str, Any]:
    """Replay fixed C02 routing evidence without launching inference or CUDA."""

    report = _empty_report()
    try:
        corpus = _load_corpus()
        report["corpus"] = {"path": corpus.descriptor.path, "sha256": corpus.descriptor.sha256}
        expected_digest = _sha256(expected_freeze_sha256, "--expected-freeze-sha256")
        freeze_raw = qualification._read_regular_path(freeze_path, "freeze manifest")
        freeze_sha256 = hashlib.sha256(freeze_raw).hexdigest()
        report["freeze_sha256"] = freeze_sha256
        if freeze_sha256 != expected_digest:
            _fail("candidate-sha-mismatch", "freeze manifest SHA-256 differs from trusted input")
        frozen = qualification._validate_freeze(qualification._parse_document(freeze_raw, "freeze manifest"))
        report["candidate_id"] = frozen.candidate_id
        report["model"] = frozen.models["smollm2"]
        reserved_paths = {
            frozen.final_manifest.path,
            frozen.final_report.path,
            *(descriptor.path for descriptor in frozen.receipts.values()),
        }

        base_raw, base_report_sha256 = qualification.revalidate_base_release_candidate(
            frozen, freeze_sha256, evidence_root
        )
        if hashlib.sha256(base_raw).hexdigest() != base_report_sha256:
            _fail("base-report-replay-digest-mismatch", "Gate E replay returned inconsistent bytes/digest")
        report["base_release_candidate_report"] = {
            "path": frozen.final_report.path,
            "sha256": base_report_sha256,
        }

        receipt_relative, receipt_raw, receipt_document = _read_relative_json(
            evidence_root, receipt_path, "routing receipt"
        )
        if receipt_relative in reserved_paths:
            _fail("reserved-output-path-collision", "raw routing receipt replaces a freeze-declared output")
        if receipt_relative == corpus.descriptor.path:
            _fail("source-contract-path-collision", "raw routing receipt may not reuse the source corpus descriptor path")
        receipt_descriptor = Descriptor(receipt_relative, hashlib.sha256(receipt_raw).hexdigest())
        report["receipt"] = {"path": receipt_descriptor.path, "sha256": receipt_descriptor.sha256}
        receipt = _validate_receipt(receipt_document, "routing receipt")
        _validate_bound_header(
            candidate_id=receipt.candidate_id,
            bindings=receipt.bindings,
            model=receipt.model,
            frozen=frozen,
            freeze_sha256=freeze_sha256,
            base_report_sha256=base_report_sha256,
            label="routing receipt",
        )
        report["bindings"] = receipt.bindings
        _validate_execution_for_frozen(receipt.execution, frozen, "routing receipt.execution")

        used_paths = {receipt_descriptor.path, corpus.descriptor.path}
        executable_descriptor = receipt.execution["test_executable"]
        _sha256_relative_executable(
            evidence_root,
            executable_descriptor,
            "routing release executable",
            used_paths=used_paths,
            reserved_paths=reserved_paths,
        )
        report["execution"] = {
            **{key: value for key, value in receipt.execution.items() if key != "test_executable"},
            "test_executable": {"path": executable_descriptor.path, "sha256": executable_descriptor.sha256},
        }

        manifest_raw, manifest_document = _read_described_json(
            evidence_root,
            receipt.trace_manifest,
            "routing trace manifest",
            used_paths=used_paths,
            reserved_paths=reserved_paths,
        )
        report["trace_manifest"] = {
            "path": receipt.trace_manifest.path,
            "sha256": hashlib.sha256(manifest_raw).hexdigest(),
        }
        (
            manifest_candidate,
            manifest_bindings,
            manifest_model,
            manifest_execution,
            manifest_corpus,
            manifest_traces,
        ) = _validate_trace_manifest(manifest_document, "routing trace manifest")
        _validate_bound_header(
            candidate_id=manifest_candidate,
            bindings=manifest_bindings,
            model=manifest_model,
            frozen=frozen,
            freeze_sha256=freeze_sha256,
            base_report_sha256=base_report_sha256,
            label="routing trace manifest",
        )
        _validate_execution_for_frozen(manifest_execution, frozen, "routing trace manifest.execution")
        if manifest_corpus != corpus.descriptor or manifest_execution["test_executable_sha256"] != executable_descriptor.sha256:
            _fail("routing-manifest-binding-mismatch", "routing trace manifest drifts from receipt execution/corpus")

        traces: list[dict[str, Any]] = []
        for index, (case_id, trace_descriptor) in enumerate(manifest_traces):
            trace_raw, trace_document = _read_described_json(
                evidence_root,
                trace_descriptor,
                f"routing trace {case_id}",
                used_paths=used_paths,
                reserved_paths=reserved_paths,
            )
            (
                trace_candidate,
                trace_bindings,
                trace_model,
                trace_execution,
                trace_corpus,
                traced_case_id,
                trace_body,
            ) = _validate_trace_document(trace_document, f"routing trace {case_id}")
            _validate_bound_header(
                candidate_id=trace_candidate,
                bindings=trace_bindings,
                model=trace_model,
                frozen=frozen,
                freeze_sha256=freeze_sha256,
                base_report_sha256=base_report_sha256,
                label=f"routing trace {case_id}",
            )
            _validate_execution_for_frozen(trace_execution, frozen, f"routing trace {case_id}.execution")
            expected_case = corpus.cases[index]
            if (
                traced_case_id != case_id
                or trace_corpus != corpus.descriptor
                or trace_execution != manifest_execution
                or trace_body != expected_case["trace"]
            ):
                _fail("fixed-routing-trace-mismatch", f"routing trace {case_id} differs from reviewed fixed release evidence")
            traces.append(
                {
                    "case_id": case_id,
                    "coverage": expected_case["coverage"],
                    "trace": {"path": trace_descriptor.path, "sha256": hashlib.sha256(trace_raw).hexdigest()},
                }
            )

        report.update(
            {
                "status": "passed",
                "passed": True,
                "traces": traces,
                "checks": [{"name": name, "passed": True} for name in CHECK_NAMES],
            }
        )
    except qualification.IncomparableError as error:
        report["status"] = "incomparable"
        report["reason_codes"] = [getattr(error, "reason_code", "incomparable-binding")]
    except qualification.GateFailure as error:
        report["reason_codes"] = [getattr(error, "reason_code", "gate-failed")]
    except (OSError, qualification.QualificationError) as error:
        report["reason_codes"] = [getattr(error, "reason_code", "invalid-input")]
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", required=True, type=Path)
    parser.add_argument("--expected-freeze-sha256", required=True)
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--receipt", required=True, help="relative raw routing receipt path below evidence root")
    parser.add_argument("--report", type=Path, help="create semantic report without overwriting")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = evaluate(
        args.freeze,
        args.evidence_root,
        args.receipt,
        expected_freeze_sha256=args.expected_freeze_sha256,
    )
    encoded = json.dumps(report, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.report is not None:
        try:
            qualification._write_create_only(args.report, report)
        except qualification.QualificationError as error:
            print(str(error), file=sys.stderr)
            return 2
    print(encoded, end="")
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
