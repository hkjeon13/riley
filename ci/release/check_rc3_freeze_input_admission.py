#!/usr/bin/env python3
"""Bind RC3 freeze inputs structurally without creating a freeze.

This checker is deliberately a read-only, pre-freeze admission boundary. It
replays the reviewed source pre-freeze checker, opens one private external
evidence root through a held no-follow descriptor, rehashes every declared
external input, and returns a canonical structural report on stdout.

It never writes a freeze manifest, marker, candidate result, Gate E report,
semantic receipt, or qualification decision. A bound result remains
not-frozen and not-run; later producers must replay the original external
request and leaves rather than treating this diagnostic as semantic or
finalization authority.
"""

from __future__ import annotations

import argparse
import fcntl
import os
import re
import sys
from dataclasses import dataclass

_BYTECODE_DISABLED_AT_STARTUP = bool(sys.flags.dont_write_bytecode)
_BYTECODE_DISABLED_ON_MODULE_ENTRY = sys.dont_write_bytecode
sys.dont_write_bytecode = True

from pathlib import Path
from typing import Any, Callable, NoReturn, Sequence, TypeVar

import check_rc3_prefreeze as prefreeze
import check_reconstructed_prior_baseline_v2 as baseline
import provenance_v2_common as common


REQUEST_VERSION = "riley.rc3-freeze-input-request.v1"
REPORT_VERSION = "riley.rc3-freeze-input-admission.v1"
STRUCTURAL_AUTHORITY = "freeze-input-structural-only"
SCOPE = "freeze-input-structural-only"
MAX_REQUEST_BYTES = 512 * 1024
MAX_TEXT_INPUT_BYTES = 8 * 1024 * 1024
MAX_MODELS = 32
MAX_WEIGHTS_PER_MODEL = 4096
MAX_EXTERNAL_DESCRIPTORS = 8192
MAX_TOTAL_EXTERNAL_INPUT_BYTES = 1 << 40
DIRECT_REQUEST_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,122}\.json$")
REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
CANDIDATE_ID_RE = re.compile(
    r"^riley-((?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\."
    r"(?:0|[1-9][0-9]*))-rc([1-9][0-9]*)$"
)
IMAGE_ID_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
ENVIRONMENT_KEY_RE = re.compile(r"^[A-Z_][A-Z0-9_]*$")
PROFILE_NAMES = ("stable-default", "max-performance-exact")
RUNNER_OWNED_ARGUMENTS = {
    "serve",
    "--help",
    "-h",
    "--version",
    "--model",
    "--bind",
    "--device",
    "--c02-candidate-id",
    "--c02-configuration-profile",
    "--c02-startup-artifact",
}
RUNNER_OWNED_ARGUMENT_PREFIXES = tuple(
    f"{value}=" for value in RUNNER_OWNED_ARGUMENTS if value.startswith("--")
)
SELF_REFERENTIAL_ARGUMENT_PREFIXES = (
    "--freeze-sha",
    "--gate-e-report-sha",
    "--configuration-sha",
    "--base-release-candidate-report-sha",
    "--c02-freeze-sha",
    "--c02-gate-e-report-sha",
    "--c02-configuration-sha",
    "--c02-base-release-candidate-report-sha",
)
FORBIDDEN_ENVIRONMENT_KEYS = {
    "RILEY_FREEZE_SHA",
    "RILEY_GATE_E_REPORT_SHA",
    "RILEY_CONFIGURATION_SHA",
    "RILEY_BASE_RELEASE_CANDIDATE_REPORT_SHA",
}


class FreezeInputAdmissionError(ValueError):
    """The submitted pre-freeze request cannot safely be structurally bound."""


T = TypeVar("T")


@dataclass(frozen=True)
class FreezeInputReplay:
    """One no-write replay of the original RC3 freeze-input request.

    This is deliberately not an admission report and is never persisted by
    this module.  Create-only frozen-candidate producers may consume it only
    while retaining the caller-held source and evidence-root descriptors.
    """

    request_descriptor: common.EvidenceDescriptor
    request: dict[str, Any]
    descriptors: tuple[common.EvidenceDescriptor, ...]
    source_prefreeze: dict[str, Any]
    reconstructed_baseline: baseline.BaselineManifest


@dataclass(frozen=True)
class FreezeInputReplayPreflight:
    """Bounded control-plane facts collected before candidate raw streaming.

    This in-memory capability is intentionally not a report or a persisted
    admission result.  A caller that needs a cross-closure resource boundary
    may inspect the parsed request descriptors before asking the matching
    completion routine to rehash any candidate raw leaf.
    """

    request_descriptor: common.EvidenceDescriptor
    request_document: dict[str, Any]
    request: dict[str, Any]
    descriptors: tuple[common.EvidenceDescriptor, ...]
    source_prefreeze_before: dict[str, Any]
    reconstructed_baseline: baseline.BaselineManifest


def _fail(code: str, message: str) -> NoReturn:
    error = FreezeInputAdmissionError(f"{code}: {message}")
    error.reason_code = code  # type: ignore[attr-defined]
    raise error


def _require_bytecode_cache_disabled() -> None:
    if not (
        _BYTECODE_DISABLED_AT_STARTUP and _BYTECODE_DISABLED_ON_MODULE_ENTRY
    ):
        _fail(
            "bytecode-cache-write-not-permitted",
            "invoke this checker with python3 -B or PYTHONDONTWRITEBYTECODE=1",
        )


def _common(call: Callable[[], T]) -> T:
    try:
        return call()
    except common.ProvenanceV2Error as error:
        _fail(getattr(error, "reason_code", "unsafe-evidence"), str(error))


def _prefreeze(call: Callable[[], T]) -> T:
    try:
        return call()
    except prefreeze.Rc3PrefreezeError as error:
        _fail(getattr(error, "reason_code", "source-prefreeze-failed"), str(error))


def _baseline(call: Callable[[], T]) -> T:
    try:
        return call()
    except baseline.BaselineError as error:
        _fail(getattr(error, "reason_code", "invalid-reconstructed-baseline"), str(error))
    except common.ProvenanceV2Error as error:
        _fail(getattr(error, "reason_code", "invalid-reconstructed-baseline"), str(error))


def _repository_root(value: Path) -> Path:
    try:
        raw = os.fspath(value)
    except TypeError as error:
        _fail("invalid-repository-root", f"--repository-root is not a path: {error}")
    if (
        type(raw) is not str
        or not raw
        or "\x00" in raw
        or not os.path.isabs(raw)
        or raw.startswith("//")
        or raw != os.path.normpath(raw)
    ):
        _fail(
            "invalid-repository-root",
            "--repository-root must be a normalized absolute path",
        )
    return Path(raw)


def _evidence_root(value: Path, repository_root: Path) -> Path:
    try:
        raw = os.fspath(value)
    except TypeError as error:
        _fail("invalid-evidence-root", f"--evidence-root is not a path: {error}")
    if (
        type(raw) is not str
        or not raw
        or "\x00" in raw
        or not os.path.isabs(raw)
        or raw.startswith("//")
        or raw != os.path.normpath(raw)
    ):
        _fail(
            "invalid-evidence-root",
            "--evidence-root must be a normalized absolute path",
        )
    root = Path(raw)
    try:
        root.relative_to(repository_root)
    except ValueError:
        return root
    _fail(
        "evidence-root-inside-source-checkout",
        "--evidence-root must be outside --repository-root",
    )


def _request_name(value: str) -> str:
    if type(value) is not str or DIRECT_REQUEST_RE.fullmatch(value) is None:
        _fail(
            "request-must-be-direct-root-leaf",
            "--request must be one direct nonhidden root JSON leaf",
        )
    return value


def _revision(value: Any, label: str) -> str:
    if (
        type(value) is not str
        or REVISION_RE.fullmatch(value) is None
        or value == "0" * 40
    ):
        _fail("invalid-source-revision", f"{label} must be a non-zero lowercase Git SHA")
    return value


def _candidate_id(value: Any, label: str) -> tuple[str, str]:
    if type(value) is not str:
        _fail("invalid-candidate-id", f"{label} must be text")
    match = CANDIDATE_ID_RE.fullmatch(value)
    if match is None:
        _fail(
            "invalid-candidate-id",
            f"{label} must be canonical riley-X.Y.Z-rcN",
        )
    return value, match.group(1)


def _expected_reconstructed_baseline_tag(candidate_id: str) -> str:
    match = CANDIDATE_ID_RE.fullmatch(candidate_id)
    assert match is not None
    rc_digits = match.group(2)
    if rc_digits == "1":
        _fail(
            "no-prior-rc-baseline",
            "an RC1 candidate has no immediately preceding reconstructed RC baseline",
        )
    previous = list(rc_digits)
    for index in range(len(previous) - 1, -1, -1):
        if previous[index] == "0":
            previous[index] = "9"
            continue
        previous[index] = str(int(previous[index]) - 1)
        break
    previous_rc = "".join(previous).lstrip("0")
    assert previous_rc
    return f"riley-{match.group(1)}-rc{previous_rc}"


def _text(value: Any, label: str, *, maximum: int = 1024) -> str:
    if (
        type(value) is not str
        or not value
        or len(value) > maximum
        or "\x00" in value
        or any(ord(character) < 0x20 or ord(character) == 0x7F for character in value)
    ):
        _fail("invalid-freeze-input-request", f"{label} must be bounded non-control text")
    return value


def _image_id(value: Any, label: str) -> str:
    if type(value) is not str or IMAGE_ID_RE.fullmatch(value) is None:
        _fail("invalid-image-id", f"{label} must be a lowercase OCI SHA-256 ID")
    if value == "sha256:" + "0" * 64:
        _fail("invalid-image-id", f"{label} must not be all zeroes")
    return value


def _exact(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != fields:
        _fail(
            "invalid-freeze-input-request",
            f"{label} must contain exactly {', '.join(sorted(fields))}",
        )
    return value


def _array(value: Any, label: str, *, minimum: int, maximum: int) -> list[Any]:
    if type(value) is not list or not minimum <= len(value) <= maximum:
        _fail(
            "invalid-freeze-input-request",
            f"{label} must contain from {minimum} through {maximum} entries",
        )
    return value


def _descriptor(
    value: Any,
    label: str,
    *,
    allow_empty: bool = False,
) -> common.EvidenceDescriptor:
    descriptor = _common(lambda: common.parse_descriptor(value, label))
    if not allow_empty and descriptor.byte_length == 0:
        _fail("empty-evidence-input", f"{label}.byte_length must be positive")
    return descriptor


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


def _read_request(
    root_fd: int,
    request_name: str,
) -> tuple[bytes, common.EvidenceDescriptor, dict[str, Any]]:
    raw = _common(
        lambda: common.read_bounded_regular_relative(
            root_fd,
            request_name,
            "freeze-input request",
            maximum_bytes=MAX_REQUEST_BYTES,
        )
    )
    descriptor = _common(
        lambda: common.descriptor_for_bytes(
            request_name,
            raw,
            "freeze-input request",
        )
    )
    document = _common(
        lambda: common.parse_canonical_json(
            raw,
            "freeze-input request",
            maximum_bytes=MAX_REQUEST_BYTES,
        )
    )
    assert isinstance(document, dict)
    return raw, descriptor, document


def _parse_source(value: Any, descriptors: list[common.EvidenceDescriptor]) -> dict[str, Any]:
    row = _exact(
        value,
        {"git_revision", "archive", "cargo_lock", "extension_registry"},
        "request.source",
    )
    archive = _descriptor(row["archive"], "request.source.archive")
    cargo_lock = _descriptor(row["cargo_lock"], "request.source.cargo_lock")
    extension_registry = _descriptor(
        row["extension_registry"],
        "request.source.extension_registry",
    )
    descriptors.extend((archive, cargo_lock, extension_registry))
    return {
        "git_revision": _revision(row["git_revision"], "request.source.git_revision"),
        "archive": archive,
        "cargo_lock": cargo_lock,
        "extension_registry": extension_registry,
    }


def _parse_release(value: Any, descriptors: list[common.EvidenceDescriptor]) -> dict[str, Any]:
    row = _exact(value, {"elf", "container"}, "request.release")
    container = _exact(
        row["container"],
        {"image_id", "image_digest", "inspect"},
        "request.release.container",
    )
    elf = _descriptor(row["elf"], "request.release.elf")
    inspect = _descriptor(
        container["inspect"],
        "request.release.container.inspect",
    )
    descriptors.extend((elf, inspect))
    return {
        "elf": elf,
        "container": {
            "image_id": _image_id(
                container["image_id"],
                "request.release.container.image_id",
            ),
            "image_digest": _image_id(
                container["image_digest"],
                "request.release.container.image_digest",
            ),
            "inspect": inspect,
        },
    }


def _parse_toolchain(
    value: Any,
    descriptors: list[common.EvidenceDescriptor],
) -> dict[str, Any]:
    row = _exact(
        value,
        {
            "probe",
            "cuda_c_abi_version",
            "rust_version",
            "nvcc_version",
            "driver_version",
            "cuda_runtime_version",
            "cuda_toolkit_version",
            "cublas_version",
        },
        "request.toolchain",
    )
    probe = _descriptor(row["probe"], "request.toolchain.probe")
    descriptors.append(probe)
    return {
        "probe": probe,
        "cuda_c_abi_version": _text(
            row["cuda_c_abi_version"],
            "request.toolchain.cuda_c_abi_version",
        ),
        "rust_version": _text(row["rust_version"], "request.toolchain.rust_version"),
        "nvcc_version": _text(row["nvcc_version"], "request.toolchain.nvcc_version"),
        "driver_version": _text(
            row["driver_version"],
            "request.toolchain.driver_version",
        ),
        "cuda_runtime_version": _text(
            row["cuda_runtime_version"],
            "request.toolchain.cuda_runtime_version",
        ),
        "cuda_toolkit_version": _text(
            row["cuda_toolkit_version"],
            "request.toolchain.cuda_toolkit_version",
        ),
        "cublas_version": _text(
            row["cublas_version"],
            "request.toolchain.cublas_version",
        ),
    }


def _parse_models(
    value: Any,
    descriptors: list[common.EvidenceDescriptor],
) -> list[dict[str, Any]]:
    rows = _array(value, "request.models", minimum=1, maximum=MAX_MODELS)
    models: list[dict[str, Any]] = []
    model_ids: set[str] = set()
    for index, raw_row in enumerate(rows):
        row = _exact(
            raw_row,
            {"model_id", "revision", "tree", "config", "tokenizer", "weights"},
            f"request.models[{index}]",
        )
        model_id = _text(row["model_id"], f"request.models[{index}].model_id")
        if model_id in model_ids:
            _fail("duplicate-model-id", f"request.models repeats {model_id!r}")
        model_ids.add(model_id)
        tree = _descriptor(row["tree"], f"request.models[{index}].tree")
        config = _descriptor(row["config"], f"request.models[{index}].config")
        tokenizer = _descriptor(row["tokenizer"], f"request.models[{index}].tokenizer")
        weights = [
            _descriptor(item, f"request.models[{index}].weights[{weight_index}]")
            for weight_index, item in enumerate(
                _array(
                    row["weights"],
                    f"request.models[{index}].weights",
                    minimum=1,
                    maximum=MAX_WEIGHTS_PER_MODEL,
                )
            )
        ]
        descriptors.extend((tree, config, tokenizer, *weights))
        models.append(
            {
                "model_id": model_id,
                "revision": _text(
                    row["revision"],
                    f"request.models[{index}].revision",
                ),
                "tree": tree,
                "config": config,
                "tokenizer": tokenizer,
                "weights": weights,
            }
        )
    return models


def _parse_launch_profiles(
    value: Any,
    descriptors: list[common.EvidenceDescriptor],
) -> list[dict[str, Any]]:
    rows = _array(value, "request.launch_profiles", minimum=2, maximum=2)
    profiles: list[dict[str, Any]] = []
    for index, expected_profile in enumerate(PROFILE_NAMES):
        row = _exact(
            rows[index],
            {"profile", "arguments", "environment"},
            f"request.launch_profiles[{index}]",
        )
        if row["profile"] != expected_profile:
            _fail(
                "invalid-launch-profile-order",
                f"request.launch_profiles[{index}].profile must be {expected_profile!r}",
            )
        arguments = _descriptor(
            row["arguments"],
            f"request.launch_profiles[{index}].arguments",
            allow_empty=True,
        )
        environment = _descriptor(
            row["environment"],
            f"request.launch_profiles[{index}].environment",
        )
        descriptors.extend((arguments, environment))
        profiles.append(
            {
                "profile": expected_profile,
                "arguments": arguments,
                "environment": environment,
            }
        )
    return profiles


def _parse_correctness(
    value: Any,
    descriptors: list[common.EvidenceDescriptor],
) -> dict[str, Any]:
    row = _exact(value, {"contract", "report"}, "request.correctness")
    contract = _descriptor(row["contract"], "request.correctness.contract")
    report = _descriptor(row["report"], "request.correctness.report")
    descriptors.extend((contract, report))
    return {"contract": contract, "report": report}


def _parse_rollback(
    value: Any,
    descriptors: list[common.EvidenceDescriptor],
) -> dict[str, Any]:
    row = _exact(
        value,
        {"reconstructed_baseline_manifest"},
        "request.rollback",
    )
    manifest = _descriptor(
        row["reconstructed_baseline_manifest"],
        "request.rollback.reconstructed_baseline_manifest",
    )
    descriptors.append(manifest)
    return {"reconstructed_baseline_manifest": manifest}


def _bounded_unique_descriptors(
    descriptors: list[common.EvidenceDescriptor],
) -> tuple[common.EvidenceDescriptor, ...]:
    if len(descriptors) > MAX_EXTERNAL_DESCRIPTORS:
        _fail(
            "too-many-external-descriptors",
            f"freeze-input request exceeds {MAX_EXTERNAL_DESCRIPTORS} external descriptors",
        )
    parsed = _common(
        lambda: common.require_unique_descriptors(
            descriptors,
            "freeze-input request descriptors",
        )
    )
    total_bytes = sum(descriptor.byte_length for descriptor in parsed)
    if total_bytes > MAX_TOTAL_EXTERNAL_INPUT_BYTES:
        _fail(
            "external-evidence-byte-budget-exceeded",
            "freeze-input request exceeds its total external evidence byte budget",
        )
    return parsed


def _parse_request(document: dict[str, Any]) -> tuple[dict[str, Any], tuple[common.EvidenceDescriptor, ...]]:
    row = _exact(
        document,
        {
            "schema_version",
            "candidate_id",
            "source",
            "release",
            "toolchain",
            "models",
            "launch_profiles",
            "correctness",
            "rollback",
        },
        "freeze-input request",
    )
    if row["schema_version"] != REQUEST_VERSION:
        _fail(
            "unsupported-freeze-input-request-version",
            "freeze-input request.schema_version is unsupported",
        )
    descriptors: list[common.EvidenceDescriptor] = []
    candidate_id, _candidate_version = _candidate_id(
        row["candidate_id"],
        "request.candidate_id",
    )
    parsed = {
        "candidate_id": candidate_id,
        "source": _parse_source(row["source"], descriptors),
        "release": _parse_release(row["release"], descriptors),
        "toolchain": _parse_toolchain(row["toolchain"], descriptors),
        "models": _parse_models(row["models"], descriptors),
        "launch_profiles": _parse_launch_profiles(
            row["launch_profiles"],
            descriptors,
        ),
        "correctness": _parse_correctness(row["correctness"], descriptors),
        "rollback": _parse_rollback(row["rollback"], descriptors),
    }
    return parsed, _bounded_unique_descriptors(descriptors)


def _descriptor_bytes_equal(
    actual: common.EvidenceDescriptor,
    expected: common.EvidenceDescriptor,
    label: str,
) -> None:
    if (
        actual.sha256 != expected.sha256
        or actual.byte_length != expected.byte_length
    ):
        _fail(
            "source-input-drift",
            f"{label} must exactly match the source pre-freeze SHA-256 and byte length",
        )


def _source_prefreeze_report(
    repository_root: Path,
    repository_root_fd: int,
    expected_revision: str,
    candidate_id: str,
) -> dict[str, Any]:
    report = _prefreeze(
        lambda: prefreeze.check_prefreeze_on_held_root_fd(
            repository_root,
            repository_root_fd,
            expected_revision,
            candidate_id,
        )
    )
    if (
        type(report) is not dict
        or report.get("schema_version") != prefreeze.PREFREEZE_REPORT_VERSION
        or report.get("scope") != "source-pre-freeze-only"
        or report.get("candidate_status") != "not-frozen"
        or report.get("qualification_status") != "not-run"
        or report.get("candidate_id") != candidate_id
        or report.get("source_revision") != expected_revision
    ):
        _fail(
            "invalid-source-prefreeze-report",
            "source pre-freeze checker returned an unexpected report",
        )
    source_inputs = report.get("source_inputs")
    if type(source_inputs) is not dict or set(source_inputs) != {
        "workspace_manifests",
        "cargo_lock",
        "extension_registry",
        "server_defaults_source",
    }:
        _fail(
            "invalid-source-prefreeze-report",
            "source pre-freeze report does not expose its reviewed source inputs",
        )
    manifests = source_inputs["workspace_manifests"]
    if type(manifests) is not list or not manifests:
        _fail(
            "invalid-source-prefreeze-report",
            "source pre-freeze report must expose workspace manifests",
        )
    _common(
        lambda: common.require_unique_descriptors(
            manifests,
            "source pre-freeze workspace manifests",
        )
    )
    for field in ("cargo_lock", "extension_registry", "server_defaults_source"):
        _common(
            lambda field=field: common.parse_descriptor(
                source_inputs[field],
                f"source pre-freeze {field}",
            )
        )
    return report


def _line_entries(raw: bytes, label: str) -> list[str]:
    if b"\r" in raw:
        _fail("invalid-launch-input", f"{label} must not contain carriage returns")
    if not raw:
        return []
    lines = raw.split(b"\n")
    if raw.endswith(b"\n"):
        lines.pop()
    result: list[str] = []
    for index, line in enumerate(lines):
        if not line:
            _fail("invalid-launch-input", f"{label} contains an empty line at {index + 1}")
        try:
            text = line.decode("utf-8")
        except UnicodeDecodeError as error:
            _fail("invalid-launch-input", f"{label} is not UTF-8: {error}")
        if "\x00" in text:
            _fail("invalid-launch-input", f"{label} contains a NUL byte")
        result.append(text)
    return result


def _validate_launch_arguments(
    root_fd: int,
    descriptor: common.EvidenceDescriptor,
    label: str,
) -> None:
    raw = _common(
        lambda: common.read_descriptor_bytes(
            root_fd,
            descriptor,
            label,
            maximum_bytes=MAX_TEXT_INPUT_BYTES,
        )
    )
    for argument in _line_entries(raw, label):
        if (
            argument in RUNNER_OWNED_ARGUMENTS
            or argument.startswith(RUNNER_OWNED_ARGUMENT_PREFIXES)
            or argument.startswith("--c02-")
        ):
            _fail(
                "runner-owned-launch-argument",
                f"{label} attempts to override a runner-owned argument: {argument!r}",
            )
        if argument.startswith(SELF_REFERENTIAL_ARGUMENT_PREFIXES):
            _fail(
                "self-referential-launch-argument",
                f"{label} contains a forbidden self-referential argument: {argument!r}",
            )


def _validate_launch_environment(
    root_fd: int,
    descriptor: common.EvidenceDescriptor,
    label: str,
) -> None:
    raw = _common(
        lambda: common.read_descriptor_bytes(
            root_fd,
            descriptor,
            label,
            maximum_bytes=MAX_TEXT_INPUT_BYTES,
        )
    )
    entries = _line_entries(raw, label)
    if not entries:
        _fail("empty-launch-environment", f"{label} must contain at least one KEY=VALUE")
    seen: set[str] = set()
    for entry in entries:
        if "=" not in entry:
            _fail("invalid-launch-environment", f"{label} must contain KEY=VALUE entries")
        key, _value = entry.split("=", 1)
        if ENVIRONMENT_KEY_RE.fullmatch(key) is None:
            _fail("invalid-launch-environment", f"{label} has an invalid key {key!r}")
        if key in seen:
            _fail("duplicate-launch-environment-key", f"{label} repeats key {key!r}")
        seen.add(key)
        if key in FORBIDDEN_ENVIRONMENT_KEYS:
            _fail(
                "self-referential-launch-environment",
                f"{label} contains forbidden self-referential input {key!r}",
            )


def _verify_opaque_inputs(root_fd: int, request: dict[str, Any]) -> None:
    source = request["source"]
    release = request["release"]
    toolchain = request["toolchain"]
    correctness = request["correctness"]
    opaque: list[tuple[common.EvidenceDescriptor, str]] = [
        (source["archive"], "source archive"),
        (source["cargo_lock"], "external Cargo.lock"),
        (source["extension_registry"], "external extension registry"),
        (release["elf"], "release ELF"),
        (release["container"]["inspect"], "container inspect"),
        (toolchain["probe"], "toolchain probe"),
        (correctness["contract"], "correctness contract"),
        (correctness["report"], "correctness report"),
    ]
    for model_index, model in enumerate(request["models"]):
        opaque.extend(
            (
                (model["tree"], f"models[{model_index}] tree"),
                (model["config"], f"models[{model_index}] config"),
                (model["tokenizer"], f"models[{model_index}] tokenizer"),
            )
        )
        opaque.extend(
            (weight, f"models[{model_index}] weights[{weight_index}]")
            for weight_index, weight in enumerate(model["weights"])
        )
    for descriptor, label in opaque:
        _common(
            lambda descriptor=descriptor, label=label: common.verify_descriptor_file(
                root_fd,
                descriptor,
                label,
            )
        )


def _validate_reconstructed_baseline(
    root_fd: int,
    descriptor: common.EvidenceDescriptor,
    candidate_id: str,
) -> baseline.BaselineManifest:
    raw = _common(
        lambda: common.read_descriptor_bytes(
            root_fd,
            descriptor,
            "reconstructed prior baseline manifest",
            maximum_bytes=MAX_REQUEST_BYTES,
        )
    )
    document = _common(
        lambda: common.parse_canonical_json(
            raw,
            "reconstructed prior baseline manifest",
            maximum_bytes=MAX_REQUEST_BYTES,
        )
    )
    assert isinstance(document, dict)
    if document.get("schema_version") == baseline.LEGACY_MANIFEST_VERSION:
        _fail(
            "rollback-binary-provenance-required",
            "freeze-input admission requires reconstructed baseline v2 with A/B server-binary binding",
        )
    manifest = _baseline(lambda: baseline.parse_manifest(document))
    expected_tag = _expected_reconstructed_baseline_tag(candidate_id)
    if manifest.source.tag_name != expected_tag:
        _fail(
            "reconstructed-baseline-tag-mismatch",
            "reconstructed baseline must name the immediately preceding RC tag "
            f"{expected_tag!r} for candidate {candidate_id!r}",
        )
    return manifest


def _as_json(value: Any) -> Any:
    if isinstance(value, common.EvidenceDescriptor):
        return value.as_json()
    if type(value) is list:
        return [_as_json(item) for item in value]
    if type(value) is dict:
        return {key: _as_json(item) for key, item in value.items()}
    return value


def _source_inputs_from_report(report: dict[str, Any]) -> dict[str, Any]:
    source_inputs = report["source_inputs"]
    return {
        "workspace_manifests": source_inputs["workspace_manifests"],
        "cargo_lock": source_inputs["cargo_lock"],
        "extension_registry": source_inputs["extension_registry"],
        "server_defaults_source": source_inputs["server_defaults_source"],
    }


def _report(
    *,
    request_descriptor: common.EvidenceDescriptor,
    request: dict[str, Any],
    source_prefreeze: dict[str, Any],
    reconstructed_baseline: baseline.BaselineManifest,
) -> dict[str, Any]:
    return {
        "schema_version": REPORT_VERSION,
        "scope": SCOPE,
        "status": "bound",
        "authority": STRUCTURAL_AUTHORITY,
        "candidate_status": "not-frozen",
        "qualification_status": "not-run",
        "candidate_id": request["candidate_id"],
        "source_revision": request["source"]["git_revision"],
        "workspace_version": source_prefreeze["workspace_version"],
        "request": request_descriptor.as_json(),
        "source_pre_freeze": {
            "schema_version": source_prefreeze["schema_version"],
            "source_inputs": _source_inputs_from_report(source_prefreeze),
        },
        "bound_inputs": _as_json(request),
        "reconstructed_baseline": {
            "baseline_id": reconstructed_baseline.baseline_id,
            "tag_name": reconstructed_baseline.source.tag_name,
            "relationship": "immediately-prior-rc-same-semver",
        },
        "rollback_scope": "reconstructed-baseline-vocabulary-only",
        "checks": [
            {"name": "bytecode-cache-disabled-before-evidence-read", "bound": True},
            {"name": "source-pre-freeze-replay-before-and-after-input-read", "bound": True},
            {"name": "external-source-lock-and-registry-match-reviewed-source", "bound": True},
            {"name": "no-follow-rehash-of-declared-external-inputs", "bound": True},
            {"name": "bounded-external-descriptor-count-and-byte-budget", "bound": True},
            {"name": "runner-owned-and-self-referential-launch-inputs-rejected", "bound": True},
            {"name": "reconstructed-baseline-vocabulary-only", "bound": True},
            {"name": "request-stable-during-admission", "bound": True},
        ],
        "reason_codes": [],
    }


def prepare_rc3_freeze_input_request_on_held_root_fd(
    repository_root: Path,
    repository_root_fd: int,
    expected_revision: str,
    candidate_id: str,
    evidence_root_fd: int,
    request_name: str,
) -> FreezeInputReplayPreflight:
    """Collect bounded request facts before any candidate raw leaf is read.

    The primitive does not open, close, lock, or write either caller-owned
    root descriptor.  It reads only the bounded request/baseline control
    plane and the source pre-freeze oracle, returning an in-memory capability
    that a same-stack caller can use to impose cross-closure limits.
    """

    _require_bytecode_cache_disabled()
    source_root = _repository_root(repository_root)
    revision = _revision(expected_revision, "--expected-revision")
    candidate, _candidate_version = _candidate_id(candidate_id, "--candidate-id")
    direct_request_name = _request_name(request_name)
    _common(
        lambda: common.require_private_evidence_directory_fd(
            evidence_root_fd,
            "freeze-input evidence root",
        )
    )

    before_prefreeze = _source_prefreeze_report(
        source_root,
        repository_root_fd,
        revision,
        candidate,
    )
    _request_raw, request_descriptor, document = _read_request(
        evidence_root_fd,
        direct_request_name,
    )
    request, descriptors = _parse_request(document)
    if request["candidate_id"] != candidate:
        _fail(
            "candidate-id-mismatch",
            "request.candidate_id must equal --candidate-id",
        )
    if request["source"]["git_revision"] != revision:
        _fail(
            "source-revision-mismatch",
            "request.source.git_revision must equal --expected-revision",
        )
    if request_descriptor.path in {descriptor.path for descriptor in descriptors}:
        _fail(
            "request-descriptor-path-reused",
            "the request leaf must not be reused as an external input descriptor",
        )
    reconstructed_baseline = _validate_reconstructed_baseline(
        evidence_root_fd,
        request["rollback"]["reconstructed_baseline_manifest"],
        candidate,
    )
    return FreezeInputReplayPreflight(
        request_descriptor=request_descriptor,
        request_document=document,
        request=request,
        descriptors=descriptors,
        source_prefreeze_before=before_prefreeze,
        reconstructed_baseline=reconstructed_baseline,
    )


def complete_rc3_freeze_input_request_on_held_root_fd(
    repository_root: Path,
    repository_root_fd: int,
    expected_revision: str,
    candidate_id: str,
    evidence_root_fd: int,
    request_name: str,
    preflight: FreezeInputReplayPreflight,
) -> FreezeInputReplay:
    """Rehash raw candidate leaves and finish one prepared held-FD replay.

    The caller owns all root FDs and any locks.  This routine deliberately
    runs only after a same-stack caller accepted the prepared descriptor
    closure; it neither opens, closes, locks, nor writes caller-owned roots.
    """

    _require_bytecode_cache_disabled()
    source_root = _repository_root(repository_root)
    revision = _revision(expected_revision, "--expected-revision")
    candidate, _candidate_version = _candidate_id(candidate_id, "--candidate-id")
    direct_request_name = _request_name(request_name)
    if type(preflight) is not FreezeInputReplayPreflight:
        _fail("invalid-freeze-input-preflight", "preflight capability has an unexpected type")
    _common(
        lambda: common.require_private_evidence_directory_fd(
            evidence_root_fd,
            "freeze-input evidence root",
        )
    )
    if preflight.request_descriptor.path != direct_request_name:
        _fail(
            "request-preflight-mismatch",
            "preflight request descriptor does not name the requested root leaf",
        )
    if preflight.request.get("candidate_id") != candidate:
        _fail(
            "candidate-id-mismatch",
            "preflight request candidate_id must equal --candidate-id",
        )
    source = preflight.request.get("source")
    if type(source) is not dict or source.get("git_revision") != revision:
        _fail(
            "source-revision-mismatch",
            "preflight request source.git_revision must equal --expected-revision",
        )
    _request_raw_start, request_descriptor_start, document_start = _read_request(
        evidence_root_fd,
        direct_request_name,
    )
    if (
        request_descriptor_start != preflight.request_descriptor
        or document_start != preflight.request_document
    ):
        _fail(
            "request-preflight-mismatch",
            "the request changed after preflight and before raw inputs were read",
        )
    parsed_request_start, descriptors_start = _parse_request(document_start)
    if (
        parsed_request_start != preflight.request
        or descriptors_start != preflight.descriptors
    ):
        _fail(
            "request-preflight-mismatch",
            "preflight parsed inputs do not match the current request leaf",
        )
    _verify_opaque_inputs(evidence_root_fd, parsed_request_start)
    for profile in preflight.request["launch_profiles"]:
        _validate_launch_arguments(
            evidence_root_fd,
            profile["arguments"],
            f"{profile['profile']} launch arguments",
        )
        _validate_launch_environment(
            evidence_root_fd,
            profile["environment"],
            f"{profile['profile']} launch environment",
        )
    after_prefreeze = _source_prefreeze_report(
        source_root,
        repository_root_fd,
        revision,
        candidate,
    )
    if (
        common.canonical_json_bytes(preflight.source_prefreeze_before)
        != common.canonical_json_bytes(after_prefreeze)
    ):
        _fail(
            "source-prefreeze-changed-during-admission",
            "reviewed source pre-freeze report changed while inputs were read",
        )
    before_inputs = preflight.source_prefreeze_before["source_inputs"]
    _descriptor_bytes_equal(
        preflight.request["source"]["cargo_lock"],
        _common(
            lambda: common.parse_descriptor(
                before_inputs["cargo_lock"],
                "source pre-freeze Cargo.lock",
            )
        ),
        "request.source.cargo_lock",
    )
    _descriptor_bytes_equal(
        preflight.request["source"]["extension_registry"],
        _common(
            lambda: common.parse_descriptor(
                before_inputs["extension_registry"],
                "source pre-freeze extension registry",
            )
        ),
        "request.source.extension_registry",
    )
    _request_raw_end, request_descriptor_end, document_end = _read_request(
        evidence_root_fd,
        direct_request_name,
    )
    if (
        request_descriptor_end != preflight.request_descriptor
        or document_end != preflight.request_document
    ):
        _fail(
            "request-changed-during-admission",
            "freeze-input request changed while its inputs were rehashed",
        )
    return FreezeInputReplay(
        request_descriptor=preflight.request_descriptor,
        request=preflight.request,
        descriptors=preflight.descriptors,
        source_prefreeze=after_prefreeze,
        reconstructed_baseline=preflight.reconstructed_baseline,
    )


def replay_rc3_freeze_input_request_on_held_root_fd(
    repository_root: Path,
    repository_root_fd: int,
    expected_revision: str,
    candidate_id: str,
    evidence_root_fd: int,
    request_name: str,
) -> FreezeInputReplay:
    """Fully replay one request through caller-held source/input root FDs."""

    preflight = prepare_rc3_freeze_input_request_on_held_root_fd(
        repository_root,
        repository_root_fd,
        expected_revision,
        candidate_id,
        evidence_root_fd,
        request_name,
    )
    return complete_rc3_freeze_input_request_on_held_root_fd(
        repository_root,
        repository_root_fd,
        expected_revision,
        candidate_id,
        evidence_root_fd,
        request_name,
        preflight,
    )


def check_rc3_freeze_input_admission(
    repository_root: Path,
    expected_revision: str,
    candidate_id: str,
    evidence_root: Path,
    request_name: str,
) -> dict[str, Any]:
    """Return only structural pre-freeze bindings for one external request."""

    _require_bytecode_cache_disabled()
    source_root = _repository_root(repository_root)
    root = _evidence_root(evidence_root, source_root)
    source_root_fd: int | None = None
    root_fd: int | None = None
    try:
        source_root_fd = _common(
            lambda: common.open_absolute_directory(source_root, "repository root")
        )
        root_fd = _common(
            lambda: common.open_private_evidence_directory(root, "--evidence-root")
        )
        _shared_lock(root_fd, "evidence-root-lock-unavailable", "evidence-root")
        replay = replay_rc3_freeze_input_request_on_held_root_fd(
            source_root,
            source_root_fd,
            expected_revision,
            candidate_id,
            root_fd,
            request_name,
        )
        return _report(
            request_descriptor=replay.request_descriptor,
            request=replay.request,
            source_prefreeze=replay.source_prefreeze,
            reconstructed_baseline=replay.reconstructed_baseline,
        )
    finally:
        _unlock_quietly(root_fd)
        _close_quietly(root_fd)
        _close_quietly(source_root_fd)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", required=True, type=Path)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--request", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = check_rc3_freeze_input_admission(
            args.repository_root,
            args.expected_revision,
            args.candidate_id,
            args.evidence_root,
            args.request,
        )
    except (OSError, FreezeInputAdmissionError) as error:
        print(f"RC3 freeze-input admission failed: {error}", file=sys.stderr)
        return 2
    sys.stdout.buffer.write(common.canonical_json_bytes(report) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
