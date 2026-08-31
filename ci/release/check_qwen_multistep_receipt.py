#!/usr/bin/env python3
"""Fail-closed C02 verifier for native Qwen public/audit captures.

The v2 contract records bytes exposed by the OpenAI completion endpoint and a
separate C02 commit audit. Public responses do not expose token IDs, so token
identity is proven by the create-only audit record; the checker then proves
that public JSON/SSE text is the safe projection of that committed sequence.
In particular, a committed empty piece remains in the audit but is absent from
public SSE delta frames.

This checker only replays evidence already captured below an immutable
external evidence root. It never launches Riley, CUDA, containers, SSH, or a
network request.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn, Sequence

import check_rc3_qualification as qualification


RECEIPT_VERSION = "riley.qwen-multistep-receipt.v2"
CASE_MANIFEST_VERSION = "riley.qwen-multistep-cases.v2"
CHECK_REPORT_VERSION = "riley.qwen-multistep-check.v2"
GOLDEN_VERSION = "riley.qwen-multistep-golden.v2"
GOLDEN_RELATIVE_PATH = "benchmarks/release/candidates/qwen-multistep-golden-v2.json"
GOLDEN_SHA256 = "740aaeb5eb99f9c0d44ba668449e7393ba9c0e00f22fa92b2df3baec8b26cec4"
GOLDEN_ID = "qwen2.5-0.5b-instruct-bf16-c02-audit-v2"
WIRE_VERSION = "riley.qwen-multistep-wire.v2"
WIRE_RELATIVE_PATH = "benchmarks/release/candidates/qwen-multistep-wire-v2.json"
WIRE_SHA256 = "1c2575da45f03608eb10184ca550c112a44bc25889e69656444e481c3a75dab0"
REFERENCE_RELATIVE_PATH = "benchmarks/reference/qwen2.5-0.5b-instruct-bf16.json"
REFERENCE_SHA256 = "f1d2d026404ab1956ff8f0131b7f650da27606eb6a70d132a1a2f0617b8e8a41"
AUDIT_VERSION = "riley.c02-generation-audit.v1"
STABLE_DEFAULT_PROFILE = "stable-default"
GOLDEN_CASE_COUNT = 3
MIN_OUTPUT_TOKENS = 4
MAX_OUTPUT_TOKENS = 4096
MAX_TOKEN_ID = 2**32 - 1
MAX_TEXT_BYTES = 1024 * 1024
MAX_HEADER_BYTES = 64 * 1024

CASE_ID_RE = re.compile(r"^qwen-[a-z0-9][a-z0-9-]{0,63}$")
REFERENCE_CASE_RE = re.compile(r"^[a-z][a-z0-9-]{0,63}$")
SERVER_REQUEST_ID_RE = re.compile(r"^cmpl-[A-Za-z0-9_-]{1,128}$")

CHECK_NAMES = (
    "gate-e-replay",
    "qwen-frozen-model-binding",
    "qwen-wire-prompt-binding",
    "case-manifest-integrity",
    "audit-committed-token-evidence",
    "public-response-audit-binding",
    "streaming-public-parity",
)


class QwenReceiptError(qualification.QualificationError):
    """A Qwen semantic receipt is malformed or does not prove its gate."""


class QwenReceiptIncomparable(qualification.IncomparableError):
    """A Qwen artifact belongs to another candidate, model, or stable arm."""


@dataclass(frozen=True)
class Descriptor:
    path: str
    sha256: str


@dataclass(frozen=True)
class Receipt:
    candidate_id: str
    bindings: dict[str, str]
    model: dict[str, str]
    golden_id: str
    golden: Descriptor
    wire: Descriptor
    case_manifest: Descriptor


@dataclass(frozen=True)
class OutputPiece:
    token_id: int
    text: str


@dataclass(frozen=True)
class GoldenCase:
    case_id: str
    prompt_token_ids: tuple[int, ...]
    max_tokens: int
    sampling: dict[str, Any]
    expected_committed_output_tokens: tuple[OutputPiece, ...]
    finish_reason: str


@dataclass(frozen=True)
class QwenGolden:
    golden_id: str
    descriptor: Descriptor
    model: dict[str, str]
    cases: tuple[GoldenCase, ...]


@dataclass(frozen=True)
class ReferenceCase:
    name: str
    messages: tuple[tuple[str, str], ...]
    rendered_prompt: str
    prompt_token_ids: tuple[int, ...]
    max_tokens: int
    expected_output_token_ids: tuple[int, ...]


@dataclass(frozen=True)
class QwenReference:
    descriptor: Descriptor
    cases: tuple[ReferenceCase, ...]


@dataclass(frozen=True)
class WireCase:
    case_id: str
    reference_case: str
    messages: tuple[tuple[str, str], ...]
    rendered_prompt: str
    prompt_sha256: str


@dataclass(frozen=True)
class QwenWire:
    descriptor: Descriptor
    golden: Descriptor
    reference: Descriptor
    cases: tuple[WireCase, ...]


@dataclass(frozen=True)
class ModeCapture:
    request_body: Descriptor
    response_headers: Descriptor
    response_body: Descriptor
    audit_record: Descriptor


@dataclass(frozen=True)
class AuditRecord:
    server_request_id: str
    committed_output_tokens: tuple[OutputPiece, ...]


@dataclass(frozen=True)
class QwenCheckReport:
    candidate_id: str
    freeze_sha256: str
    base_release_candidate_report: Descriptor
    bindings: dict[str, str]
    golden: Descriptor
    wire: Descriptor
    receipt: Descriptor
    case_manifest: Descriptor
    model: dict[str, str]
    cases: tuple[dict[str, Any], ...]


def _raise(error_type: type[qualification.QualificationError], code: str, message: str) -> NoReturn:
    error = error_type(message)
    error.reason_code = code  # type: ignore[attr-defined]
    raise error


def _fail(code: str, message: str) -> NoReturn:
    _raise(QwenReceiptError, code, message)


def _incomparable(message: str) -> NoReturn:
    _raise(QwenReceiptIncomparable, "incomparable-binding", message)


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
        _fail("invalid-case-id", f"{label} is not a valid Qwen case ID")
    return value


def _reference_case_name(value: Any, label: str) -> str:
    value = qualification._string(value, label)
    if not REFERENCE_CASE_RE.fullmatch(value):
        _fail("invalid-reference-case", f"{label} is not a valid reference case name")
    return value


def _server_request_id(value: Any, label: str) -> str:
    value = qualification._string(value, label)
    if not SERVER_REQUEST_ID_RE.fullmatch(value):
        _fail("invalid-server-request-id", f"{label} is not a bounded completion request ID")
    return value


def _positive_int(value: Any, label: str, *, maximum: int | None = None) -> int:
    if type(value) is not int or value < 0 or (maximum is not None and value > maximum):
        _fail("invalid-integer", f"{label} must be a bounded non-negative integer")
    return value


def _token_ids(value: Any, label: str, *, minimum: int = 1) -> tuple[int, ...]:
    if not isinstance(value, list) or not (minimum <= len(value) <= MAX_OUTPUT_TOKENS):
        _fail("invalid-token-sequence", f"{label} must contain {minimum}..{MAX_OUTPUT_TOKENS} token IDs")
    return tuple(
        _positive_int(token, f"{label}[{index}]", maximum=MAX_TOKEN_ID)
        for index, token in enumerate(value)
    )


def _utf8_text(value: Any, label: str, *, allow_empty: bool) -> str:
    if not isinstance(value, str) or (not allow_empty and not value) or "\x00" in value:
        _fail("invalid-text", f"{label} must be {'possibly empty' if allow_empty else 'non-empty'} UTF-8 text")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError:
        _fail("invalid-text", f"{label} is not valid UTF-8 text")
    if len(encoded) > MAX_TEXT_BYTES:
        _fail("input-too-large", f"{label} exceeds the C02 text bound")
    return value


def _token_ids_sha256(token_ids: Sequence[int]) -> str:
    return hashlib.sha256(qualification.canonical_json_bytes(list(token_ids))).hexdigest()


def _pieces_document(pieces: Sequence[OutputPiece]) -> list[dict[str, Any]]:
    return [{"token_id": piece.token_id, "text": piece.text} for piece in pieces]


def _pieces_sha256(pieces: Sequence[OutputPiece]) -> str:
    return hashlib.sha256(qualification.canonical_json_bytes(_pieces_document(pieces))).hexdigest()


def _generated_text(pieces: Sequence[OutputPiece]) -> str:
    return "".join(piece.text for piece in pieces)


def _prompt_sha256(prompt: str, label: str) -> str:
    try:
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    except UnicodeEncodeError:
        _fail("invalid-rendered-prompt", f"{label} is not valid UTF-8 text")


def _bindings(value: Any, label: str) -> dict[str, str]:
    row = _exact(value, {"freeze_sha256", "base_release_candidate_report_sha256", "configuration_profile", "configuration_sha256"}, label)
    if row["configuration_profile"] != STABLE_DEFAULT_PROFILE:
        _incomparable(f"{label}.configuration_profile is not {STABLE_DEFAULT_PROFILE}")
    return {
        "freeze_sha256": _sha256(row["freeze_sha256"], f"{label}.freeze_sha256"),
        "base_release_candidate_report_sha256": _sha256(row["base_release_candidate_report_sha256"], f"{label}.base_release_candidate_report_sha256"),
        "configuration_profile": STABLE_DEFAULT_PROFILE,
        "configuration_sha256": _sha256(row["configuration_sha256"], f"{label}.configuration_sha256"),
    }


def _model(value: Any, label: str) -> dict[str, str]:
    return qualification._model(value, label)


def _descriptor(value: Any, label: str) -> Descriptor:
    row = _exact(value, {"path", "sha256"}, label)
    return Descriptor(qualification._relative_path(row["path"], f"{label}.path"), _sha256(row["sha256"], f"{label}.sha256"))


def _descriptor_document(descriptor: Descriptor) -> dict[str, str]:
    return {"path": descriptor.path, "sha256": descriptor.sha256}


def _messages(value: Any, label: str) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, list) or not (1 <= len(value) <= 64):
        _fail("invalid-wire-messages", f"{label} must contain 1..64 semantic messages")
    messages: list[tuple[str, str]] = []
    for index, message_value in enumerate(value):
        message = _exact(message_value, {"role", "content"}, f"{label}[{index}]")
        role = qualification._string(message["role"], f"{label}[{index}].role")
        if role not in {"system", "user"}:
            _fail("invalid-wire-message-role", f"{label}[{index}].role is not supported")
        messages.append((role, _utf8_text(message["content"], f"{label}[{index}].content", allow_empty=False)))
    return tuple(messages)


def _validate_sampling(value: Any, label: str) -> None:
    row = _exact(value, {"mode", "temperature", "top_p"}, label)
    if row["mode"] != "greedy" or type(row["temperature"]) is bool or type(row["top_p"]) is bool:
        _fail("unsupported-sampling", f"{label} must be scalar greedy sampling")
    if row["temperature"] != 0 or row["top_p"] != 1:
        _fail("unsupported-sampling", f"{label} must be temperature=0 and top_p=1")


def _validate_pieces(value: Any, label: str) -> tuple[OutputPiece, ...]:
    if not isinstance(value, list) or not (MIN_OUTPUT_TOKENS <= len(value) <= MAX_OUTPUT_TOKENS):
        _fail("invalid-committed-token-sequence", f"{label} must contain a bounded multi-step sequence")
    total_bytes = 0
    pieces: list[OutputPiece] = []
    for index, item in enumerate(value):
        row = _exact(item, {"token_id", "text"}, f"{label}[{index}]")
        token_id = _positive_int(row["token_id"], f"{label}[{index}].token_id", maximum=MAX_TOKEN_ID)
        text = _utf8_text(row["text"], f"{label}[{index}].text", allow_empty=True)
        total_bytes += len(text.encode("utf-8"))
        if total_bytes > MAX_TEXT_BYTES:
            _fail("input-too-large", f"{label} cumulative text exceeds the C02 bound")
        pieces.append(OutputPiece(token_id, text))
    return tuple(pieces)


def _load_reference() -> QwenReference:
    reference_path = Path(__file__).resolve().parents[2] / REFERENCE_RELATIVE_PATH
    raw = qualification._read_regular_path(reference_path, "Qwen source reference")
    if hashlib.sha256(raw).hexdigest() != REFERENCE_SHA256:
        _fail("qwen-reference-drift", "Qwen source reference bytes differ from the reviewed digest")
    document = qualification._parse_document(raw, "Qwen source reference")
    row = _exact(document, {"schema_version", "model", "environment", "cases"}, "Qwen source reference")
    if row["schema_version"] != "riley-qwen2-compat-v1":
        _fail("unsupported-qwen-reference-version", "Qwen source reference schema_version is unsupported")
    if not isinstance(row["model"], dict) or not isinstance(row["environment"], dict):
        _fail("invalid-qwen-reference", "Qwen source reference model/environment must be objects")
    cases_value = row["cases"]
    if not isinstance(cases_value, list) or len(cases_value) != GOLDEN_CASE_COUNT:
        _fail("qwen-reference-case-count", "Qwen source reference must contain exactly three cases")
    cases: list[ReferenceCase] = []
    names: list[str] = []
    for index, value in enumerate(cases_value):
        case = _exact(value, {"name", "messages", "rendered_chat", "prompt_token_ids", "greedy", "raw_last_logits"}, f"Qwen source reference.cases[{index}]")
        if not isinstance(case["raw_last_logits"], dict):
            _fail("invalid-qwen-reference", "Qwen source reference raw logits must be an object")
        greedy = _exact(case["greedy"], {"addressable_token_count", "cache_off_token_ids", "cache_on_token_ids", "max_new_tokens"}, f"Qwen source reference.cases[{index}].greedy")
        if _positive_int(greedy["addressable_token_count"], f"Qwen source reference.cases[{index}].greedy.addressable_token_count", maximum=MAX_TOKEN_ID) < 1:
            _fail("invalid-qwen-reference", "Qwen source reference token count must be positive")
        expected_ids = _token_ids(greedy["cache_off_token_ids"], f"Qwen source reference.cases[{index}].greedy.cache_off_token_ids", minimum=MIN_OUTPUT_TOKENS)
        if _token_ids(greedy["cache_on_token_ids"], f"Qwen source reference.cases[{index}].greedy.cache_on_token_ids", minimum=MIN_OUTPUT_TOKENS) != expected_ids:
            _fail("qwen-reference-cache-mismatch", "Qwen source reference cache-on/off tokens differ")
        max_tokens = _positive_int(greedy["max_new_tokens"], f"Qwen source reference.cases[{index}].greedy.max_new_tokens", maximum=MAX_OUTPUT_TOKENS)
        if max_tokens < MIN_OUTPUT_TOKENS or len(expected_ids) != max_tokens:
            _fail("qwen-reference-token-bound-mismatch", "Qwen source reference is not a full multi-step case")
        name = _reference_case_name(case["name"], f"Qwen source reference.cases[{index}].name")
        names.append(name)
        rendered_prompt = _utf8_text(case["rendered_chat"], f"Qwen source reference.cases[{index}].rendered_chat", allow_empty=False)
        _prompt_sha256(rendered_prompt, f"Qwen source reference.cases[{index}].rendered_chat")
        cases.append(ReferenceCase(name, _messages(case["messages"], f"Qwen source reference.cases[{index}].messages"), rendered_prompt, _token_ids(case["prompt_token_ids"], f"Qwen source reference.cases[{index}].prompt_token_ids"), max_tokens, expected_ids))
    if len(set(names)) != len(names):
        _fail("duplicate-reference-case", "Qwen source reference cases must be unique")
    return QwenReference(Descriptor(REFERENCE_RELATIVE_PATH, REFERENCE_SHA256), tuple(cases))


def _load_golden() -> QwenGolden:
    golden_path = Path(__file__).resolve().parents[2] / GOLDEN_RELATIVE_PATH
    raw = qualification._read_regular_path(golden_path, "Qwen multi-step golden")
    if hashlib.sha256(raw).hexdigest() != GOLDEN_SHA256:
        _fail("qwen-golden-drift", "Qwen multi-step golden bytes differ from the reviewed digest")
    document = qualification._parse_document(raw, "Qwen multi-step golden")
    if raw != qualification.canonical_json_bytes(document) + b"\n":
        _fail("noncanonical-qwen-golden", "Qwen multi-step golden must be canonical JSON plus one newline")
    row = _exact(document, {"schema_version", "golden_id", "model", "cases"}, "Qwen multi-step golden")
    if row["schema_version"] != GOLDEN_VERSION:
        _fail("unsupported-qwen-golden-version", "Qwen multi-step golden schema_version is unsupported")
    golden_id = qualification._string(row["golden_id"], "Qwen multi-step golden.golden_id")
    if golden_id != GOLDEN_ID:
        _fail("qwen-golden-identity-mismatch", "Qwen multi-step golden ID is not the reviewed v2 anchor")
    model = _model(row["model"], "Qwen multi-step golden.model")
    cases_value = row["cases"]
    if not isinstance(cases_value, list) or len(cases_value) != GOLDEN_CASE_COUNT:
        _fail("qwen-golden-case-count", "Qwen multi-step golden must have exactly three cases")
    cases: list[GoldenCase] = []
    case_ids: list[str] = []
    for index, value in enumerate(cases_value):
        case = _exact(value, {"case_id", "prompt_token_ids", "max_tokens", "sampling", "expected_committed_output_tokens", "finish_reason"}, f"Qwen multi-step golden.cases[{index}]")
        case_id = _case_id(case["case_id"], f"Qwen multi-step golden.cases[{index}].case_id")
        prompt_token_ids = _token_ids(case["prompt_token_ids"], f"Qwen multi-step golden.cases[{index}].prompt_token_ids")
        max_tokens = _positive_int(case["max_tokens"], f"Qwen multi-step golden.cases[{index}].max_tokens", maximum=MAX_OUTPUT_TOKENS)
        if max_tokens < MIN_OUTPUT_TOKENS:
            _fail("qwen-golden-insufficient-steps", "Qwen golden must exercise at least four generation steps")
        _validate_sampling(case["sampling"], f"Qwen multi-step golden.cases[{index}].sampling")
        pieces = _validate_pieces(case["expected_committed_output_tokens"], f"Qwen multi-step golden.cases[{index}].expected_committed_output_tokens")
        if len(pieces) != max_tokens or case["finish_reason"] != "length":
            _fail("qwen-golden-terminal-mismatch", "Qwen golden must bind a full length-limited response")
        case_ids.append(case_id)
        cases.append(GoldenCase(case_id, prompt_token_ids, max_tokens, dict(case["sampling"]), pieces, "length"))
    if case_ids != sorted(case_ids) or len(set(case_ids)) != len(case_ids):
        _fail("qwen-golden-case-order", "Qwen multi-step golden cases must be unique and ordered")
    return QwenGolden(golden_id, Descriptor(GOLDEN_RELATIVE_PATH, GOLDEN_SHA256), model, tuple(cases))


def _load_wire(golden: QwenGolden) -> QwenWire:
    reference = _load_reference()
    wire_path = Path(__file__).resolve().parents[2] / WIRE_RELATIVE_PATH
    raw = qualification._read_regular_path(wire_path, "Qwen multi-step wire corpus")
    if hashlib.sha256(raw).hexdigest() != WIRE_SHA256:
        _fail("qwen-wire-drift", "Qwen wire corpus bytes differ from the reviewed digest")
    document = qualification._parse_document(raw, "Qwen multi-step wire corpus")
    if raw != qualification.canonical_json_bytes(document) + b"\n":
        _fail("noncanonical-qwen-wire", "Qwen wire corpus must be canonical JSON plus one newline")
    row = _exact(document, {"schema_version", "golden", "reference", "cases"}, "Qwen multi-step wire corpus")
    if row["schema_version"] != WIRE_VERSION:
        _fail("unsupported-qwen-wire-version", "Qwen wire corpus schema_version is unsupported")
    if _descriptor(row["golden"], "Qwen multi-step wire corpus.golden") != golden.descriptor:
        _fail("qwen-wire-golden-binding-mismatch", "Qwen wire corpus does not bind reviewed v2 golden")
    if _descriptor(row["reference"], "Qwen multi-step wire corpus.reference") != reference.descriptor:
        _fail("qwen-wire-reference-binding-mismatch", "Qwen wire corpus does not bind reviewed source reference")
    cases_value = row["cases"]
    if not isinstance(cases_value, list) or len(cases_value) != GOLDEN_CASE_COUNT:
        _fail("qwen-wire-case-count", "Qwen wire corpus must contain exactly three cases")
    reference_cases = {case.name: case for case in reference.cases}
    cases: list[WireCase] = []
    reference_names: list[str] = []
    for index, value in enumerate(cases_value):
        case = _exact(value, {"case_id", "reference_case", "messages", "rendered_prompt", "prompt_sha256"}, f"Qwen multi-step wire corpus.cases[{index}]")
        case_id = _case_id(case["case_id"], f"Qwen multi-step wire corpus.cases[{index}].case_id")
        reference_name = _reference_case_name(case["reference_case"], f"Qwen multi-step wire corpus.cases[{index}].reference_case")
        reference_names.append(reference_name)
        reference_case = reference_cases.get(reference_name)
        if reference_case is None:
            _fail("qwen-wire-reference-case-mismatch", "Qwen wire corpus names an unknown source case")
        messages = _messages(case["messages"], f"Qwen multi-step wire corpus.cases[{index}].messages")
        rendered_prompt = _utf8_text(case["rendered_prompt"], f"Qwen multi-step wire corpus.cases[{index}].rendered_prompt", allow_empty=False)
        prompt_sha256 = _sha256(case["prompt_sha256"], f"Qwen multi-step wire corpus.cases[{index}].prompt_sha256")
        if prompt_sha256 != _prompt_sha256(rendered_prompt, f"Qwen wire case {case_id}"):
            _fail("qwen-wire-prompt-hash-mismatch", "Qwen wire prompt digest does not bind literal prompt bytes")
        golden_case = golden.cases[index]
        if case_id != golden_case.case_id:
            _fail("qwen-wire-golden-case-mismatch", "Qwen wire case identity drifts from reviewed golden")
        if (messages != reference_case.messages or rendered_prompt != reference_case.rendered_prompt or reference_case.prompt_token_ids != golden_case.prompt_token_ids or reference_case.max_tokens != golden_case.max_tokens or reference_case.expected_output_token_ids != tuple(piece.token_id for piece in golden_case.expected_committed_output_tokens)):
            _fail("qwen-wire-reference-mismatch", "Qwen wire corpus drifts from source reference or golden")
        cases.append(WireCase(case_id, reference_name, messages, rendered_prompt, prompt_sha256))
    if [case.case_id for case in cases] != [case.case_id for case in golden.cases]:
        _fail("qwen-wire-case-order", "Qwen wire cases must preserve reviewed golden order")
    if len(set(reference_names)) != len(reference_names) or set(reference_names) != set(reference_cases):
        _fail("qwen-wire-reference-inventory", "Qwen wire corpus must map each source case once")
    return QwenWire(Descriptor(WIRE_RELATIVE_PATH, WIRE_SHA256), golden.descriptor, reference.descriptor, tuple(cases))


def _canonical_document(raw: bytes, label: str) -> dict[str, Any]:
    document = qualification._parse_document(raw, label)
    if raw != qualification.canonical_json_bytes(document):
        _fail("noncanonical-evidence", f"{label} must be exact canonical JSON bytes")
    return document


def _read_relative_json(evidence_root: Path, relative_path: Path | str, label: str) -> tuple[str, bytes, dict[str, Any]]:
    relative = qualification._relative_path(str(relative_path), f"{label}.path")
    raw = qualification._read_relative(evidence_root, relative, label)
    return relative, raw, _canonical_document(raw, label)


def _read_described_bytes(evidence_root: Path, descriptor: Descriptor, label: str, used_paths: set[str]) -> bytes:
    if descriptor.path in used_paths:
        _fail("duplicate-evidence-path", f"{label} reuses evidence path {descriptor.path!r}")
    used_paths.add(descriptor.path)
    raw = qualification._read_relative(evidence_root, descriptor.path, label)
    if hashlib.sha256(raw).hexdigest() != descriptor.sha256:
        _fail("evidence-hash-mismatch", f"{label} digest mismatch")
    return raw


def _read_described_json(evidence_root: Path, descriptor: Descriptor, label: str, used_paths: set[str], *, require_canonical: bool) -> tuple[bytes, dict[str, Any]]:
    raw = _read_described_bytes(evidence_root, descriptor, label, used_paths)
    document = qualification._parse_document(raw, label)
    if require_canonical and raw != qualification.canonical_json_bytes(document):
        _fail("noncanonical-evidence", f"{label} must be exact canonical JSON bytes")
    return raw, document


def _reject_reserved_output_path(descriptor: Descriptor, reserved_paths: set[str], label: str) -> None:
    if descriptor.path in reserved_paths:
        _fail("reserved-output-path-collision", f"{label} reuses a freeze-declared final report or semantic receipt path")


def _validate_bound_header(*, candidate_id: str, bindings: dict[str, str], model: dict[str, str], frozen: qualification.FrozenCandidate, freeze_sha256: str, base_report_sha256: str, label: str) -> None:
    if candidate_id != frozen.candidate_id:
        _incomparable(f"{label} belongs to another candidate")
    expected_bindings = {"freeze_sha256": freeze_sha256, "base_release_candidate_report_sha256": base_report_sha256, "configuration_profile": STABLE_DEFAULT_PROFILE, "configuration_sha256": frozen.arms["stable_default"]["configuration_sha256"]}
    if bindings != expected_bindings:
        _incomparable(f"{label} immutable bindings drifted from frozen stable-default")
    if model != frozen.models["qwen"]:
        _incomparable(f"{label} model identity drifted from frozen Qwen")


def _validate_frozen_qwen_against_golden(frozen: qualification.FrozenCandidate, golden: QwenGolden) -> None:
    if frozen.models["qwen"] != golden.model:
        _incomparable("frozen Qwen identity differs from the reviewed multi-step golden")


def _validate_golden_binding(*, golden_id: str, descriptor: Descriptor, golden: QwenGolden, label: str) -> None:
    if golden_id != golden.golden_id or descriptor != golden.descriptor:
        _fail("qwen-golden-binding-mismatch", f"{label} does not bind reviewed Qwen v2 golden")


def _validate_wire_binding(*, descriptor: Descriptor, wire: QwenWire, label: str) -> None:
    if descriptor != wire.descriptor:
        _fail("qwen-wire-binding-mismatch", f"{label} does not bind reviewed Qwen v2 wire corpus")


def _validate_receipt(document: dict[str, Any], label: str) -> Receipt:
    row = _exact(document, {"schema_version", "status", "passed", "candidate_id", "bindings", "model", "golden_id", "golden", "wire", "case_manifest"}, label)
    if row["schema_version"] != RECEIPT_VERSION:
        _fail("unsupported-qwen-receipt-version", f"{label}.schema_version is unsupported")
    if row["status"] != "passed" or row["passed"] is not True:
        _fail("qwen-receipt-not-passed", f"{label} must be passed")
    return Receipt(_candidate_id(row["candidate_id"], f"{label}.candidate_id"), _bindings(row["bindings"], f"{label}.bindings"), _model(row["model"], f"{label}.model"), qualification._string(row["golden_id"], f"{label}.golden_id"), _descriptor(row["golden"], f"{label}.golden"), _descriptor(row["wire"], f"{label}.wire"), _descriptor(row["case_manifest"], f"{label}.case_manifest"))


def _validate_case_manifest(document: dict[str, Any], label: str) -> tuple[str, dict[str, str], dict[str, str], Descriptor, Descriptor, list[dict[str, Any]]]:
    row = _exact(document, {"schema_version", "candidate_id", "bindings", "model", "golden", "wire", "cases"}, label)
    if row["schema_version"] != CASE_MANIFEST_VERSION:
        _fail("unsupported-qwen-case-manifest-version", f"{label}.schema_version is unsupported")
    cases = row["cases"]
    if not isinstance(cases, list) or len(cases) != GOLDEN_CASE_COUNT or not all(isinstance(case, dict) for case in cases):
        _fail("invalid-case-manifest", f"{label}.cases must contain exactly three objects")
    return (_candidate_id(row["candidate_id"], f"{label}.candidate_id"), _bindings(row["bindings"], f"{label}.bindings"), _model(row["model"], f"{label}.model"), _descriptor(row["golden"], f"{label}.golden"), _descriptor(row["wire"], f"{label}.wire"), cases)


def _validate_capture(value: Any, label: str) -> ModeCapture:
    row = _exact(value, {"request_body", "response_headers", "response_body", "audit_record"}, label)
    return ModeCapture(_descriptor(row["request_body"], f"{label}.request_body"), _descriptor(row["response_headers"], f"{label}.response_headers"), _descriptor(row["response_body"], f"{label}.response_body"), _descriptor(row["audit_record"], f"{label}.audit_record"))


def _validate_case(value: dict[str, Any], label: str) -> tuple[str, ModeCapture, ModeCapture]:
    row = _exact(value, {"case_id", "non_stream", "stream"}, label)
    return (_case_id(row["case_id"], f"{label}.case_id"), _validate_capture(row["non_stream"], f"{label}.non_stream"), _validate_capture(row["stream"], f"{label}.stream"))


def _validate_request_body(raw: bytes, *, golden_case: GoldenCase, wire_case: WireCase, expected_stream: bool, model_id: str, label: str) -> None:
    document = qualification._parse_document(raw, label)
    row = _exact(document, {"model", "prompt", "max_tokens", "temperature", "top_p", "stream"}, label)
    if row["model"] != model_id:
        _fail("public-request-model-mismatch", f"{label}.model differs from frozen Qwen")
    if _utf8_text(row["prompt"], f"{label}.prompt", allow_empty=False) != wire_case.rendered_prompt:
        _fail("public-request-prompt-mismatch", f"{label}.prompt differs from reviewed literal Qwen prompt")
    if _prompt_sha256(row["prompt"], f"{label}.prompt") != wire_case.prompt_sha256:
        _fail("public-request-prompt-hash-mismatch", f"{label}.prompt digest differs from reviewed wire")
    if type(row["max_tokens"]) is not int or row["max_tokens"] != golden_case.max_tokens:
        _fail("public-request-max-tokens-mismatch", f"{label}.max_tokens differs from reviewed case")
    if type(row["temperature"]) is not int or row["temperature"] != 0 or type(row["top_p"]) is not int or row["top_p"] != 1:
        _fail("public-request-sampling-mismatch", f"{label} must use literal temperature=0 and top_p=1")
    if type(row["stream"]) is not bool or row["stream"] is not expected_stream:
        _fail("public-request-stream-mismatch", f"{label}.stream has the wrong mode")


def _validate_headers(raw: bytes, *, stream: bool, response_body: bytes, label: str) -> None:
    if not raw or len(raw) > MAX_HEADER_BYTES:
        _fail("invalid-http-headers", f"{label} has an invalid byte length")
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError:
        _fail("invalid-http-headers", f"{label} must be ASCII")
    if not text.endswith("\r\n\r\n") or "\n" in text.replace("\r\n", ""):
        _fail("invalid-http-headers", f"{label} must use only CRLF header lines")
    lines = text[:-4].split("\r\n")
    if not lines or lines[0] != "HTTP/1.1 200 OK":
        _fail("invalid-http-status", f"{label} must record HTTP/1.1 200 OK")
    expected = (["Content-Type: text/event-stream; charset=utf-8", "Cache-Control: no-cache", "Connection: close", "X-Content-Type-Options: nosniff"] if stream else ["Content-Type: application/json; charset=utf-8", f"Content-Length: {len(response_body)}", "Connection: close", "X-Content-Type-Options: nosniff"])
    if lines[1:] != expected:
        _fail("http-header-contract-mismatch", f"{label} differs from fixed public response header contract")


def _validate_usage(value: Any, *, prompt_tokens: int, completion_tokens: int, label: str) -> None:
    row = _exact(value, {"prompt_tokens", "completion_tokens", "total_tokens"}, label)
    actual_prompt = _positive_int(row["prompt_tokens"], f"{label}.prompt_tokens")
    actual_completion = _positive_int(row["completion_tokens"], f"{label}.completion_tokens")
    actual_total = _positive_int(row["total_tokens"], f"{label}.total_tokens")
    if actual_prompt != prompt_tokens or actual_completion != completion_tokens or actual_total != prompt_tokens + completion_tokens:
        _fail("usage-token-mismatch", f"{label} does not match reviewed prompt/output token counts")


def _validate_audit_record(document: dict[str, Any], *, golden_case: GoldenCase, delivery_mode: str, label: str) -> AuditRecord:
    row = _exact(document, {"schema_version", "committed_output_tokens", "delivery_mode", "finish_reason", "prompt_token_ids", "server_request_id", "usage"}, label)
    if row["schema_version"] != AUDIT_VERSION:
        _fail("unsupported-audit-version", f"{label}.schema_version is unsupported")
    if row["delivery_mode"] != delivery_mode:
        _fail("audit-delivery-mode-mismatch", f"{label}.delivery_mode does not match capture arm")
    if row["finish_reason"] != golden_case.finish_reason:
        _fail("audit-finish-reason-mismatch", f"{label}.finish_reason differs from reviewed case")
    if _token_ids(row["prompt_token_ids"], f"{label}.prompt_token_ids") != golden_case.prompt_token_ids:
        _fail("audit-prompt-token-mismatch", f"{label}.prompt_token_ids differ from reviewed case")
    pieces = _validate_pieces(row["committed_output_tokens"], f"{label}.committed_output_tokens")
    if pieces != golden_case.expected_committed_output_tokens:
        _fail("audit-committed-token-mismatch", f"{label}.committed_output_tokens differ from reviewed IDs/text")
    _validate_usage(row["usage"], prompt_tokens=len(golden_case.prompt_token_ids), completion_tokens=len(pieces), label=f"{label}.usage")
    return AuditRecord(_server_request_id(row["server_request_id"], f"{label}.server_request_id"), pieces)


def _validate_public_choice(value: Any, *, expected_text: str, expected_finish_reason: str | None, label: str) -> None:
    row = _exact(value, {"text", "index", "logprobs", "finish_reason"}, label)
    if _utf8_text(row["text"], f"{label}.text", allow_empty=True) != expected_text:
        _fail("public-text-mismatch", f"{label}.text differs from audited committed text")
    if type(row["index"]) is not int or row["index"] != 0 or row["logprobs"] is not None:
        _fail("invalid-public-choice", f"{label} must be choice index zero with null logprobs")
    if row["finish_reason"] != expected_finish_reason:
        _fail("public-finish-reason-mismatch", f"{label}.finish_reason is invalid")


def _validate_non_stream_response(raw: bytes, *, audit: AuditRecord, golden_case: GoldenCase, model_id: str, label: str) -> str:
    document = qualification._parse_document(raw, label)
    row = _exact(document, {"id", "object", "created", "model", "choices", "usage"}, label)
    if _server_request_id(row["id"], f"{label}.id") != audit.server_request_id:
        _fail("non-stream-audit-request-id-mismatch", f"{label}.id does not bind non-stream audit")
    if row["object"] != "text_completion" or row["model"] != model_id:
        _fail("invalid-public-response", f"{label} has unexpected object or model")
    _positive_int(row["created"], f"{label}.created")
    choices = row["choices"]
    if not isinstance(choices, list) or len(choices) != 1:
        _fail("invalid-public-response", f"{label}.choices must contain exactly one choice")
    expected_text = _generated_text(audit.committed_output_tokens)
    _validate_public_choice(choices[0], expected_text=expected_text, expected_finish_reason=golden_case.finish_reason, label=f"{label}.choices[0]")
    _validate_usage(row["usage"], prompt_tokens=len(golden_case.prompt_token_ids), completion_tokens=len(audit.committed_output_tokens), label=f"{label}.usage")
    return expected_text


def _parse_sse_completion_frame(raw: bytes, label: str) -> dict[str, Any]:
    if not raw.startswith(b"data: ") or b"\n" in raw or b"\r" in raw:
        _fail("invalid-sse-frame", f"{label} must contain one data: JSON line")
    return qualification._parse_document(raw[6:], label)


def _validate_sse(raw: bytes, *, audit: AuditRecord, golden_case: GoldenCase, model_id: str, label: str) -> str:
    if not raw or b"\r" in raw or not raw.endswith(b"\n\n"):
        _fail("invalid-sse-transcript", f"{label} must end with LF-delimited SSE frames")
    frames = raw[:-2].split(b"\n\n")
    if not frames or any(not frame for frame in frames):
        _fail("invalid-sse-transcript", f"{label} contains an empty SSE frame")
    visible_texts = [piece.text for piece in audit.committed_output_tokens if piece.text]
    if len(frames) != len(visible_texts) + 2:
        _fail("stream-event-count-mismatch", f"{label} must expose non-empty deltas, terminal, then [DONE]")
    created: int | None = None
    for index, expected_text in enumerate(visible_texts):
        document = _parse_sse_completion_frame(frames[index], f"{label}.frames[{index}]")
        row = _exact(document, {"id", "object", "created", "model", "choices"}, f"{label}.frames[{index}]")
        if _server_request_id(row["id"], f"{label}.frames[{index}].id") != audit.server_request_id:
            _fail("stream-audit-request-id-mismatch", f"{label}.frames[{index}] does not bind stream audit")
        if row["object"] != "text_completion" or row["model"] != model_id:
            _fail("invalid-sse-frame", f"{label}.frames[{index}] has unexpected object or model")
        frame_created = _positive_int(row["created"], f"{label}.frames[{index}].created")
        if created is None:
            created = frame_created
        elif created != frame_created:
            _fail("stream-created-mismatch", f"{label} changes created time within one stream")
        choices = row["choices"]
        if not isinstance(choices, list) or len(choices) != 1:
            _fail("invalid-sse-frame", f"{label}.frames[{index}].choices must have exactly one item")
        _validate_public_choice(choices[0], expected_text=expected_text, expected_finish_reason=None, label=f"{label}.frames[{index}].choices[0]")
    terminal_index = len(visible_texts)
    terminal = _parse_sse_completion_frame(frames[terminal_index], f"{label}.frames[{terminal_index}]")
    terminal_row = _exact(terminal, {"id", "object", "created", "model", "choices"}, f"{label}.frames[{terminal_index}]")
    if _server_request_id(terminal_row["id"], f"{label}.terminal.id") != audit.server_request_id:
        _fail("stream-audit-request-id-mismatch", f"{label}.terminal does not bind stream audit")
    if terminal_row["object"] != "text_completion" or terminal_row["model"] != model_id:
        _fail("invalid-sse-terminal", f"{label}.terminal has unexpected object or model")
    terminal_created = _positive_int(terminal_row["created"], f"{label}.terminal.created")
    if created is None or terminal_created != created:
        _fail("stream-created-mismatch", f"{label}.terminal created time differs from token deltas")
    terminal_choices = terminal_row["choices"]
    if not isinstance(terminal_choices, list) or len(terminal_choices) != 1:
        _fail("invalid-sse-terminal", f"{label}.terminal.choices must have exactly one item")
    _validate_public_choice(terminal_choices[0], expected_text="", expected_finish_reason=golden_case.finish_reason, label=f"{label}.terminal.choices[0]")
    if frames[-1] != b"data: [DONE]":
        _fail("stream-done-mismatch", f"{label} must end with exactly one data: [DONE] frame")
    public_text = "".join(visible_texts)
    if public_text != _generated_text(audit.committed_output_tokens):
        _fail("stream-text-mismatch", f"{label} public text does not equal concatenated audit text")
    return public_text


def _capture_descriptors(capture: ModeCapture) -> tuple[Descriptor, ...]:
    return (capture.request_body, capture.response_headers, capture.response_body, capture.audit_record)


def _capture_document(capture: ModeCapture) -> dict[str, dict[str, str]]:
    return {"request_body": _descriptor_document(capture.request_body), "response_headers": _descriptor_document(capture.response_headers), "response_body": _descriptor_document(capture.response_body), "audit_record": _descriptor_document(capture.audit_record)}


def _validate_case_evidence(*, golden_case: GoldenCase, wire_case: WireCase, non_stream: ModeCapture, stream: ModeCapture, evidence_root: Path, used_paths: set[str], reserved_paths: set[str], model_id: str) -> tuple[dict[str, Any], tuple[int, ...]]:
    case_id = golden_case.case_id
    for mode_label, capture in (("non_stream", non_stream), ("stream", stream)):
        for descriptor in _capture_descriptors(capture):
            _reject_reserved_output_path(descriptor, reserved_paths, f"case.{case_id}.{mode_label}")
    non_request_raw = _read_described_bytes(evidence_root, non_stream.request_body, f"case.{case_id}.non_stream.request_body", used_paths)
    non_headers_raw = _read_described_bytes(evidence_root, non_stream.response_headers, f"case.{case_id}.non_stream.response_headers", used_paths)
    non_response_raw = _read_described_bytes(evidence_root, non_stream.response_body, f"case.{case_id}.non_stream.response_body", used_paths)
    _, non_audit_document = _read_described_json(evidence_root, non_stream.audit_record, f"case.{case_id}.non_stream.audit_record", used_paths, require_canonical=False)
    stream_request_raw = _read_described_bytes(evidence_root, stream.request_body, f"case.{case_id}.stream.request_body", used_paths)
    stream_headers_raw = _read_described_bytes(evidence_root, stream.response_headers, f"case.{case_id}.stream.response_headers", used_paths)
    stream_response_raw = _read_described_bytes(evidence_root, stream.response_body, f"case.{case_id}.stream.response_body", used_paths)
    _, stream_audit_document = _read_described_json(evidence_root, stream.audit_record, f"case.{case_id}.stream.audit_record", used_paths, require_canonical=False)
    _validate_request_body(non_request_raw, golden_case=golden_case, wire_case=wire_case, expected_stream=False, model_id=model_id, label=f"case.{case_id}.non_stream.request_body")
    _validate_request_body(stream_request_raw, golden_case=golden_case, wire_case=wire_case, expected_stream=True, model_id=model_id, label=f"case.{case_id}.stream.request_body")
    non_audit = _validate_audit_record(non_audit_document, golden_case=golden_case, delivery_mode="non-stream", label=f"case.{case_id}.non_stream.audit_record")
    stream_audit = _validate_audit_record(stream_audit_document, golden_case=golden_case, delivery_mode="stream", label=f"case.{case_id}.stream.audit_record")
    if non_audit.server_request_id == stream_audit.server_request_id:
        _fail("cross-mode-request-id-collision", f"case.{case_id} must bind two distinct server request IDs")
    _validate_headers(non_headers_raw, stream=False, response_body=non_response_raw, label=f"case.{case_id}.non_stream.response_headers")
    _validate_headers(stream_headers_raw, stream=True, response_body=stream_response_raw, label=f"case.{case_id}.stream.response_headers")
    non_text = _validate_non_stream_response(non_response_raw, audit=non_audit, golden_case=golden_case, model_id=model_id, label=f"case.{case_id}.non_stream.response_body")
    stream_text = _validate_sse(stream_response_raw, audit=stream_audit, golden_case=golden_case, model_id=model_id, label=f"case.{case_id}.stream.response_body")
    expected_text = _generated_text(golden_case.expected_committed_output_tokens)
    if non_text != expected_text or stream_text != expected_text:
        _fail("public-generated-text-mismatch", f"case.{case_id} public text differs from reviewed detokenization")
    expected_ids = tuple(piece.token_id for piece in golden_case.expected_committed_output_tokens)
    return ({"case_id": case_id, "prompt_token_ids_sha256": _token_ids_sha256(golden_case.prompt_token_ids), "prompt_sha256": wire_case.prompt_sha256, "output_token_ids_sha256": _token_ids_sha256(expected_ids), "output_token_pieces_sha256": _pieces_sha256(golden_case.expected_committed_output_tokens), "generated_text_sha256": hashlib.sha256(expected_text.encode("utf-8")).hexdigest(), "finish_reason": golden_case.finish_reason, "non_stream_request_id": non_audit.server_request_id, "stream_request_id": stream_audit.server_request_id, "non_stream": _capture_document(non_stream), "stream": _capture_document(stream)}, golden_case.prompt_token_ids)


def _empty_report() -> dict[str, Any]:
    return {"schema_version": CHECK_REPORT_VERSION, "status": "failed", "passed": False, "candidate_id": None, "freeze_sha256": None, "base_release_candidate_report": None, "bindings": None, "golden": None, "wire": None, "receipt": None, "case_manifest": None, "model": None, "cases": [], "checks": [], "reason_codes": []}


def _validate_report_case(value: Any, *, golden_case: GoldenCase, wire_case: WireCase, label: str) -> dict[str, Any]:
    row = _exact(value, {"case_id", "prompt_token_ids_sha256", "prompt_sha256", "output_token_ids_sha256", "output_token_pieces_sha256", "generated_text_sha256", "finish_reason", "non_stream_request_id", "stream_request_id", "non_stream", "stream"}, label)
    if _case_id(row["case_id"], f"{label}.case_id") != golden_case.case_id:
        _fail("invalid-qwen-check-report", f"{label}.case_id drifts from reviewed golden")
    expected_ids = tuple(piece.token_id for piece in golden_case.expected_committed_output_tokens)
    expected_text = _generated_text(golden_case.expected_committed_output_tokens)
    if _sha256(row["prompt_token_ids_sha256"], f"{label}.prompt_token_ids_sha256") != _token_ids_sha256(golden_case.prompt_token_ids) or _sha256(row["prompt_sha256"], f"{label}.prompt_sha256") != wire_case.prompt_sha256 or _sha256(row["output_token_ids_sha256"], f"{label}.output_token_ids_sha256") != _token_ids_sha256(expected_ids) or _sha256(row["output_token_pieces_sha256"], f"{label}.output_token_pieces_sha256") != _pieces_sha256(golden_case.expected_committed_output_tokens) or _sha256(row["generated_text_sha256"], f"{label}.generated_text_sha256") != hashlib.sha256(expected_text.encode("utf-8")).hexdigest() or row["finish_reason"] != golden_case.finish_reason:
        _fail("invalid-qwen-check-report", f"{label} drifts from reviewed static Qwen contract")
    non_stream_id = _server_request_id(row["non_stream_request_id"], f"{label}.non_stream_request_id")
    stream_id = _server_request_id(row["stream_request_id"], f"{label}.stream_request_id")
    if non_stream_id == stream_id:
        _fail("invalid-qwen-check-report", f"{label} reuses one server ID for both modes")
    non_stream = _validate_capture(row["non_stream"], f"{label}.non_stream")
    stream = _validate_capture(row["stream"], f"{label}.stream")
    return {"case_id": golden_case.case_id, "prompt_token_ids_sha256": _token_ids_sha256(golden_case.prompt_token_ids), "prompt_sha256": wire_case.prompt_sha256, "output_token_ids_sha256": _token_ids_sha256(expected_ids), "output_token_pieces_sha256": _pieces_sha256(golden_case.expected_committed_output_tokens), "generated_text_sha256": hashlib.sha256(expected_text.encode("utf-8")).hexdigest(), "finish_reason": golden_case.finish_reason, "non_stream_request_id": non_stream_id, "stream_request_id": stream_id, "non_stream": _capture_document(non_stream), "stream": _capture_document(stream)}


def validate_check_report(document: dict[str, Any]) -> QwenCheckReport:
    row = _exact(document, {"schema_version", "status", "passed", "candidate_id", "freeze_sha256", "base_release_candidate_report", "bindings", "golden", "wire", "receipt", "case_manifest", "model", "cases", "checks", "reason_codes"}, "qwen multistep check report")
    if row["schema_version"] != CHECK_REPORT_VERSION:
        _fail("unsupported-qwen-check-report-version", "Qwen check report schema_version is unsupported")
    if row["status"] != "passed" or row["passed"] is not True or row["reason_codes"] != []:
        _fail("qwen-check-not-passed", "Qwen check report must be a clean passed result")
    golden = _load_golden()
    wire = _load_wire(golden)
    if _descriptor(row["golden"], "qwen check report.golden") != golden.descriptor:
        _fail("qwen-golden-binding-mismatch", "Qwen check report does not bind reviewed v2 golden")
    _validate_wire_binding(descriptor=_descriptor(row["wire"], "qwen check report.wire"), wire=wire, label="qwen check report")
    if _model(row["model"], "qwen check report.model") != golden.model:
        _fail("qwen-golden-binding-mismatch", "Qwen check report model differs from reviewed golden")
    checks = row["checks"]
    if not isinstance(checks, list) or len(checks) != len(CHECK_NAMES):
        _fail("invalid-qwen-check-report", "Qwen check report has an invalid check inventory")
    names: list[str] = []
    for index, check in enumerate(checks):
        item = _exact(check, {"name", "passed"}, f"qwen check report.checks[{index}]")
        if item["passed"] is not True:
            _fail("qwen-check-not-passed", f"Qwen check {item['name']!r} did not pass")
        names.append(qualification._string(item["name"], f"qwen check report.checks[{index}].name"))
    if tuple(names) != CHECK_NAMES:
        _fail("invalid-qwen-check-report", "Qwen check report check inventory drifted")
    cases = row["cases"]
    if not isinstance(cases, list) or len(cases) != GOLDEN_CASE_COUNT:
        _fail("invalid-qwen-check-report", "Qwen check report has an invalid case count")
    receipt_descriptor = _descriptor(row["receipt"], "qwen check report.receipt")
    case_manifest_descriptor = _descriptor(row["case_manifest"], "qwen check report.case_manifest")
    used_paths: set[str] = set()
    for descriptor, name in ((receipt_descriptor, "receipt"), (case_manifest_descriptor, "case manifest")):
        if descriptor.path in used_paths:
            _fail("duplicate-evidence-path", f"qwen check report reuses {name} path")
        used_paths.add(descriptor.path)
    normalized_cases: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        normalized = _validate_report_case(case, golden_case=golden.cases[index], wire_case=wire.cases[index], label=f"qwen check report.cases[{index}]")
        for capture_name in ("non_stream", "stream"):
            capture = normalized[capture_name]
            assert isinstance(capture, dict)
            for role in ("request_body", "response_headers", "response_body", "audit_record"):
                descriptor = _descriptor(capture[role], f"qwen check report.cases[{index}].{capture_name}.{role}")
                if descriptor.path in used_paths:
                    _fail("duplicate-evidence-path", "qwen check report aliases a raw evidence descriptor")
                used_paths.add(descriptor.path)
        normalized_cases.append(normalized)
    return QwenCheckReport(_candidate_id(row["candidate_id"], "qwen check report.candidate_id"), _sha256(row["freeze_sha256"], "qwen check report.freeze_sha256"), _descriptor(row["base_release_candidate_report"], "qwen check report.base_release_candidate_report"), _bindings(row["bindings"], "qwen check report.bindings"), golden.descriptor, wire.descriptor, receipt_descriptor, case_manifest_descriptor, _model(row["model"], "qwen check report.model"), tuple(normalized_cases))


def evaluate(freeze_path: Path, evidence_root: Path, receipt_path: Path | str, *, expected_freeze_sha256: str) -> dict[str, Any]:
    report = _empty_report()
    try:
        golden = _load_golden()
        wire = _load_wire(golden)
        report["golden"] = _descriptor_document(golden.descriptor)
        report["wire"] = _descriptor_document(wire.descriptor)
        expected_freeze_digest = _sha256(expected_freeze_sha256, "--expected-freeze-sha256")
        freeze_raw = qualification._read_regular_path(freeze_path, "freeze manifest")
        freeze_sha256 = hashlib.sha256(freeze_raw).hexdigest()
        report["freeze_sha256"] = freeze_sha256
        if freeze_sha256 != expected_freeze_digest:
            _fail("candidate-sha-mismatch", "freeze manifest SHA-256 differs from trusted input")
        frozen = qualification._validate_freeze(qualification._parse_document(freeze_raw, "freeze manifest"))
        report["candidate_id"] = frozen.candidate_id
        report["model"] = frozen.models["qwen"]
        _validate_frozen_qwen_against_golden(frozen, golden)
        reserved_output_paths = {frozen.final_manifest.path, frozen.final_report.path, *(descriptor.path for descriptor in frozen.receipts.values())}
        base_raw, base_report_sha256 = qualification.revalidate_base_release_candidate(frozen, freeze_sha256, evidence_root)
        if hashlib.sha256(base_raw).hexdigest() != base_report_sha256:
            _fail("base-report-replay-digest-mismatch", "Gate E replay returned inconsistent bytes/digest")
        report["base_release_candidate_report"] = {"path": frozen.final_report.path, "sha256": base_report_sha256}
        receipt_relative, receipt_raw, receipt_document = _read_relative_json(evidence_root, receipt_path, "qwen raw receipt")
        if receipt_relative in reserved_output_paths:
            _fail("reserved-output-path-collision", "raw Qwen receipt must not replace a freeze-declared output")
        receipt_descriptor = Descriptor(receipt_relative, hashlib.sha256(receipt_raw).hexdigest())
        report["receipt"] = _descriptor_document(receipt_descriptor)
        receipt = _validate_receipt(receipt_document, "qwen raw receipt")
        _validate_golden_binding(golden_id=receipt.golden_id, descriptor=receipt.golden, golden=golden, label="qwen raw receipt")
        _validate_wire_binding(descriptor=receipt.wire, wire=wire, label="qwen raw receipt")
        _validate_bound_header(candidate_id=receipt.candidate_id, bindings=receipt.bindings, model=receipt.model, frozen=frozen, freeze_sha256=freeze_sha256, base_report_sha256=base_report_sha256, label="qwen raw receipt")
        report["bindings"] = receipt.bindings
        used_paths = {receipt_descriptor.path}
        _reject_reserved_output_path(receipt.case_manifest, reserved_output_paths, "qwen case manifest")
        manifest_raw, manifest_document = _read_described_json(evidence_root, receipt.case_manifest, "qwen case manifest", used_paths, require_canonical=True)
        report["case_manifest"] = {"path": receipt.case_manifest.path, "sha256": hashlib.sha256(manifest_raw).hexdigest()}
        manifest_candidate, manifest_bindings, manifest_model, manifest_golden, manifest_wire, cases = _validate_case_manifest(manifest_document, "qwen case manifest")
        if manifest_golden != golden.descriptor:
            _fail("qwen-golden-binding-mismatch", "qwen case manifest does not bind reviewed v2 golden")
        _validate_wire_binding(descriptor=manifest_wire, wire=wire, label="qwen case manifest")
        _validate_bound_header(candidate_id=manifest_candidate, bindings=manifest_bindings, model=manifest_model, frozen=frozen, freeze_sha256=freeze_sha256, base_report_sha256=base_report_sha256, label="qwen case manifest")
        resolved_cases: list[dict[str, Any]] = []
        prompt_sequences: set[tuple[int, ...]] = set()
        case_ids: list[str] = []
        for index, value in enumerate(cases):
            case_id, non_stream, stream = _validate_case(value, f"qwen case manifest.cases[{index}]")
            golden_case = golden.cases[index]
            if case_id != golden_case.case_id:
                _fail("qwen-golden-case-id-mismatch", "Qwen case manifest case order drifts from reviewed golden")
            resolved, prompt_sequence = _validate_case_evidence(golden_case=golden_case, wire_case=wire.cases[index], non_stream=non_stream, stream=stream, evidence_root=evidence_root, used_paths=used_paths, reserved_paths=reserved_output_paths, model_id=frozen.models["qwen"]["model_id"])
            if prompt_sequence in prompt_sequences:
                _fail("duplicate-qwen-prompt", "Qwen cases must use distinct prompt token sequences")
            prompt_sequences.add(prompt_sequence)
            case_ids.append(case_id)
            resolved_cases.append(resolved)
        if case_ids != [case.case_id for case in golden.cases]:
            _fail("invalid-case-order", "Qwen case IDs must preserve reviewed lexical order")
        report.update({"status": "passed", "passed": True, "cases": resolved_cases, "checks": [{"name": name, "passed": True} for name in CHECK_NAMES]})
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
    parser.add_argument("--receipt", required=True, help="relative qwen_multistep v2 raw receipt below evidence root")
    parser.add_argument("--report", type=Path, help="create without overwriting")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = evaluate(args.freeze, args.evidence_root, args.receipt, expected_freeze_sha256=args.expected_freeze_sha256)
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
