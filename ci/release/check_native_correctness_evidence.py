#!/usr/bin/env python3
"""Build and replay closed native-correctness evidence without torch.

The native E0 report is a derived view.  This module accepts it only after
opening the original FP32, BF16, and candidate safetensors sidecars and running
``replay_validate_correctness_report`` again.  The raw archive is deterministic
USTAR with a closed inventory; the candidate source archive, report, and native
executable can additionally be cross-bound to independently supplied release
candidate artifacts.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import mmap
import os
import re
import shutil
import stat
import struct
import sys
import tarfile
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, NoReturn


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_ROOT = REPOSITORY_ROOT / "tools/python/reference"
if str(REFERENCE_ROOT) not in sys.path:
    sys.path.insert(0, str(REFERENCE_ROOT))

from rustinfer_reference import calibration  # noqa: E402
from rustinfer_reference import oracle_calibration  # noqa: E402
from rustinfer_reference.constants import RUNTIME_DEPENDENCY_CLASS  # noqa: E402
from release_common import ReleaseContractError, validate_binary  # noqa: E402


SCHEMA_VERSION = "rustinfer.native-correctness-raw-evidence.v1"
PAYLOAD_NAMES = (
    "candidate-source.tar",
    "oracle-source.tar",
    "fp32-manifest.json",
    "fp32-sidecar.safetensors",
    "bf16-manifest.json",
    "bf16-sidecar.safetensors",
    "oracle-report.json",
    "candidate-manifest.json",
    "candidate-sidecar.safetensors",
    "candidate-executable",
    "correctness-report.json",
    "bundle.json",
)
ARCHIVE_NAMES = (*PAYLOAD_NAMES, "SHA256SUMS")
EXECUTABLE_NAME = "candidate-executable"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REVISION_RE = re.compile(r"^[0-9a-f]{40,64}$")
CHECKSUM_RE = re.compile(r"^([0-9a-f]{64})  ([a-z0-9][a-z0-9.-]*)$")
MAX_JSON_BYTES = 64 * 1024 * 1024
MAX_SAFETENSORS_BYTES = 2 * 1024 * 1024 * 1024
MAX_SOURCE_ARCHIVE_BYTES = 8 * 1024 * 1024 * 1024
MAX_RAW_ARCHIVE_BYTES = 16 * 1024 * 1024 * 1024
MAX_SOURCE_MEMBER_BYTES = 2 * 1024 * 1024 * 1024
MAX_SOURCE_TOTAL_BYTES = 8 * 1024 * 1024 * 1024
MAX_SAFETENSORS_HEADER_BYTES = 32 * 1024 * 1024
COPY_BLOCK_BYTES = 1024 * 1024
CANDIDATE_BINARY_MARKERS = (
    b"rustinfer-native",
    b"calibrate",
    b"--gate-manifest",
    b"--prompts",
    b"--manifest",
    b"--sidecar",
    b"--reduction-variant",
    calibration.CALIBRATION_GATE_ID.encode("ascii"),
)


class NativeCorrectnessEvidenceError(ValueError):
    """Raw native correctness evidence is unsafe, incomplete, or inconsistent."""


def _fail(path: str, message: str) -> NoReturn:
    raise NativeCorrectnessEvidenceError(f"{path}: {message}")


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail("json", f"duplicate key {key!r}")
        result[key] = value
    return result


def _nonfinite(value: str) -> NoReturn:
    _fail("json", f"non-finite number {value!r} is forbidden")


def _json_bytes(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    raw = _read_file(path, label, MAX_JSON_BYTES)
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        _fail(label, f"cannot parse JSON: {error}")
    if not isinstance(value, dict):
        _fail(label, "must be a JSON object")
    return value, raw


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _regular_file(path: Path, label: str, maximum: int) -> Path:
    try:
        metadata = path.lstat()
    except OSError as error:
        _fail(label, f"cannot inspect {path}: {error}")
    if not stat.S_ISREG(metadata.st_mode):
        _fail(label, "must be a regular file, not a link or device")
    if metadata.st_size <= 0 or metadata.st_size > maximum:
        _fail(label, f"must contain 1 through {maximum} bytes")
    return path


def _read_file(path: Path, label: str, maximum: int) -> bytes:
    path = _regular_file(path, label, maximum)
    try:
        with path.open("rb") as source:
            raw = source.read(maximum + 1)
    except OSError as error:
        _fail(label, f"cannot read {path}: {error}")
    if len(raw) > maximum:
        _fail(label, f"exceeds the {maximum}-byte bound")
    return raw


def _sha256_file(path: Path, label: str, maximum: int) -> str:
    path = _regular_file(path, label, maximum)
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for block in iter(lambda: source.read(COPY_BLOCK_BYTES), b""):
                digest.update(block)
    except OSError as error:
        _fail(label, f"cannot hash {path}: {error}")
    return digest.hexdigest()


def _safe_sibling(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(label, "must be a non-empty sibling filename")
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or len(pure.parts) != 1
        or pure.name != value
        or value in {".", ".."}
        or "\\" in value
    ):
        _fail(label, "must be a normalized sibling filename")
    return value


def _manifest_inputs(
    manifest_path: Path,
    *,
    expected_kind: str,
) -> tuple[dict[str, Any], Path, Path | None]:
    try:
        manifest = calibration.load_calibration_manifest(manifest_path)
    except (calibration.CalibrationError, OSError) as error:
        _fail(str(manifest_path), str(error))
    if manifest["artifact_kind"] != expected_kind:
        _fail(str(manifest_path), f"artifact kind must be {expected_kind!r}")
    sidecar_name = _safe_sibling(
        manifest["sidecar"]["path"], f"{manifest_path}.sidecar.path"
    )
    sidecar = manifest_path.parent / sidecar_name
    _regular_file(sidecar, f"{manifest_path}.sidecar", MAX_SAFETENSORS_BYTES)
    executable: Path | None = None
    if expected_kind == calibration.CANDIDATE_KIND:
        executable_name = _safe_sibling(
            manifest["candidate_execution"]["executable"]["path"],
            f"{manifest_path}.candidate_execution.executable.path",
        )
        executable = manifest_path.parent / executable_name
        _regular_file(executable, f"{manifest_path}.candidate executable", MAX_SAFETENSORS_BYTES)
    return manifest, sidecar, executable


def _copy_exact(source: BinaryIO, destination: BinaryIO, size: int, label: str) -> None:
    remaining = size
    while remaining:
        block = source.read(min(COPY_BLOCK_BYTES, remaining))
        if not block:
            _fail(label, "payload is truncated")
        destination.write(block)
        remaining -= len(block)
    if source.read(1):
        _fail(label, "payload exceeds declared archive size")


def _tar_mode(name: str) -> int:
    return 0o755 if name == EXECUTABLE_NAME else 0o644


def _write_canonical_tar(
    output: Path,
    payloads: Mapping[str, Path | bytes],
    *,
    exclusive: bool,
) -> None:
    if tuple(payloads) != ARCHIVE_NAMES:
        _fail("raw_evidence", "internal canonical archive order differs")
    mode = "xb" if exclusive else "wb"
    try:
        raw_output = output.open(mode)
    except OSError as error:
        _fail("raw_evidence", f"cannot create {output}: {error}")
    try:
        with raw_output, tarfile.open(
            fileobj=raw_output, mode="w", format=tarfile.USTAR_FORMAT
        ) as archive:
            for name, value in payloads.items():
                member = tarfile.TarInfo(name)
                member.mode = _tar_mode(name)
                member.uid = 0
                member.gid = 0
                member.uname = ""
                member.gname = ""
                member.mtime = 0
                if isinstance(value, bytes):
                    member.size = len(value)
                    archive.addfile(member, fileobj=_BytesReader(value))
                else:
                    source_path = _regular_file(
                        value,
                        f"raw_evidence.{name}",
                        MAX_SOURCE_ARCHIVE_BYTES
                        if name.endswith("source.tar")
                        else MAX_SAFETENSORS_BYTES,
                    )
                    member.size = source_path.stat().st_size
                    with source_path.open("rb") as source:
                        archive.addfile(member, fileobj=source)
    except BaseException:
        if exclusive:
            with contextlib.suppress(OSError):
                output.unlink()
        raise


class _BytesReader:
    def __init__(self, value: bytes) -> None:
        self._value = value
        self._offset = 0

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = len(self._value) - self._offset
        start = self._offset
        stop = min(len(self._value), start + size)
        self._offset = stop
        return self._value[start:stop]


def _checksums(payloads: Mapping[str, Path | bytes]) -> bytes:
    lines: list[bytes] = []
    for name in PAYLOAD_NAMES:
        value = payloads[name]
        digest = (
            hashlib.sha256(value).hexdigest()
            if isinstance(value, bytes)
            else _sha256_file(
                value,
                f"raw_evidence.{name}",
                MAX_SOURCE_ARCHIVE_BYTES
                if name.endswith("source.tar")
                else MAX_SAFETENSORS_BYTES,
            )
        )
        lines.append(f"{digest}  {name}\n".encode("ascii"))
    return b"".join(lines)


def _archive_revision(path: Path, label: str) -> str:
    _regular_file(path, label, MAX_SOURCE_ARCHIVE_BYTES)
    try:
        with tarfile.open(path, "r:") as archive:
            revision = archive.pax_headers.get("comment")
            if not isinstance(revision, str) or REVISION_RE.fullmatch(revision) is None:
                _fail(label, "missing lowercase Git revision pax comment")
            members = archive.getmembers()
            if not members:
                _fail(label, "source archive is empty")
            for member in members:
                if member.pax_headers.get("comment") != revision:
                    _fail(label, f"member lacks exact Git revision marker: {member.name}")
            return revision
    except (OSError, tarfile.TarError) as error:
        _fail(label, f"cannot inspect source archive: {error}")


def _source_members(archive: tarfile.TarFile, label: str) -> list[tarfile.TarInfo]:
    members = archive.getmembers()
    if not members:
        _fail(label, "source archive is empty")
    names: set[str] = set()
    total = 0
    for member in members:
        name = member.name
        pure = PurePosixPath(name)
        if (
            not name
            or "\\" in name
            or "//" in name
            or pure.is_absolute()
            or any(part in {"", ".", ".."} for part in pure.parts)
        ):
            _fail(label, f"unsafe source member path: {name!r}")
        if name in names:
            _fail(label, f"duplicate source member: {name}")
        names.add(name)
        if member.issym() or member.islnk() or member.isdev() or member.isfifo():
            _fail(label, f"links and special members are forbidden: {name}")
        if not member.isdir() and not member.isreg():
            _fail(label, f"unsupported source member type: {name}")
        if member.size < 0 or member.size > MAX_SOURCE_MEMBER_BYTES:
            _fail(label, f"source member exceeds size bound: {name}")
        total += member.size
        if total > MAX_SOURCE_TOTAL_BYTES:
            _fail(label, "source archive exceeds total extracted size bound")
    return members


def _extract_source(path: Path, destination: Path, label: str) -> str:
    revision = _archive_revision(path, label)
    try:
        with tarfile.open(path, "r:") as archive:
            members = _source_members(archive, label)
            for member in members:
                target = destination.joinpath(*PurePosixPath(member.name).parts)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    _fail(label, f"cannot read source member: {member.name}")
                with source, target.open("xb") as output:
                    _copy_exact(source, output, member.size, f"{label}:{member.name}")
                target.chmod(0o644)
    except NativeCorrectnessEvidenceError:
        raise
    except (OSError, tarfile.TarError) as error:
        _fail(label, f"cannot extract source archive: {error}")
    return revision


def _parse_checksum_file(raw: bytes) -> dict[str, str]:
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as error:
        _fail("SHA256SUMS", f"must be ASCII: {error}")
    if not text.endswith("\n"):
        _fail("SHA256SUMS", "must end with a newline")
    result: dict[str, str] = {}
    for line in text.splitlines():
        match = CHECKSUM_RE.fullmatch(line)
        if match is None:
            _fail("SHA256SUMS", f"invalid checksum record: {line!r}")
        digest, name = match.groups()
        if name in result:
            _fail("SHA256SUMS", f"duplicate checksum path: {name}")
        result[name] = digest
    if tuple(result) != PAYLOAD_NAMES:
        _fail("SHA256SUMS", "checksum inventory/order differs from canonical payload order")
    return result


def _extract_raw_archive(path: Path, destination: Path) -> dict[str, Path]:
    _regular_file(path, "raw_evidence", MAX_RAW_ARCHIVE_BYTES)
    extracted: dict[str, Path] = {}
    try:
        with tarfile.open(path, "r:") as archive:
            if archive.pax_headers:
                _fail("raw_evidence", "USTAR archive must not contain pax headers")
            members = archive.getmembers()
            if tuple(member.name for member in members) != ARCHIVE_NAMES:
                _fail("raw_evidence", "closed member inventory/order differs")
            for member in members:
                if not member.isreg():
                    _fail("raw_evidence", f"member is not regular: {member.name}")
                expected_mode = _tar_mode(member.name)
                if (
                    member.mode != expected_mode
                    or member.uid != 0
                    or member.gid != 0
                    or member.uname != ""
                    or member.gname != ""
                    or member.mtime != 0
                    or member.pax_headers
                ):
                    _fail("raw_evidence", f"noncanonical USTAR metadata: {member.name}")
                maximum = (
                    MAX_SOURCE_ARCHIVE_BYTES
                    if member.name.endswith("source.tar")
                    else MAX_SAFETENSORS_BYTES
                )
                if member.size <= 0 or member.size > maximum:
                    _fail("raw_evidence", f"invalid member size: {member.name}")
                source = archive.extractfile(member)
                if source is None:
                    _fail("raw_evidence", f"cannot read member: {member.name}")
                target = destination / member.name
                with source, target.open("xb") as output:
                    _copy_exact(source, output, member.size, f"raw_evidence:{member.name}")
                target.chmod(expected_mode)
                extracted[member.name] = target
    except NativeCorrectnessEvidenceError:
        raise
    except (OSError, tarfile.TarError) as error:
        _fail("raw_evidence", f"cannot extract raw archive: {error}")

    checksums = _parse_checksum_file(
        _read_file(extracted["SHA256SUMS"], "SHA256SUMS", MAX_JSON_BYTES)
    )
    for name, expected in checksums.items():
        actual = _sha256_file(
            extracted[name],
            f"raw_evidence:{name}",
            MAX_SOURCE_ARCHIVE_BYTES
            if name.endswith("source.tar")
            else MAX_SAFETENSORS_BYTES,
        )
        if actual != expected:
            _fail(f"raw_evidence:{name}", f"checksum mismatch: {actual}")

    rebuilt = destination / "canonical-rebuilt.tar"
    ordered: dict[str, Path | bytes] = {name: extracted[name] for name in ARCHIVE_NAMES}
    _write_canonical_tar(rebuilt, ordered, exclusive=True)
    original_sha = _sha256_file(path, "raw_evidence", MAX_RAW_ARCHIVE_BYTES)
    rebuilt_sha = _sha256_file(rebuilt, "raw_evidence.rebuilt", MAX_RAW_ARCHIVE_BYTES)
    if original_sha != rebuilt_sha or path.stat().st_size != rebuilt.stat().st_size:
        _fail("raw_evidence", "archive bytes are not canonical deterministic USTAR")
    return extracted


class _Scalar:
    def __init__(self, value: float | bool) -> None:
        self._value = value

    def item(self) -> float | bool:
        return self._value


class _MathVector:
    def __init__(self, values: Sequence[float | bool]) -> None:
        self._values = list(values)

    def __sub__(self, other: "_MathVector") -> "_MathVector":
        return _MathVector(
            [a - b for a, b in zip(self._values, other._values, strict=True)]
        )

    def __mul__(self, other: "_MathVector") -> "_MathVector":
        return _MathVector(
            [a * b for a, b in zip(self._values, other._values, strict=True)]
        )

    def __truediv__(self, other: "_MathVector") -> "_MathVector":
        return _MathVector(
            [a / b for a, b in zip(self._values, other._values, strict=True)]
        )

    def abs(self) -> "_MathVector":
        return _MathVector([abs(float(value)) for value in self._values])

    def clamp_min(self, minimum: float) -> "_MathVector":
        return _MathVector([max(float(value), minimum) for value in self._values])

    def isfinite(self) -> "_MathVector":
        return _MathVector([math.isfinite(float(value)) for value in self._values])

    def all(self) -> _Scalar:
        return _Scalar(all(bool(value) for value in self._values))

    def max(self) -> _Scalar:
        return _Scalar(max(float(value) for value in self._values))

    def sum(self) -> _Scalar:
        return _Scalar(math.fsum(float(value) for value in self._values))

    def double(self) -> "_MathVector":
        return self


class _PureTensor:
    def __init__(
        self,
        mapping: "_PureSafeTensorMapping",
        *,
        dtype: str,
        shape: tuple[int, ...],
        data_start: int,
        count: int,
    ) -> None:
        self._mapping = mapping
        self.dtype = dtype
        self.shape = shape
        self._data_start = data_start
        self._count = count

    def detach(self) -> "_PureTensor":
        return self

    def cpu(self) -> "_PureTensor":
        return self

    def float(self) -> "_PureTensor":
        return self

    def contiguous(self) -> "_PureTensor":
        return self

    def reshape(self, size: int) -> "_PureTensor":
        if size != -1:
            raise ValueError("pure reader only supports flatten")
        return self

    def numel(self) -> int:
        return self._count

    def __len__(self) -> int:
        return self._count

    def __getitem__(self, index: slice) -> _MathVector:
        if not isinstance(index, slice) or index.step not in {None, 1}:
            raise TypeError("pure reader requires contiguous slices")
        start, stop, _ = index.indices(self._count)
        return _MathVector(self._decode(start, stop))

    def tolist(self) -> list[float]:
        return self._decode(0, self._count)

    def _decode(self, start: int, stop: int) -> list[float]:
        if self.dtype == "F32":
            offset = self._data_start + start * 4
            raw = self._mapping._mmap[offset : self._data_start + stop * 4]
            return [item[0] for item in struct.iter_unpack("<f", raw)]
        offset = self._data_start + start * 2
        raw = self._mapping._mmap[offset : self._data_start + stop * 2]
        result: list[float] = []
        for item in struct.iter_unpack("<H", raw):
            result.append(struct.unpack("<f", struct.pack("<I", item[0] << 16))[0])
        return result


class _PureSafeTensorMapping(Mapping[str, object]):
    def __init__(self, path: Path) -> None:
        self._path = _regular_file(path, "safetensors", MAX_SAFETENSORS_BYTES)
        try:
            self._file = self._path.open("rb")
            self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        except OSError as error:
            _fail("safetensors", f"cannot map {path}: {error}")
        self._tensors = self._parse()

    def _parse(self) -> dict[str, _PureTensor]:
        if len(self._mmap) < 10:
            _fail("safetensors", "file is too short")
        header_length = struct.unpack("<Q", self._mmap[:8])[0]
        if header_length <= 1 or header_length > MAX_SAFETENSORS_HEADER_BYTES:
            _fail("safetensors", "header length is outside the reviewed bound")
        data_start = 8 + header_length
        if data_start > len(self._mmap):
            _fail("safetensors", "header exceeds file length")
        try:
            header_text = self._mmap[8:data_start].decode("utf-8")
            header = json.loads(
                header_text,
                object_pairs_hook=_pairs,
                parse_constant=_nonfinite,
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            _fail("safetensors", f"invalid JSON header: {error}")
        if not isinstance(header, dict):
            _fail("safetensors", "header must be an object")
        metadata = header.pop("__metadata__", None)
        if metadata is not None and (
            not isinstance(metadata, dict)
            or any(not isinstance(k, str) or not isinstance(v, str) for k, v in metadata.items())
        ):
            _fail("safetensors.__metadata__", "must map strings to strings")
        tensors: dict[str, _PureTensor] = {}
        intervals: list[tuple[int, int, str]] = []
        for name, raw in header.items():
            if not isinstance(name, str) or not name or not isinstance(raw, dict):
                _fail("safetensors", "tensor entries require non-empty names and objects")
            if set(raw) != {"dtype", "shape", "data_offsets"}:
                _fail(f"safetensors.{name}", "closed tensor metadata differs")
            dtype = raw["dtype"]
            width = {"F32": 4, "BF16": 2}.get(dtype)
            if width is None:
                _fail(f"safetensors.{name}.dtype", "only F32 and BF16 are supported")
            shape = raw["shape"]
            if (
                not isinstance(shape, list)
                or not shape
                or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in shape)
            ):
                _fail(f"safetensors.{name}.shape", "requires positive integer dimensions")
            count = math.prod(shape)
            offsets = raw["data_offsets"]
            if (
                not isinstance(offsets, list)
                or len(offsets) != 2
                or any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in offsets)
            ):
                _fail(f"safetensors.{name}.data_offsets", "requires two nonnegative integers")
            start, stop = offsets
            if stop - start != count * width or stop > len(self._mmap) - data_start:
                _fail(f"safetensors.{name}.data_offsets", "does not match dtype/shape/file")
            intervals.append((start, stop, name))
            tensors[name] = _PureTensor(
                self,
                dtype=dtype,
                shape=tuple(shape),
                data_start=data_start + start,
                count=count,
            )
        if not tensors:
            _fail("safetensors", "must contain at least one tensor")
        cursor = 0
        for start, stop, name in sorted(intervals):
            if start != cursor:
                _fail(f"safetensors.{name}.data_offsets", "tensor data has a gap or overlap")
            cursor = stop
        if cursor != len(self._mmap) - data_start:
            _fail("safetensors", "unreferenced trailing tensor data is forbidden")
        return tensors

    def __getitem__(self, key: str) -> object:
        return self._tensors[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._tensors)

    def __len__(self) -> int:
        return len(self._tensors)

    def close(self) -> None:
        tensors, self._tensors = self._tensors, {}
        del tensors
        with contextlib.suppress(Exception):
            self._mmap.close()
        with contextlib.suppress(Exception):
            self._file.close()


def pure_safetensors_loader(path: Path) -> Mapping[str, object]:
    """Load F32/BF16 safetensors with only the Python standard library."""

    return _PureSafeTensorMapping(path)


def _load_manifest(path: Path, kind: str) -> dict[str, Any]:
    try:
        manifest = calibration.load_calibration_manifest(path)
    except (calibration.CalibrationError, OSError) as error:
        _fail(str(path), str(error))
    if manifest["artifact_kind"] != kind:
        _fail(str(path), f"artifact kind must be {kind!r}")
    return manifest


@dataclass(frozen=True)
class _SourceRoots:
    candidate: Path
    candidate_revision: str
    oracle: Path
    oracle_revision: str


def _source_root(manifest: Mapping[str, object], roots: _SourceRoots) -> tuple[Path, str]:
    if manifest["artifact_kind"] == calibration.CANDIDATE_KIND:
        return roots.candidate, roots.candidate_revision
    return roots.oracle, roots.oracle_revision


def _verify_manifest_sources(
    manifest: Mapping[str, object], roots: _SourceRoots
) -> None:
    root, revision = _source_root(manifest, roots)
    if manifest["provenance"]["git_revision"] != revision:
        raise calibration.CalibrationError(
            "manifest source revision differs from preserved source archive"
        )
    for name in calibration.SOURCE_NAMES:
        source = manifest["provenance"]["sources"][name]
        path = (root / source["path"]).resolve()
        if root.resolve() != path and root.resolve() not in path.parents:
            raise calibration.CalibrationError(f"source {name} escapes repository root")
        if calibration.sha256_file(path) != source["sha256"]:
            raise calibration.CalibrationError(f"source {name} SHA-256 differs from archive")
    gate_path = root / manifest["provenance"]["sources"]["gate_manifest"]["path"]
    if calibration._load_json_object(gate_path, "correctness gate manifest") != calibration.gate_contract_document():
        raise calibration.CalibrationError(
            "language-neutral correctness gate manifest differs from tool"
        )
    from rustinfer_reference.fixture import load_prompts

    prompts_path = root / manifest["provenance"]["sources"]["prompts"]["path"]
    prompts, corpus_sha256 = load_prompts(prompts_path)
    if corpus_sha256 != manifest["provenance"]["sources"]["prompts"]["sha256"]:
        raise calibration.CalibrationError("prompt corpus digest differs after parsing")
    cases = manifest["cases"]
    if len(prompts) != len(cases):
        raise calibration.CalibrationError("ordered prompt corpus and manifest case count differ")
    for index, (prompt, case) in enumerate(zip(prompts, cases, strict=True)):
        if (
            case["prompt_id"] != prompt.prompt_id
            or case["prompt_text_sha256"]
            != hashlib.sha256(prompt.text.encode("utf-8")).hexdigest()
            or case["prompt_metadata"] != prompt.metadata
        ):
            raise calibration.CalibrationError(
                f"manifest.cases[{index}]: does not bind the ordered prompt row"
            )
        if (
            prompt.target_prompt_tokens is not None
            and case["input_token_count"] != prompt.target_prompt_tokens
        ):
            raise calibration.CalibrationError(
                f"manifest.cases[{index}]: target_prompt_tokens was not materialized exactly"
            )
    lane_path = root / manifest["provenance"]["sources"]["lane_manifest"]["path"]
    lane = calibration._load_json_object(lane_path, "lane manifest")
    if manifest["artifact_kind"] == calibration.CANDIDATE_KIND:
        if (
            lane.get("lane_id") != "rustinfer-native"
            or lane.get("implementation_id") != "rustinfer-native"
            or lane.get("runtime_dependency_class") != "native-production"
        ):
            raise calibration.CalibrationError("candidate lane manifest is not rustinfer-native")
        engine = calibration._expect_object(lane.get("engine"), "candidate lane.engine")
        if engine.get("revision") != manifest["producer"]["engine_revision"]:
            raise calibration.CalibrationError("candidate engine revision differs from lane manifest")
    elif (
        lane.get("lane_id") != "hf-transformers"
        or lane.get("implementation_id") != "hf-transformers-eager"
        or lane.get("runtime_dependency_class") != RUNTIME_DEPENDENCY_CLASS
    ):
        raise calibration.CalibrationError("oracle lane manifest is not hf-transformers")


def _stage_manifest(
    root: Path,
    manifest_payload: Path,
    sidecar_payload: Path,
    *,
    kind: str,
    executable_payload: Path | None = None,
) -> tuple[dict[str, Any], Path]:
    root.mkdir()
    manifest = _load_manifest(manifest_payload, kind)
    sidecar_name = _safe_sibling(manifest["sidecar"]["path"], "manifest.sidecar.path")
    if kind == calibration.CANDIDATE_KIND:
        capture = manifest["candidate_execution"]["capture_argv"]
        outputs = calibration._flag_values(capture, "--manifest", "candidate.capture_argv")
        if len(outputs) != 1:
            _fail("candidate.capture_argv", "must contain one manifest output")
        manifest_name = _safe_sibling(outputs[0], "candidate.capture_argv.--manifest")
    else:
        manifest_name = "manifest.json"
    manifest_path = root / manifest_name
    os.link(manifest_payload, manifest_path)
    os.link(sidecar_payload, root / sidecar_name)
    if executable_payload is not None:
        executable_name = _safe_sibling(
            manifest["candidate_execution"]["executable"]["path"],
            "candidate.executable.path",
        )
        executable_path = root / executable_name
        os.link(executable_payload, executable_path)
        executable_path.chmod(0o755)
    return manifest, manifest_path


def _validate_candidate_binary(
    path: Path, manifest: Mapping[str, object]
) -> str:
    raw = _read_file(path, "candidate-executable", MAX_SAFETENSORS_BYTES)
    try:
        validate_binary(raw)
    except ReleaseContractError as error:
        _fail("candidate-executable", f"invalid Linux x86_64 native ELF: {error}")
    missing = [
        marker.decode("ascii")
        for marker in CANDIDATE_BINARY_MARKERS
        if marker not in raw
    ]
    if missing:
        _fail(
            "candidate-executable",
            f"missing reviewed native calibration markers: {missing}",
        )
    capture = manifest["candidate_execution"]["capture_argv"]
    executable_name = manifest["candidate_execution"]["executable"]["path"]
    if capture[:2] != [executable_name, "calibrate"]:
        _fail(
            "candidate-manifest.json.candidate_execution.capture_argv",
            "does not invoke the replayed native executable calibrate ABI",
        )
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class NativeReplayResult:
    schema_version: str
    source_revision: str
    source_archive_sha256: str
    oracle_source_revision: str
    correctness_report_sha256: str
    candidate_executable_sha256: str
    case_count: int
    failure_count: int


def replay_raw_evidence(
    raw_evidence: Path,
    *,
    source_revision: str | None = None,
    source_archive: Path | None = None,
    correctness_report: Path | None = None,
    candidate_executable: Path | None = None,
) -> NativeReplayResult:
    """Replay raw tensors and optionally bind independent candidate artifacts."""

    mappings: list[_PureSafeTensorMapping] = []

    def loader(path: Path) -> Mapping[str, object]:
        mapping = _PureSafeTensorMapping(path)
        mappings.append(mapping)
        return mapping

    try:
        with tempfile.TemporaryDirectory(prefix="rustinfer-native-replay-") as directory:
            temp = Path(directory)
            raw_root = temp / "raw"
            raw_root.mkdir()
            files = _extract_raw_archive(raw_evidence, raw_root)
            bundle, _ = _json_bytes(files["bundle.json"], "bundle.json")
            if set(bundle) != {
                "schema_version",
                "candidate_source_revision",
                "oracle_source_revision",
            } or bundle.get("schema_version") != SCHEMA_VERSION:
                _fail("bundle.json", "closed schema/version differs")
            candidate_revision = bundle["candidate_source_revision"]
            oracle_revision = bundle["oracle_source_revision"]
            if (
                not isinstance(candidate_revision, str)
                or REVISION_RE.fullmatch(candidate_revision) is None
                or not isinstance(oracle_revision, str)
                or REVISION_RE.fullmatch(oracle_revision) is None
            ):
                _fail("bundle.json", "source revisions must be lowercase Git revisions")
            candidate_source_root = temp / "candidate-source"
            oracle_source_root = temp / "oracle-source"
            candidate_source_root.mkdir()
            oracle_source_root.mkdir()
            observed_candidate_revision = _extract_source(
                files["candidate-source.tar"], candidate_source_root, "candidate-source.tar"
            )
            observed_oracle_revision = _extract_source(
                files["oracle-source.tar"], oracle_source_root, "oracle-source.tar"
            )
            if observed_candidate_revision != candidate_revision:
                _fail("bundle.json.candidate_source_revision", "source archive marker differs")
            if observed_oracle_revision != oracle_revision:
                _fail("bundle.json.oracle_source_revision", "source archive marker differs")

            fp32, fp32_path = _stage_manifest(
                temp / "fp32",
                files["fp32-manifest.json"],
                files["fp32-sidecar.safetensors"],
                kind=calibration.FP32_ORACLE_KIND,
            )
            bf16, bf16_path = _stage_manifest(
                temp / "bf16",
                files["bf16-manifest.json"],
                files["bf16-sidecar.safetensors"],
                kind=calibration.BF16_ORACLE_KIND,
            )
            candidate, candidate_path = _stage_manifest(
                temp / "candidate",
                files["candidate-manifest.json"],
                files["candidate-sidecar.safetensors"],
                kind=calibration.CANDIDATE_KIND,
                executable_payload=files["candidate-executable"],
            )
            candidate_sha = _validate_candidate_binary(
                files["candidate-executable"], candidate
            )
            oracle_document, _ = _json_bytes(files["oracle-report.json"], "oracle-report.json")
            report, report_raw = _json_bytes(
                files["correctness-report.json"], "correctness-report.json"
            )
            roots = _SourceRoots(
                candidate=candidate_source_root,
                candidate_revision=candidate_revision,
                oracle=oracle_source_root,
                oracle_revision=oracle_revision,
            )

            def source_verifier(manifest: Mapping[str, object], _root: Path) -> None:
                _verify_manifest_sources(manifest, roots)

            original_calibration_verifier = calibration.verify_manifest_sources
            original_oracle_verifier = oracle_calibration.verify_manifest_sources
            calibration.verify_manifest_sources = source_verifier
            oracle_calibration.verify_manifest_sources = source_verifier
            try:
                calibration.replay_validate_correctness_report(
                    report=report,
                    fp32_manifest=fp32,
                    fp32_manifest_path=fp32_path,
                    bf16_manifest=bf16,
                    bf16_manifest_path=bf16_path,
                    oracle_calibration_report=oracle_document,
                    oracle_calibration_report_path=files["oracle-report.json"],
                    candidate_manifest=candidate,
                    candidate_manifest_path=candidate_path,
                    repo_root=candidate_source_root,
                    sidecar_loader=loader,
                )
            except (calibration.CalibrationError, OSError, ValueError) as error:
                _fail("native_correctness", f"comparator replay failed: {error}")
            finally:
                calibration.verify_manifest_sources = original_calibration_verifier
                oracle_calibration.verify_manifest_sources = original_oracle_verifier

            if candidate["candidate_execution"]["executable"]["sha256"] != candidate_sha:
                _fail("candidate-manifest.json", "executable digest differs from raw executable")
            if report["bindings"]["candidate_executable_sha256"] != candidate_sha:
                _fail("correctness-report.json", "executable binding differs from raw executable")
            report_sha = hashlib.sha256(report_raw).hexdigest()
            source_sha = _sha256_file(
                files["candidate-source.tar"],
                "candidate-source.tar",
                MAX_SOURCE_ARCHIVE_BYTES,
            )

            if source_revision is not None and source_revision != candidate_revision:
                _fail("source_revision", "does not match replayed candidate source")
            external_bindings: tuple[tuple[Path | None, Path, str, int], ...] = (
                (
                    source_archive,
                    files["candidate-source.tar"],
                    "source_archive",
                    MAX_SOURCE_ARCHIVE_BYTES,
                ),
                (
                    correctness_report,
                    files["correctness-report.json"],
                    "correctness_report",
                    MAX_JSON_BYTES,
                ),
                (
                    candidate_executable,
                    files["candidate-executable"],
                    "candidate_executable",
                    MAX_SAFETENSORS_BYTES,
                ),
            )
            for external, embedded, label, maximum in external_bindings:
                if external is None:
                    continue
                if _sha256_file(external, label, maximum) != _sha256_file(
                    embedded, f"raw_evidence.{label}", maximum
                ):
                    _fail(label, "bytes differ from raw replay bundle")

            summary = report["summary"]
            return NativeReplayResult(
                schema_version=SCHEMA_VERSION,
                source_revision=candidate_revision,
                source_archive_sha256=source_sha,
                oracle_source_revision=oracle_revision,
                correctness_report_sha256=report_sha,
                candidate_executable_sha256=candidate_sha,
                case_count=int(summary["case_count"]),
                failure_count=int(summary["failure_count"]),
            )
    finally:
        for mapping in reversed(mappings):
            mapping.close()


def build_raw_evidence(
    *,
    candidate_source_archive: Path,
    oracle_source_archive: Path,
    fp32_manifest: Path,
    bf16_manifest: Path,
    oracle_report: Path,
    candidate_manifest: Path,
    correctness_report: Path,
    output: Path,
) -> NativeReplayResult:
    """Validate inputs, write a deterministic closed archive, then replay it."""

    _, fp32_sidecar, _ = _manifest_inputs(
        fp32_manifest, expected_kind=calibration.FP32_ORACLE_KIND
    )
    _, bf16_sidecar, _ = _manifest_inputs(
        bf16_manifest, expected_kind=calibration.BF16_ORACLE_KIND
    )
    candidate_document, candidate_sidecar, executable = _manifest_inputs(
        candidate_manifest, expected_kind=calibration.CANDIDATE_KIND
    )
    assert executable is not None
    candidate_revision = _archive_revision(
        candidate_source_archive, "candidate_source_archive"
    )
    oracle_revision = _archive_revision(oracle_source_archive, "oracle_source_archive")
    if candidate_document["provenance"]["git_revision"] != candidate_revision:
        _fail("candidate_manifest", "revision differs from candidate source archive")
    bundle = _canonical_json_bytes(
        {
            "schema_version": SCHEMA_VERSION,
            "candidate_source_revision": candidate_revision,
            "oracle_source_revision": oracle_revision,
        }
    ) + b"\n"
    payloads: dict[str, Path | bytes] = {
        "candidate-source.tar": candidate_source_archive,
        "oracle-source.tar": oracle_source_archive,
        "fp32-manifest.json": fp32_manifest,
        "fp32-sidecar.safetensors": fp32_sidecar,
        "bf16-manifest.json": bf16_manifest,
        "bf16-sidecar.safetensors": bf16_sidecar,
        "oracle-report.json": oracle_report,
        "candidate-manifest.json": candidate_manifest,
        "candidate-sidecar.safetensors": candidate_sidecar,
        "candidate-executable": executable,
        "correctness-report.json": correctness_report,
        "bundle.json": bundle,
    }
    payloads["SHA256SUMS"] = _checksums(payloads)
    _write_canonical_tar(output, payloads, exclusive=True)
    try:
        return replay_raw_evidence(
            output,
            source_revision=candidate_revision,
            source_archive=candidate_source_archive,
            correctness_report=correctness_report,
            candidate_executable=executable,
        )
    except BaseException:
        with contextlib.suppress(OSError):
            output.unlink()
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-source-archive", type=Path, required=True)
    parser.add_argument("--oracle-source-archive", type=Path, required=True)
    parser.add_argument("--fp32-manifest", type=Path, required=True)
    parser.add_argument("--bf16-manifest", type=Path, required=True)
    parser.add_argument("--oracle-report", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--correctness-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = build_raw_evidence(
            candidate_source_archive=args.candidate_source_archive,
            oracle_source_archive=args.oracle_source_archive,
            fp32_manifest=args.fp32_manifest,
            bf16_manifest=args.bf16_manifest,
            oracle_report=args.oracle_report,
            candidate_manifest=args.candidate_manifest,
            correctness_report=args.correctness_report,
            output=args.output,
        )
    except (NativeCorrectnessEvidenceError, OSError) as error:
        print(f"native correctness evidence failed: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "schema_version": result.schema_version,
                "source_revision": result.source_revision,
                "source_archive_sha256": result.source_archive_sha256,
                "oracle_source_revision": result.oracle_source_revision,
                "correctness_report_sha256": result.correctness_report_sha256,
                "candidate_executable_sha256": result.candidate_executable_sha256,
                "case_count": result.case_count,
                "failure_count": result.failure_count,
                "raw_evidence_sha256": _sha256_file(
                    args.output, "raw_evidence", MAX_RAW_ARCHIVE_BYTES
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
