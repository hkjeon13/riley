"""Pure-Python fixture assembly, canonical tensor summaries, and validation."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import struct
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from .constants import (
    ATTENTION_BACKEND,
    DTYPE,
    FIXTURE_SCHEMA_VERSION,
    GENERATOR_NAME,
    GENERATOR_VERSION,
    GOLDEN_GREEDY_MAX_NEW_TOKENS,
    MAX_CONTEXT_TOKENS,
    MODEL_CONFIG_SHA256,
    MODEL_ID,
    MODEL_REVISION,
    MODEL_WEIGHTS_SHA256,
    PRIMARY_ENVIRONMENT_ID,
    PRIMARY_GPU_COMPUTE_CAPABILITY,
    PRIMARY_GPU_NAME,
    PROMPT_CONTRACT_VERSION,
    PYTHON_EXECUTABLE_SHA256,
    PYTHON_PLATFORM_MACHINE,
    PYTHON_PLATFORM_SYSTEM,
    PYTHON_VERSION,
    PYTHON_VERSION_FILE_SHA256,
    PYTHON_VERSION_RANGE,
    RNG_ALGORITHM,
    RUNTIME_DEPENDENCY_CLASS,
    SEMANTIC_CLASS,
    TORCH_VERSION,
    TOKENIZER_ARTIFACT_FILENAMES,
    TOKENIZER_FILES_SHA256,
    TOKENIZER_SHA256,
    TRANSFORMERS_VERSION,
    UINT64_MAX,
)
from .philox import Philox4x32, derive
from .environment import (
    EnvironmentContractError,
    EnvironmentProbe,
    probe_primary_environment,
    validate_environment_snapshot,
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PROMPT_KEYS = {
    "contract_version",
    "prompt_id",
    "category",
    "language",
    "text",
    "target_prompt_tokens",
    "boundary_kind",
    "expected_behavior",
    "contains_sensitive_data",
}
_CATEGORIES = {
    "short",
    "multilingual",
    "symbols-code",
    "long-repetition",
    "context-boundary",
    "minimal",
    "early-eos",
}
_LANGUAGES = {"ko", "en", "mixed", "code", "none"}
_BOUNDARY_KINDS = {"none", "near-max-context"}
_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
FIXTURE_SOURCE_PATHS = {
    "matrix": "benchmarks/matrix.yaml",
    "prompts": "benchmarks/prompts.jsonl",
    "environment": "benchmarks/environment.md",
    "lane_manifest": "benchmarks/lanes/hf-transformers.json",
    "correctness_gate": "benchmarks/correctness/smollm2-fp32-bf16-native-e0-v2.json",
    "prompt_schema": "benchmarks/schemas/prompt.schema.json",
    "fixture_schema": "benchmarks/schemas/reference-fixture.schema.json",
    "contract_validator": "benchmarks/scripts/validate_contract.py",
    "dependency_manifest": "tools/python/reference/pyproject.toml",
    "dependency_lock": "tools/python/reference/uv.lock",
    "python_version_file": "tools/python/reference/.python-version",
    "constants": "tools/python/reference/riley_reference/constants.py",
    "environment_probe": "tools/python/reference/riley_reference/environment.py",
    "fixture_generator": "tools/python/reference/riley_reference/fixture.py",
    "hf_backend": "tools/python/reference/riley_reference/hf_backend.py",
    "cli": "tools/python/reference/riley_reference/cli.py",
}


class FixtureError(ValueError):
    """Base class for reference fixture failures."""


class PromptCorpusError(FixtureError):
    """The prompt corpus does not satisfy its versioned contract."""


class CacheParityError(FixtureError):
    """Greedy cache-on and cache-off generation diverged."""


class FixtureValidationError(FixtureError):
    """A fixture is malformed or violates the immutable reference contract."""


@dataclass(frozen=True)
class PromptRecord:
    prompt_id: str
    category: str
    language: str
    text: str
    target_prompt_tokens: int | None
    boundary_kind: str
    expected_behavior: str

    @property
    def metadata(self) -> dict[str, object]:
        return {
            "category": self.category,
            "language": self.language,
            "target_prompt_tokens": self.target_prompt_tokens,
            "boundary_kind": self.boundary_kind,
            "expected_behavior": self.expected_behavior,
        }


@dataclass(frozen=True)
class BackendMetadata:
    python_version: str
    python_executable_sha256: str
    python_platform_system: str
    python_platform_machine: str
    torch_version: str
    transformers_version: str
    device: str
    device_name: str | None
    compute_capability: str | None
    local_files_only: bool
    weights_sha256: str
    config_sha256: str
    tokenizer_sha256: str
    tokenizer_files_sha256: Mapping[str, str]


@dataclass(frozen=True)
class CaseResult:
    input_token_ids: tuple[int, ...]
    hidden_state: Mapping[str, object]
    final_logits: Mapping[str, object]
    processed_log_probs: Mapping[str, object]
    cache_on_token_ids: tuple[int, ...]
    cache_off_token_ids: tuple[int, ...]
    cache_on_stop_reason: str
    cache_off_stop_reason: str


class ReferenceBackend(Protocol):
    metadata: BackendMetadata
    eos_token_ids: tuple[int, ...]

    def generate_case(
        self,
        text: str,
        *,
        max_new_tokens: int,
        hidden_state_index: int,
        top_k: int,
        target_prompt_tokens: int | None,
    ) -> CaseResult: ...


class _StableSum:
    """Neumaier accumulation without retaining a large hidden state."""

    def __init__(self) -> None:
        self.total = 0.0
        self.compensation = 0.0

    def add(self, value: float) -> None:
        updated = self.total + value
        if abs(self.total) >= abs(value):
            self.compensation += (self.total - updated) + value
        else:
            self.compensation += (value - updated) + self.total
        self.total = updated

    def value(self) -> float:
        return self.total + self.compensation


def summarize_values(
    values: Iterable[float],
    shape: Sequence[int],
    *,
    top_k: int | None = None,
) -> dict[str, object]:
    """Summarize row-major values using canonical little-endian binary32 bytes."""

    dimensions = tuple(shape)
    if not dimensions or any(
        isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0
        for dimension in dimensions
    ):
        raise FixtureError("tensor shape must contain positive integer dimensions")
    expected_numel = math.prod(dimensions)
    if top_k is not None and (isinstance(top_k, bool) or top_k <= 0):
        raise FixtureError("top_k must be a positive integer")

    checksum = hashlib.sha256()
    total = _StableSum()
    squared_total = _StableSum()
    minimum = math.inf
    maximum = -math.inf
    count = 0
    chunk: list[float] = []
    ranked_values: list[tuple[int, float]] | None = [] if top_k is not None else None

    def flush_chunk() -> None:
        nonlocal count, minimum, maximum
        if not chunk:
            return
        try:
            encoded = struct.pack(f"<{len(chunk)}f", *chunk)
        except (OverflowError, struct.error) as error:
            raise FixtureError(
                f"tensor value near flat index {count} is outside binary32 range"
            ) from error
        checksum.update(encoded)
        canonical_values = struct.unpack(f"<{len(chunk)}f", encoded)
        for value in canonical_values:
            minimum = min(minimum, value)
            maximum = max(maximum, value)
            total.add(value)
            squared_total.add(value * value)
            if ranked_values is not None:
                ranked_values.append((count, value))
            count += 1
        chunk.clear()

    for raw_value in values:
        value = float(raw_value)
        if not math.isfinite(value):
            raise FixtureError(
                f"tensor value at flat index {count + len(chunk)} is not finite"
            )
        chunk.append(value)
        if len(chunk) == 4096:
            flush_chunk()
    flush_chunk()
    if count != expected_numel:
        raise FixtureError(
            f"tensor shape implies {expected_numel} values, received {count}"
        )

    summary: dict[str, object] = {
        "canonical_dtype": "float32-le",
        "shape": list(dimensions),
        "numel": count,
        "sha256": checksum.hexdigest(),
        "stats": {
            "min": minimum,
            "max": maximum,
            "mean": total.value() / count,
            "l2_norm": math.sqrt(max(0.0, squared_total.value())),
        },
    }
    if ranked_values is not None:
        ranked_values.sort(key=lambda pair: (-pair[1], pair[0]))
        summary["top_k"] = [
            {"rank": rank, "token_id": token_id, "value": value}
            for rank, (token_id, value) in enumerate(
                ranked_values[: min(top_k or 0, count)], start=1
            )
        ]
    return summary


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise FixtureError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _read_json_object(line: str, *, location: str) -> dict[str, object]:
    try:
        value = json.loads(line, object_pairs_hook=_reject_duplicate_keys)
    except (json.JSONDecodeError, FixtureError) as error:
        raise PromptCorpusError(f"{location}: invalid JSON: {error}") from error
    if not isinstance(value, dict):
        raise PromptCorpusError(f"{location}: expected a JSON object")
    return value


def load_prompts(path: Path) -> tuple[list[PromptRecord], str]:
    """Load and strictly validate the PR 01 UTF-8 JSONL prompt corpus."""

    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise PromptCorpusError(f"cannot read UTF-8 prompt corpus {path}: {error}") from error

    prompts: list[PromptRecord] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        location = f"{path}:{line_number}"
        row = _read_json_object(line, location=location)
        if set(row) != _PROMPT_KEYS:
            missing = sorted(_PROMPT_KEYS - set(row))
            extra = sorted(set(row) - _PROMPT_KEYS)
            raise PromptCorpusError(
                f"{location}: prompt keys differ from contract; missing={missing}, extra={extra}"
            )
        if row["contract_version"] != PROMPT_CONTRACT_VERSION:
            raise PromptCorpusError(
                f"{location}: contract_version must be {PROMPT_CONTRACT_VERSION!r}"
            )
        prompt_id = row["prompt_id"]
        if not isinstance(prompt_id, str) or not prompt_id:
            raise PromptCorpusError(f"{location}: prompt_id must be a non-empty string")
        if not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", prompt_id):
            raise PromptCorpusError(f"{location}: prompt_id has invalid characters")
        if prompt_id in seen_ids:
            raise PromptCorpusError(f"{location}: duplicate prompt_id {prompt_id!r}")
        seen_ids.add(prompt_id)
        category = row["category"]
        language = row["language"]
        boundary_kind = row["boundary_kind"]
        if category not in _CATEGORIES:
            raise PromptCorpusError(f"{location}: unsupported category {category!r}")
        if language not in _LANGUAGES:
            raise PromptCorpusError(f"{location}: unsupported language {language!r}")
        if boundary_kind not in _BOUNDARY_KINDS:
            raise PromptCorpusError(
                f"{location}: unsupported boundary_kind {boundary_kind!r}"
            )
        prompt_text = row["text"]
        expected_behavior = row["expected_behavior"]
        if not isinstance(prompt_text, str):
            raise PromptCorpusError(f"{location}: text must be a string")
        if not isinstance(expected_behavior, str) or not expected_behavior:
            raise PromptCorpusError(
                f"{location}: expected_behavior must be a non-empty string"
            )
        target = row["target_prompt_tokens"]
        if target is not None and (
            isinstance(target, bool)
            or not isinstance(target, int)
            or not 1 <= target <= MAX_CONTEXT_TOKENS
        ):
            raise PromptCorpusError(
                f"{location}: target_prompt_tokens must be null or an integer in "
                f"[1, {MAX_CONTEXT_TOKENS}]"
            )
        if row["contains_sensitive_data"] is not False:
            raise PromptCorpusError(
                f"{location}: contains_sensitive_data must be false for version control"
            )
        if category == "context-boundary":
            if boundary_kind != "near-max-context" or target is None or target < 7168:
                raise PromptCorpusError(
                    f"{location}: context-boundary requires near-max-context and target >= 7168"
                )
        elif boundary_kind != "none":
            raise PromptCorpusError(
                f"{location}: only context-boundary may use a boundary_kind"
            )
        if category == "early-eos" and (
            not prompt_text
            or target is not None
            or language != "en"
            or expected_behavior != "greedy-eos-at-first-output-token"
        ):
            raise PromptCorpusError(
                f"{location}: early-eos must be a non-empty English prompt with "
                "target_prompt_tokens=null and "
                "expected_behavior='greedy-eos-at-first-output-token'"
            )
        prompts.append(
            PromptRecord(
                prompt_id=prompt_id,
                category=category,  # type: ignore[arg-type]
                language=language,  # type: ignore[arg-type]
                text=prompt_text,
                target_prompt_tokens=target,
                boundary_kind=boundary_kind,  # type: ignore[arg-type]
                expected_behavior=expected_behavior,
            )
        )
    if not prompts:
        raise PromptCorpusError(f"{path}: corpus contains no prompt rows")
    coverage = {
        "short English": any(
            prompt.category == "short" and prompt.language == "en" for prompt in prompts
        ),
        "short Korean/multilingual": any(
            prompt.category == "multilingual" and prompt.language in {"ko", "mixed"}
            for prompt in prompts
        ),
        "numbers, symbols, or code": any(
            prompt.category == "symbols-code" for prompt in prompts
        ),
        "long repetition": any(
            prompt.category == "long-repetition" for prompt in prompts
        ),
        "context boundary": any(
            prompt.category == "context-boundary" for prompt in prompts
        ),
        "minimal or empty": any(prompt.category == "minimal" for prompt in prompts),
        "early EOS": any(prompt.category == "early-eos" for prompt in prompts),
    }
    missing_coverage = [name for name, present in coverage.items() if not present]
    if missing_coverage:
        raise PromptCorpusError(
            f"{path}: corpus is missing required coverage: {', '.join(missing_coverage)}"
        )
    return prompts, hashlib.sha256(raw).hexdigest()


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _format_utc(value: datetime) -> str:
    if value.tzinfo is None:
        raise FixtureError("created_at must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise FixtureError(f"cannot hash fixture source {path}: {error}") from error
    return digest.hexdigest()


def _git(
    root: Path,
    arguments: Sequence[str],
    *,
    text: bool = False,
) -> bytes | str:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=root,
            check=True,
            capture_output=True,
            text=text,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise FixtureError(
            f"cannot resolve fixture Git provenance ({' '.join(arguments)}): {error}"
        ) from error
    return completed.stdout


def _validate_provenance(value: object, path: str = "fixture.provenance") -> None:
    provenance = _expect_object(value, path)
    _expect_exact_keys(
        provenance,
        {
            "git_revision",
            "git_tree",
            "git_dirty",
            "git_status_sha256",
            "environment_id",
            "observed_environment",
            "sources",
        },
        path,
    )
    for key in ("git_revision", "git_tree"):
        revision = _expect_string(provenance[key], f"{path}.{key}")
        if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
            raise FixtureValidationError(f"{path}.{key}: expected full lowercase Git SHA")
    if provenance["git_dirty"] is not False:
        raise FixtureValidationError(f"{path}.git_dirty: must be false")
    if provenance["git_status_sha256"] != _EMPTY_SHA256:
        raise FixtureValidationError(
            f"{path}.git_status_sha256: must bind an empty porcelain status"
        )
    if provenance["environment_id"] != PRIMARY_ENVIRONMENT_ID:
        raise FixtureValidationError(
            f"{path}.environment_id: must be {PRIMARY_ENVIRONMENT_ID!r}"
        )
    try:
        validate_environment_snapshot(
            provenance["observed_environment"], f"{path}.observed_environment"
        )
    except EnvironmentContractError as error:
        raise FixtureValidationError(str(error)) from error
    sources = _expect_object(provenance["sources"], f"{path}.sources")
    _expect_exact_keys(sources, set(FIXTURE_SOURCE_PATHS), f"{path}.sources")
    for name, relative in FIXTURE_SOURCE_PATHS.items():
        source = _expect_object(sources[name], f"{path}.sources.{name}")
        _expect_exact_keys(source, {"path", "sha256"}, f"{path}.sources.{name}")
        if source["path"] != relative:
            raise FixtureValidationError(
                f"{path}.sources.{name}.path: must be {relative!r}"
            )
        _expect_sha256(source["sha256"], f"{path}.sources.{name}.sha256")
    python_version_source = _expect_object(
        sources["python_version_file"], f"{path}.sources.python_version_file"
    )
    if python_version_source["sha256"] != PYTHON_VERSION_FILE_SHA256:
        raise FixtureValidationError(
            f"{path}.sources.python_version_file.sha256: immutable contract mismatch"
        )


def collect_fixture_provenance(
    repo_root: Path,
    *,
    observed_environment: Mapping[str, object] | None = None,
    environment_probe: EnvironmentProbe = probe_primary_environment,
) -> dict[str, object]:
    """Bind generation to a clean tracked checkout and every contract source byte."""

    root = repo_root.resolve()
    top_level_raw = _git(root, ["rev-parse", "--show-toplevel"], text=True)
    if not isinstance(top_level_raw, str):
        raise FixtureError("Git top-level query returned non-text output")
    top_level = Path(top_level_raw.strip()).resolve()
    if top_level != root:
        raise FixtureError(
            f"--repo-root must be the Git top level: expected {top_level}, got {root}"
        )
    revision_raw = _git(root, ["rev-parse", "HEAD"], text=True)
    tree_raw = _git(root, ["rev-parse", "HEAD^{tree}"], text=True)
    status = _git(
        root,
        ["status", "--porcelain=v1", "--untracked-files=all", "--", "."],
    )
    if not isinstance(revision_raw, str) or not isinstance(tree_raw, str):
        raise FixtureError("Git revision query returned non-text output")
    if not isinstance(status, bytes):
        raise FixtureError("Git status query returned text unexpectedly")
    revision = revision_raw.strip()
    tree = tree_raw.strip()
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None or re.fullmatch(
        r"[0-9a-f]{40}", tree
    ) is None:
        raise FixtureError("Git provenance requires full lowercase commit and tree SHAs")
    if status:
        preview = status.decode("utf-8", errors="replace").splitlines()[:5]
        raise FixtureError(
            "fixture generation requires a clean Git worktree including untracked "
            f"files; status={preview!r}"
        )

    sources: dict[str, dict[str, str]] = {}
    for name, relative in FIXTURE_SOURCE_PATHS.items():
        _git(root, ["ls-files", "--error-unmatch", "--", relative])
        path = (root / relative).resolve()
        if root not in path.parents:
            raise FixtureError(f"fixture source escapes repository root: {relative}")
        digest = _sha256_file(path)
        committed = _git(root, ["show", f"{revision}:{relative}"])
        if not isinstance(committed, bytes):
            raise FixtureError(f"Git source query returned text unexpectedly: {relative}")
        if hashlib.sha256(committed).hexdigest() != digest:
            raise FixtureError(f"fixture source differs from HEAD despite clean status: {relative}")
        sources[name] = {"path": relative, "sha256": digest}

    try:
        if observed_environment is None:
            observed_environment = environment_probe()
        validate_environment_snapshot(observed_environment)
    except EnvironmentContractError as error:
        raise FixtureError(f"primary environment preflight failed: {error}") from error
    provenance: dict[str, object] = {
        "git_revision": revision,
        "git_tree": tree,
        "git_dirty": False,
        "git_status_sha256": hashlib.sha256(status).hexdigest(),
        "environment_id": PRIMARY_ENVIRONMENT_ID,
        "observed_environment": observed_environment,
        "sources": sources,
    }
    _validate_provenance(provenance, "provenance")
    return provenance


def validate_fixture_against_repository(
    fixture: Mapping[str, object], repo_root: Path
) -> None:
    """Replay fixture source bindings against both the recorded commit and checkout."""

    validate_fixture(fixture)
    provenance = _expect_object(fixture["provenance"], "fixture.provenance")
    root = repo_root.resolve()
    top_level_raw = _git(root, ["rev-parse", "--show-toplevel"], text=True)
    if not isinstance(top_level_raw, str) or Path(top_level_raw.strip()).resolve() != root:
        raise FixtureValidationError("--repo-root must be the Git top level")
    revision = _expect_string(
        provenance["git_revision"], "fixture.provenance.git_revision"
    )
    recorded_tree = _expect_string(
        provenance["git_tree"], "fixture.provenance.git_tree"
    )
    actual_tree_raw = _git(root, ["rev-parse", f"{revision}^{{tree}}"], text=True)
    if not isinstance(actual_tree_raw, str) or actual_tree_raw.strip() != recorded_tree:
        raise FixtureValidationError("fixture.provenance.git_tree: commit tree mismatch")
    sources = _expect_object(provenance["sources"], "fixture.provenance.sources")
    for name, relative in FIXTURE_SOURCE_PATHS.items():
        source = _expect_object(sources[name], f"fixture.provenance.sources.{name}")
        digest = _expect_string(
            source["sha256"], f"fixture.provenance.sources.{name}.sha256"
        )
        path = (root / relative).resolve()
        if root not in path.parents or _sha256_file(path) != digest:
            raise FixtureValidationError(
                f"fixture.provenance.sources.{name}: current source bytes differ"
            )
        committed = _git(root, ["show", f"{revision}:{relative}"])
        if not isinstance(committed, bytes) or hashlib.sha256(committed).hexdigest() != digest:
            raise FixtureValidationError(
                f"fixture.provenance.sources.{name}: recorded commit bytes differ"
            )


def generate_fixture(
    prompts: Sequence[PromptRecord],
    corpus_sha256: str,
    backend: ReferenceBackend,
    *,
    max_new_tokens: int,
    hidden_state_index: int,
    top_k: int,
    seed: int,
    provenance: Mapping[str, object],
    created_at: datetime | None = None,
) -> dict[str, object]:
    """Generate a complete fixture, failing before output on any parity mismatch."""

    if not prompts:
        raise FixtureError("at least one prompt is required")
    if not _SHA256_RE.fullmatch(corpus_sha256):
        raise FixtureError("corpus_sha256 must be lowercase SHA-256 hex")
    for name, value in (
        ("max_new_tokens", max_new_tokens),
        ("hidden_state_index", hidden_state_index),
        ("top_k", top_k),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise FixtureError(f"{name} must be a positive integer")
    if max_new_tokens > GOLDEN_GREEDY_MAX_NEW_TOKENS:
        raise FixtureError(
            "max_new_tokens exceeds the predeclared exact BF16 cache-parity "
            f"window of {GOLDEN_GREEDY_MAX_NEW_TOKENS}"
        )
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= UINT64_MAX:
        raise FixtureError("seed must be an integer in [0, 2^64 - 1]")
    metadata = backend.metadata
    if metadata.python_version != PYTHON_VERSION:
        raise FixtureError(f"backend Python version must be {PYTHON_VERSION}")
    if metadata.python_executable_sha256 != PYTHON_EXECUTABLE_SHA256:
        raise FixtureError("backend Python executable checksum differs from contract")
    if (
        metadata.python_platform_system != PYTHON_PLATFORM_SYSTEM
        or metadata.python_platform_machine != PYTHON_PLATFORM_MACHINE
    ):
        raise FixtureError("backend Python platform differs from linux/x86_64 contract")
    if metadata.torch_version != TORCH_VERSION:
        raise FixtureError(f"backend torch version must be {TORCH_VERSION}")
    if metadata.transformers_version != TRANSFORMERS_VERSION:
        raise FixtureError(
            f"backend Transformers version must be {TRANSFORMERS_VERSION}"
        )
    if metadata.weights_sha256 != MODEL_WEIGHTS_SHA256:
        raise FixtureError("backend weight checksum differs from immutable contract")
    if metadata.config_sha256 != MODEL_CONFIG_SHA256:
        raise FixtureError("backend config checksum differs from immutable contract")
    if metadata.tokenizer_sha256 != TOKENIZER_SHA256:
        raise FixtureError("backend tokenizer aggregate differs from immutable contract")
    if dict(metadata.tokenizer_files_sha256) != TOKENIZER_FILES_SHA256:
        raise FixtureError("backend tokenizer file checksums differ from immutable contract")
    if metadata.device_name != PRIMARY_GPU_NAME:
        raise FixtureError(
            f"backend GPU must be {PRIMARY_GPU_NAME!r}, found {metadata.device_name!r}"
        )
    if metadata.compute_capability != PRIMARY_GPU_COMPUTE_CAPABILITY:
        raise FixtureError(
            "backend compute capability must be "
            f"{PRIMARY_GPU_COMPUTE_CAPABILITY}, found {metadata.compute_capability!r}"
        )
    _validate_provenance(provenance, "provenance")

    cases: list[dict[str, object]] = []
    for prompt in prompts:
        result = backend.generate_case(
            prompt.text,
            max_new_tokens=max_new_tokens,
            hidden_state_index=hidden_state_index,
            top_k=top_k,
            target_prompt_tokens=prompt.target_prompt_tokens,
        )
        if not result.input_token_ids:
            raise FixtureError(f"{prompt.prompt_id}: tokenizer produced no input tokens")
        if len(result.input_token_ids) + max_new_tokens > MAX_CONTEXT_TOKENS:
            raise FixtureError(
                f"{prompt.prompt_id}: {len(result.input_token_ids)} input + "
                f"{max_new_tokens} output tokens exceed context {MAX_CONTEXT_TOKENS}"
            )
        if result.cache_on_token_ids != result.cache_off_token_ids:
            mismatch = next(
                (
                    index
                    for index, pair in enumerate(
                        zip(
                            result.cache_on_token_ids,
                            result.cache_off_token_ids,
                            strict=False,
                        )
                    )
                    if pair[0] != pair[1]
                ),
                min(len(result.cache_on_token_ids), len(result.cache_off_token_ids)),
            )
            raise CacheParityError(
                f"{prompt.prompt_id}: cache on/off greedy tokens diverged "
                f"at output index {mismatch}"
            )
        if result.cache_on_stop_reason != result.cache_off_stop_reason:
            raise CacheParityError(
                f"{prompt.prompt_id}: cache stop reasons differ: "
                f"{result.cache_on_stop_reason!r} != {result.cache_off_stop_reason!r}"
            )
        if prompt.category == "early-eos" and (
            result.cache_on_stop_reason != "eos"
            or len(result.cache_on_token_ids) != 1
            or result.cache_on_token_ids[-1] not in backend.eos_token_ids
        ):
            raise FixtureError(
                f"{prompt.prompt_id}: early-eos must emit EOS at output index 0"
            )
        cases.append(
            {
                "prompt_id": prompt.prompt_id,
                "prompt_text_sha256": hashlib.sha256(
                    prompt.text.encode("utf-8")
                ).hexdigest(),
                "prompt_metadata": prompt.metadata,
                "input": {
                    "token_ids": list(result.input_token_ids),
                    "token_count": len(result.input_token_ids),
                },
                "hidden_state": dict(result.hidden_state),
                "final_logits": dict(result.final_logits),
                "processed_log_probs": dict(result.processed_log_probs),
                "rng": {
                    "stream_id": prompt.prompt_id,
                    "domain": "token-sampling",
                    "initial_snapshot": derive(
                        seed, prompt.prompt_id, "token-sampling"
                    ).snapshot(),
                    "draws_consumed": 0,
                },
                "greedy": {
                    "cache_on_token_ids": list(result.cache_on_token_ids),
                    "cache_off_token_ids": list(result.cache_off_token_ids),
                    "exact_match": True,
                    "stop_reason": result.cache_on_stop_reason,
                },
            }
        )

    fixture: dict[str, object] = {
        "schema_version": FIXTURE_SCHEMA_VERSION,
        "created_at": _format_utc(created_at or utc_now()),
        "generator": {
            "name": GENERATOR_NAME,
            "version": GENERATOR_VERSION,
            "python_version": metadata.python_version,
            "python_executable_sha256": metadata.python_executable_sha256,
            "python_platform_system": metadata.python_platform_system,
            "python_platform_machine": metadata.python_platform_machine,
            "torch_version": metadata.torch_version,
            "transformers_version": metadata.transformers_version,
            "backend": f"huggingface-transformers-{ATTENTION_BACKEND}",
            "device": metadata.device,
            "device_name": metadata.device_name,
            "compute_capability": metadata.compute_capability,
            "local_files_only": metadata.local_files_only,
        },
        "provenance": dict(provenance),
        "contract": {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "weights_sha256": metadata.weights_sha256,
            "config_sha256": metadata.config_sha256,
            "tokenizer_sha256": metadata.tokenizer_sha256,
            "tokenizer_files_sha256": dict(metadata.tokenizer_files_sha256),
            "dtype": DTYPE,
            "max_context_tokens": MAX_CONTEXT_TOKENS,
            "trust_remote_code": False,
            "attention_backend": ATTENTION_BACKEND,
            "determinism": {
                "torch_deterministic_algorithms": True,
                "cublas_workspace_config": ":4096:8",
                "tf32": False,
            },
            "python_version_range": PYTHON_VERSION_RANGE,
            "primary_gpu": {
                "name": PRIMARY_GPU_NAME,
                "compute_capability": PRIMARY_GPU_COMPUTE_CAPABILITY,
            },
            "primary_environment_id": PRIMARY_ENVIRONMENT_ID,
            "semantic_class": SEMANTIC_CLASS,
            "runtime_dependency_class": RUNTIME_DEPENDENCY_CLASS,
        },
        "corpus": {
            "contract_version": PROMPT_CONTRACT_VERSION,
            "sha256": corpus_sha256,
            "prompt_count": len(prompts),
        },
        "generation": {
            "strategy": "greedy",
            "max_new_tokens": max_new_tokens,
            "hidden_state_index": hidden_state_index,
            "hidden_state_selection": "full-prompt",
            "logits_position": "last-prompt-token",
            "top_k": top_k,
            "cache_modes": ["on", "off"],
            "eos_token_ids": list(backend.eos_token_ids),
        },
        "rng": {
            "algorithm_id": RNG_ALGORITHM,
            "master_seed": seed,
            "domain": "token-sampling",
            "greedy_draws_consumed": 0,
        },
        "cases": cases,
    }
    validate_fixture(fixture)
    return fixture


def _expect_object(value: object, path: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise FixtureValidationError(f"{path}: expected object")
    return value


def _expect_exact_keys(value: Mapping[str, object], keys: set[str], path: str) -> None:
    if set(value) != keys:
        raise FixtureValidationError(
            f"{path}: keys differ; missing={sorted(keys - set(value))}, "
            f"extra={sorted(set(value) - keys)}"
        )


def _expect_int(value: object, path: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise FixtureValidationError(f"{path}: expected integer >= {minimum}")
    return value


def _expect_string(value: object, path: str, *, nonempty: bool = True) -> str:
    if not isinstance(value, str) or (nonempty and not value):
        raise FixtureValidationError(f"{path}: expected string")
    return value


def _expect_sha256(value: object, path: str, *, nullable: bool = False) -> None:
    if nullable and value is None:
        return
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise FixtureValidationError(f"{path}: expected lowercase SHA-256 hex")


def _expect_number(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FixtureValidationError(f"{path}: expected finite number")
    result = float(value)
    if not math.isfinite(result):
        raise FixtureValidationError(f"{path}: expected finite number")
    return result


def _validate_summary(
    value: object, path: str, *, expected_top_k: int | None
) -> int:
    summary = _expect_object(value, path)
    keys = {"canonical_dtype", "shape", "numel", "sha256", "stats"}
    if expected_top_k is not None:
        keys.add("top_k")
    _expect_exact_keys(summary, keys, path)
    if summary["canonical_dtype"] != "float32-le":
        raise FixtureValidationError(f"{path}.canonical_dtype: must be 'float32-le'")
    shape = summary["shape"]
    if not isinstance(shape, list) or not shape:
        raise FixtureValidationError(f"{path}.shape: expected non-empty array")
    dimensions = [
        _expect_int(dimension, f"{path}.shape[{index}]", minimum=1)
        for index, dimension in enumerate(shape)
    ]
    numel = _expect_int(summary["numel"], f"{path}.numel", minimum=1)
    if math.prod(dimensions) != numel:
        raise FixtureValidationError(f"{path}: shape product does not equal numel")
    _expect_sha256(summary["sha256"], f"{path}.sha256")
    stats = _expect_object(summary["stats"], f"{path}.stats")
    _expect_exact_keys(stats, {"min", "max", "mean", "l2_norm"}, f"{path}.stats")
    minimum = _expect_number(stats["min"], f"{path}.stats.min")
    maximum = _expect_number(stats["max"], f"{path}.stats.max")
    mean = _expect_number(stats["mean"], f"{path}.stats.mean")
    l2_norm = _expect_number(stats["l2_norm"], f"{path}.stats.l2_norm")
    if minimum > maximum or not minimum <= mean <= maximum or l2_norm < 0:
        raise FixtureValidationError(f"{path}.stats: inconsistent bounds")

    if expected_top_k is not None:
        entries = summary["top_k"]
        if not isinstance(entries, list) or len(entries) != min(expected_top_k, numel):
            raise FixtureValidationError(f"{path}.top_k: unexpected entry count")
        previous: tuple[float, int] | None = None
        token_ids: set[int] = set()
        for index, raw_entry in enumerate(entries):
            entry_path = f"{path}.top_k[{index}]"
            entry = _expect_object(raw_entry, entry_path)
            _expect_exact_keys(entry, {"rank", "token_id", "value"}, entry_path)
            rank = _expect_int(entry["rank"], f"{entry_path}.rank", minimum=1)
            token_id = _expect_int(entry["token_id"], f"{entry_path}.token_id")
            item_value = _expect_number(entry["value"], f"{entry_path}.value")
            if rank != index + 1 or token_id >= numel or token_id in token_ids:
                raise FixtureValidationError(f"{entry_path}: invalid rank or token_id")
            token_ids.add(token_id)
            ordering = (-item_value, token_id)
            if previous is not None and ordering < previous:
                raise FixtureValidationError(
                    f"{path}.top_k: entries are not deterministically sorted"
                )
            previous = ordering
            if not minimum <= item_value <= maximum:
                raise FixtureValidationError(f"{entry_path}.value: outside summary bounds")
    return numel


def _validate_token_ids(value: object, path: str, *, allow_empty: bool) -> list[int]:
    if not isinstance(value, list) or (not allow_empty and not value):
        description = "an array" if allow_empty else "a non-empty array"
        raise FixtureValidationError(f"{path}: expected {description}")
    return [
        _expect_int(token_id, f"{path}[{index}]")
        for index, token_id in enumerate(value)
    ]


def validate_fixture(value: object) -> None:
    """Strictly validate schema shape plus cross-field semantic invariants."""

    root = _expect_object(value, "fixture")
    _expect_exact_keys(
        root,
        {
            "schema_version",
            "created_at",
            "generator",
            "provenance",
            "contract",
            "corpus",
            "generation",
            "rng",
            "cases",
        },
        "fixture",
    )
    if root["schema_version"] != FIXTURE_SCHEMA_VERSION:
        raise FixtureValidationError("fixture.schema_version: unsupported version")
    created_at = _expect_string(root["created_at"], "fixture.created_at")
    try:
        parsed_created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError as error:
        raise FixtureValidationError("fixture.created_at: invalid RFC 3339 timestamp") from error
    if (
        not created_at.endswith("Z")
        or parsed_created_at.utcoffset() != timezone.utc.utcoffset(None)
    ):
        raise FixtureValidationError("fixture.created_at: must be UTC and end in 'Z'")

    generator = _expect_object(root["generator"], "fixture.generator")
    _expect_exact_keys(
        generator,
        {
            "name",
            "version",
            "python_version",
            "python_executable_sha256",
            "python_platform_system",
            "python_platform_machine",
            "torch_version",
            "transformers_version",
            "backend",
            "device",
            "device_name",
            "compute_capability",
            "local_files_only",
        },
        "fixture.generator",
    )
    expected_generator = {
        "name": GENERATOR_NAME,
        "version": GENERATOR_VERSION,
        "torch_version": TORCH_VERSION,
        "transformers_version": TRANSFORMERS_VERSION,
        "backend": f"huggingface-transformers-{ATTENTION_BACKEND}",
    }
    for key, expected in expected_generator.items():
        if generator[key] != expected:
            raise FixtureValidationError(f"fixture.generator.{key}: must be {expected!r}")
    python_version = _expect_string(
        generator["python_version"], "fixture.generator.python_version"
    )
    if python_version != PYTHON_VERSION:
        raise FixtureValidationError(
            f"fixture.generator.python_version: must be {PYTHON_VERSION}"
        )
    if generator["python_executable_sha256"] != PYTHON_EXECUTABLE_SHA256:
        raise FixtureValidationError(
            "fixture.generator.python_executable_sha256: immutable mismatch"
        )
    if (
        generator["python_platform_system"] != PYTHON_PLATFORM_SYSTEM
        or generator["python_platform_machine"] != PYTHON_PLATFORM_MACHINE
    ):
        raise FixtureValidationError("fixture.generator: Python platform mismatch")
    _expect_string(generator["device"], "fixture.generator.device")
    if generator["device_name"] != PRIMARY_GPU_NAME:
        raise FixtureValidationError(
            f"fixture.generator.device_name: must be {PRIMARY_GPU_NAME!r}"
        )
    if generator["compute_capability"] != PRIMARY_GPU_COMPUTE_CAPABILITY:
        raise FixtureValidationError(
            "fixture.generator.compute_capability: must be "
            f"{PRIMARY_GPU_COMPUTE_CAPABILITY!r}"
        )
    if not isinstance(generator["local_files_only"], bool):
        raise FixtureValidationError("fixture.generator.local_files_only: expected boolean")

    _validate_provenance(root["provenance"])

    contract = _expect_object(root["contract"], "fixture.contract")
    _expect_exact_keys(
        contract,
        {
            "model_id",
            "model_revision",
            "weights_sha256",
            "config_sha256",
            "tokenizer_sha256",
            "tokenizer_files_sha256",
            "dtype",
            "max_context_tokens",
            "trust_remote_code",
            "attention_backend",
            "determinism",
            "python_version_range",
            "primary_gpu",
            "primary_environment_id",
            "semantic_class",
            "runtime_dependency_class",
        },
        "fixture.contract",
    )
    expected_contract = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "dtype": DTYPE,
        "max_context_tokens": MAX_CONTEXT_TOKENS,
        "trust_remote_code": False,
        "attention_backend": ATTENTION_BACKEND,
        "python_version_range": PYTHON_VERSION_RANGE,
        "semantic_class": SEMANTIC_CLASS,
        "runtime_dependency_class": RUNTIME_DEPENDENCY_CLASS,
        "primary_environment_id": PRIMARY_ENVIRONMENT_ID,
    }
    for key, expected in expected_contract.items():
        if contract[key] != expected:
            raise FixtureValidationError(f"fixture.contract.{key}: must be {expected!r}")
    determinism = _expect_object(
        contract["determinism"], "fixture.contract.determinism"
    )
    expected_determinism = {
        "torch_deterministic_algorithms": True,
        "cublas_workspace_config": ":4096:8",
        "tf32": False,
    }
    if determinism != expected_determinism:
        raise FixtureValidationError(
            "fixture.contract.determinism: immutable contract mismatch"
        )
    if contract["weights_sha256"] != MODEL_WEIGHTS_SHA256:
        raise FixtureValidationError(
            f"fixture.contract.weights_sha256: must be {MODEL_WEIGHTS_SHA256!r}"
        )
    if contract["config_sha256"] != MODEL_CONFIG_SHA256:
        raise FixtureValidationError(
            f"fixture.contract.config_sha256: must be {MODEL_CONFIG_SHA256!r}"
        )
    if contract["tokenizer_sha256"] != TOKENIZER_SHA256:
        raise FixtureValidationError(
            f"fixture.contract.tokenizer_sha256: must be {TOKENIZER_SHA256!r}"
        )
    tokenizer_files = _expect_object(
        contract["tokenizer_files_sha256"],
        "fixture.contract.tokenizer_files_sha256",
    )
    _expect_exact_keys(
        tokenizer_files,
        set(TOKENIZER_ARTIFACT_FILENAMES),
        "fixture.contract.tokenizer_files_sha256",
    )
    if tokenizer_files != TOKENIZER_FILES_SHA256:
        raise FixtureValidationError(
            "fixture.contract.tokenizer_files_sha256: immutable artifact mismatch"
        )
    primary_gpu = _expect_object(contract["primary_gpu"], "fixture.contract.primary_gpu")
    _expect_exact_keys(primary_gpu, {"name", "compute_capability"}, "fixture.contract.primary_gpu")
    if primary_gpu != {
        "name": PRIMARY_GPU_NAME,
        "compute_capability": PRIMARY_GPU_COMPUTE_CAPABILITY,
    }:
        raise FixtureValidationError("fixture.contract.primary_gpu: immutable contract mismatch")

    corpus = _expect_object(root["corpus"], "fixture.corpus")
    _expect_exact_keys(corpus, {"contract_version", "sha256", "prompt_count"}, "fixture.corpus")
    if corpus["contract_version"] != PROMPT_CONTRACT_VERSION:
        raise FixtureValidationError("fixture.corpus.contract_version: unsupported version")
    _expect_sha256(corpus["sha256"], "fixture.corpus.sha256")
    prompt_count = _expect_int(corpus["prompt_count"], "fixture.corpus.prompt_count", minimum=1)

    generation = _expect_object(root["generation"], "fixture.generation")
    _expect_exact_keys(
        generation,
        {
            "strategy",
            "max_new_tokens",
            "hidden_state_index",
            "hidden_state_selection",
            "logits_position",
            "top_k",
            "cache_modes",
            "eos_token_ids",
        },
        "fixture.generation",
    )
    if generation["strategy"] != "greedy" or generation["cache_modes"] != ["on", "off"]:
        raise FixtureValidationError("fixture.generation: only greedy cache on/off is supported")
    max_new_tokens = _expect_int(
        generation["max_new_tokens"], "fixture.generation.max_new_tokens", minimum=1
    )
    if max_new_tokens > GOLDEN_GREEDY_MAX_NEW_TOKENS:
        raise FixtureValidationError(
            "fixture.generation.max_new_tokens: exceeds exact BF16 cache-parity "
            f"window of {GOLDEN_GREEDY_MAX_NEW_TOKENS}"
        )
    hidden_state_index = _expect_int(
        generation["hidden_state_index"], "fixture.generation.hidden_state_index", minimum=1
    )
    if generation["hidden_state_selection"] != "full-prompt":
        raise FixtureValidationError(
            "fixture.generation.hidden_state_selection: must be 'full-prompt'"
        )
    if generation["logits_position"] != "last-prompt-token":
        raise FixtureValidationError(
            "fixture.generation.logits_position: must be 'last-prompt-token'"
        )
    top_k = _expect_int(generation["top_k"], "fixture.generation.top_k", minimum=1)
    eos_token_ids = set(
        _validate_token_ids(
            generation["eos_token_ids"],
            "fixture.generation.eos_token_ids",
            allow_empty=True,
        )
    )

    rng = _expect_object(root["rng"], "fixture.rng")
    _expect_exact_keys(
        rng,
        {"algorithm_id", "master_seed", "domain", "greedy_draws_consumed"},
        "fixture.rng",
    )
    if rng["algorithm_id"] != RNG_ALGORITHM:
        raise FixtureValidationError("fixture.rng.algorithm_id: unsupported RNG")
    seed = _expect_int(rng["master_seed"], "fixture.rng.master_seed")
    if seed > UINT64_MAX:
        raise FixtureValidationError("fixture.rng.master_seed: exceeds uint64")
    if rng["domain"] != "token-sampling":
        raise FixtureValidationError("fixture.rng.domain: must be 'token-sampling'")
    if rng["greedy_draws_consumed"] != 0:
        raise FixtureValidationError(
            "fixture.rng.greedy_draws_consumed: greedy generation must consume 0"
        )

    cases = root["cases"]
    if not isinstance(cases, list) or len(cases) != prompt_count:
        raise FixtureValidationError("fixture.cases: count differs from corpus.prompt_count")
    seen_prompt_ids: set[str] = set()
    for index, raw_case in enumerate(cases):
        path = f"fixture.cases[{index}]"
        case = _expect_object(raw_case, path)
        _expect_exact_keys(
            case,
            {
                "prompt_id",
                "prompt_text_sha256",
                "prompt_metadata",
                "input",
                "hidden_state",
                "final_logits",
                "processed_log_probs",
                "rng",
                "greedy",
            },
            path,
        )
        prompt_id = _expect_string(case["prompt_id"], f"{path}.prompt_id")
        if prompt_id in seen_prompt_ids:
            raise FixtureValidationError(f"{path}.prompt_id: duplicate {prompt_id!r}")
        seen_prompt_ids.add(prompt_id)
        _expect_sha256(case["prompt_text_sha256"], f"{path}.prompt_text_sha256")
        prompt_metadata = _expect_object(case["prompt_metadata"], f"{path}.prompt_metadata")
        _expect_exact_keys(
            prompt_metadata,
            {
                "category",
                "language",
                "target_prompt_tokens",
                "boundary_kind",
                "expected_behavior",
            },
            f"{path}.prompt_metadata",
        )
        if prompt_metadata["category"] not in _CATEGORIES:
            raise FixtureValidationError(f"{path}.prompt_metadata.category: unsupported value")
        if prompt_metadata["language"] not in _LANGUAGES:
            raise FixtureValidationError(f"{path}.prompt_metadata.language: unsupported value")
        if prompt_metadata["boundary_kind"] not in _BOUNDARY_KINDS:
            raise FixtureValidationError(f"{path}.prompt_metadata.boundary_kind: unsupported value")
        _expect_string(
            prompt_metadata["expected_behavior"], f"{path}.prompt_metadata.expected_behavior"
        )
        target = prompt_metadata["target_prompt_tokens"]
        if target is not None:
            _expect_int(
                target,
                f"{path}.prompt_metadata.target_prompt_tokens",
                minimum=1,
            )

        input_object = _expect_object(case["input"], f"{path}.input")
        _expect_exact_keys(input_object, {"token_ids", "token_count"}, f"{path}.input")
        input_ids = _validate_token_ids(
            input_object["token_ids"], f"{path}.input.token_ids", allow_empty=False
        )
        if input_object["token_count"] != len(input_ids):
            raise FixtureValidationError(f"{path}.input.token_count: inconsistent")
        if target is not None and len(input_ids) != target:
            raise FixtureValidationError(
                f"{path}.input.token_count: differs from target_prompt_tokens"
            )
        _validate_summary(case["hidden_state"], f"{path}.hidden_state", expected_top_k=None)
        hidden_summary = _expect_object(case["hidden_state"], f"{path}.hidden_state")
        hidden_shape = hidden_summary["shape"]
        if (
            not isinstance(hidden_shape, list)
            or len(hidden_shape) != 3
            or hidden_shape[0] != 1
            or hidden_shape[1] != len(input_ids)
        ):
            raise FixtureValidationError(
                f"{path}.hidden_state.shape: expected [1, input_tokens, hidden_size]"
            )
        _validate_summary(case["final_logits"], f"{path}.final_logits", expected_top_k=top_k)
        final_logits_summary = _expect_object(case["final_logits"], f"{path}.final_logits")
        if not isinstance(final_logits_summary["shape"], list) or len(
            final_logits_summary["shape"]
        ) != 1:
            raise FixtureValidationError(f"{path}.final_logits.shape: expected [vocab_size]")
        processed = _expect_object(
            case["processed_log_probs"], f"{path}.processed_log_probs"
        )
        _expect_exact_keys(
            processed,
            {"pipeline_id", "tensor"},
            f"{path}.processed_log_probs",
        )
        if processed["pipeline_id"] != "log-softmax-fp32-v1":
            raise FixtureValidationError(
                f"{path}.processed_log_probs.pipeline_id: unsupported pipeline"
            )
        logits_numel = _validate_summary(
            processed["tensor"],
            f"{path}.processed_log_probs.tensor",
            expected_top_k=top_k,
        )
        raw_logits = _expect_object(case["final_logits"], f"{path}.final_logits")
        if logits_numel != raw_logits["numel"]:
            raise FixtureValidationError(
                f"{path}.processed_log_probs: vocabulary size differs from raw logits"
            )

        case_rng = _expect_object(case["rng"], f"{path}.rng")
        _expect_exact_keys(
            case_rng,
            {"stream_id", "domain", "initial_snapshot", "draws_consumed"},
            f"{path}.rng",
        )
        if case_rng["stream_id"] != prompt_id or case_rng["domain"] != "token-sampling":
            raise FixtureValidationError(f"{path}.rng: stream/domain mismatch")
        if case_rng["draws_consumed"] != 0:
            raise FixtureValidationError(f"{path}.rng: greedy path consumed RNG words")
        expected_snapshot = derive(seed, prompt_id, "token-sampling").snapshot()
        try:
            restored = Philox4x32.restore(case_rng["initial_snapshot"])
        except (TypeError, ValueError) as error:
            raise FixtureValidationError(f"{path}.rng.initial_snapshot: {error}") from error
        if restored.snapshot() != expected_snapshot:
            raise FixtureValidationError(
                f"{path}.rng.initial_snapshot: derivation identity mismatch"
            )

        greedy = _expect_object(case["greedy"], f"{path}.greedy")
        _expect_exact_keys(
            greedy,
            {"cache_on_token_ids", "cache_off_token_ids", "exact_match", "stop_reason"},
            f"{path}.greedy",
        )
        cache_on = _validate_token_ids(
            greedy["cache_on_token_ids"],
            f"{path}.greedy.cache_on_token_ids",
            allow_empty=True,
        )
        cache_off = _validate_token_ids(
            greedy["cache_off_token_ids"],
            f"{path}.greedy.cache_off_token_ids",
            allow_empty=True,
        )
        if greedy["exact_match"] is not True or cache_on != cache_off:
            raise FixtureValidationError(f"{path}.greedy: cache parity failed")
        if len(cache_on) > max_new_tokens or len(input_ids) + len(cache_on) > MAX_CONTEXT_TOKENS:
            raise FixtureValidationError(f"{path}.greedy: token count exceeds contract")
        stop_reason = greedy["stop_reason"]
        if stop_reason == "eos":
            if not cache_on or cache_on[-1] not in eos_token_ids:
                raise FixtureValidationError(f"{path}.greedy: EOS stop lacks EOS token")
        elif stop_reason == "max_new_tokens":
            if len(cache_on) != max_new_tokens:
                raise FixtureValidationError(f"{path}.greedy: max token stop has wrong length")
        else:
            raise FixtureValidationError(f"{path}.greedy.stop_reason: unsupported value")
        if prompt_metadata["category"] == "early-eos" and (
            stop_reason != "eos" or len(cache_on) != 1
        ):
            raise FixtureValidationError(
                f"{path}.greedy: early-eos must emit EOS at output index 0"
            )


def load_fixture(path: Path) -> dict[str, object]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle, object_pairs_hook=_reject_duplicate_keys)
    except (OSError, json.JSONDecodeError, FixtureError) as error:
        raise FixtureValidationError(f"cannot read fixture {path}: {error}") from error
    if not isinstance(value, dict):
        raise FixtureValidationError(f"{path}: expected a JSON object")
    validate_fixture(value)
    return value


def validate_fixture_against_prompts(
    fixture: Mapping[str, object],
    prompts: Sequence[PromptRecord],
    corpus_sha256: str,
) -> None:
    """Bind a structurally valid fixture to the current source prompt bytes."""

    validate_fixture(fixture)
    corpus = _expect_object(fixture["corpus"], "fixture.corpus")
    if corpus["sha256"] != corpus_sha256:
        raise FixtureValidationError(
            "fixture.corpus.sha256: differs from the supplied prompt corpus"
        )
    if corpus["prompt_count"] != len(prompts):
        raise FixtureValidationError(
            "fixture.corpus.prompt_count: differs from the supplied prompt corpus"
        )
    cases = fixture["cases"]
    if not isinstance(cases, list):
        raise FixtureValidationError("fixture.cases: expected array")
    for index, (raw_case, prompt) in enumerate(zip(cases, prompts, strict=True)):
        path = f"fixture.cases[{index}]"
        case = _expect_object(raw_case, path)
        if case["prompt_id"] != prompt.prompt_id:
            raise FixtureValidationError(
                f"{path}.prompt_id: differs from supplied prompt order"
            )
        expected_text_sha256 = hashlib.sha256(prompt.text.encode("utf-8")).hexdigest()
        if case["prompt_text_sha256"] != expected_text_sha256:
            raise FixtureValidationError(
                f"{path}.prompt_text_sha256: differs from supplied prompt text"
            )
        if case["prompt_metadata"] != prompt.metadata:
            raise FixtureValidationError(
                f"{path}.prompt_metadata: differs from supplied prompt metadata"
            )


def write_fixture_exclusive(path: Path, fixture: Mapping[str, object]) -> None:
    """Create a fixture without ever replacing an existing path."""

    if path.exists():
        raise FixtureError(f"refusing to overwrite existing fixture: {path}")
    validate_fixture(fixture)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as error:
        raise FixtureError(f"refusing to overwrite existing fixture: {path}") from error
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(fixture, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink()
        except OSError:
            pass
        raise
