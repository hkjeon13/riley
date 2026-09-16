"""Offline-only Qwen2.5-3B teacher-forced eager-generation oracle.

This module is deliberately separate from the Rust serving process.  It builds
one source-bound Hugging Face artifact for a fixed P2048 prompt.  The cache-off
path first derives the 128-token teacher stream using the addressable greedy
argmax.  The cache-on path then consumes that same stream: its first decode
call consumes ``teacher_token_ids[0]`` at absolute position 2048, never 2049.

The artifact stores raw BF16 last-logit rows in two safetensors sidecars.  Its
``validate`` command uses only the Python standard library plus the existing
lightweight reference modules; it never imports Torch, Transformers, or
safetensors and never opens CUDA.
"""

from __future__ import annotations

import argparse
from array import array
import hashlib
import heapq
import json
import math
import os
import struct
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from . import qwen3b_serving_oracle as oracle
from .calibration import CalibrationError
from .hf_calibration import (
    SidecarWriter,
    _default_sidecar_writer,
    _write_sidecar_exclusive,
)

SCHEMA_VERSION = "riley.qwen3b-hf-eager-teacher-forced-generation.v1"
ARTIFACT_KIND = "qwen2.5-3b-hf-eager-bf16-p2048-teacher-forced-generation"
IMPLEMENTATION_ID = "riley-python-qwen3b-hf-eager-teacher-forced-generation-v1"

CACHE_OFF = "cache-off"
CACHE_ON = "cache-on"
CACHE_MODES = (CACHE_OFF, CACHE_ON)
SIDECAR_TENSOR_KEY = "teacher_forced/logits"
BF16_BYTES = 2
MAX_SAFETENSORS_HEADER_BYTES = 1_048_576

SOURCE_PATHS = {
    "qwen_serving_oracle": "tools/python/reference/riley_reference/qwen3b_serving_oracle.py",
    "generation_trace": "tools/python/reference/riley_reference/qwen3b_generation_trace.py",
    "hf_calibration": "tools/python/reference/riley_reference/hf_calibration.py",
    "reference_project": "tools/python/reference/pyproject.toml",
    "reference_lock": "tools/python/reference/uv.lock",
}

BackendFactory = Callable[..., "HuggingFaceQwen3BGenerationTraceBackend"]
SourceProvenanceFactory = Callable[[Path], dict[str, object]]


class Qwen3BGenerationTraceError(oracle.Qwen3BServingOracleError):
    """The fixed teacher-forced generation contract cannot be satisfied."""


@dataclass(frozen=True)
class StepPlan:
    """One exact model call in the cache-free or cached teacher-forced path."""

    mode: str
    step: int
    call_input_token_ids: tuple[int, ...]
    context_token_count: int
    attention_mask_token_count: int
    position_start: int
    position_end: int
    teacher_input_token_id: int | None
    cache_length_before: int | None
    cache_length_after: int | None


@dataclass(frozen=True)
class CapturedLogits:
    """One CPU BF16 logit row and its replayable compact metadata."""

    tensor: object
    raw_bf16_le_sha256: str
    addressable_bf16_le_sha256: str
    non_addressable_bf16_le_sha256: str
    raw_argmax_token_id: int
    selected_token_id: int
    selected_logit_bf16_as_f32: float
    top_token_ids: tuple[int, ...]
    top_values_bf16_as_f32: tuple[float, ...]


@dataclass(frozen=True)
class Bf16LogitAnalysis:
    """Stable, stdlib-decoded summary of one canonical BF16 logit row."""

    raw_argmax_token_id: int
    selected_token_id: int
    selected_logit_bf16_as_f32: float
    top_token_ids: tuple[int, ...]
    top_values_bf16_as_f32: tuple[float, ...]


@dataclass(frozen=True)
class CapturedStep:
    plan: StepPlan
    logits: CapturedLogits


@dataclass(frozen=True)
class GenerationCapture:
    """Both fixed paths plus the cache-off-derived teacher token sequence."""

    teacher_token_ids: tuple[int, ...]
    cache_off_steps: tuple[CapturedStep, ...]
    cache_on_steps: tuple[CapturedStep, ...]
    cache_off_logits: object
    cache_on_logits: object


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return oracle._sha256_file(path)


def _canonical_json_bytes(document: Mapping[str, object]) -> bytes:
    return oracle._canonical_json_bytes(document)


def _require_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise Qwen3BGenerationTraceError(f"{label} must be an object")
    return value


def _require_exact_keys(
    value: Mapping[str, object], expected: set[str], label: str
) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        raise Qwen3BGenerationTraceError(
            f"{label} fields differ: missing={missing!r} extra={extra!r}"
        )


def _require_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise Qwen3BGenerationTraceError(f"{label} must be a non-empty string")
    return value


def _require_sha256(value: object, label: str) -> str:
    try:
        return oracle._require_sha256(value, label)
    except oracle.Qwen3BServingOracleError as error:
        raise Qwen3BGenerationTraceError(str(error)) from error


def _require_int(
    value: object,
    label: str,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise Qwen3BGenerationTraceError(f"{label} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        raise Qwen3BGenerationTraceError(f"{label} is outside the allowed range")
    return value


def _require_finite_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Qwen3BGenerationTraceError(f"{label} must be a finite JSON number")
    result = float(value)
    if not math.isfinite(result):
        raise Qwen3BGenerationTraceError(f"{label} must be a finite JSON number")
    return result


def _require_u32_ids(
    value: object,
    *,
    label: str,
    count: int,
    upper_bound: int,
) -> tuple[int, ...]:
    if not isinstance(value, list) or len(value) != count:
        raise Qwen3BGenerationTraceError(
            f"{label} must contain exactly {count} token IDs"
        )
    result: list[int] = []
    for index, token_id in enumerate(value):
        result.append(
            _require_int(
                token_id,
                f"{label}[{index}]",
                maximum=upper_bound - 1,
            )
        )
    return tuple(result)


def _validate_teacher_token_ids(token_ids: Sequence[int]) -> tuple[int, ...]:
    result = tuple(token_ids)
    if len(result) != oracle.OUTPUT_TOKEN_COUNT:
        raise Qwen3BGenerationTraceError(
            "teacher token stream must contain the fixed 128 generated tokens"
        )
    for index, token_id in enumerate(result):
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise Qwen3BGenerationTraceError(
                f"teacher token {index} must be an integer"
            )
        if not 0 <= token_id < oracle.ADDRESSABLE_TOKEN_COUNT:
            raise Qwen3BGenerationTraceError(
                f"teacher token {index} is outside the addressable vocabulary"
            )
    return result


_BF16_EXPONENT_MASK = 0x7F80
_BF16_SIGN_MASK = 0x8000
_F32_BIG_ENDIAN = struct.Struct(">f")


def _bf16_numeric_sort_key(bits: int) -> int:
    """Map one finite BF16 bit pattern to a monotone numeric sort key.

    IEEE BF16 has two zero encodings.  They compare numerically equal, so both
    are normalized to +0; token ID supplies the documented deterministic tie
    break after this key.
    """

    if bits == _BF16_SIGN_MASK:
        bits = 0
    if bits & _BF16_SIGN_MASK:
        return (~bits) & 0xFFFF
    return bits | _BF16_SIGN_MASK


def _bf16_bits_as_f32(bits: int) -> float:
    value = _F32_BIG_ENDIAN.unpack((bits << 16).to_bytes(4, "big"))[0]
    if not math.isfinite(value):
        raise Qwen3BGenerationTraceError("BF16 conversion produced a non-finite value")
    return value


def _analyze_bf16_le_logits(raw: bytes) -> Bf16LogitAnalysis:
    """Decode one canonical raw BF16 row with a stable numerical tie rule.

    The rank is larger BF16 numeric value followed by lower token ID.  It is
    deliberately independent of Torch so the producer and post-write sidecar
    validator make the same metadata claim from the exact bytes on disk.
    """

    if len(raw) != oracle.RAW_LOGIT_BYTES:
        raise Qwen3BGenerationTraceError("raw BF16 logit row byte count differs")
    words = array("H")
    words.frombytes(raw)
    if words.itemsize != BF16_BYTES or len(words) != oracle.MODEL_VOCABULARY_SIZE:
        raise Qwen3BGenerationTraceError("raw BF16 logit row element count differs")
    if sys.byteorder == "big":
        words.byteswap()
    elif sys.byteorder != "little":
        raise Qwen3BGenerationTraceError("unsupported host byte order")

    raw_best: tuple[int, int, int] | None = None
    selected_best: tuple[int, int, int] | None = None
    top_entries: list[tuple[int, int, int, int]] = []
    for token_id, bits in enumerate(words):
        if bits & _BF16_EXPONENT_MASK == _BF16_EXPONENT_MASK:
            raise Qwen3BGenerationTraceError(
                f"raw BF16 logit at token {token_id} is non-finite"
            )
        rank = (_bf16_numeric_sort_key(bits), -token_id, token_id)
        if raw_best is None or rank[:2] > raw_best[:2]:
            raw_best = rank
        if token_id >= oracle.ADDRESSABLE_TOKEN_COUNT:
            continue
        if selected_best is None or rank[:2] > selected_best[:2]:
            selected_best = rank
        entry = (rank[0], rank[1], token_id, bits)
        if len(top_entries) < oracle.TOP_K:
            heapq.heappush(top_entries, entry)
        elif entry[:2] > top_entries[0][:2]:
            heapq.heapreplace(top_entries, entry)

    if raw_best is None or selected_best is None or len(top_entries) != oracle.TOP_K:
        raise Qwen3BGenerationTraceError("raw BF16 logit row lacks the required tokens")
    top_entries.sort(key=lambda entry: (-entry[0], entry[2]))
    top_token_ids = tuple(entry[2] for entry in top_entries)
    top_values = tuple(_bf16_bits_as_f32(entry[3]) for entry in top_entries)
    if top_token_ids[0] != selected_best[2]:
        raise Qwen3BGenerationTraceError("raw BF16 selected token differs from top-k")
    return Bf16LogitAnalysis(
        raw_argmax_token_id=raw_best[2],
        selected_token_id=selected_best[2],
        selected_logit_bf16_as_f32=top_values[0],
        top_token_ids=top_token_ids,
        top_values_bf16_as_f32=top_values,
    )


def teacher_forced_step_plan(
    mode: str, teacher_token_ids: Sequence[int]
) -> tuple[StepPlan, ...]:
    """Return the immutable call/position plan without importing ML libraries.

    For cache-on, row zero pre-fills the prompt.  Row ``i > 0`` contains one
    input token, ``teacher_token_ids[i - 1]``, and evaluates it at absolute
    position ``PROMPT_TOKEN_COUNT + i - 1``.  This is intentionally exposed as
    a pure helper so the 2048 first-decode boundary stays unit-tested.
    """

    if mode not in CACHE_MODES:
        raise Qwen3BGenerationTraceError(f"unknown cache mode {mode!r}")
    teacher = _validate_teacher_token_ids(teacher_token_ids)
    prompt = (oracle.EXPECTED_PROMPT_TOKEN_ID,) * oracle.PROMPT_TOKEN_COUNT
    plans: list[StepPlan] = []
    for step in range(oracle.OUTPUT_TOKEN_COUNT):
        context_token_count = oracle.PROMPT_TOKEN_COUNT + step
        if mode == CACHE_OFF:
            call_input = prompt + teacher[:step]
            plan = StepPlan(
                mode=mode,
                step=step,
                call_input_token_ids=call_input,
                context_token_count=context_token_count,
                attention_mask_token_count=context_token_count,
                position_start=0,
                position_end=context_token_count - 1,
                teacher_input_token_id=None,
                cache_length_before=None,
                cache_length_after=None,
            )
        elif step == 0:
            plan = StepPlan(
                mode=mode,
                step=step,
                call_input_token_ids=prompt,
                context_token_count=context_token_count,
                attention_mask_token_count=context_token_count,
                position_start=0,
                position_end=oracle.PROMPT_TOKEN_COUNT - 1,
                teacher_input_token_id=None,
                cache_length_before=0,
                cache_length_after=oracle.PROMPT_TOKEN_COUNT,
            )
        else:
            plan = StepPlan(
                mode=mode,
                step=step,
                call_input_token_ids=(teacher[step - 1],),
                context_token_count=context_token_count,
                attention_mask_token_count=context_token_count,
                position_start=context_token_count - 1,
                position_end=context_token_count - 1,
                teacher_input_token_id=teacher[step - 1],
                cache_length_before=context_token_count - 1,
                cache_length_after=context_token_count,
            )
        plans.append(plan)
    return tuple(plans)


def _canonical_bf16_le_bytes(tensor: object, torch: Any) -> bytes:
    try:
        raw = (
            tensor.detach()
            .to(device="cpu")
            .contiguous()
            .view(torch.uint8)
            .numpy()
            .tobytes()
        )
    except (AttributeError, RuntimeError, TypeError) as error:
        raise Qwen3BGenerationTraceError(
            "cannot materialize raw BF16 logit bytes"
        ) from error
    if len(raw) != oracle.RAW_LOGIT_BYTES:
        raise Qwen3BGenerationTraceError("raw BF16 logit row byte count differs")
    if sys.byteorder == "little":
        return raw
    if sys.byteorder == "big":
        return b"".join(
            raw[index : index + BF16_BYTES][::-1]
            for index in range(0, len(raw), BF16_BYTES)
        )
    raise Qwen3BGenerationTraceError("unsupported host byte order")


class HuggingFaceQwen3BGenerationTraceBackend:
    """CUDA-only adapter used only by this offline artifact producer."""

    def __init__(self, loader: oracle.HuggingFaceQwen3BBackend) -> None:
        self._loader: oracle.HuggingFaceQwen3BBackend | None = loader
        self._torch = loader._torch
        self._model = loader._model
        self._device = loader._device
        self.producer_metadata = dict(loader.producer_metadata)
        self.producer_metadata["implementation_id"] = IMPLEMENTATION_ID

    @classmethod
    def load(
        cls, *, checkpoint: oracle.CheckpointManifest, device: str
    ) -> "HuggingFaceQwen3BGenerationTraceBackend":
        loader = oracle.HuggingFaceQwen3BBackend.load(
            checkpoint=checkpoint, device=device
        )
        try:
            return cls(loader)
        except BaseException:
            loader.close()
            raise

    def _capture_logits(self, output: object) -> CapturedLogits:
        torch = self._torch
        try:
            logits = output.logits
            expected_shape = (1, 1, oracle.MODEL_VOCABULARY_SIZE)
            if tuple(int(value) for value in logits.shape) != expected_shape:
                raise Qwen3BGenerationTraceError(
                    "HF logits shape differs from [1, 1, 151936]"
                )
            row = logits[0, -1]
            if row.dtype != torch.bfloat16:
                raise Qwen3BGenerationTraceError("HF last logits are not BF16")
            cpu_row = row.detach().to(device="cpu").contiguous()
            raw = _canonical_bf16_le_bytes(cpu_row, torch)
            analysis = _analyze_bf16_le_logits(raw)
            return CapturedLogits(
                tensor=cpu_row,
                raw_bf16_le_sha256=_sha256_bytes(raw),
                addressable_bf16_le_sha256=_sha256_bytes(
                    raw[: oracle.ADDRESSABLE_LOGIT_BYTES]
                ),
                non_addressable_bf16_le_sha256=_sha256_bytes(
                    raw[oracle.ADDRESSABLE_LOGIT_BYTES :]
                ),
                raw_argmax_token_id=analysis.raw_argmax_token_id,
                selected_token_id=analysis.selected_token_id,
                selected_logit_bf16_as_f32=analysis.selected_logit_bf16_as_f32,
                top_token_ids=analysis.top_token_ids,
                top_values_bf16_as_f32=analysis.top_values_bf16_as_f32,
            )
        except (AttributeError, IndexError, RuntimeError, TypeError) as error:
            raise Qwen3BGenerationTraceError("cannot capture HF last logits") from error

    def _run_cache_off_row(
        self, *, step: int, teacher_prefix: Sequence[int]
    ) -> CapturedStep:
        model = self._model
        if model is None:
            raise Qwen3BGenerationTraceError("generation backend is closed")
        torch = self._torch
        prompt = (oracle.EXPECTED_PROMPT_TOKEN_ID,) * oracle.PROMPT_TOKEN_COUNT
        call_input = prompt + tuple(teacher_prefix)
        context_token_count = len(call_input)
        input_ids = torch.tensor(
            [list(call_input)], dtype=torch.long, device=self._device
        )
        attention_mask = torch.ones_like(input_ids)
        position_ids = torch.arange(
            context_token_count, dtype=torch.long, device=self._device
        ).unsqueeze(0)
        try:
            with torch.inference_mode():
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                    logits_to_keep=1,
                    return_dict=True,
                )
            if getattr(output, "past_key_values", None) is not None:
                raise Qwen3BGenerationTraceError(
                    "cache-off generation unexpectedly returned a KV cache"
                )
            logits = self._capture_logits(output)
            return CapturedStep(
                plan=StepPlan(
                    mode=CACHE_OFF,
                    step=step,
                    call_input_token_ids=call_input,
                    context_token_count=context_token_count,
                    attention_mask_token_count=context_token_count,
                    position_start=0,
                    position_end=context_token_count - 1,
                    teacher_input_token_id=None,
                    cache_length_before=None,
                    cache_length_after=None,
                ),
                logits=logits,
            )
        finally:
            del input_ids, attention_mask, position_ids
            try:
                del output
            except UnboundLocalError:
                pass

    @staticmethod
    def _cache_sequence_length(cache: object) -> int:
        getter = getattr(cache, "get_seq_length", None)
        if not callable(getter):
            raise Qwen3BGenerationTraceError(
                "cache-on generation did not return a DynamicCache-like get_seq_length"
            )
        try:
            length = getter()
        except (RuntimeError, TypeError, ValueError) as error:
            raise Qwen3BGenerationTraceError(
                "cannot read cache-on DynamicCache sequence length"
            ) from error
        if isinstance(length, bool) or not isinstance(length, int) or length < 0:
            raise Qwen3BGenerationTraceError(
                "cache-on DynamicCache sequence length is invalid"
            )
        return length

    def _run_cache_on_row(
        self, *, plan: StepPlan, past: object | None
    ) -> tuple[CapturedStep, object]:
        model = self._model
        if model is None:
            raise Qwen3BGenerationTraceError("generation backend is closed")
        if plan.mode != CACHE_ON:
            raise Qwen3BGenerationTraceError("cache-on row received the wrong plan")
        if plan.cache_length_before is None or plan.cache_length_after is None:
            raise Qwen3BGenerationTraceError("cache-on plan lacks cache lengths")
        if plan.step == 0:
            if past is not None or plan.cache_length_before != 0:
                raise Qwen3BGenerationTraceError("cache-on prefill cache state differs")
        else:
            if past is None:
                raise Qwen3BGenerationTraceError("cache-on decode lacks a DynamicCache")
            if self._cache_sequence_length(past) != plan.cache_length_before:
                raise Qwen3BGenerationTraceError(
                    "cache-on DynamicCache length differs before decode"
                )
        torch = self._torch
        input_ids = torch.tensor(
            [list(plan.call_input_token_ids)], dtype=torch.long, device=self._device
        )
        attention_mask = torch.ones(
            (1, plan.attention_mask_token_count),
            dtype=torch.long,
            device=self._device,
        )
        position_ids = torch.arange(
            plan.position_start,
            plan.position_end + 1,
            dtype=torch.long,
            device=self._device,
        ).unsqueeze(0)
        kwargs: dict[str, object] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "use_cache": True,
            "logits_to_keep": 1,
            "return_dict": True,
        }
        if past is not None:
            kwargs["past_key_values"] = past
        try:
            with torch.inference_mode():
                output = model(**kwargs)
            next_past = getattr(output, "past_key_values", None)
            if next_past is None:
                raise Qwen3BGenerationTraceError(
                    "cache-on generation did not return a DynamicCache"
                )
            if self._cache_sequence_length(next_past) != plan.cache_length_after:
                raise Qwen3BGenerationTraceError(
                    "cache-on DynamicCache length differs after model call"
                )
            logits = self._capture_logits(output)
            return CapturedStep(plan=plan, logits=logits), next_past
        finally:
            del input_ids, attention_mask, position_ids
            try:
                del output
            except UnboundLocalError:
                pass

    def capture(self, prompt_token_ids: Sequence[int]) -> GenerationCapture:
        """Derive a cache-off teacher stream, then replay it through DynamicCache."""

        expected_prompt = (oracle.EXPECTED_PROMPT_TOKEN_ID,) * oracle.PROMPT_TOKEN_COUNT
        if tuple(prompt_token_ids) != expected_prompt:
            raise Qwen3BGenerationTraceError(
                "teacher-forced input IDs differ from the pinned P2048 prompt"
            )
        teacher: list[int] = []
        cache_off_steps: list[CapturedStep] = []
        for step in range(oracle.OUTPUT_TOKEN_COUNT):
            captured = self._run_cache_off_row(step=step, teacher_prefix=teacher)
            cache_off_steps.append(captured)
            teacher.append(captured.logits.selected_token_id)
        teacher_ids = _validate_teacher_token_ids(teacher)
        cache_off_plan = teacher_forced_step_plan(CACHE_OFF, teacher_ids)
        if tuple(step.plan for step in cache_off_steps) != cache_off_plan:
            raise Qwen3BGenerationTraceError("cache-off execution plan differs")

        cache_on_plan = teacher_forced_step_plan(CACHE_ON, teacher_ids)
        cache_on_steps: list[CapturedStep] = []
        past: object | None = None
        for plan in cache_on_plan:
            captured, past = self._run_cache_on_row(plan=plan, past=past)
            cache_on_steps.append(captured)
        try:
            cache_off_logits = self._torch.stack(
                [step.logits.tensor for step in cache_off_steps], dim=0
            ).contiguous()
            cache_on_logits = self._torch.stack(
                [step.logits.tensor for step in cache_on_steps], dim=0
            ).contiguous()
        except (AttributeError, RuntimeError, TypeError) as error:
            raise Qwen3BGenerationTraceError(
                "cannot stack BF16 generation sidecars"
            ) from error
        expected_shape = (oracle.OUTPUT_TOKEN_COUNT, oracle.MODEL_VOCABULARY_SIZE)
        for tensor in (cache_off_logits, cache_on_logits):
            if tuple(int(value) for value in tensor.shape) != expected_shape:
                raise Qwen3BGenerationTraceError("generation sidecar shape differs")
            if tensor.dtype != self._torch.bfloat16:
                raise Qwen3BGenerationTraceError("generation sidecar dtype differs")
        del past
        return GenerationCapture(
            teacher_token_ids=teacher_ids,
            cache_off_steps=tuple(cache_off_steps),
            cache_on_steps=tuple(cache_on_steps),
            cache_off_logits=cache_off_logits,
            cache_on_logits=cache_on_logits,
        )

    def close(self) -> None:
        loader = self._loader
        if loader is None:
            return
        self._loader = None
        self._model = None
        loader.close()


def _source_record(root: Path, relative: str) -> dict[str, object]:
    try:
        source = oracle._regular_file(root / relative, f"source file {relative}")
    except oracle.Qwen3BServingOracleError as error:
        raise Qwen3BGenerationTraceError(str(error)) from error
    return {"path": relative, "sha256": _sha256_file(source)}


def collect_source_provenance(repo_root: Path) -> dict[str, object]:
    """Bind an artifact to the standalone offline oracle sources actually run."""

    try:
        root = oracle._regular_directory(repo_root.expanduser(), "repository root")
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            [
                "git",
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
                "--",
                *SOURCE_PATHS.values(),
            ],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
        for relative in SOURCE_PATHS.values():
            subprocess.run(
                ["git", "ls-files", "--error-unmatch", "--", relative],
                cwd=root,
                check=True,
                capture_output=True,
            )
    except (
        OSError,
        subprocess.CalledProcessError,
        oracle.Qwen3BServingOracleError,
    ) as error:
        raise Qwen3BGenerationTraceError(
            "cannot record teacher-forced oracle Git provenance"
        ) from error
    if oracle.GIT_REVISION_RE.fullmatch(revision) is None:
        raise Qwen3BGenerationTraceError("Git revision has an unexpected format")
    return {
        "git_revision": revision,
        "source_dirty": bool(status),
        "source_status_sha256": _sha256_bytes(status),
        "sources": {
            name: _source_record(root, relative)
            for name, relative in SOURCE_PATHS.items()
        },
    }


def _workload_document(workload: oracle.ServingWorkload) -> dict[str, object]:
    """Record the prompt binding and explicitly rule out workload output IDs as teachers."""

    return {
        "schema_version": oracle.WORKLOAD_SCHEMA_VERSION,
        "case": oracle.WORKLOAD_CASE,
        "source_sha256": workload.source_sha256,
        "prompt_token_count": len(workload.prompt_token_ids),
        "prompt_token_ids_le_u32_sha256": oracle._token_ids_sha256(
            workload.prompt_token_ids
        ),
        "workload_output_token_count": len(workload.output_token_ids),
        "workload_output_token_ids_le_u32_sha256": oracle._token_ids_sha256(
            workload.output_token_ids
        ),
        "workload_output_token_ids_used_as_teacher": False,
    }


def _execution_document() -> dict[str, object]:
    return {
        "attention_implementation": "eager",
        "batch_size": 1,
        "cublas_workspace_config": ":4096:8",
        "deterministic_algorithms": True,
        "dtype": "bfloat16",
        "explicit_attention_mask": True,
        "explicit_input_ids": True,
        "explicit_position_ids": True,
        "hf_hub_offline": True,
        "inference_mode": True,
        "local_files_only": True,
        "logits_to_keep": 1,
        "return_dict": True,
        "sampling_applied": False,
        "tf32_enabled": False,
        "transformers_offline": True,
        "trust_remote_code": False,
    }


def _teacher_forced_contract() -> dict[str, object]:
    return {
        "fixed_output_token_count": oracle.OUTPUT_TOKEN_COUNT,
        "teacher_token_derivation": "cache-off-addressable-greedy-argmax",
        "teacher_token_source_mode": CACHE_OFF,
        "addressable_token_count": oracle.ADDRESSABLE_TOKEN_COUNT,
        "vocabulary_size": oracle.MODEL_VOCABULARY_SIZE,
        "cache_modes": list(CACHE_MODES),
        "cache_on_first_decode_position": oracle.PROMPT_TOKEN_COUNT,
        "cache_on_decode_position_rule": "prompt_token_count+step-1",
        "cache_on_decode_input_rule": "teacher_token_ids[step-1]",
        "cache_position_argument": "omitted-transformers-5.15.1",
    }


def _logit_document(
    logits: CapturedLogits,
    *,
    step: int,
    teacher_token: int,
    require_teacher_match: bool,
) -> dict[str, object]:
    if (
        len(logits.top_token_ids) != oracle.TOP_K
        or len(logits.top_values_bf16_as_f32) != oracle.TOP_K
    ):
        raise Qwen3BGenerationTraceError("captured top-k cardinality differs")
    if logits.selected_token_id != logits.top_token_ids[0]:
        raise Qwen3BGenerationTraceError("captured selected token differs from top-k")
    if logits.selected_logit_bf16_as_f32 != logits.top_values_bf16_as_f32[0]:
        raise Qwen3BGenerationTraceError("captured selected logit differs from top-k")
    selection_matches_teacher = logits.selected_token_id == teacher_token
    if require_teacher_match and not selection_matches_teacher:
        raise Qwen3BGenerationTraceError(
            f"captured selected token differs from cache-off teacher at step {step}"
        )
    return {
        "sidecar_row_index": step,
        "dtype": "bfloat16",
        "element_count": oracle.MODEL_VOCABULARY_SIZE,
        "canonical_byte_order": "little-endian-u16",
        "raw_bf16_le_sha256": logits.raw_bf16_le_sha256,
        "raw_bf16_le_bytes": oracle.RAW_LOGIT_BYTES,
        "addressable_bf16_le_sha256": logits.addressable_bf16_le_sha256,
        "addressable_bf16_le_bytes": oracle.ADDRESSABLE_LOGIT_BYTES,
        "non_addressable_bf16_le_sha256": logits.non_addressable_bf16_le_sha256,
        "non_addressable_bf16_le_bytes": oracle.NON_ADDRESSABLE_LOGIT_BYTES,
        "raw_argmax_token_id": logits.raw_argmax_token_id,
        "selected_token_id": logits.selected_token_id,
        "cache_off_teacher_token_id": teacher_token,
        "selected_logit_bf16_as_f32": logits.selected_logit_bf16_as_f32,
        "top_k": oracle.TOP_K,
        "top_token_ids": list(logits.top_token_ids),
        "top_values_bf16_as_f32": list(logits.top_values_bf16_as_f32),
        "selection_matches_cache_off_teacher": selection_matches_teacher,
    }


def _step_document(step: CapturedStep, *, teacher_token: int) -> dict[str, object]:
    plan = step.plan
    return {
        "step": plan.step,
        "call_input_token_count": len(plan.call_input_token_ids),
        "call_input_token_ids_le_u32_sha256": oracle._token_ids_sha256(
            plan.call_input_token_ids
        ),
        "context_token_count": plan.context_token_count,
        "attention_mask_token_count": plan.attention_mask_token_count,
        "position_start": plan.position_start,
        "position_end": plan.position_end,
        "teacher_input_token_id": plan.teacher_input_token_id,
        "cache_length_before": plan.cache_length_before,
        "cache_length_after": plan.cache_length_after,
        "logits": _logit_document(
            step.logits,
            step=plan.step,
            teacher_token=teacher_token,
            require_teacher_match=plan.mode == CACHE_OFF,
        ),
    }


def _sidecar_document(path: Path, sha256: str) -> dict[str, object]:
    return {
        "path": path.name,
        "sha256": sha256,
        "format": "safetensors",
        "tensor_key": SIDECAR_TENSOR_KEY,
        "tensor_count": 1,
        "dtype": "bfloat16",
        "shape": [oracle.OUTPUT_TOKEN_COUNT, oracle.MODEL_VOCABULARY_SIZE],
        "canonical_byte_order": "little-endian-u16",
        "raw_bf16_le_bytes": oracle.OUTPUT_TOKEN_COUNT * oracle.RAW_LOGIT_BYTES,
    }


def _mode_document(
    *,
    mode: str,
    steps: Sequence[CapturedStep],
    teacher_token_ids: Sequence[int],
    sidecar_path: Path,
    sidecar_sha256: str,
) -> dict[str, object]:
    expected_plan = teacher_forced_step_plan(mode, teacher_token_ids)
    if tuple(step.plan for step in steps) != expected_plan:
        raise Qwen3BGenerationTraceError(f"{mode} capture plan differs")
    return {
        "mode": mode,
        "sidecar": _sidecar_document(sidecar_path, sidecar_sha256),
        "steps": [
            _step_document(step, teacher_token=teacher_token_ids[index])
            for index, step in enumerate(steps)
        ],
    }


def build_generation_artifact(
    *,
    workload: oracle.ServingWorkload,
    checkpoint: oracle.CheckpointManifest,
    capture: GenerationCapture,
    producer_metadata: Mapping[str, object],
    source_provenance: Mapping[str, object],
    cache_off_sidecar_path: Path,
    cache_on_sidecar_path: Path,
    cache_off_sidecar_sha256: str,
    cache_on_sidecar_sha256: str,
    created_at: datetime,
) -> dict[str, object]:
    """Build the JSON manifest after both create-only sidecars exist."""

    teacher = _validate_teacher_token_ids(capture.teacher_token_ids)
    document: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "performance_claim_eligible": False,
        "created_at": oracle._utc_text(created_at),
        "producer": dict(producer_metadata),
        "contract": {
            "model_id": oracle.MODEL_ID,
            "model_revision": oracle.MODEL_REVISION,
            "workload": _workload_document(workload),
            "execution": _execution_document(),
            "teacher_forced_generation": _teacher_forced_contract(),
        },
        "model": {
            "checkpoint_path": str(checkpoint.root),
            "checkpoint_verification": "riley-checkpoint-v1-receipt-sha256+regular-file-size",
            "checkpoint_receipt": checkpoint.receipt.document(),
            "files": [record.document() for record in checkpoint.files],
        },
        "provenance": {"source_repository": dict(source_provenance)},
        "generation": {
            "teacher_token_ids": list(teacher),
            "teacher_token_ids_le_u32_sha256": oracle._token_ids_sha256(teacher),
            "cache_off": _mode_document(
                mode=CACHE_OFF,
                steps=capture.cache_off_steps,
                teacher_token_ids=teacher,
                sidecar_path=cache_off_sidecar_path,
                sidecar_sha256=cache_off_sidecar_sha256,
            ),
            "cache_on": _mode_document(
                mode=CACHE_ON,
                steps=capture.cache_on_steps,
                teacher_token_ids=teacher,
                sidecar_path=cache_on_sidecar_path,
                sidecar_sha256=cache_on_sidecar_sha256,
            ),
        },
    }
    validate_generation_artifact(document)
    return document


def _validate_producer(value: object) -> None:
    producer = _require_mapping(value, "artifact producer")
    _require_exact_keys(
        producer,
        {
            "implementation_id",
            "runtime_dependency_class",
            "python_version",
            "python_executable_sha256",
            "python_platform_system",
            "python_platform_machine",
            "torch_version",
            "transformers_version",
            "safetensors_version",
            "selected_device",
            "transformers_model_source",
        },
        "artifact producer",
    )
    if producer["implementation_id"] != IMPLEMENTATION_ID:
        raise Qwen3BGenerationTraceError("artifact producer implementation differs")
    if producer["runtime_dependency_class"] != "offline-python-reference":
        raise Qwen3BGenerationTraceError("artifact producer dependency class differs")
    for name in (
        "python_version",
        "python_platform_system",
        "python_platform_machine",
        "torch_version",
        "transformers_version",
        "safetensors_version",
    ):
        _require_string(producer[name], f"artifact producer {name}")
    _require_sha256(
        producer["python_executable_sha256"], "artifact producer Python SHA-256"
    )
    device = _require_mapping(producer["selected_device"], "artifact producer device")
    _require_exact_keys(
        device,
        {
            "requested",
            "resolved",
            "index",
            "name",
            "compute_capability",
            "driver_version",
            "runtime_cuda_version",
        },
        "artifact producer device",
    )
    for name in (
        "requested",
        "resolved",
        "name",
        "compute_capability",
        "driver_version",
        "runtime_cuda_version",
    ):
        _require_string(device[name], f"artifact producer device {name}")
    _require_int(device["index"], "artifact producer device index")
    source = _require_mapping(
        producer["transformers_model_source"], "artifact producer Transformers source"
    )
    _require_exact_keys(
        source,
        {"module", "filename", "sha256"},
        "artifact producer Transformers source",
    )
    _require_string(source["module"], "artifact producer Transformers module")
    _require_string(source["filename"], "artifact producer Transformers filename")
    _require_sha256(source["sha256"], "artifact producer Transformers SHA-256")


def _validate_workload(value: object) -> None:
    workload = _require_mapping(value, "artifact workload")
    _require_exact_keys(
        workload,
        {
            "schema_version",
            "case",
            "source_sha256",
            "prompt_token_count",
            "prompt_token_ids_le_u32_sha256",
            "workload_output_token_count",
            "workload_output_token_ids_le_u32_sha256",
            "workload_output_token_ids_used_as_teacher",
        },
        "artifact workload",
    )
    expected = {
        "schema_version": oracle.WORKLOAD_SCHEMA_VERSION,
        "case": oracle.WORKLOAD_CASE,
        "source_sha256": oracle.WORKLOAD_SHA256,
        "prompt_token_count": oracle.PROMPT_TOKEN_COUNT,
        "prompt_token_ids_le_u32_sha256": oracle.PROMPT_TOKEN_IDS_SHA256,
        "workload_output_token_count": oracle.OUTPUT_TOKEN_COUNT,
        "workload_output_token_ids_le_u32_sha256": oracle.OUTPUT_TOKEN_IDS_SHA256,
        "workload_output_token_ids_used_as_teacher": False,
    }
    if dict(workload) != expected:
        raise Qwen3BGenerationTraceError("artifact workload contract differs")


def _validate_model(value: object) -> None:
    model = _require_mapping(value, "artifact model")
    _require_exact_keys(
        model,
        {"checkpoint_path", "checkpoint_verification", "checkpoint_receipt", "files"},
        "artifact model",
    )
    _require_string(model["checkpoint_path"], "artifact checkpoint path")
    if (
        model["checkpoint_verification"]
        != "riley-checkpoint-v1-receipt-sha256+regular-file-size"
    ):
        raise Qwen3BGenerationTraceError("artifact checkpoint verification differs")
    receipt = _require_mapping(
        model["checkpoint_receipt"], "artifact checkpoint receipt"
    )
    _require_exact_keys(
        receipt, {"path", "size_bytes", "sha256"}, "artifact checkpoint receipt"
    )
    if dict(receipt) != {
        "path": oracle.CHECKPOINT_RECEIPT_FILENAME,
        "size_bytes": oracle.CHECKPOINT_RECEIPT_BYTES,
        "sha256": oracle.CHECKPOINT_RECEIPT_SHA256,
    }:
        raise Qwen3BGenerationTraceError("artifact checkpoint receipt differs")
    try:
        oracle._validate_file_records(model["files"])
    except oracle.Qwen3BServingOracleError as error:
        raise Qwen3BGenerationTraceError(str(error)) from error


def _validate_source_provenance(value: object) -> None:
    source = _require_mapping(value, "artifact source provenance")
    _require_exact_keys(
        source,
        {"git_revision", "source_dirty", "source_status_sha256", "sources"},
        "artifact source provenance",
    )
    revision = _require_string(source["git_revision"], "artifact Git revision")
    if oracle.GIT_REVISION_RE.fullmatch(revision) is None:
        raise Qwen3BGenerationTraceError("artifact Git revision is malformed")
    if not isinstance(source["source_dirty"], bool):
        raise Qwen3BGenerationTraceError("artifact source dirty must be a boolean")
    _require_sha256(source["source_status_sha256"], "artifact source status SHA-256")
    records = _require_mapping(source["sources"], "artifact source records")
    if set(records) != set(SOURCE_PATHS):
        raise Qwen3BGenerationTraceError("artifact source record set differs")
    for name, relative in SOURCE_PATHS.items():
        record = _require_mapping(records[name], f"artifact source record {name}")
        _require_exact_keys(
            record, {"path", "sha256"}, f"artifact source record {name}"
        )
        if record["path"] != relative:
            raise Qwen3BGenerationTraceError("artifact source path differs")
        _require_sha256(record["sha256"], f"artifact source record {name} SHA-256")


def _validate_teacher_forced_contract(value: object) -> None:
    contract = _require_mapping(value, "artifact teacher-forced contract")
    if dict(contract) != _teacher_forced_contract():
        raise Qwen3BGenerationTraceError("artifact teacher-forced contract differs")


def _validate_sidecar(value: object, label: str) -> Mapping[str, object]:
    sidecar = _require_mapping(value, label)
    _require_exact_keys(
        sidecar,
        {
            "path",
            "sha256",
            "format",
            "tensor_key",
            "tensor_count",
            "dtype",
            "shape",
            "canonical_byte_order",
            "raw_bf16_le_bytes",
        },
        label,
    )
    path = _require_string(sidecar["path"], f"{label}.path")
    if Path(path).name != path or not path.endswith(".safetensors"):
        raise Qwen3BGenerationTraceError(f"{label}.path must be a sidecar filename")
    expected = {
        "format": "safetensors",
        "tensor_key": SIDECAR_TENSOR_KEY,
        "tensor_count": 1,
        "dtype": "bfloat16",
        "shape": [oracle.OUTPUT_TOKEN_COUNT, oracle.MODEL_VOCABULARY_SIZE],
        "canonical_byte_order": "little-endian-u16",
        "raw_bf16_le_bytes": oracle.OUTPUT_TOKEN_COUNT * oracle.RAW_LOGIT_BYTES,
    }
    for key, expected_value in expected.items():
        if sidecar[key] != expected_value:
            raise Qwen3BGenerationTraceError(f"{label}.{key} differs")
    _require_sha256(sidecar["sha256"], f"{label}.sha256")
    return sidecar


def _validate_logits(
    value: object,
    *,
    step: int,
    teacher_token: int,
    require_teacher_match: bool,
    label: str,
) -> None:
    logits = _require_mapping(value, label)
    _require_exact_keys(
        logits,
        {
            "sidecar_row_index",
            "dtype",
            "element_count",
            "canonical_byte_order",
            "raw_bf16_le_sha256",
            "raw_bf16_le_bytes",
            "addressable_bf16_le_sha256",
            "addressable_bf16_le_bytes",
            "non_addressable_bf16_le_sha256",
            "non_addressable_bf16_le_bytes",
            "raw_argmax_token_id",
            "selected_token_id",
            "cache_off_teacher_token_id",
            "selected_logit_bf16_as_f32",
            "top_k",
            "top_token_ids",
            "top_values_bf16_as_f32",
            "selection_matches_cache_off_teacher",
        },
        label,
    )
    expected = {
        "sidecar_row_index": step,
        "dtype": "bfloat16",
        "element_count": oracle.MODEL_VOCABULARY_SIZE,
        "canonical_byte_order": "little-endian-u16",
        "raw_bf16_le_bytes": oracle.RAW_LOGIT_BYTES,
        "addressable_bf16_le_bytes": oracle.ADDRESSABLE_LOGIT_BYTES,
        "non_addressable_bf16_le_bytes": oracle.NON_ADDRESSABLE_LOGIT_BYTES,
        "top_k": oracle.TOP_K,
    }
    for key, expected_value in expected.items():
        if logits[key] != expected_value:
            raise Qwen3BGenerationTraceError(f"{label}.{key} differs")
    for key in (
        "raw_bf16_le_sha256",
        "addressable_bf16_le_sha256",
        "non_addressable_bf16_le_sha256",
    ):
        _require_sha256(logits[key], f"{label}.{key}")
    _require_int(
        logits["raw_argmax_token_id"],
        f"{label}.raw_argmax_token_id",
        maximum=oracle.MODEL_VOCABULARY_SIZE - 1,
    )
    selected = _require_int(
        logits["selected_token_id"],
        f"{label}.selected_token_id",
        maximum=oracle.ADDRESSABLE_TOKEN_COUNT - 1,
    )
    recorded_teacher = _require_int(
        logits["cache_off_teacher_token_id"],
        f"{label}.cache_off_teacher_token_id",
        maximum=oracle.ADDRESSABLE_TOKEN_COUNT - 1,
    )
    if recorded_teacher != teacher_token:
        raise Qwen3BGenerationTraceError(
            f"{label}.cache_off_teacher_token_id differs from teacher"
        )
    selection_matches_teacher = logits["selection_matches_cache_off_teacher"]
    if not isinstance(selection_matches_teacher, bool):
        raise Qwen3BGenerationTraceError(
            f"{label}.selection_matches_cache_off_teacher must be a boolean"
        )
    if selection_matches_teacher != (selected == teacher_token):
        raise Qwen3BGenerationTraceError(
            f"{label}.selection_matches_cache_off_teacher differs from selected token"
        )
    if require_teacher_match and selected != teacher_token:
        raise Qwen3BGenerationTraceError(
            f"{label}.selected_token_id differs from teacher"
        )
    selected_value = _require_finite_float(
        logits["selected_logit_bf16_as_f32"], f"{label}.selected_logit_bf16_as_f32"
    )
    top_ids = _require_u32_ids(
        logits["top_token_ids"],
        label=f"{label}.top_token_ids",
        count=oracle.TOP_K,
        upper_bound=oracle.ADDRESSABLE_TOKEN_COUNT,
    )
    if len(set(top_ids)) != oracle.TOP_K:
        raise Qwen3BGenerationTraceError(f"{label}.top_token_ids must be unique")
    if top_ids[0] != selected:
        raise Qwen3BGenerationTraceError(
            f"{label}.top_token_ids[0] differs from selected"
        )
    values_value = logits["top_values_bf16_as_f32"]
    if not isinstance(values_value, list) or len(values_value) != oracle.TOP_K:
        raise Qwen3BGenerationTraceError(
            f"{label}.top_values_bf16_as_f32 cardinality differs"
        )
    values = tuple(
        _require_finite_float(value, f"{label}.top_values_bf16_as_f32[{index}]")
        for index, value in enumerate(values_value)
    )
    if any(left < right for left, right in zip(values, values[1:])):
        raise Qwen3BGenerationTraceError(
            f"{label}.top_values_bf16_as_f32 are not descending"
        )
    if values[0] != selected_value:
        raise Qwen3BGenerationTraceError(f"{label}.selected logit differs from top-k")


def _validate_step(
    value: object,
    *,
    plan: StepPlan,
    teacher_token: int,
    label: str,
) -> None:
    step = _require_mapping(value, label)
    _require_exact_keys(
        step,
        {
            "step",
            "call_input_token_count",
            "call_input_token_ids_le_u32_sha256",
            "context_token_count",
            "attention_mask_token_count",
            "position_start",
            "position_end",
            "teacher_input_token_id",
            "cache_length_before",
            "cache_length_after",
            "logits",
        },
        label,
    )
    expected = {
        "step": plan.step,
        "call_input_token_count": len(plan.call_input_token_ids),
        "call_input_token_ids_le_u32_sha256": oracle._token_ids_sha256(
            plan.call_input_token_ids
        ),
        "context_token_count": plan.context_token_count,
        "attention_mask_token_count": plan.attention_mask_token_count,
        "position_start": plan.position_start,
        "position_end": plan.position_end,
        "teacher_input_token_id": plan.teacher_input_token_id,
        "cache_length_before": plan.cache_length_before,
        "cache_length_after": plan.cache_length_after,
    }
    for key, expected_value in expected.items():
        if step[key] != expected_value:
            raise Qwen3BGenerationTraceError(f"{label}.{key} differs")
    _validate_logits(
        step["logits"],
        step=plan.step,
        teacher_token=teacher_token,
        require_teacher_match=plan.mode == CACHE_OFF,
        label=f"{label}.logits",
    )


def _validate_mode(
    value: object,
    *,
    mode: str,
    teacher_token_ids: Sequence[int],
    label: str,
) -> None:
    document = _require_mapping(value, label)
    _require_exact_keys(document, {"mode", "sidecar", "steps"}, label)
    if document["mode"] != mode:
        raise Qwen3BGenerationTraceError(f"{label}.mode differs")
    _validate_sidecar(document["sidecar"], f"{label}.sidecar")
    steps = document["steps"]
    if not isinstance(steps, list) or len(steps) != oracle.OUTPUT_TOKEN_COUNT:
        raise Qwen3BGenerationTraceError(f"{label}.steps cardinality differs")
    plans = teacher_forced_step_plan(mode, teacher_token_ids)
    for index, plan in enumerate(plans):
        _validate_step(
            steps[index],
            plan=plan,
            teacher_token=teacher_token_ids[index],
            label=f"{label}.steps[{index}]",
        )


def validate_generation_artifact(document: Mapping[str, object]) -> None:
    """Validate schema and all causal position/cache contracts without Torch."""

    _require_exact_keys(
        document,
        {
            "schema_version",
            "artifact_kind",
            "performance_claim_eligible",
            "created_at",
            "producer",
            "contract",
            "model",
            "provenance",
            "generation",
        },
        "teacher-forced artifact",
    )
    if (
        document["schema_version"] != SCHEMA_VERSION
        or document["artifact_kind"] != ARTIFACT_KIND
    ):
        raise Qwen3BGenerationTraceError("teacher-forced artifact identity differs")
    if document["performance_claim_eligible"] is not False:
        raise Qwen3BGenerationTraceError(
            "teacher-forced numerical artifact must not be performance eligible"
        )
    try:
        oracle._validate_utc_text(document["created_at"], "artifact created_at")
    except oracle.Qwen3BServingOracleError as error:
        raise Qwen3BGenerationTraceError(str(error)) from error
    _validate_producer(document["producer"])
    contract = _require_mapping(document["contract"], "artifact contract")
    _require_exact_keys(
        contract,
        {
            "model_id",
            "model_revision",
            "workload",
            "execution",
            "teacher_forced_generation",
        },
        "artifact contract",
    )
    if (
        contract["model_id"] != oracle.MODEL_ID
        or contract["model_revision"] != oracle.MODEL_REVISION
    ):
        raise Qwen3BGenerationTraceError("artifact model identity differs")
    _validate_workload(contract["workload"])
    if contract["execution"] != _execution_document():
        raise Qwen3BGenerationTraceError("artifact execution contract differs")
    _validate_teacher_forced_contract(contract["teacher_forced_generation"])
    _validate_model(document["model"])
    provenance = _require_mapping(document["provenance"], "artifact provenance")
    _require_exact_keys(provenance, {"source_repository"}, "artifact provenance")
    _validate_source_provenance(provenance["source_repository"])
    generation = _require_mapping(document["generation"], "artifact generation")
    _require_exact_keys(
        generation,
        {
            "teacher_token_ids",
            "teacher_token_ids_le_u32_sha256",
            "cache_off",
            "cache_on",
        },
        "artifact generation",
    )
    teacher = _require_u32_ids(
        generation["teacher_token_ids"],
        label="artifact generation teacher tokens",
        count=oracle.OUTPUT_TOKEN_COUNT,
        upper_bound=oracle.ADDRESSABLE_TOKEN_COUNT,
    )
    if generation["teacher_token_ids_le_u32_sha256"] != oracle._token_ids_sha256(
        teacher
    ):
        raise Qwen3BGenerationTraceError(
            "artifact generation teacher token hash differs"
        )
    _validate_mode(
        generation["cache_off"],
        mode=CACHE_OFF,
        teacher_token_ids=teacher,
        label="artifact generation cache_off",
    )
    _validate_mode(
        generation["cache_on"],
        mode=CACHE_ON,
        teacher_token_ids=teacher,
        label="artifact generation cache_on",
    )


def _duplicate_key(pairs: list[tuple[str, object]]) -> dict[str, object]:
    document: dict[str, object] = {}
    for key, value in pairs:
        if key in document:
            raise Qwen3BGenerationTraceError(f"JSON object repeats key {key!r}")
        document[key] = value
    return document


def _nonfinite(value: str) -> None:
    raise Qwen3BGenerationTraceError(f"non-finite JSON constant {value!r} is forbidden")


def _load_manifest(path: Path) -> dict[str, object]:
    try:
        source = oracle._regular_file(path.expanduser(), "teacher-forced manifest")
        raw = source.read_bytes()
        document = dict(
            _require_mapping(
                json.loads(
                    raw,
                    object_pairs_hook=_duplicate_key,
                    parse_constant=_nonfinite,
                ),
                "teacher-forced manifest",
            )
        )
    except (OSError, json.JSONDecodeError, oracle.Qwen3BServingOracleError) as error:
        raise Qwen3BGenerationTraceError(
            "teacher-forced manifest is invalid"
        ) from error
    validate_generation_artifact(document)
    if raw != _canonical_json_bytes(document):
        raise Qwen3BGenerationTraceError(
            "teacher-forced manifest JSON is not canonical"
        )
    return document


def _read_safetensors_header(path: Path) -> tuple[Mapping[str, object], int, int]:
    try:
        source = oracle._regular_file(path.expanduser(), "teacher-forced sidecar")
        size = source.stat().st_size
        with source.open("rb") as handle:
            prefix = handle.read(8)
            if len(prefix) != 8:
                raise Qwen3BGenerationTraceError(
                    "sidecar lacks an 8-byte header length"
                )
            header_bytes = int.from_bytes(prefix, "little")
            if not 0 < header_bytes <= MAX_SAFETENSORS_HEADER_BYTES:
                raise Qwen3BGenerationTraceError("sidecar header size differs")
            raw_header = handle.read(header_bytes)
    except (OSError, oracle.Qwen3BServingOracleError) as error:
        raise Qwen3BGenerationTraceError(
            "cannot read teacher-forced sidecar"
        ) from error
    if len(raw_header) != header_bytes:
        raise Qwen3BGenerationTraceError("sidecar header is truncated")
    try:
        header = _require_mapping(
            json.loads(
                raw_header,
                object_pairs_hook=_duplicate_key,
                parse_constant=_nonfinite,
            ),
            "sidecar header",
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Qwen3BGenerationTraceError("sidecar header is invalid JSON") from error
    return header, 8 + header_bytes, size


def _validate_logits_against_raw_bf16(
    logits: Mapping[str, object],
    analysis: Bf16LogitAnalysis,
    *,
    label: str,
) -> None:
    """Prove all value-derived manifest metadata from one sidecar row."""

    if logits["raw_argmax_token_id"] != analysis.raw_argmax_token_id:
        raise Qwen3BGenerationTraceError(f"{label} raw argmax differs from sidecar")
    if logits["selected_token_id"] != analysis.selected_token_id:
        raise Qwen3BGenerationTraceError(f"{label} selected token differs from sidecar")
    if logits["selected_logit_bf16_as_f32"] != analysis.selected_logit_bf16_as_f32:
        raise Qwen3BGenerationTraceError(f"{label} selected logit differs from sidecar")
    if logits["top_token_ids"] != list(analysis.top_token_ids):
        raise Qwen3BGenerationTraceError(f"{label} top token IDs differ from sidecar")
    if logits["top_values_bf16_as_f32"] != list(analysis.top_values_bf16_as_f32):
        raise Qwen3BGenerationTraceError(f"{label} top values differ from sidecar")


def validate_sidecar_against_artifact(
    *, document: Mapping[str, object], mode: str, sidecar_path: Path
) -> None:
    """Validate one raw-BF16 safetensors sidecar using no Torch or safetensors."""

    validate_generation_artifact(document)
    if mode not in CACHE_MODES:
        raise Qwen3BGenerationTraceError(f"unknown sidecar mode {mode!r}")
    generation = _require_mapping(document["generation"], "artifact generation")
    mode_document = _require_mapping(
        generation["cache_off" if mode == CACHE_OFF else "cache_on"],
        "artifact generation mode",
    )
    sidecar = _validate_sidecar(mode_document["sidecar"], "artifact sidecar")
    try:
        source = oracle._regular_file(
            sidecar_path.expanduser(), "teacher-forced sidecar"
        )
    except oracle.Qwen3BServingOracleError as error:
        raise Qwen3BGenerationTraceError(str(error)) from error
    if source.name != sidecar["path"] or _sha256_file(source) != sidecar["sha256"]:
        raise Qwen3BGenerationTraceError("sidecar binding differs")
    header, data_start, size = _read_safetensors_header(source)
    if set(header) - {"__metadata__"} != {SIDECAR_TENSOR_KEY}:
        raise Qwen3BGenerationTraceError("sidecar tensor key set differs")
    entry = _require_mapping(header[SIDECAR_TENSOR_KEY], "sidecar logits tensor")
    _require_exact_keys(
        entry, {"dtype", "shape", "data_offsets"}, "sidecar logits tensor"
    )
    if entry["dtype"] != "BF16" or entry["shape"] != sidecar["shape"]:
        raise Qwen3BGenerationTraceError("sidecar logits metadata differs")
    offsets = entry["data_offsets"]
    if (
        not isinstance(offsets, list)
        or len(offsets) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int) for value in offsets
        )
    ):
        raise Qwen3BGenerationTraceError("sidecar logits offsets differ")
    start, end = offsets
    expected_bytes = int(sidecar["raw_bf16_le_bytes"])
    if start != 0 or end != expected_bytes or data_start + end != size:
        raise Qwen3BGenerationTraceError("sidecar logits byte range differs")
    steps = mode_document["steps"]
    if not isinstance(steps, list):
        raise Qwen3BGenerationTraceError("artifact sidecar steps are invalid")
    with source.open("rb") as handle:
        for index, step in enumerate(steps):
            mapping = _require_mapping(step, f"artifact sidecar step {index}")
            logits = _require_mapping(
                mapping["logits"], f"artifact sidecar step {index} logits"
            )
            handle.seek(data_start + index * oracle.RAW_LOGIT_BYTES)
            raw = handle.read(oracle.RAW_LOGIT_BYTES)
            if len(raw) != oracle.RAW_LOGIT_BYTES:
                raise Qwen3BGenerationTraceError("sidecar logit row is truncated")
            if _sha256_bytes(raw) != logits["raw_bf16_le_sha256"]:
                raise Qwen3BGenerationTraceError("sidecar raw BF16 hash differs")
            if (
                _sha256_bytes(raw[: oracle.ADDRESSABLE_LOGIT_BYTES])
                != logits["addressable_bf16_le_sha256"]
            ):
                raise Qwen3BGenerationTraceError(
                    "sidecar addressable BF16 hash differs"
                )
            if (
                _sha256_bytes(raw[oracle.ADDRESSABLE_LOGIT_BYTES :])
                != logits["non_addressable_bf16_le_sha256"]
            ):
                raise Qwen3BGenerationTraceError(
                    "sidecar non-addressable BF16 hash differs"
                )
            _validate_logits_against_raw_bf16(
                logits,
                _analyze_bf16_le_logits(raw),
                label=f"artifact sidecar step {index} logits",
            )


def validate_artifact_bindings(
    *,
    manifest_path: Path,
    cache_off_sidecar_path: Path,
    cache_on_sidecar_path: Path,
    workload_path: Path,
    repo_root: Path,
) -> dict[str, object]:
    """Replay workload, source, and sidecar bindings without a model or CUDA."""

    document = _load_manifest(manifest_path)
    try:
        root = oracle._regular_directory(repo_root.expanduser(), "repository root")
        workload = oracle.load_workload(workload_path)
    except oracle.Qwen3BServingOracleError as error:
        raise Qwen3BGenerationTraceError(str(error)) from error
    contract = _require_mapping(document["contract"], "artifact contract")
    if contract["workload"] != _workload_document(workload):
        raise Qwen3BGenerationTraceError("artifact workload binding differs")
    observed = _require_mapping(document["provenance"], "artifact provenance")
    source = _require_mapping(
        observed["source_repository"], "artifact source provenance"
    )
    expected_source = collect_source_provenance(root)
    if source["sources"] != expected_source["sources"]:
        raise Qwen3BGenerationTraceError("artifact source hashes differ")
    validate_sidecar_against_artifact(
        document=document, mode=CACHE_OFF, sidecar_path=cache_off_sidecar_path
    )
    validate_sidecar_against_artifact(
        document=document, mode=CACHE_ON, sidecar_path=cache_on_sidecar_path
    )
    return document


def _output_paths(
    *,
    manifest_path: Path,
    cache_off_sidecar_path: Path,
    cache_on_sidecar_path: Path,
    repo_root: Path,
) -> tuple[Path, Path, Path]:
    try:
        root = oracle._regular_directory(repo_root.expanduser(), "repository root")
    except oracle.Qwen3BServingOracleError as error:
        raise Qwen3BGenerationTraceError(str(error)) from error
    inputs = (
        manifest_path.expanduser(),
        cache_off_sidecar_path.expanduser(),
        cache_on_sidecar_path.expanduser(),
    )
    if any(not path.is_absolute() for path in inputs):
        raise Qwen3BGenerationTraceError("artifact output paths must be absolute")
    for path in inputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    try:
        parent = oracle._regular_directory(
            inputs[0].parent, "artifact output directory"
        )
    except oracle.Qwen3BServingOracleError as error:
        raise Qwen3BGenerationTraceError(str(error)) from error
    outputs = tuple(parent / path.name for path in inputs)
    if any(path.parent.resolve() != parent for path in inputs):
        raise Qwen3BGenerationTraceError("artifact outputs must be sibling files")
    if len({path.name for path in outputs}) != len(outputs):
        raise Qwen3BGenerationTraceError("artifact output filenames must be distinct")
    manifest, cache_off, cache_on = outputs
    if (
        manifest.suffix != ".json"
        or cache_off.suffix != ".safetensors"
        or cache_on.suffix != ".safetensors"
    ):
        raise Qwen3BGenerationTraceError("artifact output extensions differ")
    for output in outputs:
        if output == root or root in output.parents:
            raise Qwen3BGenerationTraceError(
                "artifact outputs must be outside the repository"
            )
        if output.exists() or output.is_symlink():
            raise Qwen3BGenerationTraceError(
                f"refusing to overwrite existing artifact output: {output}"
            )
    return manifest, cache_off, cache_on


def produce_generation_trace(
    *,
    checkpoint_path: Path,
    workload_path: Path,
    manifest_path: Path,
    cache_off_sidecar_path: Path,
    cache_on_sidecar_path: Path,
    repo_root: Path,
    device: str,
    created_at: datetime | None = None,
    backend_factory: BackendFactory = HuggingFaceQwen3BGenerationTraceBackend.load,
    sidecar_writer: SidecarWriter = _default_sidecar_writer,
    source_provenance_factory: SourceProvenanceFactory = collect_source_provenance,
) -> dict[str, object]:
    """Produce a create-only teacher-forced artifact; this never starts a server."""

    manifest, cache_off, cache_on = _output_paths(
        manifest_path=manifest_path,
        cache_off_sidecar_path=cache_off_sidecar_path,
        cache_on_sidecar_path=cache_on_sidecar_path,
        repo_root=repo_root,
    )
    try:
        workload = oracle.load_workload(workload_path)
        checkpoint = oracle.inspect_checkpoint(checkpoint_path)
    except oracle.Qwen3BServingOracleError as error:
        raise Qwen3BGenerationTraceError(str(error)) from error
    provenance = source_provenance_factory(repo_root)
    backend = backend_factory(checkpoint=checkpoint, device=device)
    created_sidecars: list[Path] = []
    try:
        capture = backend.capture(workload.prompt_token_ids)
        _write_sidecar_exclusive(
            cache_off,
            {SIDECAR_TENSOR_KEY: capture.cache_off_logits},
            sidecar_writer,
        )
        created_sidecars.append(cache_off)
        _write_sidecar_exclusive(
            cache_on,
            {SIDECAR_TENSOR_KEY: capture.cache_on_logits},
            sidecar_writer,
        )
        created_sidecars.append(cache_on)
        document = build_generation_artifact(
            workload=workload,
            checkpoint=checkpoint,
            capture=capture,
            producer_metadata=backend.producer_metadata,
            source_provenance=provenance,
            cache_off_sidecar_path=cache_off,
            cache_on_sidecar_path=cache_on,
            cache_off_sidecar_sha256=_sha256_file(cache_off),
            cache_on_sidecar_sha256=_sha256_file(cache_on),
            created_at=created_at or datetime.now(timezone.utc),
        )
        validate_sidecar_against_artifact(
            document=document, mode=CACHE_OFF, sidecar_path=cache_off
        )
        validate_sidecar_against_artifact(
            document=document, mode=CACHE_ON, sidecar_path=cache_on
        )
        try:
            oracle.write_artifact_exclusive(manifest, document)
        except oracle.Qwen3BServingOracleError as error:
            raise Qwen3BGenerationTraceError(str(error)) from error
        return document
    except BaseException:
        for sidecar in reversed(created_sidecars):
            try:
                sidecar.unlink()
            except FileNotFoundError:
                pass
        raise
    finally:
        backend.close()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m riley_reference.qwen3b_generation_trace",
        description="offline Qwen2.5-3B P2048 cache-on/cache-off teacher-forced oracle",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    produce = commands.add_parser(
        "produce", help="write a create-only manifest and two BF16 safetensors sidecars"
    )
    produce.add_argument("--checkpoint", type=Path, required=True)
    produce.add_argument("--workload", type=Path, required=True)
    produce.add_argument("--manifest", type=Path, required=True)
    produce.add_argument("--cache-off-sidecar", type=Path, required=True)
    produce.add_argument("--cache-on-sidecar", type=Path, required=True)
    produce.add_argument("--repo-root", type=Path, required=True)
    produce.add_argument("--device", default="cuda:0")
    validate = commands.add_parser(
        "validate",
        help="validate manifest/source/workload/sidecars without CUDA or Torch",
    )
    validate.add_argument("manifest", type=Path)
    validate.add_argument("--cache-off-sidecar", type=Path, required=True)
    validate.add_argument("--cache-on-sidecar", type=Path, required=True)
    validate.add_argument("--workload", type=Path, required=True)
    validate.add_argument("--repo-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "produce":
            document = produce_generation_trace(
                checkpoint_path=args.checkpoint,
                workload_path=args.workload,
                manifest_path=args.manifest,
                cache_off_sidecar_path=args.cache_off_sidecar,
                cache_on_sidecar_path=args.cache_on_sidecar,
                repo_root=args.repo_root,
                device=args.device,
            )
            print(
                f"wrote {document['artifact_kind']} to {args.manifest}; "
                f"manifest_sha256={_sha256_file(args.manifest)}"
            )
            return 0
        document = validate_artifact_bindings(
            manifest_path=args.manifest,
            cache_off_sidecar_path=args.cache_off_sidecar,
            cache_on_sidecar_path=args.cache_on_sidecar,
            workload_path=args.workload,
            repo_root=args.repo_root,
        )
        print(
            f"valid {document['artifact_kind']}: {args.manifest}; "
            f"teacher_token_sha256={document['generation']['teacher_token_ids_le_u32_sha256']}"
        )
        return 0
    except (
        OSError,
        RuntimeError,
        CalibrationError,
        oracle.Qwen3BServingOracleError,
        Qwen3BGenerationTraceError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
