from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator
from unittest import mock

from riley_reference import qwen3b_generation_trace as trace
from riley_reference import qwen3b_serving_oracle as oracle

FIXED_TIME = datetime(2026, 9, 16, 2, 3, 4, tzinfo=timezone.utc)


def _checkpoint() -> oracle.CheckpointManifest:
    metadata = tuple(
        oracle.FileRecord(
            path=path,
            size_bytes=int(record["size_bytes"]),
            sha256=str(record["sha256"]),
        )
        for path, record in oracle.EXPECTED_CHECKPOINT_METADATA.items()
    )
    weights = tuple(
        oracle.FileRecord(
            path=f"model-{index:05d}-of-00002.safetensors",
            size_bytes=size_bytes,
            sha256=digest,
        )
        for index, (size_bytes, digest) in enumerate(
            sorted(oracle.EXPECTED_WEIGHT_FILES), start=1
        )
    )
    return oracle.CheckpointManifest(
        root=Path("/checkpoint"),
        receipt=oracle.FileRecord(
            path=oracle.CHECKPOINT_RECEIPT_FILENAME,
            size_bytes=oracle.CHECKPOINT_RECEIPT_BYTES,
            sha256=oracle.CHECKPOINT_RECEIPT_SHA256,
        ),
        files=tuple(sorted((*metadata, *weights), key=lambda record: record.path)),
    )


def _producer() -> dict[str, object]:
    return {
        "implementation_id": trace.IMPLEMENTATION_ID,
        "runtime_dependency_class": "offline-python-reference",
        "python_version": "3.13.15",
        "python_executable_sha256": "a" * 64,
        "python_platform_system": "linux",
        "python_platform_machine": "x86_64",
        "torch_version": "2.13.0",
        "transformers_version": "5.15.1",
        "safetensors_version": "0.8.0",
        "selected_device": {
            "requested": "cuda:0",
            "resolved": "cuda:0",
            "index": 0,
            "name": "test GPU",
            "compute_capability": "8.9",
            "driver_version": "580.178.04",
            "runtime_cuda_version": "13.0",
        },
        "transformers_model_source": {
            "module": "transformers.models.qwen2.modeling_qwen2",
            "filename": "modeling_qwen2.py",
            "sha256": "b" * 64,
        },
    }


def _source_provenance() -> dict[str, object]:
    return {
        "git_revision": "c" * 40,
        "source_dirty": False,
        "source_status_sha256": "d" * 64,
        "sources": {
            name: {"path": path, "sha256": "e" * 64}
            for name, path in trace.SOURCE_PATHS.items()
        },
    }


@contextmanager
def _two_row_contract() -> Iterator[None]:
    output_ids = (10, 11)
    with ExitStack() as stack:
        stack.enter_context(
            mock.patch.object(oracle, "OUTPUT_TOKEN_COUNT", len(output_ids))
        )
        stack.enter_context(
            mock.patch.object(
                oracle,
                "OUTPUT_TOKEN_IDS_SHA256",
                oracle._token_ids_sha256(output_ids),
            )
        )
        yield


def _workload() -> oracle.ServingWorkload:
    output_ids = (10, 11)
    return oracle.ServingWorkload(
        source_path=Path("/workload.json"),
        source_bytes=1,
        source_sha256=oracle.WORKLOAD_SHA256,
        case=oracle.WORKLOAD_CASE,
        prompt="pinned prompt",
        prompt_token_ids=(oracle.EXPECTED_PROMPT_TOKEN_ID,) * oracle.PROMPT_TOKEN_COUNT,
        output_token_ids=output_ids,
        output_text="unused by this oracle",
    )


def _raw_bf16_logits(
    selected_token_id: int, *, raw_argmax_token_id: int | None = None
) -> bytes:
    """Make one finite row whose stable addressable winner is explicit."""

    if not 0 <= selected_token_id < oracle.ADDRESSABLE_TOKEN_COUNT:
        raise ValueError("selected token must be addressable")
    raw = bytearray(b"\x80\xbf" * oracle.MODEL_VOCABULARY_SIZE)  # BF16 -1.0

    def write(token_id: int, bits: int) -> None:
        raw[token_id * 2 : token_id * 2 + 2] = bits.to_bytes(2, "little")

    write(selected_token_id, 0x3F80)  # BF16 1.0
    if raw_argmax_token_id is not None:
        write(raw_argmax_token_id, 0x4000)  # BF16 2.0
    return bytes(raw)


def _raw_rows(
    *, cache_on_selected_token_ids: tuple[int, ...] = (7, 8)
) -> dict[str, tuple[bytes, ...]]:
    return {
        trace.CACHE_OFF: tuple(_raw_bf16_logits(token_id) for token_id in (7, 8)),
        trace.CACHE_ON: tuple(
            _raw_bf16_logits(token_id) for token_id in cache_on_selected_token_ids
        ),
    }


def _logits(raw: bytes) -> trace.CapturedLogits:
    analysis = trace._analyze_bf16_le_logits(raw)
    return trace.CapturedLogits(
        tensor=object(),
        raw_bf16_le_sha256=hashlib.sha256(raw).hexdigest(),
        addressable_bf16_le_sha256=hashlib.sha256(
            raw[: oracle.ADDRESSABLE_LOGIT_BYTES]
        ).hexdigest(),
        non_addressable_bf16_le_sha256=hashlib.sha256(
            raw[oracle.ADDRESSABLE_LOGIT_BYTES :]
        ).hexdigest(),
        raw_argmax_token_id=analysis.raw_argmax_token_id,
        selected_token_id=analysis.selected_token_id,
        selected_logit_bf16_as_f32=analysis.selected_logit_bf16_as_f32,
        top_token_ids=analysis.top_token_ids,
        top_values_bf16_as_f32=analysis.top_values_bf16_as_f32,
    )


def _capture(
    raw_rows: dict[str, tuple[bytes, ...]],
) -> trace.GenerationCapture:
    teacher = (7, 8)
    off_plans = trace.teacher_forced_step_plan(trace.CACHE_OFF, teacher)
    on_plans = trace.teacher_forced_step_plan(trace.CACHE_ON, teacher)
    return trace.GenerationCapture(
        teacher_token_ids=teacher,
        cache_off_steps=tuple(
            trace.CapturedStep(
                plan=plan,
                logits=_logits(raw_rows[trace.CACHE_OFF][index]),
            )
            for index, plan in enumerate(off_plans)
        ),
        cache_on_steps=tuple(
            trace.CapturedStep(
                plan=plan,
                logits=_logits(raw_rows[trace.CACHE_ON][index]),
            )
            for index, plan in enumerate(on_plans)
        ),
        cache_off_logits=object(),
        cache_on_logits=object(),
    )


def _artifact(
    *,
    cache_off_path: Path,
    cache_on_path: Path,
    raw_rows: dict[str, tuple[bytes, ...]],
) -> dict[str, object]:
    return trace.build_generation_artifact(
        workload=_workload(),
        checkpoint=_checkpoint(),
        capture=_capture(raw_rows),
        producer_metadata=_producer(),
        source_provenance=_source_provenance(),
        cache_off_sidecar_path=cache_off_path,
        cache_on_sidecar_path=cache_on_path,
        cache_off_sidecar_sha256=(
            hashlib.sha256(cache_off_path.read_bytes()).hexdigest()
            if cache_off_path.exists()
            else "f" * 64
        ),
        cache_on_sidecar_sha256=(
            hashlib.sha256(cache_on_path.read_bytes()).hexdigest()
            if cache_on_path.exists()
            else "0" * 64
        ),
        created_at=FIXED_TIME,
    )


def _write_safetensors_sidecar(path: Path, raw_rows: tuple[bytes, ...]) -> None:
    payload = b"".join(raw_rows)
    header = {
        trace.SIDECAR_TENSOR_KEY: {
            "dtype": "BF16",
            "shape": [len(raw_rows), oracle.MODEL_VOCABULARY_SIZE],
            "data_offsets": [0, len(payload)],
        }
    }
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + payload)


class Qwen3BGenerationTraceTests(unittest.TestCase):
    def test_import_keeps_torch_transformers_and_safetensors_lazy(self) -> None:
        package_root = Path(__file__).resolve().parents[1]
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(package_root)
        command = (
            "import sys; import riley_reference.qwen3b_generation_trace; "
            "assert 'torch' not in sys.modules; "
            "assert 'transformers' not in sys.modules; "
            "assert 'safetensors' not in sys.modules"
        )
        result = subprocess.run(
            [sys.executable, "-c", command],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_cache_on_plan_places_first_decode_at_2048_and_last_at_2174(self) -> None:
        teacher = tuple(range(100, 100 + oracle.OUTPUT_TOKEN_COUNT))
        plans = trace.teacher_forced_step_plan(trace.CACHE_ON, teacher)
        self.assertEqual(len(plans), oracle.OUTPUT_TOKEN_COUNT)
        self.assertEqual(plans[0].call_input_token_ids.__len__(), 2_048)
        self.assertEqual(plans[0].position_start, 0)
        self.assertEqual(plans[0].position_end, 2_047)
        self.assertEqual(plans[0].cache_length_after, 2_048)
        self.assertEqual(plans[1].call_input_token_ids, (teacher[0],))
        self.assertEqual(plans[1].teacher_input_token_id, teacher[0])
        self.assertEqual(plans[1].cache_length_before, 2_048)
        self.assertEqual(plans[1].position_start, 2_048)
        self.assertEqual(plans[1].position_end, 2_048)
        self.assertEqual(plans[-1].call_input_token_ids, (teacher[-2],))
        self.assertEqual(plans[-1].position_start, 2_174)
        self.assertEqual(plans[-1].position_end, 2_174)
        self.assertEqual(plans[-1].cache_length_after, 2_175)

    def test_bf16_analyzer_is_stable_and_rejects_nonfinite_values(self) -> None:
        raw_argmax = oracle.ADDRESSABLE_TOKEN_COUNT
        raw = _raw_bf16_logits(7, raw_argmax_token_id=raw_argmax)
        analysis = trace._analyze_bf16_le_logits(raw)
        self.assertEqual(analysis.raw_argmax_token_id, raw_argmax)
        self.assertEqual(analysis.selected_token_id, 7)
        self.assertEqual(analysis.selected_logit_bf16_as_f32, 1.0)
        self.assertEqual(analysis.top_token_ids[:4], (7, 0, 1, 2))
        self.assertEqual(analysis.top_values_bf16_as_f32[:4], (1.0, -1.0, -1.0, -1.0))

        nonfinite = bytearray(raw)
        nonfinite[3 * 2 : 3 * 2 + 2] = (0x7F80).to_bytes(2, "little")
        with self.assertRaisesRegex(trace.Qwen3BGenerationTraceError, "non-finite"):
            trace._analyze_bf16_le_logits(bytes(nonfinite))

    def test_manifest_accepts_fixed_teacher_forced_contract_without_ml_dependencies(
        self,
    ) -> None:
        with _two_row_contract(), tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_rows = _raw_rows()
            document = _artifact(
                cache_off_path=root / "cache-off.safetensors",
                cache_on_path=root / "cache-on.safetensors",
                raw_rows=raw_rows,
            )
            trace.validate_generation_artifact(document)
            self.assertFalse(document["performance_claim_eligible"])
            self.assertEqual(
                document["generation"]["cache_on"]["steps"][1]["position_start"],
                2_048,
            )
            self.assertFalse(
                document["contract"]["workload"][
                    "workload_output_token_ids_used_as_teacher"
                ]
            )

    def test_cache_on_output_divergence_is_recorded_not_rejected(self) -> None:
        with _two_row_contract(), tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_rows = _raw_rows(cache_on_selected_token_ids=(7, 9))
            document = _artifact(
                cache_off_path=root / "cache-off.safetensors",
                cache_on_path=root / "cache-on.safetensors",
                raw_rows=raw_rows,
            )
            trace.validate_generation_artifact(document)
            row = document["generation"]["cache_on"]["steps"][1]["logits"]
            self.assertEqual(row["cache_off_teacher_token_id"], 8)
            self.assertEqual(row["selected_token_id"], 9)
            self.assertFalse(row["selection_matches_cache_off_teacher"])

    def test_manifest_rejects_cache_position_teacher_sidecar_and_source_tampering(
        self,
    ) -> None:
        with _two_row_contract(), tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_rows = _raw_rows()
            document = _artifact(
                cache_off_path=root / "cache-off.safetensors",
                cache_on_path=root / "cache-on.safetensors",
                raw_rows=raw_rows,
            )
            tampered = copy.deepcopy(document)
            tampered["generation"]["cache_on"]["steps"][1]["position_start"] = 2_049
            with self.assertRaisesRegex(
                trace.Qwen3BGenerationTraceError, "position_start"
            ):
                trace.validate_generation_artifact(tampered)

            tampered = copy.deepcopy(document)
            tampered["generation"]["teacher_token_ids"][1] = 9
            tampered["generation"]["teacher_token_ids_le_u32_sha256"] = (
                oracle._token_ids_sha256(tampered["generation"]["teacher_token_ids"])
            )
            with self.assertRaises(trace.Qwen3BGenerationTraceError):
                trace.validate_generation_artifact(tampered)

            tampered = copy.deepcopy(document)
            tampered["generation"]["cache_off"]["sidecar"][
                "path"
            ] = "../escape.safetensors"
            with self.assertRaisesRegex(
                trace.Qwen3BGenerationTraceError, "sidecar.path"
            ):
                trace.validate_generation_artifact(tampered)

            tampered = copy.deepcopy(document)
            tampered["provenance"]["source_repository"]["sources"].pop("hf_calibration")
            with self.assertRaisesRegex(
                trace.Qwen3BGenerationTraceError, "source record"
            ):
                trace.validate_generation_artifact(tampered)

    def test_stdlib_sidecar_validation_detects_raw_row_tampering(self) -> None:
        with _two_row_contract(), tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_rows = _raw_rows()
            off_path = root / "cache-off.safetensors"
            on_path = root / "cache-on.safetensors"
            _write_safetensors_sidecar(off_path, raw_rows[trace.CACHE_OFF])
            _write_safetensors_sidecar(on_path, raw_rows[trace.CACHE_ON])
            document = _artifact(
                cache_off_path=off_path,
                cache_on_path=on_path,
                raw_rows=raw_rows,
            )
            trace.validate_sidecar_against_artifact(
                document=document,
                mode=trace.CACHE_OFF,
                sidecar_path=off_path,
            )
            trace.validate_sidecar_against_artifact(
                document=document,
                mode=trace.CACHE_ON,
                sidecar_path=on_path,
            )

            payload = bytearray(on_path.read_bytes())
            header_size = int.from_bytes(payload[:8], "little")
            payload[8 + header_size] ^= 0xFF
            on_path.write_bytes(payload)
            tampered = copy.deepcopy(document)
            tampered["generation"]["cache_on"]["sidecar"]["sha256"] = hashlib.sha256(
                payload
            ).hexdigest()
            with self.assertRaisesRegex(
                trace.Qwen3BGenerationTraceError, "raw BF16 hash"
            ):
                trace.validate_sidecar_against_artifact(
                    document=tampered,
                    mode=trace.CACHE_ON,
                    sidecar_path=on_path,
                )

    def test_stdlib_sidecar_validation_rejects_metadata_tampering(self) -> None:
        with _two_row_contract(), tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_rows = _raw_rows()
            off_path = root / "cache-off.safetensors"
            on_path = root / "cache-on.safetensors"
            _write_safetensors_sidecar(off_path, raw_rows[trace.CACHE_OFF])
            _write_safetensors_sidecar(on_path, raw_rows[trace.CACHE_ON])
            document = _artifact(
                cache_off_path=off_path,
                cache_on_path=on_path,
                raw_rows=raw_rows,
            )
            tampered = copy.deepcopy(document)
            logits = tampered["generation"]["cache_on"]["steps"][1]["logits"]
            logits["selected_token_id"] = 33
            logits["top_token_ids"][0] = 33
            logits["selection_matches_cache_off_teacher"] = False
            trace.validate_generation_artifact(tampered)

            with self.assertRaisesRegex(
                trace.Qwen3BGenerationTraceError, "selected token differs from sidecar"
            ):
                trace.validate_sidecar_against_artifact(
                    document=tampered,
                    mode=trace.CACHE_ON,
                    sidecar_path=on_path,
                )

    def test_produce_preflight_rejects_existing_output_before_model_or_checkpoint(
        self,
    ) -> None:
        repository = Path(__file__).resolve().parents[4]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "artifact.json"
            manifest.write_text("already exists", encoding="utf-8")
            with self.assertRaisesRegex(trace.Qwen3BGenerationTraceError, "overwrite"):
                trace.produce_generation_trace(
                    checkpoint_path=Path("/missing-checkpoint"),
                    workload_path=Path("/missing-workload"),
                    manifest_path=manifest,
                    cache_off_sidecar_path=root / "cache-off.safetensors",
                    cache_on_sidecar_path=root / "cache-on.safetensors",
                    repo_root=repository,
                    device="cuda:0",
                )


if __name__ == "__main__":
    unittest.main()
