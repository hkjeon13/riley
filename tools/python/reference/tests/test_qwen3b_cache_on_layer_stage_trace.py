from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from riley_reference import qwen3b_cache_on_layer_stage_trace as trace
from riley_reference import qwen3b_generation_trace as generation
from riley_reference import qwen3b_serving_oracle as oracle

FIXED_TIME = "2026-09-17T03:04:05Z"


def _sha(character: str) -> str:
    return character * 64


def _workload() -> oracle.ServingWorkload:
    return oracle.ServingWorkload(
        source_path=Path("/workload.json"),
        source_bytes=1,
        source_sha256=oracle.WORKLOAD_SHA256,
        case=oracle.WORKLOAD_CASE,
        prompt="pinned prompt",
        prompt_token_ids=(oracle.EXPECTED_PROMPT_TOKEN_ID,) * oracle.PROMPT_TOKEN_COUNT,
        output_token_ids=(1,) * oracle.OUTPUT_TOKEN_COUNT,
        output_text="unused by this trace",
    )


def _teacher() -> trace.TeacherStream:
    return trace.TeacherStream(
        token_ids=(7, 8) + (9,) * (oracle.OUTPUT_TOKEN_COUNT - 2),
        full_teacher_token_ids_sha256=_sha("a"),
        artifact_filename="teacher.json",
        artifact_sha256=_sha("b"),
        artifact_schema_version=generation.SCHEMA_VERSION,
        artifact_kind=generation.ARTIFACT_KIND,
        cache_on_sidecar_filename="teacher-cache-on.safetensors",
        cache_on_sidecar_sha256=_sha("c"),
        cache_on_sidecar_tensor_key=generation.SIDECAR_TENSOR_KEY,
    )


def _producer() -> dict[str, object]:
    return {
        "implementation_id": trace.IMPLEMENTATION_ID,
        "runtime_dependency_class": "offline-python-reference",
        "torch_version": "2.13.0",
        "transformers_version": "5.15.1",
        "transformers_qwen2_source": {
            "path": trace.stage.TRANSFORMERS_QWEN2_SOURCE_PATH,
            "sha256": trace.stage.TRANSFORMERS_QWEN2_SOURCE_SHA256,
        },
    }


def _source_provenance() -> dict[str, object]:
    return {
        "git_revision": "d" * 40,
        "source_dirty": False,
        "source_status_sha256": _sha("e"),
        "sources": {
            name: {"path": path, "sha256": _sha("f")}
            for name, path in trace.SOURCE_PATHS.items()
        },
    }


def _raw_by_name() -> dict[str, bytes]:
    return {
        name: bytes([index % 251])
        * (trace._shape_element_count(trace._expected_shapes()[name]) * trace.BF16_BYTES)
        for index, name in enumerate(trace.TRACE_TENSORS)
    }


def _tensor_document(raw_by_name: dict[str, bytes]) -> dict[str, object]:
    return {
        name: {
            "key": trace._sidecar_key(name),
            "shape": list(trace._expected_shapes()[name]),
            "dtype": "bfloat16",
            "canonical_byte_order": "little-endian-u16",
            "bf16_le_sha256": hashlib.sha256(raw).hexdigest(),
            "bf16_le_bytes": len(raw),
        }
        for name, raw in raw_by_name.items()
    }


def _manifest(*, raw_by_name: dict[str, bytes], sidecar_name: str, sidecar_sha256: str) -> dict[str, object]:
    workload = _workload()
    return {
        "schema_version": trace.SCHEMA_VERSION,
        "artifact_kind": trace.ARTIFACT_KIND,
        "trace_id": trace.TRACE_ID,
        "performance_claim_eligible": False,
        "created_at": FIXED_TIME,
        "producer": _producer(),
        "trace_profile": trace._capture_profile_document(),
        "contract": {
            "model_id": oracle.MODEL_ID,
            "model_revision": oracle.MODEL_REVISION,
            "workload": trace.cache_free._workload_document(workload),
            "execution": trace._execution_document(),
            "input": trace._teacher_input_document(workload, _teacher()),
        },
        "model": {
            "checkpoint_path": "/checkpoint",
            "checkpoint_receipt_filename": oracle.CHECKPOINT_RECEIPT_FILENAME,
            "checkpoint_receipt_sha256": _sha("1"),
        },
        "provenance": {"source_repository": _source_provenance()},
        "sidecar": {
            "path": sidecar_name,
            "sha256": sidecar_sha256,
            "format": "safetensors",
            "tensor_count": len(trace.TRACE_TENSORS),
        },
        "tensors": _tensor_document(raw_by_name),
    }


def _write_sidecar(path: Path, raw_by_name: dict[str, bytes]) -> None:
    offset = 0
    header: dict[str, object] = {}
    payload = bytearray()
    for name in trace.TRACE_TENSORS:
        raw = raw_by_name[name]
        header[trace._sidecar_key(name)] = {
            "dtype": "BF16",
            "shape": list(trace._expected_shapes()[name]),
            "data_offsets": [offset, offset + len(raw)],
        }
        offset += len(raw)
        payload.extend(raw)
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + payload)


class Qwen3BCacheOnLayerStageTraceTests(unittest.TestCase):
    def test_import_keeps_ml_dependencies_lazy(self) -> None:
        package_root = Path(__file__).resolve().parents[1]
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(package_root)
        command = (
            "import sys; import riley_reference.qwen3b_cache_on_layer_stage_trace; "
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

    def test_schema_covers_p2048_prefill_and_two_m1_decodes(self) -> None:
        self.assertEqual(len(trace.TRACE_STEPS), 3)
        self.assertEqual(len(trace.TRACE_TENSORS), 156)
        self.assertEqual(
            [step.cache_length_after for step in trace.TRACE_STEPS], [2048, 2049, 2050]
        )
        self.assertEqual(
            [step.teacher_decode_token_index for step in trace.TRACE_STEPS], [None, 0, 1]
        )
        profile = trace._capture_profile_document()
        self.assertEqual(profile["tensor_count_per_step"], 52)
        self.assertEqual(profile["future_rust_quality_consumer"]["cache_layout"], "contiguous-kv-only")

    def test_manifest_rejects_mismatched_teacher_or_execution_contract(self) -> None:
        raw = _raw_by_name()
        document = _manifest(
            raw_by_name=raw,
            sidecar_name="cache-on.safetensors",
            sidecar_sha256=_sha("2"),
        )
        trace.validate_manifest(document)

        tampered = copy.deepcopy(document)
        tampered["contract"]["input"]["teacher_decode_token_ids"][1] = 33
        with self.assertRaisesRegex(trace.Qwen3BCacheOnLayerStageTraceError, "SHA-256"):
            trace.validate_manifest(tampered)

        tampered = copy.deepcopy(document)
        tampered["contract"]["execution"]["use_cache"] = False
        with self.assertRaisesRegex(trace.Qwen3BCacheOnLayerStageTraceError, "execution"):
            trace.validate_manifest(tampered)

    def test_sidecar_validator_replays_all_steps_and_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sidecar = root / "cache-on.safetensors"
            raw = _raw_by_name()
            _write_sidecar(sidecar, raw)
            document = _manifest(
                raw_by_name=raw,
                sidecar_name=sidecar.name,
                sidecar_sha256=hashlib.sha256(sidecar.read_bytes()).hexdigest(),
            )
            trace.validate_sidecar_against_manifest(document, sidecar)

            tampered = copy.deepcopy(document)
            tampered["tensors"]["decode_step_2.last_logits"]["bf16_le_sha256"] = _sha("0")
            with self.assertRaisesRegex(trace.Qwen3BCacheOnLayerStageTraceError, "raw BF16"):
                trace.validate_sidecar_against_manifest(tampered, sidecar)

    def test_trace_logits_are_checked_against_the_source_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sidecar = root / "cache-on.safetensors"
            raw = _raw_by_name()
            # Replace only last-logits tensors with unique finite source rows.
            source_rows: dict[int, bytes] = {}
            for index, step in enumerate(trace.TRACE_STEPS):
                row = bytes([index + 1, 0]) * (oracle.RAW_LOGIT_BYTES // 2)
                raw[f"{step.name}.last_logits"] = row
                source_rows[step.source_logit_row] = row
            _write_sidecar(sidecar, raw)
            document = _manifest(
                raw_by_name=raw,
                sidecar_name=sidecar.name,
                sidecar_sha256=hashlib.sha256(sidecar.read_bytes()).hexdigest(),
            )
            trace.validate_sidecar_against_manifest(document, sidecar)
            trace._validate_trace_logits_against_source(
                manifest=document, sidecar_path=sidecar, source_rows=source_rows
            )
            source_rows[2] = b"\x00\x00" * (oracle.RAW_LOGIT_BYTES // 2)
            with self.assertRaisesRegex(trace.Qwen3BCacheOnLayerStageTraceError, "decode_step_2"):
                trace._validate_trace_logits_against_source(
                    manifest=document, sidecar_path=sidecar, source_rows=source_rows
                )


if __name__ == "__main__":
    unittest.main()
