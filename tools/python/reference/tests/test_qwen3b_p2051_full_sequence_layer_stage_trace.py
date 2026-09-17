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

from riley_reference import qwen3b_generation_trace as generation
from riley_reference import qwen3b_p2051_full_sequence_layer_stage_trace as trace
from riley_reference import qwen3b_p2051_layer_stage_trace as base
from riley_reference import qwen3b_serving_oracle as oracle


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


def _teacher_prefix() -> base.TeacherPrefix:
    return base.TeacherPrefix(
        token_ids=(7, 8, 9),
        full_teacher_token_ids_sha256=_sha("a"),
        artifact_filename="teacher.json",
        artifact_sha256=_sha("b"),
        artifact_schema_version=generation.SCHEMA_VERSION,
        artifact_kind=generation.ARTIFACT_KIND,
        cache_off_sidecar_filename="teacher-cache-off.safetensors",
        cache_off_sidecar_sha256=_sha("c"),
        cache_off_sidecar_tensor_key=generation.SIDECAR_TENSOR_KEY,
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


def _tensor_document(
    raw_by_name: dict[str, bytes] | None = None,
) -> dict[str, object]:
    tensors: dict[str, object] = {}
    for index, name in enumerate(trace.TRACE_TENSORS):
        shape = trace._expected_shapes()[name]
        raw = (
            raw_by_name[name]
            if raw_by_name is not None
            else bytes([index + 1]) * (trace._shape_element_count(shape) * trace.BF16_BYTES)
        )
        tensors[name] = {
            "key": trace._sidecar_key(name),
            "shape": list(shape),
            "dtype": "bfloat16",
            "canonical_byte_order": "little-endian-u16",
            "bf16_le_sha256": hashlib.sha256(raw).hexdigest(),
            "bf16_le_bytes": len(raw),
        }
    return tensors


def _manifest(
    *,
    tensors: dict[str, object] | None = None,
    sidecar_name: str = "p2051-full.safetensors",
    sidecar_sha256: str | None = None,
) -> dict[str, object]:
    workload = _workload()
    return {
        "schema_version": trace.SCHEMA_VERSION,
        "artifact_kind": trace.ARTIFACT_KIND,
        "trace_id": trace.TRACE_ID,
        "performance_claim_eligible": False,
        "created_at": "2026-09-17T00:00:00Z",
        "producer": _producer(),
        "trace_profile": trace._capture_profile_document(),
        "contract": {
            "model_id": oracle.MODEL_ID,
            "model_revision": oracle.MODEL_REVISION,
            "workload": base._workload_document(workload),
            "execution": base._execution_document(),
            "input": base._teacher_input_document(workload, _teacher_prefix()),
        },
        "model": {
            "checkpoint_path": "/checkpoint",
            "checkpoint_receipt_filename": oracle.CHECKPOINT_RECEIPT_FILENAME,
            "checkpoint_receipt_sha256": _sha("1"),
        },
        "provenance": {"source_repository": _source_provenance()},
        "sidecar": {
            "path": sidecar_name,
            "sha256": sidecar_sha256 or _sha("2"),
            "format": "safetensors",
            "tensor_count": len(trace.TRACE_TENSORS),
        },
        "tensors": tensors or _tensor_document(),
    }


def _write_safetensors_sidecar(path: Path, raw_by_name: dict[str, bytes]) -> None:
    offset = 0
    header: dict[str, object] = {}
    with path.open("wb") as output:
        # Build the contiguous payload before emitting the header because the
        # test exercises the same ordinary safetensors layout as the oracle.
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
        output.write(len(encoded).to_bytes(8, "little"))
        output.write(encoded)
        output.write(payload)


class Qwen3BP2051FullSequenceLayerStageTraceTests(unittest.TestCase):
    def test_import_keeps_ml_dependencies_lazy(self) -> None:
        package_root = Path(__file__).resolve().parents[1]
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(package_root)
        command = (
            "import sys; import riley_reference.qwen3b_p2051_full_sequence_layer_stage_trace; "
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

    def test_schema_is_layer_one_full_sequence_only(self) -> None:
        self.assertEqual(trace.LAYER_INDEX, 1)
        self.assertEqual(len(trace.TRACE_TENSORS), 11)
        self.assertEqual(
            trace._expected_shapes()["layer1.q_proj.full"], (2_051, 2_048)
        )
        self.assertEqual(
            trace._expected_shapes()["layer1.k_proj.full"], (2_051, 256)
        )
        self.assertEqual(
            trace._expected_shapes()["layer1.q_rope.full"], (2_051, 16, 128)
        )
        self.assertEqual(
            trace._expected_shapes()["layer1.rope_cos.full"], (2_051, 128)
        )
        self.assertEqual(
            trace._capture_profile_document()["rust_consumer"]["trace_row_layout"],
            "full-sequence-token-major",
        )

    def test_manifest_accepts_fixed_contract_without_ml_dependencies(self) -> None:
        document = _manifest()
        trace.validate_manifest(document)
        self.assertFalse(document["performance_claim_eligible"])

        tampered = copy.deepcopy(document)
        tampered["trace_profile"]["layer_index"] = 2
        with self.assertRaisesRegex(
            trace.Qwen3BP2051FullSequenceLayerStageTraceError, "profile"
        ):
            trace.validate_manifest(tampered)

    def test_sidecar_validator_replays_full_sequence_hashes_and_ranges(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sidecar = root / "p2051-full.safetensors"
            raw_by_name = {
                name: bytes([index + 1])
                * (trace._shape_element_count(trace._expected_shapes()[name]) * trace.BF16_BYTES)
                for index, name in enumerate(trace.TRACE_TENSORS)
            }
            _write_safetensors_sidecar(sidecar, raw_by_name)
            document = _manifest(
                tensors=_tensor_document(raw_by_name),
                sidecar_name=sidecar.name,
                sidecar_sha256=hashlib.sha256(sidecar.read_bytes()).hexdigest(),
            )
            trace.validate_sidecar_against_manifest(document, sidecar)

            tampered = copy.deepcopy(document)
            tampered["tensors"]["layer1.output.full"]["bf16_le_sha256"] = _sha("0")
            with self.assertRaisesRegex(
                trace.Qwen3BP2051FullSequenceLayerStageTraceError, "raw BF16 hash"
            ):
                trace.validate_sidecar_against_manifest(tampered, sidecar)


if __name__ == "__main__":
    unittest.main()
