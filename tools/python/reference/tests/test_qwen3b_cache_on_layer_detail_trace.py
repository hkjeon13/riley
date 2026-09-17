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

from riley_reference import qwen3b_cache_on_layer_detail_trace as trace
from riley_reference import qwen3b_generation_trace as generation
from riley_reference import qwen3b_serving_oracle as oracle

FIXED_TIME = "2026-09-18T03:04:05Z"


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


def _teacher() -> trace.cache_on.TeacherStream:
    return trace.cache_on.TeacherStream(
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
        * (
            trace._shape_element_count(trace._expected_shapes()[name])
            * trace.BF16_BYTES
        )
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


def _source_rows() -> dict[int, bytes]:
    return {
        0: b"\x01\x00" * (oracle.RAW_LOGIT_BYTES // 2),
        1: b"\x02\x00" * (oracle.RAW_LOGIT_BYTES // 2),
    }


def _manifest(
    *, raw_by_name: dict[str, bytes], sidecar_name: str, sidecar_sha256: str
) -> dict[str, object]:
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
            "input": trace.cache_on._teacher_input_document(workload, _teacher()),
            "source_logit_bindings": trace._source_logit_bindings(_source_rows()),
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


class Qwen3BCacheOnLayerDetailTraceTests(unittest.TestCase):
    def test_import_keeps_ml_dependencies_lazy(self) -> None:
        package_root = Path(__file__).resolve().parents[1]
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(package_root)
        command = (
            "import sys; import riley_reference.qwen3b_cache_on_layer_detail_trace; "
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

    def test_schema_is_fixed_to_layer_three_m1_boundaries(self) -> None:
        self.assertEqual(trace.DETAILED_LAYER_INDEX, 3)
        self.assertEqual(len(trace.DETAIL_TENSORS), 14)
        self.assertEqual(len(trace.TRACE_TENSORS), 15)
        self.assertEqual(
            trace.TRACE_TENSORS[0], "layer3.input_norm.last"
        )
        self.assertEqual(trace.TRACE_TENSORS[-1], "last_logits")
        profile = trace._capture_profile_document()
        self.assertEqual(profile["m1_source_logit_row"], 1)
        self.assertEqual(profile["rust_consumer"]["cache_layout"], "contiguous-kv-only")

    def test_manifest_rejects_different_layer_or_source_logit_row(self) -> None:
        raw = _raw_by_name()
        raw["last_logits"] = _source_rows()[1]
        document = _manifest(
            raw_by_name=raw,
            sidecar_name="layer-detail.safetensors",
            sidecar_sha256=_sha("2"),
        )
        trace.validate_manifest(document)

        tampered = copy.deepcopy(document)
        tampered["trace_profile"]["detailed_layer_index"] = 4
        with self.assertRaisesRegex(
            trace.Qwen3BCacheOnLayerDetailTraceError, "profile"
        ):
            trace.validate_manifest(tampered)

        tampered = copy.deepcopy(document)
        tampered["contract"]["source_logit_bindings"]["decode_step_1"][
            "source_logit_row"
        ] = 2
        with self.assertRaisesRegex(
            trace.Qwen3BCacheOnLayerDetailTraceError, "source logit"
        ):
            trace.validate_manifest(tampered)

    def test_sidecar_validator_replays_all_layer_three_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sidecar = Path(directory) / "layer-detail.safetensors"
            raw = _raw_by_name()
            raw["last_logits"] = _source_rows()[1]
            _write_sidecar(sidecar, raw)
            document = _manifest(
                raw_by_name=raw,
                sidecar_name=sidecar.name,
                sidecar_sha256=hashlib.sha256(sidecar.read_bytes()).hexdigest(),
            )
            trace.validate_sidecar_against_manifest(document, sidecar)

            tampered = copy.deepcopy(document)
            tampered["tensors"]["layer3.down_proj.last"]["bf16_le_sha256"] = _sha(
                "0"
            )
            with self.assertRaisesRegex(
                trace.Qwen3BCacheOnLayerDetailTraceError, "raw BF16"
            ):
                trace.validate_sidecar_against_manifest(tampered, sidecar)


if __name__ == "__main__":
    unittest.main()
