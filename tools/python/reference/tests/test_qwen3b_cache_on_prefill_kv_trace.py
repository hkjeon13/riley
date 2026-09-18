from __future__ import annotations

import copy
import hashlib
import os
import subprocess
import sys
import unittest
from pathlib import Path

from riley_reference import qwen3b_cache_on_layer_stage_trace as cache_on
from riley_reference import qwen3b_generation_trace as generation
from riley_reference import qwen3b_p2051_layer_stage_trace as cache_free
from riley_reference import qwen3b_cache_on_prefill_kv_trace as trace
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


def _teacher() -> cache_on.TeacherStream:
    return cache_on.TeacherStream(
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


def _manifest() -> dict[str, object]:
    shape = trace._expected_shapes()[trace.TRACE_TENSORS[0]]
    byte_count = trace._shape_element_count(shape) * trace.BF16_BYTES
    raw = b"\x00\x00" * (byte_count // 2)
    tensor = {
        "shape": list(shape),
        "dtype": "bfloat16",
        "canonical_byte_order": "little-endian-u16",
        "bf16_le_sha256": hashlib.sha256(raw).hexdigest(),
        "bf16_le_bytes": byte_count,
    }
    workload = _workload()
    teacher = _teacher()
    return {
        "schema_version": trace.SCHEMA_VERSION,
        "artifact_kind": trace.ARTIFACT_KIND,
        "trace_id": trace.TRACE_ID,
        "performance_claim_eligible": False,
        "created_at": "2026-09-18T03:04:05Z",
        "producer": _producer(),
        "trace_profile": trace._capture_profile_document(),
        "contract": {
            "model_id": oracle.MODEL_ID,
            "model_revision": oracle.MODEL_REVISION,
            "workload": cache_free._workload_document(workload),
            "execution": trace._execution_document(),
            "input": cache_on._teacher_input_document(workload, teacher),
            "source_logit_binding": {
                "source_logit_row": 0,
                "bf16_le_sha256": _sha("1"),
            },
        },
        "model": {
            "checkpoint_path": "/checkpoint",
            "checkpoint_receipt_filename": oracle.CHECKPOINT_RECEIPT_FILENAME,
            "checkpoint_receipt_sha256": _sha("2"),
        },
        "provenance": {"source_repository": _source_provenance()},
        "sidecar": {
            "path": "prefill-kv.safetensors",
            "sha256": _sha("3"),
            "format": "safetensors",
            "tensor_count": len(trace.TRACE_TENSORS),
        },
        "tensors": {
            name: {"key": trace._sidecar_key(name), **tensor}
            for name in trace.TRACE_TENSORS
        },
    }


class Qwen3BCacheOnPrefillKvTraceTests(unittest.TestCase):
    def test_import_keeps_ml_dependencies_lazy(self) -> None:
        package_root = Path(__file__).resolve().parents[1]
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(package_root)
        command = (
            "import sys; import riley_reference.qwen3b_cache_on_prefill_kv_trace; "
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

    def test_contract_captures_a_bounded_layer_prefix(self) -> None:
        self.assertEqual(trace.PREFIX_LAYER_COUNT, 14)
        self.assertEqual(len(trace.TRACE_TENSORS), 28)
        self.assertEqual(trace.TRACE_TENSORS[0], "prefill.layer0.key")
        self.assertEqual(trace.TRACE_TENSORS[-1], "prefill.layer13.value")
        self.assertEqual(
            trace._expected_shapes()["prefill.layer0.key"],
            (
                cache_free.MODEL_KEY_VALUE_HEAD_COUNT,
                oracle.PROMPT_TOKEN_COUNT,
                cache_free.MODEL_HEAD_DIMENSION,
            ),
        )

    def test_manifest_requires_the_fixed_dynamic_cache_contract(self) -> None:
        document = _manifest()
        trace.validate_manifest(document)

        tampered = copy.deepcopy(document)
        tampered["trace_profile"]["selected_layer_indices"] = list(range(13))
        with self.assertRaisesRegex(
            trace.Qwen3BCacheOnPrefillKvTraceError, "profile"
        ):
            trace.validate_manifest(tampered)

        tampered = copy.deepcopy(document)
        tampered["tensors"]["prefill.layer0.key"]["shape"][0] += 1
        with self.assertRaisesRegex(
            trace.Qwen3BCacheOnPrefillKvTraceError, "metadata"
        ):
            trace.validate_manifest(tampered)


if __name__ == "__main__":
    unittest.main()
