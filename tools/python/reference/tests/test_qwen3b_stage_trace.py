from __future__ import annotations

import copy
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from riley_reference import qwen3b_serving_oracle as oracle
from riley_reference import qwen3b_stage_trace as trace


def _sha(character: str) -> str:
    return character * 64


def _source_provenance() -> dict[str, object]:
    return {
        "git_revision": _sha("a"),
        "source_dirty": False,
        "source_status_sha256": _sha("b"),
        "sources": {
            name: {"path": path, "sha256": _sha("c")}
            for name, path in trace.SOURCE_PATHS.items()
        },
    }


def _manifest() -> dict[str, object]:
    tensors: dict[str, object] = {}
    for index, name in enumerate(trace.TRACE_TENSORS):
        shape = trace._expected_shapes()[name]
        tensors[name] = {
            "key": f"trace/{name.replace('.', '/')}",
            "shape": list(shape),
            "dtype": "bfloat16",
            "canonical_byte_order": "little-endian-u16",
            "bf16_le_sha256": _sha(format(index, "x")),
            "bf16_le_bytes": trace._shape_element_count(shape) * trace.BF16_BYTES,
        }
    return {
        "schema_version": trace.SCHEMA_VERSION,
        "artifact_kind": trace.ARTIFACT_KIND,
        "trace_id": trace.TRACE_ID,
        "performance_claim_eligible": False,
        "created_at_unix_seconds": 1,
        "producer": {
            "implementation_id": trace.IMPLEMENTATION_ID,
            "runtime_dependency_class": "offline-python-reference",
            "torch_version": "2.13.0",
            "transformers_version": "5.15.1",
            "transformers_qwen2_source": {
                "path": trace.TRANSFORMERS_QWEN2_SOURCE_PATH,
                "sha256": trace.TRANSFORMERS_QWEN2_SOURCE_SHA256,
            },
        },
        "trace_profile": {
            "capture_domain": "all-2048-token-positions",
            "id": trace.TRACE_ID,
            "tensor_count": len(trace.TRACE_TENSORS),
        },
        "contract": {
            "model_id": oracle.MODEL_ID,
            "model_revision": oracle.MODEL_REVISION,
            "workload": {
                "schema_version": oracle.WORKLOAD_SCHEMA_VERSION,
                "case": oracle.WORKLOAD_CASE,
                "source_sha256": oracle.WORKLOAD_SHA256,
                "prompt_token_count": oracle.PROMPT_TOKEN_COUNT,
                "prompt_token_ids_le_u32_sha256": oracle.PROMPT_TOKEN_IDS_SHA256,
            },
            "execution": trace._execution_document(),
        },
        "model": {
            "checkpoint_path": "/checkpoint",
            "checkpoint_receipt_filename": oracle.CHECKPOINT_RECEIPT_FILENAME,
            "checkpoint_receipt_sha256": _sha("d"),
        },
        "provenance": {"source_repository": _source_provenance()},
        "sidecar": {
            "path": "qwen3b-stage0.safetensors",
            "sha256": _sha("e"),
            "format": "safetensors",
            "tensor_count": len(trace.TRACE_TENSORS),
        },
        "tensors": tensors,
    }


class Qwen3BStageTraceTests(unittest.TestCase):
    def test_import_keeps_torch_transformers_and_safetensors_lazy(self) -> None:
        package_root = Path(__file__).resolve().parents[1]
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(package_root)
        command = (
            "import sys; import riley_reference.qwen3b_stage_trace; "
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

    def test_manifest_accepts_fixed_stage0_contract(self) -> None:
        document = _manifest()
        trace.validate_manifest(document)

    def test_manifest_rejects_rotary_shape_tampering(self) -> None:
        document = _manifest()
        document["tensors"]["layer0.q_rope"]["shape"] = [2_048, 2_048]
        with self.assertRaisesRegex(trace.Qwen3BStageTraceError, "shape"):
            trace.validate_manifest(document)

    def test_output_paths_reject_repository_before_checkpoint_or_backend(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(trace.Qwen3BStageTraceError, "outside"):
                trace.produce_hf_trace(
                    checkpoint_path=Path("/missing-checkpoint"),
                    workload_path=Path("/missing-workload"),
                    manifest_path=root / "artifact.json",
                    sidecar_path=root / "artifact.safetensors",
                    repo_root=root,
                    device="cuda:0",
                )

    def test_manifest_rejects_source_record_tampering(self) -> None:
        document = copy.deepcopy(_manifest())
        document["provenance"]["source_repository"]["sources"].pop("stage_trace")
        with self.assertRaisesRegex(trace.Qwen3BStageTraceError, "source"):
            trace.validate_manifest(document)


if __name__ == "__main__":
    unittest.main()
