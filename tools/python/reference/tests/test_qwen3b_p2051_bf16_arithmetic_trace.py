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

from riley_reference import qwen3b_p2051_bf16_arithmetic_trace as trace
from riley_reference import qwen3b_serving_oracle as oracle

FIXED_TIME = "2026-09-16T06:00:00Z"


def _sha(character: str) -> str:
    return character * 64


def _raw_by_name() -> dict[str, bytes]:
    return {
        name: bytes([index]) * (trace._element_count(shape) * trace.BF16_BYTES)
        for index, (name, shape) in enumerate(trace._expected_shapes().items())
    }


def _tensor_document(raw_by_name: dict[str, bytes] | None = None) -> dict[str, object]:
    result: dict[str, object] = {}
    for index, (name, shape) in enumerate(trace._expected_shapes().items()):
        raw = raw_by_name[name] if raw_by_name is not None else bytes([index]) * (
            trace._element_count(shape) * trace.BF16_BYTES
        )
        result[name] = {
            "key": trace._sidecar_key(name),
            "shape": list(shape),
            "dtype": "bfloat16",
            "canonical_byte_order": "little-endian-u16",
            "bf16_le_bytes": len(raw),
            "bf16_le_sha256": hashlib.sha256(raw).hexdigest(),
        }
    return result


def _policies(raw_by_name: dict[str, bytes] | None = None) -> list[dict[str, object]]:
    policies: list[dict[str, object]] = []
    for index, policy in enumerate(trace.POLICIES):
        value = trace._policy_document(policy)
        tensor_name = f"raw_q/{policy.identifier}"
        raw = raw_by_name[tensor_name] if raw_by_name is not None else bytes([index + 3]) * (
            trace.M * trace.N * trace.BF16_BYTES
        )
        value.update(
            {
                "preferred_blas_actual": policy.preferred_blas,
                "allow_bf16_reduced_precision_reduction_actual": policy.allow_reduced_precision_reduction,
                "allow_bf16_reduced_precision_reduction_split_k_actual": policy.allow_split_k,
                "repeated_bf16_exact": True,
                "output_tensor": tensor_name,
                "output_bf16_le_sha256": hashlib.sha256(raw).hexdigest(),
                "p7_default_comparison": {
                    "bf16_exact": index == 0,
                    "unequal_elements": 0 if index == 0 else 1,
                    "total_elements": trace.M * trace.N,
                    "max_abs": 0.0 if index == 0 else 0.03125,
                },
            }
        )
        policies.append(value)
    return policies


def _source_provenance() -> dict[str, object]:
    return {
        "git_revision": "a" * 40,
        "source_dirty": False,
        "source_status_sha256": _sha("b"),
        "sources": {
            name: {"path": path, "sha256": _sha("c")}
            for name, path in trace.SOURCE_PATHS.items()
        },
    }


def _manifest(
    *,
    raw_by_name: dict[str, bytes] | None = None,
    sidecar_name: str = "p9.safetensors",
    sidecar_sha256: str | None = None,
) -> dict[str, object]:
    return {
        "schema_version": trace.SCHEMA_VERSION,
        "artifact_kind": trace.ARTIFACT_KIND,
        "trace_id": trace.TRACE_ID,
        "performance_claim_eligible": False,
        "vllm_comparison_eligible": False,
        "serving_selector_changed": False,
        "quality_pass": True,
        "created_at": FIXED_TIME,
        "scope": {
            "endpoint": "layer0.q_proj.raw_no_bias",
            "operator": "torch.nn.functional.linear",
            "model_id": oracle.MODEL_ID,
            "model_revision": oracle.MODEL_REVISION,
            "shape": {"m": trace.M, "n": trace.N, "k": trace.K},
            "offline_only": True,
            "serving_path": False,
        },
        "producer": {
            "implementation_id": trace.IMPLEMENTATION_ID,
            "runtime_dependency_class": "offline-python-reference",
            "torch_version": "2.13.0+cu129",
            "runtime_cuda_version": "12.9",
            "model_loader": {},
            "tf32_enabled": False,
        },
        "p7_binding": {
            "manifest_filename": "p7.json",
            "manifest_sha256": _sha("d"),
            "sidecar_filename": "p7.safetensors",
            "sidecar_sha256": _sha("e"),
            "checkpoint_receipt_sha256": _sha("f"),
            "source_revision": "1" * 40,
            "input_tensor_key": trace.P7_INPUT_KEY,
            "raw_q_tensor_key": trace.P7_RAW_Q_KEY,
        },
        "model": {
            "checkpoint_path": "/checkpoint",
            "checkpoint_receipt_filename": oracle.CHECKPOINT_RECEIPT_FILENAME,
            "checkpoint_receipt_sha256": _sha("f"),
        },
        "policies": _policies(raw_by_name),
        "comparisons": {
            "p7_default_vs_p7_shadow_raw_q": {
                "bf16_exact": True,
                "unequal_elements": 0,
                "total_elements": trace.M * trace.N,
                "max_abs": 0.0,
            },
            "policy_outputs_are_not_riley_results": True,
        },
        "provenance": {"source_repository": _source_provenance()},
        "sidecar": {
            "path": sidecar_name,
            "sha256": sidecar_sha256 or _sha("2"),
            "format": "safetensors",
            "tensor_count": len(trace.TRACE_TENSORS),
        },
        "tensors": _tensor_document(raw_by_name),
    }


def _write_safetensors_sidecar(path: Path, raw_by_name: dict[str, bytes]) -> None:
    header: dict[str, object] = {}
    payload = bytearray()
    offset = 0
    for name in trace.TRACE_TENSORS:
        raw = raw_by_name[name]
        header[trace._sidecar_key(name)] = {
            "dtype": "BF16",
            "shape": list(trace._expected_shapes()[name]),
            "data_offsets": [offset, offset + len(raw)],
        }
        payload.extend(raw)
        offset += len(raw)
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + payload)


class Qwen3BP2051Bf16ArithmeticTraceTests(unittest.TestCase):
    def test_import_keeps_ml_dependencies_lazy(self) -> None:
        package_root = Path(__file__).resolve().parents[1]
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(package_root)
        command = (
            "import sys; import riley_reference.qwen3b_p2051_bf16_arithmetic_trace; "
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

    def test_schema_has_three_explicit_blas_and_bf16_controls(self) -> None:
        self.assertEqual(
            [(policy.preferred_blas, policy.allow_reduced_precision_reduction, policy.allow_split_k) for policy in trace.POLICIES],
            [("cublas", True, True), ("cublas", False, True), ("cublaslt", False, False)],
        )
        self.assertEqual(trace._expected_shapes()[trace.P7_INPUT_NAME], (2_051, 2_048))
        self.assertEqual(trace._expected_shapes()[trace.Q_WEIGHT_NAME], (2_048, 2_048))
        self.assertEqual(trace._expected_shapes()[trace.P7_RAW_Q_NAME], (2_051, 2_048))
        self.assertEqual(trace.TRACE_TENSORS[-1], "raw_q/cublaslt_reduced_off_splitk_off")

    def test_manifest_accepts_fixed_offline_contract(self) -> None:
        document = _manifest()
        trace.validate_manifest(document)
        self.assertFalse(document["performance_claim_eligible"])
        self.assertFalse(document["vllm_comparison_eligible"])
        self.assertFalse(document["serving_selector_changed"])

    def test_manifest_rejects_policy_backend_or_output_tampering(self) -> None:
        document = _manifest()
        document["policies"][2]["preferred_blas_actual"] = "cublas"
        with self.assertRaisesRegex(trace.Qwen3BP2051Bf16ArithmeticTraceError, "policy observation"):
            trace.validate_manifest(document)

        document = _manifest()
        document["policies"][1]["output_tensor"] = "raw_q/not-allowed"
        with self.assertRaisesRegex(trace.Qwen3BP2051Bf16ArithmeticTraceError, "policy observation"):
            trace.validate_manifest(document)

    def test_manifest_rejects_serving_or_comparison_promotion(self) -> None:
        document = _manifest()
        document["performance_claim_eligible"] = True
        with self.assertRaisesRegex(trace.Qwen3BP2051Bf16ArithmeticTraceError, "identity"):
            trace.validate_manifest(document)

        document = _manifest()
        document["comparisons"]["policy_outputs_are_not_riley_results"] = False
        with self.assertRaisesRegex(trace.Qwen3BP2051Bf16ArithmeticTraceError, "comparison"):
            trace.validate_manifest(document)

    def test_sidecar_validator_replays_full_bf16_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sidecar = Path(directory) / "p9.safetensors"
            raw_by_name = _raw_by_name()
            _write_safetensors_sidecar(sidecar, raw_by_name)
            document = _manifest(
                raw_by_name=raw_by_name,
                sidecar_name=sidecar.name,
                sidecar_sha256=hashlib.sha256(sidecar.read_bytes()).hexdigest(),
            )
            trace.validate_sidecar_against_manifest(document, sidecar)

            tampered = copy.deepcopy(document)
            tampered["tensors"][trace.P7_RAW_Q_NAME]["bf16_le_sha256"] = _sha("0")
            with self.assertRaisesRegex(trace.Qwen3BP2051Bf16ArithmeticTraceError, "raw BF16 hash"):
                trace.validate_sidecar_against_manifest(tampered, sidecar)

    def test_output_paths_are_create_only_and_outside_repository(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = root / "repository"
            artifacts = root / "artifacts"
            repository.mkdir()
            artifacts.mkdir()
            existing = artifacts / "already.json"
            existing.write_text("already", encoding="utf-8")
            with self.assertRaisesRegex(trace.Qwen3BP2051Bf16ArithmeticTraceError, "overwrite"):
                trace._output_paths(existing, artifacts / "sidecar.safetensors", repository)
            with self.assertRaisesRegex(trace.Qwen3BP2051Bf16ArithmeticTraceError, "outside"):
                trace._output_paths(repository / "inside.json", repository / "inside.safetensors", repository)


if __name__ == "__main__":
    unittest.main()
