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

from riley_reference import qwen3b_p2051_qkv_default_cublas_trace as trace
from riley_reference import qwen3b_serving_oracle as oracle

FIXED_TIME = "2026-09-17T01:02:03Z"


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


def _metric(total: int) -> dict[str, object]:
    return {
        "bf16_exact": True,
        "unequal_elements": 0,
        "total_elements": total,
        "max_abs": 0.0,
    }


def _projection_results(tensors: dict[str, object]) -> dict[str, object]:
    result: dict[str, object] = {}
    for item in trace.PROJECTIONS:
        result[item.identifier] = {
            "identifier": item.identifier,
            "module_name": item.module_name,
            "weight_tensor": item.weight_name,
            "output_tensor": item.output_name,
            "checkpoint_weight_key": item.checkpoint_weight_key,
            "shape": {"m": trace.M, "n": item.width, "k": trace.K},
            "output_bf16_le_sha256": tensors[item.output_name]["bf16_le_sha256"],
            "repeated_bf16_exact": True,
            "p7_default_raw_q_comparison": _metric(trace.M * trace.Q_WIDTH)
            if item.identifier == "q"
            else None,
        }
    return result


def _manifest(
    *,
    raw_by_name: dict[str, bytes] | None = None,
    sidecar_name: str = "p11.safetensors",
    sidecar_sha256: str | None = None,
) -> dict[str, object]:
    tensors = _tensor_document(raw_by_name)
    q_metric = _metric(trace.M * trace.Q_WIDTH)
    policy = trace._default_policy_document()
    policy.update(
        {
            "preferred_blas_actual": "cublas",
            "allow_bf16_reduced_precision_reduction_actual": True,
            "allow_bf16_reduced_precision_reduction_split_k_actual": True,
            "matmul_tf32_actual": False,
            "cudnn_tf32_actual": False,
        }
    )
    return {
        "schema_version": trace.SCHEMA_VERSION,
        "artifact_kind": trace.ARTIFACT_KIND,
        "trace_id": trace.TRACE_ID,
        "performance_claim_eligible": False,
        "vllm_comparison_eligible": False,
        "serving_selector_changed": False,
        "capture_status": "captured",
        "quality_pass": True,
        "created_at": FIXED_TIME,
        "scope": {
            "endpoints": [
                "layer0.q_proj.raw_no_bias",
                "layer0.k_proj.raw_no_bias",
                "layer0.v_proj.raw_no_bias",
            ],
            "operator": "torch.nn.functional.linear",
            "model_id": oracle.MODEL_ID,
            "model_revision": oracle.MODEL_REVISION,
            "shape": {"m": trace.M, "k": trace.K, "q_n": trace.Q_WIDTH, "kv_n": trace.KV_WIDTH},
            "offline_only": True,
            "serving_path": False,
        },
        "producer": {
            "implementation_id": trace.IMPLEMENTATION_ID,
            "runtime_dependency_class": "offline-python-reference",
            "torch_version": "2.13.0",
            "runtime_cuda_version": "12.8",
            "model_loader": {"kind": "test"},
            "tf32_enabled": False,
            "cudnn_tf32_enabled": False,
        },
        "default_policy": policy,
        "p7_binding": {
            "manifest_filename": "p7.json",
            "manifest_sha256": _sha("1"),
            "sidecar_filename": "p7.safetensors",
            "sidecar_sha256": _sha("2"),
            "checkpoint_receipt_sha256": _sha("3"),
            "source_revision": "4" * 40,
            "input_tensor_key": trace.P7_INPUT_KEY,
            "raw_q_tensor_key": trace.P7_RAW_Q_KEY,
            "input_bf16_le_sha256": _sha("5"),
            "raw_q_bf16_le_sha256": _sha("6"),
        },
        "model": {
            "checkpoint_path": "/checkpoint",
            "checkpoint_receipt_filename": oracle.CHECKPOINT_RECEIPT_FILENAME,
            "checkpoint_receipt_sha256": _sha("7"),
        },
        "projection_results": _projection_results(tensors),
        "comparisons": {
            "q_default_vs_p7_shadow_raw_q": q_metric,
            "outputs_are_not_riley_results": True,
        },
        "provenance": {"source_repository": _source_provenance()},
        "sidecar": {
            "path": sidecar_name,
            "sha256": sidecar_sha256 or _sha("8"),
            "format": "safetensors",
            "tensor_count": len(trace.TRACE_TENSORS),
        },
        "tensors": tensors,
    }


def _write_safetensors_sidecar(
    path: Path, raw_by_name: dict[str, bytes], order: tuple[str, ...] | None = None
) -> None:
    ordered = order or trace.TRACE_TENSORS
    header: dict[str, object] = {}
    payload = bytearray()
    offset = 0
    for name in ordered:
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


class _FakeMatmul:
    def __init__(self) -> None:
        self._reduced = False
        self._split_k = False
        self.allow_tf32 = True

    @property
    def allow_bf16_reduced_precision_reduction(self) -> bool:
        return self._reduced

    @allow_bf16_reduced_precision_reduction.setter
    def allow_bf16_reduced_precision_reduction(self, value: tuple[bool, bool]) -> None:
        if not isinstance(value, tuple) or len(value) != 2 or not all(isinstance(item, bool) for item in value):
            raise AssertionError("P11 must assign the PyTorch BF16 policy as a bool tuple")
        self._reduced, self._split_k = value

    @property
    def allow_bf16_reduced_precision_reduction_split_k(self) -> bool:
        return self._split_k


class _FakeCuda:
    def __init__(self) -> None:
        self.matmul = _FakeMatmul()
        self._preferred = "cublaslt"

    def preferred_blas_library(self, requested: str | None = None) -> str:
        if requested is not None:
            self._preferred = requested
        return "_BlasBackend.Cublaslt" if self._preferred == "cublaslt" else "_BlasBackend.Cublas"


class _FakeCudnn:
    def __init__(self) -> None:
        self.allow_tf32 = True


class _FakeTorch:
    def __init__(self) -> None:
        self.backends = type("Backends", (), {"cuda": _FakeCuda(), "cudnn": _FakeCudnn()})()


class Qwen3BP2051QkvDefaultCublasTraceTests(unittest.TestCase):
    def test_import_keeps_ml_dependencies_lazy(self) -> None:
        package_root = Path(__file__).resolve().parents[1]
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(package_root)
        command = (
            "import sys; import riley_reference.qwen3b_p2051_qkv_default_cublas_trace; "
            "assert 'torch' not in sys.modules; "
            "assert 'transformers' not in sys.modules; "
            "assert 'safetensors' not in sys.modules"
        )
        result = subprocess.run(
            [sys.executable, "-c", command], check=False, capture_output=True, text=True, env=environment
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_schema_pins_q_k_v_default_cublas_shapes(self) -> None:
        self.assertEqual(trace.TRACE_TENSORS, (
            "p7_input_norm", "p7_shadow_raw_q", "layer0_q_proj_weight", "raw_q/default_cublas_reduced_splitk",
            "layer0_k_proj_weight", "raw_k/default_cublas_reduced_splitk", "layer0_v_proj_weight",
            "raw_v/default_cublas_reduced_splitk",
        ))
        self.assertEqual(trace._expected_shapes()["raw_q/default_cublas_reduced_splitk"], (2_051, 2_048))
        self.assertEqual(trace._expected_shapes()["raw_k/default_cublas_reduced_splitk"], (2_051, 256))
        self.assertEqual(trace._expected_shapes()["raw_v/default_cublas_reduced_splitk"], (2_051, 256))

    def test_default_policy_is_applied_and_restored_as_a_bool_tuple(self) -> None:
        fake = _FakeTorch()
        actual = trace._apply_default_policy(fake)
        self.assertEqual(actual["preferred_blas_actual"], "cublas")
        self.assertTrue(actual["allow_bf16_reduced_precision_reduction_actual"])
        self.assertTrue(actual["allow_bf16_reduced_precision_reduction_split_k_actual"])
        self.assertFalse(actual["matmul_tf32_actual"])
        self.assertFalse(actual["cudnn_tf32_actual"])
        trace._restore_policy(fake, "cublaslt", (False, False), (True, True))
        self.assertEqual(trace._backend_name(fake), "cublaslt")
        self.assertEqual(trace._policy_readback(fake), (False, False))
        self.assertEqual(trace._tf32_readback(fake), (True, True))

    def test_manifest_accepts_fixed_offline_contract(self) -> None:
        document = _manifest()
        trace.validate_manifest(document)
        self.assertFalse(document["performance_claim_eligible"])
        self.assertFalse(document["vllm_comparison_eligible"])
        self.assertFalse(document["serving_selector_changed"])

    def test_manifest_rejects_policy_promotion_or_q_hash_tampering(self) -> None:
        document = _manifest()
        document["performance_claim_eligible"] = True
        with self.assertRaisesRegex(trace.Qwen3BP2051QkvDefaultCublasTraceError, "identity"):
            trace.validate_manifest(document)

        document = _manifest()
        document["default_policy"]["preferred_blas_actual"] = "cublaslt"
        with self.assertRaisesRegex(trace.Qwen3BP2051QkvDefaultCublasTraceError, "policy"):
            trace.validate_manifest(document)

        document = _manifest()
        document["projection_results"]["q"]["output_bf16_le_sha256"] = _sha("0")
        with self.assertRaisesRegex(trace.Qwen3BP2051QkvDefaultCublasTraceError, "hash"):
            trace.validate_manifest(document)

    def test_sidecar_validator_replays_q_k_v_hashes_and_contiguous_ranges(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sidecar = Path(directory) / "p11.safetensors"
            raw_by_name = _raw_by_name()
            _write_safetensors_sidecar(sidecar, raw_by_name)
            document = _manifest(
                raw_by_name=raw_by_name,
                sidecar_name=sidecar.name,
                sidecar_sha256=hashlib.sha256(sidecar.read_bytes()).hexdigest(),
            )
            trace.validate_sidecar_against_manifest(document, sidecar)

            _write_safetensors_sidecar(sidecar, raw_by_name, tuple(reversed(trace.TRACE_TENSORS)))
            reordered = _manifest(
                raw_by_name=raw_by_name,
                sidecar_name=sidecar.name,
                sidecar_sha256=hashlib.sha256(sidecar.read_bytes()).hexdigest(),
            )
            trace.validate_sidecar_against_manifest(reordered, sidecar)

            payload = bytearray(sidecar.read_bytes())
            payload[-1] ^= 1
            sidecar.write_bytes(payload)
            tampered = copy.deepcopy(reordered)
            tampered["sidecar"]["sha256"] = hashlib.sha256(payload).hexdigest()
            with self.assertRaisesRegex(trace.Qwen3BP2051QkvDefaultCublasTraceError, "raw BF16 hash"):
                trace.validate_sidecar_against_manifest(tampered, sidecar)

    def test_checkpoint_k_weight_reader_binds_indexed_bf16_bytes(self) -> None:
        item = next(item for item in trace.PROJECTIONS if item.identifier == "k")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shard = root / "model-00001-of-00001.safetensors"
            raw = b"\x00\x00" * (item.width * trace.K)
            header = {
                item.checkpoint_weight_key: {
                    "dtype": "BF16",
                    "shape": [item.width, trace.K],
                    "data_offsets": [0, len(raw)],
                }
            }
            encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
            shard.write_bytes(len(encoded).to_bytes(8, "little") + encoded + raw)
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {item.checkpoint_weight_key: shard.name}}), encoding="utf-8"
            )
            checkpoint = oracle.CheckpointManifest(
                root=root,
                receipt=oracle.FileRecord("riley-checkpoint.json", 1, _sha("d")),
                files=(),
            )
            self.assertEqual(trace._checkpoint_weight_raw(checkpoint, item), raw)

    def test_output_paths_are_create_only_and_outside_repository(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = root / "repository"
            artifacts = root / "artifacts"
            repository.mkdir()
            artifacts.mkdir()
            existing = artifacts / "already.json"
            existing.write_text("already", encoding="utf-8")
            with self.assertRaisesRegex(trace.Qwen3BP2051QkvDefaultCublasTraceError, "overwrite"):
                trace._output_paths(existing, artifacts / "sidecar.safetensors", repository)
            with self.assertRaisesRegex(trace.Qwen3BP2051QkvDefaultCublasTraceError, "outside"):
                trace._output_paths(repository / "inside.json", repository / "inside.safetensors", repository)


if __name__ == "__main__":
    unittest.main()
