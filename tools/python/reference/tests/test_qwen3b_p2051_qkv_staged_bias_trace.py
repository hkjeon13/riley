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

from riley_reference import qwen3b_p2051_qkv_staged_bias_trace as trace
from riley_reference import qwen3b_serving_oracle as oracle

FIXED_TIME = "2026-09-17T02:03:04Z"


def _sha(character: str) -> str:
    return character * 64


def _raw_by_name() -> dict[str, bytes]:
    return {
        name: b"\x00\x00" * trace._element_count(shape)
        for name, shape in trace._expected_shapes().items()
    }


def _tensor_records(raw_by_name: dict[str, bytes]) -> dict[str, object]:
    return {
        name: {
            "key": trace._sidecar_key(name),
            "shape": list(shape),
            "dtype": "bfloat16",
            "canonical_byte_order": "little-endian-u16",
            "bf16_le_bytes": len(raw_by_name[name]),
            "bf16_le_sha256": hashlib.sha256(raw_by_name[name]).hexdigest(),
        }
        for name, shape in trace._expected_shapes().items()
    }


def _metric(total: int) -> dict[str, object]:
    return {
        "bf16_exact": True,
        "unequal_elements": 0,
        "total_elements": total,
        "max_abs": 0.0,
    }


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
    sidecar_name: str = "p12.safetensors",
    sidecar_sha256: str | None = None,
) -> dict[str, object]:
    raw = raw_by_name or _raw_by_name()
    tensors = _tensor_records(raw)
    results: dict[str, object] = {}
    actual_metrics: dict[str, object] = {}
    for item in trace.PROJECTIONS:
        metric = _metric(trace.M * item.width)
        actual_metrics[item.identifier] = metric
        results[item.identifier] = {
            "identifier": item.identifier,
            "module_name": item.module_name,
            "raw_tensor": item.p11_raw_name,
            "bias_tensor": item.bias_name,
            "staged_tensor": item.staged_name,
            "actual_tensor": item.p7_actual_name,
            "checkpoint_bias_key": item.checkpoint_bias_key,
            "shape": {"m": trace.M, "n": item.width, "k": trace.K},
            "raw_bf16_le_sha256": tensors[item.p11_raw_name]["bf16_le_sha256"],
            "bias_bf16_le_sha256": tensors[item.bias_name]["bf16_le_sha256"],
            "staged_bf16_le_sha256": tensors[item.staged_name]["bf16_le_sha256"],
            "actual_bf16_le_sha256": tensors[item.p7_actual_name]["bf16_le_sha256"],
            "raw_vs_p11_default_cublas": _metric(trace.M * item.width),
            "staged_repeated_bf16_exact": True,
            "staged_vs_p7_actual_module": metric,
        }
    checkpoint_sha = _sha("d")
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
                "layer0.q_proj.raw_default_cublas_then_staged_bias",
                "layer0.k_proj.raw_default_cublas_then_staged_bias",
                "layer0.v_proj.raw_default_cublas_then_staged_bias",
            ],
            "primary_profile": "explicit_fp32_bias_add_bf16",
            "actual_module_output": "observational-not-primary-profile",
            "model_id": oracle.MODEL_ID,
            "model_revision": oracle.MODEL_REVISION,
            "shape": {
                "m": trace.M,
                "k": trace.K,
                "q_n": trace.Q_WIDTH,
                "kv_n": trace.KV_WIDTH,
            },
            "offline_only": True,
            "serving_path": False,
        },
        "producer": {
            "implementation_id": trace.IMPLEMENTATION_ID,
            "runtime_dependency_class": "offline-python-reference",
            "torch_version": "2.13.0",
            "runtime_cuda_version": "12.8",
            "device": "cuda:0",
            "staged_bias_operator": (
                "bf16(raw_default_cublas.to(float32)+"
                "checkpoint_bf16_bias.to(float32))"
            ),
        },
        "p7_binding": {
            "manifest_filename": "p7.json",
            "manifest_sha256": trace.P7_MANIFEST_SHA256,
            "sidecar_filename": "p7.safetensors",
            "sidecar_sha256": trace.P7_SIDECAR_SHA256,
            "checkpoint_receipt_sha256": checkpoint_sha,
            "source_revision": "e" * 40,
            "input_tensor_key": trace.p11.P7_INPUT_KEY,
            "actual_tensor_keys": {
                item.identifier: f"trace/layer0/{item.module_name}"
                for item in trace.PROJECTIONS
            },
            "actual_bf16_le_sha256": {
                item.identifier: tensors[item.p7_actual_name]["bf16_le_sha256"]
                for item in trace.PROJECTIONS
            },
        },
        "p11_binding": {
            "manifest_filename": "p11.json",
            "manifest_sha256": trace.P11_MANIFEST_SHA256,
            "sidecar_filename": "p11.safetensors",
            "sidecar_sha256": trace.P11_SIDECAR_SHA256,
            "checkpoint_receipt_sha256": checkpoint_sha,
            "source_revision": "f" * 40,
            "input_tensor_name": trace.p11.P7_INPUT_NAME,
            "raw_tensor_names": {
                item.identifier: item.p11_raw_name for item in trace.PROJECTIONS
            },
            "input_bf16_le_sha256": tensors[trace.P11_INPUT_NAME]["bf16_le_sha256"],
            "raw_bf16_le_sha256": {
                item.identifier: tensors[item.p11_raw_name]["bf16_le_sha256"]
                for item in trace.PROJECTIONS
            },
        },
        "model": {
            "checkpoint_path": "/checkpoint",
            "checkpoint_receipt_filename": oracle.CHECKPOINT_RECEIPT_FILENAME,
            "checkpoint_receipt_sha256": checkpoint_sha,
        },
        "projection_results": results,
        "comparisons": {
            "primary_profile": "explicit_fp32_bias_add_bf16",
            "raw_qkv_vs_p11_default_cublas_exact": True,
            "staged_qkv_repeated_bf16_exact": True,
            "actual_module_output_is_observational": True,
            "staged_vs_p7_actual_module": actual_metrics,
        },
        "provenance": {"source_repository": _source_provenance()},
        "sidecar": {
            "path": sidecar_name,
            "sha256": sidecar_sha256 or _sha("9"),
            "format": "safetensors",
            "tensor_count": len(trace.TRACE_TENSORS),
        },
        "tensors": tensors,
    }


def _write_sidecar(path: Path, raw_by_name: dict[str, bytes]) -> None:
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


class Qwen3BP2051QkvStagedBiasTraceTests(unittest.TestCase):
    def test_import_keeps_ml_dependencies_lazy(self) -> None:
        package_root = Path(__file__).resolve().parents[1]
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(package_root)
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; import riley_reference.qwen3b_p2051_qkv_staged_bias_trace; "
                "assert 'torch' not in sys.modules; "
                "assert 'transformers' not in sys.modules; "
                "assert 'safetensors' not in sys.modules",
            ],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_schema_declares_separate_staged_and_actual_endpoints(self) -> None:
        self.assertEqual(len(trace.TRACE_TENSORS), 13)
        self.assertEqual(
            trace._expected_shapes()["staged_q/explicit_fp32_bias_add_bf16"],
            (2_051, 2_048),
        )
        self.assertEqual(
            trace._expected_shapes()["staged_k/explicit_fp32_bias_add_bf16"],
            (2_051, 256),
        )
        self.assertEqual(
            trace._expected_shapes()["p7_actual_v"], (2_051, 256)
        )

    def test_manifest_accepts_staged_primary_profile(self) -> None:
        document = _manifest()
        trace.validate_manifest(document)
        self.assertTrue(document["quality_pass"])
        self.assertFalse(document["performance_claim_eligible"])
        self.assertFalse(document["vllm_comparison_eligible"])

    def test_manifest_rejects_profile_promotion_and_lineage_tampering(self) -> None:
        document = _manifest()
        document["performance_claim_eligible"] = True
        with self.assertRaisesRegex(
            trace.Qwen3BP2051QkvStagedBiasTraceError, "identity"
        ):
            trace.validate_manifest(document)

        document = _manifest()
        document["scope"]["primary_profile"] = "hf-module-output"
        with self.assertRaisesRegex(
            trace.Qwen3BP2051QkvStagedBiasTraceError, "scope"
        ):
            trace.validate_manifest(document)

        document = _manifest()
        document["projection_results"]["q"]["raw_bf16_le_sha256"] = _sha("0")
        with self.assertRaisesRegex(
            trace.Qwen3BP2051QkvStagedBiasTraceError, "lineage"
        ):
            trace.validate_manifest(document)

    def test_sidecar_validator_replays_staged_and_actual_bf16_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sidecar = Path(directory) / "p12.safetensors"
            raw = _raw_by_name()
            _write_sidecar(sidecar, raw)
            document = _manifest(
                raw_by_name=raw,
                sidecar_name=sidecar.name,
                sidecar_sha256=hashlib.sha256(sidecar.read_bytes()).hexdigest(),
            )
            trace.validate_sidecar_against_manifest(document, sidecar)

            payload = bytearray(sidecar.read_bytes())
            payload[-1] ^= 1
            sidecar.write_bytes(payload)
            changed = copy.deepcopy(document)
            changed["sidecar"]["sha256"] = hashlib.sha256(payload).hexdigest()
            with self.assertRaisesRegex(
                trace.Qwen3BP2051QkvStagedBiasTraceError, "raw BF16 hash"
            ):
                trace.validate_sidecar_against_manifest(changed, sidecar)

    def test_output_paths_are_create_only_and_outside_repository(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = root / "repository"
            artifacts = root / "artifacts"
            repository.mkdir()
            artifacts.mkdir()
            existing = artifacts / "already.json"
            existing.write_text("already", encoding="utf-8")
            with self.assertRaisesRegex(
                trace.Qwen3BP2051QkvStagedBiasTraceError, "overwrite"
            ):
                trace._output_paths(existing, artifacts / "sidecar.safetensors", repository)
            with self.assertRaisesRegex(
                trace.Qwen3BP2051QkvStagedBiasTraceError, "outside"
            ):
                trace._output_paths(
                    repository / "inside.json",
                    repository / "inside.safetensors",
                    repository,
                )


if __name__ == "__main__":
    unittest.main()
