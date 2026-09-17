from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

from riley_reference import qwen3b_generation_trace as generation
from riley_reference import qwen3b_p2051_attention_window_trace as trace
from riley_reference import qwen3b_serving_oracle as oracle

FIXED_TIME = "2026-09-17T05:20:00Z"
LAYER_INDEX = 2


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


def _teacher_prefix() -> trace.TeacherPrefix:
    return trace.TeacherPrefix(
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
            "path": trace.base.stage.TRANSFORMERS_QWEN2_SOURCE_PATH,
            "sha256": trace.base.stage.TRANSFORMERS_QWEN2_SOURCE_SHA256,
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
    layer_index: int, raw_by_name: dict[str, bytes] | None = None
) -> dict[str, object]:
    tensors: dict[str, object] = {}
    shapes = trace._expected_shapes(layer_index)
    for name in trace.trace_tensor_names(layer_index):
        raw = (
            raw_by_name[name]
            if raw_by_name is not None
            else b"\x00\x00" * trace._shape_element_count(shapes[name])
        )
        tensors[name] = {
            "key": trace._sidecar_key(name),
            "shape": list(shapes[name]),
            "dtype": "bfloat16",
            "canonical_byte_order": "little-endian-u16",
            "bf16_le_sha256": hashlib.sha256(raw).hexdigest(),
            "bf16_le_bytes": len(raw),
        }
    return tensors


def _manifest(
    layer_index: int = LAYER_INDEX,
    *,
    tensors: dict[str, object] | None = None,
    sidecar_name: str = "window.safetensors",
    sidecar_sha256: str | None = None,
) -> dict[str, object]:
    workload = _workload()
    prefix = _teacher_prefix()
    return {
        "schema_version": trace.SCHEMA_VERSION,
        "artifact_kind": trace.ARTIFACT_KIND,
        "trace_id": trace.TRACE_ID,
        "performance_claim_eligible": False,
        "created_at": FIXED_TIME,
        "producer": _producer(),
        "trace_profile": trace._capture_profile_document(layer_index),
        "contract": {
            "model_id": oracle.MODEL_ID,
            "model_revision": oracle.MODEL_REVISION,
            "workload": trace._workload_document(workload),
            "execution": trace._execution_document(),
            "input": trace._teacher_input_document(workload, prefix),
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
            "tensor_count": len(trace.trace_tensor_names(layer_index)),
        },
        "tensors": tensors or _tensor_document(layer_index),
    }


def _write_sidecar(path: Path, layer_index: int, raw_by_name: dict[str, bytes]) -> None:
    offset = 0
    header: dict[str, object] = {}
    payload = bytearray()
    for name in trace.trace_tensor_names(layer_index):
        raw = raw_by_name[name]
        header[trace._sidecar_key(name)] = {
            "dtype": "BF16",
            "shape": list(trace._expected_shapes(layer_index)[name]),
            "data_offsets": [offset, offset + len(raw)],
        }
        offset += len(raw)
        payload.extend(raw)
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + payload)


class Qwen3BP2051AttentionWindowTraceTests(unittest.TestCase):
    def test_import_keeps_ml_dependencies_lazy(self) -> None:
        package_root = Path(__file__).resolve().parents[1]
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(package_root)
        command = (
            "import sys; import riley_reference.qwen3b_p2051_attention_window_trace; "
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

    def test_selected_layer_contract_is_generic_and_complete(self) -> None:
        self.assertEqual(len(trace.ATTENTION_SUFFIXES), 8)
        names = trace.trace_tensor_names(LAYER_INDEX)
        self.assertEqual(names[0], "layer2.attention.query_bshd")
        self.assertEqual(names[-1], "layer2.attention.mask_last")
        self.assertEqual(
            trace._expected_shapes(LAYER_INDEX)[names[0]], (1, 2051, 16, 128)
        )
        self.assertEqual(
            trace._expected_shapes(LAYER_INDEX)[names[1]], (1, 2051, 2, 128)
        )
        self.assertEqual(trace._expected_shapes(LAYER_INDEX)[names[3]], (16, 2051))
        self.assertEqual(
            trace._capture_profile_document(LAYER_INDEX)["rust_consumer"]["api"],
            "riley_cuda::PreparedPrefillAttention::execute_hf_eager_qwen_p2051_last_row_traced",
        )
        self.assertEqual(
            trace._capture_profile_document(LAYER_INDEX)["rust_consumer"]["layer_index"],
            LAYER_INDEX,
        )
        with self.assertRaisesRegex(trace.Qwen3BP2051AttentionWindowTraceError, "0..35"):
            trace.trace_tensor_names(36)

    def test_manifest_accepts_any_valid_layer_and_rejects_profile_tampering(self) -> None:
        document = _manifest()
        self.assertEqual(trace.validate_manifest(document), LAYER_INDEX)
        document = _manifest(35)
        self.assertEqual(trace.validate_manifest(document), 35)
        tampered = _manifest()
        tampered["trace_profile"]["tensor_names"][0] = "layer2.attention.context_bshd"
        with self.assertRaisesRegex(trace.Qwen3BP2051AttentionWindowTraceError, "profile"):
            trace.validate_manifest(tampered)

    def test_sidecar_validator_replays_every_hash_and_range(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sidecar = Path(directory) / "window.safetensors"
            raw_by_name = {
                name: bytes([ordinal + 1, 0])
                * trace._shape_element_count(trace._expected_shapes(LAYER_INDEX)[name])
                for ordinal, name in enumerate(trace.trace_tensor_names(LAYER_INDEX))
            }
            _write_sidecar(sidecar, LAYER_INDEX, raw_by_name)
            document = _manifest(
                tensors=_tensor_document(LAYER_INDEX, raw_by_name),
                sidecar_name=sidecar.name,
                sidecar_sha256=hashlib.sha256(sidecar.read_bytes()).hexdigest(),
            )
            trace.validate_sidecar_against_manifest(document, sidecar)
            tampered = copy.deepcopy(document)
            name = trace.trace_tensor_names(LAYER_INDEX)[6]
            tampered["tensors"][name]["bf16_le_sha256"] = _sha("0")
            with self.assertRaisesRegex(trace.Qwen3BP2051AttentionWindowTraceError, "raw BF16 hash"):
                trace.validate_sidecar_against_manifest(tampered, sidecar)

    def test_external_provenance_rechecks_source_hashes_without_git(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "repository"
            root.mkdir()
            source = root / "fixture.py"
            source.write_text("fixture = 1\n", encoding="utf-8")
            source_paths = {"fixture": "fixture.py"}
            document = {
                "git_revision": "a" * 40,
                "source_dirty": False,
                "source_status_sha256": hashlib.sha256(b"").hexdigest(),
                "sources": {
                    "fixture": {
                        "path": "fixture.py",
                        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    }
                },
            }
            provenance = Path(directory) / "source-provenance.json"
            provenance.write_text(json.dumps(document), encoding="utf-8")
            environment = {trace.SOURCE_PROVENANCE_ENV: str(provenance)}
            with (
                mock.patch.dict(os.environ, environment, clear=False),
                mock.patch.object(trace, "SOURCE_PATHS", source_paths),
                mock.patch.object(
                    trace.subprocess,
                    "run",
                    side_effect=AssertionError("external provenance must not run Git"),
                ),
            ):
                self.assertEqual(trace.collect_source_provenance(root), document)
                source.write_text("fixture = 2\n", encoding="utf-8")
                with self.assertRaisesRegex(
                    trace.Qwen3BP2051AttentionWindowTraceError, "hashes differ"
                ):
                    trace.collect_source_provenance(root)


if __name__ == "__main__":
    unittest.main()
