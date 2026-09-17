from __future__ import annotations

import copy
import hashlib
import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

from riley_reference import qwen3b_generation_trace as generation
from riley_reference import qwen3b_p2051_layer_stage_trace as trace
from riley_reference import qwen3b_serving_oracle as oracle

FIXED_TIME = "2026-09-16T02:03:04Z"


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


def _teacher_prefix(ids: tuple[int, int, int] = (7, 8, 9)) -> trace.TeacherPrefix:
    return trace.TeacherPrefix(
        token_ids=ids,
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
    *, raw_by_name: dict[str, bytes] | None = None
) -> dict[str, object]:
    tensors: dict[str, object] = {}
    for index, name in enumerate(trace.TRACE_TENSORS):
        shape = trace._expected_shapes()[name]
        raw = (
            raw_by_name[name]
            if raw_by_name is not None
            else bytes([index % 251])
            * (trace._shape_element_count(shape) * trace.BF16_BYTES)
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
    prefix: trace.TeacherPrefix | None = None,
    tensors: dict[str, object] | None = None,
    sidecar_name: str = "p2051.safetensors",
    sidecar_sha256: str | None = None,
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
            "workload": trace._workload_document(workload),
            "execution": trace._execution_document(),
            "input": trace._teacher_input_document(
                workload, prefix or _teacher_prefix()
            ),
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


class Qwen3BP2051LayerStageTraceTests(unittest.TestCase):
    def test_import_keeps_torch_transformers_and_safetensors_lazy(self) -> None:
        package_root = Path(__file__).resolve().parents[1]
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(package_root)
        command = (
            "import sys; import riley_reference.qwen3b_p2051_layer_stage_trace; "
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

    def test_schema_covers_all_last_token_layer_outputs_and_l0_boundaries(self) -> None:
        self.assertEqual(len(trace.TRACE_TENSORS), 52)
        self.assertEqual(trace.LAST_TOKEN_ROW_INDEX, 2_050)
        self.assertEqual(
            tuple(
                name
                for name in trace.TRACE_TENSORS
                if name.startswith("layer") and name.endswith(".output.last")
            ),
            tuple(f"layer{index}.output.last" for index in range(36)),
        )
        expected = trace._expected_shapes()
        self.assertEqual(expected["embedding.last"], (2_048,))
        self.assertEqual(expected["layer0.q_rope.last"], (16, 128))
        self.assertEqual(expected["layer0.k_rope.last"], (2, 128))
        self.assertEqual(expected["layer0.gated.last"], (11_008,))
        self.assertEqual(expected["last_logits"], (oracle.MODEL_VOCABULARY_SIZE,))
        self.assertEqual(
            trace._capture_profile_document()["rust_consumer"]["api"],
            "riley_runtime::llama::PreparedLlamaForward::prepare_last_token_layer_trace+execute_last_token_layer_traced",
        )

    def test_manifest_accepts_fixed_p2051_contract_without_ml_dependencies(
        self,
    ) -> None:
        document = _manifest()
        trace.validate_manifest(document)
        self.assertFalse(document["performance_claim_eligible"])
        self.assertEqual(document["contract"]["input"]["context_token_count"], 2_051)

    def test_manifest_rejects_teacher_prefix_or_cache_binding_tampering(self) -> None:
        document = _manifest()
        document["contract"]["input"]["teacher_prefix_token_ids"][1] = 17
        with self.assertRaisesRegex(trace.Qwen3BP2051LayerStageTraceError, "prefix"):
            trace.validate_manifest(document)

        document = _manifest()
        document["contract"]["input"]["teacher_source"]["cache_mode"] = "cache-on"
        with self.assertRaisesRegex(
            trace.Qwen3BP2051LayerStageTraceError, "teacher source"
        ):
            trace.validate_manifest(document)

    def test_output_sidecar_validator_replays_each_hash_and_contiguous_ranges(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sidecar = root / "p2051.safetensors"
            raw_by_name = {
                name: bytes([index % 251])
                * (
                    trace._shape_element_count(trace._expected_shapes()[name])
                    * trace.BF16_BYTES
                )
                for index, name in enumerate(trace.TRACE_TENSORS)
            }
            _write_safetensors_sidecar(sidecar, raw_by_name)
            document = _manifest(
                tensors=_tensor_document(raw_by_name=raw_by_name),
                sidecar_name=sidecar.name,
                sidecar_sha256=hashlib.sha256(sidecar.read_bytes()).hexdigest(),
            )
            trace.validate_sidecar_against_manifest(document, sidecar)

            tampered = copy.deepcopy(document)
            tampered["tensors"]["layer14.output.last"]["bf16_le_sha256"] = _sha("0")
            with self.assertRaisesRegex(
                trace.Qwen3BP2051LayerStageTraceError, "raw BF16 hash"
            ):
                trace.validate_sidecar_against_manifest(tampered, sidecar)

            nonfinite = dict(raw_by_name)
            nonfinite["embedding.last"] = b"\x80\x7f" * (
                len(nonfinite["embedding.last"]) // trace.BF16_BYTES
            )
            _write_safetensors_sidecar(sidecar, nonfinite)
            nonfinite_document = _manifest(
                tensors=_tensor_document(raw_by_name=nonfinite),
                sidecar_name=sidecar.name,
                sidecar_sha256=hashlib.sha256(sidecar.read_bytes()).hexdigest(),
            )
            with self.assertRaisesRegex(
                trace.Qwen3BP2051LayerStageTraceError, "non-finite"
            ):
                trace.validate_sidecar_against_manifest(nonfinite_document, sidecar)

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
                    trace.Qwen3BP2051LayerStageTraceError, "hashes differ"
                ):
                    trace.collect_source_provenance(root)

    def test_sidecar_is_readable_by_a_host_rust_consumer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sidecar = Path(directory) / "trace.safetensors"
            sidecar.write_bytes(b"fixture")
            sidecar.chmod(stat.S_IRUSR | stat.S_IWUSR)
            trace._ensure_sidecar_consumer_readable(sidecar)
            mode = sidecar.stat().st_mode
            self.assertTrue(mode & stat.S_IRUSR)
            self.assertTrue(mode & stat.S_IRGRP)
            self.assertTrue(mode & stat.S_IROTH)
            self.assertTrue(mode & stat.S_IWUSR)

    def test_teacher_prefix_is_loaded_from_verified_cache_off_artifact_not_literals(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "teacher.json"
            sidecar_path = root / "teacher-cache-off.safetensors"
            manifest_path.write_text("{}", encoding="utf-8")
            sidecar_path.write_bytes(b"sidecar")
            teacher_ids = [901, 902, 903] + [7] * (oracle.OUTPUT_TOKEN_COUNT - 3)
            document = {
                "schema_version": generation.SCHEMA_VERSION,
                "artifact_kind": generation.ARTIFACT_KIND,
                "generation": {
                    "teacher_token_ids": teacher_ids,
                    "teacher_token_ids_le_u32_sha256": oracle._token_ids_sha256(
                        teacher_ids
                    ),
                    "cache_off": {
                        "sidecar": {
                            "path": sidecar_path.name,
                            "tensor_key": generation.SIDECAR_TENSOR_KEY,
                        }
                    },
                },
            }
            with (
                mock.patch.object(
                    generation, "_load_manifest", return_value=document
                ) as load,
                mock.patch.object(
                    generation, "validate_sidecar_against_artifact"
                ) as verify,
            ):
                prefix = trace.load_teacher_prefix(
                    teacher_manifest_path=manifest_path,
                    teacher_cache_off_sidecar_path=sidecar_path,
                )
            self.assertEqual(prefix.token_ids, (901, 902, 903))
            load.assert_called_once()
            verify.assert_called_once_with(
                document=document,
                mode=generation.CACHE_OFF,
                sidecar_path=sidecar_path.resolve(),
            )
            input_ids = trace.build_input_token_ids(_workload(), prefix)
            self.assertEqual(input_ids[-3:], (901, 902, 903))
            self.assertEqual(len(input_ids), 2_051)

    def test_output_paths_are_create_only_and_outside_repository(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = root / "repository"
            repository.mkdir()
            artifacts = root / "artifacts"
            artifacts.mkdir()
            existing = artifacts / "existing.json"
            existing.write_text("already", encoding="utf-8")
            with self.assertRaisesRegex(
                trace.Qwen3BP2051LayerStageTraceError, "overwrite"
            ):
                trace._output_paths(
                    existing, artifacts / "existing.safetensors", repository
                )
            with self.assertRaisesRegex(
                trace.Qwen3BP2051LayerStageTraceError, "outside"
            ):
                trace._output_paths(
                    repository / "inside.json",
                    repository / "inside.safetensors",
                    repository,
                )


if __name__ == "__main__":
    unittest.main()
