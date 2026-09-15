from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

from riley_reference import qwen3b_serving_oracle as oracle

FIXED_TIME = datetime(2026, 9, 16, 1, 2, 3, tzinfo=timezone.utc)


def _output_ids() -> tuple[int, ...]:
    return (*oracle.EXPECTED_OUTPUT_PREFIX, *(17 for _ in range(120)))


def _patched_output_contract() -> ExitStack:
    output_ids = _output_ids()
    patches = ExitStack()
    patches.enter_context(
        mock.patch.object(
            oracle,
            "OUTPUT_TOKEN_IDS_SHA256",
            oracle._token_ids_sha256(output_ids),
        )
    )
    return patches


def _workload_document() -> dict[str, object]:
    return {
        "schema_version": oracle.WORKLOAD_SCHEMA_VERSION,
        "case": oracle.WORKLOAD_CASE,
        "model_id": oracle.MODEL_ID,
        "model_revision": oracle.MODEL_REVISION,
        "prompt": "pinned synthetic test prompt",
        "prompt_token_ids": [oracle.EXPECTED_PROMPT_TOKEN_ID]
        * oracle.PROMPT_TOKEN_COUNT,
        "output_token_ids": list(_output_ids()),
        "output_text": "test output",
        "finish_reason": "length",
        "sampling": {"temperature": 0.0, "top_p": 1.0},
    }


def _workload(path: Path) -> oracle.ServingWorkload:
    source = _workload_document()
    raw = json.dumps(source, sort_keys=True, separators=(",", ":")).encode("utf-8")
    path.write_bytes(raw)
    return oracle.ServingWorkload(
        source_path=path.resolve(),
        source_bytes=len(raw),
        source_sha256=hashlib.sha256(raw).hexdigest(),
        case=oracle.WORKLOAD_CASE,
        prompt=str(source["prompt"]),
        prompt_token_ids=tuple(source["prompt_token_ids"]),
        output_token_ids=tuple(source["output_token_ids"]),
        output_text=str(source["output_text"]),
    )


def _checkpoint() -> oracle.CheckpointManifest:
    metadata = tuple(
        oracle.FileRecord(
            path=path,
            size_bytes=int(record["size_bytes"]),
            sha256=str(record["sha256"]),
        )
        for path, record in oracle.EXPECTED_CHECKPOINT_METADATA.items()
    )
    weights = tuple(
        oracle.FileRecord(
            path=f"model-{index:05d}-of-00002.safetensors",
            size_bytes=size_bytes,
            sha256=digest,
        )
        for index, (size_bytes, digest) in enumerate(
            sorted(oracle.EXPECTED_WEIGHT_FILES), start=1
        )
    )
    return oracle.CheckpointManifest(
        root=Path("/checkpoint"),
        receipt=oracle.FileRecord(
            path=oracle.CHECKPOINT_RECEIPT_FILENAME,
            size_bytes=oracle.CHECKPOINT_RECEIPT_BYTES,
            sha256=oracle.CHECKPOINT_RECEIPT_SHA256,
        ),
        files=tuple(sorted((*metadata, *weights), key=lambda item: item.path)),
    )


def _capture() -> oracle.RawLogitCapture:
    token_ids = tuple(range(oracle.TOP_K))
    values = tuple(float(oracle.TOP_K - index) for index in range(oracle.TOP_K))
    return oracle.RawLogitCapture(
        raw_bf16_le_sha256="d" * 64,
        argmax_token_id=token_ids[0],
        argmax_value_bf16_as_f32=values[0],
        top_token_ids=token_ids,
        top_values_bf16_as_f32=values,
        probe_values_bf16_as_f32={
            token_id: float(token_id) for token_id in oracle.PROBE_IDS
        },
    )


def _producer_metadata() -> dict[str, object]:
    return {
        "implementation_id": oracle.IMPLEMENTATION_ID,
        "runtime_dependency_class": "offline-python-reference",
        "python_version": "3.13.15",
        "python_executable_sha256": "e" * 64,
        "python_platform_system": "linux",
        "python_platform_machine": "x86_64",
        "torch_version": "2.13.0",
        "transformers_version": "5.15.1",
        "safetensors_version": "0.8.0",
        "selected_device": {
            "requested": "cuda:0",
            "resolved": "cuda:0",
            "index": 0,
            "name": "test GPU",
            "compute_capability": "8.9",
            "driver_version": "580.178.04",
            "runtime_cuda_version": "13.0",
        },
        "transformers_model_source": {
            "module": "transformers.models.qwen2.modeling_qwen2",
            "filename": "modeling_qwen2.py",
            "sha256": "f" * 64,
        },
    }


class _ConfigObject:
    def to_dict(self) -> dict[str, object]:
        return {
            "architectures": ["Qwen2ForCausalLM"],
            "model_type": "qwen2",
            "hidden_size": 2_048,
            "intermediate_size": 11_008,
            "num_hidden_layers": 36,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "max_position_embeddings": 32_768,
            "vocab_size": oracle.MODEL_VOCABULARY_SIZE,
            "tie_word_embeddings": True,
            "attention_dropout": 0.0,
            "rms_norm_eps": 1e-6,
            "use_sliding_window": False,
            "bos_token_id": 151_643,
            "eos_token_id": 151_645,
            "rope_parameters": {
                "rope_theta": 1_000_000.0,
                "rope_type": "default",
            },
        }


def _source_provenance() -> dict[str, object]:
    return {
        "git_revision": "a" * 40,
        "source_dirty": False,
        "source_status_sha256": "b" * 64,
        "sources": {
            name: {"path": path, "sha256": "c" * 64}
            for name, path in oracle.SOURCE_PATHS.items()
        },
    }


class Qwen3BServingOracleTests(unittest.TestCase):
    def test_module_import_keeps_torch_transformers_and_safetensors_lazy(self) -> None:
        package_root = Path(__file__).resolve().parents[1]
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(package_root)
        import_script = (
            "import sys; import riley_reference.qwen3b_serving_oracle; "
            "assert 'torch' not in sys.modules; "
            "assert 'transformers' not in sys.modules; "
            "assert 'safetensors' not in sys.modules"
        )
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                import_script,
            ],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_workload_loader_binds_schema_and_exact_u32_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory, _patched_output_contract():
            path = Path(directory) / "workload.json"
            document = _workload_document()
            raw = json.dumps(document, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
            path.write_bytes(raw)
            with mock.patch.object(
                oracle, "WORKLOAD_SHA256", hashlib.sha256(raw).hexdigest()
            ):
                workload = oracle.load_workload(path)
            self.assertEqual(workload.prompt_token_ids, (3_409,) * 2_048)
            self.assertEqual(
                workload.output_token_ids[:8], oracle.EXPECTED_OUTPUT_PREFIX
            )
            self.assertEqual(
                oracle._token_ids_sha256(workload.prompt_token_ids),
                oracle.PROMPT_TOKEN_IDS_SHA256,
            )

    def test_loaded_transformers_config_uses_to_dict_without_torch(self) -> None:
        config = _ConfigObject()
        mapping = oracle._loaded_model_config_mapping(config)
        self.assertEqual(mapping["model_type"], "qwen2")
        oracle._validate_loaded_model_config(config)
        checkpoint_config = dict(mapping)
        checkpoint_config.pop("rope_parameters")
        checkpoint_config.update(
            {
                "rope_theta": 1_000_000.0,
                "sliding_window": 32_768,
                "torch_dtype": "bfloat16",
            }
        )
        oracle._validate_checkpoint_config(checkpoint_config)
        with self.assertRaisesRegex(oracle.Qwen3BServingOracleError, "to_dict"):
            oracle._loaded_model_config_mapping({})

    def test_schema_writer_is_create_only_and_rejects_cache_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory, _patched_output_contract():
            workspace = Path(directory)
            workload = _workload(workspace / "workload.json")
            with mock.patch.object(oracle, "WORKLOAD_SHA256", workload.source_sha256):
                artifact = oracle.build_oracle_artifact(
                    workload=workload,
                    checkpoint=_checkpoint(),
                    capture=_capture(),
                    producer_metadata=_producer_metadata(),
                    source_provenance=_source_provenance(),
                    created_at=FIXED_TIME,
                )
                oracle.validate_oracle_artifact(artifact)
                output = workspace / "artifact.json"
                oracle.write_artifact_exclusive(output, artifact)
                self.assertEqual(
                    output.read_bytes(), oracle._canonical_json_bytes(artifact)
                )
                with self.assertRaisesRegex(
                    oracle.Qwen3BServingOracleError, "overwrite"
                ):
                    oracle.write_artifact_exclusive(output, artifact)
                tampered = copy.deepcopy(artifact)
                tampered["contract"]["execution"]["use_cache"] = True
                with self.assertRaisesRegex(
                    oracle.Qwen3BServingOracleError, "execution"
                ):
                    oracle.validate_oracle_artifact(tampered)

    def test_producer_rejects_repository_output_before_input_or_backend(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "repo"
            root.mkdir()
            called: list[object] = []
            with self.assertRaisesRegex(oracle.Qwen3BServingOracleError, "outside"):
                oracle.produce_oracle(
                    checkpoint_path=Path("/does-not-matter"),
                    workload_path=Path("/does-not-matter"),
                    output_path=root / "artifact.json",
                    repo_root=root,
                    device="cuda:0",
                    backend_factory=lambda **_kwargs: called.append("backend"),
                )
            self.assertEqual(called, [])


if __name__ == "__main__":
    unittest.main()
