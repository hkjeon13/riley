from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import os
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from rustinfer_reference.calibration import CalibrationError
from rustinfer_reference.cli import main
from rustinfer_reference.constants import (
    MODEL_CONFIG_SHA256,
    PRIMARY_ENVIRONMENT_ID,
    PYTHON_EXECUTABLE_SHA256,
    PYTHON_PLATFORM_MACHINE,
    PYTHON_PLATFORM_SYSTEM,
    PYTHON_VERSION,
    SAFETENSORS_VERSION,
    TOKENIZER_FILES_SHA256,
    TOKENIZER_SHA256,
    TORCH_VERSION,
    TRANSFORMERS_VERSION,
)
from rustinfer_reference.environment import PRIMARY_ENVIRONMENT_SNAPSHOT
from rustinfer_reference.hf_calibration import OracleArtifactMetadata
from rustinfer_reference.pr07_trace import (
    CapturedPr07Trace,
    PR07_TRACE_ARTIFACT_KIND,
    PR07_TRACE_TENSOR_NAMES,
    PR07_TRACE_TOKEN_IDS,
    PR07_TRACE_TOKEN_IDS_SHA256,
    TRANSFORMERS_LLAMA_SOURCE_PATH,
    TRANSFORMERS_LLAMA_SOURCE_SHA256,
    _expected_trace_shapes,
    _validate_llama_topology,
    produce_pr07_trace,
    validate_pr07_trace_manifest,
)


FIXED_TIME = datetime(2026, 8, 25, 1, 2, 3, tzinfo=timezone.utc)


class FakeTensor:
    def __init__(self, shape: list[int], *, dtype: str = "torch.bfloat16") -> None:
        self.shape = tuple(shape)
        self.dtype = dtype


class FakeTraceBackend:
    transformers_llama_source_path = TRANSFORMERS_LLAMA_SOURCE_PATH
    transformers_llama_source_sha256 = TRANSFORMERS_LLAMA_SOURCE_SHA256

    def __init__(self, tensors: dict[str, object] | None = None) -> None:
        self.metadata = OracleArtifactMetadata(
            python_version=PYTHON_VERSION,
            python_executable_sha256=PYTHON_EXECUTABLE_SHA256,
            python_platform_system=PYTHON_PLATFORM_SYSTEM,
            python_platform_machine=PYTHON_PLATFORM_MACHINE,
            torch_version=TORCH_VERSION,
            transformers_version=TRANSFORMERS_VERSION,
            safetensors_version=SAFETENSORS_VERSION,
            config_sha256=MODEL_CONFIG_SHA256,
            tokenizer_sha256=TOKENIZER_SHA256,
            tokenizer_files_sha256=dict(TOKENIZER_FILES_SHA256),
        )
        self.tensors = tensors or {
            name: FakeTensor(shape)
            for name, shape in _expected_trace_shapes().items()
        }
        self.capture_calls = 0
        self.closed = False

    def capture_trace(self) -> CapturedPr07Trace:
        self.capture_calls += 1
        return CapturedPr07Trace(self.tensors)

    def close(self) -> None:
        self.closed = True


def fake_provenance(
    _repo_root: Path, *, observed_environment: dict[str, object]
) -> dict[str, object]:
    record = {
        "path": "tracked/source.py",
        "sha256": hashlib.sha256(b"source").hexdigest(),
    }
    return {
        "git_revision": "1" * 40,
        "git_dirty": False,
        "environment_id": PRIMARY_ENVIRONMENT_ID,
        "observed_environment": copy.deepcopy(observed_environment),
        "sources": {"base": dict(record)},
        "trace_sources": {"producer": dict(record)},
    }


def fake_sidecar_writer(tensors: dict[str, object], path: Path) -> None:
    rows = [
        f"{key}:{','.join(str(value) for value in tensor.shape)}:{tensor.dtype}"
        for key, tensor in sorted(tensors.items())
    ]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _typed(qualified_name: str, **attributes: object) -> object:
    module, name = qualified_name.rsplit(".", maxsplit=1)
    value_type = type(name, (), {})
    value_type.__module__ = module
    value = value_type()
    for key, attribute in attributes.items():
        setattr(value, key, attribute)
    return value


def fake_llama_topology() -> object:
    prefix = "transformers.models.llama.modeling_llama."
    config = SimpleNamespace(
        model_type="llama",
        hidden_size=576,
        intermediate_size=1536,
        num_hidden_layers=30,
        num_attention_heads=9,
        num_key_value_heads=3,
        head_dim=64,
        vocab_size=49_152,
        max_position_embeddings=8192,
        rms_norm_eps=1e-5,
        hidden_act="silu",
        attention_bias=False,
        mlp_bias=False,
        attention_dropout=0.0,
        tie_word_embeddings=True,
        rope_parameters={"rope_theta": 100_000, "rope_type": "default"},
        _attn_implementation="eager",
    )
    layers = []
    for index in range(30):
        attention = _typed(
            prefix + "LlamaAttention",
            layer_idx=index,
            q_proj=object(),
            k_proj=object(),
            v_proj=object(),
            o_proj=object(),
        )
        mlp = _typed(
            prefix + "LlamaMLP",
            gate_proj=object(),
            up_proj=object(),
            down_proj=object(),
        )
        layers.append(
            _typed(
                prefix + "LlamaDecoderLayer",
                self_attn=attention,
                mlp=mlp,
                input_layernorm=_typed(prefix + "LlamaRMSNorm"),
                post_attention_layernorm=_typed(prefix + "LlamaRMSNorm"),
            )
        )
    tied_weight = object()
    base_model = _typed(
        prefix + "LlamaModel",
        config=config,
        layers=layers,
        norm=_typed(prefix + "LlamaRMSNorm"),
        embed_tokens=SimpleNamespace(weight=tied_weight),
    )
    return _typed(
        prefix + "LlamaForCausalLM",
        config=config,
        model=base_model,
        lm_head=SimpleNamespace(weight=tied_weight),
    )


class Pr07TraceTests(unittest.TestCase):
    def test_trace_module_import_keeps_model_dependencies_lazy(self) -> None:
        package_root = Path(__file__).resolve().parents[1]
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(package_root)
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; import rustinfer_reference.pr07_trace; "
                "assert 'torch' not in sys.modules; "
                "assert 'transformers' not in sys.modules; "
                "assert 'safetensors' not in sys.modules",
            ],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_model_free_producer_writes_bound_exclusive_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            repository = workspace / "repo"
            repository.mkdir()
            artifact_dir = workspace / "artifacts"
            manifest_path = artifact_dir / "trace.json"
            sidecar_path = artifact_dir / "trace.safetensors"
            backend = FakeTraceBackend()
            calls: list[dict[str, object]] = []

            def backend_factory(**kwargs: object) -> FakeTraceBackend:
                calls.append(dict(kwargs))
                return backend

            manifest = produce_pr07_trace(
                manifest_path=manifest_path,
                sidecar_path=sidecar_path,
                repo_root=repository,
                device="cuda:0",
                local_files_only=True,
                created_at=FIXED_TIME,
                backend_factory=backend_factory,
                sidecar_writer=fake_sidecar_writer,
                environment_probe=lambda: copy.deepcopy(
                    PRIMARY_ENVIRONMENT_SNAPSHOT
                ),
                provenance_factory=fake_provenance,
            )

            self.assertEqual(
                calls, [{"device": "cuda:0", "local_files_only": True}]
            )
            self.assertEqual(backend.capture_calls, 1)
            self.assertTrue(backend.closed)
            self.assertTrue(manifest_path.is_file())
            self.assertTrue(sidecar_path.is_file())
            self.assertEqual(manifest["artifact_kind"], PR07_TRACE_ARTIFACT_KIND)
            self.assertEqual(
                manifest["contract"]["input_token_ids"], list(PR07_TRACE_TOKEN_IDS)
            )
            self.assertEqual(
                manifest["contract"]["input_token_ids_sha256"],
                PR07_TRACE_TOKEN_IDS_SHA256,
            )
            self.assertFalse(manifest["contract"]["execution"]["use_cache"])
            self.assertEqual(
                manifest["contract"]["execution"]["explicit_position_ids"],
                list(range(7)),
            )
            self.assertEqual(tuple(manifest["tensors"]), PR07_TRACE_TENSOR_NAMES)
            self.assertEqual(
                manifest["tensors"]["layer0.attention_context"]["shape"],
                [7, 9, 64],
            )
            self.assertEqual(
                manifest["sidecar"]["sha256"],
                hashlib.sha256(sidecar_path.read_bytes()).hexdigest(),
            )
            validate_pr07_trace_manifest(manifest)

    def test_existing_output_rejects_before_environment_or_backend(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            repository = workspace / "repo"
            repository.mkdir()
            manifest_path = workspace / "trace.json"
            manifest_path.write_text("owned\n", encoding="utf-8")
            calls: list[str] = []
            with self.assertRaisesRegex(CalibrationError, "overwrite"):
                produce_pr07_trace(
                    manifest_path=manifest_path,
                    sidecar_path=workspace / "trace.safetensors",
                    repo_root=repository,
                    device="cuda:0",
                    local_files_only=True,
                    created_at=FIXED_TIME,
                    backend_factory=lambda **_kwargs: calls.append("backend"),
                    environment_probe=lambda: calls.append("environment"),
                    provenance_factory=lambda *_args, **_kwargs: calls.append(
                        "provenance"
                    ),
                )
            self.assertEqual(calls, [])
            self.assertEqual(manifest_path.read_text(encoding="utf-8"), "owned\n")

    def test_environment_preflight_rejects_before_provenance_or_backend(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            repository = workspace / "repo"
            repository.mkdir()
            changed = copy.deepcopy(PRIMARY_ENVIRONMENT_SNAPSHOT)
            changed["accelerator"]["compute_process_count"] = 1
            calls: list[str] = []
            with self.assertRaisesRegex(CalibrationError, "environment preflight"):
                produce_pr07_trace(
                    manifest_path=workspace / "trace.json",
                    sidecar_path=workspace / "trace.safetensors",
                    repo_root=repository,
                    device="cuda:0",
                    local_files_only=True,
                    created_at=FIXED_TIME,
                    backend_factory=lambda **_kwargs: calls.append("backend"),
                    environment_probe=lambda: changed,
                    provenance_factory=lambda *_args, **_kwargs: calls.append(
                        "provenance"
                    ),
                )
            self.assertEqual(calls, [])

    def test_manifest_failure_removes_sidecar_and_temporary_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            repository = workspace / "repo"
            repository.mkdir()
            manifest_path = workspace / "trace.json"
            sidecar_path = workspace / "trace.safetensors"
            backend = FakeTraceBackend()
            with mock.patch(
                "rustinfer_reference.pr07_trace.write_json_exclusive",
                side_effect=OSError("injected manifest failure"),
            ):
                with self.assertRaisesRegex(OSError, "manifest failure"):
                    produce_pr07_trace(
                        manifest_path=manifest_path,
                        sidecar_path=sidecar_path,
                        repo_root=repository,
                        device="cuda:0",
                        local_files_only=True,
                        created_at=FIXED_TIME,
                        backend_factory=lambda **_kwargs: backend,
                        sidecar_writer=fake_sidecar_writer,
                        environment_probe=lambda: copy.deepcopy(
                            PRIMARY_ENVIRONMENT_SNAPSHOT
                        ),
                        provenance_factory=fake_provenance,
                    )
            self.assertTrue(backend.closed)
            self.assertFalse(manifest_path.exists())
            self.assertFalse(sidecar_path.exists())
            self.assertEqual(list(workspace.glob(".*.tmp")), [])

    def test_tensor_contract_rejects_wrong_dtype_and_order(self) -> None:
        shapes = _expected_trace_shapes()
        wrong_dtype = {
            name: FakeTensor(
                shape,
                dtype="torch.float32" if name == "last_logits" else "torch.bfloat16",
            )
            for name, shape in shapes.items()
        }
        wrong_order = dict(reversed(list(FakeTraceBackend().tensors.items())))
        for label, tensors, pattern in (
            ("dtype", wrong_dtype, "unsupported dtype"),
            ("order", wrong_order, "names/order"),
        ):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                workspace = Path(directory)
                repository = workspace / "repo"
                repository.mkdir()
                with self.assertRaisesRegex(CalibrationError, pattern):
                    produce_pr07_trace(
                        manifest_path=workspace / "trace.json",
                        sidecar_path=workspace / "trace.safetensors",
                        repo_root=repository,
                        device="cuda:0",
                        local_files_only=True,
                        created_at=FIXED_TIME,
                        backend_factory=lambda tensors=tensors, **_kwargs: FakeTraceBackend(
                            tensors
                        ),
                        sidecar_writer=fake_sidecar_writer,
                        environment_probe=lambda: copy.deepcopy(
                            PRIMARY_ENVIRONMENT_SNAPSHOT
                        ),
                        provenance_factory=fake_provenance,
                    )

    def test_transformers_topology_contract_fails_closed(self) -> None:
        model = fake_llama_topology()
        base_model, layers = _validate_llama_topology(model)
        self.assertIs(base_model, model.model)
        self.assertEqual(len(layers), 30)
        model.model.layers[14].self_attn.layer_idx = 99
        with self.assertRaisesRegex(CalibrationError, "wrong attention index"):
            _validate_llama_topology(model)

    def test_cli_routes_fixed_trace_without_importing_a_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            calls: list[dict[str, object]] = []

            def producer(**kwargs: object) -> dict[str, object]:
                calls.append(dict(kwargs))
                return {
                    "artifact_kind": PR07_TRACE_ARTIFACT_KIND,
                    "sidecar": {"tensor_count": len(PR07_TRACE_TENSOR_NAMES)},
                }

            with contextlib.redirect_stdout(io.StringIO()):
                result = main(
                    [
                        "pr07-trace-produce",
                        "--manifest",
                        str(workspace / "trace.json"),
                        "--sidecar",
                        str(workspace / "trace.safetensors"),
                        "--repo-root",
                        str(workspace / "repo"),
                        "--allow-download",
                    ],
                    pr07_trace_producer=producer,
                    environment_probe=lambda: copy.deepcopy(
                        PRIMARY_ENVIRONMENT_SNAPSHOT
                    ),
                    now=lambda: FIXED_TIME,
                )
            self.assertEqual(result, 0)
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]["device"], "cuda:0")
            self.assertFalse(calls[0]["local_files_only"])
            self.assertEqual(calls[0]["created_at"], FIXED_TIME)


if __name__ == "__main__":
    unittest.main()
