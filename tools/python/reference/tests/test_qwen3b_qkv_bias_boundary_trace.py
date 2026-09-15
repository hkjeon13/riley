from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from riley_reference import qwen3b_qkv_bias_boundary_trace as trace
from riley_reference import qwen3b_serving_oracle as oracle


def _sha(character: str) -> str:
    return character * 64


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
                "path": trace.stage.TRANSFORMERS_QWEN2_SOURCE_PATH,
                "sha256": trace.stage.TRANSFORMERS_QWEN2_SOURCE_SHA256,
            },
        },
        "trace_profile": trace._trace_profile_document(),
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
            "path": "qwen3b-qkv-boundary.safetensors",
            "sha256": _sha("e"),
            "format": "safetensors",
            "tensor_count": len(trace.TRACE_TENSORS),
        },
        "tensors": tensors,
    }


class _FakeProjection:
    def __init__(self) -> None:
        self.weight = object()
        self.bias = {"must_remain": "unchanged"}


class _FakeFunctionalLinear:
    def __init__(self, returned: object) -> None:
        self.returned = returned
        self.calls: list[tuple[object, object, object | None]] = []

    def __call__(
        self, input_value: object, weight: object, bias: object | None
    ) -> object:
        self.calls.append((input_value, weight, bias))
        return self.returned


class _FakeBatchTensor:
    shape = (1, 2)

    def __getitem__(self, index: int) -> _FakeBatchTensor:
        if index != 0:
            raise IndexError(index)
        return self

    def detach(self) -> _FakeBatchTensor:
        return self

    def to(self, *, device: str) -> _FakeBatchTensor:
        if device != "cpu":
            raise AssertionError(f"unexpected device {device}")
        return self

    def contiguous(self) -> _FakeBatchTensor:
        return self


class _FakeParameter:
    def __init__(self, shape: tuple[int, ...], dtype: object) -> None:
        self.shape = shape
        self.dtype = dtype


class _FakeProjectionParameters:
    def __init__(self, output_width: int, dtype: object) -> None:
        self.weight = _FakeParameter((output_width, trace.MODEL_HIDDEN_SIZE), dtype)
        self.bias = _FakeParameter((output_width,), dtype)


class _FakeAttentionParameters:
    def __init__(self, dtype: object) -> None:
        self.q_proj = _FakeProjectionParameters(trace.MODEL_HIDDEN_SIZE, dtype)
        kv_width = trace.MODEL_KEY_VALUE_HEAD_COUNT * trace.MODEL_HEAD_DIMENSION
        self.k_proj = _FakeProjectionParameters(kv_width, dtype)
        self.v_proj = _FakeProjectionParameters(kv_width, dtype)


class _FakeTorchDtype:
    bfloat16 = object()


class Qwen3BQkvBiasBoundaryTraceTests(unittest.TestCase):
    def test_import_keeps_torch_transformers_and_safetensors_lazy(self) -> None:
        package_root = Path(__file__).resolve().parents[1]
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(package_root)
        command = (
            "import sys; import riley_reference.qwen3b_qkv_bias_boundary_trace; "
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

    def test_manifest_accepts_fixed_qkv_boundary_contract(self) -> None:
        trace.validate_manifest(_manifest())

    def test_manifest_accepts_canonical_json_object_order(self) -> None:
        trace.validate_manifest(json.loads(json.dumps(_manifest(), sort_keys=True)))

    def test_trace_profile_explicitly_rules_out_fused_internal_claim(self) -> None:
        self.assertEqual(
            trace._trace_profile_document(),
            {
                "capture_domain": "all-2048-token-positions",
                "id": trace.TRACE_ID,
                "pair_order": "unbiased-linear-then-unmodified-module-output",
                "tensor_count": 6,
                "unbiased_linear_semantics": (
                    "post-module-hook:torch.nn.functional.linear(input,weight,bias=None)"
                ),
                "post_bias_semantics": "unmodified-module-forward-hook",
            },
        )

    def test_projection_helper_saves_unmodified_output_before_shadow_linear_without_mutation(
        self,
    ) -> None:
        module = _FakeProjection()
        input_value = object()
        output_value = object()
        unbiased_value = object()
        functional_linear = _FakeFunctionalLinear(unbiased_value)
        captured: list[tuple[str, object]] = []
        bias_before = module.bias
        bias_contents_before = dict(module.bias)

        trace._capture_projection_bias_boundary(
            capture=lambda name, tensor: captured.append((name, tensor)),
            projection_name="layer0.q_proj",
            module=module,
            arguments=(input_value,),
            output=output_value,
            functional_linear=functional_linear,
        )

        self.assertEqual(functional_linear.calls, [(input_value, module.weight, None)])
        self.assertIs(module.bias, bias_before)
        self.assertEqual(module.bias, bias_contents_before)
        self.assertEqual(
            captured,
            [
                ("layer0.q_proj", output_value),
                ("layer0.q_proj.unbiased_linear", unbiased_value),
            ],
        )

    def test_projection_helper_saves_kv_output_before_shadow_and_preserves_identity(
        self,
    ) -> None:
        for projection_name in ("layer0.k_proj", "layer0.v_proj"):
            with self.subTest(projection_name=projection_name):
                module = _FakeProjection()
                input_value = object()
                output_value = object()
                unbiased_value = object()
                captured: list[tuple[str, object]] = []
                functional_linear = _FakeFunctionalLinear(unbiased_value)

                trace._capture_projection_bias_boundary(
                    capture=lambda name, tensor, captured=captured: captured.append(
                        (name, tensor)
                    ),
                    projection_name=projection_name,
                    module=module,
                    arguments=(input_value,),
                    output=output_value,
                    functional_linear=functional_linear,
                )

                self.assertIs(captured[0][1], output_value)
                self.assertIs(captured[1][1], unbiased_value)
                self.assertEqual(
                    [name for name, _ in captured],
                    [projection_name, f"{projection_name}.unbiased_linear"],
                )

    def test_projection_helper_rejects_missing_bias_without_calling_linear(
        self,
    ) -> None:
        module = _FakeProjection()
        module.bias = None
        functional_linear = _FakeFunctionalLinear(object())
        with self.assertRaisesRegex(trace.Qwen3BQkvBiasBoundaryTraceError, "no bias"):
            trace._capture_projection_bias_boundary(
                capture=lambda _name, _tensor: None,
                projection_name="layer0.q_proj",
                module=module,
                arguments=(object(),),
                output=object(),
                functional_linear=functional_linear,
            )
        self.assertEqual(functional_linear.calls, [])

    def test_projection_helper_rejects_wrong_projection_without_calling_linear(
        self,
    ) -> None:
        module = _FakeProjection()
        functional_linear = _FakeFunctionalLinear(object())
        with self.assertRaisesRegex(
            trace.Qwen3BQkvBiasBoundaryTraceError, "projection name"
        ):
            trace._capture_projection_bias_boundary(
                capture=lambda _name, _tensor: None,
                projection_name="layer1.q_proj",
                module=module,
                arguments=(object(),),
                output=object(),
                functional_linear=functional_linear,
            )
        self.assertEqual(functional_linear.calls, [])

    def test_capture_without_batch_rejects_double_capture(self) -> None:
        backend = object.__new__(trace.HuggingFaceQwen3BQkvBiasBoundaryTraceBackend)
        captured: dict[str, object] = {}
        trace.HuggingFaceQwen3BQkvBiasBoundaryTraceBackend._capture_without_batch(
            backend,
            captured,
            "layer0.q_proj.unbiased_linear",
            _FakeBatchTensor(),
        )
        with self.assertRaisesRegex(
            trace.Qwen3BQkvBiasBoundaryTraceError, "more than once"
        ):
            trace.HuggingFaceQwen3BQkvBiasBoundaryTraceBackend._capture_without_batch(
                backend,
                captured,
                "layer0.q_proj.unbiased_linear",
                _FakeBatchTensor(),
            )

    def test_layer0_projection_parameter_contract_requires_qkv_bf16_shapes_and_biases(
        self,
    ) -> None:
        torch = _FakeTorchDtype()
        attention = _FakeAttentionParameters(torch.bfloat16)
        trace._validate_layer0_projection_parameters(attention, torch)
        attention.k_proj.bias = _FakeParameter(
            (trace.MODEL_HIDDEN_SIZE,), torch.bfloat16
        )
        with self.assertRaisesRegex(
            trace.Qwen3BQkvBiasBoundaryTraceError, "k_proj parameter contract"
        ):
            trace._validate_layer0_projection_parameters(attention, torch)

    def test_collect_source_provenance_accepts_git_sha1_revision(self) -> None:
        repository = Path(__file__).resolve().parents[4]
        provenance = trace.collect_source_provenance(repository)
        self.assertRegex(provenance["git_revision"], r"^[0-9a-f]{40,64}$")

    def test_manifest_rejects_pair_semantics_tampering(self) -> None:
        document = _manifest()
        document["trace_profile"]["unbiased_linear_semantics"] = "fused-intermediate"
        with self.assertRaisesRegex(trace.Qwen3BQkvBiasBoundaryTraceError, "profile"):
            trace.validate_manifest(document)

    def test_manifest_rejects_unbiased_projection_shape_tampering(self) -> None:
        document = _manifest()
        document["tensors"]["layer0.q_proj.unbiased_linear"]["shape"] = [2_048, 256]
        with self.assertRaisesRegex(trace.Qwen3BQkvBiasBoundaryTraceError, "shape"):
            trace.validate_manifest(document)

    def test_output_paths_reject_repository_before_checkpoint_or_backend(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(
                trace.Qwen3BQkvBiasBoundaryTraceError, "outside"
            ):
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
        document["provenance"]["source_repository"]["sources"].pop(
            "qkv_bias_boundary_trace"
        )
        with self.assertRaisesRegex(trace.Qwen3BQkvBiasBoundaryTraceError, "source"):
            trace.validate_manifest(document)


if __name__ == "__main__":
    unittest.main()
