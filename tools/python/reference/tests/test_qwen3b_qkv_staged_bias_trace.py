from __future__ import annotations

import contextlib
import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from riley_reference import qwen3b_qkv_staged_bias_trace as trace
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
            "path": "qwen3b-qkv-staged-bias.safetensors",
            "sha256": _sha("e"),
            "format": "safetensors",
            "tensor_count": len(trace.TRACE_TENSORS),
        },
        "tensors": tensors,
    }


_UNSET = object()


class _FakeProjection:
    def __init__(self, *, bias: object = _UNSET) -> None:
        self.weight = object()
        self.bias = {"must_remain": "unchanged"} if bias is _UNSET else bias


class _FakeFunctionalLinear:
    def __init__(
        self, returned: object | None, timeline: list[object] | None = None
    ) -> None:
        self.returned = returned
        self.timeline = timeline
        self.calls: list[tuple[object, object, object | None]] = []

    def __call__(
        self, input_value: object, weight: object, bias: object | None
    ) -> object:
        self.calls.append((input_value, weight, bias))
        if self.timeline is not None:
            self.timeline.append(("functional_linear", input_value, weight, bias))
        if self.returned is not None:
            return self.returned
        if not isinstance(input_value, _FakeTensor):
            raise TypeError("fake linear requires a fake tensor input")
        width = int(weight.shape[0])
        return _FakeTensor(
            (*input_value.shape[:-1], width),
            input_value.dtype,
            f"shadow-{len(self.calls)}",
            self.timeline if self.timeline is not None else [],
        )


class _FakeTensor:
    """Small operation-recording tensor used without importing Torch."""

    def __init__(
        self,
        shape: tuple[int, ...],
        dtype: object,
        label: str,
        timeline: list[object],
    ) -> None:
        self.shape = shape
        self.dtype = dtype
        self.label = label
        self.timeline = timeline

    def __getitem__(self, index: int) -> _FakeTensor:
        if index != 0:
            raise IndexError(index)
        if not self.shape:
            raise IndexError(index)
        return _FakeTensor(self.shape[1:], self.dtype, f"{self.label}[0]", self.timeline)

    def detach(self) -> _FakeTensor:
        self.timeline.append(("detach", self.label))
        return self

    def to(
        self, *, device: object | None = None, dtype: object | None = None
    ) -> _FakeTensor:
        if (device is None) == (dtype is None):
            raise AssertionError("fake tensor expects exactly one to() target")
        if dtype is not None:
            self.timeline.append(("to_dtype", self.label, dtype))
            return _FakeTensor(
                self.shape,
                dtype,
                f"{self.label}.to({dtype})",
                self.timeline,
            )
        self.timeline.append(("to_device", self.label, device))
        return _FakeTensor(
            self.shape,
            self.dtype,
            f"{self.label}.to({device})",
            self.timeline,
        )

    def add(self, other: object) -> _FakeTensor:
        if not isinstance(other, _FakeTensor):
            raise TypeError("fake staged bias must be a tensor")
        self.timeline.append(("add", self.label, other.label))
        return _FakeTensor(
            self.shape,
            self.dtype,
            f"({self.label}+{other.label})",
            self.timeline,
        )

    def contiguous(self) -> _FakeTensor:
        self.timeline.append(("contiguous", self.label))
        return self

    def unsqueeze(self, dimension: int) -> _FakeTensor:
        if dimension != 0:
            raise AssertionError(dimension)
        return _FakeTensor((1, *self.shape), self.dtype, self.label, self.timeline)


class _FakeFinite:
    def all(self) -> _FakeFinite:
        return self

    def item(self) -> bool:
        return True


class _FakeCuda:
    def __init__(self) -> None:
        self.empty_cache_calls = 0

    def empty_cache(self) -> None:
        self.empty_cache_calls += 1


class _FakeTorch:
    bfloat16 = "bf16"
    float32 = "fp32"
    long = "int64"

    def __init__(self, timeline: list[object]) -> None:
        self.timeline = timeline
        self.cuda = _FakeCuda()
        self.nn = SimpleNamespace(
            functional=SimpleNamespace(linear=_FakeFunctionalLinear(None, timeline))
        )

    def tensor(self, values: object, *, dtype: object, device: object) -> _FakeTensor:
        self.timeline.append(("tensor", dtype, device))
        return _FakeTensor(
            (1, oracle.PROMPT_TOKEN_COUNT), dtype, "input_ids", self.timeline
        )

    def ones_like(self, tensor: _FakeTensor) -> _FakeTensor:
        self.timeline.append(("ones_like", tensor.label))
        return _FakeTensor(tensor.shape, tensor.dtype, "attention_mask", self.timeline)

    def arange(self, count: int, *, dtype: object, device: object) -> _FakeTensor:
        if count != oracle.PROMPT_TOKEN_COUNT:
            raise AssertionError(count)
        self.timeline.append(("arange", count, dtype, device))
        return _FakeTensor((count,), dtype, "position_ids", self.timeline)

    def inference_mode(self) -> contextlib.AbstractContextManager[None]:
        return contextlib.nullcontext()

    def isfinite(self, tensor: _FakeTensor) -> _FakeFinite:
        self.timeline.append(("isfinite", tensor.label))
        return _FakeFinite()


class _FakeHookHandle:
    def __init__(self, hooks: list[object], callback: object) -> None:
        self._hooks = hooks
        self._callback = callback
        self.removed = False

    def remove(self) -> None:
        self.removed = True
        self._hooks.remove(self._callback)


class _FakeProjectionForCapture:
    def __init__(
        self, *, name: str, output_width: int, torch: _FakeTorch, timeline: list[object]
    ) -> None:
        self.name = name
        self.weight = SimpleNamespace(
            shape=(output_width, trace.MODEL_HIDDEN_SIZE), dtype=torch.bfloat16
        )
        self.bias = _FakeTensor((output_width,), torch.bfloat16, f"{name}.bias", timeline)
        self._output = _FakeTensor(
            (1, oracle.PROMPT_TOKEN_COUNT, output_width),
            torch.bfloat16,
            f"{name}.actual",
            timeline,
        )
        self._hooks: list[object] = []
        self._timeline = timeline

    def register_forward_hook(self, callback: object) -> _FakeHookHandle:
        self._hooks.append(callback)
        return _FakeHookHandle(self._hooks, callback)

    def invoke(self, input_value: _FakeTensor) -> None:
        self._timeline.append(("module_output", self.name))
        for callback in tuple(self._hooks):
            callback(self, (input_value,), self._output)


class _FakeModel:
    def __init__(self, attention: object, torch: _FakeTorch, timeline: list[object]) -> None:
        self.attention = attention
        self.torch = torch
        self.timeline = timeline
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs: object) -> object:
        self.calls.append(dict(kwargs))
        input_value = _FakeTensor(
            (1, oracle.PROMPT_TOKEN_COUNT, trace.MODEL_HIDDEN_SIZE),
            self.torch.bfloat16,
            "projection_input",
            self.timeline,
        )
        self.attention.q_proj.invoke(input_value)
        self.attention.k_proj.invoke(input_value)
        self.attention.v_proj.invoke(input_value)
        return SimpleNamespace(past_key_values=None)


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
    bfloat16 = "bf16"
    float32 = "fp32"


class Qwen3BQkvStagedBiasTraceTests(unittest.TestCase):
    def test_import_keeps_torch_transformers_and_safetensors_lazy(self) -> None:
        package_root = Path(__file__).resolve().parents[1]
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(package_root)
        command = (
            "import sys; import riley_reference.qwen3b_qkv_staged_bias_trace; "
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

    def test_manifest_accepts_fixed_qkv_staged_bias_contract(self) -> None:
        trace.validate_manifest(_manifest())

    def test_manifest_accepts_canonical_json_object_order(self) -> None:
        trace.validate_manifest(json.loads(json.dumps(_manifest(), sort_keys=True)))

    def test_trace_profile_explicitly_records_both_shadow_endpoint_contracts(self) -> None:
        self.assertEqual(
            trace._trace_profile_document(),
            {
                "capture_domain": "all-2048-token-positions",
                "id": trace.TRACE_ID,
                "pair_order": (
                    "unbiased-linear-then-explicit-fp32-bias-add-bf16-round-"
                    "then-unmodified-module-output"
                ),
                "tensor_count": 9,
                "unbiased_linear_semantics": (
                    "post-module-hook:torch.nn.functional.linear(input,weight,bias=None)"
                ),
                "staged_bias_bf16_semantics": (
                    "post-module-hook:bf16(no_bias.to(float32)+bias.to(float32))"
                ),
                "post_bias_semantics": "unmodified-module-forward-hook",
            },
        )

    def test_projection_helper_saves_real_output_then_v2_no_bias_then_explicit_staged_bias(
        self,
    ) -> None:
        timeline: list[object] = []
        torch = _FakeTorchDtype()
        module = _FakeProjection(
            bias=_FakeTensor((trace.MODEL_HIDDEN_SIZE,), torch.bfloat16, "bias", timeline)
        )
        input_value = object()
        output_value = object()
        unbiased_value = _FakeTensor(
            (1, 2, trace.MODEL_HIDDEN_SIZE), torch.bfloat16, "unbiased", timeline
        )
        functional_linear = _FakeFunctionalLinear(unbiased_value, timeline)
        captured: list[tuple[str, object]] = []
        bias_before = module.bias

        trace._capture_projection_staged_bias(
            capture=lambda name, tensor: (
                timeline.append(("capture", name, tensor)), captured.append((name, tensor))
            ),
            projection_name="layer0.q_proj",
            module=module,
            arguments=(input_value,),
            output=output_value,
            functional_linear=functional_linear,
            torch=torch,
        )

        self.assertEqual(functional_linear.calls, [(input_value, module.weight, None)])
        self.assertIs(module.bias, bias_before)
        self.assertEqual(
            [name for name, _ in captured],
            [
                "layer0.q_proj",
                "layer0.q_proj.unbiased_linear",
                "layer0.q_proj.staged_bias_bf16",
            ],
        )
        self.assertIs(captured[0][1], output_value)
        self.assertIs(captured[1][1], unbiased_value)
        self.assertEqual(
            timeline,
            [
                ("capture", "layer0.q_proj", output_value),
                ("functional_linear", input_value, module.weight, None),
                ("capture", "layer0.q_proj.unbiased_linear", unbiased_value),
                ("to_dtype", "unbiased", torch.float32),
                ("to_dtype", "bias", torch.float32),
                (
                    "add",
                    "unbiased.to(fp32)",
                    "bias.to(fp32)",
                ),
                (
                    "to_dtype",
                    "(unbiased.to(fp32)+bias.to(fp32))",
                    torch.bfloat16,
                ),
                ("capture", "layer0.q_proj.staged_bias_bf16", captured[2][1]),
            ],
        )
        self.assertEqual(captured[2][1].dtype, torch.bfloat16)

    def test_projection_helper_applies_the_same_triple_to_k_and_v(self) -> None:
        for projection_name, width in (
            ("layer0.k_proj", trace.MODEL_KEY_VALUE_HEAD_COUNT * trace.MODEL_HEAD_DIMENSION),
            ("layer0.v_proj", trace.MODEL_KEY_VALUE_HEAD_COUNT * trace.MODEL_HEAD_DIMENSION),
        ):
            with self.subTest(projection_name=projection_name):
                timeline: list[object] = []
                torch = _FakeTorchDtype()
                module = _FakeProjection(
                    bias=_FakeTensor((width,), torch.bfloat16, "bias", timeline)
                )
                output_value = object()
                unbiased_value = _FakeTensor((1, 2, width), torch.bfloat16, "unbiased", timeline)
                captured: list[tuple[str, object]] = []
                trace._capture_projection_staged_bias(
                    capture=lambda name, tensor: captured.append((name, tensor)),
                    projection_name=projection_name,
                    module=module,
                    arguments=(object(),),
                    output=output_value,
                    functional_linear=_FakeFunctionalLinear(unbiased_value),
                    torch=torch,
                )
                self.assertEqual(
                    [name for name, _ in captured],
                    [
                        projection_name,
                        f"{projection_name}.unbiased_linear",
                        f"{projection_name}.staged_bias_bf16",
                    ],
                )
                self.assertIs(captured[0][1], output_value)
                self.assertIs(captured[1][1], unbiased_value)
                self.assertEqual(captured[2][1].dtype, torch.bfloat16)

    def test_projection_helper_rejects_missing_bias_before_no_bias_or_staged_calls(
        self,
    ) -> None:
        module = _FakeProjection(bias=None)
        functional_linear = _FakeFunctionalLinear(object())
        with self.assertRaisesRegex(trace.Qwen3BQkvStagedBiasTraceError, "no bias"):
            trace._capture_projection_staged_bias(
                capture=lambda _name, _tensor: None,
                projection_name="layer0.q_proj",
                module=module,
                arguments=(object(),),
                output=object(),
                functional_linear=functional_linear,
                torch=_FakeTorchDtype(),
            )
        self.assertEqual(functional_linear.calls, [])

    def test_projection_helper_rejects_wrong_projection_before_no_bias_or_staged_calls(
        self,
    ) -> None:
        functional_linear = _FakeFunctionalLinear(object())
        with self.assertRaisesRegex(trace.Qwen3BQkvStagedBiasTraceError, "projection name"):
            trace._capture_projection_staged_bias(
                capture=lambda _name, _tensor: None,
                projection_name="layer1.q_proj",
                module=_FakeProjection(),
                arguments=(object(),),
                output=object(),
                functional_linear=functional_linear,
                torch=_FakeTorchDtype(),
            )
        self.assertEqual(functional_linear.calls, [])

    def test_fake_backend_captures_all_nine_contract_endpoints_and_removes_hooks(self) -> None:
        timeline: list[object] = []
        torch = _FakeTorch(timeline)
        kv_width = trace.MODEL_KEY_VALUE_HEAD_COUNT * trace.MODEL_HEAD_DIMENSION
        attention = SimpleNamespace(
            q_proj=_FakeProjectionForCapture(
                name="q_proj", output_width=trace.MODEL_HIDDEN_SIZE, torch=torch, timeline=timeline
            ),
            k_proj=_FakeProjectionForCapture(
                name="k_proj", output_width=kv_width, torch=torch, timeline=timeline
            ),
            v_proj=_FakeProjectionForCapture(
                name="v_proj", output_width=kv_width, torch=torch, timeline=timeline
            ),
        )
        model = _FakeModel(attention, torch, timeline)
        backend = object.__new__(trace.HuggingFaceQwen3BQkvStagedBiasTraceBackend)
        backend._model = model
        backend._torch = torch
        backend._layers = (SimpleNamespace(self_attn=attention),)
        backend._device = "cuda:0"

        captured = backend.capture(
            (oracle.EXPECTED_PROMPT_TOKEN_ID,) * oracle.PROMPT_TOKEN_COUNT
        )

        self.assertEqual(tuple(captured.tensors), trace.TRACE_TENSORS)
        self.assertEqual(
            [
                name
                for name, tensor in captured.tensors.items()
                if tensor.dtype == torch.bfloat16
            ],
            list(trace.TRACE_TENSORS),
        )
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(torch.cuda.empty_cache_calls, 1)
        self.assertFalse(attention.q_proj._hooks)
        self.assertFalse(attention.k_proj._hooks)
        self.assertFalse(attention.v_proj._hooks)
        self.assertEqual(
            [call[2] for call in torch.nn.functional.linear.calls],
            [None, None, None],
        )
        for projection in ("q_proj", "k_proj", "v_proj"):
            self.assertIn(("module_output", projection), timeline)
            self.assertTrue(
                any(
                    item[:2] == ("to_dtype", f"{projection}.bias")
                    for item in timeline
                    if isinstance(item, tuple)
                )
            )

    def test_capture_without_batch_rejects_double_capture(self) -> None:
        backend = object.__new__(trace.HuggingFaceQwen3BQkvStagedBiasTraceBackend)
        captured: dict[str, object] = {}
        tensor = _FakeTensor((1, 2), "bf16", "capture", [])
        trace.HuggingFaceQwen3BQkvStagedBiasTraceBackend._capture_without_batch(
            backend,
            captured,
            "layer0.q_proj.unbiased_linear",
            tensor,
        )
        with self.assertRaisesRegex(trace.Qwen3BQkvStagedBiasTraceError, "more than once"):
            trace.HuggingFaceQwen3BQkvStagedBiasTraceBackend._capture_without_batch(
                backend,
                captured,
                "layer0.q_proj.unbiased_linear",
                tensor,
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
            trace.Qwen3BQkvStagedBiasTraceError, "k_proj parameter contract"
        ):
            trace._validate_layer0_projection_parameters(attention, torch)

    def test_collect_source_provenance_binds_every_v2b_source_in_a_clean_git_repo(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory) / "repo"
            repository.mkdir()
            for relative in trace.SOURCE_PATHS.values():
                source = repository / relative
                source.parent.mkdir(parents=True, exist_ok=True)
                source.write_text(relative + "\n", encoding="utf-8")
            for command in (
                ["git", "init", "--quiet"],
                ["git", "config", "user.email", "trace@example.invalid"],
                ["git", "config", "user.name", "Trace Test"],
                ["git", "add", "."],
                ["git", "commit", "--quiet", "-m", "source fixture"],
            ):
                subprocess.run(command, cwd=repository, check=True)
            provenance = trace.collect_source_provenance(repository)
        self.assertRegex(provenance["git_revision"], r"^[0-9a-f]{40,64}$")
        self.assertFalse(provenance["source_dirty"])
        self.assertEqual(set(provenance["sources"]), set(trace.SOURCE_PATHS))

    def test_manifest_rejects_staged_semantics_tampering(self) -> None:
        document = _manifest()
        document["trace_profile"]["staged_bias_bf16_semantics"] = "fused-intermediate"
        with self.assertRaisesRegex(trace.Qwen3BQkvStagedBiasTraceError, "profile"):
            trace.validate_manifest(document)

    def test_manifest_rejects_staged_projection_shape_tampering(self) -> None:
        document = _manifest()
        document["tensors"]["layer0.q_proj.staged_bias_bf16"]["shape"] = [2_048, 256]
        with self.assertRaisesRegex(trace.Qwen3BQkvStagedBiasTraceError, "shape"):
            trace.validate_manifest(document)

    def test_output_paths_reject_repository_before_checkpoint_or_backend(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(trace.Qwen3BQkvStagedBiasTraceError, "outside"):
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
        with self.assertRaisesRegex(trace.Qwen3BQkvStagedBiasTraceError, "source"):
            trace.validate_manifest(document)


if __name__ == "__main__":
    unittest.main()
