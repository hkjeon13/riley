from __future__ import annotations

import contextlib
import hashlib
import struct
import weakref
import unittest

from riley_reference.constants import GOLDEN_GREEDY_MAX_NEW_TOKENS
from riley_reference.cli import _build_parser
from riley_reference.fixture import FixtureError, PromptRecord, generate_fixture
from riley_reference.hf_backend import HuggingFaceBackend

from .support import FakeBackend, fixture_provenance


class _FakeTensor:
    def __init__(self, data, *, dtype="long") -> None:
        self.data = data
        self.dtype = dtype

    @property
    def shape(self) -> tuple[int, ...]:
        if not isinstance(self.data, list):
            return ()
        if not self.data:
            return (0,)
        if isinstance(self.data[0], list):
            return (len(self.data), len(self.data[0]))
        return (len(self.data),)

    def unsqueeze(self, dimension: int):
        if dimension != 0 or len(self.shape) != 1:
            raise AssertionError("fake only supports vector.unsqueeze(0)")
        return _FakeTensor([list(self.data)], dtype=self.dtype)


class _FakeScalar:
    def __init__(self, value: int) -> None:
        self._value = value

    def item(self) -> int:
        return self._value


class _FakeScores:
    def __init__(self, winning_token: int) -> None:
        self.winning_token = winning_token


class _FakeLogits:
    def __init__(self, winning_token: int) -> None:
        self._scores = _FakeScores(winning_token)

    def __getitem__(self, key):
        if key != (0, -1):
            raise AssertionError(f"unexpected fake logits index: {key!r}")
        return self._scores


class _FakeCache:
    def __init__(self, length: int) -> None:
        self.length = length


class _FakeOutput:
    def __init__(self, token: int, cache: _FakeCache | None) -> None:
        self.logits = _FakeLogits(token)
        self.past_key_values = cache


class _FakeModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.output_refs: list[weakref.ReferenceType[_FakeOutput]] = []

    def __call__(self, **kwargs):
        if "cache_position" in kwargs:
            raise AssertionError("Transformers 5.15 removed cache_position")
        input_ids = kwargs["input_ids"]
        past = kwargs.get("past_key_values")
        prefix_length = input_ids.shape[1] + (0 if past is None else past.length)
        position_ids = kwargs["position_ids"]
        expected_positions = (
            [list(range(prefix_length))]
            if past is None
            else [[prefix_length - 1]]
        )
        if position_ids.data != expected_positions:
            raise AssertionError(
                f"positions {position_ids.data!r} != {expected_positions!r}"
            )
        self.calls.append(
            {
                "use_cache": kwargs["use_cache"],
                "prefix_length": prefix_length,
            }
        )
        output = _FakeOutput(
            token=100 + prefix_length,
            cache=_FakeCache(prefix_length) if kwargs["use_cache"] else None,
        )
        self.output_refs.append(weakref.ref(output))
        return output


class _FakeCuda:
    def __init__(self, model: _FakeModel) -> None:
        self._model = model
        self.live_outputs_at_flush: list[int] = []

    def empty_cache(self) -> None:
        self.live_outputs_at_flush.append(
            sum(reference() is not None for reference in self._model.output_refs)
        )


class _FakeTorch:
    long = "long"

    def __init__(self, model: _FakeModel) -> None:
        self.cuda = _FakeCuda(model)

    @staticmethod
    def inference_mode():
        return contextlib.nullcontext()

    @staticmethod
    def arange(stop: int, *, device, dtype):
        del device
        return _FakeTensor(list(range(stop)), dtype=dtype)

    @staticmethod
    def tensor(data, *, device, dtype):
        del device
        return _FakeTensor(data, dtype=dtype)

    @staticmethod
    def ones_like(tensor: _FakeTensor):
        return _FakeTensor(
            [[1 for _ in row] for row in tensor.data], dtype=tensor.dtype
        )

    @staticmethod
    def cat(tensors: tuple[_FakeTensor, _FakeTensor], *, dim: int):
        if dim != 1:
            raise AssertionError("fake only supports dim=1 concatenation")
        left, right = tensors
        return _FakeTensor(
            [left_row + right_row for left_row, right_row in zip(left.data, right.data)],
            dtype=left.dtype,
        )

    @staticmethod
    def argmax(scores: _FakeScores):
        return _FakeScalar(scores.winning_token)


def _fake_hf_backend() -> tuple[HuggingFaceBackend, _FakeModel, _FakeTorch]:
    model = _FakeModel()
    torch = _FakeTorch(model)
    backend = object.__new__(HuggingFaceBackend)
    backend._torch = torch
    backend._model = model
    backend._device = "cuda:0"
    backend.eos_token_ids = ()
    return backend, model, torch


class HuggingFaceRegressionTests(unittest.TestCase):
    def test_cli_defaults_to_predeclared_exact_golden_window(self) -> None:
        arguments = _build_parser().parse_args(
            [
                "generate",
                "--prompts",
                "prompts.jsonl",
                "--output",
                "fixture.json",
                "--repo-root",
                ".",
            ]
        )
        self.assertEqual(arguments.max_new_tokens, GOLDEN_GREEDY_MAX_NEW_TOKENS)

    def test_fake_cache_paths_use_same_positions_and_release_each_output(self) -> None:
        backend, model, torch = _fake_hf_backend()
        input_ids = _FakeTensor([[7, 8]])
        attention_mask = _FakeTensor([[1, 1]])

        cache_on = backend._greedy_cache_on(input_ids, attention_mask, 3)
        cache_off = backend._greedy_cache_off(input_ids, attention_mask, 3)

        self.assertEqual(cache_on, ((102, 103, 104), "max_new_tokens"))
        self.assertEqual(cache_off, cache_on)
        self.assertEqual(
            [(call["use_cache"], call["prefix_length"]) for call in model.calls],
            [(True, 2), (True, 3), (True, 4), (False, 2), (False, 3), (False, 4)],
        )
        self.assertGreaterEqual(len(torch.cuda.live_outputs_at_flush), 8)
        self.assertEqual(
            set(torch.cuda.live_outputs_at_flush),
            {0},
            "model outputs must die before the CUDA allocator is flushed",
        )

    def test_seventeenth_golden_step_is_rejected_before_backend_execution(self) -> None:
        class NeverCalledBackend(FakeBackend):
            def generate_case(self, *args, **kwargs):
                raise AssertionError("unstable request must fail before backend execution")

        prompt = PromptRecord(
            prompt_id="minimal",
            category="minimal",
            language="none",
            text="",
            target_prompt_tokens=None,
            boundary_kind="none",
            expected_behavior="tokenizer fallback",
        )
        with self.assertRaisesRegex(FixtureError, "cache-parity window of 16"):
            generate_fixture(
                (prompt,),
                hashlib.sha256(b"prompt-corpus").hexdigest(),
                NeverCalledBackend(),
                max_new_tokens=GOLDEN_GREEDY_MAX_NEW_TOKENS + 1,
                hidden_state_index=1,
                top_k=2,
                seed=0,
                provenance=fixture_provenance(),
            )

    def test_empty_benchmark_prompt_uses_bos_before_eos(self) -> None:
        class EmptyTokenizer:
            bos_token_id = 7
            eos_token_id = 9

            @staticmethod
            def encode(text: str, *, add_special_tokens: bool):
                del text, add_special_tokens
                return []

        backend = object.__new__(HuggingFaceBackend)
        backend._tokenizer = EmptyTokenizer()
        self.assertEqual(backend._materialize_benchmark_rows(("",), 4), [[7, 7, 7, 7]])

    def test_token_hash_is_concatenated_unsigned_little_endian(self) -> None:
        expected = hashlib.sha256(
            struct.pack("<I", 0) + struct.pack("<I", 0x12345678)
        ).hexdigest()
        self.assertEqual(
            HuggingFaceBackend._token_id_hashes([[0, 0x12345678]]),
            (expected,),
        )


if __name__ == "__main__":
    unittest.main()
