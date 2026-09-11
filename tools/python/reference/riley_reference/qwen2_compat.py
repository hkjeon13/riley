"""Produce and validate the pinned PR 12 Qwen2 compatibility golden.

``generate`` imports PyTorch and Transformers and is intentionally restricted to
the canonical remote CUDA environment. ``validate`` uses only the Python standard
library, so the checked-in artifact can be verified without loading CUDA or a
model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "riley-qwen2-compat-v1"
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
MODEL_DTYPE = "bfloat16"
ATTENTION_IMPLEMENTATION = "eager"
MODEL_VOCAB_SIZE = 151_936
ADDRESSABLE_TOKEN_COUNT = 151_665
EOS_TOKEN_IDS = frozenset((151_645,))
MAX_NEW_TOKENS = 8
TOP_K = 10
PROBE_IDS = (0, 1, 151_643, 151_644, 151_645, 151_657, 151_664, 151_665, 151_935)
MODEL_FILE_SHA256 = {
    "config.json": "18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45",
    "model.safetensors": (
        "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe"
    ),
    "tokenizer.json": (
        "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"
    ),
    "tokenizer_config.json": (
        "5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583"
    ),
}
EXPECTED_ENVIRONMENT = {
    "python": "3.13.15",
    "torch": "2.13.0+cu130",
    "transformers": "5.15.1",
    "device": "NVIDIA GeForce RTX 4090",
    "compute_capability": "8.9",
}
EXPECTED_NVIDIA_DRIVER = "580.173.02"
CANONICAL_FIXTURE_SHA256 = (
    "42cc7f3fd04098bc4d70836ee9d18dbf919f158a010da3da6fdaa3d9deeceab7"
)
DEFAULT_SYSTEM_MESSAGE = (
    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
)
CASES = (
    (
        "english",
        [{"role": "user", "content": "Explain why the sky is blue in one sentence."}],
        40,
    ),
    (
        "korean",
        [
            {
                "role": "user",
                "content": "서울의 가을을 한 문장으로 묘사해 주세요.",
            }
        ],
        46,
    ),
    (
        "code",
        [
            {"role": "system", "content": "Answer with code only."},
            {
                "role": "user",
                "content": "Write a Rust function that adds two u32 values.",
            },
        ],
        30,
    ),
)


class FixtureValidationError(ValueError):
    """The fixture does not satisfy the pinned PR 12 contract."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(document: object) -> bytes:
    return (
        json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


def _expected_rendered_chat(messages: list[dict[str, str]]) -> str:
    materialized = messages
    if not materialized or materialized[0]["role"] != "system":
        materialized = [
            {"role": "system", "content": DEFAULT_SYSTEM_MESSAGE},
            *materialized,
        ]
    rendered = "".join(
        f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
        for message in materialized
    )
    return f"{rendered}<|im_start|>assistant\n"


def _require_exact_keys(
    value: dict[str, Any], expected: set[str], location: str
) -> None:
    actual = set(value)
    if actual != expected:
        raise FixtureValidationError(
            f"{location} keys differ: expected {sorted(expected)}, got {sorted(actual)}"
        )


def _require_token_ids(
    value: object,
    *,
    location: str,
    upper_bound: int,
    expected_length: int | None = None,
) -> list[int]:
    if not isinstance(value, list) or not all(type(token) is int for token in value):
        raise FixtureValidationError(f"{location} must be a list of integer token IDs")
    if expected_length is not None and len(value) != expected_length:
        raise FixtureValidationError(
            f"{location} must contain {expected_length} IDs, got {len(value)}"
        )
    if any(token < 0 or token >= upper_bound for token in value):
        raise FixtureValidationError(
            f"{location} contains an ID outside [0, {upper_bound})"
        )
    return value


def _require_finite_floats(
    value: object, *, location: str, expected_length: int
) -> list[float]:
    if not isinstance(value, list) or len(value) != expected_length:
        raise FixtureValidationError(
            f"{location} must contain {expected_length} floating-point values"
        )
    if not all(type(item) is float and math.isfinite(item) for item in value):
        raise FixtureValidationError(f"{location} must contain only finite JSON floats")
    return value


def _validate_document(document: object) -> None:
    if not isinstance(document, dict):
        raise FixtureValidationError("fixture root must be a JSON object")
    _require_exact_keys(
        document, {"schema_version", "model", "environment", "cases"}, "root"
    )
    if document["schema_version"] != SCHEMA_VERSION:
        raise FixtureValidationError(
            f"schema_version must be {SCHEMA_VERSION!r}"
        )

    expected_model = {
        "id": MODEL_ID,
        "revision": MODEL_REVISION,
        "dtype": MODEL_DTYPE,
        "attention": ATTENTION_IMPLEMENTATION,
        "files_sha256": MODEL_FILE_SHA256,
    }
    if document["model"] != expected_model:
        raise FixtureValidationError(
            "model provenance differs from the pinned contract"
        )
    if document["environment"] != EXPECTED_ENVIRONMENT:
        raise FixtureValidationError(
            "environment differs from the canonical remote CUDA environment"
        )

    cases = document["cases"]
    if not isinstance(cases, list) or len(cases) != len(CASES):
        raise FixtureValidationError(f"cases must contain exactly {len(CASES)} entries")
    for index, (case, (expected_name, expected_messages, prompt_length)) in enumerate(
        zip(cases, CASES, strict=True)
    ):
        location = f"cases[{index}]"
        if not isinstance(case, dict):
            raise FixtureValidationError(f"{location} must be an object")
        _require_exact_keys(
            case,
            {
                "name",
                "messages",
                "rendered_chat",
                "prompt_token_ids",
                "raw_last_logits",
                "greedy",
            },
            location,
        )
        if case["name"] != expected_name or case["messages"] != expected_messages:
            raise FixtureValidationError(f"{location} prompt identity differs")
        if case["rendered_chat"] != _expected_rendered_chat(expected_messages):
            raise FixtureValidationError(f"{location}.rendered_chat differs")
        _require_token_ids(
            case["prompt_token_ids"],
            location=f"{location}.prompt_token_ids",
            upper_bound=ADDRESSABLE_TOKEN_COUNT,
            expected_length=prompt_length,
        )

        raw_logits = case["raw_last_logits"]
        if not isinstance(raw_logits, dict):
            raise FixtureValidationError(
                f"{location}.raw_last_logits must be an object"
            )
        _require_exact_keys(
            raw_logits,
            {"top_token_ids", "top_values_f32", "probe_values_f32"},
            f"{location}.raw_last_logits",
        )
        _require_token_ids(
            raw_logits["top_token_ids"],
            location=f"{location}.raw_last_logits.top_token_ids",
            upper_bound=MODEL_VOCAB_SIZE,
            expected_length=TOP_K,
        )
        top_values = _require_finite_floats(
            raw_logits["top_values_f32"],
            location=f"{location}.raw_last_logits.top_values_f32",
            expected_length=TOP_K,
        )
        if any(left < right for left, right in zip(top_values, top_values[1:])):
            raise FixtureValidationError(
                f"{location}.raw_last_logits.top_values_f32 must be descending"
            )
        probes = raw_logits["probe_values_f32"]
        expected_probe_ids = {str(token) for token in PROBE_IDS}
        if not isinstance(probes, dict) or set(probes) != expected_probe_ids:
            raise FixtureValidationError(
                f"{location}.raw_last_logits.probe_values_f32 has the wrong probe IDs"
            )
        if not all(
            type(value) is float and math.isfinite(value)
            for value in probes.values()
        ):
            raise FixtureValidationError(
                f"{location}.raw_last_logits.probe_values_f32 must contain "
                "finite floats"
            )

        greedy = case["greedy"]
        if not isinstance(greedy, dict):
            raise FixtureValidationError(f"{location}.greedy must be an object")
        _require_exact_keys(
            greedy,
            {
                "addressable_token_count",
                "max_new_tokens",
                "cache_on_token_ids",
                "cache_off_token_ids",
            },
            f"{location}.greedy",
        )
        if greedy["addressable_token_count"] != ADDRESSABLE_TOKEN_COUNT:
            raise FixtureValidationError(
                f"{location}.greedy.addressable_token_count differs"
            )
        if greedy["max_new_tokens"] != MAX_NEW_TOKENS:
            raise FixtureValidationError(f"{location}.greedy.max_new_tokens differs")
        cache_on = _require_token_ids(
            greedy["cache_on_token_ids"],
            location=f"{location}.greedy.cache_on_token_ids",
            upper_bound=ADDRESSABLE_TOKEN_COUNT,
            expected_length=MAX_NEW_TOKENS,
        )
        cache_off = _require_token_ids(
            greedy["cache_off_token_ids"],
            location=f"{location}.greedy.cache_off_token_ids",
            upper_bound=ADDRESSABLE_TOKEN_COUNT,
            expected_length=MAX_NEW_TOKENS,
        )
        if cache_on != cache_off:
            raise FixtureValidationError(f"{location} cache-on/off parity differs")


def validate_fixture(path: Path) -> None:
    """Validate schema, pinned provenance, canonical serialization, and bytes."""

    raw = path.read_bytes()
    try:
        document = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FixtureValidationError(f"invalid UTF-8 JSON: {error}") from error
    _validate_document(document)
    if raw != _canonical_json_bytes(document):
        raise FixtureValidationError("fixture is not canonical sorted, indented JSON")
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    if actual_sha256 != CANONICAL_FIXTURE_SHA256:
        raise FixtureValidationError(
            "fixture bytes differ from the canonical remote artifact: "
            f"expected {CANONICAL_FIXTURE_SHA256}, got {actual_sha256}"
        )


def _verify_checkpoint(checkpoint: Path) -> dict[str, str]:
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"checkpoint directory does not exist: {checkpoint}")
    observed: dict[str, str] = {}
    for filename, expected_sha256 in MODEL_FILE_SHA256.items():
        path = checkpoint / filename
        if not path.is_file():
            raise FileNotFoundError(f"missing pinned checkpoint file: {path}")
        actual_sha256 = _sha256_file(path)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"{filename} SHA-256 differs: expected {expected_sha256}, "
                f"got {actual_sha256}"
            )
        observed[filename] = actual_sha256
    return observed


def _greedy(model: Any, torch: Any, prompt: Any, *, use_cache: bool) -> list[int]:
    sequence = prompt.clone()
    generated: list[int] = []
    past_key_values = None
    current = prompt
    with torch.inference_mode():
        for _ in range(MAX_NEW_TOKENS):
            output = model(
                input_ids=current,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
            logits = output.logits[:, -1, :].float()
            logits[:, ADDRESSABLE_TOKEN_COUNT:] = -torch.inf
            token = int(torch.argmax(logits, dim=-1).item())
            generated.append(token)
            token_tensor = torch.tensor(
                [[token]], dtype=torch.long, device=sequence.device
            )
            sequence = torch.cat((sequence, token_tensor), dim=1)
            if token in EOS_TOKEN_IDS:
                break
            if use_cache:
                past_key_values = output.past_key_values
                current = sequence[:, -1:]
            else:
                current = sequence
    return generated


def _generate(checkpoint: Path, output: Path, device_name: str) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if platform.python_version() != EXPECTED_ENVIRONMENT["python"]:
        raise RuntimeError(
            "Python version differs from the canonical remote environment: "
            f"expected {EXPECTED_ENVIRONMENT['python']}, "
            f"got {platform.python_version()}"
        )
    file_hashes = _verify_checkpoint(checkpoint)

    workspace_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if workspace_config not in (None, ":4096:8"):
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG must be :4096:8")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # Heavy imports are confined to the remote-only generation path.
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("Qwen2 compatibility generation requires CUDA")
    if platform.system() != "Linux":
        raise RuntimeError("canonical generation requires the remote Linux CUDA host")
    device = torch.device(device_name)
    if device.type != "cuda":
        raise RuntimeError("Qwen2 compatibility generation requires a CUDA device")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("canonical generation requires exactly one visible CUDA GPU")
    driver_query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=driver_version",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    driver_versions = [line.strip() for line in driver_query.stdout.splitlines()]
    if driver_versions != [EXPECTED_NVIDIA_DRIVER]:
        raise RuntimeError(
            "NVIDIA driver differs from the canonical remote contract: "
            f"expected {[EXPECTED_NVIDIA_DRIVER]}, got {driver_versions}"
        )
    torch.cuda.set_device(device)
    observed_environment = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "device": torch.cuda.get_device_name(device),
        "compute_capability": ".".join(
            str(value) for value in torch.cuda.get_device_capability(device)
        ),
    }
    if observed_environment != EXPECTED_ENVIRONMENT:
        raise RuntimeError(
            "CUDA environment differs from the pinned remote contract: "
            f"expected {EXPECTED_ENVIRONMENT}, got {observed_environment}"
        )

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint,
        trust_remote_code=False,
        local_files_only=True,
        use_fast=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        trust_remote_code=False,
        local_files_only=True,
        use_safetensors=True,
        dtype=torch.bfloat16,
        attn_implementation=ATTENTION_IMPLEMENTATION,
    ).to(device)
    model.eval()
    if len(tokenizer) != ADDRESSABLE_TOKEN_COUNT:
        raise RuntimeError(
            f"tokenizer length must be {ADDRESSABLE_TOKEN_COUNT}, got {len(tokenizer)}"
        )
    if model.config.vocab_size != MODEL_VOCAB_SIZE:
        raise RuntimeError(
            f"model vocab size must be {MODEL_VOCAB_SIZE}, "
            f"got {model.config.vocab_size}"
        )
    if tokenizer.eos_token_id not in EOS_TOKEN_IDS:
        raise RuntimeError(f"unexpected tokenizer EOS ID: {tokenizer.eos_token_id}")

    results: list[dict[str, object]] = []
    for name, messages, _prompt_length in CASES:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        prompt_ids = (
            encoded_prompt["input_ids"]
            if hasattr(encoded_prompt, "keys")
            else encoded_prompt
        ).to(device)
        with torch.inference_mode():
            forward_output = model(input_ids=prompt_ids, use_cache=False)
            last_logits = forward_output.logits[0, -1, :].float()
        top_values, top_ids = torch.topk(
            last_logits, k=TOP_K, largest=True, sorted=True
        )
        cache_on = _greedy(model, torch, prompt_ids, use_cache=True)
        cache_off = _greedy(model, torch, prompt_ids, use_cache=False)
        if cache_on != cache_off:
            raise RuntimeError(
                f"cache parity failed for {name}: {cache_on} != {cache_off}"
            )
        results.append(
            {
                "name": name,
                "messages": messages,
                "rendered_chat": rendered,
                "prompt_token_ids": [int(value) for value in prompt_ids[0].tolist()],
                "raw_last_logits": {
                    "top_token_ids": [int(value) for value in top_ids.cpu().tolist()],
                    "top_values_f32": [
                        float(value) for value in top_values.cpu().tolist()
                    ],
                    "probe_values_f32": {
                        str(token_id): float(last_logits[token_id].item())
                        for token_id in PROBE_IDS
                    },
                },
                "greedy": {
                    "addressable_token_count": ADDRESSABLE_TOKEN_COUNT,
                    "max_new_tokens": MAX_NEW_TOKENS,
                    "cache_on_token_ids": cache_on,
                    "cache_off_token_ids": cache_off,
                },
            }
        )

    document = {
        "schema_version": SCHEMA_VERSION,
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "dtype": MODEL_DTYPE,
            "attention": ATTENTION_IMPLEMENTATION,
            "files_sha256": file_hashes,
        },
        "environment": observed_environment,
        "cases": results,
    }
    _validate_document(document)
    payload = _canonical_json_bytes(document)
    payload_sha256 = hashlib.sha256(payload).hexdigest()
    if payload_sha256 != CANONICAL_FIXTURE_SHA256:
        raise RuntimeError(
            "generated bytes differ from the canonical PR 12 artifact: "
            f"expected {CANONICAL_FIXTURE_SHA256}, got {payload_sha256}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    created_output = False
    try:
        with output.open("xb") as handle:
            created_output = True
            handle.write(payload)
    except BaseException:
        if created_output:
            output.unlink(missing_ok=True)
        raise


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m riley_reference.qwen2_compat",
        description=(
            f"Pinned {MODEL_ID}@{MODEL_REVISION} PR 12 compatibility golden"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser(
        "generate", help="generate the golden on the canonical remote CUDA host"
    )
    generate.add_argument("--checkpoint", type=Path, required=True)
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--device", default="cuda:0")

    validate = subparsers.add_parser(
        "validate", help="validate the canonical artifact without importing PyTorch"
    )
    validate.add_argument("fixture", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "generate":
        _generate(args.checkpoint, args.output, args.device)
    else:
        validate_fixture(args.fixture)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
