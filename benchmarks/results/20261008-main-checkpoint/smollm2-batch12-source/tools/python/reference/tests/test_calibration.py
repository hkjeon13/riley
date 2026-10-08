from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
import math
import subprocess
import tempfile
import unittest
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from unittest import mock

from riley_reference.calibration import (
    ALTERNATE_CANDIDATE_REDUCTION_VARIANT,
    BF16_ORACLE_KIND,
    CALIBRATION_GATE_ID,
    CALIBRATION_THRESHOLDS,
    CANONICAL_CANDIDATE_REDUCTION_VARIANT,
    CANDIDATE_KIND,
    FP32_ORACLE_KIND,
    HF_ORACLE_REDUCTION_VARIANT,
    HF_SOURCE_PATHS,
    LEGACY_NATIVE_ENGINE_REVISION,
    LEGACY_NATIVE_SOURCE_PATHS,
    NATIVE_BUILD_ARGV,
    NATIVE_ENGINE_REVISION,
    NATIVE_EXECUTABLE_FILENAME,
    NATIVE_SOURCE_PATHS,
    ORACLE_MANIFEST_GATE_ID,
    ORACLE_REQUIRED_CANDIDATE_REDUCTION_VARIANTS,
    REQUIRED_CANDIDATE_REDUCTION_VARIANTS,
    CalibrationError,
    aggregate_tokenizer_sha256,
    compare_calibrations,
    gate_contract_document,
    load_correctness_report,
    metrics_pass,
    recompute_numeric_metrics,
    recompute_numeric_metrics_from_tensors,
    replay_validate_correctness_report,
    sha256_file,
    validate_calibration_manifest,
    verify_calibration_artifact,
)
from riley_reference.cli import _build_parser, main
from riley_reference.constants import (
    MODEL_CONFIG_SHA256,
    MODEL_ID,
    MODEL_REVISION,
    MODEL_WEIGHTS_SHA256,
    PYTHON_EXECUTABLE_SHA256,
    PYTHON_PLATFORM_MACHINE,
    PYTHON_PLATFORM_SYSTEM,
    PYTHON_VERSION,
    SAFETENSORS_VERSION,
    TORCH_VERSION,
    TOKENIZER_FILES_SHA256,
    TOKENIZER_SHA256,
    TRANSFORMERS_VERSION,
)
from riley_reference.oracle_calibration import (
    compare_hf_oracles,
    replay_validate_oracle_report,
)
from riley_reference.hf_calibration import (
    CapturedOracleCase,
    OracleArtifactMetadata,
    produce_hf_oracle,
)
from riley_reference.environment import PRIMARY_ENVIRONMENT_SNAPSHOT


FIXED_TIME = datetime(2026, 8, 24, 2, 3, 4, tzinfo=timezone.utc)
EMPTY_SHA = hashlib.sha256(b"").hexdigest()


class FakeTensor:
    def __init__(self, values: list[float], shape: tuple[int, ...], dtype: str) -> None:
        self.values = list(values)
        self.shape = shape
        self.dtype = dtype

    def detach(self):
        return self

    def cpu(self):
        return self

    def float(self):
        return FakeTensor(self.values, self.shape, "float32")

    def contiguous(self):
        return self

    def reshape(self, size: int):
        if size != -1:
            raise AssertionError("fake only supports flatten")
        return FakeTensor(self.values, (len(self.values),), self.dtype)

    def tolist(self):
        return list(self.values)


class MiniScalar:
    def __init__(self, value: float | bool) -> None:
        self.value = value

    def item(self) -> float | bool:
        return self.value


class MiniVector:
    """Enough tensor arithmetic to exercise the bounded-memory code path."""

    def __init__(self, values: list[float | bool]) -> None:
        self.values = list(values)

    def detach(self):
        return self

    def cpu(self):
        return self

    def float(self):
        return MiniVector([float(value) for value in self.values])

    def double(self):
        return MiniVector([float(value) for value in self.values])

    def contiguous(self):
        return self

    def reshape(self, size: int):
        if size != -1:
            raise AssertionError("mini tensor only supports flatten")
        return self

    def tolist(self):
        return list(self.values)

    def numel(self) -> int:
        return len(self.values)

    def __getitem__(self, index: slice):
        return MiniVector(self.values[index])

    def __sub__(self, other):
        return MiniVector([a - b for a, b in zip(self.values, other.values, strict=True)])

    def __mul__(self, other):
        return MiniVector([a * b for a, b in zip(self.values, other.values, strict=True)])

    def __truediv__(self, other):
        return MiniVector([a / b for a, b in zip(self.values, other.values, strict=True)])

    def abs(self):
        return MiniVector([abs(value) for value in self.values])

    def clamp_min(self, minimum: float):
        return MiniVector([max(float(value), minimum) for value in self.values])

    def isfinite(self):
        return MiniVector([math.isfinite(float(value)) for value in self.values])

    def all(self):
        return MiniScalar(all(bool(value) for value in self.values))

    def max(self):
        return MiniScalar(max(float(value) for value in self.values))

    def sum(self):
        return MiniScalar(sum(float(value) for value in self.values))


class FakeOracleBackend:
    def __init__(self, artifact_kind: str) -> None:
        self.artifact_kind = artifact_kind
        self.closed = False
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
            tokenizer_files_sha256=TOKENIZER_FILES_SHA256,
        )

    def capture_case(self, prompt) -> CapturedOracleCase:
        count = prompt.target_prompt_tokens or 1
        return CapturedOracleCase(
            input_token_ids=tuple([1] * count),
            first_layer_hidden=FakeTensor([0.0] * (count * 2), (count, 2), "float32"),
            final_logits=FakeTensor([float(index) for index in range(10)], (10,), "float32"),
            final_log_probs=FakeTensor([-float(index) for index in range(10)], (10,), "float32"),
            semantic=None,
        )

    def close(self) -> None:
        self.closed = True


class CalibrationFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.sidecars: dict[str, dict[str, FakeTensor]] = {}
        source_prompts = (
            Path(__file__).resolve().parents[4] / "benchmarks/prompts.jsonl"
        ).read_bytes()
        self.prompt_rows = [
            json.loads(line) for line in source_prompts.decode("utf-8").splitlines()
        ]
        self.source_prompts = source_prompts
        self._write_sources()
        self._initialize_git_history()
        self.contract_base = {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "config_sha256": MODEL_CONFIG_SHA256,
            "weights_sha256": MODEL_WEIGHTS_SHA256,
            "tokenizer_sha256": TOKENIZER_SHA256,
            "tokenizer_files_sha256": dict(TOKENIZER_FILES_SHA256),
            "attention_backend": "eager",
            "tensor_capture_cache_path": "off",
            "log_prob_pipeline": "log-softmax-fp32-v1",
            "trust_remote_code": False,
            "max_context_tokens": 8192,
            "eos_token_ids": [0],
            "semantic_generation_steps": 32,
            "cross_cache_exact_window": 16,
            "top_k": 10,
            "oracle_reduction_variant": dict(HF_ORACLE_REDUCTION_VARIANT),
        }

    def _initialize_git_history(self) -> None:
        subprocess.run(["git", "init", "-q"], cwd=self.root, check=True)
        subprocess.run(
            ["git", "config", "user.email", "calibration@example.invalid"],
            cwd=self.root,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Calibration Test"],
            cwd=self.root,
            check=True,
        )
        subprocess.run(["git", "add", "."], cwd=self.root, check=True)
        subprocess.run(
            ["git", "commit", "-q", "-m", "oracle sources"],
            cwd=self.root,
            check=True,
        )
        self.oracle_revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=self.root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        subprocess.run(
            ["git", "commit", "-q", "--allow-empty", "-m", "native candidate"],
            cwd=self.root,
            check=True,
        )
        self.candidate_revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=self.root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    def _write(self, relative: str, data: bytes) -> None:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def _write_sources(self) -> None:
        self._write("benchmarks/matrix.yaml", b"matrix-v1\n")
        self._write("benchmarks/prompts.jsonl", self.source_prompts)
        self._write(
            "benchmarks/correctness/smollm2-fp32-bf16-native-e0-v2.json",
            json.dumps(
                gate_contract_document(ORACLE_MANIFEST_GATE_ID), sort_keys=True
            ).encode("utf-8"),
        )
        self._write(
            "benchmarks/correctness/smollm2-fp32-bf16-native-e0-v3.json",
            json.dumps(gate_contract_document(), sort_keys=True).encode("utf-8"),
        )
        self._write("benchmarks/environment.md", b"environment-v1\n")
        self._write(
            "tools/python/reference/riley_reference/environment.py",
            b"environment-probe-v1\n",
        )
        self._write("tools/python/reference/uv.lock", b"python-lock-v1\n")
        self._write("tools/python/reference/.python-version", b"3.13.15\n")
        self._write("Cargo.lock", b"native-lock-v1\n")
        self._write(
            "benchmarks/lanes/hf-transformers.json",
            json.dumps(
                {
                    "lane_id": "hf-transformers",
                    "implementation_id": "hf-transformers-eager",
                    "runtime_dependency_class": "python-reference",
                    "engine": {"revision": f"transformers-{TRANSFORMERS_VERSION}+torch-{TORCH_VERSION}"},
                }
            ).encode("utf-8"),
        )
        self._write(
            "benchmarks/lanes/riley-native.json",
            json.dumps(
                {
                    "lane_id": "riley-native",
                    "implementation_id": "riley-native",
                    "runtime_dependency_class": "native-production",
                    "engine": {"revision": LEGACY_NATIVE_ENGINE_REVISION},
                }
            ).encode("utf-8"),
        )
        self._write(
            "benchmarks/lanes/riley-native-v3.json",
            json.dumps(
                {
                    "lane_id": "riley-native",
                    "implementation_id": "riley-native",
                    "runtime_dependency_class": "native-production",
                    "engine": {"revision": NATIVE_ENGINE_REVISION},
                }
            ).encode("utf-8"),
        )

    def _sources(
        self, kind: str, gate_id: str = ORACLE_MANIFEST_GATE_ID
    ) -> dict[str, object]:
        if kind != CANDIDATE_KIND:
            paths = HF_SOURCE_PATHS
        elif gate_id == ORACLE_MANIFEST_GATE_ID:
            paths = LEGACY_NATIVE_SOURCE_PATHS
        else:
            paths = NATIVE_SOURCE_PATHS
        return {
            name: {"path": relative, "sha256": sha256_file(self.root / relative)}
            for name, relative in paths.items()
        }

    @staticmethod
    def _semantic(category: str) -> dict[str, object]:
        if category == "early-eos":
            return {
                "top_1_token_id": 0,
                "top_k_token_id_set": list(range(10)),
                "cache_on": {"generated_token_ids": [0], "stop_reason": "eos"},
                "cache_off": {"generated_token_ids": [0], "stop_reason": "eos"},
                "cross_cache_first_divergence_step": None,
                "cross_cache_exact_window_match": True,
            }
        cache_on = [9, *range(1, 32)]
        cache_off = list(cache_on)
        cache_off[17] = 99
        return {
            "top_1_token_id": 9,
            "top_k_token_id_set": list(range(10)),
            "cache_on": {
                "generated_token_ids": cache_on,
                "stop_reason": "max_new_tokens",
            },
            "cache_off": {
                "generated_token_ids": cache_off,
                "stop_reason": "max_new_tokens",
            },
            "cross_cache_first_divergence_step": 17,
            "cross_cache_exact_window_match": True,
        }

    def make(
        self,
        kind: str,
        *,
        candidate_gate_id: str | None = None,
    ) -> tuple[dict[str, object], Path]:
        gate_id = (
            (candidate_gate_id or CALIBRATION_GATE_ID)
            if kind == CANDIDATE_KIND
            else ORACLE_MANIFEST_GATE_ID
        )
        required_candidate_variants = (
            ORACLE_REQUIRED_CANDIDATE_REDUCTION_VARIANTS
            if gate_id == ORACLE_MANIFEST_GATE_ID
            else REQUIRED_CANDIDATE_REDUCTION_VARIANTS
        )
        variants = (
            required_candidate_variants
            if kind == CANDIDATE_KIND
            else (HF_ORACLE_REDUCTION_VARIANT,)
        )
        sidecar_name = f"{kind}.safetensors"
        sidecar_path = self.root / sidecar_name
        manifest_path = self.root / f"{kind}.json"
        sidecar_path.write_bytes(kind.encode("ascii"))
        tensor_map: dict[str, FakeTensor] = {}
        cases: list[dict[str, object]] = []
        for row in self.prompt_rows:
            token_count = row["target_prompt_tokens"] or (
                1 if row["category"] in {"minimal", "early-eos"} else 4
            )
            first_token_id = 44239 if row["category"] == "early-eos" else 1
            case_variants: dict[str, object] = {}
            for config in variants:
                variant_id = str(config["variant_id"])
                prefix = f"cases/{row['prompt_id']}/{variant_id}"
                activation_dtype = "float32" if kind == FP32_ORACLE_KIND else "bfloat16"
                delta = 0.0 if kind == FP32_ORACLE_KIND else 0.001
                hidden_values = [
                    ((index % 11) - 5) / 5.0 + (delta if index == 0 else 0.0)
                    for index in range(token_count * 2)
                ]
                definitions = {
                    "first_layer_hidden": (
                        hidden_values,
                        (token_count, 2),
                        activation_dtype,
                    ),
                    "final_logits": (
                        (
                            [20.0 + delta, *[float(index) for index in range(1, 10)]]
                            if row["category"] == "early-eos"
                            else [
                                float(index) + (delta if index == 0 else 0.0)
                                for index in range(10)
                            ]
                        ),
                        (10,),
                        activation_dtype,
                    ),
                    "final_log_probs": (
                        [-float(10 - index) + delta for index in range(10)],
                        (10,),
                        "float32",
                    ),
                }
                refs: dict[str, object] = {}
                for tensor_name, (values, shape, dtype) in definitions.items():
                    key = f"{prefix}/{tensor_name}"
                    tensor_map[key] = FakeTensor(values, shape, dtype)
                    refs[tensor_name] = {
                        "key": key,
                        "shape": list(shape),
                        "dtype": dtype,
                        "cache_path": "off",
                    }
                case_variants[variant_id] = {
                    "config": dict(config),
                    "tensors": refs,
                    "semantic": (
                        None
                        if kind == FP32_ORACLE_KIND
                        else self._semantic(row["category"])
                    ),
                }
            token_bytes = b"".join(
                int(first_token_id if index == 0 else 1).to_bytes(4, "little")
                for index in range(token_count)
            )
            cases.append(
                {
                    "prompt_id": row["prompt_id"],
                    "prompt_text_sha256": hashlib.sha256(row["text"].encode("utf-8")).hexdigest(),
                    "prompt_metadata": {
                        "category": row["category"],
                        "language": row["language"],
                        "target_prompt_tokens": row["target_prompt_tokens"],
                        "boundary_kind": row["boundary_kind"],
                        "expected_behavior": row["expected_behavior"],
                    },
                    "input_token_ids_sha256": hashlib.sha256(token_bytes).hexdigest(),
                    "input_first_token_id": first_token_id,
                    "input_token_count": token_count,
                    "hidden_anchor_positions": {
                        "first": 0,
                        "middle": (token_count - 1) // 2,
                        "last": token_count - 1,
                    },
                    "variants": case_variants,
                }
            )
        self.sidecars[sidecar_name] = tensor_map
        producer = (
            {
                "implementation_id": "riley-native",
                "engine_revision": (
                    LEGACY_NATIVE_ENGINE_REVISION
                    if gate_id == ORACLE_MANIFEST_GATE_ID
                    else NATIVE_ENGINE_REVISION
                ),
                "runtime_dependency_class": "native-production",
                "python_version": None,
                "python_executable_sha256": None,
                "python_platform_system": None,
                "python_platform_machine": None,
                "torch_version": None,
                "transformers_version": None,
                "safetensors_version": None,
            }
            if kind == CANDIDATE_KIND
            else {
                "implementation_id": "hf-transformers-eager",
                "engine_revision": f"transformers-{TRANSFORMERS_VERSION}+torch-{TORCH_VERSION}",
                "runtime_dependency_class": "python-reference",
                "python_version": PYTHON_VERSION,
                "python_executable_sha256": PYTHON_EXECUTABLE_SHA256,
                "python_platform_system": PYTHON_PLATFORM_SYSTEM,
                "python_platform_machine": PYTHON_PLATFORM_MACHINE,
                "torch_version": TORCH_VERSION,
                "transformers_version": TRANSFORMERS_VERSION,
                "safetensors_version": SAFETENSORS_VERSION,
            }
        )
        manifest: dict[str, object] = {
            "schema_version": "1.0.0",
            "artifact_kind": kind,
            "created_at": "2026-08-24T02:03:04Z",
            "producer": producer,
            "candidate_execution": None,
            "contract": {
                **copy.deepcopy(self.contract_base),
                "gate_id": gate_id,
                "required_candidate_reduction_variants": [
                    dict(item) for item in required_candidate_variants
                ],
                "dtype": "float32" if kind == FP32_ORACLE_KIND else "bfloat16",
            },
            "provenance": {
                "sources": self._sources(kind, gate_id),
                "git_revision": (
                    self.candidate_revision
                    if kind == CANDIDATE_KIND
                    else self.oracle_revision
                ),
                "git_dirty": False,
                "git_status_sha256": EMPTY_SHA,
                "environment_id": "rtx4090-ubuntu22-driver580-v1",
                "observed_environment": copy.deepcopy(
                    PRIMARY_ENVIRONMENT_SNAPSHOT
                ),
            },
            "corpus": {"prompt_count": len(cases)},
            "sidecar": {
                "path": sidecar_name,
                "sha256": sha256_file(sidecar_path),
                "format": "safetensors",
                "tensor_count": len(tensor_map),
            },
            "cases": cases,
        }
        if kind == CANDIDATE_KIND:
            executable_path = self.root / NATIVE_EXECUTABLE_FILENAME
            executable_path.write_bytes(b"fake-riley-native\n")
            manifest["candidate_execution"] = {
                "executable": {
                    "path": NATIVE_EXECUTABLE_FILENAME,
                    "sha256": sha256_file(executable_path),
                },
                "build_argv": list(NATIVE_BUILD_ARGV),
                "capture_argv": [
                    NATIVE_EXECUTABLE_FILENAME,
                    "calibrate",
                    "--repository-root",
                    "/workspace/riley",
                    "--model",
                    "/models/smollm2",
                    "--gate-manifest",
                    (
                        LEGACY_NATIVE_SOURCE_PATHS
                        if gate_id == ORACLE_MANIFEST_GATE_ID
                        else NATIVE_SOURCE_PATHS
                    )["gate_manifest"],
                    "--prompts",
                    NATIVE_SOURCE_PATHS["prompts"],
                    "--manifest",
                    manifest_path.name,
                    "--sidecar",
                    sidecar_name,
                ],
            }
            for variant in required_candidate_variants:
                manifest["candidate_execution"]["capture_argv"].extend(
                    ["--reduction-variant", variant["variant_id"]]
                )
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return manifest, manifest_path

    def loader(self, path: Path):
        return self.sidecars[path.name]

    def make_oracle_report(
        self,
        fp32: dict[str, object],
        fp32_path: Path,
        bf16: dict[str, object],
        bf16_path: Path,
    ) -> tuple[dict[str, object], Path]:
        report = compare_hf_oracles(
            fp32_manifest=fp32,
            fp32_manifest_path=fp32_path,
            bf16_manifest=bf16,
            bf16_manifest_path=bf16_path,
            repo_root=self.root,
            created_at=FIXED_TIME,
            sidecar_loader=self.loader,
        )
        path = self.root / "oracle-calibration-report.json"
        path.write_text(json.dumps(report, sort_keys=True) + "\n", encoding="utf-8")
        return report, path


class CalibrationTests(unittest.TestCase):
    def test_oracle_pair_requires_identical_power_and_application_clocks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = CalibrationFixture(Path(directory))
            fp32, fp32_path = fixture.make(FP32_ORACLE_KIND)
            bf16, bf16_path = fixture.make(BF16_ORACLE_KIND)
            changed = copy.deepcopy(bf16)
            changed["provenance"]["observed_environment"]["accelerator"][
                "power_limit_w"
            ] = 451.0
            with self.assertRaisesRegex(CalibrationError, "observed environment"):
                compare_hf_oracles(
                    fp32_manifest=fp32,
                    fp32_manifest_path=fp32_path,
                    bf16_manifest=changed,
                    bf16_manifest_path=bf16_path,
                    repo_root=fixture.root,
                    created_at=FIXED_TIME,
                    sidecar_loader=fixture.loader,
                )

    def test_predeclared_thresholds_are_inclusive_at_boundaries(self) -> None:
        for tensor_name, threshold in CALIBRATION_THRESHOLDS.items():
            metrics = {
                "max_abs": threshold["max_abs_max"],
                "mean_abs": threshold["mean_abs_max"],
                "max_relative": threshold["max_relative_max"],
                "mean_relative": threshold["mean_relative_max"],
                "cosine_similarity": threshold["cosine_min"],
            }
            self.assertTrue(metrics_pass(metrics, threshold), tensor_name)
            changed = dict(metrics)
            changed["mean_abs"] += 1e-12
            self.assertFalse(metrics_pass(changed, threshold), tensor_name)

    def test_v2_thresholds_are_exact_uniform_15_percent_headroom(self) -> None:
        activation = gate_contract_document()["numeric"]["threshold_activation"]
        observed = activation["calibration_evidence"][
            "observed_aggregate_metrics"
        ]
        scale = Decimal("1.15")
        upper_metrics = {
            "max_abs": "max_abs_max",
            "mean_abs": "mean_abs_max",
            "max_relative": "max_relative_max",
            "mean_relative": "mean_relative_max",
        }
        for tensor_name, recorded in observed.items():
            threshold = CALIBRATION_THRESHOLDS[tensor_name]
            for metric_name, threshold_name in upper_metrics.items():
                expected = float(
                    Decimal(str(recorded[metric_name])) * scale
                )
                self.assertEqual(threshold[threshold_name], expected)
            cosine = Decimal(str(recorded["cosine_similarity"]))
            expected_cosine = float(Decimal(1) - (Decimal(1) - cosine) * scale)
            self.assertEqual(threshold["cosine_min"], expected_cosine)

    def test_v1_manifest_and_report_cannot_activate_v2_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = CalibrationFixture(Path(directory))
            fp32, fp32_path = fixture.make(FP32_ORACLE_KIND)
            bf16, bf16_path = fixture.make(BF16_ORACLE_KIND)

            v1_manifest = copy.deepcopy(fp32)
            v1_manifest["contract"]["gate_id"] = (
                "smollm2-fp32-bf16-native-e0-v1"
            )
            with self.assertRaisesRegex(CalibrationError, "contract.gate_id"):
                validate_calibration_manifest(v1_manifest)

            report = compare_hf_oracles(
                fp32_manifest=fp32,
                fp32_manifest_path=fp32_path,
                bf16_manifest=bf16,
                bf16_manifest_path=bf16_path,
                repo_root=fixture.root,
                created_at=FIXED_TIME,
                sidecar_loader=fixture.loader,
            )
            report["gate_id"] = "smollm2-hf-fp32-bf16-calibration-v1"
            with self.assertRaisesRegex(CalibrationError, "oracle_report.gate_id"):
                replay_validate_oracle_report(
                    report=report,
                    fp32_manifest=fp32,
                    fp32_manifest_path=fp32_path,
                    bf16_manifest=bf16,
                    bf16_manifest_path=bf16_path,
                    repo_root=fixture.root,
                    sidecar_loader=fixture.loader,
                )

    def test_sequence_and_tensor_metric_loaders_are_exactly_equal(self) -> None:
        reference = [0.0, 0.5, -2.0, 4.0, -0.25, 16_777_217.0]
        candidate = [0.1, 0.4, -2.5, 3.0, -0.5, 16_777_216.0]
        expected = recompute_numeric_metrics(reference, candidate)
        actual = recompute_numeric_metrics_from_tensors(
            MiniVector(reference), MiniVector(candidate)
        )
        self.assertEqual(actual, expected)

    def test_metrics_round_inputs_and_element_errors_to_f32(self) -> None:
        metrics = recompute_numeric_metrics([1.0], [-(2.0**-24)])
        self.assertEqual(
            metrics,
            {
                "max_abs": 1.0,
                "mean_abs": 1.0,
                "max_relative": 1.0,
                "mean_relative": 1.0,
                "cosine_similarity": -1.0,
            },
        )
        input_rounded = recompute_numeric_metrics(
            [16_777_216.0], [16_777_217.0]
        )
        self.assertEqual(input_rounded["max_abs"], 0.0)
        relative_rounded = recompute_numeric_metrics([3.0], [4.0])
        self.assertEqual(relative_rounded["max_relative"], 0.3333333432674408)
        f32_division = recompute_numeric_metrics([1.0000001192092896], [1.0])
        self.assertEqual(
            f32_division["max_relative"], 1.1920927533992653e-07
        )
        self.assertNotEqual(
            f32_division["max_relative"], 1.1920927533992823e-07
        )

    def test_fixed_metric_chunk_boundary_includes_short_final_chunk(self) -> None:
        count = 262_145
        reference = [1.0] * count
        candidate = list(reference)
        candidate[262_143] = 2.0
        candidate[262_144] = 3.0
        expected = recompute_numeric_metrics(reference, candidate)
        actual = recompute_numeric_metrics_from_tensors(
            MiniVector(reference), MiniVector(candidate)
        )
        self.assertEqual(actual, expected)
        self.assertEqual(expected["max_abs"], 2.0)
        self.assertEqual(expected["mean_abs"], 3.0 / count)
        self.assertEqual(expected["max_relative"], 2.0)
        self.assertEqual(expected["mean_relative"], 3.0 / count)

    def test_metrics_reject_nonfinite_values_and_f32_overflow(self) -> None:
        maximum_f32 = 3.4028234663852886e38
        invalid_pairs = (
            ([math.nan], [0.0], "non-finite"),
            ([math.inf], [0.0], "non-finite"),
            ([3.5e38], [0.0], "finite float32 range"),
            ([maximum_f32], [-maximum_f32], "finite float32 range"),
        )
        for reference, candidate, message in invalid_pairs:
            with self.subTest(
                reference=reference, candidate=candidate, loader="sequence"
            ):
                with self.assertRaisesRegex(CalibrationError, message):
                    recompute_numeric_metrics(reference, candidate)
            with self.subTest(
                reference=reference, candidate=candidate, loader="tensor"
            ):
                with self.assertRaisesRegex(CalibrationError, message):
                    recompute_numeric_metrics_from_tensors(
                        MiniVector(reference), MiniVector(candidate)
                    )

    def test_manifest_rejects_self_consistent_but_noncanonical_tokenizer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = CalibrationFixture(Path(directory))
            fp32, _ = fixture.make(FP32_ORACLE_KIND)
            changed = copy.deepcopy(fp32)
            changed_files = dict(changed["contract"]["tokenizer_files_sha256"])
            changed_files["vocab.json"] = "f" * 64
            changed["contract"]["tokenizer_files_sha256"] = changed_files
            changed["contract"]["tokenizer_sha256"] = aggregate_tokenizer_sha256(
                changed_files
            )
            with self.assertRaisesRegex(CalibrationError, "tokenizer_sha256"):
                validate_calibration_manifest(changed)

    def test_manifest_rejects_non_exact_python_runtime_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = CalibrationFixture(Path(directory))
            fp32, _ = fixture.make(FP32_ORACLE_KIND)
            changed = copy.deepcopy(fp32)
            changed["producer"]["python_version"] = "3.13.14"
            with self.assertRaisesRegex(CalibrationError, "runtime contract"):
                validate_calibration_manifest(changed)

    def test_candidate_execution_provenance_is_required_and_replayed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = CalibrationFixture(Path(directory))
            candidate, candidate_path = fixture.make(CANDIDATE_KIND)
            changed = copy.deepcopy(candidate)
            changed["candidate_execution"]["build_argv"] = ["cargo", "build"]
            with self.assertRaisesRegex(CalibrationError, "build_argv"):
                validate_calibration_manifest(changed)
            changed = copy.deepcopy(candidate)
            changed["candidate_execution"]["capture_argv"].extend(
                ["--unreviewed-flag", "unreviewed-value"]
            )
            with self.assertRaisesRegex(CalibrationError, "exact ordered"):
                validate_calibration_manifest(changed)
            for index, invalid in (
                (5, "models/smollm2"),
                (11, "candidate\\manifest.json"),
            ):
                with self.subTest(invalid_capture_path=invalid):
                    changed = copy.deepcopy(candidate)
                    changed["candidate_execution"]["capture_argv"][index] = invalid
                    with self.assertRaises(CalibrationError):
                        validate_calibration_manifest(changed)
            (fixture.root / NATIVE_EXECUTABLE_FILENAME).write_bytes(b"tampered\n")
            with self.assertRaisesRegex(CalibrationError, "executable SHA-256"):
                verify_calibration_artifact(
                    manifest=candidate,
                    manifest_path=candidate_path,
                    repo_root=fixture.root,
                    sidecar_loader=fixture.loader,
                )

    def test_native_contract_build_and_versioned_engines_are_exact(self) -> None:
        self.assertEqual(NATIVE_ENGINE_REVISION, "riley-native-contract-v3")
        self.assertEqual(
            LEGACY_NATIVE_ENGINE_REVISION, "riley-native-contract-v2"
        )
        self.assertEqual(
            NATIVE_BUILD_ARGV,
            (
                "cargo",
                "build",
                "--locked",
                "--release",
                "--package",
                "riley-native",
                "--no-default-features",
                "--features",
                "cuda",
                "--bin",
                "riley-native",
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            fixture = CalibrationFixture(Path(directory))
            candidate, _ = fixture.make(CANDIDATE_KIND)
            changed = copy.deepcopy(candidate)
            changed["producer"]["engine_revision"] = "riley-native-contract-v1"
            with self.assertRaisesRegex(CalibrationError, "native-production"):
                validate_calibration_manifest(changed)

    def test_offline_fp32_producer_writes_full_corpus_and_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            repo = base / "repo"
            repo.mkdir()
            fixture = CalibrationFixture(repo)
            output = base / "output"
            output.mkdir()
            manifest_path = output / "fp32-manifest.json"
            sidecar_path = output / "fp32.safetensors"
            backend = FakeOracleBackend(FP32_ORACLE_KIND)
            provenance = {
                "sources": fixture._sources(FP32_ORACLE_KIND),
                "git_revision": fixture.oracle_revision,
                "git_dirty": False,
                "git_status_sha256": EMPTY_SHA,
                "environment_id": "rtx4090-ubuntu22-driver580-v1",
                "observed_environment": copy.deepcopy(
                    PRIMARY_ENVIRONMENT_SNAPSHOT
                ),
            }

            def write_fake_sidecar(_tensors, path: Path) -> None:
                path.write_bytes(b"fake-safetensors")

            with mock.patch(
                "riley_reference.hf_calibration.repository_provenance",
                return_value=provenance,
            ):
                manifest = produce_hf_oracle(
                    artifact_kind=FP32_ORACLE_KIND,
                    prompts_path=repo / HF_SOURCE_PATHS["prompts"],
                    manifest_path=manifest_path,
                    sidecar_path=sidecar_path,
                    repo_root=repo,
                    device="fake",
                    local_files_only=True,
                    created_at=FIXED_TIME,
                    backend_factory=lambda **_kwargs: backend,
                    sidecar_writer=write_fake_sidecar,
                    environment_probe=lambda: copy.deepcopy(
                        PRIMARY_ENVIRONMENT_SNAPSHOT
                    ),
                )
                self.assertEqual(len(manifest["cases"]), 31)
                self.assertIsNone(manifest["candidate_execution"])
                self.assertTrue(backend.closed)
                with self.assertRaisesRegex(CalibrationError, "overwrite"):
                    produce_hf_oracle(
                        artifact_kind=FP32_ORACLE_KIND,
                        prompts_path=repo / HF_SOURCE_PATHS["prompts"],
                        manifest_path=manifest_path,
                        sidecar_path=sidecar_path,
                        repo_root=repo,
                        device="fake",
                        local_files_only=True,
                        created_at=FIXED_TIME,
                        backend_factory=lambda **_kwargs: backend,
                        sidecar_writer=write_fake_sidecar,
                        environment_probe=lambda: copy.deepcopy(
                            PRIMARY_ENVIRONMENT_SNAPSHOT
                        ),
                    )

    def test_language_neutral_schemas_accept_recomputed_fake_bundle(self) -> None:
        from benchmarks.scripts.validate_contract import ContractError, validate_instance

        schema_root = Path(__file__).resolve().parents[4] / "benchmarks/schemas"
        schemas = {
            name: json.loads((schema_root / name).read_text(encoding="utf-8"))
            for name in (
                "correctness-gate.schema.json",
                "correctness-gate-v3.schema.json",
                "correctness-calibration-manifest.schema.json",
                "correctness-calibration-manifest-v3.schema.json",
                "oracle-calibration-report.schema.json",
                "correctness-report-v3.schema.json",
            )
        }
        validate_instance(
            gate_contract_document(ORACLE_MANIFEST_GATE_ID),
            schemas["correctness-gate.schema.json"],
        )
        validate_instance(
            gate_contract_document(), schemas["correctness-gate-v3.schema.json"]
        )
        with tempfile.TemporaryDirectory() as directory:
            fixture = CalibrationFixture(Path(directory))
            fp32, fp32_path = fixture.make(FP32_ORACLE_KIND)
            bf16, bf16_path = fixture.make(BF16_ORACLE_KIND)
            candidate, candidate_path = fixture.make(CANDIDATE_KIND)
            for manifest in (fp32, bf16):
                validate_instance(
                    manifest,
                    schemas["correctness-calibration-manifest.schema.json"],
                )
            validate_instance(
                candidate,
                schemas["correctness-calibration-manifest-v3.schema.json"],
            )
            with self.assertRaises(ContractError):
                validate_instance(
                    fp32,
                    schemas["correctness-calibration-manifest-v3.schema.json"],
                )
            wrong_count = copy.deepcopy(candidate)
            wrong_count["sidecar"]["tensor_count"] = 94
            with self.assertRaises(ContractError):
                validate_instance(
                    wrong_count,
                    schemas["correctness-calibration-manifest-v3.schema.json"],
                )
            oracle_report, oracle_path = fixture.make_oracle_report(
                fp32, fp32_path, bf16, bf16_path
            )
            validate_instance(
                oracle_report, schemas["oracle-calibration-report.schema.json"]
            )
            report = compare_calibrations(
                fp32_manifest=fp32,
                fp32_manifest_path=fp32_path,
                bf16_manifest=bf16,
                bf16_manifest_path=bf16_path,
                oracle_calibration_report=oracle_report,
                oracle_calibration_report_path=oracle_path,
                candidate_manifest=candidate,
                candidate_manifest_path=candidate_path,
                repo_root=fixture.root,
                created_at=FIXED_TIME,
                sidecar_loader=fixture.loader,
            )
            validate_instance(report, schemas["correctness-report-v3.schema.json"])

    def test_v3_manifest_requires_full_hidden_and_canonical_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = CalibrationFixture(Path(directory))
            candidate, _ = fixture.make(CANDIDATE_KIND)
            validate_calibration_manifest(candidate)
            sampled = copy.deepcopy(candidate)
            for variant in sampled["cases"][0]["variants"].values():
                variant["tensors"]["first_layer_hidden"]["shape"] = [3, 2]
            with self.assertRaisesRegex(CalibrationError, "every valid token"):
                validate_calibration_manifest(sampled)
            missing = copy.deepcopy(candidate)
            del missing["cases"][0]["variants"][
                CANONICAL_CANDIDATE_REDUCTION_VARIANT["variant_id"]
            ]
            with self.assertRaisesRegex(CalibrationError, "required reduction"):
                validate_calibration_manifest(missing)
            extra = copy.deepcopy(candidate)
            extra["contract"]["required_candidate_reduction_variants"].append(
                dict(ALTERNATE_CANDIDATE_REDUCTION_VARIANT)
            )
            with self.assertRaisesRegex(CalibrationError, "execution contract"):
                validate_calibration_manifest(extra)
            wrong_count = copy.deepcopy(candidate)
            wrong_count["sidecar"]["tensor_count"] = 31 * 2 * 3
            with self.assertRaisesRegex(CalibrationError, "tensor_count"):
                validate_calibration_manifest(wrong_count)
            extra_source = copy.deepcopy(candidate)
            extra_source["provenance"]["sources"]["matrix"] = {
                "path": HF_SOURCE_PATHS["matrix"],
                "sha256": sha256_file(
                    fixture.root / HF_SOURCE_PATHS["matrix"]
                ),
            }
            with self.assertRaisesRegex(CalibrationError, "source set"):
                validate_calibration_manifest(extra_source)

    def test_oracle_and_candidate_gate_roles_cannot_be_swapped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = CalibrationFixture(Path(directory))
            fp32, _ = fixture.make(FP32_ORACLE_KIND)
            candidate, _ = fixture.make(CANDIDATE_KIND)
            changed_oracle = copy.deepcopy(fp32)
            changed_oracle["contract"]["gate_id"] = CALIBRATION_GATE_ID
            changed_oracle["contract"]["required_candidate_reduction_variants"] = [
                dict(CANONICAL_CANDIDATE_REDUCTION_VARIANT)
            ]
            with self.assertRaisesRegex(CalibrationError, "frozen v2 gate"):
                validate_calibration_manifest(changed_oracle)
            changed_candidate = copy.deepcopy(candidate)
            changed_candidate["contract"]["gate_id"] = ORACLE_MANIFEST_GATE_ID
            with self.assertRaisesRegex(CalibrationError, "native-production|flag inventory"):
                validate_calibration_manifest(changed_candidate)

    def test_comparator_accepts_v2_oracles_and_v3_canonical_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = CalibrationFixture(Path(directory))
            fp32, fp32_path = fixture.make(FP32_ORACLE_KIND)
            bf16, bf16_path = fixture.make(BF16_ORACLE_KIND)
            oracle_report, oracle_report_path = fixture.make_oracle_report(
                fp32, fp32_path, bf16, bf16_path
            )
            candidate, candidate_path = fixture.make(CANDIDATE_KIND)
            report = compare_calibrations(
                fp32_manifest=fp32,
                fp32_manifest_path=fp32_path,
                bf16_manifest=bf16,
                bf16_manifest_path=bf16_path,
                oracle_calibration_report=oracle_report,
                oracle_calibration_report_path=oracle_report_path,
                candidate_manifest=candidate,
                candidate_manifest_path=candidate_path,
                repo_root=fixture.root,
                created_at=FIXED_TIME,
                sidecar_loader=fixture.loader,
            )
            self.assertEqual(report["status"], "pass")
            self.assertNotEqual(
                report["bindings"]["oracle_git_revision"],
                report["bindings"]["candidate_git_revision"],
            )
            self.assertEqual(
                set(report["summary"]["variants"]),
                {CANONICAL_CANDIDATE_REDUCTION_VARIANT["variant_id"]},
            )
            self.assertEqual(report["gate_id"], CALIBRATION_GATE_ID)
            self.assertEqual(
                report["bindings"]["oracle_manifest_gate_id"],
                ORACLE_MANIFEST_GATE_ID,
            )
            self.assertIn("oracle_matrix_sha256", report["bindings"])
            self.assertIn("oracle_gate_manifest_sha256", report["bindings"])
            self.assertIn("candidate_gate_manifest_sha256", report["bindings"])
            self.assertNotIn("matrix_sha256", report["bindings"])
            self.assertNotIn("report_sha256", report)

    def test_historical_v2_two_variant_candidate_report_still_replays(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = CalibrationFixture(Path(directory))
            fp32, fp32_path = fixture.make(FP32_ORACLE_KIND)
            bf16, bf16_path = fixture.make(BF16_ORACLE_KIND)
            oracle_report, oracle_report_path = fixture.make_oracle_report(
                fp32, fp32_path, bf16, bf16_path
            )
            candidate, candidate_path = fixture.make(
                CANDIDATE_KIND, candidate_gate_id=ORACLE_MANIFEST_GATE_ID
            )
            report = compare_calibrations(
                fp32_manifest=fp32,
                fp32_manifest_path=fp32_path,
                bf16_manifest=bf16,
                bf16_manifest_path=bf16_path,
                oracle_calibration_report=oracle_report,
                oracle_calibration_report_path=oracle_report_path,
                candidate_manifest=candidate,
                candidate_manifest_path=candidate_path,
                repo_root=fixture.root,
                created_at=FIXED_TIME,
                sidecar_loader=fixture.loader,
            )
            self.assertEqual(report["gate_id"], ORACLE_MANIFEST_GATE_ID)
            self.assertEqual(
                set(report["summary"]["variants"]),
                {
                    CANONICAL_CANDIDATE_REDUCTION_VARIANT["variant_id"],
                    ALTERNATE_CANDIDATE_REDUCTION_VARIANT["variant_id"],
                },
            )
            self.assertIn("matrix_sha256", report["bindings"])
            self.assertIn("gate_manifest_sha256", report["bindings"])
            replay_validate_correctness_report(
                report=report,
                fp32_manifest=fp32,
                fp32_manifest_path=fp32_path,
                bf16_manifest=bf16,
                bf16_manifest_path=bf16_path,
                oracle_calibration_report=oracle_report,
                oracle_calibration_report_path=oracle_report_path,
                candidate_manifest=candidate,
                candidate_manifest_path=candidate_path,
                repo_root=fixture.root,
                sidecar_loader=fixture.loader,
            )

    def test_top_k_metadata_is_recomputed_from_raw_logits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = CalibrationFixture(Path(directory))
            candidate, candidate_path = fixture.make(CANDIDATE_KIND)
            candidate["cases"][0]["variants"][
                CANONICAL_CANDIDATE_REDUCTION_VARIANT["variant_id"]
            ]["semantic"]["top_1_token_id"] = 8
            with self.assertRaisesRegex(CalibrationError, "top-k metadata"):
                verify_calibration_artifact(
                    manifest=candidate,
                    manifest_path=candidate_path,
                    repo_root=fixture.root,
                    sidecar_loader=fixture.loader,
                )

    def test_oracle_report_is_explicitly_not_e0_candidate_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = CalibrationFixture(Path(directory))
            fp32, fp32_path = fixture.make(FP32_ORACLE_KIND)
            bf16, bf16_path = fixture.make(BF16_ORACLE_KIND)
            report = compare_hf_oracles(
                fp32_manifest=fp32,
                fp32_manifest_path=fp32_path,
                bf16_manifest=bf16,
                bf16_manifest_path=bf16_path,
                repo_root=fixture.root,
                created_at=FIXED_TIME,
                sidecar_loader=fixture.loader,
            )
            self.assertEqual(report["status"], "pass")
            self.assertIs(report["e0_candidate_evidence"], False)
            replay_validate_oracle_report(
                report=report,
                fp32_manifest=fp32,
                fp32_manifest_path=fp32_path,
                bf16_manifest=bf16,
                bf16_manifest_path=bf16_path,
                repo_root=fixture.root,
                sidecar_loader=fixture.loader,
            )

    def test_forged_coherent_report_is_rejected_by_sidecar_replay(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = CalibrationFixture(Path(directory))
            fp32, fp32_path = fixture.make(FP32_ORACLE_KIND)
            bf16, bf16_path = fixture.make(BF16_ORACLE_KIND)
            oracle_report, oracle_report_path = fixture.make_oracle_report(
                fp32, fp32_path, bf16, bf16_path
            )
            candidate, candidate_path = fixture.make(CANDIDATE_KIND)
            report = compare_calibrations(
                fp32_manifest=fp32,
                fp32_manifest_path=fp32_path,
                bf16_manifest=bf16,
                bf16_manifest_path=bf16_path,
                oracle_calibration_report=oracle_report,
                oracle_calibration_report_path=oracle_report_path,
                candidate_manifest=candidate,
                candidate_manifest_path=candidate_path,
                repo_root=fixture.root,
                created_at=FIXED_TIME,
                sidecar_loader=fixture.loader,
            )
            forged = copy.deepcopy(report)
            variant_id = CANONICAL_CANDIDATE_REDUCTION_VARIANT["variant_id"]
            forged_metrics = forged["cases"][0]["variants"][variant_id]["numeric"][
                "first_layer_hidden"
            ]["metrics"]
            forged_metrics["max_abs"] = 0.2
            forged["summary"]["variants"][variant_id]["aggregate_numeric"][
                "first_layer_hidden"
            ]["metrics"]["max_abs"] = 0.2
            with self.assertRaisesRegex(CalibrationError, "comparator replay"):
                replay_validate_correctness_report(
                    report=forged,
                    fp32_manifest=fp32,
                    fp32_manifest_path=fp32_path,
                    bf16_manifest=bf16,
                    bf16_manifest_path=bf16_path,
                    oracle_calibration_report=oracle_report,
                    oracle_calibration_report_path=oracle_report_path,
                    candidate_manifest=candidate,
                    candidate_manifest_path=candidate_path,
                    repo_root=fixture.root,
                    sidecar_loader=fixture.loader,
                )

    def test_cli_compare_and_replay_use_raw_file_hash_not_self_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = CalibrationFixture(Path(directory))
            fp32, fp32_path = fixture.make(FP32_ORACLE_KIND)
            bf16, bf16_path = fixture.make(BF16_ORACLE_KIND)
            _, oracle_report_path = fixture.make_oracle_report(
                fp32, fp32_path, bf16, bf16_path
            )
            _, candidate_path = fixture.make(CANDIDATE_KIND)
            output = fixture.root / "correctness-report.json"
            arguments = [
                "calibrate-compare",
                "--fp32-manifest",
                str(fp32_path),
                "--bf16-manifest",
                str(bf16_path),
                "--candidate-manifest",
                str(candidate_path),
                "--oracle-report",
                str(oracle_report_path),
                "--output",
                str(output),
                "--repo-root",
                str(fixture.root),
            ]
            stdout = io.StringIO()
            with mock.patch(
                "riley_reference.calibration._default_sidecar_loader",
                fixture.loader,
            ), contextlib.redirect_stdout(stdout):
                self.assertEqual(main(arguments, now=lambda: FIXED_TIME), 0)
            report = load_correctness_report(output)
            self.assertNotIn("report_sha256", report)
            self.assertIn(sha256_file(output), stdout.getvalue())
            validate_arguments = [
                "calibrate-validate-report",
                str(output),
                "--fp32-manifest",
                str(fp32_path),
                "--bf16-manifest",
                str(bf16_path),
                "--candidate-manifest",
                str(candidate_path),
                "--oracle-report",
                str(oracle_report_path),
                "--repo-root",
                str(fixture.root),
            ]
            with mock.patch(
                "riley_reference.calibration._default_sidecar_loader",
                fixture.loader,
            ), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(validate_arguments), 0)
            variant_id = CANONICAL_CANDIDATE_REDUCTION_VARIANT["variant_id"]
            report["cases"][0]["variants"][variant_id]["numeric"][
                "first_layer_hidden"
            ]["metrics"]["max_abs"] = 0.2
            report["summary"]["variants"][variant_id]["aggregate_numeric"][
                "first_layer_hidden"
            ]["metrics"]["max_abs"] = 0.2
            output.write_text(json.dumps(report, sort_keys=True) + "\n", encoding="utf-8")
            with mock.patch(
                "riley_reference.calibration._default_sidecar_loader",
                fixture.loader,
            ), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(validate_arguments), 2)

    def test_cli_contract_exposes_separate_oracle_and_candidate_flows(self) -> None:
        commands = _build_parser()._subparsers._group_actions[0].choices
        for command in (
            "calibrate-produce",
            "calibrate-validate-manifest",
            "calibrate-oracles",
            "calibrate-compare",
            "calibrate-validate-oracles",
            "calibrate-validate-report",
        ):
            self.assertIn(command, commands)
        fake_manifest = {
            "artifact_kind": FP32_ORACLE_KIND,
            "cases": [{} for _ in range(31)],
        }
        with tempfile.TemporaryDirectory() as directory, mock.patch(
            "riley_reference.hf_calibration.produce_hf_oracle",
            return_value=fake_manifest,
        ) as producer, contextlib.redirect_stdout(io.StringIO()):
            base = Path(directory)
            result = main(
                [
                    "calibrate-produce",
                    "--role",
                    "fp32",
                    "--prompts",
                    str(base / "prompts.jsonl"),
                    "--manifest",
                    str(base / "fp32.json"),
                    "--sidecar",
                    str(base / "fp32.safetensors"),
                    "--repo-root",
                    str(base),
                ],
                now=lambda: FIXED_TIME,
            )
        self.assertEqual(result, 0)
        self.assertTrue(producer.call_args.kwargs["local_files_only"])


if __name__ == "__main__":
    unittest.main()
