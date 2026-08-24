from __future__ import annotations

import hashlib
import importlib.util
import json
import struct
import sys
import tempfile
import unittest
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from rustinfer_vllm_benchmark.adapter import (  # noqa: E402
    MODEL_REVISION,
    MODEL_WEIGHTS_SHA256,
    PRIMARY_COMPUTE_CAPABILITY,
    PRIMARY_DRIVER_VERSION,
    PRIMARY_GPU_NAME,
    PRIMARY_RAM_BYTES,
    VLLM_VERSION,
    AdapterError,
    BackendMetadata,
    NvmlProcessTreeSampler,
    ObservabilityMeasurement,
    VllmBackend,
    _llm_options,
    _pin_vllm_environment,
    _engine_timing_durations,
    _validate_engine_timing_sanity,
    prompt_token_ids_sha256,
    run_benchmark,
    verify_snapshot_artifacts,
)


FIXED_TIME = datetime(2026, 8, 24, 12, 34, 56, tzinfo=timezone.utc)


class StepTimer:
    def __init__(self, step: float = 0.05, start: float = 0.0) -> None:
        self.value = start
        self.step = step

    def __call__(self) -> float:
        result = self.value
        self.value += self.step
        return result


class FakeTokenizer:
    bos_token_id = 7

    def __init__(self) -> None:
        self.calls: list[tuple[str, bool]] = []

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        self.calls.append((text, add_special_tokens))
        if not text:
            return []
        return [self.bos_token_id, 11, 13]


class FakeTokensPrompt(dict):
    def __init__(self, *, prompt_token_ids: list[int]) -> None:
        super().__init__(prompt_token_ids=prompt_token_ids)


class FakeSamplingParams:
    created: list[dict[str, object]] = []

    def __init__(self, **kwargs: object) -> None:
        self.values = dict(kwargs)
        type(self).created.append(self.values)


class FakeRequestOutputKind(Enum):
    CUMULATIVE = 0
    DELTA = 1
    FINAL_ONLY = 2


@dataclass
class FakeMetrics:
    first_token_latency: float | None = None
    arrival_time: float = 10.0
    scheduled_ts: float = 20.0
    first_token_ts: float = 20.002
    last_token_ts: float = 20.033


@dataclass
class FakeCompletion:
    token_ids: tuple[int, ...]


@dataclass
class FakeRequestOutput:
    request_id: str
    prompt_token_ids: list[int]
    outputs: list[FakeCompletion]
    metrics: FakeMetrics | None
    finished: bool


class FakeEngineCore:
    def __init__(self) -> None:
        self.shutdown_calls = 0

    def shutdown(self) -> None:
        self.shutdown_calls += 1


class FakeLLMEngine:
    def __init__(self, *, expose_metrics: bool = True) -> None:
        self.expose_metrics = expose_metrics
        self.engine_core = FakeEngineCore()
        self.active: dict[str, dict[str, object]] = {}
        self.calls: list[list[dict[str, object]]] = []
        self.abort_calls: list[tuple[list[str], bool]] = []
        self.emit_multiple_tokens = False

    def add_request(
        self,
        request_id: str,
        prompt: dict,
        params: FakeSamplingParams,
        *,
        arrival_time: float,
    ) -> str:
        if not self.active:
            self.calls.append([])
        index = len(self.calls[-1])
        call = {
            "request_id": request_id,
            "prompt": prompt,
            "params": params,
            "arrival_time": arrival_time,
        }
        self.calls[-1].append(call)
        self.active[request_id] = {
            **call,
            "index": index,
            "step": 0,
            "generated": 0,
        }
        return request_id

    def has_unfinished_requests(self) -> bool:
        return bool(self.active)

    def step(self) -> list[FakeRequestOutput]:
        outputs: list[FakeRequestOutput] = []
        for request_id, state in list(self.active.items()):
            state["step"] = int(state["step"]) + 1
            # The second logical request starts two scheduler steps later, so
            # concurrent requests have observably different completion times.
            if int(state["step"]) <= 2 * int(state["index"]):
                continue
            token_index = int(state["generated"])
            params = state["params"]
            assert isinstance(params, FakeSamplingParams)
            max_tokens = int(params.values["max_tokens"])
            state["generated"] = token_index + 1
            finished = int(state["generated"]) == max_tokens
            metrics = None
            if self.expose_metrics:
                metrics = FakeMetrics(
                    first_token_latency=0.002 + int(state["index"]) * 0.001,
                    first_token_ts=20.002 + int(state["index"]) * 0.001,
                    last_token_ts=(
                        20.002
                        + int(state["index"]) * 0.001
                        + token_index * 0.001
                    ),
                )
            prompt = state["prompt"]
            assert isinstance(prompt, dict)
            outputs.append(
                FakeRequestOutput(
                    request_id=request_id,
                    prompt_token_ids=list(prompt["prompt_token_ids"]),
                    outputs=[
                        FakeCompletion(
                            (token_index, token_index + 1000)
                            if self.emit_multiple_tokens
                            else (token_index,)
                        )
                    ],
                    metrics=metrics,
                    finished=finished,
                )
            )
            if finished:
                del self.active[request_id]
        return outputs

    def abort_request(self, request_ids: list[str], internal: bool = False) -> None:
        self.abort_calls.append((list(request_ids), internal))
        for request_id in request_ids:
            self.active.pop(request_id, None)


class FakeLLM:
    def __init__(self, *, expose_metrics: bool = True) -> None:
        self.llm_engine = FakeLLMEngine(expose_metrics=expose_metrics)

    @property
    def calls(self) -> list[list[dict[str, object]]]:
        return self.llm_engine.calls


class FakeObservabilitySampler:
    def __init__(self) -> None:
        self.start_calls = 0
        self.stop_calls: list[float] = []

    def start(self) -> None:
        self.start_calls += 1

    def stop(self, *, wall_seconds: float) -> ObservabilityMeasurement:
        self.stop_calls.append(wall_seconds)
        return ObservabilityMeasurement(
            cpu_utilization_percent=123.0,
            gpu_utilization_percent=45.0,
            peak_gpu_memory_bytes=987_654_321,
        )


@dataclass
class FakeCpuTimes:
    user: float
    system: float


class FakeProcess:
    def __init__(
        self,
        pid: int,
        created: float,
        *,
        user: float,
        system: float,
        children: tuple["FakeProcess", ...] = (),
    ) -> None:
        self.pid = pid
        self.created = created
        self.user = user
        self.system = system
        self.descendants = children

    def create_time(self) -> float:
        return self.created

    def cpu_times(self) -> FakeCpuTimes:
        return FakeCpuTimes(self.user, self.system)

    def children(self, *, recursive: bool) -> list["FakeProcess"]:
        if not recursive:
            raise AssertionError("sampler must include the recursive process tree")
        return list(self.descendants)


class FakePsutil:
    def __init__(self, root: FakeProcess) -> None:
        self.root = root

    def Process(self, pid: int) -> FakeProcess:  # noqa: N802 - mirrors psutil
        self.requested_pid = pid
        return self.root


@dataclass(frozen=True)
class FakeMemoryInfo:
    used: int


@dataclass(frozen=True)
class FakeUtilization:
    gpu: float


class FakeNvml:
    def __init__(self) -> None:
        self.initialized = False
        self.memory_samples = iter((100, 250))
        self.utilization_samples = iter((10.0, 30.0))

    def nvmlInit(self) -> None:  # noqa: N802 - mirrors pynvml
        self.initialized = True

    @staticmethod
    def nvmlDeviceGetHandleByIndex(index: int) -> str:  # noqa: N802
        if index != 0:
            raise AssertionError("sampler must observe primary GPU zero")
        return "gpu-0"

    def nvmlDeviceGetMemoryInfo(self, handle: str) -> FakeMemoryInfo:  # noqa: N802
        if handle != "gpu-0":
            raise AssertionError("unexpected NVML handle")
        return FakeMemoryInfo(next(self.memory_samples))

    def nvmlDeviceGetUtilizationRates(self, handle: str) -> FakeUtilization:  # noqa: N802
        if handle != "gpu-0":
            raise AssertionError("unexpected NVML handle")
        return FakeUtilization(next(self.utilization_samples))


def primary_environment() -> dict[str, object]:
    return {
        "gpu_model": PRIMARY_GPU_NAME,
        "compute_capability": PRIMARY_COMPUTE_CAPABILITY,
        "gpu_count": 1,
        "cpu_model": "Intel Core i7-13700K (fake)",
        "ram_bytes": PRIMARY_RAM_BYTES,
        "os": "Ubuntu 22.04, Linux fake, x86_64",
        "nvidia_driver_version": PRIMARY_DRIVER_VERSION,
        "cuda_toolkit_version": "wheel-build-fake",
        "cuda_runtime_version": "fake",
    }


def make_backend(
    *, expose_metrics: bool = True, wall_timer: StepTimer | None = None
) -> tuple[VllmBackend, FakeLLM, FakeTokenizer, FakeObservabilitySampler]:
    llm = FakeLLM(expose_metrics=expose_metrics)
    tokenizer = FakeTokenizer()
    sampler = FakeObservabilitySampler()
    backend = VllmBackend(
        llm=llm,
        tokenizer=tokenizer,
        sampling_params_type=FakeSamplingParams,
        tokens_prompt_type=FakeTokensPrompt,
        request_output_kind_delta=FakeRequestOutputKind.DELTA,
        metadata=BackendMetadata(
            engine_version=VLLM_VERSION,
            model_revision=MODEL_REVISION,
            weights_sha256=MODEL_WEIGHTS_SHA256,
            local_files_only=True,
        ),
        environment=primary_environment(),
        observability_sampler=sampler,
        wall_timer=wall_timer or StepTimer(step=0.001, start=1_700_000_000.0),
    )
    return backend, llm, tokenizer, sampler


class AdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        FakeSamplingParams.created.clear()

    def _run(
        self, root: Path, *, warm_state: str
    ) -> tuple[
        list[dict],
        VllmBackend,
        FakeLLM,
        FakeTokenizer,
        FakeObservabilitySampler,
    ]:
        backend, llm, tokenizer, sampler = make_backend()
        factory_calls: list[dict[str, object]] = []

        def factory(**kwargs: object) -> VllmBackend:
            factory_calls.append(dict(kwargs))
            return backend

        result_dir = root / f"result-{warm_state}"
        count = run_benchmark(
            matrix_path=REPOSITORY_ROOT / "benchmarks/matrix.yaml",
            prompts_path=REPOSITORY_ROOT / "benchmarks/prompts.jsonl",
            result_dir=result_dir,
            run_index=2,
            run_id=f"vllm-fake-{warm_state}-2",
            warm_state=warm_state,
            concurrency=2,
            prompt_tokens=128,
            output_tokens=32,
            backend_factory=factory,
            now=lambda: FIXED_TIME,
            timer=StepTimer(),
        )
        rows = [
            json.loads(line)
            for line in (result_dir / "raw.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(count, len(rows))
        self.assertEqual(
            factory_calls, [{"local_files_only": True, "max_num_seqs": 2}]
        )
        return rows, backend, llm, tokenizer, sampler

    def _validate_schema(self, rows: list[dict]) -> None:
        validator_path = REPOSITORY_ROOT / "benchmarks/scripts/validate_contract.py"
        spec = importlib.util.spec_from_file_location("contract_validator", validator_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        schema = json.loads(
            (REPOSITORY_ROOT / "benchmarks/schemas/result.schema.json").read_text(
                encoding="utf-8"
            )
        )
        for row in rows:
            module.validate_instance(row, schema)

    def test_import_does_not_load_vllm(self) -> None:
        self.assertNotIn("vllm", sys.modules)

    def test_token_materialization_uses_bos_repeat_and_u32_le_hash(self) -> None:
        backend, _, tokenizer, _ = make_backend()
        rows = backend.materialize_token_ids(("", "seed"), prompt_tokens=8)
        self.assertEqual(rows[0], (7,) * 8)
        self.assertEqual(rows[1], (7, 11, 13, 7, 11, 13, 7, 11))
        self.assertTrue(all(add_special_tokens for _, add_special_tokens in tokenizer.calls))
        expected = hashlib.sha256(
            b"".join(struct.pack("<I", token_id) for token_id in rows[1])
        ).hexdigest()
        self.assertEqual(prompt_token_ids_sha256(rows[1]), expected)

    def test_cold_cell_emits_one_strict_trial(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            rows, _, llm, tokenizer, sampler = self._run(
                Path(directory), warm_state="cold"
            )
            self.assertEqual(len(rows), 1)
            self.assertEqual(len(llm.calls), 1)
            self.assertEqual(len(tokenizer.calls), 2)
            row = rows[0]
            self.assertEqual(row["trial_index"], 1)
            self.assertEqual(row["implementation_id"], "vllm")
            self.assertEqual(row["metrics"]["model_load_ms"], 50.0)
            self.assertAlmostEqual(row["metrics"]["batch_wall_ms"], 1850.0)
            self.assertEqual(row["metrics"]["peak_vram_bytes"], 987_654_321)
            self.assertEqual(row["metrics"]["gpu_utilization_percent"], 45.0)
            self.assertEqual(row["metrics"]["cpu_utilization_percent"], 123.0)
            self.assertAlmostEqual(row["requests"][0]["ttft_ms"], 100.0)
            self.assertAlmostEqual(row["requests"][0]["mean_tpot_ms"], 50.0)
            self.assertAlmostEqual(row["requests"][0]["end_to_end_ms"], 1650.0)
            self.assertEqual(len(row["requests"][0]["itl_ms"]), 31)
            self.assertTrue(
                all(
                    abs(value - 50.0) < 1e-9
                    for value in row["requests"][0]["itl_ms"]
                )
            )
            self.assertAlmostEqual(row["requests"][1]["ttft_ms"], 150.0)
            self.assertAlmostEqual(row["requests"][1]["end_to_end_ms"], 1700.0)
            self.assertNotEqual(
                row["requests"][0]["end_to_end_ms"],
                row["requests"][1]["end_to_end_ms"],
            )
            self.assertEqual(sampler.start_calls, 1)
            self.assertEqual(len(sampler.stop_calls), 1)
            self.assertRegex(
                row["requests"][0]["prompt_token_ids_sha256"], r"^[0-9a-f]{64}$"
            )
            expected_output_hash = hashlib.sha256(
                b"".join(struct.pack("<I", token_id) for token_id in range(32))
            ).hexdigest()
            self.assertEqual(
                row["requests"][0]["generated_token_ids_sha256"],
                expected_output_hash,
            )
            self._validate_schema(rows)

    def test_engine_timing_sanity_prefers_direct_first_token_latency(self) -> None:
        ttft, end_to_end, mean_tpot = _engine_timing_durations(
            FakeMetrics(first_token_latency=0.007),
            generated_tokens=32,
        )
        self.assertEqual(ttft, 0.007)
        self.assertAlmostEqual(end_to_end, 0.038)
        self.assertAlmostEqual(mean_tpot, 0.001)

    def test_legacy_engine_timing_sanity_includes_queue_latency(self) -> None:
        ttft, end_to_end, mean_tpot = _engine_timing_durations(
            {
                "arrival_time": 10.0,
                "first_scheduled_time": 10.003,
                "first_token_time": 10.008,
                "last_token_time": 10.039,
                "finished_time": 10.041,
            },
            generated_tokens=32,
        )
        self.assertAlmostEqual(ttft, 0.008)
        self.assertAlmostEqual(end_to_end, 0.041)
        self.assertAlmostEqual(mean_tpot, 0.001)

    def test_engine_wall_arrival_is_separate_from_monotonic_row_clock(self) -> None:
        wall_timer = StepTimer(step=0.001, start=1_700_000_000.0)
        backend, llm, _, _ = make_backend(wall_timer=wall_timer)
        token_rows = backend.materialize_token_ids(("seed",), prompt_tokens=4)
        measurement = backend.generate_batch(
            token_rows, max_new_tokens=2, timer=StepTimer(step=0.01)
        )
        self.assertGreater(llm.calls[0][0]["arrival_time"], 1_000_000_000)
        self.assertLess(measurement.requests[0].ttft_seconds, 1.0)

    def test_engine_timing_sanity_rejects_duration_later_than_host(self) -> None:
        with self.assertRaisesRegex(AdapterError, "engine TTFT"):
            _validate_engine_timing_sanity(
                FakeMetrics(first_token_latency=1.0),
                generated_tokens=32,
                host_ttft=0.01,
                host_end_to_end=0.10,
                host_itl=(0.002,) * 31,
            )

    def test_sampler_includes_worker_cpu_and_device_wide_gpu_metrics(self) -> None:
        child = FakeProcess(202, 2.0, user=0.3, system=0.2)
        root = FakeProcess(
            101,
            1.0,
            user=0.2,
            system=0.1,
            children=(child,),
        )
        nvml = FakeNvml()
        sampler = NvmlProcessTreeSampler(
            nvml_module=nvml,
            psutil_module=FakePsutil(root),
            sample_interval_seconds=60.0,
        )
        sampler.start()
        root.user += 0.08
        root.system += 0.02
        child.user += 0.15
        child.system += 0.05
        measurement = sampler.stop(wall_seconds=0.1)
        self.assertTrue(nvml.initialized)
        self.assertAlmostEqual(measurement.cpu_utilization_percent, 300.0)
        self.assertEqual(measurement.gpu_utilization_percent, 20.0)
        self.assertEqual(measurement.peak_gpu_memory_bytes, 250)

    def test_warm_cell_runs_five_warmups_and_thirty_trials(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            rows, _, llm, _, _ = self._run(Path(directory), warm_state="warm")
            self.assertEqual(len(rows), 30)
            self.assertEqual(len(llm.calls), 35)
            self.assertEqual([row["trial_index"] for row in rows], list(range(1, 31)))
            self.assertEqual(len({row["trial_id"] for row in rows}), 30)
            self._validate_schema(rows)

    def test_public_vllm_call_contract_is_exact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, _, llm, _, _ = self._run(Path(directory), warm_state="cold")
            calls = llm.calls[0]
            self.assertEqual(len(calls), 2)
            self.assertEqual(len(calls[0]["prompt"]["prompt_token_ids"]), 128)
            sampling = calls[0]["params"]
            self.assertEqual(
                sampling.values,
                {
                    "temperature": 0,
                    "max_tokens": 32,
                    "min_tokens": 32,
                    "ignore_eos": True,
                    "detokenize": False,
                    "output_kind": FakeRequestOutputKind.DELTA,
                },
            )

    def test_llm_disables_cross_trial_prefix_caching(self) -> None:
        options = _llm_options(max_num_seqs=8)
        self.assertIs(options["enable_prefix_caching"], False)
        self.assertIs(options["disable_log_stats"], False)
        self.assertEqual(options["max_num_seqs"], 8)
        self.assertEqual(options["revision"], MODEL_REVISION)

    def test_runtime_pins_non_jit_vllm_sampler_before_import(self) -> None:
        variable = "VLLM_USE_FLASHINFER_SAMPLER"
        with mock.patch.dict("os.environ", {}, clear=True):
            _pin_vllm_environment()
            self.assertEqual(__import__("os").environ[variable], "0")
        with mock.patch.dict("os.environ", {variable: "1"}, clear=True):
            with self.assertRaisesRegex(AdapterError, variable):
                _pin_vllm_environment()

    def test_missing_request_metrics_fail_instead_of_fabricating_ttft(self) -> None:
        backend, _, _, _ = make_backend(expose_metrics=False)
        token_rows = backend.materialize_token_ids(("seed",), prompt_tokens=4)
        with self.assertRaisesRegex(AdapterError, "timing metric family"):
            backend.generate_batch(
                token_rows, max_new_tokens=2, timer=StepTimer(step=0.25)
            )

    def test_multi_token_delta_is_rejected_and_pending_requests_are_aborted(self) -> None:
        backend, llm, _, _ = make_backend()
        llm.llm_engine.emit_multiple_tokens = True
        token_rows = backend.materialize_token_ids(("seed", "other"), prompt_tokens=4)
        with self.assertRaisesRegex(AdapterError, "multiple or zero tokens"):
            backend.generate_batch(
                token_rows, max_new_tokens=2, timer=StepTimer(step=0.01)
            )
        self.assertEqual(llm.llm_engine.engine_core.shutdown_calls, 1)
        self.assertEqual(len(llm.llm_engine.abort_calls), 1)
        aborted, internal = llm.llm_engine.abort_calls[0]
        self.assertEqual(len(aborted), 2)
        self.assertTrue(internal)

    def test_snapshot_hash_verification_binds_weights_config_and_tokenizer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory)
            payloads = {
                "model.safetensors": b"tiny fake safetensors",
                "config.json": b"fake config",
                "merges.txt": b"fake merges",
                "special_tokens_map.json": b"fake special tokens",
                "tokenizer.json": b"fake tokenizer",
                "tokenizer_config.json": b"fake tokenizer config",
                "vocab.json": b"fake vocab",
            }
            for filename, payload in payloads.items():
                (snapshot / filename).write_bytes(payload)
            digests = {
                filename: hashlib.sha256(payload).hexdigest()
                for filename, payload in payloads.items()
            }
            tokenizer_files = {
                filename: digests[filename]
                for filename in (
                    "merges.txt",
                    "special_tokens_map.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "vocab.json",
                )
            }
            aggregate = hashlib.sha256(
                json.dumps(
                    tokenizer_files,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            self.assertEqual(
                verify_snapshot_artifacts(
                    snapshot,
                    expected_weights_sha256=digests["model.safetensors"],
                    expected_config_sha256=digests["config.json"],
                    expected_tokenizer_files_sha256=tokenizer_files,
                    expected_tokenizer_sha256=aggregate,
                ),
                digests["model.safetensors"],
            )
            with self.assertRaisesRegex(AdapterError, "tokenizer aggregate SHA-256"):
                verify_snapshot_artifacts(
                    snapshot,
                    expected_weights_sha256=digests["model.safetensors"],
                    expected_config_sha256=digests["config.json"],
                    expected_tokenizer_files_sha256=tokenizer_files,
                    expected_tokenizer_sha256="0" * 64,
                )
            for filename in ("config.json", "tokenizer.json"):
                with self.subTest(filename=filename):
                    original = (snapshot / filename).read_bytes()
                    (snapshot / filename).write_bytes(original + b"tampered")
                    with self.assertRaisesRegex(AdapterError, f"{filename} SHA-256"):
                        verify_snapshot_artifacts(
                            snapshot,
                            expected_weights_sha256=digests["model.safetensors"],
                            expected_config_sha256=digests["config.json"],
                            expected_tokenizer_files_sha256=tokenizer_files,
                            expected_tokenizer_sha256=aggregate,
                        )
                    (snapshot / filename).write_bytes(original)

    def test_existing_result_directory_is_rejected_before_engine_load(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result_dir = Path(directory) / "existing"
            result_dir.mkdir()
            called = False

            def factory(**kwargs: object) -> VllmBackend:
                del kwargs
                nonlocal called
                called = True
                return make_backend()[0]

            with self.assertRaisesRegex(AdapterError, "reuse"):
                run_benchmark(
                    matrix_path=REPOSITORY_ROOT / "benchmarks/matrix.yaml",
                    prompts_path=REPOSITORY_ROOT / "benchmarks/prompts.jsonl",
                    result_dir=result_dir,
                    run_index=1,
                    run_id="existing-test",
                    warm_state="cold",
                    concurrency=1,
                    prompt_tokens=128,
                    output_tokens=32,
                    backend_factory=factory,
                )
            self.assertFalse(called)


if __name__ == "__main__":
    unittest.main()
