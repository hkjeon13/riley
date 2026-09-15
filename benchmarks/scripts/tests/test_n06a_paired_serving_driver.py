from __future__ import annotations

import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest import mock


SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import n03b_n06a_d128_repeat_summary as summary
import n06a_paired_serving_driver as driver


class FakeProcess:
    _next_pid = 90_000

    def __init__(self, *, stdin: object | None) -> None:
        self.pid = FakeProcess._next_pid
        FakeProcess._next_pid += 1
        self.stdin = stdin
        self.returncode: int | None = None

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def terminate(self) -> None:
        self.returncode = -15

    def kill(self) -> None:
        self.returncode = -9


class FakePopen:
    def __init__(self, events: list[str] | None = None) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.events = events

    def __call__(self, argv: list[str], **kwargs: object) -> FakeProcess:
        if self.events is not None:
            self.events.append("popen")
        self.calls.append(tuple(argv))
        process = FakeProcess(stdin=io.BytesIO() if kwargs["stdin"] is not None else None)
        stdout = kwargs["stdout"]
        stderr = kwargs["stderr"]
        assert hasattr(stdout, "write") and hasattr(stderr, "write")
        if "--decode-attention-backend" in argv:
            stderr.write(
                (
                    "RILEY_DECODE_ATTENTION "
                    "requested_backend=native-bf16-paged-split-gqa-d128-two-stage "
                    f"resolved_ragged_backend={driver.RESOLVED_BACKEND_ID} "
                    "fallback_reason=none query_heads=16 key_value_heads=2 "
                    "head_size=128 page_size=16 graph=false\n"
                ).encode("utf-8")
            )
        else:
            stdout.write(
                b"INFO Using FLASH_ATTN attention backend out of potential backends: "
                b"FLASH_ATTN, XFORMERS.\n"
            )
        stdout.flush()
        stderr.flush()
        return process


class FakeGpuMemorySampler:
    def __init__(self, *, monotonic_ns: object, **_: object) -> None:
        self.started = False
        assert callable(monotonic_ns)
        self.monotonic_ns = monotonic_ns
        self.initial_started_ns = 0
        self.initial_finished_ns = 0

    def start(self) -> None:
        self.started = True
        self.initial_started_ns = self.monotonic_ns()
        self.initial_finished_ns = self.monotonic_ns()

    def stop(self) -> dict[str, object]:
        assert self.started
        final_started_ns = self.monotonic_ns()
        return {
            "schema_version": "riley.n06a-whole-gpu-memory-samples.v1",
            "gpu_index": 0,
            "sampling_interval_seconds": 0.25,
            "samples": [
                {
                    "started_ns": self.initial_started_ns,
                    "finished_ns": self.initial_finished_ns,
                    "monotonic_ns": self.initial_finished_ns,
                    "memory_used_bytes": 18_000_000_000,
                },
                {
                    "started_ns": self.initial_finished_ns + 1,
                    "finished_ns": final_started_ns - 1,
                    "monotonic_ns": final_started_ns - 1,
                    "memory_used_bytes": 18_500_000_000,
                },
                {
                    "started_ns": final_started_ns,
                    "finished_ns": final_started_ns + 1,
                    "monotonic_ns": final_started_ns + 1,
                    "memory_used_bytes": 18_250_000_000,
                },
            ],
            "sample_count": 3,
            "peak_used_bytes": 18_500_000_000,
            "errors": [],
        }


class OverBudgetGpuMemorySampler(FakeGpuMemorySampler):
    def stop(self) -> dict[str, object]:
        result = super().stop()
        result["samples"] = [
            {**result["samples"][0], "memory_used_bytes": 18_000_000_000},
            {**result["samples"][1], "memory_used_bytes": 19_000_000_001},
            result["samples"][2],
        ]
        result["peak_used_bytes"] = 19_000_000_001
        return result


class LifecycleGapGpuMemorySampler(FakeGpuMemorySampler):
    def stop(self) -> dict[str, object]:
        result = super().stop()
        samples = result["samples"]
        assert isinstance(samples, list)
        samples[1] = {
            **samples[1],
            "started_ns": int(samples[1]["started_ns"]) + 2_000_000_000,
            "finished_ns": int(samples[1]["finished_ns"]) + 2_000_000_000,
            "monotonic_ns": int(samples[1]["monotonic_ns"]) + 2_000_000_000,
        }
        samples[2] = {
            **samples[2],
            "started_ns": int(samples[2]["started_ns"]) + 2_000_000_000,
            "finished_ns": int(samples[2]["finished_ns"]) + 2_000_000_000,
            "monotonic_ns": int(samples[2]["monotonic_ns"]) + 2_000_000_000,
        }
        return result


class SyntheticRunner:
    def __init__(self, *, fail_warmup: bool = False, inconsistent_warmup: bool = False) -> None:
        self.calls: list[dict[str, object]] = []
        self.fail_warmup = fail_warmup
        self.inconsistent_warmup = inconsistent_warmup
        self._time = 1_000_000_000

    def __call__(self, **kwargs: object) -> tuple[list[dict[str, object]], dict[str, object]]:
        self.calls.append(dict(kwargs))
        workload = kwargs["workload"]
        assert isinstance(workload, driver.Workload)
        count = int(kwargs["count"])
        phase = str(kwargs["phase"])
        rows: list[dict[str, object]] = []
        for index in range(count):
            start = self._time + index * 10
            finished = start + 1_000_000
            rows.append(
                {
                    "schema_version": "riley.http-token-observation.v2",
                    "status": "success",
                    "index": index,
                    "phase": phase,
                    "warmup": phase != "retained",
                    "started_ns": start,
                    "finished_ns": finished,
                    "call_finished_ns": finished,
                    "response_identity": {"id": f"{phase}-{index}"},
                    "protocol_valid": True,
                    "reference_match": True,
                    "streaming": True,
                    "mode": "strict",
                    "token_ids": list(workload.output_token_ids),
                    "prompt_token_ids": list(workload.prompt_token_ids),
                    "token_delivery_groups": {"frame_token_counts": [1] * workload.max_output_tokens},
                    "token_arrival_ns": [start + 100_000, start + 900_000],
                    "metrics": {
                        "token_ttft_ns": 100_000,
                        "token_tpot_ns": 800_000,
                        "e2e_ns": 1_000_000,
                    },
                    "frames": [{"arrived_ns": start + 100_000, "data": "fixture"}],
                }
            )
        self._time += 10_000_000
        complete = not (phase == "server-warmup" and self.fail_warmup)
        inconsistent = phase == "server-warmup" and self.inconsistent_warmup
        return rows, {
            "schema_version": "riley.n06a-streaming-phase.v1",
            "phase": phase,
            "warmup": phase != "retained",
            "offered_concurrency": int(kwargs["concurrency"]),
            "requested": count,
            "attempted": count,
            "succeeded": count if complete and not inconsistent else count - 1,
            "failed": 1 if inconsistent or not complete else 0,
            "phase_wall_ns": 2_000_000,
            "completed": complete,
        }


class FakeTokenClient:
    def __init__(self, *, grouped: bool = False) -> None:
        self.grouped = grouped
        self.aborted = False
        self._lock = threading.Lock()
        self._next = 0

    def __enter__(self) -> "FakeTokenClient":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def abort_pending(self, _: str) -> None:
        self.aborted = True

    def request(self, _: int, __: dict[str, object], reference: object, **___: object) -> dict[str, object]:
        assert isinstance(reference, driver.token_client.TokenReference)
        with self._lock:
            index = self._next
            self._next += 1
        start = 10_000 + index * 100
        counts = [2] if self.grouped else [1, 1]
        return {
            "schema_version": "riley.http-token-observation.v2",
            "status": "success",
            "started_ns": start,
            "finished_ns": start + 50,
            "call_finished_ns": start + 50,
            "response_identity": {"id": f"response-{index}"},
            "protocol_valid": True,
            "reference_match": True,
            "streaming": True,
            "mode": "strict",
            "token_ids": list(reference.output_token_ids),
            "prompt_token_ids": list(reference.prompt_token_ids),
            "token_delivery_groups": {"frame_token_counts": counts},
            "token_arrival_ns": [start + 10, start + 40],
            "metrics": {"token_ttft_ns": 10, "token_tpot_ns": 30, "e2e_ns": 50},
            "frames": [{"arrived_ns": start + 10, "data": "raw-frame"}],
        }


class N06aPairedServingDriverTests(unittest.TestCase):
    def make_workload(self, directory: Path) -> driver.Workload:
        path = directory / "workload.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": driver.WORKLOAD_SCHEMA_VERSION,
                    "case": "qwen3b-c2-p4-o2",
                    "model_id": driver.MODEL_ID,
                    "model_revision": driver.MODEL_REVISION,
                    "prompt": "test prompt",
                    "prompt_token_ids": [10, 11, 12, 13],
                    "output_token_ids": [20, 21],
                    "output_text": "ok",
                    "finish_reason": "length",
                    "sampling": {"temperature": 0.0, "top_p": 1.0},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return driver.load_workload(path)

    def make_model_identity(self, directory: Path, model: Path) -> driver.ModelIdentityManifest:
        metadata = {
            "config.json": b'{"architectures":["Qwen2ForCausalLM"]}\n',
            "tokenizer_config.json": b'{"model_max_length":32768}\n',
        }
        shards = {
            "model-00001-of-00002.safetensors": b"first fixture shard",
            "model-00002-of-00002.safetensors": b"second fixture shard",
        }
        for relative_path, payload in {**metadata, **shards}.items():
            (model / relative_path).write_bytes(payload)
        manifest_path = directory / "model-identity-manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": driver.MODEL_IDENTITY_MANIFEST_SCHEMA_VERSION,
                    "model_id": driver.MODEL_ID,
                    "model_revision": driver.MODEL_REVISION,
                    "model_path": str(model.resolve()),
                    "metadata_files": [
                        {
                            "path": relative_path,
                            "size_bytes": len(payload),
                            "sha256": hashlib.sha256(payload).hexdigest(),
                        }
                        for relative_path, payload in metadata.items()
                    ],
                    "shards": [
                        {
                            "path": relative_path,
                            "size_bytes": len(payload),
                            "lfs_oid_sha256": hashlib.sha256(("lfs:" + relative_path).encode("utf-8")).hexdigest(),
                        }
                        for relative_path, payload in shards.items()
                    ],
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return driver._load_model_identity_manifest(manifest_path, model_path=model.resolve())

    def make_config(self, directory: Path) -> driver.DriverConfig:
        model = directory / "model"
        model.mkdir()
        riley = directory / "riley"
        vllm = directory / "docker"
        nvidia_smi = directory / "nvidia-smi"
        model_git = directory / "git"
        for executable in (riley, vllm, nvidia_smi, model_git):
            executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            executable.chmod(0o755)
        digest = "sha256:" + "a" * 64
        template_path = directory / "vllm-command.json"
        template_path.write_text(
            json.dumps(
                [
                    str(vllm),
                    "run",
                    "--rm",
                    "--gpus",
                    "device=0",
                    "--ipc",
                    "host",
                    "--network",
                    "host",
                    "-v",
                    "{model_path}:/model:ro",
                    "vllm/vllm-openai@" + digest,
                    "--model",
                    "/model",
                    "--served-model-name",
                    "{model_id}",
                    "--dtype",
                    "bfloat16",
                    "--kv-cache-dtype",
                    "bfloat16",
                    "--enable-chunked-prefill",
                    "--no-enable-prefix-caching",
                    "--port",
                    "{port}",
                    "--max-model-len",
                    "{max_model_len}",
                    "--max-num-seqs",
                    "{concurrency}",
                    "--max-num-batched-tokens",
                    "{batch_token_budget}",
                    "--gpu-memory-utilization",
                    "0.65",
                ]
            ),
            encoding="utf-8",
        )
        workload = self.make_workload(directory)
        model_identity = self.make_model_identity(directory, model)
        return driver.DriverConfig(
            artifact_root=directory / "artifacts",
            workload=workload,
            model_path=model.resolve(),
            model_identity=model_identity,
            model_git=model_git.resolve(),
            model_git_sha256=hashlib.sha256(model_git.read_bytes()).hexdigest(),
            riley_binary=riley.resolve(),
            vllm_template=tuple(json.loads(template_path.read_text(encoding="utf-8"))),
            vllm_command_path=template_path.resolve(),
            vllm_command_sha256=hashlib.sha256(template_path.read_bytes()).hexdigest(),
            riley_binary_sha256=hashlib.sha256(riley.read_bytes()).hexdigest(),
            vllm_launcher_sha256=hashlib.sha256(vllm.read_bytes()).hexdigest(),
            vllm_backend_requested=driver.VLLM_AUTO_BACKEND_REQUESTED,
            vllm_backend_receipt_regex=__import__("re").compile(
                r"Using (?P<backend_resolved>[A-Z0-9_]+) attention backend out of potential backends:"
            ),
            vllm_startup_required_fragments=("Using FLASH_ATTN attention backend out of potential backends:",),
            vllm_image_digest=digest,
            vllm_gpu_memory_utilization=driver.VLLM_GPU_MEMORY_UTILIZATION,
            whole_gpu_sampled_peak_limit_bytes=driver.WHOLE_GPU_SAMPLED_PEAK_LIMIT_BYTES,
            nvidia_smi=nvidia_smi.resolve(),
            gpu_memory_sampling_interval_seconds=0.25,
            riley_port=18080,
            vllm_port=18081,
            concurrency=2,
            batch_token_budget=2,
            retained_requests=100,
            warmup_requests=2,
            max_model_len=32,
            kv_blocks=2,
            startup_timeout_seconds=5.0,
            request_timeout_seconds=5.0,
            shutdown_timeout_seconds=5.0,
            cuda_visible_devices="0",
            vllm_version_template=None,
        )

    def dependencies(
        self,
        popen: FakePopen,
        runner: SyntheticRunner,
        *,
        gpu_sampler_factory: object = FakeGpuMemorySampler,
        events: list[str] | None = None,
        lane_psi_snapshot: object | None = None,
    ) -> driver.DriverDependencies:
        stopped: list[int] = []

        def fake_wait(process: FakeProcess, _: int, lane: str, __: float) -> dict[str, object]:
            return {
                "endpoint": "/readyz" if lane == "riley" else "/health",
                "http_status": 200,
                "ready_ns": next(values),
            }

        def fake_stop(managed: driver.ManagedProcess, _: float) -> dict[str, object]:
            if events is not None:
                events.append("cleanup")
            managed.process.returncode = 0
            for stream in (managed.stdout_stream, managed.stderr_stream):
                with contextlib.suppress(Exception):
                    stream.flush()
                with contextlib.suppress(Exception):
                    stream.close()
            stopped.append(managed.process.pid)
            return {"lane": managed.lane, "pid": managed.process.pid, "returncode": 0, "cleanup_verified": True, "errors": []}

        def fake_container_cleanup(launcher: str, container_name: str, _: float) -> dict[str, object]:
            if events is not None:
                events.append("container-cleanup")
            self.assertTrue(launcher)
            self.assertRegex(container_name, r"^riley-n06a-vllm-(?:warmup|timed)-")
            return {
                "schema_version": driver.DOCKER_CONTAINER_CLEANUP_SCHEMA_VERSION,
                "launcher": launcher,
                "container_name": container_name,
                "commands": [],
                "notes": [],
                "errors": [],
                "final_state": "absent",
                "cleanup_verified": True,
            }

        def command(argv: list[str], stdout: str, *, monotonic_ns: object) -> dict[str, object]:
            assert callable(monotonic_ns)
            started_ns = monotonic_ns()
            finished_ns = monotonic_ns()
            return {
                "argv": argv,
                "started_ns": started_ns,
                "finished_ns": finished_ns,
                "timeout_seconds": driver.MODEL_IDENTITY_GIT_TIMEOUT_SECONDS,
                "returncode": 0,
                "stdout_text": stdout,
                "stdout_bytes": len(stdout.encode("utf-8")),
                "stdout_sha256": hashlib.sha256(stdout.encode("utf-8")).hexdigest(),
                "stderr_text": "",
                "stderr_bytes": 0,
                "stderr_sha256": hashlib.sha256(b"").hexdigest(),
            }

        def fake_model_identity_validator(**kwargs: object) -> dict[str, object]:
            manifest = kwargs["manifest"]
            git = kwargs["git"]
            git_sha256 = kwargs["git_sha256"]
            monotonic_ns = kwargs["monotonic_ns"]
            assert isinstance(manifest, driver.ModelIdentityManifest)
            assert isinstance(git, Path) and isinstance(git_sha256, str)
            lfs_oids = {entry.relative_path: entry.lfs_oid_sha256 for entry in manifest.shards}
            lfs_stdout = "".join(f"{lfs_oids[path]} * {path}\n" for path in sorted(lfs_oids))
            return {
                "schema_version": driver.MODEL_IDENTITY_VALIDATION_SCHEMA_VERSION,
                "manifest_sha256": manifest.source_sha256,
                "model_path": str(manifest.model_path),
                "model_revision": driver.MODEL_REVISION,
                "git": {"path": str(git), "sha256": git_sha256},
                "commands": [
                    command(
                        [str(git), "-C", str(manifest.model_path), "rev-parse", "HEAD"],
                        driver.MODEL_REVISION + "\n",
                        monotonic_ns=monotonic_ns,
                    ),
                    command(
                        [str(git), "-C", str(manifest.model_path), "lfs", "ls-files", "-l"],
                        lfs_stdout,
                        monotonic_ns=monotonic_ns,
                    ),
                ],
                "observed_git_head": driver.MODEL_REVISION,
                "observed_lfs_oids": {path: lfs_oids[path] for path in sorted(lfs_oids)},
                "validated": True,
                "errors": [],
            }

        def fake_gpu_idle_census(**kwargs: object) -> dict[str, object]:
            if events is not None:
                events.append("gpu-idle-census")
            nvidia_smi = kwargs["nvidia_smi"]
            timeout_seconds = kwargs["timeout_seconds"]
            monotonic_ns = kwargs["monotonic_ns"]
            assert isinstance(nvidia_smi, Path) and callable(monotonic_ns)

            def census_command(suffix: list[str], stdout: str) -> dict[str, object]:
                started_ns = monotonic_ns()
                finished_ns = monotonic_ns()
                return {
                    "argv": [str(nvidia_smi), *suffix],
                    "started_ns": started_ns,
                    "finished_ns": finished_ns,
                    "timeout_seconds": timeout_seconds,
                    "returncode": 0,
                    "stdout_text": stdout,
                    "stdout_bytes": len(stdout.encode("utf-8")),
                    "stdout_sha256": hashlib.sha256(stdout.encode("utf-8")).hexdigest(),
                    "stderr_text": "",
                    "stderr_bytes": 0,
                    "stderr_sha256": hashlib.sha256(b"").hexdigest(),
                }

            return {
                "schema_version": driver.GPU_IDLE_CENSUS_SCHEMA_VERSION,
                "gpu_index": 0,
                "max_idle_memory_bytes": driver.GPU_IDLE_CENSUS_MAX_USED_BYTES,
                "commands": [
                    census_command(
                        ["--id=0", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], ""
                    ),
                    census_command(
                        ["--id=0", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], "64 MiB\n"
                    ),
                ],
                "compute_process_pids": [],
                "memory_used_bytes": 64 * 1024 * 1024,
                "compliant": True,
                "errors": [],
            }

        values = iter(range(2_000_000_000, 3_000_000_000, 10_000))
        psi_snapshot_index = 0

        def default_lane_psi_snapshot(**kwargs: object) -> dict[str, object]:
            nonlocal psi_snapshot_index
            proc_root = kwargs["proc_root"]
            monotonic_ns = kwargs["monotonic_ns"]
            assert isinstance(proc_root, Path) and callable(monotonic_ns)
            psi_snapshot_index += 1
            base = psi_snapshot_index * 1_000

            def resource(offset: int, *, full: bool = True) -> dict[str, object]:
                value = float(base + offset)
                return {
                    "status": "ok",
                    "source": str(proc_root / "pressure" / ("cpu" if offset == 0 else "io" if offset == 100 else "memory")),
                    "some": {
                        "avg10": value,
                        "avg60": value + 0.1,
                        "avg300": value + 0.2,
                        "total": base + offset,
                    },
                    "full": None
                    if not full
                    else {
                        "avg10": value / 2.0,
                        "avg60": value / 2.0 + 0.1,
                        "avg300": value / 2.0 + 0.2,
                        "total": base + offset,
                    },
                    "error": None,
                }

            started_ns = monotonic_ns()
            finished_ns = monotonic_ns()
            return {
                "schema_version": driver.LANE_PSI_SCHEMA_VERSION,
                "policy": driver.LANE_PSI_POLICY,
                "snapshot_started_ns": started_ns,
                "snapshot_finished_ns": finished_ns,
                "psi": {
                    "cpu": resource(0, full=False),
                    "io": resource(100),
                    "memory": resource(200),
                },
            }

        return driver.DriverDependencies(
            popen=popen,
            wait_ready=fake_wait,
            run_streaming_phase=runner,
            stop_process=fake_stop,
            cleanup_vllm_container=fake_container_cleanup,
            gpu_memory_sampler_factory=gpu_sampler_factory,
            gpu_idle_census=fake_gpu_idle_census,
            lane_psi_snapshot=(
                lane_psi_snapshot
                if callable(lane_psi_snapshot)
                else default_lane_psi_snapshot
            ),
            model_identity_validator=fake_model_identity_validator,
            monotonic_ns=lambda: next(values),
        )

    def test_timed_even_attempt_runs_vllm_then_riley_with_server_warmups_and_exact_markers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            config = self.make_config(directory)
            popen = FakePopen()
            runner = SyntheticRunner()
            attempt_dir, markers = driver.execute_pair(
                config,
                driver.AttemptContext(phase="timed", index=2),
                dependencies=self.dependencies(popen, runner),
            )

            self.assertEqual(len(markers), 2)
            riley_argv = driver._riley_argv(config)
            self.assertEqual(riley_argv[riley_argv.index("--batch-token-budget") + 1], "2")
            vllm_argv = driver._vllm_argv(config)
            self.assertEqual(vllm_argv[vllm_argv.index("--max-num-batched-tokens") + 1], "2")
            self.assertIn("lane=vllm", markers[0])
            self.assertIn("lane=riley", markers[1])
            parsed_vllm = summary._parse_marker_tokens(
                markers[0], prefix=driver.MARKER_PREFIX, label="fixture vLLM marker"
            )
            parsed_riley = summary._parse_marker_tokens(
                markers[1], prefix=driver.MARKER_PREFIX, label="fixture Riley marker"
            )
            self.assertEqual(parsed_vllm["backend_resolved"], "FLASH_ATTN")
            self.assertEqual(parsed_riley["batch_token_budget"], "2")
            self.assertEqual(parsed_riley["max_model_len"], "32")
            self.assertEqual([call["port"] for call in runner.calls], [18081, 18081, 18080, 18080])
            self.assertEqual([call["phase"] for call in runner.calls], ["server-warmup", "retained", "server-warmup", "retained"])
            self.assertTrue((attempt_dir / "vllm.server-warmup.requests.jsonl").is_file())
            self.assertTrue((attempt_dir / "riley.server-warmup.requests.jsonl").is_file())
            self.assertTrue((attempt_dir / "vllm.startup.stdout.log").is_file())
            self.assertTrue((attempt_dir / "vllm.startup.stderr.log").is_file())
            self.assertTrue((attempt_dir / "riley.startup.stderr.log").is_file())
            self.assertTrue((attempt_dir / "vllm.whole-gpu-memory.samples.jsonl").is_file())
            self.assertTrue((attempt_dir / "riley.whole-gpu-memory.peak.json").is_file())
            attempt = json.loads((attempt_dir / "attempt.config.json").read_text(encoding="utf-8"))
            self.assertEqual(attempt["launch_provenance"]["riley_binary"]["sha256"], config.riley_binary_sha256)
            self.assertEqual(attempt["launch_provenance"]["vllm_command_template"]["sha256"], config.vllm_command_sha256)
            vllm_provenance = json.loads((attempt_dir / "vllm.provenance.json").read_text(encoding="utf-8"))
            self.assertEqual(vllm_provenance["vllm_startup_snapshot"]["backend_resolved"], "FLASH_ATTN")
            self.assertEqual(vllm_provenance["vllm_startup_snapshot"]["stdout_sha256"], hashlib.sha256((attempt_dir / "vllm.startup.stdout.log").read_bytes()).hexdigest())
            self.assertTrue(vllm_provenance["cleanup"]["container"]["cleanup_verified"])
            self.assertEqual(vllm_provenance["cleanup"]["container"]["final_state"], "absent")
            self.assertEqual(attempt["vllm_argv"].count("--name"), 1)
            self.assertEqual(
                attempt["vllm_argv"][attempt["vllm_argv"].index("--name") + 1],
                attempt["launch_provenance"]["vllm_docker_container"]["name"],
            )

    def test_lane_psi_artifacts_are_marker_bound_and_high_pressure_is_not_a_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            attempt_dir, markers = driver.execute_pair(
                self.make_config(directory),
                driver.AttemptContext(phase="timed", index=1),
                dependencies=self.dependencies(FakePopen(), SyntheticRunner()),
            )
            marker_by_lane = {
                summary._parse_marker_tokens(marker, prefix=driver.MARKER_PREFIX, label="fixture marker")["lane"]: marker
                for marker in markers
            }
            self.assertEqual(set(marker_by_lane), {"riley", "vllm"})
            for lane in ("riley", "vllm"):
                provenance_path = attempt_dir / f"{lane}.provenance.json"
                pressure_path = attempt_dir / f"{lane}.lane-psi.json"
                provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
                pressure = json.loads(pressure_path.read_text(encoding="utf-8"))
                marker_fields = summary._parse_marker_tokens(
                    marker_by_lane[lane], prefix=driver.MARKER_PREFIX, label=f"fixture {lane} marker"
                )
                self.assertEqual(marker_fields["lane_provenance_path"], str(provenance_path.resolve()))
                self.assertEqual(
                    marker_fields["lane_provenance_sha256"],
                    hashlib.sha256(provenance_path.read_bytes()).hexdigest(),
                )
                self.assertEqual(provenance["lane_pressure"]["path"], str(pressure_path.resolve()))
                self.assertEqual(
                    provenance["lane_pressure"]["sha256"],
                    hashlib.sha256(pressure_path.read_bytes()).hexdigest(),
                )
                self.assertEqual(pressure["policy"], driver.LANE_PSI_POLICY)
                self.assertGreaterEqual(pressure["pre"]["psi"]["io"]["some"]["avg10"], 1_000.0)
                self.assertLessEqual(
                    pressure["pre"]["snapshot_finished_ns"], provenance["started_ns"]
                )
                self.assertGreaterEqual(
                    pressure["post"]["snapshot_started_ns"], provenance["cleanup_completed_ns"]
                )

    def test_lane_psi_capture_retains_high_unavailable_and_malformed_observations(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            proc_root = Path(temporary)
            pressure_root = proc_root / "pressure"
            pressure_root.mkdir()
            (pressure_root / "cpu").write_text(
                "some avg10=999.25 avg60=17.50 avg300=4.25 total=12345\n",
                encoding="utf-8",
            )
            (pressure_root / "io").write_text("some broken-record\n", encoding="utf-8")
            ticks = iter((100, 200))
            snapshot = driver.capture_lane_psi_snapshot(
                proc_root=proc_root,
                monotonic_ns=lambda: next(ticks),
            )

        self.assertEqual(snapshot["snapshot_started_ns"], 100)
        self.assertEqual(snapshot["snapshot_finished_ns"], 200)
        self.assertEqual(snapshot["psi"]["cpu"]["status"], "ok")
        self.assertEqual(snapshot["psi"]["cpu"]["some"]["avg10"], 999.25)
        self.assertEqual(snapshot["psi"]["io"]["status"], "malformed")
        self.assertEqual(snapshot["psi"]["memory"]["status"], "unavailable")

    def test_unavailable_or_malformed_lane_psi_is_retained_without_blocking_markers(self) -> None:
        snapshots = iter(("unavailable", "malformed", "unavailable", "malformed"))

        def pressure_snapshot(**kwargs: object) -> dict[str, object]:
            status = next(snapshots)
            proc_root = kwargs["proc_root"]
            monotonic_ns = kwargs["monotonic_ns"]
            assert isinstance(proc_root, Path) and callable(monotonic_ns)
            started_ns = monotonic_ns()
            finished_ns = monotonic_ns()
            return {
                "schema_version": driver.LANE_PSI_SCHEMA_VERSION,
                "policy": driver.LANE_PSI_POLICY,
                "snapshot_started_ns": started_ns,
                "snapshot_finished_ns": finished_ns,
                "psi": {
                    resource: {
                        "status": status,
                        "source": str(proc_root / "pressure" / resource),
                        "some": None,
                        "full": None,
                        "error": f"fixture {status}",
                    }
                    for resource in driver.PSI_RESOURCES
                },
            }

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            attempt_dir, markers = driver.execute_pair(
                self.make_config(directory),
                driver.AttemptContext(phase="timed", index=1),
                dependencies=self.dependencies(
                    FakePopen(), SyntheticRunner(), lane_psi_snapshot=pressure_snapshot
                ),
            )
            self.assertEqual(len(markers), 2)
            riley_pressure = json.loads(
                (attempt_dir / "riley.lane-psi.json").read_text(encoding="utf-8")
            )
            self.assertEqual(riley_pressure["pre"]["psi"]["io"]["status"], "unavailable")
            self.assertEqual(riley_pressure["post"]["psi"]["io"]["status"], "malformed")

    def test_warmup_outer_attempt_retains_per_server_warmup_but_emits_no_markers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            config = self.make_config(directory)
            popen = FakePopen()
            runner = SyntheticRunner()
            attempt_dir, markers = driver.execute_pair(
                config,
                driver.AttemptContext(phase="warmup", index=1),
                dependencies=self.dependencies(popen, runner),
            )
            self.assertEqual(markers, ())
            self.assertEqual([call["phase"] for call in runner.calls], ["server-warmup", "server-warmup"])
            complete = json.loads((attempt_dir / "attempt.complete.json").read_text(encoding="utf-8"))
            self.assertEqual(complete["markers"], [])

    def test_failed_server_warmup_preserves_failure_and_never_emits_a_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            config = self.make_config(directory)
            popen = FakePopen()
            runner = SyntheticRunner(fail_warmup=True)
            with self.assertRaisesRegex(driver.DriverError, "strict per-server warmup"):
                driver.execute_pair(
                    config,
                    driver.AttemptContext(phase="timed", index=1),
                    dependencies=self.dependencies(popen, runner),
                )
            failure = directory / "artifacts" / "timed-001" / "attempt.failure.json"
            self.assertTrue(failure.is_file())
            self.assertEqual(json.loads(failure.read_text(encoding="utf-8"))["emitted_markers"], [])

    def test_server_warmup_requires_zero_failures_even_if_a_runner_claims_completion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            config = self.make_config(directory)
            popen = FakePopen()
            runner = SyntheticRunner(inconsistent_warmup=True)
            with self.assertRaisesRegex(driver.DriverError, "strict per-server warmup has invalid"):
                driver.execute_pair(
                    config,
                    driver.AttemptContext(phase="timed", index=1),
                    dependencies=self.dependencies(popen, runner),
                )
            failure = directory / "artifacts" / "timed-001" / "attempt.failure.json"
            self.assertTrue(failure.is_file())
            self.assertEqual(json.loads(failure.read_text(encoding="utf-8"))["emitted_markers"], [])

    def test_over_budget_whole_gpu_sampled_peak_prevents_a_marker_and_retains_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            config = self.make_config(directory)
            popen = FakePopen()
            runner = SyntheticRunner()
            with self.assertRaisesRegex(driver.DriverError, "whole-GPU sampled peak is not compliant"):
                driver.execute_pair(
                    config,
                    driver.AttemptContext(phase="timed", index=1),
                    dependencies=self.dependencies(
                        popen,
                        runner,
                        gpu_sampler_factory=OverBudgetGpuMemorySampler,
                    ),
                )
            attempt = directory / "artifacts" / "timed-001"
            peak = json.loads((attempt / "riley.whole-gpu-memory.peak.json").read_text(encoding="utf-8"))
            self.assertFalse(peak["compliant"])
            self.assertEqual(peak["peak_used_bytes"], 19_000_000_001)
            self.assertEqual(json.loads((attempt / "attempt.failure.json").read_text(encoding="utf-8"))["emitted_markers"], [])

    def test_lifecycle_gap_in_whole_gpu_sampling_prevents_a_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            config = self.make_config(directory)
            with self.assertRaisesRegex(driver.DriverError, "whole-GPU sampled peak is not compliant"):
                driver.execute_pair(
                    config,
                    driver.AttemptContext(phase="timed", index=1),
                    dependencies=self.dependencies(
                        FakePopen(),
                        SyntheticRunner(),
                        gpu_sampler_factory=LifecycleGapGpuMemorySampler,
                    ),
                )
            peak = json.loads(
                (directory / "artifacts" / "timed-001" / "riley.whole-gpu-memory.peak.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertFalse(peak["compliant"])
            self.assertTrue(peak["errors"])

    def test_sampler_starts_before_launch_and_stops_after_cleanup(self) -> None:
        events: list[str] = []

        class OrderedSampler(FakeGpuMemorySampler):
            def start(self) -> None:
                events.append("sampler-start")
                super().start()

            def stop(self) -> dict[str, object]:
                events.append("sampler-stop")
                return super().stop()

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            driver.execute_pair(
                self.make_config(directory),
                driver.AttemptContext(phase="timed", index=1),
                dependencies=self.dependencies(
                    FakePopen(events),
                    SyntheticRunner(),
                    gpu_sampler_factory=OrderedSampler,
                    events=events,
                ),
            )
        self.assertEqual(
            events,
            [
                "sampler-start",
                "popen",
                "cleanup",
                "sampler-stop",
                "gpu-idle-census",
                "sampler-start",
                "popen",
                "cleanup",
                "container-cleanup",
                "sampler-stop",
                "gpu-idle-census",
            ],
        )

    def test_context_is_explicit_and_cannot_be_implicitly_ordered(self) -> None:
        with self.assertRaisesRegex(driver.DriverError, driver.N01_PHASE_ENV):
            driver.attempt_context_from_environment({})
        self.assertEqual(
            driver.attempt_context_from_environment(
                {driver.N01_PHASE_ENV: "timed", driver.N01_INDEX_ENV: "2"}
            ).pair_order,
            "vllm-riley",
        )

    def test_build_config_requires_the_vllm_fixed_prompt_and_memory_envelope(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            workload = self.make_workload(directory)
            model = directory / "model"
            model.mkdir()
            riley = directory / "riley"
            docker = directory / "docker"
            nvidia_smi = directory / "nvidia-smi"
            model_git = directory / "git"
            for executable in (riley, docker, nvidia_smi, model_git):
                executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
                executable.chmod(0o755)
            digest = "sha256:" + "b" * 64
            template_path = directory / "vllm-command.json"
            template = [
                str(docker),
                "run",
                "--rm",
                "--gpus",
                "device=0",
                "--ipc",
                "host",
                "--network",
                "host",
                "-v",
                "{model_path}:/model:ro",
                "vllm/vllm-openai@" + digest,
                "--model",
                "/model",
                "--served-model-name",
                "{model_id}",
                "--host",
                "127.0.0.1",
                "--port",
                "{port}",
                "--dtype",
                "bfloat16",
                "--kv-cache-dtype",
                "bfloat16",
                "--enable-chunked-prefill",
                "--no-enable-prefix-caching",
                "--max-num-seqs",
                "{concurrency}",
                "--max-num-batched-tokens",
                "{batch_token_budget}",
                "--max-model-len",
                "{max_model_len}",
                "--gpu-memory-utilization",
                "0.65",
            ]
            template_path.write_text(json.dumps(template), encoding="utf-8")
            model_identity = self.make_model_identity(directory, model)
            arguments = driver.build_parser().parse_args(
                [
                    "--artifact-root",
                    str(directory / "artifacts"),
                    "--workload",
                    str(workload.source_path),
                    "--model-path",
                    str(model),
                    "--model-identity-manifest",
                    str(model_identity.source_path),
                    "--model-git",
                    str(model_git),
                    "--riley-binary",
                    str(riley),
                    "--vllm-command-json",
                    str(template_path),
                    "--vllm-image-digest",
                    digest,
                    "--vllm-backend-receipt-regex",
                    r"Using (?P<backend_resolved>[A-Z0-9_]+) attention backend out of potential backends:",
                    "--vllm-startup-required-fragment",
                    "Using FLASH_ATTN attention backend out of potential backends:",
                    "--concurrency",
                    "2",
                    "--batch-token-budget",
                    "2",
                    "--retained-requests",
                    "2",
                    "--warmup-requests",
                    "2",
                    "--max-model-len",
                    "32",
                    "--kv-blocks",
                    "2",
                    "--nvidia-smi",
                    str(nvidia_smi),
                ]
            )
            config = driver.build_config(arguments)
            self.assertEqual(config.vllm_backend_requested, driver.VLLM_AUTO_BACKEND_REQUESTED)
            self.assertEqual(config.vllm_image_digest, digest)
            self.assertEqual(config.vllm_gpu_memory_utilization, "0.65")
            wrong_image = list(template)
            image_index = wrong_image.index("vllm/vllm-openai@" + digest)
            wrong_image[image_index] = "untrusted-vllm:latest"
            # The digest still appears exactly once, but only as an inert
            # post-image argument.  It must not satisfy the image pin check.
            wrong_image.append("vllm/vllm-openai@" + digest)
            template_path.write_text(json.dumps(wrong_image), encoding="utf-8")
            with self.assertRaisesRegex(driver.DriverError, "first non-option image operand"):
                driver.build_config(arguments)
            template_path.write_text(json.dumps(template), encoding="utf-8")
            post_image_gpu = list(template)
            gpu_index = post_image_gpu.index("--gpus")
            del post_image_gpu[gpu_index : gpu_index + 2]
            image_index = post_image_gpu.index("vllm/vllm-openai@" + digest)
            post_image_gpu[image_index + 1 : image_index + 1] = ["--gpus", "device=0"]
            template_path.write_text(json.dumps(post_image_gpu), encoding="utf-8")
            with self.assertRaisesRegex(
                driver.DriverError, r"Docker control option '--gpus' must occur before"
            ):
                driver.build_config(arguments)
            template_path.write_text(json.dumps(template), encoding="utf-8")
            template_path.write_text(
                json.dumps([token for token in template if token != "--no-enable-prefix-caching"]),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(driver.DriverError, "no-enable-prefix-caching"):
                driver.build_config(arguments)
            template_path.write_text(
                json.dumps([token for token in template if token not in {"--gpus", "device=0"}]),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(driver.DriverError, "physical GPU 0"):
                driver.build_config(arguments)
            template_path.write_text(
                json.dumps([token for token in template if token != "--rm"]), encoding="utf-8"
            )
            arguments.cuda_visible_devices = "0"
            with self.assertRaisesRegex(driver.DriverError, "Docker --rm"):
                driver.build_config(arguments)
            template_path.write_text(json.dumps(template), encoding="utf-8")
            template_path.write_text(json.dumps([*template, "--detach"]), encoding="utf-8")
            arguments.cuda_visible_devices = "0"
            with self.assertRaisesRegex(driver.DriverError, "must not detach"):
                driver.build_config(arguments)
            template_path.write_text(json.dumps(template), encoding="utf-8")
            arguments.cuda_visible_devices = "1"
            with self.assertRaisesRegex(driver.DriverError, "exactly '0'"):
                driver.build_config(arguments)

    def test_docker_control_prefix_rejects_each_control_after_image(self) -> None:
        image = "vllm/vllm-openai@sha256:" + "c" * 64
        prefix = [
            "/usr/bin/docker",
            "run",
            "--name",
            "riley-n06a-vllm-timed-0001-0123456789abcdef0123",
            "--rm",
            "--gpus",
            "device=0",
            "--ipc",
            "host",
            "--network",
            "host",
            "-v",
            "/models:/model:ro",
            "--volume",
            "/scratch:/scratch:ro",
            image,
            "--model",
            "/model",
        ]
        controls = (
            ("--name", ("riley-n06a-vllm-timed-0001-0123456789abcdef0123",)),
            ("--rm", ()),
            ("--gpus", ("device=0",)),
            ("--ipc", ("host",)),
            ("--network", ("host",)),
            ("-v", ("/models:/model:ro",)),
            ("--volume", ("/scratch:/scratch:ro",)),
        )
        for flag, values in controls:
            with self.subTest(flag=flag):
                argv = list(prefix)
                flag_index = argv.index(flag)
                del argv[flag_index : flag_index + 1 + len(values)]
                image_index = argv.index(image)
                argv[image_index + 1 : image_index + 1] = [flag, *values]
                resolved_image_index = driver._docker_run_image_operand_index(
                    argv, label="fixture Docker argv"
                )
                self.assertEqual(argv[resolved_image_index], image)
                with self.assertRaisesRegex(
                    driver.DriverError, "Docker control option"
                ):
                    driver._require_docker_controls_before_image(
                        argv,
                        image_operand_index=resolved_image_index,
                        label="fixture Docker argv",
                    )

    def test_default_docker_cleanup_stops_waits_and_proves_daemon_absence_without_a_shell(self) -> None:
        calls: list[tuple[list[str], dict[str, object]]] = []
        container_name = "riley-n06a-vllm-timed-0001-0123456789abcdef0123"
        inventory_calls = 0

        def fake_run(argv: list[str], **kwargs: object) -> object:
            nonlocal inventory_calls
            calls.append((list(argv), dict(kwargs)))
            arguments = argv[1:]
            if arguments[:3] == ["container", "ls", "--all"]:
                inventory_calls += 1
                stdout = (
                    f"{container_name}\t0123456789ab\n".encode("utf-8")
                    if inventory_calls == 1
                    else b""
                )
                return type("Completed", (), {"returncode": 0, "stdout": stdout, "stderr": b""})()
            if arguments[:2] == ["container", "stop"]:
                return type("Completed", (), {"returncode": 0, "stdout": b"stopped\n", "stderr": b""})()
            if arguments[:2] == ["container", "wait"]:
                return type("Completed", (), {"returncode": 1, "stdout": b"", "stderr": b"removed\n"})()
            if arguments[:2] == ["container", "inspect"]:
                return type("Completed", (), {"returncode": 1, "stdout": b"", "stderr": b"No such container\n"})()
            self.fail(f"unexpected Docker cleanup argv: {argv!r}")

        with mock.patch.object(driver.subprocess, "run", side_effect=fake_run):
            receipt = driver._default_cleanup_vllm_container("/usr/bin/docker", container_name, 5.0)
        self.assertTrue(receipt["cleanup_verified"])
        self.assertEqual(receipt["final_state"], "absent")
        self.assertEqual(receipt["errors"], [])
        self.assertEqual(
            [command["operation"] for command in receipt["commands"]],
            ["inventory-before", "stop", "wait", "inventory-after-stop", "inspect-absence"],
        )
        self.assertTrue(all(arguments["shell"] is False for _, arguments in calls))
        self.assertTrue(all(float(arguments["timeout"]) <= 30.0 for _, arguments in calls))

    def test_grouped_sse_is_retained_but_cannot_complete_the_phase(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workload = self.make_workload(Path(temporary))
            fake = FakeTokenClient(grouped=True)
            rows, accounting = driver.run_streaming_phase(
                port=18080,
                workload=workload,
                concurrency=1,
                count=1,
                request_timeout_seconds=5.0,
                client_factory=lambda: fake,
            )
            self.assertFalse(accounting["completed"])
            self.assertEqual(rows[0]["status"], "failed")
            self.assertIn("n06a_validation_error", rows[0])
            self.assertEqual(rows[0]["frames"], [{"arrived_ns": 10_010, "data": "raw-frame"}])


if __name__ == "__main__":
    unittest.main()
