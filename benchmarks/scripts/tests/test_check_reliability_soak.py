from __future__ import annotations

import copy
import hashlib
import importlib.util
import io
import json
import os
import re
import shutil
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "check_reliability_soak.py"
SPEC = importlib.util.spec_from_file_location("check_reliability_soak", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
checker = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = checker
SPEC.loader.exec_module(checker)


def digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def write_raw_tar(path: Path, payloads: dict[str, bytes]) -> None:
    with tarfile.open(path, "w:", format=tarfile.USTAR_FORMAT) as archive:
        for name in sorted(payloads):
            contents = payloads[name]
            member = tarfile.TarInfo(name)
            member.size = len(contents)
            member.mode = 0o644
            member.uid = 0
            member.gid = 0
            member.uname = ""
            member.gname = ""
            member.mtime = 0
            archive.addfile(member, io.BytesIO(contents))


def read_raw_tar(path: Path) -> dict[str, bytes]:
    with tarfile.open(path, "r:") as archive:
        return {
            member.name: archive.extractfile(member).read()
            for member in archive.getmembers()
        }


class SoakFixture:
    KINDS = [
        ("steady", "steady", "iteration-batch"),
        ("burst-idle", "burst-idle", "iteration-batch"),
        ("mixed-short-long", "mixed", "iteration-batch"),
        ("invalid", "invalid", "iteration-batch"),
        ("overload", "overload", "iteration-batch"),
        ("cancellation-disconnect", "cancellation-disconnect", "iteration-batch"),
        ("near-kv", "near-kv", "iteration-batch"),
        ("graceful-restart", "graceful-restart", "iteration-batch"),
        ("rollback-iteration-batch", "rollback", "iteration-batch"),
        ("rollback-per-operation", "rollback", "per-operation"),
    ]

    def __init__(self, root: Path) -> None:
        self.root = root
        self.run_directory = root / "run"
        self.run_directory.mkdir()
        self.runtime_receipts_directory = root / "runtime-receipts"
        self.runtime_receipts_directory.mkdir()
        self.golden = digest("golden completion")
        self.source = {
            "git_commit": "a" * 40,
            "git_dirty": False,
            "source_archive_sha256": "b" * 64,
            "binary_sha256": "c" * 64,
            "image_sha256": "d" * 64,
            "model_sha256": "e" * 64,
            "model_id": "fixture/model",
            "model_revision": "f" * 40,
        }
        self.config_sha256 = digest("config")
        self.weights_sha256 = digest("weights")
        self.tokenizer_aggregate_sha256 = digest("tokenizer aggregate")
        self.tokenizer_json_sha256 = digest("tokenizer json")
        self.native_correctness_report = self._native_correctness_report()
        self.native_correctness_report_path = root / "native-correctness-report.json"
        self.native_correctness_report_path.write_text(
            json.dumps(self.native_correctness_report, sort_keys=True) + "\n"
        )
        self.native_correctness_report_sha256 = hashlib.sha256(
            self.native_correctness_report_path.read_bytes()
        ).hexdigest()
        self.correctness_golden = self._correctness_golden()
        self.correctness_golden_path = root / "correctness-golden.json"
        self.correctness_golden_path.write_text(
            json.dumps(self.correctness_golden, sort_keys=True) + "\n"
        )
        self.manifest = self._manifest()
        self.events: list[dict[str, object]] = []
        self.binding = checker._canonical_sha256(self.source)
        self.release_image_id = "sha256:" + self.source["image_sha256"]
        self.test_image_id = "sha256:" + digest("soak test layer")
        self.container_id = digest("soak container")
        self.container_name = (
            f"riley-soak-{self.source['git_commit'][:12]}-20260826T000000Z"
        )
        self.release_layers = ["sha256:" + digest("release layer")]
        self.test_layers = [
            *self.release_layers,
            "sha256:" + digest("test layer packages"),
            "sha256:" + digest("test layer scripts"),
        ]
        self.release_image_environment = [
            f"{name}={setting}"
            for name, setting in checker.SOAK_RELEASE_ENVIRONMENT.items()
        ]
        self.test_image_environment = [
            *self.release_image_environment,
            "DEBIAN_FRONTEND=noninteractive",
            "LC_ALL=C",
            "TZ=UTC",
        ]
        self._build_events()
        self.write()

    def _native_correctness_report(self) -> dict[str, object]:
        return {
            "schema_version": "1.0.0",
            "gate_id": "smollm2-fp32-bf16-native-e0-v3",
            "status": "pass",
            "bindings": {
                "candidate_git_revision": self.source["git_commit"],
                "candidate_git_status_sha256": hashlib.sha256(b"").hexdigest(),
                "candidate_executable_sha256": digest(
                    "native calibration executable"
                ),
                "model_id": self.source["model_id"],
                "model_revision": self.source["model_revision"],
                "config_sha256": self.config_sha256,
                "weights_sha256": self.weights_sha256,
                "tokenizer_sha256": self.tokenizer_aggregate_sha256,
            },
        }

    def _correctness_golden(self) -> dict[str, object]:
        return {
            "schema_version": "riley.python-free-release-e2e-golden.v1",
            "correctness_gate_id": "smollm2-fp32-bf16-native-e0-v3",
            "correctness_report_sha256": self.native_correctness_report_sha256,
            "source_revision": self.source["git_commit"],
            "model_id": self.source["model_id"],
            "model_revision": self.source["model_revision"],
            "config_sha256": self.config_sha256,
            "weights_sha256": self.weights_sha256,
            "tokenizer_aggregate_sha256": self.tokenizer_aggregate_sha256,
            "tokenizer_json_sha256": self.tokenizer_json_sha256,
            "prompt": "fixture prompt",
            "max_tokens": 8,
            "expected_greedy_text_sha256": self.golden,
        }

    def _manifest(self) -> dict[str, object]:
        scenarios = []
        for scenario_id, kind, mode in self.KINDS:
            request_profile = {
                "invalid": "invalid",
                "overload": "long",
                "cancellation-disconnect": "long",
                "near-kv": "near_kv",
            }.get(kind, "short")
            scenario = {
                "id": scenario_id, "kind": kind, "required": True,
                "duration_seconds": 2, "concurrency": 2,
                "cycle_interval_ms": 1000 if kind == "burst-idle" else 0,
                "request_profile": request_profile, "execution_completion": mode,
            }
            if kind == "mixed":
                scenario["secondary_request_profile"] = "long"
            scenarios.append(scenario)
        return {
            "schema_version": "riley.reliability-soak-manifest.v1",
            "contract_id": "fixture",
            "target": {
                "kind": "process", "binary": "/bin/riley", "model_path": "/models/fixture",
                "bind": "127.0.0.1:18080", "completion_path": "/v1/completions",
                "health_path": "/readyz", "metrics_path": "/metrics",
                "shutdown_signal": "TERM",
                "launch_arguments": list(checker.EXPECTED_LAUNCH_ARGUMENTS),
            },
            "thresholds": {
                "sample_interval_ms": 1000, "maximum_sample_gap_ms": 1500,
                "minimum_samples_per_scenario": 2, "plateau_tail_fraction": 1.0,
                "maximum_rss_plateau_growth_bytes": 1024,
                "maximum_rss_slope_bytes_per_hour": 1024,
                "maximum_vram_plateau_growth_bytes": 1024,
                "maximum_vram_slope_bytes_per_hour": 1024,
                "minimum_cancellations": 1, "minimum_disconnects": 1,
                "minimum_overloads": 1, "graceful_shutdown_deadline_ms": 30000,
            },
            "requests": {
                "short": {
                    "model": self.source["model_id"],
                    "prompt": "fixture prompt",
                    "max_tokens": 8,
                    "temperature": 0.0,
                },
                "long": {},
                "near_kv": {},
                "invalid": {},
            },
            "golden": {
                "request_profile": "short", "digest_domain": "completion-text-utf8",
                "generated_sha256": self.golden,
                "provenance_sha256": self.native_correctness_report_sha256,
            },
            "scenarios": scenarios,
        }

    def _event(self, kind: str, scenario_id: str | None, **values: object) -> None:
        self.events.append({
            "schema_version": "riley.reliability-soak-event.v1",
            "sequence": len(self.events) + 1,
            "monotonic_ns": (len(self.events) + 1) * 1_000_000_000,
            "kind": kind, "scenario_id": scenario_id,
            "binding_sha256": self.binding, **values,
        })

    @staticmethod
    def metrics(*, active: int = 0) -> dict[str, object]:
        return {
            "active_requests": active, "waiting_requests": 0, "kv_allocated_blocks": 0,
            "allocation": {"device_live_count": 0, "device_live_bytes": 0, "pinned_live_count": 0, "pinned_live_bytes": 0},
            "batch_shapes": {
                "metrics_degraded": False, "last": None, "bucket_count": 0,
                "buckets": [{
                    "dense_rows": 0, "hit_count": 0, "latency_sample_count": 0,
                    "gpu_execution_ns_total": 0, "gpu_execution_ns_average": 0,
                    "gpu_execution_ns_maximum": 0, "gpu_execution_ns_last": 0,
                } for _ in range(10)],
            },
            "counters": {"cancellations": 1, "disconnects": 1, "overloads": 1, "dropped_observations": 0},
        }

    def _sample(self, scenario_id: str | None) -> None:
        stopped = scenario_id is None
        self._event(
            "sample", scenario_id,
            process={
                "pid": 0 if stopped else 123,
                "rss_bytes": 0 if stopped else 1000,
                "hwm_bytes": 0 if stopped else 1200,
                "fd_count": 0 if stopped else 8,
                "thread_count": 0 if stopped else 4,
                "children": [],
            },
            gpu={"vram_bytes": 0 if stopped else 2000},
            metrics=self.metrics(),
            sample_dropped=False,
        )

    def _request(self, scenario_id: str, outcome: str) -> None:
        generated = self.golden if outcome == "success" else None
        scenario = next(
            scenario
            for scenario in self.manifest["scenarios"]
            if scenario["id"] == scenario_id
        )
        action = {
            "invalid": "invalid",
            "overload": "overload",
            "cancelled": "cancel",
            "disconnected": "disconnect",
        }.get(outcome, "normal")
        status = {
            "success": 200,
            "invalid": 400,
            "overload": 429,
            "cancelled": 0,
            "disconnected": 200,
        }[outcome]
        response_bytes = {
            "cancelled": 0,
            "disconnected": checker.DISCONNECT_RESPONSE_BYTES,
        }.get(outcome, 128)
        response_sha256 = (
            checker.EMPTY_SHA256
            if response_bytes == 0
            else digest(f"response body {scenario_id} {outcome}")
        )
        request_profile = scenario["request_profile"]
        request_stream = action == "disconnect"
        request_body = dict(self.manifest["requests"][request_profile])
        if "prompt_repeat" in request_body:
            request_body["prompt"] = (
                request_body["prompt"] * request_body.pop("prompt_repeat")
            )
        request_body["stream"] = request_stream
        self._event(
            "request",
            scenario_id,
            request_id=f"request-{len(self.events)}",
            request_profile=request_profile,
            client_action=action,
            request_stream=request_stream,
            curl_exit_code={"cancel": 28, "disconnect": 23}.get(action, 0),
            request_body_sha256=hashlib.sha256(
                checker._jq_1_6_request_json_bytes(request_body)
            ).hexdigest(),
            response_body_sha256=response_sha256,
            response_bytes=response_bytes,
            outcome=outcome,
            http_status=status,
            latency_ms=1.0,
            generated_sha256=generated,
        )

    def _build_events(self) -> None:
        self._event("run_start", None)
        for scenario_id, kind, mode in self.KINDS:
            self._event("scenario_start", scenario_id, execution_completion=mode)
            self._sample(scenario_id)
            self._sample(scenario_id)
            if kind == "invalid":
                self._request(scenario_id, "invalid")
            elif kind == "overload":
                self._request(scenario_id, "overload")
            elif kind == "cancellation-disconnect":
                self._request(scenario_id, "cancelled")
                self._request(scenario_id, "disconnected")
            else:
                self._request(scenario_id, "success")
            if kind == "graceful-restart":
                self._event("restart", scenario_id, graceful=True, exit_code=0, elapsed_ms=100.0, before_generated_sha256=self.golden, after_generated_sha256=self.golden)
            self._event("scenario_end", scenario_id, status="success")
        self._sample(None)
        self._event("run_end", None, status="success")

    def write(self) -> None:
        self.manifest_path = self.root / "manifest.json"
        self.manifest_path.write_text(json.dumps(self.manifest, sort_keys=True) + "\n")
        manifest_sha = hashlib.sha256(self.manifest_path.read_bytes()).hexdigest()
        run = {
            "schema_version": "riley.reliability-soak-run.v1",
            "run_id": "soak-20260826T000004Z-aaaaaaaaaaaa",
            "manifest_sha256": manifest_sha, "binding_sha256": self.binding, "source": self.source,
            "target": {"kind": "process", "pid": 123, "image_id": "sha256:" + "d" * 64, "command_sha256": "1" * 64},
            "started_at_utc": "2026-08-26T00:00:04Z",
        }
        (self.run_directory / "run.json").write_text(json.dumps(run, sort_keys=True) + "\n")
        (self.run_directory / "events.jsonl").write_text("".join(json.dumps(event, sort_keys=True) + "\n" for event in self.events))
        self._write_runtime_receipts(run)

    def _container_environment(self) -> list[str]:
        environment = dict(item.split("=", 1) for item in self.test_image_environment)
        environment.update(
            {
                "RILEY_SOAK_MANIFEST": "/run-input/reliability-soak-v1.json",
                "RILEY_SOAK_OUTPUT": "/evidence/run",
                "RILEY_SOURCE_REVISION": self.source["git_commit"],
                "RILEY_SOURCE_ARCHIVE_SHA256": self.source[
                    "source_archive_sha256"
                ],
                "RILEY_BINARY_SHA256": self.source["binary_sha256"],
                "RILEY_IMAGE_SHA256": self.source["image_sha256"],
                "RILEY_MODEL_SHA256": self.source["model_sha256"],
                "RILEY_MODEL_ID": self.source["model_id"],
                "RILEY_MODEL_REVISION": self.source["model_revision"],
                "RILEY_SOAK_FINAL_METRICS_JSON": "/evidence/final-metrics.json",
                "RILEY_SOAK_BINARY": "/opt/riley/bin/riley",
                "RILEY_SOAK_MODEL_PATH": "/model",
                "RILEY_SOAK_BIND": "127.0.0.1:18080",
                "NVIDIA_DRIVER_CAPABILITIES": "compute,utility",
                "ALL_PROXY": "",
                "FTP_PROXY": "",
                "HTTP_PROXY": "",
                "HTTPS_PROXY": "",
                "NO_PROXY": "",
                "all_proxy": "",
                "ftp_proxy": "",
                "http_proxy": "",
                "https_proxy": "",
                "no_proxy": "",
            }
        )
        return [f"{key}={value}" for key, value in environment.items()]

    def _network_settings(self) -> dict[str, object]:
        none = {
            "Gateway": "",
            "IPAddress": "",
            "IPPrefixLen": 0,
            "IPv6Gateway": "",
            "GlobalIPv6Address": "",
            "GlobalIPv6PrefixLen": 0,
            "MacAddress": "",
        }
        return {
            **none,
            "Ports": None,
            "Networks": {"none": none.copy()},
        }

    def _test_image_labels(self) -> dict[str, str]:
        return {
            "org.riley.reliability-soak.release-image-id": (
                self.release_image_id
            ),
            "org.riley.reliability-soak.source-revision": self.source[
                "git_commit"
            ],
            "org.riley.reliability-soak.source-archive-sha256": self.source[
                "source_archive_sha256"
            ],
            "org.riley.reliability-soak.release-binary-sha256": self.source[
                "binary_sha256"
            ],
        }

    def _container_inspect(self, *, post_run: bool) -> list[dict[str, object]]:
        return [
            {
                "Id": self.container_id,
                "Name": "/" + self.container_name,
                "Image": self.test_image_id,
                "Path": "/opt/riley-soak/ci/run_release_soak.sh",
                "Args": [],
                "Created": "2026-08-26T00:00:02.123456789Z",
                "Config": {
                    "Image": self.test_image_id,
                    "User": "65532:65532",
                    "Entrypoint": ["/opt/riley-soak/ci/run_release_soak.sh"],
                    "Cmd": [],
                    "WorkingDir": "",
                    "Healthcheck": {"Test": ["NONE"]},
                    "Labels": self._test_image_labels(),
                    "Env": self._container_environment(),
                },
                "HostConfig": {
                    "NetworkMode": "none",
                    "PidMode": "host",
                    "IpcMode": "private",
                    "UTSMode": "",
                    "UsernsMode": "",
                    "CgroupnsMode": "private",
                    "Runtime": "runc",
                    "ReadonlyRootfs": True,
                    "PidsLimit": 8192,
                    "Privileged": False,
                    "AutoRemove": False,
                    "PublishAllPorts": False,
                    "PortBindings": {},
                    "RestartPolicy": {"Name": "no", "MaximumRetryCount": 0},
                    "CapDrop": ["ALL"],
                    "CapAdd": None,
                    "Binds": None,
                    "DeviceCgroupRules": None,
                    "Devices": [],
                    "ExtraHosts": None,
                    "GroupAdd": None,
                    "Links": None,
                    "Sysctls": None,
                    "VolumesFrom": None,
                    "SecurityOpt": ["no-new-privileges:true"],
                    "Tmpfs": {
                        "/tmp": "rw,nosuid,nodev,noexec,size=67108864"
                    },
                    "DeviceRequests": [
                        {
                            "Driver": "",
                            "Count": 0,
                            "DeviceIDs": [checker.DESIGNATED_GPU["gpu_uuid"]],
                            "Capabilities": [["gpu"]],
                            "Options": {},
                        }
                    ],
                },
                "Mounts": [
                    {
                        "Type": "bind",
                        "Source": "/fixture/model",
                        "Destination": "/model",
                        "RW": False,
                        "Mode": "",
                        "Propagation": "rprivate",
                    },
                    {
                        "Type": "bind",
                        "Source": "/fixture/manifest.json",
                        "Destination": "/run-input/reliability-soak-v1.json",
                        "RW": False,
                        "Mode": "",
                        "Propagation": "rprivate",
                    },
                    {
                        "Type": "bind",
                        "Source": "/fixture/evidence",
                        "Destination": "/evidence",
                        "RW": True,
                        "Mode": "",
                        "Propagation": "rprivate",
                    },
                ],
                "NetworkSettings": self._network_settings(),
                "RestartCount": 0,
                "State": {
                    "Status": "exited" if post_run else "created",
                    "Running": False,
                    "Paused": False,
                    "Restarting": False,
                    "OOMKilled": False,
                    "Dead": False,
                    "Pid": 0,
                    "ExitCode": 0,
                    "Error": "",
                    "StartedAt": (
                        "2026-08-26T00:00:03.123456789Z"
                        if post_run
                        else "0001-01-01T00:00:00Z"
                    ),
                    "FinishedAt": (
                        "2026-08-26T07:16:03.123456789Z"
                        if post_run
                        else "0001-01-01T00:00:00Z"
                    ),
                },
            }
        ]

    def _write_runtime_receipts(self, run: dict[str, object]) -> None:
        host = {"hostname": checker.DESIGNATED_HOSTNAME, **checker.DESIGNATED_GPU}
        runtime_closure = (
            "/lib64/ld-linux-x86-64.so.2\t/lib64/ld-linux-x86-64.so.2\t"
            f"/usr/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2\t{digest('loader')}\n"
            "libc.so.6\t/lib/x86_64-linux-gnu/libc.so.6\t"
            f"/usr/lib/x86_64-linux-gnu/libc.so.6\t{digest('libc')}\n"
            "libcuda.so.1\tNOT_FOUND\t-\t-\n"
        ).encode("ascii")
        launcher = {
            "schema_version": checker.LAUNCHER_RECEIPT_VERSION,
            "host": host,
            "source": {
                "git_revision": self.source["git_commit"],
                "source_archive_sha256": self.source["source_archive_sha256"],
                "release_binary_sha256": self.source["binary_sha256"],
                "model_tree_sha256": self.source["model_sha256"],
                "manifest_sha256": run["manifest_sha256"],
                "correctness_golden_sha256": hashlib.sha256(
                    self.correctness_golden_path.read_bytes()
                ).hexdigest(),
                "native_correctness_report_sha256": (
                    self.native_correctness_report_sha256
                ),
            },
            "evidence": {
                "run_json_sha256": hashlib.sha256(
                    (self.run_directory / "run.json").read_bytes()
                ).hexdigest(),
                "events_jsonl_sha256": hashlib.sha256(
                    (self.run_directory / "events.jsonl").read_bytes()
                ).hexdigest(),
                "release_runtime_closure_sha256": hashlib.sha256(
                    runtime_closure
                ).hexdigest(),
            },
            "images": {
                "release_image_id": self.release_image_id,
                "test_layer_image_id": self.test_image_id,
            },
            "container": {
                "id": self.container_id,
                "name": self.container_name,
                "exit_code": 0,
            },
        }
        release_image = [
            {
                "Id": self.release_image_id,
                "Os": "linux",
                "Architecture": "amd64",
                "Config": {
                    "User": "65532:65532",
                    "WorkingDir": "",
                    "Labels": {},
                    "Env": self.release_image_environment,
                },
                "RootFS": {"Type": "layers", "Layers": self.release_layers},
            }
        ]
        test_image = [
            {
                "Id": self.test_image_id,
                "Os": "linux",
                "Architecture": "amd64",
                "Config": {
                    "User": "65532:65532",
                    "Entrypoint": ["/opt/riley-soak/ci/run_release_soak.sh"],
                    "Cmd": [],
                    "WorkingDir": "",
                    "Env": self.test_image_environment,
                    "Labels": self._test_image_labels(),
                },
                "RootFS": {"Type": "layers", "Layers": self.test_layers},
            }
        ]
        receipts: dict[str, object] = {
            "launcher-receipt.json": launcher,
            "release-image-inspect.json": release_image,
            "test-layer-image-inspect.json": test_image,
            "container-inspect-pre.json": self._container_inspect(post_run=False),
            "container-inspect-post.json": self._container_inspect(post_run=True),
        }
        gpu = checker.DESIGNATED_GPU
        (self.runtime_receipts_directory / "host-gpu.csv").write_text(
            f"{gpu['gpu_name']}, {gpu['gpu_uuid']}, {gpu['compute_capability']}, "
            f"{gpu['memory_total_mib']}, {gpu['driver_version']}\n",
            encoding="ascii",
        )
        (self.runtime_receipts_directory / "release-runtime-closure.tsv").write_bytes(
            runtime_closure
        )
        for filename, receipt in receipts.items():
            (self.runtime_receipts_directory / filename).write_text(
                json.dumps(receipt, sort_keys=True) + "\n",
                encoding="utf-8",
            )

    def renumber(self) -> None:
        for index, event in enumerate(self.events, 1):
            event["sequence"] = index
            event["monotonic_ns"] = index * 1_000_000_000

    def trusted_arguments(self) -> dict[str, Path]:
        return {
            "runtime_receipts_directory": self.runtime_receipts_directory,
            "correctness_golden": self.correctness_golden_path,
            "native_correctness_report": self.native_correctness_report_path,
        }

    def correctness_arguments(self) -> dict[str, Path]:
        return {
            "correctness_golden": self.correctness_golden_path,
            "native_correctness_report": self.native_correctness_report_path,
        }

    def mutate_receipt(self, filename: str, mutate) -> None:  # type: ignore[no-untyped-def]
        path = self.runtime_receipts_directory / filename
        value = json.loads(path.read_text(encoding="utf-8"))
        mutate(value)
        path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


class ReliabilitySoakCheckerTests(unittest.TestCase):
    @staticmethod
    def mutate_container_pair(fixture: SoakFixture, mutate) -> None:  # type: ignore[no-untyped-def]
        for filename in (
            "container-inspect-pre.json",
            "container-inspect-post.json",
        ):
            fixture.mutate_receipt(filename, lambda value: mutate(value[0]))

    @staticmethod
    def refresh_launcher_evidence(fixture: SoakFixture) -> None:
        fixture.mutate_receipt(
            "launcher-receipt.json",
            lambda value: value.__setitem__(
                "evidence",
                {
                    "run_json_sha256": hashlib.sha256(
                        (fixture.run_directory / "run.json").read_bytes()
                    ).hexdigest(),
                    "events_jsonl_sha256": hashlib.sha256(
                        (fixture.run_directory / "events.jsonl").read_bytes()
                    ).hexdigest(),
                    "release_runtime_closure_sha256": hashlib.sha256(
                        (
                            fixture.runtime_receipts_directory
                            / "release-runtime-closure.tsv"
                        ).read_bytes()
                    ).hexdigest(),
                },
            ),
        )

    def evaluate(self, mutate=None):  # type: ignore[no-untyped-def]
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        fixture = SoakFixture(Path(directory.name))
        if mutate is not None:
            mutate(fixture)
            fixture.renumber()
            fixture.write()
        fixture_contract = checker._normalized_manifest_sha256(fixture.manifest)
        with mock.patch.object(
            checker,
            "REVIEWED_MANIFEST_TEMPLATE_CANONICAL_SHA256",
            fixture_contract,
        ):
            return checker.evaluate(
                fixture.manifest_path,
                fixture.run_directory,
                **fixture.trusted_arguments(),
            )

    def evaluate_receipt(self, mutate):  # type: ignore[no-untyped-def]
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        fixture = SoakFixture(Path(directory.name))
        mutate(fixture)
        fixture_contract = checker._normalized_manifest_sha256(fixture.manifest)
        with mock.patch.object(
            checker,
            "REVIEWED_MANIFEST_TEMPLATE_CANONICAL_SHA256",
            fixture_contract,
        ):
            return checker.evaluate(
                fixture.manifest_path,
                fixture.run_directory,
                **fixture.trusted_arguments(),
            )

    def test_short_complete_fixture_passes(self) -> None:
        report = self.evaluate()
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(
            report["schema_version"],
            "riley.reliability-soak-report.v2",
        )
        self.assertEqual(len(report["scenario_summaries"]), 10)
        self.assertEqual(
            report["bindings"]["trusted_correctness"]["generated_text_sha256"],
            digest("golden completion"),
        )
        runtime = report["bindings"]["runtime_provenance"]
        self.assertEqual(runtime["hostname"], checker.DESIGNATED_HOSTNAME)
        self.assertEqual(runtime["gpu_uuid"], checker.DESIGNATED_GPU["gpu_uuid"])
        self.assertRegex(runtime["launcher_receipt_sha256"], r"^[0-9a-f]{64}$")
        self.assertRegex(runtime["release_runtime_closure_sha256"], r"^[0-9a-f]{64}$")

    def test_held_fd_streaming_raw_replay_matches_legacy_without_legacy_calls(self) -> None:
        """The Gate E path must not reopen or fully buffer the raw archive."""

        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        fixture = SoakFixture(Path(directory.name))
        raw_evidence = Path(directory.name) / "bound-soak.tar"
        contract = checker._normalized_manifest_sha256(fixture.manifest)
        with mock.patch.object(
            checker,
            "REVIEWED_MANIFEST_TEMPLATE_CANONICAL_SHA256",
            contract,
        ):
            expected = checker.evaluate(
                fixture.manifest_path,
                fixture.run_directory,
                **fixture.trusted_arguments(),
            )
            self.assertTrue(expected["passed"], expected)
            checker.package_raw_evidence(
                fixture.manifest_path,
                fixture.run_directory,
                raw_evidence,
                **fixture.trusted_arguments(),
            )
            raw = raw_evidence.read_bytes()
            raw_sha256 = hashlib.sha256(raw).hexdigest()
            raw_fd = os.open(raw_evidence, os.O_RDONLY)
            try:
                with mock.patch.object(
                    checker,
                    "evaluate",
                    side_effect=AssertionError("legacy evaluate must not be called"),
                ), mock.patch.object(
                    checker,
                    "replay_raw_evidence_archive",
                    side_effect=AssertionError("legacy raw replay must not be called"),
                ):
                    replay = checker.replay_bound_raw_evidence_fd(
                        raw_fd,
                        expected_sha256=raw_sha256,
                        expected_byte_length=len(raw),
                        correctness_golden_raw=fixture.correctness_golden_path.read_bytes(),
                        native_correctness_report_raw=(
                            fixture.native_correctness_report_path.read_bytes()
                        ),
                    )
                self.assertEqual(replay["report"], expected)
                validated = checker.validate_bound_reliability_soak_evidence(
                    expected,
                    raw_fd,
                    correctness_golden_raw=fixture.correctness_golden_path.read_bytes(),
                    native_correctness_report_raw=(
                        fixture.native_correctness_report_path.read_bytes()
                    ),
                    source_revision=fixture.source["git_commit"],
                    source_archive_sha256=fixture.source["source_archive_sha256"],
                    release_binary_sha256=fixture.source["binary_sha256"],
                    release_image_id="sha256:" + fixture.source["image_sha256"],
                    candidate_id="riley-0.1.0-rc1",
                    correctness_golden_sha256=hashlib.sha256(
                        fixture.correctness_golden_path.read_bytes()
                    ).hexdigest(),
                    native_correctness_report_sha256=hashlib.sha256(
                        fixture.native_correctness_report_path.read_bytes()
                    ).hexdigest(),
                    raw_evidence_sha256=raw_sha256,
                    raw_evidence_byte_length=len(raw),
                    model_tree_sha256=fixture.source["model_sha256"],
                )
            finally:
                os.close(raw_fd)
        self.assertEqual(validated["report"], expected)

    def test_held_fd_streaming_rejects_bound_length_before_archive_parse(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            raw_path = Path(temporary) / "raw.tar"
            raw_path.write_bytes(b"not-a-canonical-archive")
            raw_fd = os.open(raw_path, os.O_RDONLY)
            try:
                with self.assertRaises(checker.InputError) as raised:
                    checker.replay_bound_raw_evidence_fd(
                        raw_fd,
                        expected_sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest(),
                        expected_byte_length=raw_path.stat().st_size + 1,
                        correctness_golden_raw=b"{}",
                        native_correctness_report_raw=b"{}",
                    )
            finally:
                os.close(raw_fd)
        self.assertIn("length differs", str(raised.exception))

    def test_held_fd_streaming_rejects_noncanonical_checksum_manifest(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        fixture = SoakFixture(Path(directory.name))
        original = Path(directory.name) / "original.tar"
        mutated = Path(directory.name) / "mutated.tar"
        contract = checker._normalized_manifest_sha256(fixture.manifest)
        with mock.patch.object(
            checker,
            "REVIEWED_MANIFEST_TEMPLATE_CANONICAL_SHA256",
            contract,
        ):
            checker.package_raw_evidence(
                fixture.manifest_path,
                fixture.run_directory,
                original,
                **fixture.trusted_arguments(),
            )
            payloads = read_raw_tar(original)
            payloads["SHA256SUMS"] = b"0" * 64 + b"  events.jsonl\n"
            write_raw_tar(mutated, payloads)
            raw = mutated.read_bytes()
            raw_fd = os.open(mutated, os.O_RDONLY)
            try:
                with self.assertRaises(checker.InputError) as raised:
                    checker.replay_bound_raw_evidence_fd(
                        raw_fd,
                        expected_sha256=hashlib.sha256(raw).hexdigest(),
                        expected_byte_length=len(raw),
                        correctness_golden_raw=fixture.correctness_golden_path.read_bytes(),
                        native_correctness_report_raw=(
                            fixture.native_correctness_report_path.read_bytes()
                        ),
                    )
            finally:
                os.close(raw_fd)
        self.assertIn("SHA256SUMS", str(raised.exception))

    def test_held_fd_streaming_rejects_oversized_event_line_without_buffering_it(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        fixture = SoakFixture(Path(directory.name))
        original = Path(directory.name) / "original.tar"
        mutated = Path(directory.name) / "oversized-event.tar"
        contract = checker._normalized_manifest_sha256(fixture.manifest)
        with mock.patch.object(
            checker,
            "REVIEWED_MANIFEST_TEMPLATE_CANONICAL_SHA256",
            contract,
        ):
            checker.package_raw_evidence(
                fixture.manifest_path,
                fixture.run_directory,
                original,
                **fixture.trusted_arguments(),
            )
            payloads = read_raw_tar(original)
            payloads["events.jsonl"] = (
                b"{" + b"a" * checker.MAX_BOUND_EVENT_LINE_BYTES + b"}\n"
            )
            launcher = json.loads(payloads["launcher-receipt.json"])
            launcher["evidence"]["events_jsonl_sha256"] = hashlib.sha256(
                payloads["events.jsonl"]
            ).hexdigest()
            payloads["launcher-receipt.json"] = (
                json.dumps(launcher, sort_keys=True) + "\n"
            ).encode("utf-8")
            payloads["SHA256SUMS"] = b"".join(
                f"{hashlib.sha256(payloads[name]).hexdigest()}  {name}\n".encode("ascii")
                for name in checker.RAW_ARCHIVE_PAYLOADS
            )
            write_raw_tar(mutated, payloads)
            raw = mutated.read_bytes()
            raw_fd = os.open(mutated, os.O_RDONLY)
            try:
                with self.assertRaises(checker.InputError) as raised:
                    checker.replay_bound_raw_evidence_fd(
                        raw_fd,
                        expected_sha256=hashlib.sha256(raw).hexdigest(),
                        expected_byte_length=len(raw),
                        correctness_golden_raw=fixture.correctness_golden_path.read_bytes(),
                        native_correctness_report_raw=(
                            fixture.native_correctness_report_path.read_bytes()
                        ),
                    )
            finally:
                os.close(raw_fd)
        self.assertIn("bounded line limit", str(raised.exception))

    def test_remote_jq_1_6_integral_request_bytes_are_replayed_exactly(self) -> None:
        request = {
            "model": "fixture/model",
            "prompt": "fixture prompt",
            "max_tokens": 8,
            "temperature": 0.0,
            "stream": False,
        }
        self.assertEqual(
            checker._jq_1_6_request_json_bytes(request),
            b'{"max_tokens":8,"model":"fixture/model","prompt":"fixture prompt",'
            b'"stream":false,"temperature":0}',
        )
        report = self.evaluate()
        self.assertTrue(report["passed"], report)

    def test_trusted_correctness_inputs_are_mandatory(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        fixture = SoakFixture(Path(directory.name))
        raw_evidence = Path(directory.name) / "soak.evidence.tar"
        missing_runtime_evidence = Path(directory.name) / "missing-runtime.tar"
        fixture_contract = checker._normalized_manifest_sha256(fixture.manifest)
        with mock.patch.object(
            checker,
            "REVIEWED_MANIFEST_TEMPLATE_CANONICAL_SHA256",
            fixture_contract,
        ):
            report = checker.evaluate(fixture.manifest_path, fixture.run_directory)
            runtime_report = checker.evaluate(
                fixture.manifest_path,
                fixture.run_directory,
                **fixture.correctness_arguments(),
            )
            with self.assertRaisesRegex(checker.InputError, "--correctness-golden"):
                checker.package_raw_evidence(
                    fixture.manifest_path,
                    fixture.run_directory,
                    raw_evidence,
                )
            self.assertFalse(raw_evidence.exists())
            with self.assertRaisesRegex(
                checker.InputError, "--runtime-receipts-directory"
            ):
                checker.package_raw_evidence(
                    fixture.manifest_path,
                    fixture.run_directory,
                    missing_runtime_evidence,
                    **fixture.correctness_arguments(),
                )
            self.assertFalse(missing_runtime_evidence.exists())
            checker.package_raw_evidence(
                fixture.manifest_path,
                fixture.run_directory,
                raw_evidence,
                **fixture.trusted_arguments(),
            )
            replay = checker.replay_raw_evidence_archive(raw_evidence)
        self.assertFalse(report["passed"])
        self.assertIn("--correctness-golden", report["errors"][0])
        self.assertFalse(runtime_report["passed"])
        self.assertIn("--runtime-receipts-directory", runtime_report["errors"][0])
        self.assertFalse(replay["report"]["passed"])
        self.assertIn("--correctness-golden", replay["report"]["errors"][0])

    def test_runtime_receipts_fail_closed_on_missing_or_tampered_inputs(self) -> None:
        missing = self.evaluate_receipt(
            lambda fixture: (
                fixture.runtime_receipts_directory / "launcher-receipt.json"
            ).unlink()
        )
        self.assertEqual(missing["status"], "error")
        self.assertIn("launcher-receipt.json", missing["errors"][0])

        def tamper_host(fixture: SoakFixture) -> None:
            path = fixture.runtime_receipts_directory / "host-gpu.csv"
            path.write_bytes(path.read_bytes() + b"\n")

        tampered = self.evaluate_receipt(tamper_host)
        self.assertEqual(tampered["status"], "error")
        self.assertIn("exact designated single GPU row", tampered["errors"][0])

        legacy = self.evaluate_receipt(
            lambda fixture: fixture.mutate_receipt(
                "launcher-receipt.json",
                lambda value: value.__setitem__(
                    "schema_version",
                    "riley.reliability-soak-launcher-receipt.v1",
                ),
            )
        )
        self.assertEqual(legacy["status"], "error")
        self.assertIn(checker.LAUNCHER_RECEIPT_VERSION, legacy["errors"][0])

    def test_runtime_closure_receipt_rejects_addresses_and_noncanonical_order(self) -> None:
        def replace_closure(fixture: SoakFixture, raw: bytes) -> None:
            path = fixture.runtime_receipts_directory / "release-runtime-closure.tsv"
            path.write_bytes(raw)
            fixture.mutate_receipt(
                "launcher-receipt.json",
                lambda receipt: receipt["evidence"].__setitem__(
                    "release_runtime_closure_sha256", hashlib.sha256(raw).hexdigest()
                ),
            )

        valid_rows = (
            "/lib64/ld-linux-x86-64.so.2\t/lib64/ld-linux-x86-64.so.2\t"
            f"/usr/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2\t{digest('loader')}\n"
            "libc.so.6\t/lib/x86_64-linux-gnu/libc.so.6\t"
            f"/usr/lib/x86_64-linux-gnu/libc.so.6\t{digest('libc')}\n"
            "libcuda.so.1\tNOT_FOUND\t-\t-\n"
        ).encode("ascii").splitlines(keepends=True)
        mutations = {
            "ASLR address": (
                valid_rows[0].rstrip(b"\n")
                + b"\t(0x1234)\n"
                + b"".join(valid_rows[1:])
            ),
            "unsorted rows": valid_rows[1] + valid_rows[0] + valid_rows[2],
            "relative target": valid_rows[0].replace(
                b"/usr/lib/x86_64-linux-gnu/ld-linux", b"usr/lib/x86_64-linux-gnu/ld-linux"
            ) + b"".join(valid_rows[1:]),
            "unresolved row with digest": (
                valid_rows[0]
                + valid_rows[1]
                + valid_rows[2].replace(
                    b"NOT_FOUND\t-\t-",
                    f"NOT_FOUND\t-\t{digest('missing')}".encode("ascii"),
                )
            ),
            "unresolved absolute dependency": (
                valid_rows[0]
                + valid_rows[1]
                + valid_rows[2].replace(b"libcuda.so.1", b"/libcuda.so.1")
            ),
            "unexpected unresolved soname": (
                valid_rows[0]
                + valid_rows[1]
                + valid_rows[2].replace(b"libcuda.so.1", b"libmissing.so.1")
            ),
            "missing unresolved libcuda": valid_rows[0] + valid_rows[1],
            "duplicate unresolved libcuda": b"".join(valid_rows)
            + b"libcuda.so.1\tNOT_FOUND\t-\t-\n",
        }
        for name, raw in mutations.items():
            with self.subTest(name=name):
                report = self.evaluate_receipt(
                    lambda fixture, value=raw: replace_closure(fixture, value)
                )
                self.assertEqual(report["status"], "error", report)
                self.assertIn("release-runtime-closure.tsv", report["errors"][0])

    def test_launcher_receipt_hashes_exact_exported_run_and_event_bytes(self) -> None:
        def alter_run_bytes(fixture: SoakFixture) -> None:
            path = fixture.run_directory / "run.json"
            path.write_bytes(path.read_bytes() + b" ")

        def alter_event_bytes(fixture: SoakFixture) -> None:
            path = fixture.run_directory / "events.jsonl"
            raw = path.read_bytes()
            path.write_bytes(raw.replace(b"\n", b" \n", 1))

        for name, mutation in (
            ("run.json", alter_run_bytes),
            ("events.jsonl", alter_event_bytes),
        ):
            with self.subTest(name=name):
                report = self.evaluate_receipt(mutation)
                self.assertEqual(report["status"], "error", report)
                self.assertIn(
                    "exact exported run.json/events.jsonl bytes and runtime closure",
                    report["errors"][0],
                )

    def test_run_timestamp_cannot_be_spliced_from_container_lifecycle(self) -> None:
        def splice_run_timestamp(fixture: SoakFixture) -> None:
            path = fixture.run_directory / "run.json"
            run = json.loads(path.read_text(encoding="utf-8"))
            run["started_at_utc"] = "2099-12-31T23:59:59Z"
            run["run_id"] = "soak-20991231T235959Z-aaaaaaaaaaaa"
            path.write_text(json.dumps(run, sort_keys=True) + "\n", encoding="utf-8")
            self.refresh_launcher_evidence(fixture)

        report = self.evaluate_receipt(splice_run_timestamp)
        self.assertEqual(report["status"], "error", report)
        self.assertIn("Docker StartedAt..FinishedAt", report["errors"][0])

    def test_run_id_stamp_must_equal_strict_started_timestamp(self) -> None:
        def mismatch_run_id(fixture: SoakFixture) -> None:
            path = fixture.run_directory / "run.json"
            run = json.loads(path.read_text(encoding="utf-8"))
            run["run_id"] = "soak-20260826T000005Z-aaaaaaaaaaaa"
            path.write_text(json.dumps(run, sort_keys=True) + "\n", encoding="utf-8")
            self.refresh_launcher_evidence(fixture)

        report = self.evaluate_receipt(mismatch_run_id)
        self.assertEqual(report["status"], "error", report)
        self.assertIn("exact started_at_utc stamp", report["errors"][0])

    def test_run_timestamp_must_be_valid_strict_utc(self) -> None:
        def invalid_timestamp(fixture: SoakFixture) -> None:
            path = fixture.run_directory / "run.json"
            run = json.loads(path.read_text(encoding="utf-8"))
            run["started_at_utc"] = "2026-02-30T00:00:04Z"
            run["run_id"] = "soak-20260230T000004Z-aaaaaaaaaaaa"
            path.write_text(json.dumps(run, sort_keys=True) + "\n", encoding="utf-8")
            self.refresh_launcher_evidence(fixture)

        report = self.evaluate_receipt(invalid_timestamp)
        self.assertEqual(report["status"], "error", report)
        self.assertIn("valid Gregorian UTC timestamp", report["errors"][0])

    def test_container_name_stamp_is_strict_and_immediately_precedes_create(self) -> None:
        names = {
            "invalid date": "riley-soak-aaaaaaaaaaaa-20260230T000000Z",
            "after create": "riley-soak-aaaaaaaaaaaa-20260826T000003Z",
            "too early": "riley-soak-aaaaaaaaaaaa-20260825T230000Z",
        }
        for name, container_name in names.items():
            with self.subTest(name=name):
                def mutate(fixture: SoakFixture, value: str = container_name) -> None:
                    fixture.mutate_receipt(
                        "launcher-receipt.json",
                        lambda receipt: receipt["container"].__setitem__("name", value),
                    )
                    self.mutate_container_pair(
                        fixture,
                        lambda container: container.__setitem__("Name", f"/{value}"),
                    )

                report = self.evaluate_receipt(mutate)
                self.assertEqual(report["status"], "error", report)
                self.assertIn("launcher-receipt.json.container.name", report["errors"][0])

    def test_runtime_receipt_security_and_source_contract_is_fail_closed(self) -> None:
        mutations = {
            "wrong image": lambda fixture: fixture.mutate_receipt(
                "test-layer-image-inspect.json",
                lambda value: value[0].__setitem__(
                    "Id", "sha256:" + digest("substituted image")
                ),
            ),
            "wrong rootfs": lambda fixture: fixture.mutate_receipt(
                "test-layer-image-inspect.json",
                lambda value: value[0]["RootFS"]["Layers"].__setitem__(
                    0, "sha256:" + digest("unrelated rootfs")
                ),
            ),
            "wrong user": lambda fixture: fixture.mutate_receipt(
                "release-image-inspect.json",
                lambda value: value[0]["Config"].__setitem__("User", "0:0"),
            ),
            "wrong network": lambda fixture: fixture.mutate_receipt(
                "container-inspect-pre.json",
                lambda value: value[0]["HostConfig"].__setitem__(
                    "NetworkMode", "bridge"
                ),
            ),
            "wrong PID mode": lambda fixture: fixture.mutate_receipt(
                "container-inspect-pre.json",
                lambda value: value[0]["HostConfig"].__setitem__(
                    "PidMode", "private"
                ),
            ),
            "wrong GPU": lambda fixture: fixture.mutate_receipt(
                "container-inspect-pre.json",
                lambda value: value[0]["HostConfig"]["DeviceRequests"][0].__setitem__(
                    "DeviceIDs", ["GPU-00000000-0000-0000-0000-000000000000"]
                ),
            ),
            "wrong mount": lambda fixture: fixture.mutate_receipt(
                "container-inspect-pre.json",
                lambda value: value[0]["Mounts"][0].__setitem__("RW", True),
            ),
            "nonzero exit": lambda fixture: fixture.mutate_receipt(
                "container-inspect-post.json",
                lambda value: value[0]["State"].__setitem__("ExitCode", 1),
            ),
            "OOM": lambda fixture: fixture.mutate_receipt(
                "container-inspect-post.json",
                lambda value: value[0]["State"].__setitem__("OOMKilled", True),
            ),
            "restart": lambda fixture: fixture.mutate_receipt(
                "container-inspect-post.json",
                lambda value: value[0].__setitem__("RestartCount", 1),
            ),
            "wrong labels": lambda fixture: fixture.mutate_receipt(
                "test-layer-image-inspect.json",
                lambda value: value[0]["Config"]["Labels"].__setitem__(
                    "org.riley.reliability-soak.source-revision", "9" * 40
                ),
            ),
            "wrong source": lambda fixture: fixture.mutate_receipt(
                "launcher-receipt.json",
                lambda value: value["source"].__setitem__(
                    "source_archive_sha256", digest("substituted source")
                ),
            ),
        }
        for name, mutation in mutations.items():
            with self.subTest(name=name):
                report = self.evaluate_receipt(mutation)
                self.assertEqual(report["status"], "error", report)
                self.assertFalse(report["passed"])

    def test_runtime_environment_inheritance_and_overrides_are_fail_closed(self) -> None:
        def append_environment(
            fixture: SoakFixture, filename: str, setting: str
        ) -> None:
            fixture.mutate_receipt(
                filename,
                lambda value: value[0]["Config"]["Env"].append(setting),
            )

        def replace_environment(
            fixture: SoakFixture,
            filename: str,
            name: str,
            setting: str,
        ) -> None:
            def replace(value):  # type: ignore[no-untyped-def]
                environment = value[0]["Config"]["Env"]
                matches = [
                    index
                    for index, item in enumerate(environment)
                    if item.startswith(f"{name}=")
                ]
                if len(matches) != 1:
                    raise AssertionError(f"expected one fixture environment key {name}")
                environment[matches[0]] = f"{name}={setting}"

            fixture.mutate_receipt(filename, replace)

        for forbidden in (
            "BASH_ENV",
            "BASH_FUNC_hook%%",
            "ENV",
            "GLIBC_TUNABLES",
            "LD_PRELOAD",
            "LD_AUDIT",
            "LD_DEBUG",
            "HOME",
            "CURL_HOME",
            "XDG_CONFIG_HOME",
        ):
            with self.subTest(scope="release image", forbidden=forbidden):
                report = self.evaluate_receipt(
                    lambda fixture, setting=f"{forbidden}=/unreviewed": (
                        append_environment(
                            fixture, "release-image-inspect.json", setting
                        )
                    )
                )
                self.assertEqual(report["status"], "error", report)
                self.assertIn("forbidden shell/loader", report["errors"][0])

            with self.subTest(scope="test image", forbidden=forbidden):
                report = self.evaluate_receipt(
                    lambda fixture, setting=f"{forbidden}=/unreviewed": (
                        append_environment(
                            fixture, "test-layer-image-inspect.json", setting
                        )
                    )
                )
                self.assertEqual(report["status"], "error", report)
                self.assertIn("forbidden shell/loader", report["errors"][0])

            with self.subTest(scope="container", forbidden=forbidden):
                def mutate_container(
                    fixture: SoakFixture, setting: str = f"{forbidden}=/unreviewed"
                ) -> None:
                    self.mutate_container_pair(
                        fixture,
                        lambda container: container["Config"]["Env"].append(
                            setting
                        ),
                    )

                report = self.evaluate_receipt(mutate_container)
                self.assertEqual(report["status"], "error", report)
                self.assertIn("forbidden shell/loader", report["errors"][0])

        exact_value_mutations = {
            "PATH": "/unreviewed/shims:/usr/bin:/bin",
            "LD_LIBRARY_PATH": "/unreviewed/lib",
            "NVIDIA_VISIBLE_DEVICES": checker.DESIGNATED_GPU["gpu_uuid"],
            "NVIDIA_DRIVER_CAPABILITIES": "all",
        }
        for name, setting in exact_value_mutations.items():
            with self.subTest(scope="consistent inheritance", name=name):
                def mutate_exact_environment(
                    fixture: SoakFixture,
                    variable: str = name,
                    replacement: str = setting,
                ) -> None:
                    for filename in (
                        "release-image-inspect.json",
                        "test-layer-image-inspect.json",
                        "container-inspect-pre.json",
                        "container-inspect-post.json",
                    ):
                        replace_environment(
                            fixture,
                            filename,
                            variable,
                            replacement,
                        )

                report = self.evaluate_receipt(mutate_exact_environment)
                self.assertEqual(report["status"], "error", report)
                self.assertIn(name, report["errors"][0])
                self.assertIn("exact reviewed release-image value", report["errors"][0])

        def add_unreviewed_image_environment(fixture: SoakFixture) -> None:
            append_environment(
                fixture, "test-layer-image-inspect.json", "SAFE_BUT_UNREVIEWED=1"
            )
            self.mutate_container_pair(
                fixture,
                lambda container: container["Config"]["Env"].append(
                    "SAFE_BUT_UNREVIEWED=1"
                ),
            )

        inherited = self.evaluate_receipt(add_unreviewed_image_environment)
        self.assertEqual(inherited["status"], "error", inherited)
        self.assertIn("exact soak-layer overrides", inherited["errors"][0])

    def test_container_receipt_rejects_consistent_unreviewed_runtime_fields(self) -> None:
        mutations = {
            "healthcheck": lambda container: container["Config"].__setitem__(
                "Healthcheck", {"Test": ["CMD-SHELL", "/unreviewed"]}
            ),
            "path": lambda container: container.__setitem__("Path", "/unreviewed"),
            "devices": lambda container: container["HostConfig"].__setitem__(
                "Devices",
                [
                    {
                        "PathOnHost": "/dev/sda",
                        "PathInContainer": "/dev/sda",
                        "CgroupPermissions": "rwm",
                    }
                ],
            ),
            "device cgroup rules": lambda container: container[
                "HostConfig"
            ].__setitem__("DeviceCgroupRules", ["c 1:3 rwm"]),
            "host IPC": lambda container: container["HostConfig"].__setitem__(
                "IpcMode", "host"
            ),
            "host UTS": lambda container: container["HostConfig"].__setitem__(
                "UTSMode", "host"
            ),
            "host userns": lambda container: container[
                "HostConfig"
            ].__setitem__("UsernsMode", "host"),
            "host cgroupns": lambda container: container[
                "HostConfig"
            ].__setitem__("CgroupnsMode", "host"),
            "supplemental group": lambda container: container[
                "HostConfig"
            ].__setitem__("GroupAdd", ["0"]),
            "sysctl": lambda container: container["HostConfig"].__setitem__(
                "Sysctls", {"kernel.domainname": "unreviewed"}
            ),
            "runtime": lambda container: container["HostConfig"].__setitem__(
                "Runtime", "nvidia"
            ),
            "GPU driver alias": lambda container: container["HostConfig"][
                "DeviceRequests"
            ][0].__setitem__("Driver", "nvidia"),
            "all GPU count": lambda container: container["HostConfig"][
                "DeviceRequests"
            ][0].__setitem__("Count", -1),
            "bind mode": lambda container: container["Mounts"][0].__setitem__(
                "Mode", "z"
            ),
            "bind propagation": lambda container: container["Mounts"][0].__setitem__(
                "Propagation", "rshared"
            ),
        }
        for name, mutation in mutations.items():
            with self.subTest(name=name):
                report = self.evaluate_receipt(
                    lambda fixture, mutate=mutation: self.mutate_container_pair(
                        fixture, mutate
                    )
                )
                self.assertEqual(report["status"], "error", report)
                self.assertFalse(report["passed"])

    def test_container_receipt_requires_real_bounded_execution_timestamps(self) -> None:
        mutations = {
            "missing start": lambda value: value[0]["State"].pop("StartedAt"),
            "missing finish": lambda value: value[0]["State"].pop("FinishedAt"),
            "invalid nanoseconds": lambda value: value[0]["State"].__setitem__(
                "FinishedAt", "2026-08-26T07:16:00.1234567890Z"
            ),
            "reversed": lambda value: value[0]["State"].__setitem__(
                "FinishedAt", "2026-08-25T23:59:59.123456789Z"
            ),
            "short": lambda value: value[0]["State"].__setitem__(
                "FinishedAt", "2026-08-26T07:14:59.123456789Z"
            ),
            "nonzero pid": lambda value: value[0]["State"].__setitem__("Pid", 7),
        }
        for name, mutation in mutations.items():
            with self.subTest(name=name):
                report = self.evaluate_receipt(
                    lambda fixture, mutate=mutation: fixture.mutate_receipt(
                        "container-inspect-post.json", mutate
                    )
                )
                self.assertEqual(report["status"], "error", report)
                self.assertFalse(report["passed"])

        missing_created = self.evaluate_receipt(
            lambda fixture: fixture.mutate_receipt(
                "container-inspect-post.json",
                lambda value: value[0].pop("Created"),
            )
        )
        self.assertEqual(missing_created["status"], "error", missing_created)

        started_pre_run = self.evaluate_receipt(
            lambda fixture: fixture.mutate_receipt(
                "container-inspect-pre.json",
                lambda value: value[0]["State"].__setitem__(
                    "StartedAt", "2026-08-26T00:00:00.123456789Z"
                ),
            )
        )
        self.assertEqual(started_pre_run["status"], "error", started_pre_run)
        self.assertIn("zero start/finish", started_pre_run["errors"][0])

    def test_container_runtime_must_cover_preserved_event_span(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        fixture = SoakFixture(Path(directory.name))
        fixture.events[-1]["monotonic_ns"] = (
            fixture.events[0]["monotonic_ns"] + 26_161_000_000_000
        )
        fixture.write()
        reviewed = checker._normalized_manifest_sha256(fixture.manifest)
        with mock.patch.object(
            checker, "REVIEWED_MANIFEST_TEMPLATE_CANONICAL_SHA256", reviewed
        ):
            report = checker.evaluate(
                fixture.manifest_path,
                fixture.run_directory,
                **fixture.trusted_arguments(),
            )
        self.assertEqual(report["status"], "error", report)
        self.assertIn("shorter than the preserved monotonic event span", report["errors"][0])

    def test_internally_consistent_generated_hash_cannot_self_authorize(self) -> None:
        def mutate(fixture: SoakFixture) -> None:
            substituted = digest("self-authorized completion")
            fixture.manifest["golden"]["generated_sha256"] = substituted
            for event in fixture.events:
                if event["kind"] == "request" and event["outcome"] == "success":
                    event["generated_sha256"] = substituted
                elif event["kind"] == "restart":
                    event["before_generated_sha256"] = substituted
                    event["after_generated_sha256"] = substituted

        report = self.evaluate(mutate)
        self.assertEqual(report["status"], "error")
        self.assertIn("trusted E2E correctness golden", report["errors"][0])

    def test_provenance_must_hash_submitted_native_report(self) -> None:
        def mutate(fixture: SoakFixture) -> None:
            fixture.manifest["golden"]["provenance_sha256"] = digest(
                "self-authorized report"
            )

        report = self.evaluate(mutate)
        self.assertEqual(report["status"], "error")
        self.assertIn("submitted native correctness report", report["errors"][0])

    def test_frozen_v2_correctness_cannot_authorize_v3_soak(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        fixture = SoakFixture(Path(directory.name))
        old_gate = "smollm2-fp32-bf16-native-e0-v2"
        fixture.native_correctness_report["gate_id"] = old_gate
        fixture.native_correctness_report_path.write_text(
            json.dumps(fixture.native_correctness_report, sort_keys=True) + "\n"
        )
        fixture.native_correctness_report_sha256 = hashlib.sha256(
            fixture.native_correctness_report_path.read_bytes()
        ).hexdigest()
        fixture.correctness_golden["correctness_gate_id"] = old_gate
        fixture.correctness_golden["correctness_report_sha256"] = (
            fixture.native_correctness_report_sha256
        )
        fixture.correctness_golden_path.write_text(
            json.dumps(fixture.correctness_golden, sort_keys=True) + "\n"
        )
        fixture.manifest["golden"]["provenance_sha256"] = (
            fixture.native_correctness_report_sha256
        )
        fixture.write()
        reviewed = checker._normalized_manifest_sha256(fixture.manifest)
        with mock.patch.object(
            checker, "REVIEWED_MANIFEST_TEMPLATE_CANONICAL_SHA256", reviewed
        ):
            report = checker.evaluate(
                fixture.manifest_path,
                fixture.run_directory,
                **fixture.trusted_arguments(),
            )
        self.assertEqual(report["status"], "error", report)
        self.assertIn(checker.NATIVE_CORRECTNESS_GATE, report["errors"][0])

    def test_native_calibration_executable_digest_is_required_but_distinct(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        fixture = SoakFixture(Path(directory.name))
        native_digest = fixture.native_correctness_report["bindings"][
            "candidate_executable_sha256"
        ]
        self.assertNotEqual(native_digest, fixture.source["binary_sha256"])

        fixture.native_correctness_report["bindings"][
            "candidate_executable_sha256"
        ] = "not-a-sha256"
        fixture.native_correctness_report_path.write_text(
            json.dumps(fixture.native_correctness_report, sort_keys=True) + "\n"
        )
        fixture.native_correctness_report_sha256 = hashlib.sha256(
            fixture.native_correctness_report_path.read_bytes()
        ).hexdigest()
        fixture.correctness_golden["correctness_report_sha256"] = (
            fixture.native_correctness_report_sha256
        )
        fixture.correctness_golden_path.write_text(
            json.dumps(fixture.correctness_golden, sort_keys=True) + "\n"
        )
        fixture.manifest["golden"]["provenance_sha256"] = (
            fixture.native_correctness_report_sha256
        )
        fixture.write()
        reviewed = checker._normalized_manifest_sha256(fixture.manifest)
        with mock.patch.object(
            checker, "REVIEWED_MANIFEST_TEMPLATE_CANONICAL_SHA256", reviewed
        ):
            report = checker.evaluate(
                fixture.manifest_path,
                fixture.run_directory,
                **fixture.trusted_arguments(),
            )
        self.assertEqual(report["status"], "error", report)
        self.assertIn("candidate_executable_sha256", report["errors"][0])

    def test_golden_request_identity_is_cross_bound(self) -> None:
        def mutate(fixture: SoakFixture) -> None:
            fixture.manifest["requests"]["short"]["prompt"] = "different prompt"

        report = self.evaluate(mutate)
        self.assertEqual(report["status"], "error")
        self.assertIn("trusted E2E correctness golden", report["errors"][0])

    def test_missing_required_scenario_end_fails(self) -> None:
        report = self.evaluate(lambda fixture: fixture.events.pop(next(i for i, event in enumerate(fixture.events) if event["kind"] == "scenario_end")))
        self.assertEqual(report["status"], "error", report)
        self.assertIn("active scenario", report["errors"][0])

    def test_failure_event_fails(self) -> None:
        def mutate(fixture: SoakFixture) -> None:
            fixture.events.insert(-1, {
                "schema_version": "riley.reliability-soak-event.v1", "sequence": 0,
                "monotonic_ns": 0, "kind": "failure", "scenario_id": None,
                "binding_sha256": fixture.binding, "stage": "fixture", "message": "forced",
            })
        report = self.evaluate(mutate)
        self.assertFalse(next(check for check in report["checks"] if check["name"] == "no_failure_events")["passed"])

    def test_execution_completion_claim_must_match_manifest(self) -> None:
        def mutate(fixture: SoakFixture) -> None:
            event = next(event for event in fixture.events if event["scenario_id"] == "rollback-per-operation" and event["kind"] == "scenario_start")
            event["execution_completion"] = "iteration-batch"
        report = self.evaluate(mutate)
        self.assertFalse(next(check for check in report["checks"] if check["name"] == "rollback-per-operation.execution_completion")["passed"])

    def test_request_transport_action_stream_and_exit_code_are_cross_bound(self) -> None:
        def tamper(
            fixture: SoakFixture,
            action: str,
            field: str,
            value: object,
        ) -> None:
            request = next(
                event
                for event in fixture.events
                if event["kind"] == "request"
                and event["client_action"] == action
            )
            request[field] = value

        cases = (
            ("normal-exit", "normal", "curl_exit_code", 28),
            ("invalid-action", "invalid", "client_action", "normal"),
            ("overload-stream", "overload", "request_stream", True),
            ("cancel-action", "cancel", "client_action", "disconnect"),
            ("cancel-stream", "cancel", "request_stream", True),
            ("cancel-exit", "cancel", "curl_exit_code", 23),
            ("disconnect-stream", "disconnect", "request_stream", False),
            ("disconnect-exit", "disconnect", "curl_exit_code", 28),
            ("disconnect-bytes", "disconnect", "response_bytes", 0),
            (
                "request-body",
                "normal",
                "request_body_sha256",
                digest("different exact request body"),
            ),
        )
        for name, action, field, value in cases:
            with self.subTest(name=name):
                report = self.evaluate(
                    lambda fixture, a=action, f=field, v=value: tamper(
                        fixture, a, f, v
                    )
                )
                self.assertEqual(report["status"], "error", report)

    def test_scenarios_must_follow_manifest_order_without_overlap(self) -> None:
        def reorder(fixture: SoakFixture) -> None:
            first_start = next(
                index
                for index, event in enumerate(fixture.events)
                if event["kind"] == "scenario_start"
                and event["scenario_id"] == "steady"
            )
            first_end = next(
                index
                for index, event in enumerate(fixture.events)
                if event["kind"] == "scenario_end"
                and event["scenario_id"] == "steady"
            )
            second_start = next(
                index
                for index, event in enumerate(fixture.events)
                if event["kind"] == "scenario_start"
                and event["scenario_id"] == "burst-idle"
            )
            second_end = next(
                index
                for index, event in enumerate(fixture.events)
                if event["kind"] == "scenario_end"
                and event["scenario_id"] == "burst-idle"
            )
            first = fixture.events[first_start : first_end + 1]
            second = fixture.events[second_start : second_end + 1]
            fixture.events[first_start : second_end + 1] = second + first

        def overlap(fixture: SoakFixture) -> None:
            first_end = next(
                index
                for index, event in enumerate(fixture.events)
                if event["kind"] == "scenario_end"
                and event["scenario_id"] == "steady"
            )
            second_start = next(
                index
                for index, event in enumerate(fixture.events)
                if event["kind"] == "scenario_start"
                and event["scenario_id"] == "burst-idle"
            )
            fixture.events[first_end], fixture.events[second_start] = (
                fixture.events[second_start],
                fixture.events[first_end],
            )

        for name, mutation, message in (
            ("reordered", reorder, "manifest order"),
            ("overlapping", overlap, "overlaps active scenario"),
        ):
            with self.subTest(name=name):
                report = self.evaluate(mutation)
                self.assertEqual(report["status"], "error", report)
                self.assertIn(message, report["errors"][0])

    def test_python_child_fails(self) -> None:
        def mutate(fixture: SoakFixture) -> None:
            sample = next(event for event in fixture.events if event["kind"] == "sample")
            sample["process"]["children"] = [{"pid": 77, "comm": "python3", "executable": "/usr/bin/python3"}]
        report = self.evaluate(mutate)
        self.assertFalse(next(check for check in report["checks"] if check["name"] == "no_python_children")["passed"])

    def test_final_nonzero_allocation_fails(self) -> None:
        def mutate(fixture: SoakFixture) -> None:
            samples = [event for event in fixture.events if event["kind"] == "sample"]
            samples[-1]["metrics"]["allocation"]["device_live_count"] = 1
        report = self.evaluate(mutate)
        self.assertFalse(next(check for check in report["checks"] if check["name"] == "final_quiescence")["passed"])

    def test_final_nonzero_process_or_gpu_state_fails(self) -> None:
        for field, value in (
            ("process.pid", 123),
            ("process.rss_bytes", 4096),
            ("process.children", [{"pid": 77, "comm": "riley", "executable": "/opt/riley/bin/riley"}]),
            ("gpu.vram_bytes", 4096),
        ):
            with self.subTest(field=field):
                def mutate(fixture: SoakFixture, field: str = field, value: object = value) -> None:
                    final = [
                        event
                        for event in fixture.events
                        if event["kind"] == "sample" and event["scenario_id"] is None
                    ][-1]
                    section, key = field.split(".", 1)
                    final[section][key] = value

                report = self.evaluate(mutate)
                check = next(
                    check
                    for check in report["checks"]
                    if check["name"] == "final_quiescence"
                )
                self.assertFalse(check["passed"], report)

    def test_dropped_sample_fails(self) -> None:
        def mutate(fixture: SoakFixture) -> None:
            samples = [event for event in fixture.events if event["scenario_id"] == "steady" and event["kind"] == "sample"]
            samples[1]["sample_dropped"] = True
        report = self.evaluate(mutate)
        self.assertFalse(next(check for check in report["checks"] if check["name"] == "no_dropped_samples")["passed"])

    def test_bounded_request_observation_eviction_is_not_sample_loss(self) -> None:
        def mutate(fixture: SoakFixture) -> None:
            samples = [
                event
                for event in fixture.events
                if event["scenario_id"] == "steady" and event["kind"] == "sample"
            ]
            samples[1]["metrics"]["counters"]["dropped_observations"] = 10_000

        report = self.evaluate(mutate)
        self.assertTrue(
            next(
                check
                for check in report["checks"]
                if check["name"] == "no_dropped_samples"
            )["passed"]
        )

    def test_sample_gap_fails(self) -> None:
        def mutate(fixture: SoakFixture) -> None:
            fixture.manifest["thresholds"]["maximum_sample_gap_ms"] = 500
        report = self.evaluate(mutate)
        self.assertFalse(next(check for check in report["checks"] if check["name"] == "steady.sample_gap_ms")["passed"])

    def test_truncated_scenario_cannot_claim_planned_duration(self) -> None:
        def mutate(fixture: SoakFixture) -> None:
            scenario = next(
                scenario
                for scenario in fixture.manifest["scenarios"]
                if scenario["id"] == "steady"
            )
            scenario["duration_seconds"] = 60

        report = self.evaluate(mutate)
        failed = {check["name"] for check in report["checks"] if not check["passed"]}
        self.assertIn("steady.duration_seconds", failed)
        self.assertIn("steady.sample_coverage_seconds", failed)

    def test_manifest_contract_is_pinned_except_materialized_golden(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        fixture = SoakFixture(Path(directory.name))
        reviewed_digest = checker._normalized_manifest_sha256(fixture.manifest)
        fixture.manifest["thresholds"]["minimum_overloads"] = 2
        fixture.write()
        with mock.patch.object(
            checker,
            "REVIEWED_MANIFEST_TEMPLATE_CANONICAL_SHA256",
            reviewed_digest,
        ):
            report = checker.evaluate(
                fixture.manifest_path,
                fixture.run_directory,
                **fixture.trusted_arguments(),
            )
        self.assertEqual(report["status"], "error")
        self.assertIn("reviewed PR16 soak contract", report["errors"][0])

    def test_checked_in_template_digest_is_reviewed(self) -> None:
        template = json.loads(
            (SCRIPT.parents[1] / "soak/reliability-soak-v1.json").read_text()
        )
        self.assertEqual(
            checker._normalized_manifest_sha256(template),
            checker.REVIEWED_MANIFEST_TEMPLATE_CANONICAL_SHA256,
        )

    def test_manifest_requires_complete_conservative_rollback_command(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        fixture = SoakFixture(Path(directory.name))
        reviewed_digest = checker._normalized_manifest_sha256(fixture.manifest)
        arguments = fixture.manifest["target"]["launch_arguments"]
        index = arguments.index("--residual-rmsnorm")
        del arguments[index : index + 2]
        fixture.write()
        with mock.patch.object(
            checker,
            "REVIEWED_MANIFEST_TEMPLATE_CANONICAL_SHA256",
            reviewed_digest,
        ):
            report = checker.evaluate(
                fixture.manifest_path,
                fixture.run_directory,
                **fixture.trusted_arguments(),
            )
        self.assertEqual(report["status"], "error")
        self.assertIn("exact canonical-v1/completion-mode/separate", report["errors"][0])

    def test_manifest_schema_pins_complete_conservative_command(self) -> None:
        schema = json.loads(
            (
                SCRIPT.parents[1]
                / "schemas/reliability-soak-manifest.schema.json"
            ).read_text()
        )
        self.assertEqual(
            schema["properties"]["target"]["properties"]["launch_arguments"]["const"],
            checker.EXPECTED_LAUNCH_ARGUMENTS,
        )

    def test_report_schema_accepts_only_sha1_or_sha256_git_object_ids(self) -> None:
        schema = json.loads(
            (
                SCRIPT.parents[1]
                / "schemas/reliability-soak-report.schema.json"
            ).read_text()
        )
        pattern = schema["$defs"]["gitRevision"]["pattern"]
        self.assertIsNotNone(re.fullmatch(pattern, "a" * 40))
        self.assertIsNotNone(re.fullmatch(pattern, "b" * 64))
        self.assertIsNotNone(checker.GIT_RE.fullmatch("a" * 40))
        self.assertIsNotNone(checker.GIT_RE.fullmatch("b" * 64))
        self.assertIsNone(re.fullmatch(pattern, "c" * 41))
        self.assertIsNone(re.fullmatch(pattern, "A" * 40))

    def test_schemas_require_stream_hashes_and_strict_run_stamps(self) -> None:
        schema_root = SCRIPT.parents[1] / "schemas"
        report_schema = json.loads(
            (schema_root / "reliability-soak-report.schema.json").read_text()
        )
        runtime = report_schema["$defs"]["runtimeProvenance"]
        self.assertIn("run_json_sha256", runtime["required"])
        self.assertIn("events_jsonl_sha256", runtime["required"])
        self.assertIn("release_runtime_closure_sha256", runtime["required"])

        run_schema = json.loads(
            (schema_root / "reliability-soak-run.schema.json").read_text()
        )
        run_id_pattern = run_schema["properties"]["run_id"]["pattern"]
        timestamp_pattern = run_schema["properties"]["started_at_utc"]["pattern"]
        self.assertIsNotNone(
            re.fullmatch(run_id_pattern, "soak-20260826T000004Z-aaaaaaaaaaaa")
        )
        self.assertIsNotNone(
            re.fullmatch(timestamp_pattern, "2026-08-26T00:00:04Z")
        )
        self.assertIsNone(re.fullmatch(run_id_pattern, "fixture-run"))
        self.assertIsNone(
            re.fullmatch(timestamp_pattern, "2026-08-26T00:00:04+00:00")
        )

        event_schema = json.loads(
            (schema_root / "reliability-soak-event.schema.json").read_text()
        )
        event_rules = {
            rule["properties"]["kind"]["const"]: rule
            for reference in event_schema["oneOf"]
            for rule in [
                event_schema["$defs"][reference["$ref"].rsplit("/", 1)[1]]
            ]
        }
        request_rule = event_rules["request"]
        self.assertTrue(
            {
                "request_profile",
                "client_action",
                "request_stream",
                "curl_exit_code",
                "request_body_sha256",
                "response_body_sha256",
                "response_bytes",
            }
            <= set(request_rule["required"])
        )

    def test_event_schema_closes_every_kind_and_requires_kind_fields(self) -> None:
        event_schema = json.loads(
            (
                SCRIPT.parents[1]
                / "schemas/reliability-soak-event.schema.json"
            ).read_text()
        )
        event_rules = {
            rule["properties"]["kind"]["const"]: rule
            for reference in event_schema["oneOf"]
            for rule in [
                event_schema["$defs"][reference["$ref"].rsplit("/", 1)[1]]
            ]
        }
        common = {
            "schema_version",
            "sequence",
            "monotonic_ns",
            "kind",
            "scenario_id",
            "binding_sha256",
        }
        kind_fields = {
            "run_start": set(),
            "scenario_start": {"execution_completion"},
            "sample": {"process", "gpu", "metrics", "sample_dropped"},
            "request": {
                "request_id",
                "request_profile",
                "client_action",
                "request_stream",
                "curl_exit_code",
                "request_body_sha256",
                "response_body_sha256",
                "response_bytes",
                "outcome",
                "http_status",
                "latency_ms",
                "generated_sha256",
            },
            "restart": {
                "graceful",
                "exit_code",
                "elapsed_ms",
                "before_generated_sha256",
                "after_generated_sha256",
            },
            "scenario_end": {"status"},
            "failure": {"stage", "message"},
            "run_end": {"status"},
        }
        self.assertEqual(set(event_rules), set(kind_fields))
        for kind, fields in kind_fields.items():
            with self.subTest(kind=kind):
                rule = event_rules[kind]
                self.assertIs(rule["additionalProperties"], False)
                self.assertEqual(set(rule["properties"]), common | fields)
                self.assertEqual(set(rule["required"]), common | fields)

        def schema_accepts_shape(event: dict[str, object]) -> bool:
            rule = event_rules.get(str(event.get("kind")))
            if rule is None:
                return False
            fields = set(event)
            return set(rule["required"]) <= fields <= set(rule["properties"])

        def common_event(kind: str, scenario_id: str | None) -> dict[str, object]:
            return {
                "schema_version": "riley.reliability-soak-event.v1",
                "sequence": 1,
                "monotonic_ns": 1,
                "kind": kind,
                "scenario_id": scenario_id,
                "binding_sha256": "a" * 64,
            }

        scenario_start = common_event("scenario_start", "steady")
        scenario_start["execution_completion"] = "iteration-batch"
        self.assertTrue(schema_accepts_shape(scenario_start))
        scenario_start.pop("execution_completion")
        self.assertFalse(schema_accepts_shape(scenario_start))

        failure = common_event("failure", None)
        failure.update({"stage": "request", "message": "failed"})
        self.assertTrue(schema_accepts_shape(failure))
        for missing in ("stage", "message"):
            incomplete_failure = dict(failure)
            incomplete_failure.pop(missing)
            self.assertFalse(schema_accepts_shape(incomplete_failure))

        run_end = common_event("run_end", None)
        run_end["status"] = "success"
        self.assertTrue(schema_accepts_shape(run_end))
        run_end.pop("status")
        self.assertFalse(schema_accepts_shape(run_end))

        run_start = common_event("run_start", None)
        self.assertTrue(schema_accepts_shape(run_start))
        run_start["status"] = "success"
        self.assertFalse(schema_accepts_shape(run_start))

    def test_cancellation_and_overload_must_be_observed(self) -> None:
        def mutate(fixture: SoakFixture) -> None:
            for event in fixture.events:
                if event["kind"] == "request" and event["outcome"] in {"cancelled", "disconnected", "overload"}:
                    event["outcome"] = "success"
                    event["http_status"] = 200
                    event["generated_sha256"] = fixture.golden
        report = self.evaluate(mutate)
        self.assertEqual(report["status"], "error", report)
        self.assertIn("transport contract", report["errors"][0])

    def test_restart_and_rollback_must_match_bound_golden(self) -> None:
        def mutate(fixture: SoakFixture) -> None:
            restart = next(event for event in fixture.events if event["kind"] == "restart")
            restart["after_generated_sha256"] = digest("wrong")
            request = next(event for event in fixture.events if event["scenario_id"] == "rollback-per-operation" and event["kind"] == "request")
            request["generated_sha256"] = digest("also wrong")
        report = self.evaluate(mutate)
        names = {check["name"] for check in report["checks"] if not check["passed"]}
        self.assertIn("graceful_restart_golden_parity", names)
        self.assertIn("rollback_golden_parity", names)

    def test_steady_short_requests_must_match_bound_golden(self) -> None:
        def mutate(fixture: SoakFixture) -> None:
            request = next(
                event
                for event in fixture.events
                if event["scenario_id"] == "steady" and event["kind"] == "request"
            )
            request["generated_sha256"] = digest("drifted completion")

        report = self.evaluate(mutate)
        self.assertFalse(
            next(
                check
                for check in report["checks"]
                if check["name"] == "steady.golden_parity"
            )["passed"]
        )

    def test_source_binding_mismatch_is_an_input_error(self) -> None:
        def mutate(fixture: SoakFixture) -> None:
            fixture.source["binary_sha256"] = "9" * 64
        report = self.evaluate(mutate)
        self.assertEqual(report["status"], "error")
        self.assertIn("binding_sha256", report["errors"][0])

    def test_rss_slope_is_bounded(self) -> None:
        def mutate(fixture: SoakFixture) -> None:
            samples = [event for event in fixture.events if event["scenario_id"] == "steady" and event["kind"] == "sample"]
            samples[1]["process"]["rss_bytes"] = 1_000_000
        report = self.evaluate(mutate)
        self.assertFalse(next(check for check in report["checks"] if check["name"] == "steady.rss_slope_per_hour")["passed"])

    def test_raw_packager_is_deterministic_and_replays_exact_report(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        fixture = SoakFixture(Path(directory.name))
        reviewed_digest = checker._normalized_manifest_sha256(fixture.manifest)
        first = fixture.root / "first.tar"
        second = fixture.root / "second.tar"
        with mock.patch.object(
            checker,
            "REVIEWED_MANIFEST_TEMPLATE_CANONICAL_SHA256",
            reviewed_digest,
        ):
            expected = checker.evaluate(
                fixture.manifest_path,
                fixture.run_directory,
                **fixture.trusted_arguments(),
            )
            first_replay = checker.package_raw_evidence(
                fixture.manifest_path,
                fixture.run_directory,
                first,
                **fixture.trusted_arguments(),
            )
            checker.package_raw_evidence(
                fixture.manifest_path,
                fixture.run_directory,
                second,
                **fixture.trusted_arguments(),
            )
        self.assertEqual(first.read_bytes(), second.read_bytes())
        self.assertEqual(first_replay["report"], expected)
        self.assertEqual(
            first_replay["archive_sha256"],
            hashlib.sha256(first.read_bytes()).hexdigest(),
        )
        self.assertEqual(
            first_replay["report"]["bindings"]["runtime_provenance"][
                "container_inspect_post_sha256"
            ],
            first_replay["container-inspect-post.json_sha256"],
        )
        self.assertEqual(
            first_replay["report"]["bindings"]["runtime_provenance"][
                "run_json_sha256"
            ],
            first_replay["run.json_sha256"],
        )
        self.assertEqual(
            first_replay["report"]["bindings"]["runtime_provenance"][
                "events_jsonl_sha256"
            ],
            first_replay["events.jsonl_sha256"],
        )
        with tarfile.open(first, "r:") as archive:
            self.assertEqual(
                [member.name for member in archive.getmembers()],
                sorted(checker.RAW_ARCHIVE_MEMBERS),
            )

    def test_runtime_receipt_directory_inventory_and_file_types_are_exact(self) -> None:
        for kind in ("extra file", "extra directory", "symlink", "fifo"):
            with self.subTest(kind=kind):
                directory = tempfile.TemporaryDirectory()
                self.addCleanup(directory.cleanup)
                fixture = SoakFixture(Path(directory.name))
                if kind == "extra file":
                    (fixture.runtime_receipts_directory / "failure.json").write_text(
                        '{"failed":true}\n', encoding="utf-8"
                    )
                elif kind == "extra directory":
                    (fixture.runtime_receipts_directory / "unreviewed").mkdir()
                else:
                    target = fixture.runtime_receipts_directory / "host-gpu.csv"
                    target.unlink()
                    if kind == "symlink":
                        target.symlink_to(fixture.correctness_golden_path)
                    else:
                        os.mkfifo(target)
                reviewed = checker._normalized_manifest_sha256(fixture.manifest)
                output = fixture.root / f"rejected-{kind.replace(' ', '-')}.tar"
                with mock.patch.object(
                    checker,
                    "REVIEWED_MANIFEST_TEMPLATE_CANONICAL_SHA256",
                    reviewed,
                ):
                    report = checker.evaluate(
                        fixture.manifest_path,
                        fixture.run_directory,
                        **fixture.trusted_arguments(),
                    )
                    with self.assertRaises(checker.InputError):
                        checker.package_raw_evidence(
                            fixture.manifest_path,
                            fixture.run_directory,
                            output,
                            **fixture.trusted_arguments(),
                        )
                self.assertEqual(report["status"], "error", report)
                self.assertFalse(output.exists())

    def test_raw_packager_rejects_receipt_inventory_mutation_after_open(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        fixture = SoakFixture(Path(directory.name))
        reviewed = checker._normalized_manifest_sha256(fixture.manifest)
        output = fixture.root / "mutated-inventory.tar"
        original_stream = checker._stream_sha256
        mutated = False

        def add_extra_after_first_hash(handle):  # type: ignore[no-untyped-def]
            nonlocal mutated
            result = original_stream(handle)
            if not mutated:
                mutated = True
                (fixture.runtime_receipts_directory / "late-extra.json").write_text(
                    '{}\n', encoding="utf-8"
                )
            return result

        with (
            mock.patch.object(
                checker,
                "REVIEWED_MANIFEST_TEMPLATE_CANONICAL_SHA256",
                reviewed,
            ),
            mock.patch.object(
                checker,
                "_stream_sha256",
                side_effect=add_extra_after_first_hash,
            ),
        ):
            with self.assertRaisesRegex(
                checker.InputError, "exact receipt inventory"
            ):
                checker.package_raw_evidence(
                    fixture.manifest_path,
                    fixture.run_directory,
                    output,
                    **fixture.trusted_arguments(),
                )
        self.assertTrue(mutated)
        self.assertFalse(output.exists())

    def test_raw_replay_rejects_path_replacement_after_hash(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        fixture = SoakFixture(Path(directory.name))
        reviewed = checker._normalized_manifest_sha256(fixture.manifest)
        first = fixture.root / "first-held.tar"
        second = fixture.root / "second-substitution.tar"
        raced = fixture.root / "raced.tar"
        with mock.patch.object(
            checker,
            "REVIEWED_MANIFEST_TEMPLATE_CANONICAL_SHA256",
            reviewed,
        ):
            checker.package_raw_evidence(
                fixture.manifest_path,
                fixture.run_directory,
                first,
                **fixture.trusted_arguments(),
            )
            fixture.mutate_receipt(
                "release-image-inspect.json",
                lambda value: value[0].__setitem__("Comment", "second archive"),
            )
            checker.package_raw_evidence(
                fixture.manifest_path,
                fixture.run_directory,
                second,
                **fixture.trusted_arguments(),
            )
            first_sha256 = hashlib.sha256(first.read_bytes()).hexdigest()
            second_sha256 = hashlib.sha256(second.read_bytes()).hexdigest()
            shutil.copyfile(first, raced)
            original_stream = checker._stream_sha256
            swapped = False

            def swap_path_after_first_hash(handle):  # type: ignore[no-untyped-def]
                nonlocal swapped
                result = original_stream(handle)
                if not swapped:
                    swapped = True
                    os.replace(second, raced)
                return result

            with mock.patch.object(
                checker, "_stream_sha256", side_effect=swap_path_after_first_hash
            ):
                with self.assertRaisesRegex(
                    checker.InputError, "held evidence file changed"
                ):
                    checker.replay_raw_evidence_archive(
                        raced, **fixture.correctness_arguments()
                    )
        self.assertNotEqual(first_sha256, second_sha256)
        self.assertEqual(hashlib.sha256(raced.read_bytes()).hexdigest(), second_sha256)

    def test_raw_replay_rejects_symlinks_and_fifos_without_blocking(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        fixture = SoakFixture(Path(directory.name))
        reviewed = checker._normalized_manifest_sha256(fixture.manifest)
        original = fixture.root / "original-regular.tar"
        with mock.patch.object(
            checker,
            "REVIEWED_MANIFEST_TEMPLATE_CANONICAL_SHA256",
            reviewed,
        ):
            checker.package_raw_evidence(
                fixture.manifest_path,
                fixture.run_directory,
                original,
                **fixture.trusted_arguments(),
            )
            linked = fixture.root / "linked.tar"
            linked.symlink_to(original)
            with self.assertRaisesRegex(checker.InputError, "regular file"):
                checker.replay_raw_evidence_archive(
                    linked, **fixture.correctness_arguments()
                )
            fifo = fixture.root / "fifo.tar"
            os.mkfifo(fifo)
            with self.assertRaisesRegex(checker.InputError, "regular file"):
                checker.replay_raw_evidence_archive(
                    fifo, **fixture.correctness_arguments()
                )

    def test_raw_replay_rejects_checksum_tampering(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        fixture = SoakFixture(Path(directory.name))
        reviewed_digest = checker._normalized_manifest_sha256(fixture.manifest)
        original = fixture.root / "original.tar"
        tampered = fixture.root / "tampered.tar"
        with mock.patch.object(
            checker,
            "REVIEWED_MANIFEST_TEMPLATE_CANONICAL_SHA256",
            reviewed_digest,
        ):
            checker.package_raw_evidence(
                fixture.manifest_path,
                fixture.run_directory,
                original,
                **fixture.trusted_arguments(),
            )
            payloads = read_raw_tar(original)
            payloads["events.jsonl"] += b"\n"
            write_raw_tar(tampered, payloads)
            with self.assertRaisesRegex(checker.InputError, "SHA256SUMS"):
                checker.replay_raw_evidence_archive(
                    tampered,
                    **fixture.correctness_arguments(),
                )

    def test_raw_replay_rejects_extra_member(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        fixture = SoakFixture(Path(directory.name))
        reviewed_digest = checker._normalized_manifest_sha256(fixture.manifest)
        original = fixture.root / "original.tar"
        expanded = fixture.root / "expanded.tar"
        with mock.patch.object(
            checker,
            "REVIEWED_MANIFEST_TEMPLATE_CANONICAL_SHA256",
            reviewed_digest,
        ):
            checker.package_raw_evidence(
                fixture.manifest_path,
                fixture.run_directory,
                original,
                **fixture.trusted_arguments(),
            )
            payloads = read_raw_tar(original)
            payloads["self-asserted-report.json"] = b'{"passed":true}\n'
            write_raw_tar(expanded, payloads)
            with self.assertRaisesRegex(checker.InputError, "exact ordered inventory"):
                checker.replay_raw_evidence_archive(
                    expanded,
                    **fixture.correctness_arguments(),
                )

    def test_raw_replay_rejects_legacy_three_payload_archive(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        fixture = SoakFixture(Path(directory.name))
        legacy = fixture.root / "legacy-three-payload.tar"
        payloads = {
            "events.jsonl": (fixture.run_directory / "events.jsonl").read_bytes(),
            "manifest.json": fixture.manifest_path.read_bytes(),
            "run.json": (fixture.run_directory / "run.json").read_bytes(),
        }
        payloads["SHA256SUMS"] = b"".join(
            f"{hashlib.sha256(payloads[name]).hexdigest()}  {name}\n".encode("ascii")
            for name in sorted(payloads)
        )
        write_raw_tar(legacy, payloads)
        with self.assertRaisesRegex(checker.InputError, "exact ordered inventory"):
            checker.replay_raw_evidence_archive(
                legacy,
                **fixture.correctness_arguments(),
            )

    def test_raw_packager_is_create_only(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        fixture = SoakFixture(Path(directory.name))
        reviewed_digest = checker._normalized_manifest_sha256(fixture.manifest)
        output = fixture.root / "existing.tar"
        output.write_bytes(b"owner data")
        with mock.patch.object(
            checker,
            "REVIEWED_MANIFEST_TEMPLATE_CANONICAL_SHA256",
            reviewed_digest,
        ):
            with self.assertRaises(FileExistsError):
                checker.package_raw_evidence(
                    fixture.manifest_path,
                    fixture.run_directory,
                    output,
                    **fixture.trusted_arguments(),
                )
        self.assertEqual(output.read_bytes(), b"owner data")

    def test_raw_packager_refuses_failed_run(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        fixture = SoakFixture(Path(directory.name))
        fixture.events[-1]["status"] = "failure"
        fixture.write()
        reviewed_digest = checker._normalized_manifest_sha256(fixture.manifest)
        output = fixture.root / "failed.tar"
        with mock.patch.object(
            checker,
            "REVIEWED_MANIFEST_TEMPLATE_CANONICAL_SHA256",
            reviewed_digest,
        ):
            with self.assertRaisesRegex(checker.InputError, "non-passing"):
                checker.package_raw_evidence(
                    fixture.manifest_path,
                    fixture.run_directory,
                    output,
                    **fixture.trusted_arguments(),
                )
        self.assertFalse(output.exists())

    def test_raw_packager_removes_output_for_rejected_runtime_receipt(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        fixture = SoakFixture(Path(directory.name))
        fixture.mutate_receipt(
            "container-inspect-post.json",
            lambda value: value[0]["State"].__setitem__("OOMKilled", True),
        )
        reviewed_digest = checker._normalized_manifest_sha256(fixture.manifest)
        output = fixture.root / "rejected-runtime.tar"
        with mock.patch.object(
            checker,
            "REVIEWED_MANIFEST_TEMPLATE_CANONICAL_SHA256",
            reviewed_digest,
        ):
            with self.assertRaisesRegex(checker.InputError, "non-passing"):
                checker.package_raw_evidence(
                    fixture.manifest_path,
                    fixture.run_directory,
                    output,
                    **fixture.trusted_arguments(),
                )
        self.assertFalse(output.exists())


class SoakRunnerStaticTests(unittest.TestCase):
    def test_runner_recomputes_the_canonical_model_tree_binding(self) -> None:
        repository = Path(__file__).resolve().parents[3]
        runner = (repository / "ci/run_release_soak.sh").read_text(encoding="utf-8")
        self.assertIn('test -d "$model_path"', runner)
        self.assertIn('test ! -L "$model_path"', runner)
        self.assertIn('find "$model_path" -mindepth 1 ! -type d ! -type f', runner)
        self.assertIn("-type f -print0 | sort -z", runner)
        self.assertIn("printf '%s  %s\\n'", runner)
        self.assertIn(
            'test "$computed_model_sha256" = "$RILEY_MODEL_SHA256"',
            runner,
        )

    def test_runner_uses_distinct_reviewed_cancel_and_disconnect_transports(self) -> None:
        repository = Path(__file__).resolve().parents[3]
        runner = (repository / "ci/run_release_soak.sh").read_text(encoding="utf-8")
        self.assertIn('--max-time 0.05 -o "$output"', runner)
        self.assertIn('[ "$curl_code" -eq 28 ]', runner)
        self.assertIn('--no-buffer --max-time 300 --limit-rate 1K', runner)
        self.assertIn('| head -c 1024 >"$output"', runner)
        self.assertEqual(
            runner.count('pipeline_codes=("${PIPESTATUS[@]}")'), 2
        )
        capture = runner.index('pipeline_codes=("${PIPESTATUS[@]}")')
        curl_exit = runner.index('curl_code=${pipeline_codes[0]}', capture)
        head_exit = runner.index('head_code=${pipeline_codes[1]}', curl_exit)
        self.assertLess(capture, curl_exit)
        self.assertLess(curl_exit, head_exit)
        self.assertIn(
            '[ "$curl_code" -eq 23 ] && [ "$head_code" -eq 0 ]', runner
        )
        self.assertIn('[ "$response_bytes" -eq 1024 ]', runner)
        self.assertIn('jq -cS --arg profile "$profile"', runner)


if __name__ == "__main__":
    unittest.main()
