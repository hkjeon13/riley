from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import re
import shlex
import struct
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "check_python_free_release_e2e.py"
REPOSITORY_ROOT = SCRIPT.parents[2]
DRIVER = REPOSITORY_ROOT / "ci/run_python_free_release_e2e.sh"
PACKAGER = REPOSITORY_ROOT / "ci/release/package_python_free_e2e_evidence.py"
RELEASE_DIR = REPOSITORY_ROOT / "ci/release"
sys.path.insert(0, str(RELEASE_DIR))
from build_release_bundle import build_bundle  # noqa: E402
from release_common import (  # noqa: E402
    MIT_LICENSE_BYTES,
    SERVER_DEFAULTS_SOURCE_PATH,
    native_manifest_bytes,
)
from test_release import (  # noqa: E402
    DEPENDENCIES,
    EPOCH,
    fixture_elf,
    install_reviewed_server_defaults_source,
)

SPEC = importlib.util.spec_from_file_location("check_python_free_release_e2e", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
checker = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = checker
SPEC.loader.exec_module(checker)
PACKAGE_SPEC = importlib.util.spec_from_file_location("package_python_free_e2e_evidence", PACKAGER)
assert PACKAGE_SPEC is not None and PACKAGE_SPEC.loader is not None
packager = importlib.util.module_from_spec(PACKAGE_SPEC)
sys.modules[PACKAGE_SPEC.name] = packager
PACKAGE_SPEC.loader.exec_module(packager)


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode("utf-8")


def compact_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def http_json(value: object, content_type: str = "application/json") -> bytes:
    body = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return (
        b"HTTP/1.1 200 OK\r\n"
        + f"Content-Type: {content_type}\r\nContent-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode("ascii")
        + body
    )


def safetensors_fixture() -> bytes:
    header = json.dumps(
        {"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    header += b" " * (-len(header) % 8)
    return struct.pack("<Q", len(header)) + header + b"\0\0\0\0"


def write_raw_tar(path: Path, payloads: dict[str, bytes], *, link_name: str | None = None) -> None:
    with tarfile.open(path, "w", format=tarfile.USTAR_FORMAT) as archive:
        for name in sorted(payloads):
            contents = payloads[name]
            member = tarfile.TarInfo(name)
            member.size = len(contents)
            member.mode = 0o755 if name == "image-binary" else 0o644
            member.uid = 0
            member.gid = 0
            member.uname = ""
            member.gname = ""
            member.mtime = 0
            if name == link_name:
                member.type = tarfile.SYMTYPE
                member.linkname = "raw-evidence.json"
                member.size = 0
                archive.addfile(member)
            else:
                archive.addfile(member, io.BytesIO(contents))


def read_raw_tar(path: Path) -> dict[str, bytes]:
    with tarfile.open(path, "r:") as archive:
        result: dict[str, bytes] = {}
        for member in archive.getmembers():
            source = archive.extractfile(member)
            if source is not None:
                result[member.name] = source.read()
        return result


def refresh_checksums(payloads: dict[str, bytes]) -> None:
    payloads["SHA256SUMS"] = b"".join(
        f"{sha_bytes(payloads[name])}  {name}\n".encode("ascii")
        for name in sorted(payloads)
        if name != "SHA256SUMS"
    )


class E2EFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.revision = "a" * 40
        self.image_id = "sha256:" + "b" * 64
        self.source_archive = root / "source.tar"
        self.release_binary = root / "rustinfer"
        self.release_bundle = root / "rustinfer.tar.gz"
        self.model_dir = root / "model"
        self.model_dir.mkdir()
        self.config = self.model_dir / "config.json"
        self.weights = self.model_dir / "model.safetensors"
        self.tokenizer = self.model_dir / "tokenizer.json"
        self.golden = root / "correctness-golden.json"
        self.correctness_report = root / "correctness-report.json"
        self.shutdown_metrics = root / "shutdown-metrics.json"
        self.repeat_shutdown_metrics = root / "repeat-shutdown-metrics.json"
        self.evidence = root / "raw-evidence.json"
        self.raw_archive = root / "python-free-evidence.tar"
        self.evidence_dir = root / "raw-files"
        self.evidence_dir.mkdir()

        self.source_archive.write_bytes(b"source archive fixture")
        self.release_binary.write_bytes(fixture_elf())
        self.release_binary.chmod(0o755)
        self.config.write_bytes(b'{"model_type":"fixture"}\n')
        self.weights.write_bytes(safetensors_fixture())
        tokenizer_contents = {
            "merges.txt": b"#version: 0.2\na b\n",
            "special_tokens_map.json": b'{"eos_token":"<eos>"}\n',
            "tokenizer.json": b'{"version":"1.0","model":{}}\n',
            "tokenizer_config.json": b'{"model_max_length":1024}\n',
            "vocab.json": b'{"<eos>":0}\n',
        }
        for name, contents in tokenizer_contents.items():
            (self.model_dir / name).write_bytes(contents)
        (self.model_dir / "rustinfer-checkpoint.json").write_bytes(b'{"format":"rustinfer-checkpoint-v1"}\n')
        self.model_id = checker.MODEL_ID
        self.model_revision = checker.MODEL_REVISION
        self.expected_text = "fixture completion"
        self.expected_text_sha256 = sha_bytes(self.expected_text.encode())
        self.model_files = {
            path.relative_to(self.model_dir).as_posix(): sha_bytes(path.read_bytes())
            for path in sorted(self.model_dir.iterdir())
        }
        self.tokenizer_aggregate_sha256 = checker._tokenizer_aggregate_sha256(self.model_files)
        correctness = {
            "gate_id": checker.CORRECTNESS_GATE,
            "status": "pass",
            "bindings": {
                "candidate_git_revision": self.revision,
                "candidate_git_status_sha256": sha_bytes(b""),
                "model_id": self.model_id,
                "model_revision": self.model_revision,
                "config_sha256": self.model_files["config.json"],
                "weights_sha256": self.model_files["model.safetensors"],
                "tokenizer_sha256": self.tokenizer_aggregate_sha256,
            },
        }
        self.correctness_report.write_bytes(json_bytes(correctness))
        golden = {
            "schema_version": checker.GOLDEN_SCHEMA,
            "correctness_gate_id": checker.CORRECTNESS_GATE,
            "correctness_report_sha256": sha_bytes(self.correctness_report.read_bytes()),
            "source_revision": self.revision,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "config_sha256": self.model_files["config.json"],
            "weights_sha256": self.model_files["model.safetensors"],
            "tokenizer_aggregate_sha256": self.tokenizer_aggregate_sha256,
            "tokenizer_json_sha256": self.model_files["tokenizer.json"],
            "prompt": "A bounded release probe",
            "max_tokens": 8,
            "expected_greedy_text_sha256": self.expected_text_sha256,
        }
        self.golden.write_bytes(json_bytes(golden))
        repository = root / "repository"
        repository.mkdir()
        (repository / "Cargo.toml").write_text(
            '[workspace]\nmembers = []\n[workspace.package]\nversion = "0.1.0"\nlicense = "MIT"\n',
            encoding="utf-8",
        )
        (repository / "LICENSE").write_bytes(MIT_LICENSE_BYTES)
        install_reviewed_server_defaults_source(repository)
        build_bundle(
            binary_path=self.release_binary,
            output=self.release_bundle,
            repository_root=repository,
            source_revision=self.revision,
            source_date_epoch=EPOCH,
        )
        self.hashes = {
            "archive": sha_bytes(self.source_archive.read_bytes()),
            "binary": sha_bytes(self.release_binary.read_bytes()),
            "bundle": sha_bytes(self.release_bundle.read_bytes()),
            "model": checker.model_tree_sha256(self.model_dir),
            "config": self.model_files["config.json"],
            "weights": self.model_files["model.safetensors"],
            "tokenizer_json": self.model_files["tokenizer.json"],
            "tokenizer_aggregate": self.tokenizer_aggregate_sha256,
            "golden": sha_bytes(self.golden.read_bytes()),
            "correctness": sha_bytes(self.correctness_report.read_bytes()),
        }
        self.metrics_before = self._metrics(0, 0)
        self.metrics_after = self._metrics(1, 1)
        self.final_metrics = self._metrics(1, 1)
        self.repeat_final_metrics = self._metrics(0, 0)
        self.shutdown_metrics.write_bytes(json_bytes(self.final_metrics))
        self.repeat_shutdown_metrics.write_bytes(json_bytes(self.repeat_final_metrics))
        self.raw = self._raw()
        self._write_raw_files()
        self.write()

    @staticmethod
    def _metrics(cancellations: int, disconnects: int) -> dict[str, object]:
        return {
            "active_requests": 0,
            "waiting_requests": 0,
            "kv_allocated_blocks": 0,
            "allocation": {"device_live_count": 0, "device_live_bytes": 0, "pinned_live_count": 0, "pinned_live_bytes": 0},
            "counters": {"cancellations": cancellations, "disconnects": disconnects, "overloads": 0, "dropped_observations": 0},
        }

    def _processes(self, pid: int) -> list[dict[str, object]]:
        return [{"pid": pid, "ppid": 1, "comm": "rustinfer", "args": "/opt/rustinfer/bin/rustinfer serve --model /models/checkpoint"}]

    def _raw(self) -> dict[str, object]:
        ldd_lines = [f"{dependency} => /usr/lib/{dependency} (0x00000001)" for dependency in DEPENDENCIES]
        return {
            "schema_version": checker.RAW_SCHEMA,
            "run_id": "python-free-e2e-fixture",
            "recorded_at_utc": "2026-08-26T12:34:56Z",
            "status": "success",
            "source": {"git_revision": self.revision, "git_dirty": False, "source_archive_sha256": self.hashes["archive"]},
            "release": {"binary_sha256": self.hashes["binary"], "bundle_sha256": self.hashes["bundle"], "image_sha256": self.image_id.removeprefix("sha256:")},
            "model": {
                "model_id": self.model_id, "model_revision": self.model_revision,
                "model_tree_sha256": self.hashes["model"], "config_sha256": self.hashes["config"],
                "weights_sha256": self.hashes["weights"], "tokenizer_aggregate_sha256": self.hashes["tokenizer_aggregate"],
                "tokenizer_json_sha256": self.hashes["tokenizer_json"], "correctness_gate_id": checker.CORRECTNESS_GATE,
                "correctness_report_sha256": self.hashes["correctness"], "correctness_golden_sha256": self.hashes["golden"],
            },
            "runtime": {"container_ids": ["c" * 64, "e" * 64], "network_mode": "none", "image_id": self.image_id, "image_binary_sha256": self.hashes["binary"]},
            "observations": {
                "readyz": {"http_status": 200, "ready": True, "accepting": True},
                "models": {"http_status": 200, "model_ids": [self.model_id]},
                "greedy": {
                    "non_stream_http_status": 200, "stream_http_status": 200,
                    "non_stream_text_sha256": self.expected_text_sha256, "stream_text_sha256": self.expected_text_sha256,
                    "approved_text_sha256": self.expected_text_sha256, "completion_tokens": 2, "stream_token_events": 2,
                    "finish_reason": "length", "stream_done": True,
                    "prompt_sha256": sha_bytes(b"A bounded release probe"), "max_tokens": 8,
                },
                "sampling": {
                    "seed": 424242, "temperature": 0.8, "top_p": 0.95,
                    "first_http_status": 200, "second_http_status": 200,
                    "first_completion_tokens": 3, "second_completion_tokens": 3,
                    "first_finish_reason": "length", "second_finish_reason": "length",
                    "first_text_sha256": sha_bytes(b"sample output"), "second_text_sha256": sha_bytes(b"sample output"),
                },
                "cancellation": {
                    "disconnect_probe_sent": True, "cancellations_before": 0, "cancellations_after": 1,
                    "disconnects_before": 0, "disconnects_after": 1, "active_requests_after": 0, "waiting_requests_after": 0,
                },
                "shutdown": {
                    "signal": "SIGTERM", "exit_code": 0, "metrics": self.final_metrics,
                    "metrics_sha256": sha_bytes(self.shutdown_metrics.read_bytes()), "repeat_exit_code": 0,
                    "repeat_metrics": self.repeat_final_metrics,
                    "repeat_metrics_sha256": sha_bytes(self.repeat_shutdown_metrics.read_bytes()),
                },
                "python_free": {
                    "forbidden_executables": [], "forbidden_artifact_count": 0,
                    "processes": self._processes(101), "manifest_dependencies": DEPENDENCIES,
                    "loader_dependencies": ldd_lines, "unresolved_dependencies": [], "forbidden_dependency_matches": [],
                },
            },
        }

    def _container(self, container_id: str, pid: int, running: bool, ordinal: str) -> bytes:
        state = {
            "Running": running, "Pid": pid if running else 0, "ExitCode": 0,
            "OOMKilled": False, "Error": "", "StartedAt": f"2026-08-26T12:0{ordinal}:00Z",
            "FinishedAt": "0001-01-01T00:00:00Z" if running else f"2026-08-26T12:1{ordinal}:00Z",
        }
        value = [{
            "Id": container_id, "Image": self.image_id, "Path": "/opt/rustinfer/bin/rustinfer",
            "Args": ["serve", "--model", "/models/checkpoint", "--model-id", self.model_id, "--bind", "127.0.0.1:8080", "--max-output-tokens", "1024"],
            "Created": f"2026-08-26T12:0{ordinal}:00Z", "Config": {"Image": self.image_id},
            "HostConfig": {"NetworkMode": "none", "ReadonlyRootfs": True, "DeviceRequests": [{"Driver": "nvidia", "Capabilities": [["gpu"]]}]},
            "State": state, "Mounts": [{"Destination": "/models/checkpoint", "RW": False}, {"Destination": "/evidence", "RW": True}],
        }]
        return json_bytes(value)

    def _write_raw_files(self) -> None:
        files: dict[str, bytes] = {}
        files["image-binary"] = self.release_binary.read_bytes()
        files["image-native-dependencies.txt"] = native_manifest_bytes(DEPENDENCIES)
        files["image-ldd.txt"] = b"".join(f"{dependency} => /usr/lib/{dependency} (0x00000001)\n".encode() for dependency in DEPENDENCIES)
        files["image-readelf.txt"] = (
            b"Class:                             ELF64\n"
            b"Type:                              DYN (Position-Independent Executable file)\n"
            b"Machine:                           Advanced Micro Devices X86-64\n"
            + b"".join(f" 0x0000000000000001 (NEEDED)             Shared library: [{dependency}]\n".encode() for dependency in DEPENDENCIES)
        )
        files["image-python-scan.txt"] = b"[forbidden-executables]\n[forbidden-artifacts]\n"
        files["image-inspect.json"] = json_bytes([{
            "Id": self.image_id, "Architecture": "amd64", "Os": "linux",
            "Config": {"Entrypoint": ["/opt/rustinfer/bin/rustinfer"], "Cmd": ["--help"], "User": "65532:65532", "Env": ["PATH=/opt/rustinfer/bin:/usr/bin"]},
            "RootFS": {"Type": "layers", "Layers": ["sha256:" + "d" * 64]},
        }])
        for ordinal, container_id, pid, digit in (("first", "c" * 64, 101, "1"), ("second", "e" * 64, 202, "2")):
            files[f"container-{ordinal}-pre.json"] = self._container(container_id, pid, True, digit)
            files[f"container-{ordinal}-runtime.json"] = self._container(container_id, pid, True, digit)
            files[f"container-{ordinal}-post.json"] = self._container(container_id, pid, False, digit)
            process = f"PID PPID COMMAND COMMAND\n{pid} 1 rustinfer /opt/rustinfer/bin/rustinfer serve --model /models/checkpoint\n".encode()
            files[f"process-{ordinal}-pre.txt"] = process
            files[f"process-{ordinal}-runtime.txt"] = process
        greedy_request = {"model": self.model_id, "prompt": "A bounded release probe", "max_tokens": 8, "temperature": 0, "top_p": 1, "stream": False}
        files["request-greedy.json"] = compact_json(greedy_request)
        files["request-greedy-stream.json"] = compact_json({**greedy_request, "stream": True})
        files["request-sampling.json"] = compact_json({"model": self.model_id, "prompt": "A bounded release probe", "max_tokens": 16, "temperature": 0.8, "top_p": 0.95, "seed": 424242, "stream": False})
        files["http-readyz.raw"] = http_json({"ready": True, "accepting": True})
        files["http-models.raw"] = http_json({"data": [{"id": self.model_id}]})
        files["http-greedy.raw"] = http_json({"choices": [{"text": self.expected_text, "finish_reason": "length"}], "usage": {"completion_tokens": 2}})
        sse = (
            'data: {"choices":[{"finish_reason":null,"text":"fixture "}]}\n\n'
            'data: {"choices":[{"finish_reason":null,"text":"completion"}]}\n\n'
            'data: {"choices":[{"finish_reason":"length","text":""}]}\n\n'
            "data: [DONE]\n\n"
        ).encode()
        files["http-greedy-stream.raw"] = (
            b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
            + f"Content-Length: {len(sse)}\r\nConnection: close\r\n\r\n".encode()
            + sse
        )
        sampling_response = http_json({"choices": [{"text": "sample output", "finish_reason": "length"}], "usage": {"completion_tokens": 3}})
        files["http-sampling-first.raw"] = sampling_response
        files["http-sampling-second.raw"] = sampling_response
        files["http-metrics-before.raw"] = http_json(self.metrics_before)
        files["http-metrics-after.raw"] = http_json(self.metrics_after)
        cancel_body = json.dumps({"model": self.model_id, "prompt": "A bounded release probe", "max_tokens": 512, "temperature": 0, "top_p": 1, "stream": True}, sort_keys=True, separators=(",", ":")).encode()
        files["cancellation-request.raw"] = (
            f"POST /v1/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {len(cancel_body)}\r\nConnection: close\r\n\r\n".encode()
            + cancel_body
        )
        files["cancellation-response-prefix.raw"] = b"HTTP/1.1 200"
        for name, contents in files.items():
            path = self.evidence_dir / name
            path.write_bytes(contents)
            if name == "image-binary":
                path.chmod(0o755)

    def package_inputs(self) -> dict[str, Path]:
        special = {
            "correctness-golden.json": self.golden,
            "model-SHA256SUMS": self.root / "model-SHA256SUMS",
            "raw-evidence.json": self.evidence,
            "repeat-shutdown-metrics.json": self.repeat_shutdown_metrics,
            "shutdown-metrics.json": self.shutdown_metrics,
        }
        special["model-SHA256SUMS"].write_bytes(checker.model_tree_manifest_bytes(self.model_dir))
        return {name: special.get(name, self.evidence_dir / name) for name in packager.PAYLOAD_ARGUMENTS}

    def write(self) -> None:
        self.evidence.write_bytes(json_bytes(self.raw))
        if self.raw_archive.exists():
            self.raw_archive.unlink()
        packager.package(self.raw_archive, self.package_inputs(), self.model_dir)

    def evaluate(self) -> tuple[dict[str, object], str | None]:
        return checker.evaluate(
            self.evidence, raw_archive=self.raw_archive, source_revision=self.revision,
            source_archive=self.source_archive, release_binary=self.release_binary,
            release_bundle=self.release_bundle, image_id=self.image_id, model_dir=self.model_dir,
            expected_model_tree_sha256=self.hashes["model"], weights=self.weights,
            expected_weights_sha256=self.hashes["weights"], tokenizer=self.tokenizer,
            expected_tokenizer_json_sha256=self.hashes["tokenizer_json"],
            expected_tokenizer_aggregate_sha256=self.hashes["tokenizer_aggregate"],
            correctness_golden=self.golden,
            expected_correctness_golden_sha256=self.hashes["golden"],
            correctness_report=self.correctness_report,
            expected_correctness_report_sha256=self.hashes["correctness"],
            shutdown_metrics=self.shutdown_metrics,
            repeat_shutdown_metrics=self.repeat_shutdown_metrics,
        )

    def replay(self, archive: dict[str, object] | None = None) -> tuple[dict[str, object], str | None]:
        return checker.validate_bound_raw_archive(
            archive or checker.load_raw_evidence_archive(self.raw_archive),
            source_revision=self.revision, source_archive_sha256=self.hashes["archive"],
            release_binary_sha256=self.hashes["binary"], release_bundle_sha256=self.hashes["bundle"],
            image_id=self.image_id,
            correctness_report=json.loads(self.correctness_report.read_text()),
            correctness_report_sha256=self.hashes["correctness"],
            correctness_golden_sha256=self.hashes["golden"],
        )

    def argv(self, report: Path) -> list[str]:
        return [
            "--evidence", str(self.evidence), "--raw-archive", str(self.raw_archive),
            "--source-revision", self.revision, "--source-archive", str(self.source_archive),
            "--release-binary", str(self.release_binary), "--release-bundle", str(self.release_bundle),
            "--image-id", self.image_id, "--model-dir", str(self.model_dir),
            "--model-tree-sha256", self.hashes["model"], "--weights", str(self.weights),
            "--weights-sha256", self.hashes["weights"], "--tokenizer", str(self.tokenizer),
            "--tokenizer-json-sha256", self.hashes["tokenizer_json"],
            "--tokenizer-aggregate-sha256", self.hashes["tokenizer_aggregate"],
            "--correctness-golden", str(self.golden), "--correctness-golden-sha256", self.hashes["golden"],
            "--correctness-report", str(self.correctness_report), "--correctness-report-sha256", self.hashes["correctness"],
            "--shutdown-metrics", str(self.shutdown_metrics), "--repeat-shutdown-metrics", str(self.repeat_shutdown_metrics),
            "--report", str(report),
        ]


class PythonFreeReleaseE2EV2Tests(unittest.TestCase):
    def test_complete_bound_raw_runtime_replay_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = E2EFixture(Path(directory))
            report, diagnostic = fixture.evaluate()
            self.assertIsNone(diagnostic)
            self.assertEqual(report["status"], "passed")
            self.assertEqual({row["id"] for row in report["checks"]}, set(checker.CHECK_IDS))
            self.assertEqual(report["raw_evidence_sha256"], sha_bytes(fixture.raw_archive.read_bytes()))

    def test_frozen_v2_golden_cannot_authorize_v3_release_e2e(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = E2EFixture(Path(directory))
            old_gate = "smollm2-fp32-bf16-native-e0-v2"
            correctness = json.loads(fixture.correctness_report.read_text())
            correctness["gate_id"] = old_gate
            fixture.correctness_report.write_bytes(json_bytes(correctness))
            fixture.hashes["correctness"] = sha_bytes(
                fixture.correctness_report.read_bytes()
            )
            golden = json.loads(fixture.golden.read_text())
            golden["correctness_gate_id"] = old_gate
            golden["correctness_report_sha256"] = fixture.hashes["correctness"]
            fixture.golden.write_bytes(json_bytes(golden))
            fixture.hashes["golden"] = sha_bytes(fixture.golden.read_bytes())
            fixture.raw["model"]["correctness_gate_id"] = old_gate
            fixture.raw["model"]["correctness_report_sha256"] = fixture.hashes[
                "correctness"
            ]
            fixture.raw["model"]["correctness_golden_sha256"] = fixture.hashes[
                "golden"
            ]
            fixture.write()

            report, diagnostic = fixture.evaluate()
            self.assertEqual(report["status"], "error")
            self.assertIn("native E0 v3 gate", diagnostic)

    def test_packager_is_deterministic_and_create_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = E2EFixture(Path(directory))
            output = fixture.root / "copy.tar"
            packager.package(output, fixture.package_inputs(), fixture.model_dir)
            self.assertEqual(output.read_bytes(), fixture.raw_archive.read_bytes())
            original = output.read_bytes()
            with self.assertRaises(FileExistsError):
                packager.package(output, fixture.package_inputs(), fixture.model_dir)
            self.assertEqual(output.read_bytes(), original)

    def test_v1_summary_only_fixture_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payloads = {
                "raw-evidence.json": b'{"schema_version":"rustinfer.python-free-release-e2e-raw.v1"}\n',
                "correctness-golden.json": b"{}\n", "model-SHA256SUMS": b"0" * 64 + b"  config.json\n",
                "shutdown-metrics.json": b"{}\n", "repeat-shutdown-metrics.json": b"{}\n",
            }
            refresh_checksums(payloads)
            archive = root / "synthetic-v1.tar"
            write_raw_tar(archive, payloads)
            with self.assertRaises(checker.EvidenceError) as caught:
                checker.load_raw_evidence_archive(archive)
            self.assertIn("missing fixed v2", str(caught.exception))

    def test_image_inspect_and_process_transition_cannot_be_self_asserted(self) -> None:
        mutations = {
            "image": lambda payloads: payloads.__setitem__("image-inspect.json", json_bytes([{"Id": "sha256:" + "f" * 64}])),
            "container": lambda payloads: payloads.__setitem__("container-first-post.json", payloads["container-first-runtime.json"]),
        }
        for expected, mutate in mutations.items():
            with self.subTest(expected), tempfile.TemporaryDirectory() as directory:
                fixture = E2EFixture(Path(directory))
                payloads = read_raw_tar(fixture.raw_archive)
                mutate(payloads)
                refresh_checksums(payloads)
                write_raw_tar(fixture.raw_archive, payloads)
                report, diagnostic = fixture.replay(checker.load_raw_evidence_archive(fixture.raw_archive))
                self.assertEqual(report["status"], "error")
                self.assertIn(expected, diagnostic.lower())

    def test_readiness_requires_json_booleans_not_equal_integers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = E2EFixture(Path(directory))
            payloads = read_raw_tar(fixture.raw_archive)
            raw = json.loads(payloads["raw-evidence.json"])
            raw["observations"]["readyz"]["ready"] = 1
            raw["observations"]["readyz"]["accepting"] = 1
            payloads["raw-evidence.json"] = json_bytes(raw)
            payloads["http-readyz.raw"] = http_json(
                {"ready": 1, "accepting": 1}
            )
            refresh_checksums(payloads)
            write_raw_tar(fixture.raw_archive, payloads)
            report, diagnostic = fixture.replay(
                checker.load_raw_evidence_archive(fixture.raw_archive)
            )
            self.assertEqual(report["status"], "error")
            self.assertIn("boolean", diagnostic)

    def test_request_numbers_cannot_be_replaced_by_json_booleans(self) -> None:
        for mutation in ("greedy", "sampling", "cancellation"):
            with self.subTest(mutation), tempfile.TemporaryDirectory() as directory:
                fixture = E2EFixture(Path(directory))
                payloads = read_raw_tar(fixture.raw_archive)
                if mutation == "greedy":
                    for name in ("request-greedy.json", "request-greedy-stream.json"):
                        request = json.loads(payloads[name])
                        request["temperature"] = False
                        request["top_p"] = True
                        payloads[name] = compact_json(request)
                elif mutation == "sampling":
                    request = json.loads(payloads["request-sampling.json"])
                    request["temperature"] = False
                    request["top_p"] = True
                    payloads["request-sampling.json"] = compact_json(request)
                else:
                    _, _, body = payloads["cancellation-request.raw"].partition(
                        b"\r\n\r\n"
                    )
                    request = json.loads(body)
                    request["temperature"] = False
                    request["top_p"] = True
                    body = json.dumps(
                        request, sort_keys=True, separators=(",", ":")
                    ).encode("utf-8")
                    payloads["cancellation-request.raw"] = (
                        b"POST /v1/completions HTTP/1.1\r\n"
                        b"Host: localhost\r\n"
                        b"Content-Type: application/json\r\n"
                        + f"Content-Length: {len(body)}\r\n".encode("ascii")
                        + b"Connection: close\r\n\r\n"
                        + body
                    )
                refresh_checksums(payloads)
                write_raw_tar(fixture.raw_archive, payloads)
                report, diagnostic = fixture.replay(
                    checker.load_raw_evidence_archive(fixture.raw_archive)
                )
                self.assertEqual(report["status"], "error")
                self.assertIn("number", diagnostic)

    def test_reviewed_golden_prevents_transcript_self_authorization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = E2EFixture(Path(directory))
            payloads = read_raw_tar(fixture.raw_archive)
            golden = json.loads(payloads["correctness-golden.json"])
            golden["expected_greedy_text_sha256"] = sha_bytes(b"fabricated completion")
            payloads["correctness-golden.json"] = json_bytes(golden)
            raw = json.loads(payloads["raw-evidence.json"])
            raw["model"]["correctness_golden_sha256"] = sha_bytes(payloads["correctness-golden.json"])
            payloads["raw-evidence.json"] = json_bytes(raw)
            refresh_checksums(payloads)
            write_raw_tar(fixture.raw_archive, payloads)
            report, diagnostic = fixture.replay(checker.load_raw_evidence_archive(fixture.raw_archive))
            self.assertEqual(report["status"], "error")
            self.assertIn("reviewed", diagnostic)

    def test_actual_model_and_image_binary_bytes_are_checksum_subjects(self) -> None:
        for member in ("model/model.safetensors", "image-binary"):
            with self.subTest(member), tempfile.TemporaryDirectory() as directory:
                fixture = E2EFixture(Path(directory))
                payloads = read_raw_tar(fixture.raw_archive)
                payloads[member] += b"tamper"
                refresh_checksums(payloads)
                write_raw_tar(fixture.raw_archive, payloads)
                if member.startswith("model/"):
                    with self.assertRaises(checker.EvidenceError):
                        checker.load_raw_evidence_archive(fixture.raw_archive)
                else:
                    report, diagnostic = fixture.replay(checker.load_raw_evidence_archive(fixture.raw_archive))
                    self.assertEqual(report["status"], "error")
                    self.assertIn("image", diagnostic)

    def test_archive_inventory_metadata_and_checksums_fail_closed(self) -> None:
        for mutation in ("missing", "extra", "checksum", "link"):
            with self.subTest(mutation), tempfile.TemporaryDirectory() as directory:
                fixture = E2EFixture(Path(directory))
                payloads = read_raw_tar(fixture.raw_archive)
                link = None
                if mutation == "missing":
                    del payloads["http-readyz.raw"]
                    refresh_checksums(payloads)
                elif mutation == "extra":
                    payloads["unreviewed.txt"] = b"x\n"
                    refresh_checksums(payloads)
                elif mutation == "checksum":
                    payloads["SHA256SUMS"] = b"0" * len(payloads["SHA256SUMS"])
                else:
                    link = "image-inspect.json"
                write_raw_tar(fixture.raw_archive, payloads, link_name=link)
                with self.assertRaises(checker.EvidenceError):
                    checker.load_raw_evidence_archive(fixture.raw_archive)

    def test_archived_json_must_be_utf8_not_utf16(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = E2EFixture(Path(directory))
            payloads = read_raw_tar(fixture.raw_archive)
            raw = json.loads(payloads["raw-evidence.json"])
            payloads["raw-evidence.json"] = json.dumps(
                raw, sort_keys=True
            ).encode("utf-16")
            refresh_checksums(payloads)
            write_raw_tar(fixture.raw_archive, payloads)
            with self.assertRaisesRegex(checker.EvidenceError, "strict UTF-8"):
                checker.load_raw_evidence_archive(fixture.raw_archive)

    def test_replay_requires_independently_reviewed_golden_digest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = E2EFixture(Path(directory))
            archive = checker.load_raw_evidence_archive(fixture.raw_archive)
            report, diagnostic = checker.validate_bound_raw_archive(
                archive, source_revision=fixture.revision,
                source_archive_sha256=fixture.hashes["archive"],
                release_binary_sha256=fixture.hashes["binary"],
                release_bundle_sha256=fixture.hashes["bundle"], image_id=fixture.image_id,
                correctness_report=json.loads(fixture.correctness_report.read_text()),
                correctness_report_sha256=fixture.hashes["correctness"],
            )
            self.assertEqual(report["status"], "error")
            self.assertIn("independently reviewed", diagnostic)

    def test_cli_report_is_create_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = E2EFixture(Path(directory))
            report = fixture.root / "attestation.json"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(checker.main(fixture.argv(report)), 0)
            original = report.read_bytes()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(checker.main(fixture.argv(report)), 2)
            self.assertEqual(report.read_bytes(), original)

    def test_remote_driver_preserves_v2_raw_observations(self) -> None:
        source = DRIVER.read_text(encoding="utf-8")
        for required in (
            "rustinfer.python-free-release-e2e-raw.v2", "docker image inspect", "docker cp",
            "readelf --file-header --program-headers --dynamic", "container-first-pre",
            "container-first-runtime", "container-first-post", "process-first-runtime",
            "cancellation-request", "cancellation-response-prefix", "--model-dir",
            "--network none", "docker kill --signal TERM",
        ):
            self.assertIn(required, source)
        self.assertNotIn("--network host", source)

    def test_remote_driver_and_checker_share_exact_default_serve_arguments(self) -> None:
        source = DRIVER.read_text(encoding="utf-8")
        match = re.search(r"(?ms)^launch_container\(\) \{\n(?P<body>.*?)^\}", source)
        self.assertIsNotNone(match)
        body = match.group("body")
        marker = '"$RUSTINFER_E2E_IMAGE_ID" \\\n'
        self.assertIn(marker, body)
        command = body.split(marker, 1)[1].replace("\\\n", " ")
        command = command.replace('"$model_id"', shlex.quote(checker.MODEL_ID))
        command = command.replace('"$RUSTINFER_E2E_BIND"', "127.0.0.1:8080")
        runner_args = shlex.split(command)
        literal_default_args = [
            "serve", "--model", "/models/checkpoint", "--model-id", checker.MODEL_ID,
            "--bind", "127.0.0.1:8080", "--max-output-tokens", "1024",
        ]
        checker_args = checker.expected_container_args(checker.MODEL_ID)
        self.assertEqual(checker_args, literal_default_args)
        self.assertEqual(runner_args, checker_args)
        self.assertEqual(
            checker.STABLE_OPTIMIZATION_DEFAULTS,
            checker.release_common.STABLE_OPTIMIZATION_DEFAULTS,
        )
        defaults_source = (
            REPOSITORY_ROOT / checker.release_common.SERVER_DEFAULTS_SOURCE_PATH
        ).read_bytes()
        self.assertEqual(
            hashlib.sha256(defaults_source).hexdigest(),
            checker.release_common.SERVER_DEFAULTS_SOURCE_SHA256,
        )
        for flag in checker.OPTIMIZATION_SELECTION_FLAGS:
            self.assertNotIn(flag, body)
            self.assertNotIn(flag, runner_args)
            self.assertNotIn(flag, checker_args)

    def test_checker_rejects_explicit_optimization_selection_arguments(self) -> None:
        values = {
            "--execution-completion": "iteration-batch",
            "--residual-rmsnorm": "separate",
            "--reduction-profile": "canonical-v1",
        }
        self.assertEqual(set(values), set(checker.OPTIMIZATION_SELECTION_FLAGS))
        for flag, value in values.items():
            with self.subTest(flag=flag), tempfile.TemporaryDirectory() as directory:
                fixture = E2EFixture(Path(directory))
                payloads = read_raw_tar(fixture.raw_archive)
                snapshot = json.loads(payloads["container-first-pre.json"])
                snapshot[0]["Args"].extend([flag, value])
                payloads["container-first-pre.json"] = json_bytes(snapshot)
                refresh_checksums(payloads)
                write_raw_tar(fixture.raw_archive, payloads)
                report, diagnostic = fixture.replay(
                    checker.load_raw_evidence_archive(fixture.raw_archive)
                )
                self.assertEqual(report["status"], "error")
                self.assertIn("arguments mismatch", diagnostic)


if __name__ == "__main__":
    unittest.main()
