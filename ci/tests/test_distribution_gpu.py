"""Exercise the GPU image check's evidence bindings and cleanup without CUDA."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ci"))
import check_distribution_gpu as checker
sys.path.pop(0)


class DistributionGpuTest(unittest.TestCase):
    gpu = "GPU-031a7f9d-5789-04b8-c9d6-63f55ee1e6a7"
    container = "c" * 64
    pid = 2345

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.cpu_report = self.root / "cpu-report.json"
        inputs = {"schema_version": "riley.distribution-image-check.v1", "status": "passed",
                  "image_id": "sha256:" + "a" * 64, "source_revision": "b" * 40,
                  "source_archive_sha256": "d" * 64, "binaries": {checker.RILEY: "e" * 64}}
        self.cpu_report.write_text(json.dumps(inputs))
        self.image = inputs["image_id"]
        self.binary = inputs["binaries"][checker.RILEY]
        self.image_inspect = {
            "Id": self.image, "Os": "linux", "Architecture": "amd64",
            "Config": {
                "User": "65532:65532", "Entrypoint": ["/opt/riley/bin/riley-docker"], "Cmd": ["help"],
                "Volumes": {"/models": {}}, "ExposedPorts": {"8080/tcp": {}},
                "Env": ["RILEY_MODEL_DIR=/models", "RILEY_BIND=0.0.0.0:8080"],
                "Labels": {"org.opencontainers.image.revision": inputs["source_revision"],
                           "io.riley.source-archive-sha256": inputs["source_archive_sha256"]},
            },
        }
        model = self.root / checker.CHECKPOINT
        model.write_text(json.dumps({"source_model": "test-model", "source_revision": "a" * 40}))
        self.model_hash = checker.digest(model)
        self.exit_code = 0
        self.device_pids = set()
        self.api_pass = True
        self.smoke_disconnected = True
        self.soak_growth_kib = 0
        self.create_interrupted = False
        self.name = None
        self.calls = []
        self.out = self.root / "evidence"

    def docker(self, *args):
        self.calls.append(args)
        match args:
            case ("image", "inspect", self.image):
                return json.dumps([self.image_inspect])
            case ("create", *rest):
                self.assertIn("--pull=never", rest)
                self.assertIn(f"device={self.gpu}", rest)
                self.assertIn("127.0.0.1::8080", rest)
                self.assertIn(self.image, rest)
                self.name = rest[rest.index("--name") + 1]
                if self.create_interrupted:
                    raise subprocess.TimeoutExpired(args, 30)
                return self.container
            case ("start", self.container):
                self.device_pids = {self.pid}
            case ("container", "inspect", self.container):
                return json.dumps([{
                    "Id": self.container, "Image": self.image,
                    "State": {"Running": bool(self.device_pids), "Pid": self.pid, "ExitCode": self.exit_code},
                    "NetworkSettings": {"Ports": {"8080/tcp": [{"HostIp": "127.0.0.1", "HostPort": "44321"}]}},
                }])
            case ("container", "inspect", self.name):
                return json.dumps([{"Id": self.container, "Name": "/" + self.name,
                                    "Config": {"Labels": {"io.riley.gpu-check": self.name}}}])
            case ("exec", self.container, "nvidia-smi", *_):
                return f"{self.gpu}, A100, 550\n"
            case ("exec", self.container, "/bin/sh", *_):
                return f"{self.binary}  /proc/1/exe\n"
            case ("stop", "--time", "20", self.container):
                self.device_pids = set()
            case ("rm", "--force", "--volumes", self.container):
                self.device_pids = set()
            case _:
                self.fail(f"unexpected Docker command {args!r}")

    def check(self, model_hash=None, **kwargs):
        def log_command(out, file, argv, _seconds, **_kwargs):
            (out / file).write_text("fixture log\n")
            if file == "api.log":
                self.assertIn("127.0.0.1:44321", argv)
                (out / "api.json").write_text(json.dumps({"results": [{"passed": self.api_pass}]}))
            if file == "smoke.log":
                self.assertIn("--smoke", argv)
                self.assertNotIn("--require-zero-rss-growth", argv)
                self.assertEqual(argv[argv.index("--docker-container") + 1], self.container)
                self.assertEqual(argv[argv.index("--pid") + 1], str(self.pid))
                checks = dict.fromkeys(checker.MIXED_LOAD_CHECKS, True)
                checks["disconnects_observed"] = self.smoke_disconnected
                passed = all(checks.values())
                (out / "smoke.json").write_text(json.dumps({
                    "status": "passed" if passed else "failed", "mode": "smoke",
                    "seconds": checker.SMOKE_SECONDS, "rate": 8, "shared_prefixes": 8,
                    "counts": {}, "counter_changes": {}, "checks": checks,
                    "rss_source": {"container_id": self.container, "host_pid": self.pid, "container_pid": 1},
                }))
                return subprocess.CompletedProcess(argv, int(not passed))
            if file == "soak.log":
                self.assertIn("--require-zero-rss-growth", argv)
                self.assertEqual(argv[argv.index("--docker-container") + 1], self.container)
                self.assertEqual(argv[argv.index("--pid") + 1], str(self.pid))
                passed = self.soak_growth_kib <= 0
                (out / "soak.json").write_text(json.dumps({
                    "status": "passed" if passed else "failed", "mode": "mixed_soak", "seconds": 1800,
                    "rate": 8, "shared_prefixes": 8, "counts": {"plain": {"completed": 1}},
                    "rss_source": {"container_id": self.container, "host_pid": self.pid, "container_pid": 1},
                    "checks": {"rss_growth_second_half_at_most_zero": passed, "no_unexpected_http_or_stream_error": True,
                               "completed_schema_and_tool_covered": True, "disconnects_observed": True,
                               "idle_after_drain": True, "native_allocation_plateau": True, "workload_coverage": True,
                               "p95_drift_at_most_5_percent": True},
                    "smaps_rollup_second_half": {"rss_growth_kib": self.soak_growth_kib},
                }))
                return subprocess.CompletedProcess(argv, int(not passed))

        def copy_shutdown(argv, **_kwargs):
            self.assertEqual(argv[:3], ["docker", "cp", f"{self.container}:{checker.SHUTDOWN}"])
            Path(argv[3]).write_text(json.dumps({"active_requests": 0, "waiting_requests": 0,
                                                "kv_allocated_blocks": 0, "allocation": {"live_bytes": 0}}))

        with (patch.object(checker, "docker", self.docker), patch.object(checker, "holder_pids", return_value=set()),
              patch.object(checker, "gpu_pids", side_effect=lambda _: self.device_pids),
              patch.object(checker, "wait_ready", return_value={"ready": True, "accepting": True}),
              patch.object(checker, "log_command", log_command),
              patch.object(checker.subprocess, "run", copy_shutdown)):
            return checker.check_gpu(self.cpu_report, self.gpu, self.root, model_hash or self.model_hash, self.out, False,
                                     **kwargs)

    def test_report_binds_the_image_process_and_api(self):
        self.check()
        report = json.loads((self.out / "report.json").read_bytes())
        self.assertEqual(report["image_id"], self.image)
        self.assertEqual(report["riley_binary_sha256"], self.binary)
        self.assertEqual(report["gpu_uuid"], self.gpu)
        self.assertEqual(report["artifacts_sha256"]["api.json"], checker.digest(self.out / "api.json"))
        self.assertEqual(self.calls[-1], ("rm", "--force", "--volumes", self.container))

    def test_accuracy_suites_use_the_same_container_address_and_c32_limit(self):
        golden = self.root / "golden.json.gz"
        holdout = self.root / "holdout.json.gz"
        holdout_golden = self.root / "holdout-golden.json.gz"
        with patch.object(checker, "check_accuracy", return_value={"verified_requests": 10}) as accuracy:
            self.check(accuracy_baseline=golden, accuracy_holdout=(holdout, holdout_golden))
        self.assertEqual([call.args for call in accuracy.call_args_list], [
            (self.out, "127.0.0.1:44321", "baseline", golden),
            (self.out, "127.0.0.1:44321", "holdout", holdout_golden, holdout),
        ])
        report = json.loads((self.out / "report.json").read_bytes())
        self.assertEqual(set(report["accuracy_suites"]), {"baseline", "holdout"})
        self.assertEqual(report["accuracy_server_arguments"][:2], ["--max-active-sequences", "32"])
        create = next(call for call in self.calls if call[0] == "create")
        self.assertEqual(list(create[-6:]), report["accuracy_server_arguments"])

    def test_positive_soak_growth_fails_after_graceful_shutdown(self):
        self.soak_growth_kib = 512
        self.assertEqual(self.check(mixed_soak="cache-on"), 1)
        report = json.loads((self.out / "report.json").read_bytes())
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["shutdown_exit"], 0)
        self.assertTrue((self.out / "shutdown.json").is_file())
        self.assertEqual(report["mixed_soak"]["failed_checks"], ["rss_growth_second_half_at_most_zero"])
        self.assertEqual(report["soak_server_arguments"][-4:], ["--kv-blocks", "16384", "--prefix-cache-blocks", "128"])
        create = next(call for call in self.calls if call[0] == "create")
        self.assertEqual(list(create[-10:]), report["soak_server_arguments"])

    def test_cache_off_soak_does_not_supply_an_invalid_zero_block_budget(self):
        self.assertEqual(self.check(mixed_soak="cache-off"), 0)
        report = json.loads((self.out / "report.json").read_bytes())
        create = next(call for call in self.calls if call[0] == "create")
        self.assertNotIn("--prefix-cache-blocks", create)
        self.assertEqual(report["soak_server_arguments"][-2:], ["--kv-blocks", "16384"])
        self.assertEqual(report["mixed_soak"]["mode"], "cache-off")

    def test_serving_smoke_reports_disconnect_failure_after_graceful_shutdown(self):
        self.smoke_disconnected = False
        self.assertEqual(self.check(serving_smoke=True), 1)
        report = json.loads((self.out / "report.json").read_bytes())
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["shutdown_exit"], 0)
        self.assertEqual(report["serving_smoke"]["failed_checks"], ["disconnects_observed"])
        self.assertEqual(report["serving_smoke"]["server_arguments"][-4:],
                         ["--kv-blocks", "4096", "--prefix-cache-blocks", "128"])
        self.assertEqual(report["artifacts_sha256"]["smoke.json"], checker.digest(self.out / "smoke.json"))
        create = next(call for call in self.calls if call[0] == "create")
        self.assertEqual(list(create[-10:]), report["serving_smoke"]["server_arguments"])

    def test_serving_smoke_cannot_be_confused_with_strict_soak_or_long_accuracy(self):
        for options in ({"mixed_soak": "cache-off"}, {"accuracy_baseline": self.root / "golden"}):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, "select serving smoke separately"):
                self.check(serving_smoke=True, **options)
        self.assertFalse(self.out.exists())
        self.assertFalse(self.calls)

    def test_checkpoint_drift_prevents_container_creation(self):
        with self.assertRaisesRegex(ValueError, "checkpoint manifest differs"):
            self.check("0" * 64)
        self.assertEqual(self.calls, [("image", "inspect", self.image)])
        self.assertFalse(self.out.exists())

    def test_binary_drift_and_api_failure_remove_the_container(self):
        digest = self.binary
        for changed_binary in (True, False):
            with self.subTest(changed_binary=changed_binary):
                self.binary = "0" * 64 if changed_binary else digest
                self.api_pass = False
                self.out = self.root / f"failure-{changed_binary}"
                regex = "running Riley differs" if changed_binary else "API suite failed"
                with self.assertRaisesRegex(ValueError, regex):
                    self.check()
                self.assertEqual(self.calls[-1], ("rm", "--force", "--volumes", self.container))
                self.assertFalse((self.out / "report.json").exists())

    def test_forced_shutdown_is_a_failure(self):
        self.exit_code = 137
        with self.assertRaisesRegex(ValueError, "stop gracefully"):
            self.check()
        self.assertEqual(self.calls[-1], ("rm", "--force", "--volumes", self.container))
        self.assertFalse((self.out / "report.json").exists())

    def test_docker_create_timeout_removes_the_identified_container(self):
        self.create_interrupted = True
        with self.assertRaises(subprocess.TimeoutExpired):
            self.check()
        self.assertEqual(self.calls[-2], ("container", "inspect", self.name))
        self.assertEqual(self.calls[-1], ("rm", "--force", "--volumes", self.container))
        self.assertFalse((self.out / "report.json").exists())
