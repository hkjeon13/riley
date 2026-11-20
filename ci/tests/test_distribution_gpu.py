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
    peer = "GPU-d9e89dc4-e4e2-00d2-8a22-9d696d5b9ff0"
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
        self.tp_workload = None
        self.peer_busy = False
        self.api_pass = True
        self.smoke_disconnected = True
        self.soak_growth_kib = 0
        self.create_interrupted = False
        self.name = None
        self.calls = []
        self.client_commands = {}
        self.out = self.root / "evidence"

    def docker(self, *args):
        self.calls.append(args)
        match args:
            case ("image", "inspect", self.image):
                return json.dumps([self.image_inspect])
            case ("create", *rest):
                self.assertIn("--pull=never", rest)
                requested = f'device={self.gpu}' if self.tp_workload is None else f'"device={self.gpu},{self.peer}"'
                self.assertIn(requested, rest)
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
                return f"{self.gpu}, A100, 550\n" + (f"{self.peer}, A100, 550\n" if self.tp_workload else "")
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
                self.client_commands[file] = argv
                self.assertIn("127.0.0.1:44321", argv)
                (out / "api.json").write_text(json.dumps({"results": [{"passed": self.api_pass}]}))
            if file == "mixed-stage.log":
                self.assertIn("127.0.0.1:44321", argv)
                (out / "mixed-stage.json").write_text(json.dumps({
                    "status": "passed", "prefill_chunk_tokens": 256,
                    "trials": [{"matching_responses": 2, "witnesses": [{}]} for _ in range(3)],
                }))
            if file == "tp-holdout.log":
                self.assertIn(str(self.tp_workload.resolve()), argv)
                (out / "tp-holdout.json").write_text(json.dumps({
                    "status": "passed", "workload_sha256": checker.digest(self.tp_workload),
                    "modes": {"c1": {"requests": 11}, "c8": {"requests": 11, "exact_matches": 11},
                              "c32": {"requests": 32, "exact_matches": 32}},
                }))
            if file == "smoke.log":
                self.client_commands[file] = argv
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
                    "expect_tools_unsupported": "--expect-tools-unsupported" in argv,
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

        def visible_pids(uuid):
            if uuid == self.peer and self.peer_busy:
                return {9000}
            return self.device_pids

        with (patch.object(checker, "docker", self.docker), patch.object(checker, "holder_pids", return_value=set()),
              patch.object(checker, "gpu_pids", side_effect=visible_pids),
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

    def test_serving_smoke_can_use_a_reviewed_smaller_kv_pool(self):
        self.assertEqual(self.check(serving_smoke=True, smoke_kv_blocks=2576), 0)
        report = json.loads((self.out / "report.json").read_bytes())
        create = next(call for call in self.calls if call[0] == "create")
        self.assertEqual(list(create[-10:]), report["serving_smoke"]["server_arguments"])
        self.assertEqual(list(create[-4:]), ["--kv-blocks", "2576", "--prefix-cache-blocks", "128"])

    def test_mixed_stage_binds_the_observation_to_the_server_capacity_and_receipt(self):
        self.assertEqual(self.check(mixed_stage=True), 0)
        report = json.loads((self.out / "report.json").read_bytes())
        mixed = report["mixed_stage"]
        self.assertEqual(mixed["matching_responses"], 6)
        self.assertEqual(mixed["observed_mixed_iterations"], 3)
        create = next(call for call in self.calls if call[0] == "create")
        self.assertEqual(list(create[-16:]), mixed["server_arguments"])
        self.assertEqual(list(create[-6:-2]), ["--batch-token-budget", "512", "--prefill-chunk-tokens", "256"])
        self.assertEqual(create[-2], "--batch-shape-buckets")
        self.assertEqual(create[-1].split(",")[-1], "512")
        self.assertEqual(report["artifacts_sha256"]["mixed-stage.json"], checker.digest(self.out / "mixed-stage.json"))

    def test_tp2_holdout_binds_both_devices_and_exact_responses(self):
        self.tp_workload = self.root / "holdout.json.gz"
        self.tp_workload.write_bytes(b"pinned workload")
        self.assertEqual(self.check(tp_holdout=(self.peer, self.tp_workload)), 0)
        report = json.loads((self.out / "report.json").read_bytes())["tensor_parallel"]
        self.assertEqual(report["gpu_uuids"], [self.gpu, self.peer])
        create = next(call for call in self.calls if call[0] == "create")
        self.assertIn(f"CUDA_VISIBLE_DEVICES={self.gpu},{self.peer}", create)
        self.assertEqual(list(create[-12:]), report["server_arguments"])
        self.assertEqual(report["holdout"]["modes"]["c32"]["exact_matches"], 32)

    def test_tp2_holdout_rejects_a_busy_peer_before_creating_the_container(self):
        workload = self.root / "holdout.json.gz"
        workload.write_bytes(b"pinned workload")
        self.peer_busy = True
        with self.assertRaisesRegex(ValueError, "unexpected process before startup"):
            self.check(tp_holdout=(self.peer, workload))
        self.assertFalse(self.out.exists())
        self.assertFalse(any(call[0] == "create" for call in self.calls))

    def test_smoke_kv_pool_requires_smoke_and_positive_capacity(self):
        for options in ({"smoke_kv_blocks": 2576}, {"serving_smoke": True, "smoke_kv_blocks": 0}):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, "requires serving smoke"):
                self.check(**options)
        self.assertFalse(self.calls)
        self.assertFalse(self.out.exists())

    def test_serving_smoke_checks_unsupported_tools_in_api_and_mixed_traffic(self):
        self.assertEqual(self.check(serving_smoke=True, expect_tools_unsupported=True), 0)
        for argv in self.client_commands.values():
            self.assertIn("--expect-tools-unsupported", argv)
            self.assertNotIn("--tools", argv)
        report = json.loads((self.out / "report.json").read_bytes())
        self.assertIs(report["serving_smoke"]["expect_tools_unsupported"], True)

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
