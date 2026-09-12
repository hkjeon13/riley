"""Local synthetic receipt/identity tests; never execute CUDA or serving lanes."""

import copy
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import build_batch3 as builder
import prepare_batch3_plan as qualifier


class Batch3ScriptsTest(unittest.TestCase):
    def put(self, path, contents):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")
        return path

    def record(self, path, value):
        return self.put(path, json.dumps(value, indent=2) + "\n")

    def commit(self, source, message):
        subprocess.run(["git", "add", "."], cwd=source, check=True, capture_output=True)
        subprocess.run(["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@local",
                        "commit", "--quiet", "-m", message], cwd=source, check=True, capture_output=True)
        return builder.git(source, "rev-parse", "HEAD")

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="riley-batch3-script-test-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name).resolve()
        self.root, self.base, self.model = (self.directory / name for name in ("opt", "base", "model"))
        old_source = self.root / "batch2-source"
        old_source.mkdir(parents=True)
        subprocess.run(["git", "init", "--quiet", str(old_source)], check=True, capture_output=True)
        for relative in builder.FROZEN_FILES:
            self.put(old_source / relative, "frozen fixture " + relative)
        self.put(old_source / builder.SERVICE, "old HTTP source")
        for name in ("check_vllm_profile_run.py", "preflight.sh"):
            self.put(old_source / "benchmarks/scripts" / name, "fixture checker")
        self.old_commit = self.commit(old_source, "frozen fixture")
        self.baseline = {
            "source_commit": self.old_commit,
            "source_files": {name: builder.sha(old_source / name) for name in builder.FROZEN_FILES},
            "binaries": {str(self.put(self.root / "batch2-target/release" / name, "old " + name)):
                         builder.sha(self.root / "batch2-target/release" / name)
                         for name in ("riley", "riley-profile")},
        }
        self.record(self.root / "batch2-build.json", self.baseline)
        source = self.root / "batch3-source"
        subprocess.run(["git", "clone", "--quiet", str(old_source), str(source)], check=True, capture_output=True)
        self.put(source / builder.SERVICE, "new HTTP source")
        service_sha = builder.sha(source / builder.SERVICE)
        self.commit_id = self.commit(source, "HTTP only")
        for module in (builder, qualifier):
            patch = mock.patch.multiple(module, BATCH2_COMMIT=self.old_commit, SERVICE_SHA=service_sha)
            patch.start()
            self.addCleanup(patch.stop)
        self.parity = builder.component_parity(self.root, self.baseline)
        cpu_log = self.put(self.root / "batch3-cpu-tests.log", "\n".join(
            "test service::tests::" + name + " ... ok" for name in builder.CPU_TESTS
        ) + "\ntest result: ok. 3 passed; 0 failed; 1 ignored;\n")
        cpu = {
            "passed": True, "source_commit": self.commit_id, "source_clean": True,
            "service_sha256": service_sha, "argv": builder.CPU_COMMAND, "features": ["server"],
            "target_dir": str(self.root / "batch3-cpu-target"), "log": builder.evidence(cpu_log),
            "test_counts": builder.validate_cpu_log(cpu_log), "required_tests": list(builder.CPU_TESTS),
            "gpu_executed": False, "transport_timing_executed": False,
        }
        cpu_path = self.record(self.root / "batch3-cpu-tests.json", cpu)
        build_log = self.put(self.root / "batch3-build.log", "synthetic build receipt fixture\n")
        self.build = {
            "source_root": str(source), "source_commit": self.commit_id, "source_clean": True,
            "parent_source_commit": self.old_commit, "only_changed_source_files": [builder.SERVICE],
            "service_sha256": service_sha, "source_files": {**self.baseline["source_files"], builder.SERVICE: service_sha},
            "batch2_build": builder.evidence(self.root / "batch2-build.json"),
            "batch2_component_parity": {"count": 17, "all_equal": True, "files": self.parity},
            "gpu_tests_executed": False, "performance_measured": False,
            "binaries": {str(self.put(self.root / "batch3-target/release" / name, "new " + name)):
                         builder.sha(self.root / "batch3-target/release" / name)
                         for name in ("riley", "riley-profile")},
            "build_argv": ["cargo", "build", "--release", "-p", "riley-server", "--features",
                           "server,bench,cuda", "--bin", "riley", "--bin", "riley-profile"],
            "build_environment": {
                "CUDA_HOME": "/data/riley-g04-cuda13", "CUDAToolkit_ROOT": "/data/riley-g04-cuda13",
                "CMAKE": "/data/cmake-3.31.12/bin/cmake", "CMAKE_BUILD_PARALLEL_LEVEL": "4",
                "CARGO_BUILD_JOBS": "4", "LD_LIBRARY_PATH": "/data/riley-g04-cuda13/lib",
            }, "build_log": builder.evidence(build_log), "cpu_tests": builder.evidence(cpu_path),
        }
        self.record(self.root / "batch3-build.json", self.build)
        weights = self.put(self.model / "model.safetensors", "fixture weights")
        tokenizer = self.put(self.model / "tokenizer.json", "fixture tokenizer")
        binding = {
            "source": {"correctness_gate_id": qualifier.GATE}, "input_token_ids": [1] * 128,
            "generated_token_ids": list(range(32)),
            "workload": {"concurrency": 1, "prompt_tokens": 128, "output_tokens": 32,
                         "warmups": 5, "measured_iterations": 30, "sampling_id": "greedy",
                         "weights_sha256": builder.sha(weights), "tokenizer_sha256": builder.sha(tokenizer)},
        }
        binding_path = self.record(self.base / "native-binding.json", binding)
        self.record(self.base / "request.json", {"prompt_token_ids": binding["input_token_ids"]})
        reference_source = self.directory / "reference-source"
        self.put(reference_source / "benchmarks/lanes/vllm/adapter.py", "fixture adapter")
        self.record(self.base / "candidate.json", {"source_root": str(reference_source)})
        for name in ("verification.json", "prompts.jsonl", "vllm_lane.py", "environment.json"):
            self.put(self.base / name, "{}")
        gpu_checks = []
        for name, marker in qualifier.prior.REQUIRED_TESTS.items():
            path = self.put(self.root / ("batch2-" + name + ".log"),
                            "test fixture::" + name + " ... ok\n" + marker
                            + "\ntest result: ok. 1 passed; 0 failed; 0 ignored;\n")
            gpu_checks.append({"name": name, "passed": True, **builder.evidence(path)})
        self.gpu = {
            "passed": True, "source_commit": self.old_commit, "binaries": self.baseline["binaries"],
            "reference_binding_sha256": builder.sha(binding_path), "vllm_reference_tokens_exact": True,
            "full_logits_and_kv_exact": True, "performance_claim_eligible": False, "checks": gpu_checks,
        }
        gpu_path = self.record(self.root / "batch2-gpu-tests.json", self.gpu)
        qualification_path = self.record(self.root / "batch2-qualification.json", {
            "passed": True, "source_commit": self.old_commit, "binaries": self.baseline["binaries"],
            "build": builder.evidence(self.root / "batch2-build.json"), "gpu_tests": builder.evidence(gpu_path),
            "correctness_gate_id": qualifier.GATE, "numerical_profile": qualifier.PROFILE,
        })
        http_argv = [str(self.root / "batch3-target/release/riley"), "serve", "--model", str(self.model)]
        for flag, value in {
            "--model-id": "g04-smol", "--max-active-sequences": "1", "--batch-token-budget": "128",
            "--prefill-chunk-tokens": "128", "--max-sequence-tokens": "160", "--max-output-tokens": "32",
            "--kv-blocks": "10", "--residual-rmsnorm": "separate", "--execution-completion": "iteration-batch",
            "--metadata-transport": "packed-async", "--execution-graph-policy": "require",
            "--graph-numerics": qualifier.PROFILE,
        }.items():
            http_argv.extend((flag, value))
        http_results = []
        for sampling in ("cpu", "gpu-greedy"):
            log = self.put(self.root / "batch3-http-correctness" / ("http-" + sampling + ".log"), "fixture HTTP log")
            http_results.append({
                "sampling": sampling, "full_text_exact": True, "streaming_exact": True,
                "post_disconnect_reuse_exact": True, "invalid_request_statuses": [400, 400, 400],
                "argv": http_argv + ["--sampling-backend", sampling], "server_exit_code": 0,
                "server_log": builder.evidence(log),
            })
        self.record(self.root / "batch3-http-correctness/http-validation.json", {
            "passed": True, "profile": qualifier.PROFILE, "source_commit": self.commit_id,
            "service_sha256": service_sha, "build": builder.evidence(self.root / "batch3-build.json"),
            "binary_sha256": self.build["binaries"][http_argv[0]], "reference_binding": builder.evidence(binding_path),
            "reference_tokens": binding["generated_token_ids"], "performance_trials": 0,
            "gpu_component_tests_rerun": False, "results": http_results,
        })
        interpreter = self.put(self.directory / "reference-python", "fixture executable")
        lane = self.put(self.directory / "reference-lane.py", "fixture lane")
        matrix = self.put(self.directory / "reference-matrix.json", "{}")
        self.http_runner = self.put(self.root / "run_serving_optimization.py", "fixture runner")
        self.engine_runner = self.put(self.root / "run_engine_optimization.py", "fixture runner")
        engine_argv = [str(self.root / "batch2-target/release/riley-profile"), "--model", str(self.model)]
        for flag, value in {"--correctness-gate-id": qualifier.GATE, "--git-commit": self.old_commit,
                            "--git-dirty": "false", "--implementation-id": "g06-p128-prefill-v1",
                            "--correctness-report-sha256": "old", "--executable-sha256": "old",
                            "--run-id": "g06-riley-{index}"}.items():
            engine_argv.extend((flag, value))
        pins = [*self.base.iterdir(), weights, tokenizer, qualification_path]
        self.template = self.record(self.root / "batch2-engine-plan.json", {
            "source_root": str(old_source), "source_commit": self.old_commit,
            "implementation_id": "g06-p128-prefill-v1", "numerical_profile": qualifier.PROFILE,
            "http_lanes": {"riley": {"argv": http_argv + ["--sampling-backend", "cpu"]}},
            "engine_lanes": {"riley": {"argv": engine_argv},
                             "vllm": {"argv": [str(interpreter), str(lane), "--matrix", str(matrix)]}},
            "immutable_files": {str(path): builder.sha(path) for path in pins},
            "reference_checker_python": str(interpreter),
        })

    def prepare(self):
        return qualifier.prepare(self.root, self.base, self.template, self.http_runner, self.engine_runner)

    def test_preparation_keeps_old_gpu_identity_and_new_http_identity(self):
        result = self.prepare()
        self.assertFalse(result["measurement_started"])
        qualification = builder.read(self.root / "batch3-qualification.json")
        self.assertEqual(qualification["source_commit"], self.commit_id)
        self.assertFalse(qualification["gpu_component_tests_rerun_on_batch3"])
        self.assertEqual(qualification["gpu_component_evidence"]["source_commit"], self.old_commit)
        self.assertEqual(qualification["gpu_component_evidence"]["binaries"], self.baseline["binaries"])
        self.assertEqual(qualification["gpu_component_evidence"]["file_count"], 17)
        for mode in ("http", "engine"):
            plan = builder.read(self.root / ("batch3-" + mode + "-plan.json"))
            self.assertEqual(plan["implementation_id"], "g07-http-wakeup-v1")
            self.assertEqual(plan["measurement_mode"], mode)
            self.assertEqual(qualifier.prior.option(plan["engine_lanes"]["riley"]["argv"], "--correctness-gate-id"), qualifier.GATE)
            self.assertEqual(qualifier.prior.option(plan["http_lanes"]["riley"]["argv"], "--prefill-chunk-tokens"), "128")
        before = builder.sha(self.root / "batch3-qualification.json")
        with self.assertRaisesRegex(ValueError, "refusing to replace"):
            self.prepare()
        self.assertEqual(builder.sha(self.root / "batch3-qualification.json"), before)

    def test_gpu4_file_cannot_enter_service_only_snapshot(self):
        source = self.root / "batch3-source"
        self.put(source / "kernels/src/graph_resources.cu", "unwanted GPU4 change")
        subprocess.run(["git", "add", "."], cwd=source, check=True, capture_output=True)
        subprocess.run(["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@local",
                        "commit", "--amend", "--no-edit", "--quiet"], cwd=source, check=True, capture_output=True)
        with self.assertRaisesRegex(ValueError, "change only service.rs"):
            builder.component_parity(self.root, self.baseline)

    def test_new_binary_tampering_is_rejected(self):
        self.put(self.root / "batch3-target/release/riley", "another binary")
        with self.assertRaisesRegex(ValueError, "binary hashes differ"):
            self.prepare()
        self.assertFalse((self.root / "batch3-qualification.json").exists())

    def test_gpu_tests_cannot_be_relabelled_as_new_source(self):
        altered = copy.deepcopy(self.gpu)
        altered["source_commit"] = self.commit_id
        self.record(self.root / "batch2-gpu-tests.json", altered)
        with self.assertRaisesRegex(ValueError, "GPU source identity differs"):
            self.prepare()
        self.assertFalse((self.root / "batch3-qualification.json").exists())

    def test_missing_new_cpu_test_is_rejected(self):
        path = self.put(self.root / "incomplete-cpu.log", "test result: ok. 3 passed; 0 failed; 1 ignored;\n")
        with self.assertRaisesRegex(ValueError, "lifecycle test did not pass"):
            builder.validate_cpu_log(path)

    def test_changed_build_features_are_rejected(self):
        altered = copy.deepcopy(self.build)
        altered["build_argv"][altered["build_argv"].index("--features") + 1] = "server,cuda"
        self.record(self.root / "batch3-build.json", altered)
        with self.assertRaisesRegex(ValueError, "build command differs"):
            self.prepare()


if __name__ == "__main__":
    unittest.main()
