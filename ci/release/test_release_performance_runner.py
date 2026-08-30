#!/usr/bin/env python3
"""CPU-only tests for the remote five-run release-performance contract."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
HOST_RUNNER = ROOT / "ci" / "run_remote_release_performance.sh"
CONTAINER_RUNNER = ROOT / "ci" / "release" / "run_release_performance_once.sh"
RUNNER_DOC = ROOT / "ci" / "release" / "RELEASE_PERFORMANCE_RUNNER.md"
PREFLIGHT = ROOT / "benchmarks" / "scripts" / "preflight.sh"
sys.path.insert(0, str(ROOT / "ci" / "release"))

import validate_release_performance_runner as contract


REVISION = "a" * 40
SOURCE_SHA256 = "b" * 64
IMAGE_ID = "sha256:" + "c" * 64
MODEL_TREE_SHA256 = "d" * 64
SUPERVISOR_TOKEN = "9" * 64
IMAGE_LABELS = {
    "maintainer": "NVIDIA CORPORATION <cudatools@nvidia.com>",
    "org.opencontainers.image.ref.name": "ubuntu",
    "org.opencontainers.image.version": "22.04",
}
CAPTURE_IDS = [
    contract.performance._runner_capture_id(SUPERVISOR_TOKEN, pair_index)
    for pair_index in range(1, 6)
]


SUPERVISOR_CONTRACT_MARKERS = (
    "exec /usr/bin/env -i",
    "/usr/bin/python3.10 -I -S -E -c",
    "os.O_NONBLOCK",
    "os.O_CLOEXEC",
    "os.set_inheritable(lock_fd, False)",
    "os.set_inheritable(lock_fd, True)",
    "PR_SET_PDEATHSIG",
    "signal.pthread_sigmask(signal.SIG_BLOCK, forwarded_signals)",
    '[[ ${PPID} == "${RILEY_PERF_SUPERVISOR_PID}" ]]',
    "/proc/${PERF_SUPERVISOR_PID}/fdinfo/${PERF_SUPERVISOR_LOCK_FD}",
    "${fdinfo_type} == FLOCK",
    "${fdinfo_kind} == ADVISORY",
    "${fdinfo_mode} == WRITE",
    '[[ ${parent_flock_pid} == "${PERF_SUPERVISOR_PID}" ]]',
    'eval "exec ${PERF_SUPERVISOR_LOCK_FD}>&-"',
    "close_fds=True",
    "timeout=15",
    "except Exception as error:",
    "if os.WIFEXITED(wait_status) and os.WEXITSTATUS(wait_status) == 0:",
    "org.riley.release-performance-supervisor",
    '"container", "ls", "--all", "--quiet", "--no-trunc"',
    'if container_status in ("exited", "dead"):',
    'if container_status not in ("created", "running", "paused", "restarting", "removing"):',
    '"container", "rm", "--force", "--volumes", container_id',
)


def _assert_static_supervisor_contract(source: str) -> None:
    missing = [marker for marker in SUPERVISOR_CONTRACT_MARKERS if marker not in source]
    if missing:
        raise AssertionError(f"missing supervisor contract marker: {missing[0]}")
    if source.index("os.set_inheritable(lock_fd, True)") > source.index("os.execve("):
        raise AssertionError("child lock descriptor must be made inheritable before Bash exec")
    if source.index('eval "exec ${PERF_SUPERVISOR_LOCK_FD}>&-"') > source.index(
        "for unsafe_name in"
    ):
        raise AssertionError("Bash must close its authentication descriptor before setup")
    if source.index(
        "if os.WIFEXITED(wait_status) and os.WEXITSTATUS(wait_status) == 0:"
    ) > source.index("docker_environment = {"):
        raise AssertionError("normal success must bypass supervisor Docker cleanup")


def _embedded_supervisor_program(source: str) -> str:
    marker = "/usr/bin/python3.10 -I -S -E -c '\n"
    start = source.index(marker) + len(marker)
    end = source.index("\n' \"$0\" \"$@\"", start)
    return source[start:end]


def _write(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _preflight_values() -> dict[str, str]:
    return {
        **contract.FIXED_PREFLIGHT,
        "git_revision": REVISION,
        "memory_used_mib": "0",
        "temperature_c": "35",
        "power_limit_w": "450.00",
        "graphics_clock_mhz": "[N/A]",
        "memory_clock_mhz": "[N/A]",
        "staging_available_bytes": str(30 * 1024**3),
    }


def _optimizer_report() -> dict[str, object]:
    passed_log = "1" * 64
    return {
        "schema_version": 1,
        "gate_id": contract.GATE_ID,
        "recorded_at_utc": "2026-08-26T00:00:00Z",
        "status": "passed",
        "semantic_class": "E0",
        "source": {
            "git_commit": REVISION,
            "git_dirty": False,
            "archive_sha256": SOURCE_SHA256,
        },
        "build": {
            "container_image_sha256": IMAGE_ID.removeprefix("sha256:"),
            "network": "none",
            "cargo_locked": True,
            "cargo_offline": True,
            "rustc": "1.85.0",
            "cuda_toolkit": contract.CUDA_TOOLKIT_VERSION,
            "cuda_architecture": contract.CUDA_ARCHITECTURE,
        },
        "gpu": {
            "model": contract.GPU_NAME,
            "uuid": contract.GPU_UUID,
            "pci_bus_id": contract.GPU_PCI_BUS_ID,
            "compute_capability": contract.GPU_COMPUTE_CAPABILITY,
            "vram_mib": contract.GPU_MEMORY_MIB,
            "driver_version": contract.DRIVER_VERSION,
        },
        "model": {
            "model_id": contract.MODEL_ID,
            "revision": contract.MODEL_REVISION,
            "dtype": "bf16",
            "manifest_sha256": MODEL_TREE_SHA256,
            "weights_sha256": contract.WEIGHTS_SHA256,
            "tokenizer_sha256": contract.TOKENIZER_SHA256,
        },
        "implementations": {
            "baseline": "per-operation",
            "candidate": "iteration-batch",
            "residual_rmsnorm": "separate",
            "rollback": "--execution-completion per-operation",
        },
        "tests": [
            {"id": "cuda-compile-only", "log_sha256": passed_log, "result": "passed"},
            {
                "id": "workspace-all-features-all-targets",
                "log_sha256": passed_log,
                "result": "passed",
            },
            {
                "id": "command-batch-lifecycle",
                "log_sha256": passed_log,
                "result": "passed",
                "one_shot_finish": True,
                "drop_restores_stream": True,
            },
            {
                "id": "command-batch-resource-ledger",
                "log_sha256": passed_log,
                "result": "passed",
                "queued_chain_raw_byte_mismatches": 0,
                "cuda_live_allocation_delta": 0,
                "owner_close_live_allocation_count": 0,
                "validation_fail_closed": True,
                "stream_reuse_after_finish": True,
            },
            {
                "id": "smollm2-multi-step-greedy-exact",
                "log_sha256": passed_log,
                "result": "passed",
                "decode_steps": 16,
                "committed_iterations": 16,
                "generated_token_ids": list(contract.performance.OPTIMIZATION_GOLDEN_TOKEN_IDS),
                "raw_logit_mismatches": 0,
                "token_id_mismatches": 0,
                "cuda_live_allocation_delta": 0,
                "owner_close_live_allocation_count": 0,
            },
            {
                "id": "fixed37-production-batch-e0",
                "log_sha256": passed_log,
                "result": "passed",
                "gate_id": contract.performance.FIXED37_PRODUCTION_BATCH_GATE_ID,
                "fixture_sha256": contract.performance.FIXED37_GOLDEN_FIXTURE_SHA256,
                "generated_token_ids_sha256": contract.performance.FIXED37_GOLDEN_TOKEN_IDS_SHA256,
                "cases": 31,
                "compared_steps": 481,
                "exact_window": 16,
                "fixed_profile": "fixed-contiguous-37-balanced-v1",
                "canonical_profile": "canonical-v1",
                "residual_rmsnorm": "separate",
                "execution_completion": "iteration-batch",
                "fixed_prefill_raw_logit_mismatches": 0,
                "fixed_cached_growing_token_id_mismatches": 0,
                "fixed_cached_growing_cosine_min": contract.performance.FIXED37_CACHED_GROWING_COSINE_MIN,
                "fixed_cached_growing_max_abs_max": contract.performance.FIXED37_CACHED_GROWING_MAX_ABS_MAX,
                "fixed_cached_growing_mean_abs_max": contract.performance.FIXED37_CACHED_GROWING_MEAN_ABS_MAX,
                "fixed_cached_growing_worst_cosine": 0.999,
                "fixed_cached_growing_worst_max_abs": 1.0,
                "fixed_cached_growing_worst_mean_abs": 0.25,
                "fixed_cached_growing_threshold_violations": 0,
                "fixed_golden_token_id_mismatches": 0,
                "canonical_golden_token_id_mismatches": 0,
                "cuda_live_allocation_delta": 0,
                "owner_close_live_allocation_count": 0,
                "compile_command_id": "compile-fixed37-production-batch-e0",
                "execute_command_id": "fixed37-production-batch-e0",
                "compile_log_sha256": "2" * 64,
                "test_binary_sha256": "3" * 64,
            },
        ],
    }


def _gpu() -> dict[str, object]:
    return {
        "name": contract.GPU_NAME,
        "uuid": contract.GPU_UUID,
        "pci_bus_id": contract.GPU_PCI_BUS_ID,
        "driver_version": contract.DRIVER_VERSION,
        "compute_capability": contract.GPU_COMPUTE_CAPABILITY,
        "memory_total_mib": contract.GPU_MEMORY_MIB,
    }


def _container_environment() -> dict[str, str]:
    return {
        "RILEY_PERF_SOURCE_REVISION": REVISION,
        "RILEY_PERF_SOURCE_ARCHIVE_SHA256": SOURCE_SHA256,
        "RILEY_PERF_PROFILE_BINARY_SHA256": "e" * 64,
        "RILEY_PERF_OPTIMIZER_REPORT_SHA256": "f" * 64,
        "RILEY_PERF_OPTIMIZER_IMAGE_SHA256": IMAGE_ID.removeprefix("sha256:"),
        "RILEY_PERF_MODEL_TREE_SHA256": MODEL_TREE_SHA256,
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


def _mount_sources() -> dict[str, str]:
    return {
        "/input/source.tar": "/evidence/source.tar",
        "/input/riley-profile": "/evidence/inputs/riley-profile",
        "/input/optimizer-correctness-report.json": "/evidence/inputs/optimizer.json",
        "/model": "/evidence/inputs/model",
        "/evidence": "/evidence/raw",
    }


def _container_receipt(pair_index: int, *, after: bool) -> list[dict[str, object]]:
    container_id = format(pair_index, "x") * 64
    environment = _container_environment()
    environment["RILEY_PERF_PAIR_INDEX"] = str(pair_index)
    environment["RILEY_PERF_CAPTURE_ID"] = CAPTURE_IDS[pair_index - 1]
    mounts = [
        {
            "Type": "bind",
            "Source": source,
            "Destination": destination,
            "RW": destination == "/evidence",
            "Mode": "",
            "Propagation": "rprivate",
        }
        for destination, source in _mount_sources().items()
    ]
    mounts.append(
        {
            "Type": "volume",
            "Source": f"volume-{pair_index}",
            "Destination": "/workspace",
            "RW": True,
        }
    )
    state: dict[str, object] = {
        "Status": "exited" if after else "created",
        "Running": False,
        "ExitCode": 0,
        "Paused": False,
        "Restarting": False,
        "OOMKilled": False,
        "Dead": False,
        "Error": "",
        "StartedAt": (
            f"2026-08-26T12:00:{(pair_index - 1) * 10 + 1:02d}.000000000Z"
            if after
            else contract.performance.RUNNER_ZERO_TIME
        ),
        "FinishedAt": (
            f"2026-08-26T12:00:{(pair_index - 1) * 10 + 3:02d}.000000000Z"
            if after
            else contract.performance.RUNNER_ZERO_TIME
        ),
    }
    if not after:
        state["Pid"] = 0
    return [
        {
            "Id": container_id,
            "Image": IMAGE_ID,
            "Path": contract.performance.RUNNER_CONTAINER_ENTRYPOINT[0],
            "Args": contract.performance.RUNNER_CONTAINER_CMD,
            "RestartCount": 0,
            "Created": f"2026-08-26T12:00:{(pair_index - 1) * 10:02d}.000000000Z",
            "Config": {
                "Image": IMAGE_ID,
                "User": "0:0",
                "WorkingDir": "/workspace",
                "Entrypoint": ["/bin/bash"],
                "Cmd": contract.performance.RUNNER_CONTAINER_CMD,
                "Healthcheck": {"Test": ["NONE"]},
                "Labels": {
                    **IMAGE_LABELS,
                    contract.performance.RUNNER_SUPERVISOR_LABEL: SUPERVISOR_TOKEN,
                },
                "Env": [f"{name}={value}" for name, value in environment.items()],
            },
            "HostConfig": {
                "NetworkMode": "none",
                "ReadonlyRootfs": True,
                "AutoRemove": False,
                "CapDrop": ["ALL"],
                "CapAdd": None,
                "SecurityOpt": ["no-new-privileges:true"],
                "PidsLimit": 512,
                "Privileged": False,
                "Tmpfs": {"/tmp": "rw,nosuid,nodev,noexec,size=2147483648"},
                "RestartPolicy": {"Name": "no", "MaximumRetryCount": 0},
                "PidMode": "",
                "IpcMode": "private",
                "UTSMode": "",
                "UsernsMode": "",
                "CgroupnsMode": "private",
                "Runtime": "runc",
                "CpuShares": 0,
                "Memory": 0,
                "NanoCpus": 0,
                "CpuPeriod": 0,
                "CpuQuota": 0,
                "CpusetCpus": "",
                "CpusetMems": "",
                "MemoryReservation": 0,
                "MemorySwap": 0,
                "Devices": [],
                "DeviceCgroupRules": None,
                "DeviceRequests": [
                    {
                        "Driver": "",
                        "Count": 0,
                        "DeviceIDs": [contract.GPU_UUID],
                        "Capabilities": [["gpu"]],
                        "Options": {},
                    }
                ],
            },
            "NetworkSettings": {"Networks": {}},
            "Mounts": mounts,
            "State": state,
        }
    ]


def _validate_container_receipts(
    before_paths: list[Path], after_paths: list[Path], **kwargs: object
) -> dict[str, object]:
    return contract.validate_container_receipts(
        before_paths,
        after_paths,
        supervisor_token=SUPERVISOR_TOKEN,
        capture_ids=CAPTURE_IDS,
        image_labels=IMAGE_LABELS,
        **kwargs,
    )


def _validate_gpu_monitors(paths: list[Path]) -> list[dict[str, object]]:
    return contract.validate_gpu_monitors(
        paths,
        capture_ids=CAPTURE_IDS,
        container_ids=[format(index, "x") * 64 for index in range(1, 6)],
    )


class ReleasePerformanceRunnerTests(unittest.TestCase):
    def test_shell_scripts_are_syntactically_valid(self) -> None:
        for script in (HOST_RUNNER, CONTAINER_RUNNER):
            with self.subTest(script=script.name):
                completed = subprocess.run(
                    ["/bin/bash", "-n", str(script)],
                    cwd=ROOT,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
        host_source = HOST_RUNNER.read_text(encoding="utf-8")
        compile(
            _embedded_supervisor_program(host_source),
            "ci/run_remote_release_performance.sh:supervisor",
            "exec",
        )

    def test_host_help_is_cpu_only_and_missing_args_fail(self) -> None:
        help_result = subprocess.run(
            ["/bin/bash", str(HOST_RUNNER), "--help"],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn("--optimizer-image sha256:", help_result.stdout)
        missing = subprocess.run(
            ["/bin/bash", str(HOST_RUNNER)],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(missing.returncode, 2)

    def test_direct_supervised_marker_and_poisoned_environment_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bash_env = Path(directory) / "bash-env"
            bash_env.write_text(
                "export RILEY_PERF_SUPERVISOR_PID=$PPID\n"
                "export RILEY_PERF_SUPERVISOR_EXE=/usr/bin/python3.10\n"
                "export RILEY_PERF_SUPERVISOR_LOCK_FD=9\n"
                "export RILEY_PERF_SUPERVISOR_LOCK_ID=1:1\n"
                f"export RILEY_PERF_SUPERVISOR_TOKEN={'0' * 64}\n",
                encoding="utf-8",
            )
            environment = dict(os.environ)
            environment["BASH_ENV"] = str(bash_env)
            completed = subprocess.run(
                ["/bin/bash", str(HOST_RUNNER), "--gpu-lock-supervised"],
                cwd=ROOT,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("release performance:", completed.stderr)

    def test_static_reviewed_tool_inventory_equals_manifest(self) -> None:
        host = HOST_RUNNER.read_text(encoding="utf-8")
        self.assertIn(
            'test -f "${tool_path}" && test ! -L "${tool_path}" && test -x "${tool_path}"',
            host,
        )
        self.assertNotIn("/usr/bin/awk", host)
        shell_tools = {
            name: {"path": path, "sha256": digest}
            for name, path, digest in re.findall(
                r"^\s*'([^|']+)\|(/(?:usr/)?bin/[^|']+)\|([0-9a-f]{64})'$",
                host,
                flags=re.MULTILINE,
            )
        }
        self.assertEqual(shell_tools, contract.performance.RUNNER_REVIEWED_TOOLS)
        self.assertEqual(
            shell_tools["mawk"],
            {
                "path": "/usr/bin/mawk",
                "sha256": "dc157030a32367742480403025a6f731275b07d039238d167ade535e6f3eb98e",
            },
        )
        absolute_tool_paths = set(
            re.findall(r"/(?:usr/)?bin/[A-Za-z0-9_.+-]+", host)
        )
        # These two exact /bin/bash references are immutable paths inside the
        # reviewed optimizer image, not host executables.
        self.assertEqual(len(re.findall(r"(?<!/usr)/bin/bash", host)), 2)
        absolute_tool_paths.remove("/bin/bash")
        self.assertEqual(
            absolute_tool_paths,
            {receipt["path"] for receipt in shell_tools.values()},
        )
        preflight = PREFLIGHT.read_text(encoding="utf-8")
        preflight_paths = set(
            re.findall(r"/usr/bin/[A-Za-z0-9_.+-]+", preflight)
        ) - {"/usr/bin/env"}
        self.assertEqual(
            preflight_paths,
            {
                shell_tools[name]["path"]
                for name in (
                    "df",
                    "git",
                    "head",
                    "mawk",
                    "nvidia-smi",
                    "sed",
                    "timedatectl",
                    "tr",
                    "uname",
                    "wc",
                )
            },
        )
        for unreviewed in ("$(awk ", "| awk ", "$(nvidia-smi", "$(git ", "$(df "):
            self.assertNotIn(unreviewed, preflight)
        documentation = RUNNER_DOC.read_text(encoding="utf-8")
        for name, receipt in shell_tools.items():
            with self.subTest(tool=name):
                self.assertIn(f"| {name} | `{receipt['path']}` | `{receipt['sha256']}` |", documentation)

    def test_supervisor_fd_and_lifecycle_contract_is_mutation_sensitive(self) -> None:
        host = HOST_RUNNER.read_text(encoding="utf-8")
        _assert_static_supervisor_contract(host)
        self.assertNotIn("--gpu-lock-held", host)
        self.assertNotIn("RILEY_PERF_GPU_LOCK_FD", host)
        for marker in SUPERVISOR_CONTRACT_MARKERS:
            with self.subTest(marker=marker):
                mutated = host.replace(marker, "MUTATED_CONTRACT_MARKER")
                with self.assertRaisesRegex(AssertionError, "missing supervisor contract marker"):
                    _assert_static_supervisor_contract(mutated)

    def test_static_five_fresh_container_contract(self) -> None:
        host = HOST_RUNNER.read_text(encoding="utf-8")
        inner = CONTAINER_RUNNER.read_text(encoding="utf-8")
        self.assertIn("for pair_index in 1 2 3 4 5; do", host)
        self.assertEqual(host.count('container_id=$("${DOCKER_BIN}" create'), 1)
        for required in (
            "/var/tmp/riley-server-4096-gpu-evidence.lock",
            "os.O_APPEND",
            "os.O_NONBLOCK",
            "os.O_CLOEXEC",
            "follow_symlinks=False",
            "${fdinfo_type} == FLOCK",
            '[[ ${parent_flock_pid} == "${PERF_SUPERVISOR_PID}" ]]',
            "require_shared_gpu_lock",
            "--network none",
            "--read-only",
            "--no-healthcheck",
            '--gpus "device=${DESIGNATED_GPU_UUID}"',
            "type=volume,destination=/workspace,volume-nocopy",
            "benchmarks/scripts/preflight.sh",
            '"${GIT_BIN}" status --porcelain=v1 --untracked-files=all',
            '"${GIT_BIN}" get-tar-commit-id',
            "optimizer-image-inspect-after.json",
            "--container-inspect-before",
            "--container-inspect-after",
            '--gpu-monitor "${gpu_monitor_receipts[@]}"',
            '"gpu_monitors": receipt["gpu_monitors"]',
            '"runner_manifest": receipt["manifest"]',
            '"executions": receipt["executions"]',
            'execution-receipt.json',
            '--supervisor-token "${PERF_SUPERVISOR_TOKEN}"',
            '--capture-id "${capture_ids[@]}"',
            '--execution-receipt-output "${execution_receipt_outputs[@]}"',
            'RILEY_PERF_CAPTURE_ID=${capture_id}',
            "/usr/bin/mkdir -m 0733 \"${run_evidence_dir}\"",
            "/usr/bin/find \"${model_snapshot}\" -type d -exec /usr/bin/chmod 0555",
            "/usr/bin/find \"${model_snapshot}\" -type f -exec /usr/bin/chmod 0444",
            "--security-opt no-new-privileges:true",
            "--tmpfs /tmp:rw,nosuid,nodev,noexec,size=2147483648",
        ):
            self.assertIn(required, host)
        self.assertGreaterEqual(host.count("require_exact_clean_checkout"), 4)
        self.assertGreaterEqual(host.count("revalidate_immutable_inputs"), 4)
        self.assertNotIn("rm -rf", host)
        self.assertNotIn('/usr/bin/chmod 0444 "${raw_run}"', host)
        loop = host[host.index("for pair_index in 1 2 3 4 5; do") :]
        self.assertLess(
            loop.index("revalidate_immutable_inputs accepted-preflight"),
            loop.index('container_id=$("${DOCKER_BIN}" create'),
        )
        self.assertLess(
            loop.index("revalidate_immutable_inputs immediate-pre-start"),
            loop.index('"${DOCKER_BIN}" start --attach'),
        )
        self.assertGreater(
            loop.index("revalidate_immutable_inputs post-exit"),
            loop.index('wait "${attach_pid}"'),
        )
        for required in (
            "--role candidate",
            "--runtime-flag-name execution_completion",
            "--runtime-flag-value iteration-batch",
            "--implementation-id native-iteration-command-batch",
            "--concurrency 1",
            "--prompt-tokens 128",
            "--output-tokens 32",
            "--warmups 5",
            "--measured-iterations 30",
            "--sampling-id greedy",
            "--seed none",
            "candidate-${RILEY_PERF_PAIR_INDEX}.json",
            "Ubuntu 22.04.5 LTS",
            '--os-release "${os_pretty_name}"',
            '/usr/bin/chmod 0444 "${output}"',
            ': "${RILEY_PERF_CAPTURE_ID:?missing capture ID}"',
            'date -u +%Y-%m-%dT%H:%M:%S.%NZ',
            '${RILEY_PERF_CAPTURE_ID}-pair${RILEY_PERF_PAIR_INDEX}',
        ):
            self.assertIn(required, inner)
        self.assertNotIn("python", inner.lower())

    def test_permission_modes_are_cpu_only_reproducible(self) -> None:
        """Exercise the host-side POSIX contract without Docker, a model, or a GPU."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            nested = model / "nested"
            nested.mkdir(parents=True)
            weight = nested / "weight.bin"
            weight.write_bytes(b"fixture")
            evidence = root / "run-evidence"
            evidence.mkdir()

            for path in (model, nested):
                path.chmod(0o555)
            weight.chmod(0o444)
            evidence.chmod(0o733)

            self.assertEqual(stat.S_IMODE(model.stat().st_mode), 0o555)
            self.assertEqual(stat.S_IMODE(nested.stat().st_mode), 0o555)
            self.assertEqual(stat.S_IMODE(weight.stat().st_mode), 0o444)
            self.assertEqual(stat.S_IMODE(evidence.stat().st_mode), 0o733)

        documentation = RUNNER_DOC.read_text(encoding="utf-8")
        probe = documentation.split(
            "Replace the image placeholder with the already", 1
        )[1].split("```sh", 1)[1].split("```", 1)[0]
        self.assertIn("ssh server-4096", probe)
        self.assertIn("/usr/bin/docker run", probe)
        self.assertIn("--cap-drop ALL", probe)
        self.assertIn("--network none", probe)
        self.assertNotIn("--gpus", probe)

    def test_model_tree_recheck_detects_mutation_and_accepts_exact_restore(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model"
            model.mkdir()
            weight = model / "weight.bin"
            original = b"reviewed model bytes"
            weight.write_bytes(original)
            expected, count = contract.canonical_model_tree_sha256(model)
            self.assertEqual(count, 1)

            weight.write_bytes(b"temporary replacement")
            changed, _ = contract.canonical_model_tree_sha256(model)
            self.assertNotEqual(changed, expected)

            weight.write_bytes(original)
            restored, _ = contract.canonical_model_tree_sha256(model)
            self.assertEqual(restored, expected)

    def test_gpu_readers_reject_symlinks_and_fifos_without_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "gpu-target.csv"
            target.write_text(", ".join(contract.performance.RUNNER_GPU_ROW) + "\n")
            symlink = root / "gpu-link.csv"
            symlink.symlink_to(target)
            with self.assertRaisesRegex(contract.ContractError, "regular file"):
                contract.validate_gpu_csv(symlink)

            fifo = root / "gpu.fifo"
            os.mkfifo(fifo)
            with self.assertRaisesRegex(contract.ContractError, "regular file"):
                contract.validate_gpu_csv(fifo)

    def test_gpu_reader_open_flags_fail_closed_without_fallbacks(self) -> None:
        for flag in ("O_CLOEXEC", "O_NOFOLLOW", "O_NONBLOCK"):
            with self.subTest(flag=flag), mock.patch.object(contract.os, flag, 0):
                with self.assertRaisesRegex(contract.ContractError, f"os\\.{flag}"):
                    contract._regular_file_read_open_flags()

        source = (ROOT / "ci" / "release" / "validate_release_performance_runner.py").read_text(
            encoding="utf-8"
        )
        for fallback in (
            'getattr(os, "O_CLOEXEC", 0)',
            'getattr(os, "O_NOFOLLOW", 0)',
            'getattr(os, "O_NONBLOCK", 0)',
        ):
            self.assertNotIn(fallback, source)

    def test_gpu_reader_rejects_same_inode_mutation_during_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gpu.csv"
            original = (", ".join(contract.performance.RUNNER_GPU_ROW) + "\n").encode()
            path.write_bytes(original)
            real_read = os.read
            mutated = False

            def mutate_after_first_read(descriptor: int, maximum: int) -> bytes:
                nonlocal mutated
                block = real_read(descriptor, maximum)
                if not mutated:
                    path.write_bytes(original + b"tamper")
                    mutated = True
                return block

            with mock.patch.object(contract.os, "read", side_effect=mutate_after_first_read):
                with self.assertRaisesRegex(contract.ContractError, "changed while it was read"):
                    contract.validate_gpu_csv(path)

    def test_gpu_monitor_receipts_reject_foreign_process_and_power_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths: list[Path] = []
            header = ",".join(contract.performance.RUNNER_GPU_MONITOR_HEADER)
            for pair_index in range(1, 6):
                path = root / f"monitor-{pair_index}.csv"
                path.write_text(
                    header
                    + f"\n{CAPTURE_IDS[pair_index - 1]},{format(pair_index, 'x') * 64},pre_start,0,450.00,[N/A],[N/A],35,0,none"
                    + f"\n{CAPTURE_IDS[pair_index - 1]},{format(pair_index, 'x') * 64},running,1,450.00,[N/A],[N/A],55,1024,container:{1000 + pair_index}"
                    + f"\n{CAPTURE_IDS[pair_index - 1]},{format(pair_index, 'x') * 64},post_exit,2,450.00,[N/A],[N/A],40,0,none\n",
                    encoding="utf-8",
                )
                paths.append(path)
            self.assertEqual(len(_validate_gpu_monitors(paths)), 5)

            paths[2].write_text(
                paths[2].read_text(encoding="utf-8").replace(
                    "container:1003", "foreign:1003"
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(contract.ContractError, "compute_processes"):
                _validate_gpu_monitors(paths)

            paths[2].write_text(
                paths[2].read_text(encoding="utf-8").replace(
                    "foreign:1003", "container:1003"
                ).replace("450.00", "451.00", 1),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(contract.ContractError, "power/application-clock"):
                _validate_gpu_monitors(paths)

    def test_preflight_contract_and_cross_run_clock_stability(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.snapshot"
            second = root / "second.snapshot"
            values = _preflight_values()
            first.write_text(
                "".join(f"{key}={value}\n" for key, value in values.items()),
                encoding="utf-8",
            )
            second.write_text(first.read_text(encoding="utf-8"), encoding="utf-8")
            parsed = contract.validate_preflights([first, second], REVISION)
            self.assertEqual(len(parsed), 2)
            drift = dict(values)
            drift["power_limit_w"] = "451.00"
            second.write_text(
                "".join(f"{key}={value}\n" for key, value in drift.items()),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(contract.ContractError, "power_limit_w"):
                contract.validate_preflights([first, second], REVISION)
            drift = dict(values)
            drift["kernel_release"] = "6.8.0-139-generic"
            second.write_text(
                "".join(f"{key}={value}\n" for key, value in drift.items()),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(contract.ContractError, "kernel_release"):
                contract.validate_preflights([second], REVISION)
            with self.assertRaisesRegex(contract.ContractError, "distinct receipt path"):
                contract.validate_preflights([first, first], REVISION)

    def test_actual_gpu_and_image_facts_are_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gpu_path = root / "gpu.csv"
            gpu_path.write_text(
                f"{contract.GPU_NAME}, {contract.GPU_UUID}, "
                f"{contract.GPU_PCI_BUS_ID}, {contract.GPU_MEMORY_MIB}, "
                f"{contract.DRIVER_VERSION}, {contract.GPU_COMPUTE_CAPABILITY}\n",
                encoding="utf-8",
            )
            self.assertEqual(contract.validate_gpu_csv(gpu_path)["uuid"], contract.GPU_UUID)
            gpu_path.write_text(
                f"{contract.GPU_NAME}, GPU-other, {contract.GPU_PCI_BUS_ID}, "
                f"{contract.GPU_MEMORY_MIB}, {contract.DRIVER_VERSION}, "
                f"{contract.GPU_COMPUTE_CAPABILITY}\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(contract.ContractError, "uuid"):
                contract.validate_gpu_csv(gpu_path)

            inspect_path = root / "inspect.json"
            _write(
                inspect_path,
                [
                    {
                        "Id": IMAGE_ID,
                        "Os": "linux",
                        "Architecture": "amd64",
                        "Config": {
                            "Env": [f"CUDA_VERSION={contract.CUDA_RUNTIME_VERSION}"],
                            "WorkingDir": "/workspace",
                            "Labels": IMAGE_LABELS,
                        },
                    }
                ],
            )
            self.assertEqual(
                contract.validate_image_inspect(inspect_path, IMAGE_ID)["id"], IMAGE_ID
            )
            document = json.loads(inspect_path.read_text(encoding="utf-8"))
            document[0]["Architecture"] = "arm64"
            _write(inspect_path, document)
            with self.assertRaisesRegex(contract.ContractError, "Architecture"):
                contract.validate_image_inspect(inspect_path, IMAGE_ID)
            document[0]["Architecture"] = "amd64"
            document[0]["Config"]["Env"].append("LD_AUDIT=/tmp/audit.so")
            _write(inspect_path, document)
            with self.assertRaisesRegex(contract.ContractError, "LD_AUDIT"):
                contract.validate_image_inspect(inspect_path, IMAGE_ID)

    def test_optimizer_report_is_bound_to_external_inputs_and_actual_gpu(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "optimizer.json"
            report = _optimizer_report()
            _write(path, report)
            report_sha = hashlib.sha256(path.read_bytes()).hexdigest()
            result = contract.validate_optimizer_report(
                path,
                expected_sha256=report_sha,
                revision=REVISION,
                source_archive_sha256=SOURCE_SHA256,
                image_digest=IMAGE_ID.removeprefix("sha256:"),
                model_tree_sha256=MODEL_TREE_SHA256,
                gpu=_gpu(),
            )
            self.assertEqual(result["status"], "passed")

            mutation = copy.deepcopy(report)
            mutation["source"]["git_commit"] = "e" * 40
            _write(path, mutation)
            mutation_sha = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(contract.ContractError, "source"):
                contract.validate_optimizer_report(
                    path,
                    expected_sha256=mutation_sha,
                    revision=REVISION,
                    source_archive_sha256=SOURCE_SHA256,
                    image_digest=IMAGE_ID.removeprefix("sha256:"),
                    model_tree_sha256=MODEL_TREE_SHA256,
                    gpu=_gpu(),
                )

            mutation = copy.deepcopy(report)
            fixed37 = next(
                row
                for row in mutation["tests"]
                if row["id"] == "fixed37-production-batch-e0"
            )
            fixed37["fixed_golden_token_id_mismatches"] = 1
            _write(path, mutation)
            mutation_sha = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(
                contract.ContractError,
                "fixed_golden_token_id_mismatches",
            ):
                contract.validate_optimizer_report(
                    path,
                    expected_sha256=mutation_sha,
                    revision=REVISION,
                    source_archive_sha256=SOURCE_SHA256,
                    image_digest=IMAGE_ID.removeprefix("sha256:"),
                    model_tree_sha256=MODEL_TREE_SHA256,
                    gpu=_gpu(),
                )

            _write(path, report)
            with self.assertRaisesRegex(contract.ContractError, "external SHA"):
                contract.validate_optimizer_report(
                    path,
                    expected_sha256="f" * 64,
                    revision=REVISION,
                    source_archive_sha256=SOURCE_SHA256,
                    image_digest=IMAGE_ID.removeprefix("sha256:"),
                    model_tree_sha256=MODEL_TREE_SHA256,
                    gpu=_gpu(),
                )

    def test_container_receipts_prove_five_fresh_isolated_exit_zero_runs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            before_paths: list[Path] = []
            after_paths: list[Path] = []
            for pair_index in range(1, 6):
                before_path = root / f"before-{pair_index}.json"
                after_path = root / f"after-{pair_index}.json"
                _write(before_path, _container_receipt(pair_index, after=False))
                _write(after_path, _container_receipt(pair_index, after=True))
                before_paths.append(before_path)
                after_paths.append(after_path)
            result = _validate_container_receipts(
                before_paths,
                after_paths,
                image_id=IMAGE_ID,
                expected_environment=_container_environment(),
                expected_mount_sources=_mount_sources(),
            )
            self.assertEqual(result["count"], 5)
            self.assertEqual(len(set(result["distinct_container_ids"])), 5)

            invalid = _container_receipt(3, after=True)
            invalid[0]["State"]["ExitCode"] = 9
            _write(after_paths[2], invalid)
            with self.assertRaisesRegex(contract.ContractError, "exit-zero"):
                _validate_container_receipts(
                    before_paths,
                    after_paths,
                    image_id=IMAGE_ID,
                    expected_environment=_container_environment(),
                    expected_mount_sources=_mount_sources(),
                )

            invalid = _container_receipt(3, after=True)
            invalid[0]["State"]["ExitCode"] = False
            _write(after_paths[2], invalid)
            with self.assertRaisesRegex(contract.ContractError, "exit-zero"):
                _validate_container_receipts(
                    before_paths,
                    after_paths,
                    image_id=IMAGE_ID,
                    expected_environment=_container_environment(),
                    expected_mount_sources=_mount_sources(),
                )

            _write(after_paths[2], _container_receipt(3, after=True))
            duplicate_before = _container_receipt(1, after=False)
            duplicate_after = _container_receipt(1, after=True)
            duplicate_before[0]["Config"]["Env"] = _container_receipt(2, after=False)[0][
                "Config"
            ]["Env"]
            duplicate_after[0]["Config"]["Env"] = _container_receipt(2, after=True)[0][
                "Config"
            ]["Env"]
            _write(before_paths[1], duplicate_before)
            _write(after_paths[1], duplicate_after)
            with self.assertRaisesRegex(contract.ContractError, "distinct fresh"):
                _validate_container_receipts(
                    before_paths,
                    after_paths,
                    image_id=IMAGE_ID,
                    expected_environment=_container_environment(),
                    expected_mount_sources=_mount_sources(),
                )

            _write(before_paths[1], _container_receipt(2, after=False))
            _write(after_paths[1], _container_receipt(2, after=True))
            invalid = _container_receipt(4, after=False)
            invalid[0]["HostConfig"]["NetworkMode"] = "bridge"
            _write(before_paths[3], invalid)
            with self.assertRaisesRegex(contract.ContractError, "NetworkMode"):
                _validate_container_receipts(
                    before_paths,
                    after_paths,
                    image_id=IMAGE_ID,
                    expected_environment=_container_environment(),
                    expected_mount_sources=_mount_sources(),
                )

            invalid = _container_receipt(4, after=False)
            invalid[0]["HostConfig"]["ReadonlyRootfs"] = False
            _write(before_paths[3], invalid)
            with self.assertRaisesRegex(contract.ContractError, "ReadonlyRootfs"):
                _validate_container_receipts(
                    before_paths,
                    after_paths,
                    image_id=IMAGE_ID,
                    expected_environment=_container_environment(),
                    expected_mount_sources=_mount_sources(),
                )

            for field, value in (
                ("SecurityOpt", ["no-new-privileges"]),
                ("Tmpfs", {"/tmp": "rw,nosuid,nodev,noexec,size=2g"}),
                ("CapAdd", []),
                ("CapAdd", ["NET_ADMIN"]),
            ):
                with self.subTest(normalized_docker_field=field):
                    invalid = _container_receipt(4, after=False)
                    invalid[0]["HostConfig"][field] = value
                    _write(before_paths[3], invalid)
                    with self.assertRaisesRegex(contract.ContractError, field):
                        _validate_container_receipts(
                            before_paths,
                            after_paths,
                            image_id=IMAGE_ID,
                            expected_environment=_container_environment(),
                            expected_mount_sources=_mount_sources(),
                        )

            for field, mutate, expected in (
                (
                    "missing-capadd",
                    lambda row: row[0]["HostConfig"].pop("CapAdd"),
                    "HostConfig.CapAdd",
                ),
                (
                    "labels",
                    lambda row: row[0]["Config"]["Labels"].__setitem__(
                        "unreviewed", "label"
                    ),
                    "Config.Labels",
                ),
                (
                    "driver",
                    lambda row: row[0]["HostConfig"]["DeviceRequests"][0].__setitem__(
                        "Driver", "nvidia"
                    ),
                    "DeviceRequests\\[0\\].Driver",
                ),
                (
                    "count",
                    lambda row: row[0]["HostConfig"]["DeviceRequests"][0].__setitem__(
                        "Count", 1
                    ),
                    "without count-based",
                ),
                (
                    "bool-count",
                    lambda row: row[0]["HostConfig"]["DeviceRequests"][0].__setitem__(
                        "Count", False
                    ),
                    "without count-based",
                ),
                (
                    "options",
                    lambda row: row[0]["HostConfig"]["DeviceRequests"][0].__setitem__(
                        "Options", {"capabilities": "all"}
                    ),
                    "without count-based",
                ),
            ):
                with self.subTest(normalized_docker_field=field):
                    invalid = _container_receipt(4, after=False)
                    mutate(invalid)
                    _write(before_paths[3], invalid)
                    with self.assertRaisesRegex(contract.ContractError, expected):
                        _validate_container_receipts(
                            before_paths,
                            after_paths,
                            image_id=IMAGE_ID,
                            expected_environment=_container_environment(),
                            expected_mount_sources=_mount_sources(),
                        )

            invalid = _container_receipt(4, after=False)
            next(
                mount
                for mount in invalid[0]["Mounts"]
                if mount["Destination"] == "/model"
            )["Mode"] = "ro"
            _write(before_paths[3], invalid)
            with self.assertRaisesRegex(contract.ContractError, "access mode, or propagation"):
                _validate_container_receipts(
                    before_paths,
                    after_paths,
                    image_id=IMAGE_ID,
                    expected_environment=_container_environment(),
                    expected_mount_sources=_mount_sources(),
                )

            invalid = _container_receipt(4, after=False)
            invalid[0]["HostConfig"]["DeviceRequests"][0]["DeviceIDs"] = ["GPU-other"]
            _write(before_paths[3], invalid)
            with self.assertRaisesRegex(contract.ContractError, "DeviceIDs"):
                _validate_container_receipts(
                    before_paths,
                    after_paths,
                    image_id=IMAGE_ID,
                    expected_environment=_container_environment(),
                    expected_mount_sources=_mount_sources(),
                )

            invalid = _container_receipt(4, after=True)
            invalid[0]["RestartCount"] = 1
            _write(before_paths[3], _container_receipt(4, after=False))
            _write(after_paths[3], invalid)
            with self.assertRaisesRegex(contract.ContractError, "RestartCount"):
                _validate_container_receipts(
                    before_paths,
                    after_paths,
                    image_id=IMAGE_ID,
                    expected_environment=_container_environment(),
                    expected_mount_sources=_mount_sources(),
                )

            _write(after_paths[3], _container_receipt(4, after=True))
            for field, mutate, expected in (
                (
                    "cmd",
                    lambda row: row[0]["Config"].__setitem__("Cmd", ["-c", "true"]),
                    "Config.Cmd",
                ),
                (
                    "environment",
                    lambda row: row[0]["Config"]["Env"].append("BASH_ENV=/tmp/x"),
                    "Config.Env",
                ),
                (
                    "image",
                    lambda row: row[0].__setitem__("Image", "sha256:" + "9" * 64),
                    "immutable optimizer image",
                ),
            ):
                with self.subTest(field=field):
                    invalid = _container_receipt(4, after=False)
                    mutate(invalid)
                    _write(before_paths[3], invalid)
                    with self.assertRaisesRegex(contract.ContractError, expected):
                        _validate_container_receipts(
                            before_paths,
                            after_paths,
                            image_id=IMAGE_ID,
                            expected_environment=_container_environment(),
                            expected_mount_sources=_mount_sources(),
                        )

            _write(before_paths[3], _container_receipt(4, after=False))
            oom = _container_receipt(4, after=True)
            oom[0]["State"]["OOMKilled"] = True
            _write(after_paths[3], oom)
            with self.assertRaisesRegex(contract.ContractError, "exit-zero"):
                _validate_container_receipts(
                    before_paths,
                    after_paths,
                    image_id=IMAGE_ID,
                    expected_environment=_container_environment(),
                    expected_mount_sources=_mount_sources(),
                )

            duplicate_volume_before = _container_receipt(4, after=False)
            duplicate_volume_after = _container_receipt(4, after=True)
            for receipt in (duplicate_volume_before, duplicate_volume_after):
                workspace = next(
                    mount
                    for mount in receipt[0]["Mounts"]
                    if mount["Destination"] == "/workspace"
                )
                workspace["Source"] = "volume-1"
            _write(after_paths[3], duplicate_volume_after)
            _write(before_paths[3], duplicate_volume_before)
            with self.assertRaisesRegex(contract.ContractError, "workspace volumes"):
                _validate_container_receipts(
                    before_paths,
                    after_paths,
                    image_id=IMAGE_ID,
                    expected_environment=_container_environment(),
                    expected_mount_sources=_mount_sources(),
                )

if __name__ == "__main__":
    unittest.main()
