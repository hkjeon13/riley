from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
PREFLIGHT = REPOSITORY_ROOT / "benchmarks/scripts/preflight.sh"
HOST_PROFILE = PREFLIGHT.parent / "host_profiles/rtx4090-ubuntu22-driver580-host-v3.env"


NVIDIA_SMI = r"""#!/usr/bin/env sh
case "$*" in
  *--list-gpus*)
    echo 'GPU 0: NVIDIA GeForce RTX 4090 (UUID: GPU-fake)'
    ;;
  *--query-compute-apps*)
    if [ "${FAKE_COMPUTE_QUERY_OK:-yes}" != yes ]; then
      echo 'synthetic compute query failure' >&2
      exit 92
    fi
    ;;
  *--query-gpu=name*)
    printf '%s, 8.9, %s, %s, %s, %s, %s, 450.00, 2520, 10501\n' \
      'NVIDIA GeForce RTX 4090' "${FAKE_MEMORY_TOTAL_MIB:-24564}" \
      "${FAKE_IDLE_MEMORY_MIB:-0}" "${FAKE_DRIVER_VERSION:-580.173.02}" \
      "${FAKE_PERSISTENCE_MODE:-Disabled}" "${FAKE_TEMPERATURE_C:-35}"
    ;;
  *)
    echo "unexpected nvidia-smi argv: $*" >&2
    exit 90
    ;;
esac
"""


TIMEDATECTL = r"""#!/usr/bin/env sh
printf '%s\n' "${FAKE_CLOCK_SYNCHRONIZED:-yes}"
"""


DF = r"""#!/usr/bin/env sh
echo 'Filesystem 1024-blocks Used Available Capacity Mounted on'
printf '/dev/fake 100000000 1 %s 1%% /fake\n' "${FAKE_AVAILABLE_KIB:-31457280}"
"""


GIT = r"""#!/usr/bin/env sh
case "$*" in
  *--is-inside-work-tree*) echo true ;;
  *'rev-parse HEAD'*) echo aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa ;;
  *status*) ;;
  *) echo "unexpected git argv: $*" >&2; exit 91 ;;
esac
"""


UNAME = r"""#!/usr/bin/env sh
case "$1" in
  -r) printf '%s\n' "${FAKE_KERNEL_RELEASE:-6.8.0-138-generic}" ;;
  -m) printf '%s\n' "${FAKE_MACHINE:-x86_64}" ;;
  *) echo "unexpected uname argv: $*" >&2; exit 93 ;;
esac
"""

MAWK = r"""#!/usr/bin/env sh
exec /usr/bin/awk "$@"
"""


class PreflightTests(unittest.TestCase):
    def _program(self, root: Path, name: str, source: str) -> Path:
        path = root / name
        path.write_text(source, encoding="utf-8")
        path.chmod(0o755)
        return path

    def _run(
        self,
        root: Path,
        *,
        clock: str = "yes",
        available_kib: int = 30 * 1024 * 1024,
        persistence: str = "Disabled",
        governor: str = "powersave",
        second_governor: str | None = None,
        governor_policy_count: int = 24,
        memory_total_mib: int = 24_564,
        idle_memory_mib: int = 0,
        driver_version: str = "580.173.02",
        temperature_c: int = 35,
        kernel_release: str = "6.8.0-138-generic",
        ram_kib: int = 65_610_936,
        compute_query_ok: bool = True,
        environment_id: str = "rtx4090-ubuntu22-driver580-v1",
        extra_environment: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        tools = root / "fake-bin"
        tools.mkdir()
        self._program(tools, "nvidia-smi", NVIDIA_SMI)
        self._program(tools, "timedatectl", TIMEDATECTL)
        self._program(tools, "df", DF)
        self._program(tools, "git", GIT)
        self._program(tools, "uname", UNAME)
        self._program(tools, "mawk", MAWK)
        preflight = root / "preflight-under-test.sh"
        profile_directory = root / "host_profiles"
        profile_directory.mkdir()
        (profile_directory / HOST_PROFILE.name).write_text(
            HOST_PROFILE.read_text(encoding="utf-8"), encoding="utf-8"
        )
        preflight_source = PREFLIGHT.read_text(encoding="utf-8")
        for name in ("nvidia-smi", "timedatectl", "df", "git", "uname", "mawk"):
            reviewed_path = f"/usr/bin/{name}"
            self.assertIn(reviewed_path, preflight_source)
            preflight_source = preflight_source.replace(
                reviewed_path, str(tools / name)
            )
        preflight.write_text(preflight_source, encoding="utf-8")
        preflight.chmod(0o755)
        staging = root / "staging"
        staging.mkdir()
        host_root = root / "host"
        (host_root / "etc").mkdir(parents=True)
        (host_root / "proc").mkdir()
        (host_root / "etc/os-release").write_text(
            'ID=ubuntu\nVERSION_ID="22.04"\n', encoding="utf-8"
        )
        cpu_records = []
        for logical in range(24):
            cpu_records.extend(
                (
                    f"processor : {logical}",
                    "model name : 13th Gen Intel(R) Core(TM) i7-13700K",
                    "physical id : 0",
                    f"core id : {logical % 16}",
                    "",
                )
            )
        (host_root / "proc/cpuinfo").write_text(
            "\n".join(cpu_records), encoding="utf-8"
        )
        (host_root / "proc/meminfo").write_text(
            f"MemTotal:       {ram_kib} kB\n", encoding="utf-8"
        )
        governor_root = root / "cpufreq"
        for index in range(governor_policy_count):
            value = second_governor if index == 1 and second_governor else governor
            governor_path = governor_root / f"policy{index}" / "scaling_governor"
            governor_path.parent.mkdir(parents=True)
            governor_path.write_text(value + "\n", encoding="utf-8")
        environment = os.environ.copy()
        environment.update(
            {
                "PATH": str(tools) + os.pathsep + environment.get("PATH", ""),
                "FAKE_CLOCK_SYNCHRONIZED": clock,
                "FAKE_AVAILABLE_KIB": str(available_kib),
                "FAKE_PERSISTENCE_MODE": persistence,
                "FAKE_MEMORY_TOTAL_MIB": str(memory_total_mib),
                "FAKE_IDLE_MEMORY_MIB": str(idle_memory_mib),
                "FAKE_DRIVER_VERSION": driver_version,
                "FAKE_TEMPERATURE_C": str(temperature_c),
                "FAKE_KERNEL_RELEASE": kernel_release,
                "FAKE_COMPUTE_QUERY_OK": "yes" if compute_query_ok else "no",
                "RILEY_CPU_GOVERNOR_ROOT": str(governor_root),
                "RILEY_HOST_ROOT": str(host_root),
                "RILEY_PREFLIGHT_OUTPUT_ROOT": str(staging),
                "RILEY_PREFLIGHT_ENVIRONMENT_ID": environment_id,
            }
        )
        if extra_environment:
            environment.update(extra_environment)
        return subprocess.run(
            ["/bin/bash", str(preflight)],
            cwd=REPOSITORY_ROOT,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

    def test_emits_machine_readable_clock_and_disk_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            completed = self._run(Path(directory))
        self.assertEqual(completed.returncode, 0, completed.stderr)
        values = dict(
            line.split("=", 1) for line in completed.stdout.splitlines() if line
        )
        self.assertEqual(values["driver_version"], "580.173.02")
        self.assertEqual(values["driver_version_prefix"], "580.")
        self.assertEqual(values["persistence_mode"], "Disabled")
        self.assertEqual(values["cpu_governor"], "powersave")
        self.assertEqual(values["cpu_governor_policy_count"], "24")
        self.assertEqual(values["environment_id"], "rtx4090-ubuntu22-driver580-v1")
        self.assertEqual(values["host_profile_schema_version"], "riley.host-profile.v1")
        self.assertEqual(values["host_profile_id"], "rtx4090-ubuntu22-driver580-host-v3")
        self.assertEqual(values["host_profile_version"], "3")
        self.assertEqual(values["os_id"], "ubuntu")
        self.assertEqual(values["os_version_id"], "22.04")
        self.assertEqual(values["kernel_release"], "6.8.0-138-generic")
        self.assertEqual(values["machine"], "x86_64")
        self.assertEqual(values["cpu_model"], "Intel Core i7-13700K")
        self.assertIn("i7-13700K", values["cpu_model_observed"])
        self.assertEqual(values["physical_cpu_cores"], "16")
        self.assertEqual(values["logical_cpu_threads"], "24")
        self.assertEqual(values["ram_bytes"], "67185598464")
        self.assertEqual(values["ram_bytes_target"], "67185594368")
        self.assertEqual(values["ram_bytes_tolerance_bytes"], str(16 * 1024 * 1024))
        self.assertEqual(values["memory_total_mib"], "24564")
        self.assertEqual(values["idle_memory_limit_mib"], "512")
        self.assertEqual(values["start_temperature_limit_c"], "48")
        self.assertEqual(values["clock_synchronized"], "yes")
        self.assertEqual(values["staging_available_bytes"], str(30 * 1024**3))
        self.assertEqual(values["staging_minimum_bytes"], str(20 * 1024**3))

    def test_checked_in_profile_tolerates_boot_page_variation_but_not_a_different_host(self) -> None:
        snapshot = "rtx4090-ubuntu22-driver580-20260911-v2"
        accepted = (
            (snapshot, 65_610_932),
            (snapshot, 65_610_936),
            ("rtx4090-ubuntu22-driver580-v1", 65_610_932),
            ("rtx4090-ubuntu22-driver580-v1", 65_610_932 + 8 * 1024),
        )
        for identity, ram in accepted:
            with self.subTest(identity=identity, ram=ram), tempfile.TemporaryDirectory() as directory:
                result = self._run(Path(directory), environment_id=identity, ram_kib=ram)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("host_profile_id=rtx4090-ubuntu22-driver580-host-v3", result.stdout)
        rejected = (
            (snapshot, 65_610_932 + 16 * 1024 + 1),
            ("arbitrary-host", 65_610_932),
        )
        for identity, ram in rejected:
            with self.subTest(identity=identity, ram=ram), tempfile.TemporaryDirectory() as directory:
                result = self._run(Path(directory), environment_id=identity, ram_kib=ram)
                self.assertNotEqual(result.returncode, 0)

    def test_profile_accepts_the_driver_branch_and_bounded_desktop_idle_memory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = self._run(
                Path(directory), driver_version="580.200.01", idle_memory_mib=512
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        for label, overrides in {
            "driver branch": {"driver_version": "581.1.0"},
            "idle memory": {"idle_memory_mib": 513},
        }.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                result = self._run(Path(directory), **overrides)
                self.assertNotEqual(result.returncode, 0)
                self.assertTrue(result.stderr.startswith("preflight:"))

    def test_ambient_host_policy_overrides_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = self._run(
                Path(directory), extra_environment={"RILEY_MAX_IDLE_MEMORY_MIB": "999999"}
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("checked-in host profile revision", result.stderr)

    def test_primary_environment_checks_fail_closed(self) -> None:
        cases = {
            "clock": {"clock": "no"},
            "disk": {"available_kib": 19 * 1024 * 1024},
            "persistence": {"persistence": "Enabled"},
            "governor": {"governor": "performance"},
            "one governor": {"second_governor": "performance"},
            "governor policy count": {"governor_policy_count": 23},
            "GPU memory": {"memory_total_mib": 24_563},
            "kernel": {"kernel_release": "6.8.0-139-generic"},
            "RAM": {"ram_kib": 65_610_932 + 16 * 1024 + 1},
            "compute query": {"compute_query_ok": False},
        }
        for label, overrides in cases.items():
            with self.subTest(label), tempfile.TemporaryDirectory() as directory:
                completed = self._run(Path(directory), **overrides)
                self.assertNotEqual(completed.returncode, 0)
                self.assertTrue(completed.stderr.startswith("preflight:"))


if __name__ == "__main__":
    unittest.main()
