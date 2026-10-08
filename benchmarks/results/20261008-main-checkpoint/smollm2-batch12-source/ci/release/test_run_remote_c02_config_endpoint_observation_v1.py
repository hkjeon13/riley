#!/usr/bin/env python3
"""Static safety checks for the isolated config endpoint bridge wrapper."""

from __future__ import annotations

import os
import stat
import subprocess
import unittest
from pathlib import Path


class RunRemoteC02ConfigEndpointObservationV1Test(unittest.TestCase):
    def test_wrapper_is_executable_isolated_and_never_operates_a_service(self) -> None:
        wrapper = Path(__file__).with_name("run_remote_c02_config_endpoint_observation_v1.sh")
        source = wrapper.read_text(encoding="utf-8")
        self.assertTrue(source.startswith("#!/usr/bin/env bash\n"))
        self.assertTrue(os.stat(wrapper).st_mode & stat.S_IXUSR)
        subprocess.run(["bash", "-n", str(wrapper)], check=True)
        for required in (
            "set -euo pipefail",
            "capture_c02_config_endpoint_observation_v1.py",
            "/usr/bin/env -i",
            "PATH=/usr/bin:/bin",
            "LC_ALL=C",
            "TZ=UTC",
            "PYTHONDONTWRITEBYTECODE=1",
            "/usr/bin/python3 -B -I -S",
        ):
            self.assertIn(required, source)
        for forbidden in (
            "docker ", "podman ", "systemctl ", "ssh ", "nvidia-smi",
            "check_rc3_qualification", "check_soak_v2_receipt", "curl ", "wget ",
        ):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
