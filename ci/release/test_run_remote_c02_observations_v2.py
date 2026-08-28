#!/usr/bin/env python3
"""Static checks for the isolated v2 raw-observation wrapper."""

from __future__ import annotations

import unittest
from pathlib import Path


class RunRemoteC02ObservationsV2Test(unittest.TestCase):
    def test_wrapper_uses_isolated_python_and_never_operates_the_service(self) -> None:
        wrapper = Path(__file__).with_name("run_remote_c02_observations_v2.sh")
        source = wrapper.read_text(encoding="utf-8")
        self.assertTrue(source.startswith("#!/usr/bin/env bash\n"))
        self.assertIn("set -euo pipefail", source)
        self.assertIn("capture_c02_observations_v2.py", source)
        self.assertIn("/usr/bin/env -i", source)
        self.assertIn("PATH=/usr/bin:/bin", source)
        self.assertIn("PYTHONDONTWRITEBYTECODE=1", source)
        self.assertIn("/usr/bin/python3 -B -I -S", source)
        for forbidden in (
            "docker ",
            "podman ",
            "systemctl ",
            "ssh ",
            "nvidia-smi",
            "check_rc3_qualification",
            "check_soak_v2_receipt",
            "check_rc3_rollback_receipt",
        ):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
