#!/usr/bin/env python3
"""Static guards for the local-only C02 raw soak manifest wrapper."""

from __future__ import annotations

import unittest
from pathlib import Path


class RunBindRawC02SoakV2Test(unittest.TestCase):
    def test_wrapper_has_one_static_local_python_entry_point(self) -> None:
        wrapper = Path(__file__).with_name("run_bind_raw_c02_soak_v2.sh")
        source = wrapper.read_text(encoding="utf-8")
        self.assertTrue(source.startswith("#!/usr/bin/env bash\n"))
        self.assertIn("set -euo pipefail", source)
        self.assertIn("bind_raw_c02_soak_v2.py", source)
        self.assertIn("/usr/bin/env -i", source)
        self.assertIn("PATH=/usr/bin:/bin", source)
        self.assertIn("LC_ALL=C", source)
        self.assertIn("TZ=UTC", source)
        self.assertIn("PYTHONDONTWRITEBYTECODE=1", source)
        self.assertIn("PYTHONPATH=", source)
        self.assertIn("/usr/bin/python3 -B -S", source)
        for forbidden in (
            "docker ",
            "podman ",
            "systemctl ",
            "ssh ",
            "nvidia-smi",
            "curl ",
            "wget ",
            "check_rc3_qualification",
            "check_soak_v2_receipt",
        ):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
