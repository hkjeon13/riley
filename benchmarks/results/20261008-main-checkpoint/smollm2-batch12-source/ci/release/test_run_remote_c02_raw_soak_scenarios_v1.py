#!/usr/bin/env python3
"""Static guards for the local-only C02 raw scenario capture wrapper."""

from __future__ import annotations

import os
import subprocess
import unittest
from pathlib import Path


class RunRemoteC02RawSoakScenariosV1Tests(unittest.TestCase):
    def test_wrapper_is_static_local_python_only(self) -> None:
        wrapper = Path(__file__).with_name("run_remote_c02_raw_soak_scenarios_v1.sh")
        source = wrapper.read_text(encoding="utf-8")
        self.assertTrue(source.startswith("#!/usr/bin/env bash\n"))
        self.assertIn("set -euo pipefail", source)
        self.assertIn("unset CDPATH", source)
        self.assertIn("capture_c02_raw_soak_scenarios_v1.py", source)
        self.assertIn("/usr/bin/env -i", source)
        self.assertIn("PATH=/usr/bin:/bin", source)
        self.assertIn("LC_ALL=C", source)
        self.assertIn("TZ=UTC", source)
        self.assertIn("PYTHONDONTWRITEBYTECODE=1", source)
        self.assertIn("/usr/bin/python3 -B -I -S", source)
        self.assertIn("-f $PRODUCER && ! -L $PRODUCER", source)
        self.assertTrue(os.access(wrapper, os.X_OK))
        for forbidden in ("docker ", "podman ", "systemctl ", "ssh ", "nvidia-smi", "curl ", "wget "):
            self.assertNotIn(forbidden, source)

    def test_wrapper_has_bash_syntax_and_help_needs_no_capture(self) -> None:
        wrapper = Path(__file__).with_name("run_remote_c02_raw_soak_scenarios_v1.sh")
        subprocess.run(["bash", "-n", str(wrapper)], check=True)
        completed = subprocess.run(["bash", str(wrapper), "--help"], text=True, capture_output=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--scenario-contract", completed.stdout)
        self.assertIn("--audit-dir-name", completed.stdout)
        self.assertIn("--configuration-sha256", completed.stdout)


if __name__ == "__main__":
    unittest.main()
