"""Keep the distribution wrapper's serve help and model defaults usable."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ENTRYPOINT = Path(__file__).resolve().parents[2] / "deploy/docker/entrypoint.sh"


class DistributionEntrypointTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        binary = self.root / "riley"
        binary.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\"\n", encoding="utf-8")
        binary.chmod(0o755)
        self.env = {"PATH": f"{self.root}{os.pathsep}{os.defpath}"}

    def run_entrypoint(self, *arguments):
        return subprocess.run(
            ["sh", str(ENTRYPOINT), *arguments],
            env=self.env, capture_output=True, text=True, timeout=5,
        )

    def test_serve_help_reaches_riley_without_a_model(self):
        for flag in ("--help", "-h"):
            with self.subTest(flag=flag):
                result = self.run_entrypoint("serve", flag)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.splitlines(), ["serve", flag])

    def test_serve_still_requires_a_model(self):
        result = self.run_entrypoint("serve")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("serve needs --model", result.stderr)

    def test_local_model_still_gets_defaults_and_an_explicit_profile_wins(self):
        checkpoint = self.root / "example"
        checkpoint.mkdir()
        result = self.run_entrypoint("serve", "--model", str(checkpoint), "--inference-profile", "native-bf16-v1")
        self.assertEqual(result.returncode, 0, result.stderr)
        command, *options = result.stdout.splitlines()
        self.assertEqual(command, "serve")
        self.assertEqual(len(options), 8)
        self.assertEqual(dict(zip(options[::2], options[1::2], strict=True)), {
            "--bind": "0.0.0.0:8080",
            "--model-id": "example",
            "--model": str(checkpoint),
            "--inference-profile": "native-bf16-v1",
        })
