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

    def test_manifest_weight_limit_and_bind_default_can_be_overridden(self):
        checkpoint = self.root / "example"
        checkpoint.mkdir()
        (checkpoint / "riley-checkpoint.json").write_text(
            '{"files": [\n{"bytes": 100},\n{"bytes": 200}\n]}\n', encoding="utf-8",
        )
        self.env["RILEY_BIND"] = "0.0.0.0:9090"
        model = ["serve", "--model", str(checkpoint)]

        for overrides, expected in (
            ([], {"--bind": "0.0.0.0:9090", "--max-weight-bytes": "300", "--model-id": "example"}),
            (["--bind", "127.0.0.1:8181", "--max-weight-bytes", "400", "--model-id", "custom"],
             {"--bind": "127.0.0.1:8181", "--max-weight-bytes": "400", "--model-id": "custom"}),
        ):
            with self.subTest(overrides=overrides):
                result = self.run_entrypoint(*model, *overrides)
                self.assertEqual(result.returncode, 0, result.stderr)
                command, *options = result.stdout.splitlines()
                self.assertEqual(command, "serve")
                self.assertEqual(len(options), 10)
                self.assertEqual(dict(zip(options[::2], options[1::2], strict=True)), {
                    "--model": str(checkpoint),
                    "--inference-profile": "native-bf16-varlen-v1",
                    **expected,
                })

    def test_cached_hub_revision_uses_repository_name_as_api_model_id(self):
        repository = "Qwen/Qwen2.5-7B-Instruct"
        revision = "a" * 40
        model_dir = self.root / "models"
        checkpoint = model_dir / "hub" / repository.replace("/", "--") / revision
        checkpoint.mkdir(parents=True)
        (checkpoint / "riley-checkpoint.json").write_text('{"bytes": 100}\n', encoding="utf-8")
        self.env["RILEY_MODEL_DIR"] = str(model_dir)

        result = self.run_entrypoint("serve", "--model", f"{repository}@{revision}")

        self.assertEqual(result.returncode, 0, result.stderr)
        command, *options = result.stdout.splitlines()
        self.assertEqual(command, "serve")
        self.assertEqual(len(options), 10)
        self.assertEqual(dict(zip(options[::2], options[1::2], strict=True)), {
            "--model": str(checkpoint),
            "--model-id": repository,
            "--inference-profile": "native-bf16-varlen-v1",
            "--max-weight-bytes": "100",
            "--bind": "0.0.0.0:8080",
        })
