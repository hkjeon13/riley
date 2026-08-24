from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import tomllib
import unittest
from argparse import Namespace
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

from rustinfer_reference.cli import _build_parser, _run_benchmark, main
from rustinfer_reference.fixture import (
    CacheParityError,
    FixtureValidationError,
    FIXTURE_SOURCE_PATHS,
    collect_fixture_provenance,
    generate_fixture,
    load_prompts,
    validate_fixture_against_prompts,
    validate_fixture_against_repository,
    validate_fixture,
    write_fixture_exclusive,
)
from rustinfer_reference.environment import PRIMARY_ENVIRONMENT_SNAPSHOT

from .support import FakeBackend, fixture_provenance, prompt_rows, write_prompts


FIXED_TIME = datetime(2026, 8, 24, 1, 2, 3, tzinfo=timezone.utc)


class FixtureAndCliTests(unittest.TestCase):
    @property
    def repository(self) -> Path:
        return Path(__file__).resolve().parents[4]

    def _validate_with_repository_schema(self, fixture: dict[str, object]) -> None:
        validator_path = self.repository / "benchmarks/scripts/validate_contract.py"
        spec = importlib.util.spec_from_file_location(
            "reference_fixture_contract_validator", validator_path
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        schema = json.loads(
            (self.repository / "benchmarks/schemas/reference-fixture.schema.json").read_text(
                encoding="utf-8"
            )
        )
        module.validate_instance(fixture, schema)

    def test_prompt_corpus_coverage_and_digest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.jsonl"
            write_prompts(path)
            prompts, digest = load_prompts(path)
            self.assertEqual(len(prompts), 7)
            self.assertEqual(digest, hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(prompts[4].target_prompt_tokens, 7168)

    def test_missing_required_coverage_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.jsonl"
            write_prompts(path)
            rows = path.read_text(encoding="utf-8").splitlines()
            path.write_text("\n".join(rows[:-1]) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "early EOS"):
                load_prompts(path)

    def test_prompt_loader_rejects_zero_target_length(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.jsonl"
            rows = prompt_rows()
            rows[0]["target_prompt_tokens"] = 0
            path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, r"\[1, 8192\]"):
                load_prompts(path)

    def test_fixture_generation_is_deterministic_and_semantically_valid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            prompts_path = Path(directory) / "prompts.jsonl"
            write_prompts(prompts_path)
            prompts, digest = load_prompts(prompts_path)
            backend = FakeBackend()
            first = generate_fixture(
                prompts,
                digest,
                backend,
                max_new_tokens=2,
                hidden_state_index=1,
                top_k=3,
                seed=42,
                provenance=fixture_provenance(),
                created_at=FIXED_TIME,
            )
            second = generate_fixture(
                prompts,
                digest,
                backend,
                max_new_tokens=2,
                hidden_state_index=1,
                top_k=3,
                seed=42,
                provenance=fixture_provenance(),
                created_at=FIXED_TIME,
            )
            self.assertEqual(first, second)
            validate_fixture(first)
            validate_fixture_against_prompts(first, prompts, digest)
            self._validate_with_repository_schema(first)
            boundary = first["cases"][4]
            self.assertEqual(boundary["input"]["token_count"], 7168)
            self.assertEqual(
                boundary["processed_log_probs"]["pipeline_id"],
                "log-softmax-fp32-v1",
            )
            self.assertNotEqual(
                boundary["processed_log_probs"]["tensor"]["sha256"],
                boundary["final_logits"]["sha256"],
            )
            self.assertEqual(boundary["rng"]["stream_id"], "boundary")
            self.assertEqual(boundary["rng"]["draws_consumed"], 0)

    def test_fixture_cross_validation_rejects_stale_prompt_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            prompts_path = Path(directory) / "prompts.jsonl"
            write_prompts(prompts_path)
            prompts, digest = load_prompts(prompts_path)
            fixture = generate_fixture(
                prompts,
                digest,
                FakeBackend(),
                max_new_tokens=1,
                hidden_state_index=1,
                top_k=2,
                seed=0,
                provenance=fixture_provenance(),
                created_at=FIXED_TIME,
            )
            rows = prompts_path.read_text(encoding="utf-8")
            prompts_path.write_text(
                rows.replace("The sky is", "The ocean is", 1), encoding="utf-8"
            )
            changed_prompts, changed_digest = load_prompts(prompts_path)
            with self.assertRaisesRegex(FixtureValidationError, "corpus.sha256"):
                validate_fixture_against_prompts(
                    fixture, changed_prompts, changed_digest
                )

    def test_fixture_provenance_replays_recorded_commit_and_current_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            for relative in FIXTURE_SOURCE_PATHS.values():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f"fixture source: {relative}\n", encoding="utf-8")
            (root / FIXTURE_SOURCE_PATHS["python_version_file"]).write_text(
                "3.13.15\n", encoding="utf-8"
            )
            prompts_path = root / FIXTURE_SOURCE_PATHS["prompts"]
            write_prompts(prompts_path)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            subprocess.run(["git", "add", "."], cwd=root, check=True)
            subprocess.run(
                [
                    "git",
                    "-c",
                    "user.name=fixture-test",
                    "-c",
                    "user.email=fixture-test@example.invalid",
                    "commit",
                    "-q",
                    "-m",
                    "fixture sources",
                ],
                cwd=root,
                check=True,
            )
            provenance = collect_fixture_provenance(
                root, environment_probe=lambda: copy.deepcopy(PRIMARY_ENVIRONMENT_SNAPSHOT)
            )
            prompts, digest = load_prompts(prompts_path)
            fixture = generate_fixture(
                prompts,
                digest,
                FakeBackend(),
                max_new_tokens=1,
                hidden_state_index=1,
                top_k=2,
                seed=0,
                provenance=provenance,
                created_at=FIXED_TIME,
            )
            validate_fixture_against_repository(fixture, root)
            changed = root / FIXTURE_SOURCE_PATHS["constants"]
            changed.write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(FixtureValidationError, "current source bytes"):
                validate_fixture_against_repository(fixture, root)
            with self.assertRaisesRegex(ValueError, "clean Git worktree"):
                collect_fixture_provenance(
                    root,
                    environment_probe=lambda: copy.deepcopy(
                        PRIMARY_ENVIRONMENT_SNAPSHOT
                    ),
                )

    def test_validator_rejects_cache_and_rng_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            prompts_path = Path(directory) / "prompts.jsonl"
            write_prompts(prompts_path)
            prompts, digest = load_prompts(prompts_path)
            fixture = generate_fixture(
                prompts,
                digest,
                FakeBackend(),
                max_new_tokens=1,
                hidden_state_index=1,
                top_k=2,
                seed=7,
                provenance=fixture_provenance(),
                created_at=FIXED_TIME,
            )
            changed = copy.deepcopy(fixture)
            changed["cases"][0]["greedy"]["cache_off_token_ids"] = [123]
            with self.assertRaisesRegex(FixtureValidationError, "cache parity"):
                validate_fixture(changed)
            changed = copy.deepcopy(fixture)
            changed["cases"][0]["rng"]["initial_snapshot"][
                "derivation_digest_hex"
            ] = "0" * 64
            with self.assertRaisesRegex(FixtureValidationError, "identity mismatch"):
                validate_fixture(changed)

    def test_generation_aborts_on_cache_divergence(self) -> None:
        class DivergingBackend(FakeBackend):
            def generate_case(self, *args, **kwargs):
                result = super().generate_case(*args, **kwargs)
                return replace(result, cache_off_token_ids=(123,))

        with tempfile.TemporaryDirectory() as directory:
            prompts_path = Path(directory) / "prompts.jsonl"
            write_prompts(prompts_path)
            prompts, digest = load_prompts(prompts_path)
            with self.assertRaises(CacheParityError):
                generate_fixture(
                    prompts,
                    digest,
                    DivergingBackend(),
                    max_new_tokens=1,
                    hidden_state_index=1,
                    top_k=2,
                    seed=0,
                    provenance=fixture_provenance(),
                    created_at=FIXED_TIME,
                )

    def test_generation_rejects_non_primary_gpu_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            prompts_path = Path(directory) / "prompts.jsonl"
            write_prompts(prompts_path)
            prompts, digest = load_prompts(prompts_path)
            backend = FakeBackend()
            backend.metadata = replace(backend.metadata, device_name="Different GPU")
            with self.assertRaisesRegex(ValueError, "backend GPU"):
                generate_fixture(
                    prompts,
                    digest,
                    backend,
                    max_new_tokens=1,
                    hidden_state_index=1,
                    top_k=2,
                    seed=0,
                    provenance=fixture_provenance(),
                    created_at=FIXED_TIME,
                )

    def test_generation_rejects_non_exact_python_runtime_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            prompts_path = Path(directory) / "prompts.jsonl"
            write_prompts(prompts_path)
            prompts, digest = load_prompts(prompts_path)
            backend = FakeBackend()
            backend.metadata = replace(backend.metadata, python_version="3.13.14")
            with self.assertRaisesRegex(ValueError, "Python version"):
                generate_fixture(
                    prompts,
                    digest,
                    backend,
                    max_new_tokens=1,
                    hidden_state_index=1,
                    top_k=2,
                    seed=0,
                    provenance=fixture_provenance(),
                    created_at=FIXED_TIME,
                )

    def test_cli_generate_defaults_offline_and_never_overwrites(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            root = workspace / "repo"
            prompts = root / "benchmarks/prompts.jsonl"
            output = workspace / "staging/fixture.json"
            prompts.parent.mkdir(parents=True)
            write_prompts(prompts)
            calls: list[dict[str, object]] = []

            def provenance(
                _root: Path, *, observed_environment
            ) -> dict[str, object]:
                value = fixture_provenance()
                value["observed_environment"] = observed_environment
                value["sources"]["prompts"]["sha256"] = hashlib.sha256(
                    prompts.read_bytes()
                ).hexdigest()
                return value

            def factory(**kwargs):
                calls.append(kwargs)
                return FakeBackend(local_files_only=kwargs["local_files_only"])

            result = main(
                [
                    "generate",
                    "--prompts",
                    str(prompts),
                    "--output",
                    str(output),
                    "--repo-root",
                    str(root),
                    "--max-new-tokens",
                    "1",
                    "--top-k",
                    "2",
                ],
                backend_factory=factory,
                provenance_factory=provenance,
                environment_probe=lambda: copy.deepcopy(PRIMARY_ENVIRONMENT_SNAPSHOT),
                now=lambda: FIXED_TIME,
            )
            self.assertEqual(result, 0)
            self.assertEqual(calls, [{"device": "cuda:0", "local_files_only": True}])
            before = output.read_bytes()
            self.assertEqual(
                main(
                    [
                        "generate",
                        "--prompts",
                        str(prompts),
                        "--output",
                        str(output),
                        "--repo-root",
                        str(root),
                    ],
                    backend_factory=factory,
                    provenance_factory=provenance,
                    environment_probe=lambda: copy.deepcopy(
                        PRIMARY_ENVIRONMENT_SNAPSHOT
                    ),
                ),
                2,
            )
            self.assertEqual(output.read_bytes(), before)
            self.assertEqual(len(calls), 1, "overwrite rejection must precede model load")
            self.assertEqual(
                main(
                    [
                        "validate",
                        str(output),
                        "--prompts",
                        str(prompts),
                        "--repo-root",
                        str(root),
                    ],
                    repository_validator=lambda *_: None,
                ),
                0,
            )
            prompts.write_text(
                prompts.read_text(encoding="utf-8").replace(
                    "The sky is", "The ocean is", 1
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                main(
                    [
                        "validate",
                        str(output),
                        "--prompts",
                        str(prompts),
                        "--repo-root",
                        str(root),
                    ],
                    repository_validator=lambda *_: None,
                ),
                2,
            )

    def test_exclusive_writer_refuses_existing_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.json"
            path.write_text("owned by user\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "overwrite"):
                write_fixture_exclusive(path, {})
            self.assertEqual(path.read_text(encoding="utf-8"), "owned by user\n")

    def test_cli_import_does_not_import_heavy_dependencies(self) -> None:
        package_root = Path(__file__).resolve().parents[1]
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(package_root)
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; import rustinfer_reference.cli; "
                "assert 'torch' not in sys.modules; "
                "assert 'transformers' not in sys.modules; "
                "assert 'pynvml' not in sys.modules; "
                "assert 'psutil' not in sys.modules",
            ],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_reference_project_pins_observability_dependencies(self) -> None:
        project = tomllib.loads(
            (self.repository / "tools/python/reference/pyproject.toml").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            project["project"]["dependencies"],
            [
                "nvidia-ml-py==13.610.43",
                "psutil==7.2.2",
                "safetensors==0.8.0",
                "torch==2.13.0",
                "transformers==5.15.1",
            ],
        )

    def test_hf_backend_import_keeps_benchmark_sampler_dependencies_lazy(self) -> None:
        package_root = self.repository / "tools/python/reference"
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(package_root)
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; import rustinfer_reference.hf_backend; "
                "assert 'pynvml' not in sys.modules; "
                "assert 'psutil' not in sys.modules",
            ],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_benchmark_cli_requires_shared_independent_run_id(self) -> None:
        arguments = [
            "benchmark",
            "--matrix",
            "matrix.yaml",
            "--prompts",
            "prompts.jsonl",
            "--result-dir",
            "results/run-1-cell",
            "--warm-state",
            "cold",
            "--concurrency",
            "1",
            "--prompt-tokens",
            "128",
            "--output-tokens",
            "32",
        ]
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                _build_parser().parse_args(arguments)
        parsed = _build_parser().parse_args(
            [*arguments, "--run-id", "hf-transformers-run-1"]
        )
        self.assertEqual(parsed.run_id, "hf-transformers-run-1")

    def test_benchmark_command_enables_observability_sampler(self) -> None:
        factory_calls: list[dict[str, object]] = []

        def factory(**kwargs):
            factory_calls.append(kwargs)
            return object()

        def fake_run_benchmark(**kwargs):
            kwargs["backend_factory"](
                device=kwargs["device"],
                local_files_only=kwargs["local_files_only"],
            )
            return 1

        arguments = Namespace(
            matrix=Path("matrix.yaml"),
            prompts=Path("prompts.jsonl"),
            result_dir=Path("results/cell"),
            device="cuda:0",
            allow_download=False,
            run_index=1,
            run_id="hf-transformers-run-1",
            warm_state="cold",
            concurrency=1,
            prompt_tokens=128,
            output_tokens=32,
        )
        with mock.patch(
            "rustinfer_reference.benchmark.run_benchmark",
            side_effect=fake_run_benchmark,
        ):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    _run_benchmark(
                        arguments,
                        backend_factory=factory,
                        now=lambda: FIXED_TIME,
                    ),
                    0,
                )
        self.assertEqual(
            factory_calls,
            [
                {
                    "device": "cuda:0",
                    "local_files_only": True,
                    "enable_observability": True,
                }
            ],
        )

    def test_reference_schema_is_valid_json_and_covers_fixture_keys(self) -> None:
        schema = json.loads(
            (self.repository / "benchmarks/schemas/reference-fixture.schema.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(schema["$schema"], "https://json-schema.org/draft/2020-12/schema")
        self.assertEqual(
            schema["$defs"]["generation"]["properties"]["max_new_tokens"]["maximum"],
            16,
        )
        self.assertIn("processed_log_probs", schema["$defs"]["case"]["required"])
        self.assertIn("rng", schema["$defs"]["case"]["required"])
        self.assertEqual(
            set(schema["$defs"]["provenance"]["properties"]["sources"]["required"]),
            set(FIXTURE_SOURCE_PATHS),
        )


if __name__ == "__main__":
    unittest.main()
