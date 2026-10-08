from __future__ import annotations

import copy
import importlib.util
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from riley_reference.benchmark import _validate_matrix, run_benchmark

from .support import FakeBackend, write_prompts


FIXED_TIME = datetime(2026, 8, 24, 2, 3, 4, tzinfo=timezone.utc)


class BenchmarkTests(unittest.TestCase):
    @property
    def repository(self) -> Path:
        return Path(__file__).resolve().parents[4]

    def _run(self, root: Path, *, warm_state: str) -> tuple[list[dict], FakeBackend]:
        prompts = root / "prompts.jsonl"
        result_dir = root / f"result-{warm_state}"
        write_prompts(prompts)
        backend = FakeBackend()

        def factory(**kwargs):
            self.assertEqual(kwargs["device"], "cuda:0")
            self.assertTrue(kwargs["local_files_only"])
            return backend

        timer_values = iter((10.0, 10.125))
        row_count = run_benchmark(
            matrix_path=self.repository / "benchmarks/matrix.yaml",
            prompts_path=prompts,
            result_dir=result_dir,
            backend_factory=factory,
            device="cuda:0",
            local_files_only=True,
            run_index=2,
            run_id="fake-run-2",
            warm_state_filter=warm_state,
            concurrency_filter=2,
            prompt_tokens_filter=128,
            output_tokens_filter=32,
            now=lambda: FIXED_TIME,
            timer=lambda: next(timer_values),
        )
        rows = [json.loads(line) for line in (result_dir / "raw.jsonl").read_text().splitlines()]
        self.assertEqual(row_count, len(rows))
        return rows, backend

    def _validate_rows_with_repository_schema(self, rows: list[dict]) -> None:
        validator_path = self.repository / "benchmarks/scripts/validate_contract.py"
        spec = importlib.util.spec_from_file_location("pr01_contract_validator", validator_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        schema = json.loads(
            (self.repository / "benchmarks/schemas/result.schema.json").read_text()
        )
        for row in rows:
            module.validate_instance(row, schema)

    def test_cold_cell_emits_one_schema_shaped_trial(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            rows, backend = self._run(Path(directory), warm_state="cold")
            self.assertEqual(len(rows), 1)
            self.assertEqual(backend.benchmark_calls, 1)
            row = rows[0]
            self.assertEqual(row["run_id"], "fake-run-2")
            self.assertEqual(row["trial_index"], 1)
            self.assertEqual(
                row["environment_id"], "rtx4090-ubuntu22-driver580-v1"
            )
            self.assertEqual(row["metrics"]["model_load_ms"], 125.0)
            self.assertEqual(row["metrics"]["cpu_utilization_percent"], 12.5)
            self.assertEqual(row["metrics"]["gpu_utilization_percent"], 45.0)
            self.assertEqual(row["metrics"]["peak_vram_bytes"], 987_654_321)
            self.assertIsNone(row["correctness_gate_id"])
            self.assertIsNone(row["correctness_report_sha256"])
            self.assertEqual(len(row["requests"]), 2)
            self.assertRegex(row["requests"][0]["prompt_token_ids_sha256"], r"^[0-9a-f]{64}$")
            self.assertRegex(
                row["requests"][0]["generated_token_ids_sha256"],
                r"^[0-9a-f]{64}$",
            )
            schema = json.loads(
                (self.repository / "benchmarks/schemas/result.schema.json").read_text()
            )
            self.assertEqual(set(row), set(schema["required"]))
            request_required = schema["$defs"]["requestObservation"]["required"]
            self.assertEqual(set(row["requests"][0]), set(request_required))
            self._validate_rows_with_repository_schema(rows)

    def test_warm_cell_honors_five_warmups_and_thirty_measurements(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            rows, backend = self._run(Path(directory), warm_state="warm")
            self.assertEqual(len(rows), 30)
            self.assertEqual(backend.benchmark_calls, 35)
            self.assertEqual([row["trial_index"] for row in rows], list(range(1, 31)))
            self.assertEqual(len({row["trial_id"] for row in rows}), 30)
            self._validate_rows_with_repository_schema(rows)

    def test_existing_result_directory_is_rejected_before_backend_load(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompts = root / "prompts.jsonl"
            result_dir = root / "existing"
            write_prompts(prompts)
            result_dir.mkdir()
            called = False

            def factory(**kwargs):
                nonlocal called
                called = True
                return FakeBackend()

            with self.assertRaisesRegex(ValueError, "reuse"):
                run_benchmark(
                    matrix_path=self.repository / "benchmarks/matrix.yaml",
                    prompts_path=prompts,
                    result_dir=result_dir,
                    backend_factory=factory,
                    device="cuda:0",
                    local_files_only=True,
                    run_index=1,
                    run_id="x",
                    warm_state_filter="cold",
                    concurrency_filter=1,
                    prompt_tokens_filter=128,
                    output_tokens_filter=32,
                )
            self.assertFalse(called)

    def test_matrix_lifecycle_is_per_independent_run(self) -> None:
        matrix = json.loads(
            (self.repository / "benchmarks/matrix.yaml").read_text(encoding="utf-8")
        )
        _validate_matrix(matrix)
        changed = copy.deepcopy(matrix)
        changed["measurement"]["warm"]["reuse_model_within_run"] = False
        with self.assertRaisesRegex(ValueError, "reuse_model_within_run"):
            _validate_matrix(changed)


if __name__ == "__main__":
    unittest.main()
