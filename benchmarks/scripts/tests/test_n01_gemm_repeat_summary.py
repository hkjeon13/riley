from __future__ import annotations

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "n01_gemm_repeat_summary.py"
SPEC = importlib.util.spec_from_file_location("n01_gemm_repeat_summary", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
summary = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = summary
SPEC.loader.exec_module(summary)


def psi(status: str, value: float) -> dict[str, object]:
    if status != "ok":
        return {"status": status, "some": None, "full": None, "error": "fixture unavailable"}
    return {
        "status": "ok",
        "some": {"avg10": value, "avg60": value + 0.1, "avg300": value + 0.2, "total": int(value * 100)},
        "full": {"avg10": 0.0, "avg60": 0.0, "avg300": 0.0, "total": 0},
        "error": None,
    }


def gemm_record(case: summary.ExpectedCase, *, offset: float) -> str:
    return (
        "riley-cuda-gemm "
        f"case={case.label} m={case.m} n={case.n} k={case.k} "
        "gemm_reduction_policy=strict-no-split-v1 latency_scope=ffi_execute_sync "
        f"latency_median_ms={1.0 + offset:.6f} latency_p95_ms={1.2 + offset:.6f} "
        f"effective_median_tflops={10.0 + offset:.6f} temporary_bytes=0 "
        "implementation_id=cublaslt:algo=1:tile=2:stages=3:split_k=1:reduction=0:swizzle=0:custom=0 "
        "numerical_flags=0x0 cc=8.9 runtime_version=12080 cublaslt_version=12080 "
        "explicit_stream=true python_free=true"
    )


CONTROL_STATUS = (
    "riley-cuda-qwen2_5-3b-native-control schema_version=1 "
    "input_weight_fixture=deterministic-synthetic cases=qkv:1,8,32,128;gate-up:1,8,32,128 "
    "reduction_policy=strict-no-split-v1 full_model_fusion=false full_model_serving=false status=passed"
)


class N01GemmRepeatSummaryTests(unittest.TestCase):
    def make_log(self, directory: Path, index: int, *, offset: float) -> Path:
        path = directory / f"timed-{index:03d}.stdout.log"
        rows = [gemm_record(case, offset=offset) for case in summary.EXPECTED_CASES]
        path.write_text("\n".join([*rows, CONTROL_STATUS, "test result: ok"]) + "\n", encoding="utf-8")
        return path

    def make_receipt(
        self,
        directory: Path,
        *,
        statuses: tuple[str, ...] = ("succeeded", "failed", "succeeded"),
    ) -> Path:
        output = directory / "evidence"
        output.mkdir()
        runs: list[dict[str, object]] = []
        for index, status in enumerate(statuses, start=1):
            stdout_path = output / f"timed-{index:03d}.stdout.log"
            if status == "succeeded":
                stdout_path = self.make_log(output, index, offset=float(index) / 10.0)
            else:
                stdout_path.write_text("the failed attempt is retained but not parsed\n", encoding="utf-8")
            runs.append(
                {
                    "kind": "timed",
                    "index": index,
                    "status": status,
                    "exit_code": 0 if status == "succeeded" else 17,
                    "timed_out": False,
                    "wall_time_ms": 10.0 + index,
                    "stdout_path": str(stdout_path),
                    "pre": {"psi": {"io": psi("ok", float(index))}},
                    "post": {"psi": {"io": psi("ok", float(index) + 0.5)}},
                }
            )
        receipt_path = output / "n01-repeat-control-receipt.json"
        receipt = {
            "schema_version": summary.REPEAT_RECEIPT_SCHEMA_VERSION,
            "receipt_path": str(receipt_path),
            "output_dir": str(output),
            "configuration": {
                "timed_repeats": len(statuses),
                "bootstrap_resamples": 400,
                "bootstrap_seed": 260913,
            },
            "status": "completed" if all(status == "succeeded" for status in statuses) else "completed-with-failures",
            "timed_runs": runs,
        }
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        return receipt_path

    def test_summarizes_each_control_case_and_retains_all_io_covariates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            receipt_path = self.make_receipt(Path(temporary_directory))
            first = summary.summarize_receipt(receipt_path)
            second = summary.summarize_receipt(receipt_path)

        self.assertEqual(first, second)
        self.assertEqual(first["schema_version"], summary.SUMMARY_SCHEMA_VERSION)
        self.assertFalse(first["scope"]["full_model_serving"])
        self.assertFalse(first["scope"]["vllm_comparison"])
        self.assertEqual(first["receipt"]["successful_timed_runs"], 2)
        self.assertEqual(first["receipt"]["non_successful_timed_runs"], 1)
        self.assertEqual(len(first["timed_run_io_psi_covariates"]), 3)
        self.assertEqual(first["timed_run_io_psi_covariates"][1]["status"], "failed")
        self.assertEqual(
            first["timed_run_io_psi_covariates"][1]["io_psi"]["pre"]["some"]["avg10"],
            2.0,
        )
        self.assertEqual(len(first["per_case"]), len(summary.EXPECTED_CASES))
        case = first["per_case"][0]
        self.assertEqual([item["timed_index"] for item in case["observations"]], [1, 3])
        self.assertEqual(case["latency_median_ms"]["count"], 2)
        self.assertAlmostEqual(case["latency_median_ms"]["median"], 1.2)
        self.assertGreater(case["latency_median_ms"]["sample_stddev"], 0.0)
        interval = case["latency_median_ms"]["deterministic_bootstrap_median_95_ci"]
        self.assertEqual(interval["resamples"], 400)
        self.assertLessEqual(interval["lower"], case["latency_median_ms"]["median"])
        self.assertGreaterEqual(interval["upper"], case["latency_median_ms"]["median"])

    def test_duplicate_expected_record_fails_explicitly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            receipt_path = self.make_receipt(Path(temporary_directory), statuses=("succeeded",))
            payload = json.loads(receipt_path.read_text(encoding="utf-8"))
            log_path = Path(payload["timed_runs"][0]["stdout_path"])
            log_path.write_text(
                log_path.read_text(encoding="utf-8")
                + gemm_record(summary.EXPECTED_CASES[0], offset=9.0)
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(summary.SummaryError, "duplicates expected GEMM case"):
                summary.summarize_receipt(receipt_path)

    def test_incomplete_expected_records_fail_explicitly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            receipt_path = self.make_receipt(Path(temporary_directory), statuses=("succeeded",))
            payload = json.loads(receipt_path.read_text(encoding="utf-8"))
            log_path = Path(payload["timed_runs"][0]["stdout_path"])
            rows = [gemm_record(case, offset=0.1) for case in summary.EXPECTED_CASES[:-1]]
            log_path.write_text("\n".join([*rows, CONTROL_STATUS]) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(summary.SummaryError, "missing expected GEMM records"):
                summary.summarize_receipt(receipt_path)

    def test_malformed_latency_and_cli_do_not_write_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            receipt_path = self.make_receipt(Path(temporary_directory), statuses=("succeeded",))
            before = receipt_path.read_bytes()
            payload = json.loads(before)
            log_path = Path(payload["timed_runs"][0]["stdout_path"])
            log_path.write_text(
                log_path.read_text(encoding="utf-8").replace(
                    "latency_median_ms=1.100000", "latency_median_ms=not-a-number", 1
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(summary.SummaryError, "latency_median_ms"):
                summary.summarize_receipt(receipt_path)
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                result = summary.main(["--receipt", str(receipt_path)])
            self.assertEqual(result, 2)
            self.assertIn("n01_gemm_repeat_summary: error:", stderr.getvalue())
            self.assertEqual(receipt_path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
