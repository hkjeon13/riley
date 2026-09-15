from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "n03a_paged_d128_repeat_summary.py"
SPEC = importlib.util.spec_from_file_location("n03a_paged_d128_repeat_summary", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
summary = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = summary
SPEC.loader.exec_module(summary)


def psi(status: str, value: float) -> dict[str, object]:
    if status != "ok":
        return {"status": status, "some": None, "full": None, "error": "fixture unavailable"}
    return {
        "status": "ok",
        "some": {
            "avg10": value,
            "avg60": value + 0.1,
            "avg300": value + 0.2,
            "total": int(value * 100),
        },
        "full": {"avg10": value / 2.0, "avg60": value / 2.0, "avg300": value / 2.0, "total": 0},
        "error": None,
    }


def n03a_record(case: summary.ExpectedCase, *, offset: float) -> str:
    native_median = 0.200000 + offset + case.logical_tokens / 1_000_000.0
    reference_median = native_median * 1.6
    native_p95 = native_median * 1.1
    reference_p95 = reference_median * 1.1
    return (
        "riley-cuda-n03a-paged-gqa "
        f"schema_version=2 case={case.label} logical_tokens={case.logical_tokens} "
        "batch=1 query_heads=16 key_value_heads=2 head_size=128 page_size=16 "
        "fixture=patterned shuffled_page_ids=true partial_last_page=false "
        "timing_scope=prepared_paged_decode_execute_cuda_event "
        "internal_warmups_per_backend=8 paired_rounds=24 paired_order=ABBA "
        f"native_median_ms={native_median:.6f} native_p95_ms={native_p95:.6f} "
        f"reference_median_ms={reference_median:.6f} reference_p95_ms={reference_p95:.6f} "
        f"paired_speedup_ratio={reference_median / native_median:.6f} "
        f"paired_delta_ms={reference_median - native_median:.6f} "
        "native_workspace_bytes=270400 native_v2_state_prefix_bytes=266240 "
        "native_v2_transition_step_bytes=4096 native_v2_normalizer_bytes=64 "
        "reference_workspace_bytes=266240 "
        f"implementation_id={summary.IMPLEMENTATION_ID} "
        "implementation_version=2 "
        "graph_capture_supported=false operator_parity=passed allocation_delta=0 "
        "python_free=true full_model_serving=false vllm_comparison=false status=passed"
    )


class N03aPagedD128RepeatSummaryTests(unittest.TestCase):
    def test_requires_transition_v2_implementation_id(self) -> None:
        self.assertEqual(
            summary.IMPLEMENTATION_ID,
            "riley.cuda.native-bf16-paged-split-gqa.qwen2.5-3b.d128.qh16.kvh2.block16.transition-v2",
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            receipt_path = self.make_receipt(Path(temporary_directory), statuses=("succeeded",))
            payload = json.loads(receipt_path.read_text(encoding="utf-8"))
            log_path = Path(payload["timed_runs"][0]["stdout_path"])
            log_path.write_text(
                log_path.read_text(encoding="utf-8").replace(
                    summary.IMPLEMENTATION_ID,
                    "riley.cuda.native-bf16-paged-split-gqa.qwen2.5-3b.d128.qh16.kvh2.block16",
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(summary.SummaryError, "implementation_id"):
                summary.summarize_receipt(receipt_path)

    def make_log(self, directory: Path, index: int, *, offset: float) -> Path:
        path = directory / f"timed-{index:03d}.stdout.log"
        rows = [n03a_record(case, offset=offset) for case in reversed(summary.EXPECTED_CASES)]
        rows[0] = "test n03a_paged_d128_paired_cuda_event_control ... " + rows[0]
        path.write_text("\n".join(["running ignored GPU control", *rows, "test result: ok"]) + "\n", encoding="utf-8")
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
            stderr_path = output / f"timed-{index:03d}.stderr.log"
            if status == "succeeded":
                stdout_path = self.make_log(output, index, offset=float(index) / 100.0)
            else:
                stdout_path.write_text("failed attempt remains in receipt\n", encoding="utf-8")
            stderr_path.write_text("test stderr\n", encoding="utf-8")
            runs.append(
                {
                    "kind": "timed",
                    "index": index,
                    "status": status,
                    "exit_code": 0 if status == "succeeded" else 17,
                    "timed_out": False,
                    "wall_time_ms": 10.0 + index,
                    "error": None if status == "succeeded" else "fixture failure retained",
                    "stdout_path": str(stdout_path),
                    "stderr_path": str(stderr_path),
                    "pre": {"psi": {"io": psi("ok", 80.0 + index)}},
                    "post": {"psi": {"io": psi("ok", 90.0 + index)}},
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

    def test_aggregates_outer_process_metrics_and_retains_high_pressure_failures(self) -> None:
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
        failed = first["timed_run_io_psi_covariates"][1]
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["io_psi"]["pre"]["some"]["avg10"], 82.0)
        self.assertIn("fixture failure retained", failed["error"])
        self.assertEqual(len(first["per_case"]), 2)
        case = first["per_case"][0]
        self.assertEqual(case["case"], "b1-c2048-qh16-kvh2-d128")
        self.assertEqual([item["timed_index"] for item in case["observations"]], [1, 3])
        metrics = case["outer_process_metrics"]
        self.assertEqual(metrics["native_median_ms"]["count"], 2)
        self.assertGreater(metrics["paired_speedup_ratio"]["median"], 1.0)
        self.assertGreater(metrics["paired_delta_ms"]["median"], 0.0)
        interval = metrics["native_median_ms"]["deterministic_bootstrap_median_95_ci"]
        self.assertEqual(interval["resamples"], 400)
        self.assertEqual(interval["seed"], 260913)
        self.assertLessEqual(interval["lower"], metrics["native_median_ms"]["median"])
        self.assertGreaterEqual(interval["upper"], metrics["native_median_ms"]["median"])

    def test_every_successful_log_must_contain_both_exact_markers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            receipt_path = self.make_receipt(Path(temporary_directory), statuses=("succeeded",))
            payload = json.loads(receipt_path.read_text(encoding="utf-8"))
            log_path = Path(payload["timed_runs"][0]["stdout_path"])
            log_path.write_text(
                n03a_record(summary.EXPECTED_CASES[0], offset=0.01) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(summary.SummaryError, "missing expected N03a records"):
                summary.summarize_receipt(receipt_path)

    def test_duplicate_or_extra_marker_field_fails_explicitly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            receipt_path = self.make_receipt(Path(temporary_directory), statuses=("succeeded",))
            payload = json.loads(receipt_path.read_text(encoding="utf-8"))
            log_path = Path(payload["timed_runs"][0]["stdout_path"])
            log_path.write_text(
                log_path.read_text(encoding="utf-8")
                + n03a_record(summary.EXPECTED_CASES[0], offset=0.03)
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(summary.SummaryError, "duplicates expected N03a case"):
                summary.summarize_receipt(receipt_path)

            log_path.write_text(
                "\n".join(
                    [
                        n03a_record(case, offset=0.01) + " unapproved_field=true"
                        for case in summary.EXPECTED_CASES
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(summary.SummaryError, "unsupported marker fields"):
                summary.summarize_receipt(receipt_path)

    def test_required_field_and_cli_failure_do_not_write_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            receipt_path = self.make_receipt(Path(temporary_directory), statuses=("succeeded",))
            before = receipt_path.read_bytes()
            payload = json.loads(before)
            log_path = Path(payload["timed_runs"][0]["stdout_path"])
            log_path.write_text(
                log_path.read_text(encoding="utf-8").replace(" python_free=true", "", 1),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(summary.SummaryError, "lacks required marker fields"):
                summary.summarize_receipt(receipt_path)
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                result = summary.main(["--receipt", str(receipt_path)])
            self.assertEqual(result, 2)
            self.assertIn("n03a_paged_d128_repeat_summary: error:", stderr.getvalue())
            self.assertEqual(receipt_path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
