"""Focused hostile-input tests for the native-D128 trace collector."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "collect_qwen3b_native_d128_trace.py"
REPOSITORY_ROOT = SCRIPT.parents[2]
SPEC = importlib.util.spec_from_file_location(
    "collect_qwen3b_native_d128_trace", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
collector = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = collector
SPEC.loader.exec_module(collector)


def digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def top_ids(start: int) -> list[int]:
    return list(range(start, start + collector.TOP_K))


def top_values() -> list[float]:
    return [float(100 - index) for index in range(collector.TOP_K)]


def hf_row(
    step: int, native: dict[str, object], teacher_token_id: int
) -> dict[str, object]:
    ids = top_ids(teacher_token_id)
    values = top_values()
    hf_hash = str(native["row_bf16_le_sha256"])
    return {
        "step": step,
        "input_token_count": collector.PROMPT_TOKEN_COUNT + step,
        "attention_mask_token_count": collector.PROMPT_TOKEN_COUNT + step,
        "position_start": 0,
        "position_end": collector.PROMPT_TOKEN_COUNT - 1 + step,
        "logits_bf16_le_sha256": hf_hash,
        "raw_argmax_token_id": int(native["raw_argmax_token_id"]),
        "selected_token_id": teacher_token_id,
        "selected_logit_bf16_as_f32": values[0],
        "top_token_ids": ids,
        "top_values_f32": values,
        "raw_hash_matches": True,
        "selected_token_matches": int(native["selected_token_id"]) == teacher_token_id,
        "raw_argmax_matches": True,
        "top32_order_matches_tie_sensitive": list(native["top_token_ids"]) == ids,
        "top32_set_overlap": len(set(native["top_token_ids"]).intersection(ids)),
    }


def trace_row(step: int, *, selected_token_id: int) -> dict[str, object]:
    ids = top_ids(selected_token_id)
    values = top_values()
    native: dict[str, object] = {
        "step": step,
        "scheduler_iteration_id": 64 + step,
        "kind": "prefill" if step == 0 else "decode",
        "iteration_input_token_count": 32 if step == 0 else 1,
        "target_logical_length": collector.PROMPT_TOKEN_COUNT + step,
        "row_bf16_le_sha256": digest(f"native-{step}"),
        "addressable_bf16_le_sha256": digest(f"addressable-{step}"),
        "raw_argmax_token_id": selected_token_id,
        "selected_token_id": selected_token_id,
        "selected_logit_bf16_as_f32": values[0],
        "top_token_ids": ids,
        "top_values_f32": values,
    }
    native["hf_cache_off"] = hf_row(step, native, selected_token_id)
    return native


def mode(backend: str, selected_token_ids: list[int]) -> dict[str, object]:
    rows = [
        trace_row(step, selected_token_id=token)
        for step, token in enumerate(selected_token_ids)
    ]
    return {
        "projection_bias_backend": backend,
        "prefill_iteration_count": 64,
        "decode_iteration_count": collector.TRACE_OUTPUT_ROWS - 1,
        "first_hf_cache_off_selected_token_mismatch": None,
        "rows": rows,
    }


def valid_trace() -> dict[str, object]:
    teacher_token_ids = [100 + step * 40 for step in range(collector.TRACE_OUTPUT_ROWS)]
    return {
        "schema_version": collector.TRACE_SCHEMA_VERSION,
        "artifact_kind": collector.TRACE_ARTIFACT_KIND,
        "performance_claim_eligible": False,
        "trace_contract": {
            "scheduler_executor_replay": True,
            "http_transport_replayed": False,
            "sampling": "teacher-forced-hf-cache-off-selected-token-after-scheduler-commit",
            "hf_reference": "full-prefix-cache-off; cache-on excluded because its recorded decode position starts at 2049",
            "projection_bias_modes": [
                collector.STRICT_PROJECTION_BIAS_BACKEND,
                collector.FUSED_PROJECTION_BIAS_BACKEND,
            ],
            "native_d128_backend": collector.NATIVE_D128_BACKEND_ID,
            "max_active_sequences": 8,
            "batch_token_budget": 32,
            "prefill_chunk_tokens": 32,
            "max_sequence_tokens": 2_176,
            "physical_kv_blocks": 1_088,
            "trace_output_rows": collector.TRACE_OUTPUT_ROWS,
            "submitted_request_count": 1,
            "scheduled_batch_size": 1,
            "capacity_configuration_only": True,
            "request_max_new_tokens": collector.TRACE_OUTPUT_ROWS,
            "execution_graph_policy": "disabled",
            "residual_rmsnorm": "separate",
            "execution_completion": "iteration-batch",
            "metadata_transport": "synchronous",
            "batch_shape_policy": "fixed-maximum",
            "reduction_profile": "canonical-v1",
            "top32_order_comparison": "observational-tie-sensitive",
        },
        "model": {
            "id": collector.QWEN3B_MODEL_ID,
            "revision": collector.QWEN3B_MODEL_REVISION,
        },
        "workload": {
            "sha256": collector.QWEN3B_WORKLOAD_SHA256,
            "case": collector.QWEN3B_WORKLOAD_CASE,
            "prompt_token_count": collector.PROMPT_TOKEN_COUNT,
            "hf_teacher_token_ids": teacher_token_ids,
        },
        "hf_generation_oracle": {
            "schema_version": collector.HF_GENERATION_ORACLE_SCHEMA,
            "artifact_sha256": digest("hf-generation-oracle"),
            "mode": "cache_off",
        },
        "modes": [
            mode(collector.STRICT_PROJECTION_BIAS_BACKEND, teacher_token_ids),
            mode(collector.FUSED_PROJECTION_BIAS_BACKEND, teacher_token_ids),
        ],
    }


def marker(trace: dict[str, object]) -> str:
    return collector.MARKER_PREFIX + json.dumps(
        trace, allow_nan=False, separators=(",", ":")
    )


class NativeD128TraceCollectorTests(unittest.TestCase):
    def write_log(self, root: Path, content: str) -> Path:
        path = root / "native-d128-test.stdout.log"
        path.write_text(content, encoding="utf-8")
        return path

    def test_collects_one_marker_as_canonical_create_only_envelope(self) -> None:
        trace = valid_trace()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = (
                "running ignored Rust test\n" + marker(trace) + "\ntest result: ok\n"
            )
            log_path = self.write_log(root, source)
            output = root / "evidence" / "trace.json"
            result = collector.collect_trace(
                test_stdout_path=log_path, output_path=output
            )
            raw = output.read_bytes()

        self.assertEqual(raw, collector._canonical_json_bytes(result))
        self.assertEqual(result["schema_version"], collector.COLLECTED_SCHEMA_VERSION)
        self.assertFalse(result["performance_claim_eligible"])
        self.assertEqual(result["trace"], trace)
        self.assertEqual(
            result["source_log_sha256"],
            hashlib.sha256(source.encode("utf-8")).hexdigest(),
        )

    def test_cli_collects_the_same_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            log_path = self.write_log(root, marker(valid_trace()) + "\n")
            output = root / "trace.json"
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--test-stdout",
                    str(log_path),
                    "--output",
                    str(output),
                ],
                cwd=REPOSITORY_ROOT,
                check=False,
                text=True,
                capture_output=True,
            )
            document = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("source_log_sha256=", result.stdout)
        self.assertFalse(document["performance_claim_eligible"])

    def test_rejects_missing_or_duplicate_markers_without_writing_output(self) -> None:
        cases = {
            "missing": "test result: ok\n",
            "duplicate": marker(valid_trace()) + "\n" + marker(valid_trace()) + "\n",
        }
        for label, source in cases.items():
            with self.subTest(
                label=label
            ), tempfile.TemporaryDirectory() as temporary_directory:
                root = Path(temporary_directory)
                log_path = self.write_log(root, source)
                output = root / "trace.json"
                with self.assertRaisesRegex(
                    collector.TraceCollectorError, "exactly one"
                ):
                    collector.collect_trace(
                        test_stdout_path=log_path, output_path=output
                    )
                self.assertFalse(output.exists())

    def test_rejects_performance_eligible_and_inconsistent_comparison_bindings(
        self,
    ) -> None:
        performance = valid_trace()
        performance["performance_claim_eligible"] = True
        inconsistent = valid_trace()
        inconsistent["modes"][0]["rows"][0]["hf_cache_off"][
            "selected_token_matches"
        ] = False
        for label, trace, expected in (
            ("performance", performance, "must be false"),
            ("comparison", inconsistent, "selected_token_matches"),
        ):
            with self.subTest(
                label=label
            ), tempfile.TemporaryDirectory() as temporary_directory:
                root = Path(temporary_directory)
                log_path = self.write_log(root, marker(trace) + "\n")
                output = root / "trace.json"
                with self.assertRaisesRegex(collector.TraceCollectorError, expected):
                    collector.collect_trace(
                        test_stdout_path=log_path, output_path=output
                    )
                self.assertFalse(output.exists())

    def test_rejects_duplicate_json_keys_and_existing_output_without_replacement(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            log_path = self.write_log(
                root,
                collector.MARKER_PREFIX
                + '{"schema_version":"first","schema_version":"second"}\n',
            )
            output = root / "trace.json"
            with self.assertRaisesRegex(collector.TraceCollectorError, "repeats key"):
                collector.collect_trace(test_stdout_path=log_path, output_path=output)
            self.assertFalse(output.exists())

            valid_log = self.write_log(root, marker(valid_trace()) + "\n")
            output.write_bytes(b"do not replace")
            with self.assertRaisesRegex(
                collector.TraceCollectorError, "refusing to overwrite"
            ):
                collector.collect_trace(test_stdout_path=valid_log, output_path=output)
            self.assertEqual(output.read_bytes(), b"do not replace")

    def test_rejects_unknown_or_missing_capacity_and_provenance_contract_fields(
        self,
    ) -> None:
        unknown = valid_trace()
        unknown["trace_contract"]["capacity_provenance"] = "unbound"
        missing = valid_trace()
        del missing["trace_contract"]["scheduled_batch_size"]
        for label, trace in (("unknown", unknown), ("missing", missing)):
            with self.subTest(
                label=label
            ), tempfile.TemporaryDirectory() as temporary_directory:
                root = Path(temporary_directory)
                log_path = self.write_log(root, marker(trace) + "\n")
                with self.assertRaisesRegex(
                    collector.TraceCollectorError, "unexpected|missing"
                ):
                    collector.collect_trace(
                        test_stdout_path=log_path, output_path=root / "trace.json"
                    )

    def test_rejects_noncompact_marker_outer_whitespace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            log_path = self.write_log(
                root, collector.MARKER_PREFIX + " " + json.dumps(valid_trace()) + "\n"
            )
            with self.assertRaisesRegex(
                collector.TraceCollectorError, "outer whitespace"
            ):
                collector.collect_trace(
                    test_stdout_path=log_path, output_path=root / "trace.json"
                )


if __name__ == "__main__":
    unittest.main()
