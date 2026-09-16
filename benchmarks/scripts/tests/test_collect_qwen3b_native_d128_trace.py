"""Focused hostile-input tests for the native-D128 trace collector v3."""

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
    *,
    mode: str,
    step: int,
    native: dict[str, object],
    teacher_token_ids: list[int],
    teacher_token_id: int,
    previous_teacher_token_id: int | None,
    selected_token_id: int | None = None,
) -> dict[str, object]:
    selected = teacher_token_id if selected_token_id is None else selected_token_id
    ids = top_ids(selected)
    values = top_values()
    context = collector.PROMPT_TOKEN_COUNT + step
    if mode == "cache-off":
        plan = {
            "call_input_token_count": context,
            "context_token_count": context,
            "attention_mask_token_count": context,
            "position_start": 0,
            "position_end": context - 1,
            "teacher_input_token_id": None,
            "cache_length_before": None,
            "cache_length_after": None,
        }
    elif step == 0:
        plan = {
            "call_input_token_count": collector.PROMPT_TOKEN_COUNT,
            "context_token_count": collector.PROMPT_TOKEN_COUNT,
            "attention_mask_token_count": collector.PROMPT_TOKEN_COUNT,
            "position_start": 0,
            "position_end": collector.PROMPT_TOKEN_COUNT - 1,
            "teacher_input_token_id": None,
            "cache_length_before": 0,
            "cache_length_after": collector.PROMPT_TOKEN_COUNT,
        }
    else:
        assert previous_teacher_token_id is not None
        plan = {
            "call_input_token_count": 1,
            "context_token_count": context,
            "attention_mask_token_count": context,
            "position_start": context - 1,
            "position_end": context - 1,
            "teacher_input_token_id": previous_teacher_token_id,
            "cache_length_before": context - 1,
            "cache_length_after": context,
        }
    call_input_hash = collector._expected_hf_call_input_sha256(
        mode, step, teacher_token_ids
    )
    return {
        "mode": mode,
        "step": step,
        "call_input_token_ids_le_u32_sha256": call_input_hash,
        **plan,
        "logits_bf16_le_sha256": str(native["row_bf16_le_sha256"]),
        "addressable_bf16_le_sha256": str(native["addressable_bf16_le_sha256"]),
        "raw_argmax_token_id": int(native["raw_argmax_token_id"]),
        "selected_token_id": selected,
        "cache_off_teacher_token_id": teacher_token_id,
        "selection_matches_cache_off_teacher": selected == teacher_token_id,
        "selected_logit_bf16_as_f32": values[0],
        "top_token_ids": ids,
        "top_values_bf16_as_f32": values,
        "raw_hash_matches": True,
        "selected_token_matches": int(native["selected_token_id"]) == selected,
        "raw_argmax_matches": True,
        "top32_order_matches_bf16_numeric_tie_break": list(native["top_token_ids"])
        == ids,
        "top32_set_overlap": len(set(native["top_token_ids"]).intersection(ids)),
    }


def trace_row(
    step: int,
    *,
    teacher_token_ids: list[int],
    cache_on_selected_token_id: int | None = None,
) -> dict[str, object]:
    selected = teacher_token_ids[step]
    ids = top_ids(selected)
    values = top_values()
    native: dict[str, object] = {
        "step": step,
        "scheduler_iteration_id": 64 + step,
        "kind": "prefill" if step == 0 else "decode",
        "iteration_input_token_count": 32 if step == 0 else 1,
        "target_logical_length": collector.PROMPT_TOKEN_COUNT + step,
        "row_bf16_le_sha256": digest(f"native-{step}"),
        "addressable_bf16_le_sha256": digest(f"addressable-{step}"),
        "raw_argmax_token_id": selected,
        "selected_token_id": selected,
        "selected_logit_bf16_as_f32": values[0],
        "top_token_ids": ids,
        "top_values_bf16_as_f32": values,
    }
    native["hf_cache_off"] = hf_row(
        mode="cache-off",
        step=step,
        native=native,
        teacher_token_ids=teacher_token_ids,
        teacher_token_id=selected,
        previous_teacher_token_id=None,
    )
    native["hf_cache_on"] = hf_row(
        mode="cache-on",
        step=step,
        native=native,
        teacher_token_ids=teacher_token_ids,
        teacher_token_id=selected,
        previous_teacher_token_id=(None if step == 0 else teacher_token_ids[step - 1]),
        selected_token_id=cache_on_selected_token_id,
    )
    return native


def mode(
    variant: dict[str, object],
    teacher_token_ids: list[int],
    *,
    cache_on_divergence_step: int | None = None,
) -> dict[str, object]:
    rows = [
        trace_row(
            step,
            teacher_token_ids=teacher_token_ids,
            cache_on_selected_token_id=(
                teacher_token_ids[step] + 200
                if step == cache_on_divergence_step
                else None
            ),
        )
        for step in range(collector.TRACE_OUTPUT_ROWS)
    ]
    return {
        "variant_id": str(variant["id"]),
        "projection_bias_backend": str(variant["projection_bias_backend"]),
        "batch_shape_policy": str(variant["batch_shape_policy"]),
        "prefill_dense_rows": int(variant["prefill_dense_rows"]),
        "decode_dense_rows": int(variant["decode_dense_rows"]),
        "prefill_iteration_count": 64,
        "decode_iteration_count": collector.TRACE_OUTPUT_ROWS - 1,
        "first_hf_cache_off_selected_token_mismatch": None,
        "first_hf_cache_on_selected_token_mismatch": cache_on_divergence_step,
        "rows": rows,
    }


def valid_trace(*, cache_on_divergence_step: int | None = None) -> dict[str, object]:
    teacher_token_ids = [100 + step * 40 for step in range(collector.TRACE_OUTPUT_ROWS)]
    return {
        "schema_version": collector.TRACE_SCHEMA_VERSION,
        "artifact_kind": collector.TRACE_ARTIFACT_KIND,
        "performance_claim_eligible": False,
        "trace_contract": {
            "scheduler_executor_replay": True,
            "http_transport_replayed": False,
            "sampling": "teacher-forced-cache-off-addressable-greedy-argmax-after-scheduler-commit",
            "cache_on_scheduler_reference": "step0-p2048-prefill;step>0-teacher_token_ids[step-1]-at-position-2048+step-1",
            "cache_off_control_reference": "full-prefix-cache-off-at-position-0-through-2047+step",
            "trace_variants": [dict(variant) for variant in collector.TRACE_VARIANTS],
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
            "reduction_profile": "canonical-v1",
            "top32_order_comparison": "bf16-numeric-descending-token-id-ascending-tie-break",
        },
        "model": {
            "id": collector.QWEN3B_MODEL_ID,
            "revision": collector.QWEN3B_MODEL_REVISION,
        },
        "workload": {
            "sha256": collector.QWEN3B_WORKLOAD_SHA256,
            "case": collector.QWEN3B_WORKLOAD_CASE,
            "prompt_token_count": collector.PROMPT_TOKEN_COUNT,
            "prompt_token_ids_le_u32_sha256": collector.QWEN3B_PROMPT_TOKEN_IDS_SHA256,
            "teacher_token_ids": teacher_token_ids,
            "teacher_token_ids_le_u32_sha256": collector._u32_le_sha256(
                teacher_token_ids
            ),
        },
        "hf_teacher_forced_oracle": {
            "schema_version": collector.HF_TEACHER_FORCED_ORACLE_SCHEMA,
            "artifact_kind": collector.HF_TEACHER_FORCED_ARTIFACT_KIND,
            "artifact_sha256": digest("hf-teacher-forced-oracle"),
            "teacher_token_ids_le_u32_sha256": digest("all-128-teacher-tokens"),
            "cache_off_sidecar": {
                "basename": "cache-off-logits.safetensors",
                "sha256": digest("cache-off-sidecar"),
            },
            "cache_on_sidecar": {
                "basename": "cache-on-logits.safetensors",
                "sha256": digest("cache-on-sidecar"),
            },
        },
        "modes": [
            mode(
                variant,
                teacher_token_ids,
                cache_on_divergence_step=cache_on_divergence_step,
            )
            for variant in collector.TRACE_VARIANTS
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

    def test_allows_cache_on_selected_token_to_differ_from_teacher(self) -> None:
        trace = valid_trace(cache_on_divergence_step=3)
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            result = collector.collect_trace(
                test_stdout_path=self.write_log(root, marker(trace) + "\n"),
                output_path=root / "trace.json",
            )
        self.assertEqual(
            result["trace"]["modes"][0]["first_hf_cache_on_selected_token_mismatch"],
            3,
        )
        self.assertFalse(
            result["trace"]["modes"][0]["rows"][3]["hf_cache_on"][
                "selection_matches_cache_off_teacher"
            ]
        )

    def test_rejects_active_row_bucket_trace_with_misreported_m1_decode_rows(
        self,
    ) -> None:
        trace = valid_trace()
        trace["modes"][1]["decode_dense_rows"] = 32
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            with self.assertRaisesRegex(
                collector.TraceCollectorError, "decode_dense_rows"
            ):
                collector.collect_trace(
                    test_stdout_path=self.write_log(root, marker(trace) + "\n"),
                    output_path=root / "trace.json",
                )

    def test_rejects_cache_off_non_teacher_selection_and_bad_cache_on_schedule(
        self,
    ) -> None:
        non_teacher = valid_trace()
        hf_off = non_teacher["modes"][0]["rows"][1]["hf_cache_off"]
        hf_off["selected_token_id"] = 777
        hf_off["selection_matches_cache_off_teacher"] = False
        hf_off["top_token_ids"] = top_ids(777)
        hf_off["top32_order_matches_bf16_numeric_tie_break"] = False
        hf_off["top32_set_overlap"] = 0
        schedule = valid_trace()
        schedule["modes"][0]["rows"][1]["hf_cache_on"]["position_start"] = 2_049
        for label, trace, expected in (
            ("cache-off", non_teacher, "selected_token_id"),
            ("cache-on", schedule, "position_start"),
        ):
            with self.subTest(
                label=label
            ), tempfile.TemporaryDirectory() as temporary_directory:
                root = Path(temporary_directory)
                with self.assertRaisesRegex(collector.TraceCollectorError, expected):
                    collector.collect_trace(
                        test_stdout_path=self.write_log(root, marker(trace) + "\n"),
                        output_path=root / "trace.json",
                    )

    def test_rejects_cache_on_decode_input_hash_not_derived_from_teacher(self) -> None:
        trace = valid_trace()
        trace["modes"][0]["rows"][1]["hf_cache_on"][
            "call_input_token_ids_le_u32_sha256"
        ] = digest("wrong-cache-on-decode-input")
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            with self.assertRaisesRegex(
                collector.TraceCollectorError, "call_input_token_ids_le_u32_sha256"
            ):
                collector.collect_trace(
                    test_stdout_path=self.write_log(root, marker(trace) + "\n"),
                    output_path=root / "trace.json",
                )

    def test_rejects_forged_pinned_prompt_input_hashes(self) -> None:
        cache_off = valid_trace()
        cache_off["modes"][0]["rows"][4]["hf_cache_off"][
            "call_input_token_ids_le_u32_sha256"
        ] = digest("wrong-cache-off-full-prefix-input")
        cache_on_prefill = valid_trace()
        cache_on_prefill["modes"][0]["rows"][0]["hf_cache_on"][
            "call_input_token_ids_le_u32_sha256"
        ] = digest("wrong-cache-on-prompt-prefill-input")
        workload = valid_trace()
        workload["workload"]["prompt_token_ids_le_u32_sha256"] = digest(
            "wrong-pinned-prompt-binding"
        )
        for label, trace in (
            ("cache-off", cache_off),
            ("cache-on-prefill", cache_on_prefill),
            ("workload", workload),
        ):
            with self.subTest(
                label=label
            ), tempfile.TemporaryDirectory() as temporary_directory:
                root = Path(temporary_directory)
                with self.assertRaisesRegex(
                    collector.TraceCollectorError,
                    "call_input_token_ids_le_u32_sha256|prompt_token_ids_le_u32_sha256",
                ):
                    collector.collect_trace(
                        test_stdout_path=self.write_log(root, marker(trace) + "\n"),
                        output_path=root / "trace.json",
                    )

    def test_rejects_equal_bf16_top_k_values_with_descending_token_ids(self) -> None:
        trace = valid_trace()
        row = trace["modes"][0]["rows"][0]
        row["top_values_bf16_as_f32"][1] = row["top_values_bf16_as_f32"][0]
        row["top_token_ids"][0], row["top_token_ids"][1] = (
            row["top_token_ids"][1],
            row["top_token_ids"][0],
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            with self.assertRaisesRegex(
                collector.TraceCollectorError, "ascending token ID"
            ):
                collector.collect_trace(
                    test_stdout_path=self.write_log(root, marker(trace) + "\n"),
                    output_path=root / "trace.json",
                )

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

    def test_rejects_performance_eligible_and_invalid_sidecar_binding(self) -> None:
        performance = valid_trace()
        performance["performance_claim_eligible"] = True
        binding = valid_trace()
        binding["hf_teacher_forced_oracle"]["cache_on_sidecar"][
            "basename"
        ] = "../bad.safetensors"
        for label, trace, expected in (
            ("performance", performance, "must be false"),
            ("binding", binding, "must be a safetensors basename"),
        ):
            with self.subTest(
                label=label
            ), tempfile.TemporaryDirectory() as temporary_directory:
                root = Path(temporary_directory)
                with self.assertRaisesRegex(collector.TraceCollectorError, expected):
                    collector.collect_trace(
                        test_stdout_path=self.write_log(root, marker(trace) + "\n"),
                        output_path=root / "trace.json",
                    )

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

    def test_rejects_unknown_or_missing_contract_fields_and_noncompact_marker(
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
                with self.assertRaisesRegex(
                    collector.TraceCollectorError, "unexpected|missing"
                ):
                    collector.collect_trace(
                        test_stdout_path=self.write_log(root, marker(trace) + "\n"),
                        output_path=root / "trace.json",
                    )
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            with self.assertRaisesRegex(
                collector.TraceCollectorError, "outer whitespace"
            ):
                collector.collect_trace(
                    test_stdout_path=self.write_log(
                        root,
                        collector.MARKER_PREFIX
                        + " "
                        + json.dumps(valid_trace())
                        + "\n",
                    ),
                    output_path=root / "trace.json",
                )


if __name__ == "__main__":
    unittest.main()
