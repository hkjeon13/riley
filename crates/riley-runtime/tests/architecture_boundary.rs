//! Source-level architecture contracts for incremental executor extraction.

const EXECUTOR_METRICS: &str = include_str!("../src/llama/executor/metrics.rs");
const EXECUTOR_ERROR: &str = include_str!("../src/llama/executor/error.rs");
const EXECUTOR_SHAPE: &str = include_str!("../src/llama/executor/shape.rs");
const EXECUTOR_BUFFERS: &str = include_str!("../src/llama/executor/buffers.rs");
const EXECUTOR_DEVICE_VIEWS: &str = include_str!("../src/llama/executor/device_views.rs");
const EXECUTOR_DISPATCH: &str = include_str!("../src/llama/executor/dispatch.rs");
const EXECUTOR_GEMM_PLAN: &str = include_str!("../src/llama/executor/gemm_plan.rs");
const EXECUTOR_METADATA: &str = include_str!("../src/llama/executor/metadata.rs");
const EXECUTOR_OUTPUT: &str = include_str!("../src/llama/executor/output.rs");
const EXECUTOR_POISON: &str = include_str!("../src/llama/executor/poison.rs");
const EXECUTOR_ROPE: &str = include_str!("../src/llama/executor/rope.rs");

#[test]
fn executor_metrics_do_not_own_runtime_resources_or_scheduling_policy() {
    for forbidden in [
        "riley_scheduler",
        "riley_server",
        "CudaDeviceBuffer",
        "CudaPinnedHostBuffer",
        "CudaUploadedWeights",
        "KvLayout",
        "LlamaExecutionPlan",
        "PreparedLlamaForward",
        "PreparedLlamaBatchExecutor",
        "LoadedModel",
        "Vec",
        "Box",
        "String",
        "format!",
    ] {
        assert!(
            !EXECUTOR_METRICS.contains(forbidden),
            "executor metrics crossed its value-only boundary with {forbidden:?}"
        );
    }
}

#[test]
fn executor_error_vocabulary_does_not_own_runtime_resources_or_scheduling_policy() {
    for required in [
        "cuda_error",
        "LlamaBatchExecutorError::Cuda",
        "record_close",
        "LlamaBatchExecutorError::Cleanup",
    ] {
        assert!(
            EXECUTOR_ERROR.contains(required),
            "executor error vocabulary omitted required routing token {required:?}"
        );
    }
    for forbidden in [
        "riley_scheduler",
        "riley_server",
        "CudaContext",
        "CudaDeviceBuffer",
        "CudaPinnedHostBuffer",
        "CudaUploadedWeights",
        "KvLayout",
        "LlamaExecutionPlan",
        "PreparedLlamaForward",
        "PreparedLlamaBatchExecutor",
        "LoadedModel",
        "Vec",
        "Box",
        "String",
        "format!",
    ] {
        assert!(
            !EXECUTOR_ERROR.contains(forbidden),
            "executor error vocabulary crossed its value-only boundary with {forbidden:?}"
        );
    }
}

#[test]
fn executor_shape_does_not_own_runtime_resources_or_scheduling_policy() {
    let required = "select_prepared_dense_rows";
    assert!(
        EXECUTOR_SHAPE.contains(required),
        "executor shape omitted required scalar-selection token {required:?}"
    );
    for forbidden in [
        "riley_scheduler",
        "riley_server",
        "CudaContext",
        "CudaDeviceBuffer",
        "CudaPinnedHostBuffer",
        "CudaStream",
        "CudaUploadedWeights",
        "KvLayout",
        "LlamaExecutionPlan",
        "PreparedLlamaForward",
        "PreparedLlamaBatchExecutor",
        "PreparedLlamaBatchShape",
        "LoadedModel",
        "Vec",
        "Box",
        "String",
        "format!",
    ] {
        assert!(
            !EXECUTOR_SHAPE.contains(forbidden),
            "executor shape crossed its value-only boundary with {forbidden:?}"
        );
    }
}

#[test]
fn executor_buffers_do_not_own_model_or_execution_policy() {
    for forbidden in [
        "riley_scheduler",
        "riley_server",
        "LoadedModel",
        "CudaUploadedWeights",
        "KvLayout",
        "LlamaExecutionPlan",
        "PreparedLlamaForward",
        "PreparedLlamaBatchExecutor",
        "PreparedLlamaBatchShape",
        "GemmPlans",
        "CudaStream",
        "CudaExecutionStream",
        "CudaBufferSpan",
        "LlamaPackedBatchMetadata",
        "BatchMetadataTransport",
        "greedy_results",
        "GREEDY_RESULT_BYTES",
    ] {
        assert!(
            !EXECUTOR_BUFFERS.contains(forbidden),
            "executor buffers crossed its raw-input boundary with {forbidden:?}"
        );
    }
}

#[test]
fn executor_device_views_only_bind_borrowed_cuda_metadata() {
    for required in [
        "struct BatchDeviceViews",
        "per_operation_device_views",
        "packed_device_views",
        "PackedBatchV1::new(",
        "CudaBufferSpan::new(",
    ] {
        assert!(
            EXECUTOR_DEVICE_VIEWS.contains(required),
            "executor device views omitted required borrowed-binding token {required:?}"
        );
    }
    for forbidden in [
        "riley_scheduler",
        "riley_server",
        "CudaContext",
        "CudaPinnedHostBuffer",
        "CudaStream",
        "CudaExecutionStream",
        "CudaCommand",
        "CudaBufferSpanMut",
        "LoadedModel",
        "CudaUploadedWeights",
        "KvLayout",
        "LlamaExecutionPlan",
        "PreparedLlamaForward",
        "PreparedLlamaBatchExecutor",
        "PreparedLlamaBatchShape",
        "GemmPlans",
        "BatchDeviceInput",
        "BatchHostInput",
        "IterationBatchHostWorkspace",
        "BatchMetadataTransport",
        "ExecutionCompletionImplementation",
        "LlamaOp",
        "batch_executor",
        "allocate_device_buffer",
        "allocate_pinned_host_buffer",
        "upload_from_slice",
        "copy_from_pinned_in_command_batch",
        "Vec",
        "Box",
        "String",
        "format!",
    ] {
        assert!(
            !EXECUTOR_DEVICE_VIEWS.contains(forbidden),
            "executor device views crossed its borrowed-binding boundary with {forbidden:?}"
        );
    }
}

#[test]
fn executor_dispatch_only_binds_borrowed_output_primitives() {
    for required in [
        "struct OutputPrimitiveDispatch",
        "dispatch_output_primitives",
        "CudaExecutionStream",
        "CudaBufferSpan",
        "CudaBufferSpanMut::new(",
        "RowGatherParams",
        "row_gather",
        "Bf16ArgmaxParams",
        "deterministic_bf16_argmax",
        "span(",
        "output_logits_bytes",
        "greedy_result_bytes",
        "cuda_error as dispatch_cuda",
    ] {
        assert!(
            EXECUTOR_DISPATCH.contains(required),
            "executor dispatch omitted required borrowed-output token {required:?}"
        );
    }
    for forbidden in [
        "batch_executor",
        "ForwardBuffers",
        "LlamaExecutionPlan",
        "BatchOutputMode",
        "PreparedLlamaBatchExecutor",
        "PreparedLlamaForward",
        "BatchHostInput",
        "BatchDeviceInput",
        "BatchMetadataTransport",
        "LlamaPackedBatchMetadata",
        "PackedBatchV1",
        "CudaContext",
        "CudaStream",
        "CudaPinnedHostBuffer",
        "CudaUploadedWeights",
        "KvLayout",
        "GemmPlans",
        "RopeTableParams",
        "indexed_rope",
        "ragged_paged_attention",
        "allocate_device_buffer",
        "allocate_pinned_host_buffer",
        "upload_from_slice",
        "copy_from_pinned",
        "close(",
        "poison",
        "Vec",
        "Box",
        "String",
        "format!",
    ] {
        assert!(
            !EXECUTOR_DISPATCH.contains(forbidden),
            "executor dispatch crossed its borrowed-output boundary with {forbidden:?}"
        );
    }
}

#[test]
fn executor_gemm_plan_only_prepares_anchored_shape_variants() {
    for required in [
        "struct PreparedLlamaBatchShape",
        "prepare_shape_variants",
        "validate_shape_buckets",
        "prepare_batch_shape_variant",
        "try_reserve_exact",
        "is_anchored_gemm_not_supported",
        "CudaErrorKind::NotSupported",
        "variant.close()",
        "prepare anchored CUDA GEMM plan",
    ] {
        assert!(
            EXECUTOR_GEMM_PLAN.contains(required),
            "executor GEMM plan omitted required cold-preparation token {required:?}"
        );
    }
    for forbidden in [
        "riley_scheduler",
        "riley_server",
        "batch_executor",
        "PreparedLlamaBatchExecutor",
        "PreparedLlamaBatchExecutorConfig",
        "LlamaBatchShapeHistory",
        "CudaStream",
        "CudaExecutionStream",
        "CudaDeviceBuffer",
        "CudaPinnedHostBuffer",
        "CudaBufferSpan",
        "CudaUploadedWeights",
        "KvLayout",
        "ForwardBuffers",
        "BatchDeviceInput",
        "BatchHostInput",
        "PreparedLlamaBatchMetadata",
        "LlamaPackedBatchMetadata",
        "PackedBatchV1",
        "BatchMetadataTransport",
        "ExecutionCompletionImplementation",
        "LlamaOp",
        "execute_gemm",
        "execute_fixed_graph",
        "upload_from_slice",
        "copy_from_pinned",
        "allocate_device_buffer",
        "allocate_pinned_host_buffer",
    ] {
        assert!(
            !EXECUTOR_GEMM_PLAN.contains(forbidden),
            "executor GEMM plan crossed its cold-variant boundary with {forbidden:?}"
        );
    }
}

#[test]
fn executor_output_only_decodes_and_sizes_canonical_host_results() {
    for required in [
        "decode_greedy_tokens",
        "GREEDY_RESULT_BYTES",
        "BF16_ARGMAX_STATUS_SUCCESS",
        "BF16_ARGMAX_STATUS_NON_FINITE",
        "GreedyLogitsNonFinite",
        "InvalidGreedyResult",
        "chunks_exact",
        "output_logits_bytes",
        "LlamaBatchExecutorResource::GatheredLogits",
        "greedy_result_bytes",
        "LlamaBatchExecutorResource::GreedyResults",
        "ArithmeticOverflow",
    ] {
        assert!(
            EXECUTOR_OUTPUT.contains(required),
            "executor output omitted required host-decoding token {required:?}"
        );
    }
    for forbidden in [
        "riley_scheduler",
        "riley_server",
        "batch_executor",
        "PreparedLlamaBatchExecutor",
        "CudaContext",
        "CudaDeviceBuffer",
        "CudaPinnedHostBuffer",
        "CudaStream",
        "CudaExecutionStream",
        "CudaBufferSpan",
        "CudaUploadedWeights",
        "KvLayout",
        "ForwardBuffers",
        "BatchDeviceInput",
        "BatchHostInput",
        "PreparedLlamaBatchMetadata",
        "LlamaPackedBatchMetadata",
        "PackedBatchV1",
        "BatchMetadataTransport",
        "ExecutionCompletionImplementation",
        "Bf16ArgmaxParams",
        "row_gather",
        "download_to_slice",
        "execute_gemm",
        "allocate_device_buffer",
        "allocate_pinned_host_buffer",
        "Vec",
        "Box",
        "String",
        "format!",
    ] {
        assert!(
            !EXECUTOR_OUTPUT.contains(forbidden),
            "executor output crossed its host-decoding boundary with {forbidden:?}"
        );
    }
}

#[test]
fn executor_rope_only_materializes_cold_host_table_bytes() {
    for required in [
        "build_absolute_rope_angles",
        "build_absolute_cpu_rope_tables",
        "RopeTableBytes",
        "theta.powf",
        "angle.sin_cos",
        "to_ne_bytes",
        "try_reserve_exact",
        "LlamaBatchExecutorError::ArithmeticOverflow",
        "LlamaBatchExecutorError::HostAllocation",
        "LlamaBatchExecutorResource::RopeCos",
        "LlamaBatchExecutorResource::HostWorkspace",
    ] {
        assert!(
            EXECUTOR_ROPE.contains(required),
            "executor rope omitted required host-table token {required:?}"
        );
    }
    for forbidden in [
        "riley_scheduler",
        "riley_server",
        "riley_cuda",
        "batch_executor",
        "super::buffers",
        "PreparedLlamaBatchExecutor",
        "PreparedLlamaForward",
        "LlamaRopeTableProfile",
        "LlamaForwardError",
        "LlamaForwardResource",
        "LlamaDecodeError",
        "LlamaDecodeResource",
        "CudaContext",
        "CudaDeviceBuffer",
        "CudaPinnedHostBuffer",
        "CudaStream",
        "CudaExecutionStream",
        "CudaBufferSpan",
        "CudaUploadedWeights",
        "KvLayout",
        "LlamaExecutionPlan",
        "LoadedModel",
        "ForwardBuffers",
        "BatchMetadataTransport",
        "RopeTableParams",
        "rope_table(",
        "upload_from_slice",
        "ExecutionSite",
        "LlamaOp",
        "allocate_device_buffer",
        "allocate_pinned_host_buffer",
    ] {
        assert!(
            !EXECUTOR_ROPE.contains(forbidden),
            "executor rope crossed its host-table boundary with {forbidden:?}"
        );
    }
}

#[test]
fn executor_poison_only_routes_borrowed_failure_state() {
    for required in [
        "enum BatchDispatchDisposition",
        "CommandSubmissionStarted",
        "mutation_may_have_occurred",
        "poison_for_batch_error",
        "forward_gemms_poisoned",
        "LlamaBatchExecutorError::Cuda",
        "LlamaBatchExecutorError::Forward",
        "LlamaBatchExecutorError::InvalidConfiguration",
        "LlamaBatchExecutorError::ArithmeticOverflow",
        "poison_for_cuda_error",
        "poison_for_forward_error",
    ] {
        assert!(
            EXECUTOR_POISON.contains(required),
            "executor poison omitted required typed-routing token {required:?}"
        );
    }
    for forbidden in [
        "riley_scheduler",
        "riley_server",
        "batch_executor",
        "PreparedLlamaBatchExecutor",
        "PreparedLlamaForward",
        "CudaContext",
        "CudaDeviceBuffer",
        "CudaPinnedHostBuffer",
        "CudaStream",
        "CudaExecutionStream",
        "CudaBufferSpan",
        "CudaUploadedWeights",
        "KvLayout",
        "LlamaExecutionPlan",
        "ForwardBuffers",
        "BatchDeviceInput",
        "BatchHostInput",
        "LlamaPackedBatchMetadata",
        "PackedBatchV1",
        "BatchMetadataTransport",
        "ExecutionCompletionImplementation",
        "LlamaOp",
        "execute_gemm",
        "execute_fixed_graph",
        "upload_from_slice",
        "copy_from_pinned",
        "download_to_slice",
        "allocate_device_buffer",
        "allocate_pinned_host_buffer",
        "Vec",
        "Box",
        "String",
        "format!",
    ] {
        assert!(
            !EXECUTOR_POISON.contains(forbidden),
            "executor poison crossed its borrowed-routing boundary with {forbidden:?}"
        );
    }
}

#[test]
fn executor_metadata_is_a_checked_layout_and_host_packing_boundary() {
    for required in [
        "validate_for_execution",
        "LLAMA_BATCH_METADATA_V1_VERSION",
        "TokenOutOfRange",
        "PositionOutOfRange",
    ] {
        assert!(
            EXECUTOR_METADATA.contains(required),
            "executor metadata omitted required host-preflight token {required:?}"
        );
    }
    for forbidden in [
        "riley_scheduler",
        "riley_server",
        "super::buffers",
        "AttentionReductionProfile",
        "FIXED37_RAGGED_MAX_LOGICAL_TOKENS",
        "CudaContext",
        "CudaError",
        "CudaDType",
        "CudaDeviceBuffer",
        "CudaPinnedHostBuffer",
        "CudaStream",
        "CudaExecutionStream",
        "CudaBufferSpan",
        "LoadedModel",
        "CudaUploadedWeights",
        "KvLayout",
        "LlamaExecutionPlan",
        "PreparedLlamaForward",
        "PreparedLlamaBatchExecutor",
        "PreparedLlamaBatchShape",
        "GemmPlans",
        "BatchMetadataTransport",
        "ExecutionCompletionImplementation",
        "BatchDeviceInput",
        "BatchHostInput",
        "PerOperationDeviceMetadata",
        "IterationBatchHostWorkspace",
        "PackedBatchV1",
        "upload_from_slice",
        "copy_from_pinned_in_command_batch",
        "Vec",
        "Box",
        "String",
        "format!",
    ] {
        assert!(
            !EXECUTOR_METADATA.contains(forbidden),
            "executor metadata crossed its checked-layout/host-packing boundary with {forbidden:?}"
        );
    }
}
