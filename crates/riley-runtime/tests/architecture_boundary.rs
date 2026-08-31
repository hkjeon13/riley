//! Source-level architecture contracts for incremental executor extraction.

const EXECUTOR_METRICS: &str = include_str!("../src/llama/executor/metrics.rs");
const EXECUTOR_ERROR: &str = include_str!("../src/llama/executor/error.rs");
const EXECUTOR_SHAPE: &str = include_str!("../src/llama/executor/shape.rs");
const EXECUTOR_BUFFERS: &str = include_str!("../src/llama/executor/buffers.rs");
const EXECUTOR_DEVICE_VIEWS: &str = include_str!("../src/llama/executor/device_views.rs");
const EXECUTOR_METADATA: &str = include_str!("../src/llama/executor/metadata.rs");

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
fn executor_metadata_is_a_checked_layout_and_host_packing_boundary() {
    for forbidden in [
        "riley_scheduler",
        "riley_server",
        "super::buffers",
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
