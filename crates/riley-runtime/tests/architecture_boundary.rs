//! Source-level architecture contracts for incremental executor extraction.

const EXECUTOR_METRICS: &str = include_str!("../src/llama/executor/metrics.rs");
const EXECUTOR_ERROR: &str = include_str!("../src/llama/executor/error.rs");
const EXECUTOR_SHAPE: &str = include_str!("../src/llama/executor/shape.rs");

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
