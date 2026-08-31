use riley_runtime::llama::{
    ExecutionGraphPolicy, ExecutionMode, GraphCaptureSafety, GraphDispatchDecision,
    GraphDispatchEligibility, GraphDispatchError, GraphDispatchRequest, GraphFallbackReason,
    GraphInventoryState, GraphOperatorCapability, GraphSamplingBackend, GraphWorkloadStage,
    select_execution_graph,
};

const GRAPH_DISPATCH_SOURCE: &str = include_str!("../src/llama/executor/graph.rs");

const fn admitted_safety() -> GraphCaptureSafety {
    GraphCaptureSafety::new(
        GraphSamplingBackend::GpuGreedy,
        GraphOperatorCapability::Supported,
        true,
    )
}

const fn request(
    policy: ExecutionGraphPolicy,
    stage: GraphWorkloadStage,
    inventory: GraphInventoryState,
) -> GraphDispatchRequest {
    GraphDispatchRequest::new(
        policy,
        GraphDispatchEligibility::new(stage, true, true, true, admitted_safety()),
        inventory,
    )
}

#[test]
fn graph_dispatch_is_cpu_only_and_keeps_disabled_as_exact_eager() {
    let disabled = select_execution_graph(GraphDispatchRequest::new(
        ExecutionGraphPolicy::Disabled,
        GraphDispatchEligibility::new(
            GraphWorkloadStage::Unsupported,
            false,
            false,
            false,
            GraphCaptureSafety::new(
                GraphSamplingBackend::Unsupported,
                GraphOperatorCapability::Unknown,
                false,
            ),
        ),
        GraphInventoryState::Poisoned,
    ))
    .expect("disabled must choose the exact eager path without inspecting facts");

    assert_eq!(disabled.mode(), ExecutionMode::ExactEager);
    assert_eq!(
        disabled.fallback_reason(),
        Some(GraphFallbackReason::PolicyDisabled)
    );
}

#[test]
fn auto_selects_only_exactly_matching_graph_entries() {
    assert_eq!(
        select_execution_graph(request(
            ExecutionGraphPolicy::Auto,
            GraphWorkloadStage::PureDecode,
            GraphInventoryState::PreparedFull,
        )),
        Ok(GraphDispatchDecision::FullGraph)
    );
    assert_eq!(
        select_execution_graph(request(
            ExecutionGraphPolicy::Auto,
            GraphWorkloadStage::Mixed,
            GraphInventoryState::PreparedPiecewise,
        )),
        Ok(GraphDispatchDecision::PiecewiseGraph)
    );
    assert_eq!(
        select_execution_graph(request(
            ExecutionGraphPolicy::Auto,
            GraphWorkloadStage::PureDecode,
            GraphInventoryState::PreparedPiecewise,
        )),
        Ok(GraphDispatchDecision::ExactEager(
            GraphFallbackReason::NotPrepared
        ))
    );
    assert_eq!(
        select_execution_graph(GraphDispatchRequest::new(
            ExecutionGraphPolicy::Auto,
            GraphDispatchEligibility::new(
                GraphWorkloadStage::Mixed,
                true,
                true,
                true,
                GraphCaptureSafety::new(
                    GraphSamplingBackend::Unsupported,
                    GraphOperatorCapability::Supported,
                    true,
                ),
            ),
            GraphInventoryState::PreparedPiecewise,
        )),
        Ok(GraphDispatchDecision::PiecewiseGraph)
    );
}

#[test]
fn unknown_capability_and_poison_never_select_a_graph() {
    let unknown = GraphDispatchRequest::new(
        ExecutionGraphPolicy::Auto,
        GraphDispatchEligibility::new(
            GraphWorkloadStage::PureDecode,
            true,
            true,
            true,
            GraphCaptureSafety::new(
                GraphSamplingBackend::GpuGreedy,
                GraphOperatorCapability::Unknown,
                true,
            ),
        ),
        GraphInventoryState::PreparedFull,
    );
    assert_eq!(
        select_execution_graph(unknown),
        Ok(GraphDispatchDecision::ExactEager(
            GraphFallbackReason::OperatorCapabilityUnknown
        ))
    );
    assert_eq!(
        select_execution_graph(request(
            ExecutionGraphPolicy::Auto,
            GraphWorkloadStage::Prefill,
            GraphInventoryState::Poisoned,
        )),
        Ok(GraphDispatchDecision::ExactEager(
            GraphFallbackReason::GraphPoisoned
        ))
    );
}

#[test]
fn require_rejects_instead_of_silently_running_eager() {
    assert_eq!(
        select_execution_graph(request(
            ExecutionGraphPolicy::Require,
            GraphWorkloadStage::PureDecode,
            GraphInventoryState::NotPrepared,
        )),
        Err(GraphDispatchError::RequiredGraphUnavailable {
            reason: GraphFallbackReason::NotPrepared,
        })
    );
}

#[test]
fn auto_reports_every_pre_lookup_eligibility_miss_with_a_closed_reason() {
    let cases = [
        (
            GraphDispatchEligibility::new(
                GraphWorkloadStage::Unsupported,
                true,
                true,
                true,
                admitted_safety(),
            ),
            GraphFallbackReason::UnsupportedStage,
        ),
        (
            GraphDispatchEligibility::new(
                GraphWorkloadStage::PureDecode,
                false,
                true,
                true,
                admitted_safety(),
            ),
            GraphFallbackReason::UnsupportedShape,
        ),
        (
            GraphDispatchEligibility::new(
                GraphWorkloadStage::PureDecode,
                true,
                false,
                true,
                admitted_safety(),
            ),
            GraphFallbackReason::LayoutMismatch,
        ),
        (
            GraphDispatchEligibility::new(
                GraphWorkloadStage::PureDecode,
                true,
                true,
                true,
                GraphCaptureSafety::new(
                    GraphSamplingBackend::Unsupported,
                    GraphOperatorCapability::Supported,
                    true,
                ),
            ),
            GraphFallbackReason::UnsupportedSampling,
        ),
        (
            GraphDispatchEligibility::new(
                GraphWorkloadStage::PureDecode,
                true,
                true,
                true,
                GraphCaptureSafety::new(
                    GraphSamplingBackend::GpuGreedy,
                    GraphOperatorCapability::Unsupported,
                    true,
                ),
            ),
            GraphFallbackReason::BackendNotCaptureSafe,
        ),
    ];

    for (eligibility, reason) in cases {
        assert_eq!(
            select_execution_graph(GraphDispatchRequest::new(
                ExecutionGraphPolicy::Auto,
                eligibility,
                GraphInventoryState::PreparedFull,
            )),
            Ok(GraphDispatchDecision::ExactEager(reason))
        );
    }
}

#[test]
fn graph_dispatch_policy_does_not_own_cuda_or_model_execution() {
    for forbidden in [
        "riley_cuda",
        "batch_executor",
        "PreparedLlama",
        "LlamaBatchExecutor",
        "CudaContext",
        "extern \"C\"",
        "unsafe",
    ] {
        assert!(
            !GRAPH_DISPATCH_SOURCE.contains(forbidden),
            "graph dispatch crossed its scalar-only boundary with {forbidden:?}"
        );
    }
}
