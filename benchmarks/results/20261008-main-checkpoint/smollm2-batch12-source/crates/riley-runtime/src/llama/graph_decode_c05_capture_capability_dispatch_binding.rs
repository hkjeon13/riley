//! Fail-closed binding from exact C05 evidence to one C06 dispatch request.
//!
//! C07-29 establishes narrow C05 primitive facts, while C06 consumes one
//! aggregate operator-capability value.  This adapter is the only bridge
//! between those two value boundaries: it replaces that one request fact and
//! preserves every other C06 eligibility and inventory fact unchanged.
//!
//! The binding does not create a CUDA runtime or context, allocate, capture,
//! instantiate, launch, look up a graph, record metrics, or touch an executor.
//! A partial C05 inventory must therefore remain an `Unknown` C06 gate rather
//! than becoming permission to select or replay a graph.

use riley_cuda::CudaResult;

use super::{
    graph::{GraphDispatchRequest, GraphOperatorCapability, GraphWorkloadStage},
    graph_decode_c05_capture_capability_evidence::pure_decode_graph_v1_c05_capture_capability_evidence,
    graph_decode_capture_inventory::PureDecodeGraphV1CaptureCapabilityInventory,
};

/// Binds an already observed pure-decode inventory to a C06 request.
///
/// The inventory's fail-closed aggregate replaces only a `PureDecode`
/// request's operator capability.  All other request stages are explicitly
/// demoted to `Unknown`: this C07 V1 evidence does not review prefill or mixed
/// chains. Policy, stage, shape, layout, inventory-state, sampling, and
/// backend-safety facts are retained verbatim.
#[must_use]
pub(crate) fn bind_pure_decode_graph_v1_capture_capability_inventory(
    request: GraphDispatchRequest,
    inventory: PureDecodeGraphV1CaptureCapabilityInventory,
) -> GraphDispatchRequest {
    let operator_capability = if request.eligibility().stage() == GraphWorkloadStage::PureDecode {
        inventory.operator_capability()
    } else {
        GraphOperatorCapability::Unknown
    };

    request.with_eligibility(
        request
            .eligibility()
            .with_operator_capability(operator_capability),
    )
}

/// Queries exact C05 primitive evidence and binds it to one C06 request.
///
/// The C05 query is a native vocabulary lookup only.  A query error is
/// propagated instead of being widened into capture or execution permission.
///
/// # Errors
///
/// Returns the unchanged C05 query error when reviewed primitive evidence
/// cannot be established.
pub(crate) fn bind_pure_decode_graph_v1_exact_c05_capture_evidence(
    request: GraphDispatchRequest,
) -> CudaResult<GraphDispatchRequest> {
    let inventory = pure_decode_graph_v1_c05_capture_capability_evidence()?;
    Ok(bind_pure_decode_graph_v1_capture_capability_inventory(
        request, inventory,
    ))
}

#[cfg(test)]
mod tests {
    use super::{
        bind_pure_decode_graph_v1_capture_capability_inventory,
        bind_pure_decode_graph_v1_exact_c05_capture_evidence,
    };
    use crate::llama::graph::{
        ExecutionGraphPolicy, GraphCaptureSafety, GraphDispatchDecision, GraphDispatchEligibility,
        GraphDispatchError, GraphDispatchRequest, GraphFallbackReason, GraphInventoryState,
        GraphOperatorCapability, GraphSamplingBackend, GraphWorkloadStage, select_execution_graph,
    };
    use crate::llama::graph_decode_capture_inventory::{
        PURE_DECODE_GRAPH_V1_CAPTURE_OPERATION_COUNT, PureDecodeGraphV1CaptureCapabilityInventory,
        PureDecodeGraphV1CaptureOperation,
    };

    fn inventory(
        capability: GraphOperatorCapability,
    ) -> PureDecodeGraphV1CaptureCapabilityInventory {
        PureDecodeGraphV1CaptureCapabilityInventory::new(
            [capability; PURE_DECODE_GRAPH_V1_CAPTURE_OPERATION_COUNT],
        )
    }

    fn request(
        policy: ExecutionGraphPolicy,
        stage: GraphWorkloadStage,
        inventory_state: GraphInventoryState,
        operator_capability: GraphOperatorCapability,
    ) -> GraphDispatchRequest {
        GraphDispatchRequest::new(
            policy,
            GraphDispatchEligibility::new(
                stage,
                true,
                true,
                true,
                GraphCaptureSafety::new(GraphSamplingBackend::GpuGreedy, operator_capability, true),
            ),
            inventory_state,
        )
    }

    #[test]
    fn partial_inventory_forces_auto_eager_and_require_rejection() {
        let partial = PureDecodeGraphV1CaptureCapabilityInventory::default()
            .with_capability(
                PureDecodeGraphV1CaptureOperation::MetadataH2d,
                GraphOperatorCapability::Supported,
            )
            .with_capability(
                PureDecodeGraphV1CaptureOperation::MlpSiluBf16,
                GraphOperatorCapability::Supported,
            );

        let auto = bind_pure_decode_graph_v1_capture_capability_inventory(
            request(
                ExecutionGraphPolicy::Auto,
                GraphWorkloadStage::PureDecode,
                GraphInventoryState::PreparedFull,
                GraphOperatorCapability::Supported,
            ),
            partial,
        );
        assert_eq!(
            select_execution_graph(auto),
            Ok(GraphDispatchDecision::ExactEager(
                GraphFallbackReason::OperatorCapabilityUnknown
            ))
        );

        let require = bind_pure_decode_graph_v1_capture_capability_inventory(
            request(
                ExecutionGraphPolicy::Require,
                GraphWorkloadStage::PureDecode,
                GraphInventoryState::PreparedFull,
                GraphOperatorCapability::Supported,
            ),
            partial,
        );
        assert_eq!(
            select_execution_graph(require),
            Err(GraphDispatchError::RequiredGraphUnavailable {
                reason: GraphFallbackReason::OperatorCapabilityUnknown,
            })
        );
    }

    #[test]
    fn binding_changes_only_operator_evidence() {
        let original = GraphDispatchRequest::new(
            ExecutionGraphPolicy::Auto,
            GraphDispatchEligibility::new(
                GraphWorkloadStage::Mixed,
                false,
                true,
                false,
                GraphCaptureSafety::new(
                    GraphSamplingBackend::Unsupported,
                    GraphOperatorCapability::Supported,
                    false,
                ),
            ),
            GraphInventoryState::Poisoned,
        );

        assert_eq!(
            bind_pure_decode_graph_v1_capture_capability_inventory(
                original,
                PureDecodeGraphV1CaptureCapabilityInventory::default(),
            ),
            GraphDispatchRequest::new(
                ExecutionGraphPolicy::Auto,
                GraphDispatchEligibility::new(
                    GraphWorkloadStage::Mixed,
                    false,
                    true,
                    false,
                    GraphCaptureSafety::new(
                        GraphSamplingBackend::Unsupported,
                        GraphOperatorCapability::Unknown,
                        false,
                    ),
                ),
                GraphInventoryState::Poisoned,
            )
        );
    }

    #[test]
    fn complete_synthetic_evidence_only_passes_the_operator_gate() {
        let complete = inventory(GraphOperatorCapability::Supported);
        assert_eq!(
            select_execution_graph(bind_pure_decode_graph_v1_capture_capability_inventory(
                request(
                    ExecutionGraphPolicy::Auto,
                    GraphWorkloadStage::PureDecode,
                    GraphInventoryState::PreparedFull,
                    GraphOperatorCapability::Unknown,
                ),
                complete,
            )),
            Ok(GraphDispatchDecision::FullGraph)
        );
        assert_eq!(
            select_execution_graph(bind_pure_decode_graph_v1_capture_capability_inventory(
                request(
                    ExecutionGraphPolicy::Auto,
                    GraphWorkloadStage::PureDecode,
                    GraphInventoryState::NotPrepared,
                    GraphOperatorCapability::Unknown,
                ),
                complete,
            )),
            Ok(GraphDispatchDecision::ExactEager(
                GraphFallbackReason::NotPrepared
            ))
        );
    }

    #[test]
    fn pure_decode_evidence_never_admits_prefill_or_mixed_piecewise_graphs() {
        let complete = inventory(GraphOperatorCapability::Supported);

        for stage in [GraphWorkloadStage::Prefill, GraphWorkloadStage::Mixed] {
            let auto = bind_pure_decode_graph_v1_capture_capability_inventory(
                request(
                    ExecutionGraphPolicy::Auto,
                    stage,
                    GraphInventoryState::PreparedPiecewise,
                    GraphOperatorCapability::Supported,
                ),
                complete,
            );
            assert_eq!(
                select_execution_graph(auto),
                Ok(GraphDispatchDecision::ExactEager(
                    GraphFallbackReason::OperatorCapabilityUnknown
                )),
                "{stage:?} must not inherit C07 V1 pure-decode evidence",
            );

            let require = bind_pure_decode_graph_v1_capture_capability_inventory(
                request(
                    ExecutionGraphPolicy::Require,
                    stage,
                    GraphInventoryState::PreparedPiecewise,
                    GraphOperatorCapability::Supported,
                ),
                complete,
            );
            assert_eq!(
                select_execution_graph(require),
                Err(GraphDispatchError::RequiredGraphUnavailable {
                    reason: GraphFallbackReason::OperatorCapabilityUnknown,
                }),
                "{stage:?} must reject required piecewise replay without its own evidence",
            );
        }
    }

    #[test]
    fn exact_c05_evidence_still_denies_the_incomplete_decode_chain() {
        let bound = bind_pure_decode_graph_v1_exact_c05_capture_evidence(request(
            ExecutionGraphPolicy::Auto,
            GraphWorkloadStage::PureDecode,
            GraphInventoryState::PreparedFull,
            GraphOperatorCapability::Supported,
        ))
        .expect("reviewed C05 evidence query must link without CUDA initialization");

        assert_eq!(
            select_execution_graph(bound),
            Ok(GraphDispatchDecision::ExactEager(
                GraphFallbackReason::OperatorCapabilityUnknown
            ))
        );
    }
}
