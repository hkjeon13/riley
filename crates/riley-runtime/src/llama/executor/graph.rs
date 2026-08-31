//! Closed and allocation-free execution-graph dispatch policy.
//!
//! This C06-0 module owns neither a CUDA Graph nor a model executor. It turns
//! already-observed scalar eligibility and inventory facts into one of three
//! modes, or a fail-closed `require` rejection. Signature construction,
//! native graph lookup, and runtime wiring remain separate follow-up slices.

use std::error;
use std::fmt;

/// Operator-specific graph-capture admission result supplied to the dispatcher.
///
/// This is a runtime policy value, not a CUDA ABI handle. A later C06 adapter
/// maps the C05 native capability record into this closed vocabulary.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
pub enum GraphOperatorCapability {
    /// No reviewed capture-safety result is available for the selected work.
    #[default]
    Unknown,
    /// At least one selected operator is not capture-safe.
    Unsupported,
    /// Every selected operator is admitted for the requested graph mode.
    Supported,
}

/// Workload stage considered by the C06 execution-graph policy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum GraphWorkloadStage {
    /// A prefill-only iteration.
    Prefill,
    /// A pure decode iteration.
    PureDecode,
    /// An iteration that mixes prefill and decode work.
    Mixed,
    /// A stage with no reviewed execution-graph policy.
    Unsupported,
}

/// Sampling/output backend considered for full-graph admission.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum GraphSamplingBackend {
    /// The fixed GPU greedy backend required by the initial full-graph path.
    GpuGreedy,
    /// Any other or unreviewed sampling/output backend.
    ///
    /// It can remain an eager boundary around a piecewise graph, but cannot be
    /// part of a full graph replay.
    Unsupported,
}

/// Operator and backend facts that must both admit graph dispatch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GraphCaptureSafety {
    sampling_backend: GraphSamplingBackend,
    operator_capability: GraphOperatorCapability,
    backend_capture_safe: bool,
}

impl GraphCaptureSafety {
    /// Creates an allocation-free capture-safety fact bundle.
    #[must_use]
    pub const fn new(
        sampling_backend: GraphSamplingBackend,
        operator_capability: GraphOperatorCapability,
        backend_capture_safe: bool,
    ) -> Self {
        Self {
            sampling_backend,
            operator_capability,
            backend_capture_safe,
        }
    }
}

/// Scalar eligibility facts observed before a graph inventory lookup result is
/// applied. Every fact is caller-owned and has no CUDA side effect.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GraphDispatchEligibility {
    stage: GraphWorkloadStage,
    active_row_bucket_supported: bool,
    metadata_layout_matches: bool,
    inventory_enabled: bool,
    capture_safety: GraphCaptureSafety,
}

impl GraphDispatchEligibility {
    /// Creates the fixed policy input for one proposed execution-graph lookup.
    #[must_use]
    pub const fn new(
        stage: GraphWorkloadStage,
        active_row_bucket_supported: bool,
        metadata_layout_matches: bool,
        inventory_enabled: bool,
        capture_safety: GraphCaptureSafety,
    ) -> Self {
        Self {
            stage,
            active_row_bucket_supported,
            metadata_layout_matches,
            inventory_enabled,
            capture_safety,
        }
    }
}

/// Policy selected at startup or by an explicit benchmark/qualification caller.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
pub enum ExecutionGraphPolicy {
    /// Preserve the established exact eager path without inspecting graph facts.
    #[default]
    Disabled,
    /// Select a matching prepared graph, otherwise execute exact eager work.
    Auto,
    /// Reject an iteration without a matching prepared graph.
    Require,
}

/// One cold-prepared inventory lookup result for the exact proposed signature.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum GraphInventoryState {
    /// No graph was prepared for the requested signature.
    NotPrepared,
    /// A complete pure-decode graph is ready for replay.
    PreparedFull,
    /// An admitted fixed segment is ready for piecewise replay.
    PreparedPiecewise,
    /// A graph entry had an unrecoverable launch/completion failure.
    Poisoned,
}

/// Complete, scalar-only input for one dispatcher decision.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GraphDispatchRequest {
    policy: ExecutionGraphPolicy,
    eligibility: GraphDispatchEligibility,
    inventory: GraphInventoryState,
}

impl GraphDispatchRequest {
    /// Combines policy, eligibility, and the exact signature's inventory state.
    #[must_use]
    pub const fn new(
        policy: ExecutionGraphPolicy,
        eligibility: GraphDispatchEligibility,
        inventory: GraphInventoryState,
    ) -> Self {
        Self {
            policy,
            eligibility,
            inventory,
        }
    }
}

/// Mode selected for one iteration after graph policy evaluation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ExecutionMode {
    /// Replay a fully prepared pure-decode CUDA Graph.
    FullGraph,
    /// Replay a prepared graph segment around dynamic work boundaries.
    PiecewiseGraph,
    /// Use the existing, exact command-batch execution path.
    ExactEager,
}

/// Closed explanation for a graph miss or exact-eager fallback.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum GraphFallbackReason {
    /// The configured policy deliberately disables graph dispatch.
    PolicyDisabled,
    /// The requested signature has no prepared graph entry.
    NotPrepared,
    /// The iteration stage has no reviewed graph mode.
    UnsupportedStage,
    /// The active-row bucket is not part of the prepared graph inventory.
    UnsupportedShape,
    /// The sampling/output backend is not the admitted GPU-greedy backend.
    UnsupportedSampling,
    /// The runtime metadata layout differs from the prepared graph layout.
    LayoutMismatch,
    /// A selected backend or operator is not capture-safe.
    BackendNotCaptureSafe,
    /// The exact graph entry is poisoned and must never be replayed.
    GraphPoisoned,
    /// Cold graph inventory capacity was disabled or exhausted.
    CapacityDisabled,
    /// At least one selected operator's capture capability is unknown.
    OperatorCapabilityUnknown,
}

impl GraphFallbackReason {
    /// Stable allocation-free identifier for metrics and traces.
    #[must_use]
    pub const fn id(self) -> &'static str {
        match self {
            Self::PolicyDisabled => "policy-disabled",
            Self::NotPrepared => "not-prepared",
            Self::UnsupportedStage => "unsupported-stage",
            Self::UnsupportedShape => "unsupported-shape",
            Self::UnsupportedSampling => "unsupported-sampling",
            Self::LayoutMismatch => "layout-mismatch",
            Self::BackendNotCaptureSafe => "backend-not-capture-safe",
            Self::GraphPoisoned => "graph-poisoned",
            Self::CapacityDisabled => "capacity-disabled",
            Self::OperatorCapabilityUnknown => "operator-capability-unknown",
        }
    }
}

/// A successful graph-dispatch decision.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GraphDispatchDecision {
    /// A full graph is ready and eligible for pure decode.
    FullGraph,
    /// A piecewise graph is ready and eligible for prefill or mixed work.
    PiecewiseGraph,
    /// No graph runs; the established exact eager path remains correct.
    ExactEager(GraphFallbackReason),
}

impl GraphDispatchDecision {
    /// Execution mode selected by this decision.
    #[must_use]
    pub const fn mode(self) -> ExecutionMode {
        match self {
            Self::FullGraph => ExecutionMode::FullGraph,
            Self::PiecewiseGraph => ExecutionMode::PiecewiseGraph,
            Self::ExactEager(_) => ExecutionMode::ExactEager,
        }
    }

    /// Exact-eager explanation, if no graph is selected.
    #[must_use]
    pub const fn fallback_reason(self) -> Option<GraphFallbackReason> {
        match self {
            Self::ExactEager(reason) => Some(reason),
            Self::FullGraph | Self::PiecewiseGraph => None,
        }
    }
}

/// Fail-closed outcome when the `require` policy cannot select a graph.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GraphDispatchError {
    /// The required graph path is unavailable for the exact request facts.
    RequiredGraphUnavailable {
        /// Closed reason that prevented graph selection.
        reason: GraphFallbackReason,
    },
}

impl fmt::Display for GraphDispatchError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RequiredGraphUnavailable { reason } => write!(
                formatter,
                "execution graph is required but unavailable: {}",
                reason.id()
            ),
        }
    }
}

impl error::Error for GraphDispatchError {}

/// Selects graph replay, exact eager work, or a fail-closed `require` error.
///
/// This function is pure and allocation-free. It never opens a capture, looks
/// up a raw handle, mutates an executor, or turns a graph miss into capture on
/// the iteration hot path.
///
/// # Errors
///
/// Returns `GraphDispatchError::RequiredGraphUnavailable` only when the request
/// uses `ExecutionGraphPolicy::Require` and no eligible, matching prepared graph
/// is available.
pub const fn select_execution_graph(
    request: GraphDispatchRequest,
) -> Result<GraphDispatchDecision, GraphDispatchError> {
    match request.policy {
        ExecutionGraphPolicy::Disabled => {
            return Ok(GraphDispatchDecision::ExactEager(
                GraphFallbackReason::PolicyDisabled,
            ));
        }
        ExecutionGraphPolicy::Auto | ExecutionGraphPolicy::Require => {}
    }

    let eligibility = request.eligibility;
    match eligibility.stage {
        GraphWorkloadStage::Unsupported => {
            return fallback(request.policy, GraphFallbackReason::UnsupportedStage);
        }
        GraphWorkloadStage::Prefill
        | GraphWorkloadStage::PureDecode
        | GraphWorkloadStage::Mixed => {}
    }
    if !eligibility.inventory_enabled {
        return fallback(request.policy, GraphFallbackReason::CapacityDisabled);
    }
    if !eligibility.active_row_bucket_supported {
        return fallback(request.policy, GraphFallbackReason::UnsupportedShape);
    }
    if !eligibility.metadata_layout_matches {
        return fallback(request.policy, GraphFallbackReason::LayoutMismatch);
    }
    match eligibility.stage {
        GraphWorkloadStage::PureDecode => match eligibility.capture_safety.sampling_backend {
            GraphSamplingBackend::GpuGreedy => {}
            GraphSamplingBackend::Unsupported => {
                return fallback(request.policy, GraphFallbackReason::UnsupportedSampling);
            }
        },
        GraphWorkloadStage::Prefill | GraphWorkloadStage::Mixed => {}
        GraphWorkloadStage::Unsupported => {
            return fallback(request.policy, GraphFallbackReason::UnsupportedStage);
        }
    }
    match eligibility.capture_safety.operator_capability {
        GraphOperatorCapability::Unknown => {
            return fallback(
                request.policy,
                GraphFallbackReason::OperatorCapabilityUnknown,
            );
        }
        GraphOperatorCapability::Unsupported => {
            return fallback(request.policy, GraphFallbackReason::BackendNotCaptureSafe);
        }
        GraphOperatorCapability::Supported => {}
    }
    if !eligibility.capture_safety.backend_capture_safe {
        return fallback(request.policy, GraphFallbackReason::BackendNotCaptureSafe);
    }

    match (eligibility.stage, request.inventory) {
        (GraphWorkloadStage::PureDecode, GraphInventoryState::PreparedFull) => {
            Ok(GraphDispatchDecision::FullGraph)
        }
        (
            GraphWorkloadStage::Prefill | GraphWorkloadStage::Mixed,
            GraphInventoryState::PreparedPiecewise,
        ) => Ok(GraphDispatchDecision::PiecewiseGraph),
        (_, GraphInventoryState::Poisoned) => {
            fallback(request.policy, GraphFallbackReason::GraphPoisoned)
        }
        _ => fallback(request.policy, GraphFallbackReason::NotPrepared),
    }
}

const fn fallback(
    policy: ExecutionGraphPolicy,
    reason: GraphFallbackReason,
) -> Result<GraphDispatchDecision, GraphDispatchError> {
    match policy {
        ExecutionGraphPolicy::Disabled | ExecutionGraphPolicy::Auto => {
            Ok(GraphDispatchDecision::ExactEager(reason))
        }
        ExecutionGraphPolicy::Require => {
            Err(GraphDispatchError::RequiredGraphUnavailable { reason })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ExecutionGraphPolicy, ExecutionMode, GraphCaptureSafety, GraphDispatchDecision,
        GraphDispatchEligibility, GraphDispatchError, GraphDispatchRequest, GraphFallbackReason,
        GraphInventoryState, GraphOperatorCapability, GraphSamplingBackend, GraphWorkloadStage,
        select_execution_graph,
    };

    const fn capture_safety() -> GraphCaptureSafety {
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
            GraphDispatchEligibility::new(stage, true, true, true, capture_safety()),
            inventory,
        )
    }

    #[test]
    fn disabled_policy_never_selects_a_ready_graph() {
        let decision = select_execution_graph(request(
            ExecutionGraphPolicy::Disabled,
            GraphWorkloadStage::PureDecode,
            GraphInventoryState::PreparedFull,
        ))
        .expect("disabled policy must preserve eager execution");

        assert_eq!(decision.mode(), ExecutionMode::ExactEager);
        assert_eq!(
            decision.fallback_reason(),
            Some(GraphFallbackReason::PolicyDisabled)
        );
    }

    #[test]
    fn auto_selects_only_the_stage_matching_prepared_mode() {
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
                GraphWorkloadStage::Prefill,
                GraphInventoryState::PreparedPiecewise,
            )),
            Ok(GraphDispatchDecision::PiecewiseGraph)
        );
        assert_eq!(
            select_execution_graph(request(
                ExecutionGraphPolicy::Auto,
                GraphWorkloadStage::Mixed,
                GraphInventoryState::PreparedPiecewise,
            )),
            Ok(GraphDispatchDecision::PiecewiseGraph)
        );
    }

    #[test]
    fn auto_reports_closed_fallback_reasons_without_graph_work() {
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
                GraphWorkloadStage::PureDecode,
                GraphInventoryState::Poisoned,
            )),
            Ok(GraphDispatchDecision::ExactEager(
                GraphFallbackReason::GraphPoisoned
            ))
        );
        assert_eq!(GraphFallbackReason::LayoutMismatch.id(), "layout-mismatch");
    }

    #[test]
    fn require_turns_every_miss_into_a_fail_closed_error() {
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

        let unsupported_sampling = GraphDispatchRequest::new(
            ExecutionGraphPolicy::Require,
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
            GraphInventoryState::PreparedFull,
        );
        assert_eq!(
            select_execution_graph(unsupported_sampling),
            Err(GraphDispatchError::RequiredGraphUnavailable {
                reason: GraphFallbackReason::UnsupportedSampling,
            })
        );
    }
}
