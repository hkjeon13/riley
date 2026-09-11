//! Closed CPU-only bridge from a graph registry snapshot to dispatch policy.
//!
//! The bridge owns no graph resource and does not launch or capture work. It
//! handles `disabled` before it inspects signature or registry facts. For
//! `auto` and `require`, it verifies signature coherence and then asks the
//! scalar dispatcher to preflight with `NotPrepared`; every pre-lookup
//! rejection returns without touching the registry.

use super::graph::{
    ExecutionGraphPolicy, ExecutionMode, GraphDispatchDecision, GraphDispatchError,
    GraphDispatchRequest, GraphFallbackReason, GraphInventoryState, GraphSignature,
    GraphWorkloadStage, select_execution_graph,
};
use super::graph_registry::{GraphRegistry, GraphRegistryLookup, GraphReplayMode, GraphReplaySlot};

/// Dispatch result paired with the logical replay slot selected from a registry.
///
/// A slot is a non-owning C06 metadata value. A later C07 owner resolves it to
/// an executable graph only after its resource lifetime checks have succeeded.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum GraphRegistryDispatchDecision {
    /// A full graph was selected with its exact logical replay slot.
    FullGraph {
        /// Non-address logical slot selected from the immutable registry.
        replay_slot: GraphReplaySlot,
    },
    /// A piecewise graph was selected with its exact logical replay slot.
    PiecewiseGraph {
        /// Non-address logical slot selected from the immutable registry.
        replay_slot: GraphReplaySlot,
    },
    /// No graph was selected; exact eager execution remains required.
    ExactEager {
        /// Closed explanation for the exact-eager path.
        reason: GraphFallbackReason,
    },
}

impl GraphRegistryDispatchDecision {
    /// Returns the generic dispatcher decision without exposing a resource.
    #[must_use]
    pub const fn decision(self) -> GraphDispatchDecision {
        match self {
            Self::FullGraph { .. } => GraphDispatchDecision::FullGraph,
            Self::PiecewiseGraph { .. } => GraphDispatchDecision::PiecewiseGraph,
            Self::ExactEager { reason } => GraphDispatchDecision::ExactEager(reason),
        }
    }

    /// Returns the selected execution mode.
    #[must_use]
    pub const fn mode(self) -> ExecutionMode {
        self.decision().mode()
    }

    /// Returns the exact eager fallback reason, if a graph was not selected.
    #[must_use]
    pub const fn fallback_reason(self) -> Option<GraphFallbackReason> {
        self.decision().fallback_reason()
    }

    /// Returns the logical slot only for a selected graph path.
    #[must_use]
    pub const fn replay_slot(self) -> Option<GraphReplaySlot> {
        match self {
            Self::FullGraph { replay_slot } | Self::PiecewiseGraph { replay_slot } => {
                Some(replay_slot)
            }
            Self::ExactEager { .. } => None,
        }
    }

    const fn exact_eager(reason: GraphFallbackReason) -> Self {
        Self::ExactEager { reason }
    }
}

trait GraphInventorySource {
    fn lookup(
        &self,
        signature: GraphSignature,
        replay_mode: GraphReplayMode,
    ) -> GraphRegistryLookup<'_>;
}

impl<const MAX_ENTRIES: usize> GraphInventorySource for GraphRegistry<MAX_ENTRIES> {
    fn lookup(
        &self,
        signature: GraphSignature,
        replay_mode: GraphReplayMode,
    ) -> GraphRegistryLookup<'_> {
        Self::lookup(self, signature, replay_mode)
    }
}

/// Selects an exact graph entry from a bounded registry after policy preflight.
///
/// `disabled` and every closed pre-lookup rejection return before the registry
/// is queried. An admitted request must also have the same
/// stage and sampling backend in its eligibility facts and graph signature.
/// The result carries only a logical replay slot; this function never resolves,
/// captures, launches, or mutates a graph resource.
///
/// # Errors
///
/// Returns `GraphDispatchError::RequiredGraphUnavailable` when `require`
/// cannot select an exact, admitted, prepared registry entry.
pub fn select_registered_execution_graph<const MAX_ENTRIES: usize>(
    request: GraphDispatchRequest,
    signature: GraphSignature,
    registry: &GraphRegistry<MAX_ENTRIES>,
) -> Result<GraphRegistryDispatchDecision, GraphDispatchError> {
    select_from_inventory(request, signature, registry)
}

fn select_from_inventory(
    request: GraphDispatchRequest,
    signature: GraphSignature,
    inventory_source: &impl GraphInventorySource,
) -> Result<GraphRegistryDispatchDecision, GraphDispatchError> {
    if request.policy() == ExecutionGraphPolicy::Disabled {
        return without_registry_slot(select_execution_graph(
            request.with_inventory(GraphInventoryState::NotPrepared),
        ));
    }

    let request_stage = request.eligibility().stage();
    let iteration = signature.iteration();
    if iteration.stage() != request_stage
        || iteration.sampling_backend() != request.eligibility().sampling_backend()
    {
        return signature_mismatch(request.policy());
    }

    let preflight =
        select_execution_graph(request.with_inventory(GraphInventoryState::NotPrepared));
    if !requires_registry_lookup(preflight) {
        return without_registry_slot(preflight);
    }
    let Some(replay_mode) = replay_mode_for_stage(request_stage) else {
        return without_registry_slot(preflight);
    };

    let lookup =
        inventory_from_lookup(inventory_source.lookup(signature, replay_mode), replay_mode);
    let (inventory, replay_slot) = match lookup {
        RegistryInventory::CapacityDisabled => {
            let capacity_disabled = request
                .with_eligibility(request.eligibility().with_inventory_enabled(false))
                .with_inventory(GraphInventoryState::NotPrepared);
            return without_registry_slot(select_execution_graph(capacity_disabled));
        }
        RegistryInventory::State {
            inventory,
            replay_slot,
        } => (inventory, replay_slot),
    };
    match select_execution_graph(request.with_inventory(inventory)) {
        Ok(GraphDispatchDecision::FullGraph) => match replay_slot {
            Some(replay_slot) => Ok(GraphRegistryDispatchDecision::FullGraph { replay_slot }),
            None => without_registry_slot(preflight),
        },
        Ok(GraphDispatchDecision::PiecewiseGraph) => match replay_slot {
            Some(replay_slot) => Ok(GraphRegistryDispatchDecision::PiecewiseGraph { replay_slot }),
            None => without_registry_slot(preflight),
        },
        Ok(GraphDispatchDecision::ExactEager(reason)) => {
            Ok(GraphRegistryDispatchDecision::exact_eager(reason))
        }
        Err(error) => Err(error),
    }
}

const fn requires_registry_lookup(
    preflight: Result<GraphDispatchDecision, GraphDispatchError>,
) -> bool {
    matches!(
        preflight,
        Ok(GraphDispatchDecision::ExactEager(
            GraphFallbackReason::NotPrepared
        )) | Err(GraphDispatchError::RequiredGraphUnavailable {
            reason: GraphFallbackReason::NotPrepared,
        })
    )
}

const fn replay_mode_for_stage(stage: GraphWorkloadStage) -> Option<GraphReplayMode> {
    match stage {
        GraphWorkloadStage::PureDecode => Some(GraphReplayMode::FullGraph),
        GraphWorkloadStage::Prefill | GraphWorkloadStage::Mixed => {
            Some(GraphReplayMode::PiecewiseGraph)
        }
        GraphWorkloadStage::Unsupported => None,
    }
}

enum RegistryInventory {
    CapacityDisabled,
    State {
        inventory: GraphInventoryState,
        replay_slot: Option<GraphReplaySlot>,
    },
}

fn inventory_from_lookup(
    lookup: GraphRegistryLookup<'_>,
    replay_mode: GraphReplayMode,
) -> RegistryInventory {
    match lookup {
        GraphRegistryLookup::CapacityDisabled => RegistryInventory::CapacityDisabled,
        GraphRegistryLookup::NotPrepared => RegistryInventory::State {
            inventory: GraphInventoryState::NotPrepared,
            replay_slot: None,
        },
        GraphRegistryLookup::Prepared(entry) => {
            let inventory = match replay_mode {
                GraphReplayMode::FullGraph => GraphInventoryState::PreparedFull,
                GraphReplayMode::PiecewiseGraph => GraphInventoryState::PreparedPiecewise,
            };
            RegistryInventory::State {
                inventory,
                replay_slot: Some(entry.replay_slot()),
            }
        }
        GraphRegistryLookup::Poisoned(_) => RegistryInventory::State {
            inventory: GraphInventoryState::Poisoned,
            replay_slot: None,
        },
    }
}

fn without_registry_slot(
    result: Result<GraphDispatchDecision, GraphDispatchError>,
) -> Result<GraphRegistryDispatchDecision, GraphDispatchError> {
    result.map(|decision| match decision {
        GraphDispatchDecision::ExactEager(reason) => {
            GraphRegistryDispatchDecision::exact_eager(reason)
        }
        GraphDispatchDecision::FullGraph | GraphDispatchDecision::PiecewiseGraph => {
            GraphRegistryDispatchDecision::exact_eager(GraphFallbackReason::NotPrepared)
        }
    })
}

const fn signature_mismatch(
    policy: ExecutionGraphPolicy,
) -> Result<GraphRegistryDispatchDecision, GraphDispatchError> {
    match policy {
        ExecutionGraphPolicy::Disabled => Ok(GraphRegistryDispatchDecision::ExactEager {
            reason: GraphFallbackReason::PolicyDisabled,
        }),
        ExecutionGraphPolicy::Auto => Ok(GraphRegistryDispatchDecision::ExactEager {
            reason: GraphFallbackReason::SignatureMismatch,
        }),
        ExecutionGraphPolicy::Require => Err(GraphDispatchError::RequiredGraphUnavailable {
            reason: GraphFallbackReason::SignatureMismatch,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::{GraphInventorySource, select_from_inventory};
    use crate::llama::graph::{
        ExecutionGraphPolicy, GraphCaptureSafety, GraphComputeType, GraphDataType,
        GraphDeviceSignature, GraphDispatchEligibility, GraphDispatchRequest, GraphFallbackReason,
        GraphGemmPlanSetId, GraphGeometrySignature, GraphImplementationId,
        GraphImplementationSignature, GraphInventoryState, GraphIterationSignature,
        GraphLayoutSignature, GraphMetadataLayoutSignature, GraphModelArchitecture,
        GraphModelSignature, GraphOperatorCapability, GraphReductionPolicyId,
        GraphRevisionFingerprint, GraphSamplingBackend, GraphSignature, GraphStaticSignature,
        GraphTensorSignature, GraphWorkloadStage,
    };
    use crate::llama::graph_registry::{GraphRegistryLookup, GraphReplayMode};

    struct NeverLookup;

    impl GraphInventorySource for NeverLookup {
        fn lookup(&self, _: GraphSignature, _: GraphReplayMode) -> GraphRegistryLookup<'_> {
            panic!("disabled policy must not query a graph inventory source");
        }
    }

    #[test]
    fn disabled_policy_bypasses_the_registry_lookup_source() {
        let request = GraphDispatchRequest::new(
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
        );

        let decision = select_from_inventory(request, signature(), &NeverLookup)
            .expect("disabled policy must preserve exact eager execution");
        assert_eq!(
            decision.fallback_reason(),
            Some(GraphFallbackReason::PolicyDisabled)
        );
        assert!(decision.replay_slot().is_none());
    }

    #[test]
    fn closed_preflight_rejection_bypasses_the_registry_lookup_source() {
        let request = GraphDispatchRequest::new(
            ExecutionGraphPolicy::Auto,
            GraphDispatchEligibility::new(
                GraphWorkloadStage::PureDecode,
                false,
                true,
                true,
                GraphCaptureSafety::new(
                    GraphSamplingBackend::GpuGreedy,
                    GraphOperatorCapability::Supported,
                    true,
                ),
            ),
            GraphInventoryState::NotPrepared,
        );

        let decision = select_from_inventory(request, signature(), &NeverLookup)
            .expect("closed preflight miss must retain exact eager execution");
        assert_eq!(
            decision.fallback_reason(),
            Some(GraphFallbackReason::UnsupportedShape)
        );
        assert!(decision.replay_slot().is_none());
    }

    fn signature() -> GraphSignature {
        let model = GraphModelSignature::new(
            GraphModelArchitecture::LlamaDecoder,
            1,
            GraphRevisionFingerprint::from_bytes([1; 32]),
            1,
        );
        let device = GraphDeviceSignature::new(8, 9, 12_804, 12_804, 1);
        let tensors = GraphTensorSignature::new(
            GraphDataType::BFloat16,
            GraphDataType::BFloat16,
            GraphComputeType::Float32,
        );
        let geometry = GraphGeometrySignature::new(32, 4_096, 11_008, 32_000, 32, 8, 128);
        let layout = GraphLayoutSignature::new(
            8_192,
            16,
            1,
            GraphMetadataLayoutSignature::new(1, [0xA1; 32]),
        );
        let implementations = GraphImplementationSignature::new(
            GraphImplementationId::new(1),
            GraphImplementationId::new(2),
            GraphImplementationId::new(3),
            GraphImplementationId::new(4),
            GraphGemmPlanSetId::new(5),
            GraphReductionPolicyId::new(6),
        );
        GraphSignature::new(
            GraphStaticSignature::new(model, device, tensors, geometry, layout, implementations),
            GraphIterationSignature::new(
                GraphWorkloadStage::PureDecode,
                1,
                GraphSamplingBackend::GpuGreedy,
            ),
        )
    }
}
