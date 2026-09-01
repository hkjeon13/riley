//! C07 V1 bridge into the immutable C06 graph-registry dispatcher.
//!
//! C07-21 consumes only a complete C07-20 cache-key value. An already
//! ineligible C07 candidate remains outside C06; a bound candidate delegates
//! its unchanged caller request and exact signature once to the existing C06
//! registry adapter. This module resolves no replay slot and owns no resource.

use super::graph::{GraphDispatchError, GraphDispatchRequest};
use super::graph_decode_c06_signature::{
    PureDecodeGraphV1C06Signature, PureDecodeGraphV1C06SignatureBinding,
};
use super::graph_decode_preflight::PureDecodeGraphV1Ineligibility;
use super::graph_registry::GraphRegistry;
use super::graph_registry_dispatch::{
    GraphRegistryDispatchDecision, select_registered_execution_graph,
};

/// C07 facts paired with one opaque C06 registry-selection decision.
///
/// The logical replay slot remains entirely inside C06's decision value. A
/// later native owner must resolve it only after its own resource-lifetime
/// checks have succeeded.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[must_use]
pub(crate) struct PureDecodeGraphV1C06RegistryDispatchBinding {
    signature_binding: PureDecodeGraphV1C06SignatureBinding,
    decision: GraphRegistryDispatchDecision,
}

impl PureDecodeGraphV1C06RegistryDispatchBinding {
    /// Returns the original exact C07-20 complete-signature binding.
    pub(crate) const fn signature_binding(self) -> PureDecodeGraphV1C06SignatureBinding {
        self.signature_binding
    }

    /// Returns C06's opaque immutable registry-selection outcome.
    #[must_use]
    pub(crate) const fn decision(self) -> GraphRegistryDispatchDecision {
        self.decision
    }
}

/// Closed result of applying C06 registry selection to one C07-20 value.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[must_use]
#[allow(clippy::large_enum_variant)] // C07 keeps this bridge allocation-free.
pub(crate) enum PureDecodeGraphV1C06RegistryDispatch {
    /// An exact C07 complete key was passed unchanged through the C06 adapter.
    Bound(PureDecodeGraphV1C06RegistryDispatchBinding),
    /// The V1 candidate was already ineligible before request or registry inspection.
    Ineligible(PureDecodeGraphV1Ineligibility),
}

/// Result of C07-to-C06 immutable registry-selection delegation.
pub(crate) type PureDecodeGraphV1C06RegistryDispatchResult =
    Result<PureDecodeGraphV1C06RegistryDispatch, GraphDispatchError>;

/// Selects an opaque C06 registry decision for one bound C07-20 key.
///
/// The caller's C06 request is passed through unchanged. C07 does not repair
/// stage, sampling, layout-match, inventory, policy, or capture-safety facts:
/// the existing C06 adapter preserves its own signature-coherence, preflight,
/// exact-lookup, and `require` error ordering. An ineligible C07 value returns
/// before this function inspects the request or registry.
pub(crate) fn select_pure_decode_graph_v1_c06_registry_dispatch<const MAX_ENTRIES: usize>(
    candidate: &PureDecodeGraphV1C06Signature,
    request: GraphDispatchRequest,
    registry: &GraphRegistry<MAX_ENTRIES>,
) -> PureDecodeGraphV1C06RegistryDispatchResult {
    match *candidate {
        PureDecodeGraphV1C06Signature::Ineligible(reason) => {
            Ok(PureDecodeGraphV1C06RegistryDispatch::Ineligible(reason))
        }
        PureDecodeGraphV1C06Signature::Bound(binding) => {
            select_registered_execution_graph(request, binding.signature(), registry).map(
                |decision| {
                    PureDecodeGraphV1C06RegistryDispatch::Bound(
                        PureDecodeGraphV1C06RegistryDispatchBinding {
                            signature_binding: binding,
                            decision,
                        },
                    )
                },
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        PureDecodeGraphV1C06RegistryDispatch, PureDecodeGraphV1C06RegistryDispatchBinding,
        select_pure_decode_graph_v1_c06_registry_dispatch,
    };
    use crate::llama::graph::{
        ExecutionGraphPolicy, GraphCaptureSafety, GraphComputeType, GraphDataType,
        GraphDeviceSignature, GraphDispatchEligibility, GraphDispatchError, GraphDispatchRequest,
        GraphFallbackReason, GraphGemmPlanSetId, GraphGeometrySignature, GraphImplementationId,
        GraphImplementationSignature, GraphInventoryState, GraphLayoutSignature,
        GraphMetadataLayoutSignature, GraphModelArchitecture, GraphModelSignature,
        GraphOperatorCapability, GraphReductionPolicyId, GraphRevisionFingerprint,
        GraphSamplingBackend, GraphSignature, GraphStaticSignature, GraphTensorSignature,
        GraphWorkloadStage,
    };
    use crate::llama::graph_decode_binding::PureDecodeGraphMetadataBinding;
    use crate::llama::graph_decode_c06_identity::bind_pure_decode_graph_v1_c06_identity;
    use crate::llama::graph_decode_c06_registry_observation::observe_pure_decode_graph_v1_c06_registry_dispatch;
    use crate::llama::graph_decode_c06_signature::{
        PureDecodeGraphV1C06Signature, PureDecodeGraphV1C06SignatureBinding,
        compose_pure_decode_graph_v1_c06_signature,
    };
    use crate::llama::graph_decode_layout::{
        PureDecodeGraphMetadataLayout, PureDecodeGraphMetadataLayoutSpec,
    };
    use crate::llama::graph_decode_padding::plan_pure_decode_graph_padding;
    use crate::llama::graph_decode_preflight::PureDecodeGraphV1Ineligibility;
    use crate::llama::graph_decode_preflight_binding::PureDecodeGraphV1LayoutBinding;
    use crate::llama::graph_metrics::GraphDispatchMetrics;
    use crate::llama::graph_registry::{
        GraphEntryFootprint, GraphRegistry, GraphRegistryEntry, GraphRegistryEntryState,
        GraphRegistryLimits, GraphReplayMode, GraphReplaySlot,
    };
    use crate::llama::graph_registry_dispatch::GraphRegistryDispatchDecision;

    fn layout() -> PureDecodeGraphMetadataLayout {
        PureDecodeGraphMetadataLayout::try_new(PureDecodeGraphMetadataLayoutSpec::new(4, 4, 1, 1))
            .expect("M4 C07 cold layout must be representable")
    }

    fn static_signature(
        metadata_layout: GraphMetadataLayoutSignature,
        revision: u8,
    ) -> GraphStaticSignature {
        let revision_u32 = u32::from(revision);

        GraphStaticSignature::new(
            GraphModelSignature::new(
                GraphModelArchitecture::LlamaDecoder,
                revision_u32,
                GraphRevisionFingerprint::from_bytes([revision; 32]),
                revision_u32,
            ),
            GraphDeviceSignature::new(8, 9, 12_804, 12_804, revision_u32),
            GraphTensorSignature::new(
                GraphDataType::BFloat16,
                GraphDataType::BFloat16,
                GraphComputeType::Float32,
            ),
            GraphGeometrySignature::new(32, 4_096, 11_008, 32_000, 32, 8, 128),
            GraphLayoutSignature::new(8_192, 16, revision_u32, metadata_layout),
            GraphImplementationSignature::new(
                GraphImplementationId::new(revision_u32),
                GraphImplementationId::new(revision_u32 + 1),
                GraphImplementationId::new(revision_u32 + 2),
                GraphImplementationId::new(revision_u32 + 3),
                GraphGemmPlanSetId::new(revision_u32),
                GraphReductionPolicyId::new(revision_u32),
            ),
        )
    }

    fn complete_candidate(sampling_backend: GraphSamplingBackend) -> PureDecodeGraphV1C06Signature {
        let layout = layout();
        let padding = plan_pure_decode_graph_padding(4).expect("A4 selects M4");
        let c07_binding = PureDecodeGraphV1LayoutBinding::Bound(
            PureDecodeGraphMetadataBinding::try_new(layout, padding)
                .expect("matching M4 facts must bind"),
        );
        let partial = bind_pure_decode_graph_v1_c06_identity(&c07_binding, sampling_backend);
        let metadata_layout = match partial {
            crate::llama::graph_decode_c06_identity::PureDecodeGraphV1C06Identity::Bound(
                binding,
            ) => binding.metadata_layout_signature(),
            crate::llama::graph_decode_c06_identity::PureDecodeGraphV1C06Identity::Ineligible(
                reason,
            ) => panic!("expected a C07 partial identity, got {reason:?}"),
        };
        compose_pure_decode_graph_v1_c06_signature(&partial, static_signature(metadata_layout, 1))
            .expect("matching static metadata must compose a complete C07 key")
    }

    fn expect_bound(
        result: &PureDecodeGraphV1C06RegistryDispatch,
    ) -> PureDecodeGraphV1C06RegistryDispatchBinding {
        match *result {
            PureDecodeGraphV1C06RegistryDispatch::Bound(binding) => binding,
            PureDecodeGraphV1C06RegistryDispatch::Ineligible(reason) => {
                panic!("expected a C06 registry decision, got {reason:?}")
            }
        }
    }

    fn expect_signature_binding(
        candidate: &PureDecodeGraphV1C06Signature,
    ) -> PureDecodeGraphV1C06SignatureBinding {
        match *candidate {
            PureDecodeGraphV1C06Signature::Bound(binding) => binding,
            PureDecodeGraphV1C06Signature::Ineligible(reason) => {
                panic!("expected a complete C07 key, got {reason:?}")
            }
        }
    }

    fn request(
        policy: ExecutionGraphPolicy,
        stage: GraphWorkloadStage,
        sampling_backend: GraphSamplingBackend,
        active_row_bucket_supported: bool,
        metadata_layout_matches: bool,
        inventory_enabled: bool,
    ) -> GraphDispatchRequest {
        GraphDispatchRequest::new(
            policy,
            GraphDispatchEligibility::new(
                stage,
                active_row_bucket_supported,
                metadata_layout_matches,
                inventory_enabled,
                GraphCaptureSafety::new(sampling_backend, GraphOperatorCapability::Supported, true),
            ),
            GraphInventoryState::NotPrepared,
        )
    }

    fn registry(
        signature: &crate::llama::graph::GraphSignature,
        state: GraphRegistryEntryState,
    ) -> GraphRegistry<1> {
        GraphRegistry::try_new(
            GraphRegistryLimits::new(1, 1, 0, 16, 16),
            &[GraphRegistryEntry::new(
                *signature,
                GraphReplayMode::FullGraph,
                GraphReplaySlot::new(17),
                state,
                GraphEntryFootprint::new(1, 1),
            )],
        )
        .expect("one exact full entry must fit the immutable registry")
    }

    #[test]
    fn bound_candidate_preserves_the_exact_c06_full_decision_and_slot() {
        let candidate = complete_candidate(GraphSamplingBackend::GpuGreedy);
        let signature_binding = expect_signature_binding(&candidate);
        let registry = registry(
            &signature_binding.signature(),
            GraphRegistryEntryState::Prepared,
        );
        let result = select_pure_decode_graph_v1_c06_registry_dispatch(
            &candidate,
            request(
                ExecutionGraphPolicy::Auto,
                GraphWorkloadStage::PureDecode,
                GraphSamplingBackend::GpuGreedy,
                true,
                true,
                true,
            ),
            &registry,
        )
        .expect("a matching prepared full entry must select under auto");
        let binding = expect_bound(&result);

        assert_eq!(binding.signature_binding(), signature_binding);
        assert_eq!(
            binding.decision(),
            GraphRegistryDispatchDecision::FullGraph {
                replay_slot: GraphReplaySlot::new(17),
            }
        );
    }

    #[test]
    fn bound_candidate_preserves_c06_miss_poison_and_capacity_outcomes() {
        let candidate = complete_candidate(GraphSamplingBackend::GpuGreedy);
        let signature_binding = expect_signature_binding(&candidate);
        let prepared = registry(
            &signature_binding.signature(),
            GraphRegistryEntryState::Prepared,
        );
        let missing_signature = GraphSignature::new(
            static_signature(signature_binding.identity().metadata_layout_signature(), 2),
            signature_binding.signature().iteration(),
        );
        let missing = registry(&missing_signature, GraphRegistryEntryState::Prepared);
        let poisoned = registry(
            &signature_binding.signature(),
            GraphRegistryEntryState::Poisoned,
        );
        let request = request(
            ExecutionGraphPolicy::Auto,
            GraphWorkloadStage::PureDecode,
            GraphSamplingBackend::GpuGreedy,
            true,
            true,
            true,
        );

        for (registry, expected) in [
            (
                &prepared,
                GraphRegistryDispatchDecision::FullGraph {
                    replay_slot: GraphReplaySlot::new(17),
                },
            ),
            (
                &missing,
                GraphRegistryDispatchDecision::ExactEager {
                    reason: GraphFallbackReason::NotPrepared,
                },
            ),
            (
                &poisoned,
                GraphRegistryDispatchDecision::ExactEager {
                    reason: GraphFallbackReason::GraphPoisoned,
                },
            ),
        ] {
            assert_eq!(
                expect_bound(
                    &select_pure_decode_graph_v1_c06_registry_dispatch(
                        &candidate, request, registry
                    )
                    .expect("auto must preserve C06 closed outcomes"),
                )
                .decision(),
                expected,
            );
        }

        let capacity_disabled = GraphRegistry::<0>::capacity_disabled();
        assert_eq!(
            expect_bound(
                &select_pure_decode_graph_v1_c06_registry_dispatch(
                    &candidate,
                    request,
                    &capacity_disabled,
                )
                .expect("auto capacity miss must remain an opaque C06 decision"),
            )
            .decision(),
            GraphRegistryDispatchDecision::ExactEager {
                reason: GraphFallbackReason::CapacityDisabled,
            },
        );
    }

    #[test]
    fn caller_request_mismatches_and_require_errors_are_not_repaired_by_c07() {
        let candidate = complete_candidate(GraphSamplingBackend::GpuGreedy);
        let signature_binding = expect_signature_binding(&candidate);
        let registry = registry(
            &signature_binding.signature(),
            GraphRegistryEntryState::Prepared,
        );

        assert_eq!(
            expect_bound(
                &select_pure_decode_graph_v1_c06_registry_dispatch(
                    &candidate,
                    request(
                        ExecutionGraphPolicy::Auto,
                        GraphWorkloadStage::PureDecode,
                        GraphSamplingBackend::Unsupported,
                        true,
                        true,
                        true,
                    ),
                    &registry,
                )
                .expect("auto must retain C06 signature mismatch as exact eager"),
            )
            .decision(),
            GraphRegistryDispatchDecision::ExactEager {
                reason: GraphFallbackReason::SignatureMismatch,
            },
        );
        assert_eq!(
            select_pure_decode_graph_v1_c06_registry_dispatch(
                &candidate,
                request(
                    ExecutionGraphPolicy::Require,
                    GraphWorkloadStage::Mixed,
                    GraphSamplingBackend::GpuGreedy,
                    true,
                    true,
                    true,
                ),
                &registry,
            ),
            Err(GraphDispatchError::RequiredGraphUnavailable {
                reason: GraphFallbackReason::SignatureMismatch,
            }),
        );
    }

    #[test]
    fn bound_candidate_preserves_c06_disabled_precedence() {
        let candidate = complete_candidate(GraphSamplingBackend::GpuGreedy);
        let registry = GraphRegistry::<0>::capacity_disabled();

        assert_eq!(
            expect_bound(
                &select_pure_decode_graph_v1_c06_registry_dispatch(
                    &candidate,
                    request(
                        ExecutionGraphPolicy::Disabled,
                        GraphWorkloadStage::Unsupported,
                        GraphSamplingBackend::Unsupported,
                        false,
                        false,
                        false,
                    ),
                    &registry,
                )
                .expect("disabled must preserve C06's pre-registry eager decision"),
            )
            .decision(),
            GraphRegistryDispatchDecision::ExactEager {
                reason: GraphFallbackReason::PolicyDisabled,
            },
        );
    }

    #[test]
    fn bound_candidate_preserves_c06_preflight_precedence() {
        let candidate = complete_candidate(GraphSamplingBackend::GpuGreedy);
        let signature_binding = expect_signature_binding(&candidate);
        let registry = registry(
            &signature_binding.signature(),
            GraphRegistryEntryState::Prepared,
        );

        assert_eq!(
            expect_bound(
                &select_pure_decode_graph_v1_c06_registry_dispatch(
                    &candidate,
                    request(
                        ExecutionGraphPolicy::Auto,
                        GraphWorkloadStage::PureDecode,
                        GraphSamplingBackend::GpuGreedy,
                        false,
                        false,
                        false,
                    ),
                    &registry,
                )
                .expect("auto must preserve C06 preflight fallback"),
            )
            .decision(),
            GraphRegistryDispatchDecision::ExactEager {
                reason: GraphFallbackReason::CapacityDisabled,
            },
        );
    }

    #[test]
    fn ineligible_candidate_short_circuits_before_c06_request_or_registry_handling() {
        let reason = PureDecodeGraphV1Ineligibility::UnsupportedActiveRows { active_rows: 33 };
        let registry = GraphRegistry::<0>::capacity_disabled();

        assert_eq!(
            select_pure_decode_graph_v1_c06_registry_dispatch(
                &PureDecodeGraphV1C06Signature::Ineligible(reason),
                request(
                    ExecutionGraphPolicy::Require,
                    GraphWorkloadStage::Unsupported,
                    GraphSamplingBackend::Unsupported,
                    false,
                    false,
                    false,
                ),
                &registry,
            ),
            Ok(PureDecodeGraphV1C06RegistryDispatch::Ineligible(reason)),
        );
    }

    #[test]
    fn c07_22_observes_bound_full_and_eager_decisions_once() {
        let candidate = complete_candidate(GraphSamplingBackend::GpuGreedy);
        let signature_binding = expect_signature_binding(&candidate);
        let prepared = registry(
            &signature_binding.signature(),
            GraphRegistryEntryState::Prepared,
        );
        let request = request(
            ExecutionGraphPolicy::Auto,
            GraphWorkloadStage::PureDecode,
            GraphSamplingBackend::GpuGreedy,
            true,
            true,
            true,
        );
        let mut metrics = GraphDispatchMetrics::new();

        let full =
            select_pure_decode_graph_v1_c06_registry_dispatch(&candidate, request, &prepared);
        observe_pure_decode_graph_v1_c06_registry_dispatch(&full, &mut metrics);
        let full = full.expect("C07-22 must preserve the prepared C06 decision");
        assert_eq!(
            expect_bound(&full).decision(),
            GraphRegistryDispatchDecision::FullGraph {
                replay_slot: GraphReplaySlot::new(17),
            },
        );
        assert_eq!(metrics.snapshot().full_graph_selected_count(), 1);
        assert_eq!(metrics.snapshot().exact_eager_count(), 0);

        let capacity_disabled = GraphRegistry::<0>::capacity_disabled();
        let eager = select_pure_decode_graph_v1_c06_registry_dispatch(
            &candidate,
            request,
            &capacity_disabled,
        );
        observe_pure_decode_graph_v1_c06_registry_dispatch(&eager, &mut metrics);
        let eager = eager.expect("C07-22 must preserve C06's capacity eager outcome");
        assert_eq!(
            expect_bound(&eager).decision(),
            GraphRegistryDispatchDecision::ExactEager {
                reason: GraphFallbackReason::CapacityDisabled,
            },
        );
        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.full_graph_selected_count(), 1);
        assert_eq!(snapshot.exact_eager_count(), 1);
        assert_eq!(
            snapshot.fallback_reason_count(GraphFallbackReason::CapacityDisabled),
            1,
        );
    }
}
