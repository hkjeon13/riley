//! Private C07 association between one selected full-graph identity and C05 ownership.
//!
//! C06 registry selection owns only immutable metadata and exposes a logical
//! replay slot. C05-6 owns an instantiated native graph with its capture
//! stream and fixed device buffer by value. This narrow C07-24 boundary joins
//! one exact C07 full-graph selection identity to one such C05 owner without
//! treating the current fixed-fill graph as a Llama decode graph.
//!
//! It deliberately performs no C06 selection or observation, no H2D, no
//! graph launch, and no executor work. A caller receives a mutable borrow of
//! the retained owner only after full signature and logical slot equality are
//! established. The future dynamic-input/capture owner must establish that a
//! real decode graph has the same resource and semantic contract before it
//! can add a production launch path.

use std::error;
use std::fmt;

use riley_cuda::{CudaResult, OwnedGraphExec, OwnedGraphFillResources};

use super::graph::{GraphFallbackReason, GraphSignature};
use super::graph_decode_c06_registry_dispatch::PureDecodeGraphV1C06RegistryDispatchBinding;
use super::graph_decode_c06_signature::PureDecodeGraphV1C06SignatureBinding;
use super::graph_registry::GraphReplaySlot;
use super::graph_registry_dispatch::GraphRegistryDispatchDecision;

/// Cold owner of one exact C07 full-graph identity and one C05 executable.
///
/// The stored signature is intentionally the complete C06 key, rather than a
/// fingerprint or a bucket-only value. A replay slot is publicly constructible
/// and can be reused by a later immutable registry snapshot, so slot equality
/// alone is not authority to borrow this owner. This value is private to the
/// CUDA C07 owner that prepared it; it is not a process-wide resolver.
#[must_use]
pub(crate) struct PureDecodeGraphV1C05OwnedReplaySlot {
    signature: GraphSignature,
    replay_slot: GraphReplaySlot,
    exec: OwnedGraphExec,
}

impl PureDecodeGraphV1C05OwnedReplaySlot {
    /// Binds one already-composed C07 full-graph identity to its C05 owner.
    ///
    /// The caller must cold-prepare the C05 executable in the same runtime
    /// owner that will later supply the C07 registry-selection binding. This
    /// constructor neither validates native graph semantics nor publishes a
    /// C06 registry entry; it records only the exact immutable identity used
    /// for a later fail-closed borrow.
    #[must_use]
    pub(crate) const fn new(
        signature_binding: PureDecodeGraphV1C06SignatureBinding,
        replay_slot: GraphReplaySlot,
        exec: OwnedGraphExec,
    ) -> Self {
        Self {
            signature: signature_binding.signature(),
            replay_slot,
            exec,
        }
    }

    /// Returns this private owner's complete immutable graph identity.
    #[must_use]
    pub(crate) const fn signature(&self) -> GraphSignature {
        self.signature
    }

    /// Returns this private owner's non-address logical replay slot.
    #[must_use]
    pub(crate) const fn replay_slot(&self) -> GraphReplaySlot {
        self.replay_slot
    }

    /// Borrows the C05 executable only for an exact C07 full-graph selection.
    ///
    /// Exact eager decisions pass through unchanged because they select no
    /// graph resource. A C07 pure-decode owner rejects a selected piecewise
    /// graph rather than attaching its full-graph C05 owner to a different
    /// mode. A selected full graph must match the complete C06 signature and
    /// logical slot before the executable can be borrowed.
    pub(crate) fn resolve(
        &mut self,
        selection: PureDecodeGraphV1C06RegistryDispatchBinding,
    ) -> Result<PureDecodeGraphV1C05ReplayResolution<'_>, PureDecodeGraphV1C05ReplayResolveError>
    {
        match validate_selected_full_graph(
            self.signature,
            self.replay_slot,
            selection.signature_binding().signature(),
            selection.decision(),
        )? {
            PureDecodeGraphV1C05ReplaySelection::ExactEager(reason) => {
                Ok(PureDecodeGraphV1C05ReplayResolution::ExactEager { reason })
            }
            PureDecodeGraphV1C05ReplaySelection::FullGraph => Ok(
                PureDecodeGraphV1C05ReplayResolution::FullGraph(PureDecodeGraphV1C05ResolvedExec {
                    replay_slot: self.replay_slot,
                    exec: &mut self.exec,
                }),
            ),
        }
    }

    /// Closes the retained C05 executable without retry or fallback policy.
    ///
    /// C05 returns the stream/buffer pair only after its native close proves
    /// release. Any close error keeps C05's existing fail-closed ownership
    /// contract; this C07 association neither recovers nor replaces it.
    pub(crate) fn close(self) -> CudaResult<OwnedGraphFillResources> {
        self.exec.close()
    }
}

/// Closed outcome of resolving one C07 selection against one private owner.
#[must_use]
pub(crate) enum PureDecodeGraphV1C05ReplayResolution<'owner> {
    /// C06 selected exact eager work, so no graph owner was borrowed.
    ExactEager {
        /// The unchanged C06 fallback explanation.
        reason: GraphFallbackReason,
    },
    /// One exact C07 full-graph selection borrowed the matching C05 owner.
    FullGraph(PureDecodeGraphV1C05ResolvedExec<'owner>),
}

/// Exclusive borrow of the C05 executable resolved from an exact C07 choice.
///
/// This type intentionally exposes no production launch or close method.
/// Holding it keeps the enclosing C07 owner mutably borrowed, so Rust blocks
/// a second resolution or owner close until the borrow ends. The test-only
/// accessor below exercises C05 replay lifecycle parity without claiming the
/// fixed-fill graph executes a C07 decode graph.
#[must_use]
pub(crate) struct PureDecodeGraphV1C05ResolvedExec<'owner> {
    replay_slot: GraphReplaySlot,
    exec: &'owner mut OwnedGraphExec,
}

impl PureDecodeGraphV1C05ResolvedExec<'_> {
    /// Returns the exact logical slot that selected this exclusive borrow.
    #[must_use]
    pub(crate) const fn replay_slot(&self) -> GraphReplaySlot {
        self.replay_slot
    }

    /// Exposes C05 replay only to the isolated CUDA lifecycle regression.
    #[cfg(test)]
    pub(crate) fn exec_for_gpu_test(&mut self) -> &mut OwnedGraphExec {
        self.exec
    }
}

/// Typed reason one C07 decision cannot borrow this private full-graph owner.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub(crate) enum PureDecodeGraphV1C05ReplayResolveError {
    /// A selected full graph named a different complete immutable identity.
    SignatureMismatch {
        /// Complete identity carried by the C07 selection binding.
        selected: GraphSignature,
        /// Complete identity retained by this private C05 owner.
        owned: GraphSignature,
    },
    /// A selected full graph used another logical replay slot.
    SlotMismatch {
        /// Slot carried by the C07 selection decision.
        selected: GraphReplaySlot,
        /// Slot retained by this private C05 owner.
        owned: GraphReplaySlot,
    },
    /// C07 V1 owns only a full graph, never a piecewise graph resource.
    PiecewiseGraphSelected {
        /// The incompatible selected logical replay slot.
        replay_slot: GraphReplaySlot,
    },
}

impl fmt::Display for PureDecodeGraphV1C05ReplayResolveError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SignatureMismatch { .. } => formatter.write_str(
                "C07 full-graph selection identity does not match the private C05 graph owner",
            ),
            Self::SlotMismatch { selected, owned } => write!(
                formatter,
                "C07 full-graph selection slot {} does not match private C05 owner slot {}",
                selected.value(),
                owned.value(),
            ),
            Self::PiecewiseGraphSelected { replay_slot } => write!(
                formatter,
                "C07 V1 cannot resolve piecewise graph slot {} through a full-graph C05 owner",
                replay_slot.value(),
            ),
        }
    }
}

impl error::Error for PureDecodeGraphV1C05ReplayResolveError {}

#[derive(Debug)]
enum PureDecodeGraphV1C05ReplaySelection {
    ExactEager(GraphFallbackReason),
    FullGraph,
}

/// Validates the metadata-only half of a private C05 owner resolution.
///
/// The helper is separate so unit tests can cover C06 decision ordering
/// without constructing a native graph. It must remain the only place that
/// interprets a selected C06 decision in this C07 boundary.
fn validate_selected_full_graph(
    owned_signature: GraphSignature,
    owned_slot: GraphReplaySlot,
    selected_signature: GraphSignature,
    decision: GraphRegistryDispatchDecision,
) -> Result<PureDecodeGraphV1C05ReplaySelection, PureDecodeGraphV1C05ReplayResolveError> {
    match decision {
        GraphRegistryDispatchDecision::ExactEager { reason } => {
            Ok(PureDecodeGraphV1C05ReplaySelection::ExactEager(reason))
        }
        GraphRegistryDispatchDecision::PiecewiseGraph { replay_slot } => {
            Err(PureDecodeGraphV1C05ReplayResolveError::PiecewiseGraphSelected { replay_slot })
        }
        GraphRegistryDispatchDecision::FullGraph { replay_slot } => {
            if selected_signature != owned_signature {
                return Err(PureDecodeGraphV1C05ReplayResolveError::SignatureMismatch {
                    selected: selected_signature,
                    owned: owned_signature,
                });
            }
            if replay_slot != owned_slot {
                return Err(PureDecodeGraphV1C05ReplayResolveError::SlotMismatch {
                    selected: replay_slot,
                    owned: owned_slot,
                });
            }
            Ok(PureDecodeGraphV1C05ReplaySelection::FullGraph)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use riley_cuda::{CudaContext, CudaGraphCaptureMode, CudaRuntime};

    use super::{
        PureDecodeGraphV1C05OwnedReplaySlot, PureDecodeGraphV1C05ReplayResolution,
        PureDecodeGraphV1C05ReplayResolveError, PureDecodeGraphV1C05ReplaySelection,
        validate_selected_full_graph,
    };
    use crate::llama::graph::{
        ExecutionGraphPolicy, GraphCaptureSafety, GraphComputeType, GraphDataType,
        GraphDeviceSignature, GraphDispatchEligibility, GraphDispatchRequest, GraphFallbackReason,
        GraphGemmPlanSetId, GraphGeometrySignature, GraphImplementationId,
        GraphImplementationSignature, GraphInventoryState, GraphLayoutSignature,
        GraphMetadataLayoutSignature, GraphModelArchitecture, GraphModelSignature,
        GraphOperatorCapability, GraphReductionPolicyId, GraphRevisionFingerprint,
        GraphSamplingBackend, GraphStaticSignature, GraphTensorSignature, GraphWorkloadStage,
    };
    use crate::llama::graph_decode_binding::PureDecodeGraphMetadataBinding;
    use crate::llama::graph_decode_c06_identity::bind_pure_decode_graph_v1_c06_identity;
    use crate::llama::graph_decode_c06_registry_dispatch::{
        PureDecodeGraphV1C06RegistryDispatch, PureDecodeGraphV1C06RegistryDispatchBinding,
        select_pure_decode_graph_v1_c06_registry_dispatch,
    };
    use crate::llama::graph_decode_c06_signature::{
        PureDecodeGraphV1C06Signature, PureDecodeGraphV1C06SignatureBinding,
        compose_pure_decode_graph_v1_c06_signature,
    };
    use crate::llama::graph_decode_layout::{
        PureDecodeGraphMetadataLayout, PureDecodeGraphMetadataLayoutSpec,
    };
    use crate::llama::graph_decode_layout_signature::pure_decode_graph_v1_metadata_layout_signature;
    use crate::llama::graph_decode_padding::plan_pure_decode_graph_padding;
    use crate::llama::graph_decode_preflight_binding::PureDecodeGraphV1LayoutBinding;
    use crate::llama::graph_registry::{
        GraphEntryFootprint, GraphRegistry, GraphRegistryEntry, GraphRegistryEntryState,
        GraphRegistryLimits, GraphReplayMode, GraphReplaySlot,
    };
    use crate::llama::graph_registry_dispatch::GraphRegistryDispatchDecision;

    type TestResult<T = ()> = Result<T, Box<dyn Error>>;

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

    fn signature_binding(revision: u8) -> PureDecodeGraphV1C06SignatureBinding {
        let layout = PureDecodeGraphMetadataLayout::try_new(
            PureDecodeGraphMetadataLayoutSpec::new(1, 1, 1, 1),
        )
        .expect("M1 C07 layout must be representable");
        let padding = plan_pure_decode_graph_padding(1).expect("M1 must select M1");
        let layout_binding = PureDecodeGraphV1LayoutBinding::Bound(
            PureDecodeGraphMetadataBinding::try_new(layout, padding)
                .expect("matching M1 layout and padding must bind"),
        );
        let identity = bind_pure_decode_graph_v1_c06_identity(
            &layout_binding,
            GraphSamplingBackend::GpuGreedy,
        );
        let composed = compose_pure_decode_graph_v1_c06_signature(
            &identity,
            static_signature(
                pure_decode_graph_v1_metadata_layout_signature(layout),
                revision,
            ),
        )
        .expect("matching C07/C06 metadata identities must compose");
        match composed {
            PureDecodeGraphV1C06Signature::Bound(binding) => binding,
            PureDecodeGraphV1C06Signature::Ineligible(reason) => {
                panic!("M1 fixture unexpectedly ineligible: {reason:?}")
            }
        }
    }

    fn request() -> GraphDispatchRequest {
        GraphDispatchRequest::new(
            ExecutionGraphPolicy::Auto,
            GraphDispatchEligibility::new(
                GraphWorkloadStage::PureDecode,
                true,
                true,
                true,
                GraphCaptureSafety::new(
                    GraphSamplingBackend::GpuGreedy,
                    GraphOperatorCapability::Supported,
                    true,
                ),
            ),
            GraphInventoryState::NotPrepared,
        )
    }

    fn selected_binding(
        signature_binding: PureDecodeGraphV1C06SignatureBinding,
        replay_slot: GraphReplaySlot,
    ) -> PureDecodeGraphV1C06RegistryDispatchBinding {
        let registry = GraphRegistry::<1>::try_new(
            GraphRegistryLimits::new(1, 1, 0, 16, 16),
            &[GraphRegistryEntry::new(
                signature_binding.signature(),
                GraphReplayMode::FullGraph,
                replay_slot,
                GraphRegistryEntryState::Prepared,
                GraphEntryFootprint::new(1, 1),
            )],
        )
        .expect("one full C07 graph entry must fit");
        match select_pure_decode_graph_v1_c06_registry_dispatch(
            &PureDecodeGraphV1C06Signature::Bound(signature_binding),
            request(),
            &registry,
        )
        .expect("prepared matching C07 registry entry must select")
        {
            PureDecodeGraphV1C06RegistryDispatch::Bound(binding) => binding,
            PureDecodeGraphV1C06RegistryDispatch::Ineligible(reason) => {
                panic!("M1 fixture unexpectedly ineligible during selection: {reason:?}")
            }
        }
    }

    #[test]
    fn exact_eager_preserves_its_reason_without_owner_identity_lookup() {
        let owned = signature_binding(1).signature();
        let selected = signature_binding(2).signature();
        let result = validate_selected_full_graph(
            owned,
            GraphReplaySlot::new(17),
            selected,
            GraphRegistryDispatchDecision::ExactEager {
                reason: GraphFallbackReason::BackendNotCaptureSafe,
            },
        )
        .expect("exact eager does not resolve an owner");

        assert!(matches!(
            result,
            PureDecodeGraphV1C05ReplaySelection::ExactEager(
                GraphFallbackReason::BackendNotCaptureSafe
            )
        ));
    }

    #[test]
    fn full_graph_requires_complete_signature_equality_before_slot_equality() {
        let owned = signature_binding(1).signature();
        let selected = signature_binding(2).signature();
        let error = validate_selected_full_graph(
            owned,
            GraphReplaySlot::new(17),
            selected,
            GraphRegistryDispatchDecision::FullGraph {
                replay_slot: GraphReplaySlot::new(17),
            },
        )
        .expect_err("a stale full-graph identity must not borrow the same logical slot");

        assert_eq!(
            error,
            PureDecodeGraphV1C05ReplayResolveError::SignatureMismatch { selected, owned },
        );
    }

    #[test]
    fn matching_full_graph_identity_rejects_a_different_logical_slot() {
        let signature = signature_binding(1).signature();
        let error = validate_selected_full_graph(
            signature,
            GraphReplaySlot::new(17),
            signature,
            GraphRegistryDispatchDecision::FullGraph {
                replay_slot: GraphReplaySlot::new(18),
            },
        )
        .expect_err("a different logical slot must not borrow this owner");

        assert_eq!(
            error,
            PureDecodeGraphV1C05ReplayResolveError::SlotMismatch {
                selected: GraphReplaySlot::new(18),
                owned: GraphReplaySlot::new(17),
            },
        );
    }

    #[test]
    fn selected_piecewise_graph_never_borrows_a_c07_full_graph_owner() {
        let signature = signature_binding(1).signature();
        let error = validate_selected_full_graph(
            signature,
            GraphReplaySlot::new(17),
            signature,
            GraphRegistryDispatchDecision::PiecewiseGraph {
                replay_slot: GraphReplaySlot::new(17),
            },
        )
        .expect_err("C07 V1 full owner must reject piecewise selection");

        assert_eq!(
            error,
            PureDecodeGraphV1C05ReplayResolveError::PiecewiseGraphSelected {
                replay_slot: GraphReplaySlot::new(17),
            },
        );
    }

    fn first_context() -> TestResult<CudaContext> {
        let runtime = CudaRuntime::initialize()?;
        assert!(runtime.device_count() > 0, "remote runner has no CUDA GPU");
        let context = runtime.device(0)?.create_context()?;
        assert!(context.allocation_stats()?.is_zero());
        Ok(context)
    }

    #[test]
    #[ignore = "requires a remote CUDA GPU"]
    fn c07_24_resolves_one_exact_c05_owner_without_c07_execution_wiring() -> TestResult {
        const ELEMENT_COUNT: u64 = 128;
        const FINAL_VALUE: f32 = -3.25;

        let context = first_context()?;
        let allocation_baseline = context.allocation_stats()?;
        let capture_stream = context.create_stream()?;
        let byte_len = ELEMENT_COUNT
            .checked_mul(u64::try_from(std::mem::size_of::<f32>())?)
            .ok_or("C07-24 fixture byte length overflow")?;
        let buffer = context.allocate_device_buffer(byte_len)?;
        let mut capture = capture_stream.begin_owned_graph_fill_capture(
            buffer,
            ELEMENT_COUNT,
            CudaGraphCaptureMode::ThreadLocal,
        )?;
        capture.enqueue_fill(FINAL_VALUE)?;
        let exec = capture.end()?.instantiate()?;
        let signature_binding = signature_binding(1);
        let matching_selection = selected_binding(signature_binding, GraphReplaySlot::new(17));
        let mismatched_selection = selected_binding(signature_binding, GraphReplaySlot::new(18));
        let mut owner = PureDecodeGraphV1C05OwnedReplaySlot::new(
            signature_binding,
            GraphReplaySlot::new(17),
            exec,
        );

        assert_eq!(owner.signature(), signature_binding.signature());
        assert_eq!(owner.replay_slot(), GraphReplaySlot::new(17));
        let mismatch = match owner.resolve(mismatched_selection) {
            Ok(_) => panic!("a mismatched slot unexpectedly borrowed the retained owner"),
            Err(error) => error,
        };
        assert_eq!(
            mismatch,
            PureDecodeGraphV1C05ReplayResolveError::SlotMismatch {
                selected: GraphReplaySlot::new(18),
                owned: GraphReplaySlot::new(17),
            },
            "a rejected selection must leave the retained owner usable",
        );
        {
            let mut resolved = match owner.resolve(matching_selection)? {
                PureDecodeGraphV1C05ReplayResolution::FullGraph(resolved) => resolved,
                PureDecodeGraphV1C05ReplayResolution::ExactEager { reason } => {
                    return Err(
                        format!("matching prepared full graph fell back: {reason:?}").into(),
                    );
                }
            };
            assert_eq!(resolved.replay_slot(), GraphReplaySlot::new(17));
            resolved.exec_for_gpu_test().launch()?.finish()?;
        }

        let resources = owner.close()?;
        resources.close()?;
        assert_eq!(context.allocation_stats()?, allocation_baseline);
        context.close()?;
        println!(
            "c07-24-owned-exec-resolution slot=17 elements={ELEMENT_COUNT} value={FINAL_VALUE} status=passed"
        );
        Ok(())
    }
}
