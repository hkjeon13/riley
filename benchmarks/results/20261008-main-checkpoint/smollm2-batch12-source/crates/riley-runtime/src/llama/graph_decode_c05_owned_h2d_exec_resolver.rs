//! Private C07 association between one selected full-graph identity and C05 H2D ownership.
//!
//! C06 registry selection owns only immutable metadata and exposes a logical
//! replay slot. C05-7 owns an instantiated graph containing one fixed-address
//! H2D node, together with its capture stream, pinned source, and device
//! destination. This narrow C07-25 boundary joins one exact C07 full-graph
//! selection identity to one such C05 owner without treating the one-node H2D
//! graph as a Llama decode graph.
//!
//! It deliberately performs no C06 selection or observation, no production
//! H2D replay, and no executor work. A caller receives a mutable borrow of the
//! retained owner only after full signature and logical slot equality are
//! established. A future dynamic-input/capture owner must establish the real
//! decode graph's semantic/resource contract before adding an execution path.

use std::error;
use std::fmt;

use riley_cuda::{CudaResult, OwnedGraphH2DExec, OwnedGraphH2DResources};

use super::graph::{GraphFallbackReason, GraphSignature};
use super::graph_decode_c06_registry_dispatch::PureDecodeGraphV1C06RegistryDispatchBinding;
use super::graph_decode_c06_signature::PureDecodeGraphV1C06SignatureBinding;
use super::graph_registry::GraphReplaySlot;
use super::graph_registry_dispatch::GraphRegistryDispatchDecision;

/// Cold owner of one exact C07 full-graph identity and one C05 H2D executable.
///
/// The stored signature is intentionally the complete C06 key, rather than a
/// fingerprint or bucket-only value. A replay slot is publicly constructible
/// and may be reused by a later immutable registry snapshot, so slot equality
/// alone is not authority to borrow this owner. This value stays private to the
/// CUDA C07 owner that prepared it; it is not a process-wide resolver.
#[must_use]
pub(crate) struct PureDecodeGraphV1C05OwnedH2DReplaySlot {
    signature: GraphSignature,
    replay_slot: GraphReplaySlot,
    exec: OwnedGraphH2DExec,
}

impl PureDecodeGraphV1C05OwnedH2DReplaySlot {
    /// Binds one already-composed C07 full-graph identity to its C05 H2D owner.
    ///
    /// The caller must cold-prepare the C05 executable in the same runtime
    /// owner that will later supply the C07 registry-selection binding. This
    /// constructor neither validates native graph semantics nor publishes a
    /// C06 registry entry; it records only the exact immutable identity used
    /// for a later fail-closed exclusive borrow.
    #[must_use]
    pub(crate) const fn new(
        signature_binding: PureDecodeGraphV1C06SignatureBinding,
        replay_slot: GraphReplaySlot,
        exec: OwnedGraphH2DExec,
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

    /// Borrows the C05 H2D executable only for an exact C07 full-graph
    /// selection.
    ///
    /// Exact eager decisions pass through unchanged because they select no
    /// graph resource. A C07 pure-decode owner rejects a selected piecewise
    /// graph rather than attaching its full-graph owner to another mode. A
    /// selected full graph must match the complete C06 signature and logical
    /// slot before the executable can be borrowed.
    pub(crate) fn resolve(
        &mut self,
        selection: PureDecodeGraphV1C06RegistryDispatchBinding,
    ) -> Result<
        PureDecodeGraphV1C05H2DReplayResolution<'_>,
        PureDecodeGraphV1C05H2DReplayResolveError,
    > {
        match validate_selected_h2d_full_graph(
            self.signature,
            self.replay_slot,
            selection.signature_binding().signature(),
            selection.decision(),
        )? {
            PureDecodeGraphV1C05H2DReplaySelection::ExactEager(reason) => {
                Ok(PureDecodeGraphV1C05H2DReplayResolution::ExactEager { reason })
            }
            PureDecodeGraphV1C05H2DReplaySelection::FullGraph => {
                Ok(PureDecodeGraphV1C05H2DReplayResolution::FullGraph(
                    PureDecodeGraphV1C05ResolvedH2DExec {
                        replay_slot: self.replay_slot,
                        exec: &mut self.exec,
                    },
                ))
            }
        }
    }

    /// Closes the retained C05 H2D executable without retry or fallback policy.
    ///
    /// C05 returns the stream/source/destination triple only after native close
    /// proves release. Any close error keeps C05's existing fail-closed
    /// ownership contract; this C07 association neither recovers nor replaces
    /// it.
    pub(crate) fn close(self) -> CudaResult<OwnedGraphH2DResources> {
        self.exec.close()
    }
}

/// Closed outcome of resolving one C07 selection against one private H2D owner.
#[must_use]
pub(crate) enum PureDecodeGraphV1C05H2DReplayResolution<'owner> {
    /// C06 selected exact eager work, so no graph owner was borrowed.
    ExactEager {
        /// The unchanged C06 fallback explanation.
        reason: GraphFallbackReason,
    },
    /// One exact C07 full-graph selection borrowed the matching C05 H2D owner.
    FullGraph(PureDecodeGraphV1C05ResolvedH2DExec<'owner>),
}

/// Exclusive borrow of the C05 H2D executable resolved from an exact C07 choice.
///
/// This type intentionally exposes no production replay or close method.
/// Holding it keeps the enclosing C07 owner mutably borrowed, so Rust blocks a
/// second resolution or owner close until the borrow ends. The test-only
/// accessor below exercises C05-7 lifecycle parity without claiming the
/// one-node H2D graph executes a C07 decode graph.
#[must_use]
pub(crate) struct PureDecodeGraphV1C05ResolvedH2DExec<'owner> {
    replay_slot: GraphReplaySlot,
    exec: &'owner mut OwnedGraphH2DExec,
}

impl PureDecodeGraphV1C05ResolvedH2DExec<'_> {
    /// Returns the exact logical slot that selected this exclusive borrow.
    #[must_use]
    pub(crate) const fn replay_slot(&self) -> GraphReplaySlot {
        self.replay_slot
    }

    /// Exposes C05-7 replay only to the isolated CUDA lifecycle regression.
    #[cfg(test)]
    pub(crate) fn exec_for_gpu_test(&mut self) -> &mut OwnedGraphH2DExec {
        self.exec
    }
}

/// Typed reason one C07 decision cannot borrow this private full-graph owner.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub(crate) enum PureDecodeGraphV1C05H2DReplayResolveError {
    /// A selected full graph named a different complete immutable identity.
    SignatureMismatch {
        /// Complete identity carried by the C07 selection binding.
        selected: GraphSignature,
        /// Complete identity retained by this private C05 H2D owner.
        owned: GraphSignature,
    },
    /// A selected full graph used another logical replay slot.
    SlotMismatch {
        /// Slot carried by the C07 selection decision.
        selected: GraphReplaySlot,
        /// Slot retained by this private C05 H2D owner.
        owned: GraphReplaySlot,
    },
    /// C07 V1 owns only a full graph, never a piecewise graph resource.
    PiecewiseGraphSelected {
        /// The incompatible selected logical replay slot.
        replay_slot: GraphReplaySlot,
    },
}

impl fmt::Display for PureDecodeGraphV1C05H2DReplayResolveError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SignatureMismatch { .. } => formatter.write_str(
                "C07 full-graph selection identity does not match the private C05 H2D graph owner",
            ),
            Self::SlotMismatch { selected, owned } => write!(
                formatter,
                "C07 full-graph selection slot {} does not match private C05 H2D owner slot {}",
                selected.value(),
                owned.value(),
            ),
            Self::PiecewiseGraphSelected { replay_slot } => write!(
                formatter,
                "C07 V1 cannot resolve piecewise graph slot {} through a full-graph C05 H2D owner",
                replay_slot.value(),
            ),
        }
    }
}

impl error::Error for PureDecodeGraphV1C05H2DReplayResolveError {}

#[derive(Debug)]
enum PureDecodeGraphV1C05H2DReplaySelection {
    ExactEager(GraphFallbackReason),
    FullGraph,
}

/// Validates the metadata-only half of a private C05 H2D owner resolution.
///
/// The helper is separate so unit tests can cover C06 decision ordering without
/// constructing a native graph. It must remain the only place that interprets
/// a selected C06 decision in this C07 boundary.
fn validate_selected_h2d_full_graph(
    owned_signature: GraphSignature,
    owned_slot: GraphReplaySlot,
    selected_signature: GraphSignature,
    decision: GraphRegistryDispatchDecision,
) -> Result<PureDecodeGraphV1C05H2DReplaySelection, PureDecodeGraphV1C05H2DReplayResolveError> {
    match decision {
        GraphRegistryDispatchDecision::ExactEager { reason } => {
            Ok(PureDecodeGraphV1C05H2DReplaySelection::ExactEager(reason))
        }
        GraphRegistryDispatchDecision::PiecewiseGraph { replay_slot } => {
            Err(PureDecodeGraphV1C05H2DReplayResolveError::PiecewiseGraphSelected { replay_slot })
        }
        GraphRegistryDispatchDecision::FullGraph { replay_slot } => {
            if selected_signature != owned_signature {
                return Err(
                    PureDecodeGraphV1C05H2DReplayResolveError::SignatureMismatch {
                        selected: selected_signature,
                        owned: owned_signature,
                    },
                );
            }
            if replay_slot != owned_slot {
                return Err(PureDecodeGraphV1C05H2DReplayResolveError::SlotMismatch {
                    selected: replay_slot,
                    owned: owned_slot,
                });
            }
            Ok(PureDecodeGraphV1C05H2DReplaySelection::FullGraph)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use riley_cuda::{CudaContext, CudaGraphCaptureMode, CudaRuntime};

    use super::{
        PureDecodeGraphV1C05H2DReplayResolution, PureDecodeGraphV1C05H2DReplayResolveError,
        PureDecodeGraphV1C05H2DReplaySelection, PureDecodeGraphV1C05OwnedH2DReplaySlot,
        validate_selected_h2d_full_graph,
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
    fn exact_eager_preserves_its_reason_without_h2d_owner_identity_lookup() {
        let owned = signature_binding(1).signature();
        let selected = signature_binding(2).signature();
        let result = validate_selected_h2d_full_graph(
            owned,
            GraphReplaySlot::new(17),
            selected,
            GraphRegistryDispatchDecision::ExactEager {
                reason: GraphFallbackReason::BackendNotCaptureSafe,
            },
        )
        .expect("exact eager does not resolve an H2D owner");

        assert!(matches!(
            result,
            PureDecodeGraphV1C05H2DReplaySelection::ExactEager(
                GraphFallbackReason::BackendNotCaptureSafe
            )
        ));
    }

    #[test]
    fn full_graph_requires_complete_signature_equality_before_slot_equality() {
        let owned = signature_binding(1).signature();
        let selected = signature_binding(2).signature();
        let error = validate_selected_h2d_full_graph(
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
            PureDecodeGraphV1C05H2DReplayResolveError::SignatureMismatch { selected, owned },
        );
    }

    #[test]
    fn matching_full_graph_identity_rejects_a_different_logical_slot() {
        let signature = signature_binding(1).signature();
        let error = validate_selected_h2d_full_graph(
            signature,
            GraphReplaySlot::new(17),
            signature,
            GraphRegistryDispatchDecision::FullGraph {
                replay_slot: GraphReplaySlot::new(18),
            },
        )
        .expect_err("a different logical slot must not borrow this H2D owner");

        assert_eq!(
            error,
            PureDecodeGraphV1C05H2DReplayResolveError::SlotMismatch {
                selected: GraphReplaySlot::new(18),
                owned: GraphReplaySlot::new(17),
            },
        );
    }

    #[test]
    fn selected_piecewise_graph_never_borrows_a_c07_full_h2d_owner() {
        let signature = signature_binding(1).signature();
        let error = validate_selected_h2d_full_graph(
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
            PureDecodeGraphV1C05H2DReplayResolveError::PiecewiseGraphSelected {
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

    fn h2d_payload(byte_len: usize, replay: usize) -> Vec<u8> {
        (0..byte_len)
            .map(|index| {
                let mixed = index
                    .wrapping_mul(29)
                    .wrapping_add(replay.wrapping_mul(71))
                    .wrapping_add(replay.rotate_left(3));
                (mixed & 0xff) as u8
            })
            .collect()
    }

    #[test]
    #[ignore = "requires a remote CUDA GPU"]
    fn c07_25_resolves_one_exact_c05_h2d_owner_without_c07_execution_wiring() -> TestResult {
        const REPLAYS: usize = 32;

        let context = first_context()?;
        let allocation_baseline = context.allocation_stats()?;
        let layout = PureDecodeGraphMetadataLayout::try_new(
            PureDecodeGraphMetadataLayoutSpec::new(1, 1, 1, 1),
        )?;
        let byte_len = layout.total_bytes();
        let capture_stream = context.create_stream()?;
        let source = context.allocate_pinned_host_buffer(byte_len)?;
        let destination = context.allocate_device_buffer(byte_len)?;
        let mut capture = capture_stream.begin_owned_graph_h2d_capture(
            source,
            destination,
            CudaGraphCaptureMode::ThreadLocal,
        )?;
        capture.enqueue_h2d()?;
        let exec = capture.end()?.instantiate()?;
        let owner_signature_binding = signature_binding(1);
        let mismatched_signature_binding = signature_binding(2);
        let matching_selection =
            selected_binding(owner_signature_binding, GraphReplaySlot::new(17));
        let signature_mismatched_selection =
            selected_binding(mismatched_signature_binding, GraphReplaySlot::new(17));
        let mismatched_selection =
            selected_binding(owner_signature_binding, GraphReplaySlot::new(18));
        let mut owner = PureDecodeGraphV1C05OwnedH2DReplaySlot::new(
            owner_signature_binding,
            GraphReplaySlot::new(17),
            exec,
        );

        assert_eq!(owner.signature(), owner_signature_binding.signature());
        assert_eq!(owner.replay_slot(), GraphReplaySlot::new(17));
        let signature_mismatch = match owner.resolve(signature_mismatched_selection) {
            Ok(_) => panic!("a mismatched signature unexpectedly borrowed the retained H2D owner"),
            Err(error) => error,
        };
        assert_eq!(
            signature_mismatch,
            PureDecodeGraphV1C05H2DReplayResolveError::SignatureMismatch {
                selected: mismatched_signature_binding.signature(),
                owned: owner_signature_binding.signature(),
            },
            "a rejected signature must leave the retained H2D owner usable",
        );
        let mismatch = match owner.resolve(mismatched_selection) {
            Ok(_) => panic!("a mismatched slot unexpectedly borrowed the retained H2D owner"),
            Err(error) => error,
        };
        assert_eq!(
            mismatch,
            PureDecodeGraphV1C05H2DReplayResolveError::SlotMismatch {
                selected: GraphReplaySlot::new(18),
                owned: GraphReplaySlot::new(17),
            },
            "a rejected selection must leave the retained H2D owner usable",
        );

        let byte_len = usize::try_from(byte_len)?;
        let mut expected = Vec::new();
        {
            let mut resolved = match owner.resolve(matching_selection)? {
                PureDecodeGraphV1C05H2DReplayResolution::FullGraph(resolved) => resolved,
                PureDecodeGraphV1C05H2DReplayResolution::ExactEager { reason } => {
                    return Err(
                        format!("matching prepared full graph fell back: {reason:?}").into(),
                    );
                }
            };
            assert_eq!(resolved.replay_slot(), GraphReplaySlot::new(17));
            for replay in 0..REPLAYS {
                let payload = h2d_payload(byte_len, replay);
                resolved
                    .exec_for_gpu_test()
                    .launch_with_source(&payload)?
                    .finish()?;
                expected = payload;
            }
        }

        let resources = owner.close()?;
        let (mut capture_stream, mut source, mut destination) = resources.into_parts();
        let mut actual = vec![0_u8; byte_len];
        destination.download_to_slice(0, &mut actual, &mut source, &mut capture_stream)?;
        assert_eq!(
            actual, expected,
            "last staged H2D payload must replay byte-exactly"
        );
        source.close()?;
        destination.close()?;
        capture_stream.close()?;
        assert_eq!(context.allocation_stats()?, allocation_baseline);
        context.close()?;
        println!(
            "c07-25-owned-h2d-exec-resolution slot=17 replays={REPLAYS} bytes={byte_len} status=passed"
        );
        Ok(())
    }
}
