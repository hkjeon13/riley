//! Private C07 cold preparation of one provenance-bound C05 H2D graph owner.
//!
//! C07-25 verifies a selected complete signature and logical replay slot
//! against an already-instantiated C05 owner. This C07-26 boundary establishes
//! the missing cold-resource provenance before that association: one exact V1
//! host-slab lease is staged into the same exact pinned slab that is moved with
//! the same exact device slab through one C05 capture/instantiate transition.
//!
//! It deliberately creates no production replay path or decode execution. The
//! resulting owner retains only cold layout provenance and C07-25's exclusive
//! resolution boundary. Dynamic input staging and real Llama graph semantics
//! remain later work.

use std::error;
use std::fmt;

use riley_cuda::{CudaError, CudaGraphCaptureMode, CudaResult, CudaStream, OwnedGraphH2DResources};

use super::graph::GraphSignature;
use super::graph_decode_c05_owned_h2d_exec_resolver::{
    PureDecodeGraphV1C05H2DReplayResolution, PureDecodeGraphV1C05H2DReplayResolveError,
    PureDecodeGraphV1C05OwnedH2DReplaySlot,
};
use super::graph_decode_c06_registry_dispatch::PureDecodeGraphV1C06RegistryDispatchBinding;
use super::graph_decode_c06_signature::PureDecodeGraphV1C06SignatureBinding;
use super::graph_decode_exact_device_slab::{
    PureDecodeGraphV1ExactDeviceSlab, pure_decode_graph_v1_exact_metadata_layouts_match,
};
use super::graph_decode_exact_host_slab::PureDecodeGraphV1ExactHostSlabLease;
use super::graph_decode_exact_pinned_host_slab::{
    PureDecodeGraphV1ExactPinnedHostSlab, PureDecodeGraphV1ExactPinnedHostSlabStageError,
};
use super::graph_decode_layout::{
    PureDecodeGraphMetadataGeometryDigest, PureDecodeGraphMetadataLayout,
};
use super::graph_registry::GraphReplaySlot;

/// One C07 metadata resource checked against the signature-bound cold layout.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub(crate) enum PureDecodeGraphV1C05H2DMetadataResource {
    /// The successful exact host-slab lease being staged.
    HostSlab,
    /// The C07-owned pinned H2D source allocation.
    PinnedHostSlab,
    /// The C07-owned fixed device destination allocation.
    DeviceSlab,
}

impl fmt::Display for PureDecodeGraphV1C05H2DMetadataResource {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::HostSlab => formatter.write_str("host slab"),
            Self::PinnedHostSlab => formatter.write_str("pinned host slab"),
            Self::DeviceSlab => formatter.write_str("device slab"),
        }
    }
}

/// Recoverable or terminal reason C07 cannot prepare its C05 H2D owner.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub(crate) enum PureDecodeGraphV1C05H2DMetadataOwnerPrepareErrorKind {
    /// A resource named a different complete C07 cold layout.
    LayoutMismatch {
        /// Resource rejected before any capture work.
        resource: PureDecodeGraphV1C05H2DMetadataResource,
        /// Geometry retained by the authoritative C06 signature binding.
        expected_geometry_digest: PureDecodeGraphMetadataGeometryDigest,
        /// Geometry retained by the rejected C07 resource.
        actual_geometry_digest: PureDecodeGraphMetadataGeometryDigest,
    },
    /// Equal layout values carried inconsistent geometry provenance.
    GeometryDigestMismatch {
        /// Resource rejected before any capture work.
        resource: PureDecodeGraphV1C05H2DMetadataResource,
        /// Geometry retained by the authoritative C06 signature binding.
        expected_geometry_digest: PureDecodeGraphMetadataGeometryDigest,
        /// Geometry retained by the rejected C07 resource.
        actual_geometry_digest: PureDecodeGraphMetadataGeometryDigest,
    },
    /// A retained exact resource had a length different from the cold layout.
    PayloadByteLenMismatch {
        /// Resource rejected before any capture work.
        resource: PureDecodeGraphV1C05H2DMetadataResource,
        /// Byte length derived from the authoritative complete layout.
        expected_byte_len: u64,
        /// Byte length observed through the typed resource.
        actual_byte_len: u64,
    },
    /// A host slice length cannot be represented by C05's fixed u64 ABI.
    HostPayloadByteLenNotRepresentable {
        /// Host length before conversion to C05's fixed u64 ABI.
        actual_byte_len: usize,
    },
    /// The exact synchronous host-to-pinned stage did not succeed.
    PinnedStage(PureDecodeGraphV1ExactPinnedHostSlabStageError),
    /// C05 rejected or could not enter capture.
    CaptureBegin(CudaError),
    /// C05 could not record its one fixed H2D node.
    CaptureEnqueue(CudaError),
    /// C05 could not complete capture.
    CaptureEnd(CudaError),
    /// C05 could not instantiate the captured graph.
    Instantiate(CudaError),
}

impl fmt::Display for PureDecodeGraphV1C05H2DMetadataOwnerPrepareErrorKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LayoutMismatch { resource, .. } => write!(
                formatter,
                "C07 exact {resource} layout does not match the C06 signature-bound cold layout"
            ),
            Self::GeometryDigestMismatch { resource, .. } => write!(
                formatter,
                "C07 exact {resource} geometry provenance does not match the C06 signature-bound cold layout"
            ),
            Self::PayloadByteLenMismatch {
                resource,
                expected_byte_len,
                actual_byte_len,
            } => write!(
                formatter,
                "C07 exact {resource} has {actual_byte_len} bytes, but its signature-bound cold layout requires {expected_byte_len}"
            ),
            Self::HostPayloadByteLenNotRepresentable { actual_byte_len } => write!(
                formatter,
                "C07 exact host slab length {actual_byte_len} cannot be represented by C05's fixed u64 ABI"
            ),
            Self::PinnedStage(source) => write!(
                formatter,
                "could not stage the exact C07 host slab into its pinned source: {source}"
            ),
            Self::CaptureBegin(source) => {
                write!(formatter, "could not begin C05 H2D graph capture: {source}")
            }
            Self::CaptureEnqueue(source) => {
                write!(
                    formatter,
                    "could not enqueue C05's fixed H2D graph node: {source}"
                )
            }
            Self::CaptureEnd(source) => {
                write!(formatter, "could not end C05 H2D graph capture: {source}")
            }
            Self::Instantiate(source) => {
                write!(
                    formatter,
                    "could not instantiate C05 H2D graph capture: {source}"
                )
            }
        }
    }
}

/// Failed preparation together with recoverable C07 cold resources when safe.
///
/// All C07 layout/provenance and pinned-stage failures retain the original
/// typed resources. A C05 Rust-side capture preflight failure is also
/// recoverable because C05 returns its untouched triple. After native capture
/// ownership starts, or after a later C05 transition fails, no resources are
/// reconstructed and this error is terminal.
#[must_use]
pub(crate) struct PureDecodeGraphV1C05H2DMetadataOwnerPrepareError {
    kind: PureDecodeGraphV1C05H2DMetadataOwnerPrepareErrorKind,
    resources: Option<PureDecodeGraphV1C05H2DColdResources>,
}

impl PureDecodeGraphV1C05H2DMetadataOwnerPrepareError {
    fn recoverable(
        kind: PureDecodeGraphV1C05H2DMetadataOwnerPrepareErrorKind,
        resources: PureDecodeGraphV1C05H2DColdResources,
    ) -> Self {
        Self {
            kind,
            resources: Some(resources),
        }
    }

    fn terminal(kind: PureDecodeGraphV1C05H2DMetadataOwnerPrepareErrorKind) -> Self {
        Self {
            kind,
            resources: None,
        }
    }

    /// Returns the precise failed preparation phase.
    #[must_use]
    pub(crate) const fn kind(&self) -> &PureDecodeGraphV1C05H2DMetadataOwnerPrepareErrorKind {
        &self.kind
    }

    /// Returns original C07 resources only when ownership never entered a
    /// terminal C05 native transition.
    #[must_use]
    pub(crate) fn into_resources(self) -> Option<PureDecodeGraphV1C05H2DColdResources> {
        self.resources
    }
}

impl fmt::Debug for PureDecodeGraphV1C05H2DMetadataOwnerPrepareError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PureDecodeGraphV1C05H2DMetadataOwnerPrepareError")
            .field("kind", &self.kind)
            .field("resources_recoverable", &self.resources.is_some())
            .finish()
    }
}

impl fmt::Display for PureDecodeGraphV1C05H2DMetadataOwnerPrepareError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.kind.fmt(formatter)
    }
}

impl error::Error for PureDecodeGraphV1C05H2DMetadataOwnerPrepareError {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match &self.kind {
            PureDecodeGraphV1C05H2DMetadataOwnerPrepareErrorKind::PinnedStage(source) => {
                Some(source)
            }
            PureDecodeGraphV1C05H2DMetadataOwnerPrepareErrorKind::CaptureBegin(source)
            | PureDecodeGraphV1C05H2DMetadataOwnerPrepareErrorKind::CaptureEnqueue(source)
            | PureDecodeGraphV1C05H2DMetadataOwnerPrepareErrorKind::CaptureEnd(source)
            | PureDecodeGraphV1C05H2DMetadataOwnerPrepareErrorKind::Instantiate(source) => {
                Some(source)
            }
            PureDecodeGraphV1C05H2DMetadataOwnerPrepareErrorKind::LayoutMismatch { .. }
            | PureDecodeGraphV1C05H2DMetadataOwnerPrepareErrorKind::GeometryDigestMismatch {
                ..
            }
            | PureDecodeGraphV1C05H2DMetadataOwnerPrepareErrorKind::PayloadByteLenMismatch {
                ..
            }
            | PureDecodeGraphV1C05H2DMetadataOwnerPrepareErrorKind::HostPayloadByteLenNotRepresentable {
                ..
            } => None,
        }
    }
}

/// Immutable C07 facts retained across C05 resource transitions.
///
/// Only the checked C06 signature binding constructs this value. Therefore a
/// caller cannot supply an unrelated digest, byte length, or partial identity
/// beside an otherwise valid C05 resource triple.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PureDecodeGraphV1C05H2DMetadataProvenance {
    signature_binding: PureDecodeGraphV1C06SignatureBinding,
    layout: PureDecodeGraphMetadataLayout,
    geometry_digest: PureDecodeGraphMetadataGeometryDigest,
    payload_byte_len: u64,
}

impl PureDecodeGraphV1C05H2DMetadataProvenance {
    fn from_signature_binding(signature_binding: PureDecodeGraphV1C06SignatureBinding) -> Self {
        let metadata_binding = signature_binding.identity().metadata_binding();
        let layout = metadata_binding.layout();
        let geometry_digest = metadata_binding.geometry_digest();
        debug_assert_eq!(layout.geometry_digest(), geometry_digest);
        Self {
            signature_binding,
            layout,
            geometry_digest,
            payload_byte_len: layout.total_bytes(),
        }
    }

    const fn signature_binding(self) -> PureDecodeGraphV1C06SignatureBinding {
        self.signature_binding
    }

    const fn layout(self) -> PureDecodeGraphMetadataLayout {
        self.layout
    }

    const fn geometry_digest(self) -> PureDecodeGraphMetadataGeometryDigest {
        self.geometry_digest
    }

    const fn payload_byte_len(self) -> u64 {
        self.payload_byte_len
    }
}

/// C07-owned stream and exact pinned/device slabs ready for C05 preparation.
///
/// This private owner accepts only a checked complete C06 signature binding,
/// then verifies its two C07 slab owners against that authoritative full
/// layout. It has no graph capability until a successful exact host lease is
/// synchronously staged and C05 instantiation completes.
#[must_use]
pub(crate) struct PureDecodeGraphV1C05H2DColdResources {
    provenance: PureDecodeGraphV1C05H2DMetadataProvenance,
    stream: CudaStream,
    pinned: PureDecodeGraphV1ExactPinnedHostSlab,
    device: PureDecodeGraphV1ExactDeviceSlab,
}

impl PureDecodeGraphV1C05H2DColdResources {
    /// Binds C07's typed pinned/device owners to one complete C06 signature.
    ///
    /// Layout comparison is complete and precedes every digest and byte-length
    /// check. A mismatch returns every moved resource untouched and does not
    /// enter CUDA capture or submit a transfer.
    pub(crate) fn try_new(
        signature_binding: PureDecodeGraphV1C06SignatureBinding,
        stream: CudaStream,
        pinned: PureDecodeGraphV1ExactPinnedHostSlab,
        device: PureDecodeGraphV1ExactDeviceSlab,
    ) -> Result<Self, PureDecodeGraphV1C05H2DMetadataOwnerPrepareError> {
        let resources = Self {
            provenance: PureDecodeGraphV1C05H2DMetadataProvenance::from_signature_binding(
                signature_binding,
            ),
            stream,
            pinned,
            device,
        };
        if let Err(kind) = resources.validate_retained_slab_resources() {
            return Err(
                PureDecodeGraphV1C05H2DMetadataOwnerPrepareError::recoverable(kind, resources),
            );
        }
        Ok(resources)
    }

    /// Returns the authoritative complete C06 graph identity.
    #[must_use]
    pub(crate) const fn signature(&self) -> GraphSignature {
        self.provenance.signature_binding().signature()
    }

    /// Returns the exact C07 layout retained with these cold resources.
    #[must_use]
    pub(crate) const fn layout(&self) -> PureDecodeGraphMetadataLayout {
        self.provenance.layout()
    }

    /// Returns the exact C07 cold geometry retained with these resources.
    #[must_use]
    pub(crate) const fn geometry_digest(&self) -> PureDecodeGraphMetadataGeometryDigest {
        self.provenance.geometry_digest()
    }

    /// Returns the fixed payload byte length derived from the C07 layout.
    #[must_use]
    pub(crate) const fn payload_byte_len(&self) -> u64 {
        self.provenance.payload_byte_len()
    }

    /// Stages one exact host lease and internally completes C05 capture.
    ///
    /// The capture mode is intentionally fixed to the one reviewed C05 mode.
    /// The host lease is retained only for the synchronous pinned write; it is
    /// never stored in the graph owner. No C07 execution entry point is made
    /// available by this method.
    pub(crate) fn instantiate_exact_owner(
        mut self,
        host_lease: PureDecodeGraphV1ExactHostSlabLease<'_>,
        replay_slot: GraphReplaySlot,
    ) -> Result<
        PureDecodeGraphV1C05ExactMetadataH2DOwner,
        PureDecodeGraphV1C05H2DMetadataOwnerPrepareError,
    > {
        if let Err(kind) = self.validate_retained_slab_resources() {
            return Err(PureDecodeGraphV1C05H2DMetadataOwnerPrepareError::recoverable(kind, self));
        }
        let host_payload_byte_len = match host_lease_byte_len(host_lease) {
            Ok(payload_byte_len) => payload_byte_len,
            Err(kind) => {
                return Err(
                    PureDecodeGraphV1C05H2DMetadataOwnerPrepareError::recoverable(kind, self),
                );
            }
        };
        if let Err(kind) = validate_exact_metadata_resource(
            self.provenance,
            PureDecodeGraphV1C05H2DMetadataResource::HostSlab,
            host_lease.layout(),
            host_lease.geometry_digest(),
            host_payload_byte_len,
        ) {
            return Err(PureDecodeGraphV1C05H2DMetadataOwnerPrepareError::recoverable(kind, self));
        }

        match self.pinned.stage_from_host_lease(host_lease) {
            Ok(pinned_lease) => drop(pinned_lease),
            Err(error) => {
                return Err(
                    PureDecodeGraphV1C05H2DMetadataOwnerPrepareError::recoverable(
                        PureDecodeGraphV1C05H2DMetadataOwnerPrepareErrorKind::PinnedStage(error),
                        self,
                    ),
                );
            }
        }

        let Self {
            provenance,
            stream,
            pinned,
            device,
        } = self;
        let source = pinned.into_c05_owned_graph_h2d_source();
        let destination = device.into_c05_owned_graph_h2d_destination();
        let mut capture = match stream.begin_owned_graph_h2d_capture(
            source,
            destination,
            CudaGraphCaptureMode::ThreadLocal,
        ) {
            Ok(capture) => capture,
            Err(error) => {
                let kind = PureDecodeGraphV1C05H2DMetadataOwnerPrepareErrorKind::CaptureBegin(
                    error.error().clone(),
                );
                return match error.into_resources() {
                    Some(resources) => Err(
                        PureDecodeGraphV1C05H2DMetadataOwnerPrepareError::recoverable(
                            kind,
                            Self::from_known_c05_graph_release(provenance, resources),
                        ),
                    ),
                    None => Err(PureDecodeGraphV1C05H2DMetadataOwnerPrepareError::terminal(
                        kind,
                    )),
                };
            }
        };
        if let Err(error) = capture.enqueue_h2d() {
            return Err(PureDecodeGraphV1C05H2DMetadataOwnerPrepareError::terminal(
                PureDecodeGraphV1C05H2DMetadataOwnerPrepareErrorKind::CaptureEnqueue(error),
            ));
        }
        let captured = match capture.end() {
            Ok(captured) => captured,
            Err(error) => {
                return Err(PureDecodeGraphV1C05H2DMetadataOwnerPrepareError::terminal(
                    PureDecodeGraphV1C05H2DMetadataOwnerPrepareErrorKind::CaptureEnd(error),
                ));
            }
        };
        let exec = match captured.instantiate() {
            Ok(exec) => exec,
            Err(error) => {
                return Err(PureDecodeGraphV1C05H2DMetadataOwnerPrepareError::terminal(
                    PureDecodeGraphV1C05H2DMetadataOwnerPrepareErrorKind::Instantiate(error),
                ));
            }
        };
        Ok(PureDecodeGraphV1C05ExactMetadataH2DOwner {
            provenance,
            resolver: PureDecodeGraphV1C05OwnedH2DReplaySlot::new(
                provenance.signature_binding(),
                replay_slot,
                exec,
            ),
        })
    }

    /// Closes every cold resource after it has not entered a C05 graph owner.
    pub(crate) fn close(self) -> CudaResult<()> {
        let Self {
            stream,
            pinned,
            device,
            ..
        } = self;
        device.close()?;
        pinned.close()?;
        stream.close()
    }

    fn validate_retained_slab_resources(
        &self,
    ) -> Result<(), PureDecodeGraphV1C05H2DMetadataOwnerPrepareErrorKind> {
        validate_exact_metadata_resource(
            self.provenance,
            PureDecodeGraphV1C05H2DMetadataResource::PinnedHostSlab,
            self.pinned.layout(),
            self.pinned.geometry_digest(),
            self.pinned.payload_byte_len(),
        )?;
        validate_exact_metadata_resource(
            self.provenance,
            PureDecodeGraphV1C05H2DMetadataResource::DeviceSlab,
            self.device.layout(),
            self.device.geometry_digest(),
            self.device.payload_byte_len(),
        )
    }

    fn from_known_c05_graph_release(
        provenance: PureDecodeGraphV1C05H2DMetadataProvenance,
        resources: OwnedGraphH2DResources,
    ) -> Self {
        let (stream, source, destination) = resources.into_parts();
        Self {
            provenance,
            stream,
            pinned: PureDecodeGraphV1ExactPinnedHostSlab::recover_from_c05_owned_graph_h2d_source(
                provenance.layout(),
                source,
            ),
            device: PureDecodeGraphV1ExactDeviceSlab::recover_from_c05_owned_graph_h2d_destination(
                provenance.layout(),
                destination,
            ),
        }
    }
}

/// One instantiated C05 H2D graph tied to exact C07 metadata provenance.
///
/// C07-25 remains the sole selection resolver. This wrapper adds no executable
/// access and retains the original full layout/digest facts so a successful
/// C05 close can restore the same typed cold-resource bundle.
#[must_use]
pub(crate) struct PureDecodeGraphV1C05ExactMetadataH2DOwner {
    provenance: PureDecodeGraphV1C05H2DMetadataProvenance,
    resolver: PureDecodeGraphV1C05OwnedH2DReplaySlot,
}

impl PureDecodeGraphV1C05ExactMetadataH2DOwner {
    /// Returns the complete C06 identity bound to this exact C07 owner.
    #[must_use]
    pub(crate) const fn signature(&self) -> GraphSignature {
        self.provenance.signature_binding().signature()
    }

    /// Returns the exact C07 cold layout whose slabs C05 retained.
    #[must_use]
    pub(crate) const fn layout(&self) -> PureDecodeGraphMetadataLayout {
        self.provenance.layout()
    }

    /// Returns the C07 geometry identity retained beside the C05 owner.
    #[must_use]
    pub(crate) const fn geometry_digest(&self) -> PureDecodeGraphMetadataGeometryDigest {
        self.provenance.geometry_digest()
    }

    /// Returns the C07 layout-derived fixed H2D payload byte length.
    #[must_use]
    pub(crate) const fn payload_byte_len(&self) -> u64 {
        self.provenance.payload_byte_len()
    }

    /// Returns the C06 logical replay slot bound to the C05 owner.
    #[must_use]
    pub(crate) const fn replay_slot(&self) -> GraphReplaySlot {
        self.resolver.replay_slot()
    }

    /// Applies C07-25's existing exact-signature and logical-slot resolver.
    ///
    /// A successful full-graph result only holds an exclusive owner borrow; it
    /// exposes no production execution operation at this boundary.
    pub(crate) fn resolve(
        &mut self,
        selection: PureDecodeGraphV1C06RegistryDispatchBinding,
    ) -> Result<
        PureDecodeGraphV1C05H2DReplayResolution<'_>,
        PureDecodeGraphV1C05H2DReplayResolveError,
    > {
        self.resolver.resolve(selection)
    }

    /// Closes C05 and rewraps its known-released resources with their original
    /// C07 provenance. A C05 close failure remains terminal and is not retried.
    pub(crate) fn close(self) -> CudaResult<PureDecodeGraphV1C05H2DColdResources> {
        let Self {
            provenance,
            resolver,
        } = self;
        let resources = resolver.close()?;
        Ok(
            PureDecodeGraphV1C05H2DColdResources::from_known_c05_graph_release(
                provenance, resources,
            ),
        )
    }
}

fn validate_exact_metadata_resource(
    provenance: PureDecodeGraphV1C05H2DMetadataProvenance,
    resource: PureDecodeGraphV1C05H2DMetadataResource,
    layout: PureDecodeGraphMetadataLayout,
    geometry_digest: PureDecodeGraphMetadataGeometryDigest,
    payload_byte_len: u64,
) -> Result<(), PureDecodeGraphV1C05H2DMetadataOwnerPrepareErrorKind> {
    if !pure_decode_graph_v1_exact_metadata_layouts_match(provenance.layout(), layout) {
        return Err(
            PureDecodeGraphV1C05H2DMetadataOwnerPrepareErrorKind::LayoutMismatch {
                resource,
                expected_geometry_digest: provenance.geometry_digest(),
                actual_geometry_digest: geometry_digest,
            },
        );
    }
    if provenance.geometry_digest() != geometry_digest {
        return Err(
            PureDecodeGraphV1C05H2DMetadataOwnerPrepareErrorKind::GeometryDigestMismatch {
                resource,
                expected_geometry_digest: provenance.geometry_digest(),
                actual_geometry_digest: geometry_digest,
            },
        );
    }
    if provenance.payload_byte_len() != payload_byte_len {
        return Err(
            PureDecodeGraphV1C05H2DMetadataOwnerPrepareErrorKind::PayloadByteLenMismatch {
                resource,
                expected_byte_len: provenance.payload_byte_len(),
                actual_byte_len: payload_byte_len,
            },
        );
    }
    Ok(())
}

fn host_lease_byte_len(
    host_lease: PureDecodeGraphV1ExactHostSlabLease<'_>,
) -> Result<u64, PureDecodeGraphV1C05H2DMetadataOwnerPrepareErrorKind> {
    let actual_byte_len = host_lease.bytes().len();
    u64::try_from(actual_byte_len).map_err(|_| {
        PureDecodeGraphV1C05H2DMetadataOwnerPrepareErrorKind::HostPayloadByteLenNotRepresentable {
            actual_byte_len,
        }
    })
}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use riley_cuda::{CudaContext, CudaRuntime};

    use super::{
        PureDecodeGraphV1C05ExactMetadataH2DOwner, PureDecodeGraphV1C05H2DColdResources,
        PureDecodeGraphV1C05H2DMetadataOwnerPrepareErrorKind,
        PureDecodeGraphV1C05H2DMetadataProvenance, PureDecodeGraphV1C05H2DMetadataResource,
        validate_exact_metadata_resource,
    };
    use crate::llama::batch::{
        LlamaBatchBlockTable, LlamaBatchMetadataConfig, LlamaBatchRow, LlamaBatchRowKind,
        PreparedLlamaBatchMetadata,
    };
    use crate::llama::graph::{
        GraphComputeType, GraphDataType, GraphDeviceSignature, GraphGemmPlanSetId,
        GraphGeometrySignature, GraphImplementationId, GraphImplementationSignature,
        GraphLayoutSignature, GraphMetadataLayoutSignature, GraphModelArchitecture,
        GraphModelSignature, GraphReductionPolicyId, GraphRevisionFingerprint,
        GraphSamplingBackend, GraphStaticSignature, GraphTensorSignature,
    };
    use crate::llama::graph_decode_binding::PureDecodeGraphMetadataBinding;
    use crate::llama::graph_decode_c06_identity::bind_pure_decode_graph_v1_c06_identity;
    use crate::llama::graph_decode_c06_signature::{
        PureDecodeGraphV1C06Signature, PureDecodeGraphV1C06SignatureBinding,
        compose_pure_decode_graph_v1_c06_signature,
    };
    use crate::llama::graph_decode_exact_device_slab::PureDecodeGraphV1ExactDeviceSlab;
    use crate::llama::graph_decode_exact_host_slab::{
        PureDecodeGraphV1ExactHostSlab, PureDecodeGraphV1ExactHostSlabWrite,
    };
    use crate::llama::graph_decode_exact_pinned_host_slab::PureDecodeGraphV1ExactPinnedHostSlab;
    use crate::llama::graph_decode_layout::{
        PureDecodeGraphMetadataLayout, PureDecodeGraphMetadataLayoutSpec,
    };
    use crate::llama::graph_decode_layout_signature::pure_decode_graph_v1_metadata_layout_signature;
    use crate::llama::graph_decode_padding::plan_pure_decode_graph_padding;
    use crate::llama::graph_decode_preflight_binding::PureDecodeGraphV1LayoutBinding;
    use crate::llama::graph_registry::GraphReplaySlot;
    use crate::paged_kv::BLOCK_TABLE_V1_VERSION;

    type TestResult<T = ()> = Result<T, Box<dyn Error>>;

    fn layout(
        bucket_rows: u32,
        block_entry_capacity: u64,
        header_bytes: u64,
        control_status_bytes: u64,
    ) -> PureDecodeGraphMetadataLayout {
        PureDecodeGraphMetadataLayout::try_new(PureDecodeGraphMetadataLayoutSpec::new(
            bucket_rows,
            block_entry_capacity,
            header_bytes,
            control_status_bytes,
        ))
        .expect("C07 test layout must be valid")
    }

    fn static_signature(metadata_layout: GraphMetadataLayoutSignature) -> GraphStaticSignature {
        GraphStaticSignature::new(
            GraphModelSignature::new(
                GraphModelArchitecture::LlamaDecoder,
                1,
                GraphRevisionFingerprint::from_bytes([0xC7; 32]),
                1,
            ),
            GraphDeviceSignature::new(8, 9, 12_804, 12_804, 1),
            GraphTensorSignature::new(
                GraphDataType::BFloat16,
                GraphDataType::BFloat16,
                GraphComputeType::Float32,
            ),
            GraphGeometrySignature::new(32, 4_096, 11_008, 32_000, 32, 8, 128),
            GraphLayoutSignature::new(8_192, 16, 1, metadata_layout),
            GraphImplementationSignature::new(
                GraphImplementationId::new(1),
                GraphImplementationId::new(2),
                GraphImplementationId::new(3),
                GraphImplementationId::new(4),
                GraphGemmPlanSetId::new(1),
                GraphReductionPolicyId::new(1),
            ),
        )
    }

    fn signature_binding(
        layout: PureDecodeGraphMetadataLayout,
    ) -> PureDecodeGraphV1C06SignatureBinding {
        let padding = plan_pure_decode_graph_padding(layout.bucket_rows())
            .expect("every exact C07 layout bucket must have a padding plan");
        let binding = PureDecodeGraphV1LayoutBinding::Bound(
            PureDecodeGraphMetadataBinding::try_new(layout, padding)
                .expect("matching exact C07 layout and padding must bind"),
        );
        let identity =
            bind_pure_decode_graph_v1_c06_identity(&binding, GraphSamplingBackend::GpuGreedy);
        match compose_pure_decode_graph_v1_c06_signature(
            &identity,
            static_signature(pure_decode_graph_v1_metadata_layout_signature(layout)),
        )
        .expect("matching C07 and C06 metadata identities must compose")
        {
            PureDecodeGraphV1C06Signature::Bound(binding) => binding,
            PureDecodeGraphV1C06Signature::Ineligible(reason) => {
                panic!("exact C07 test layout unexpectedly ineligible: {reason:?}")
            }
        }
    }

    #[test]
    fn c07_26_uses_complete_layout_before_digest_or_payload_diagnostics() {
        let exact = layout(1, 1, 3, 5);
        let changed_geometry = layout(1, 1, 4, 5);
        let provenance = PureDecodeGraphV1C05H2DMetadataProvenance::from_signature_binding(
            signature_binding(exact),
        );

        assert_eq!(
            validate_exact_metadata_resource(
                provenance,
                PureDecodeGraphV1C05H2DMetadataResource::PinnedHostSlab,
                changed_geometry,
                changed_geometry.geometry_digest(),
                0,
            ),
            Err(
                PureDecodeGraphV1C05H2DMetadataOwnerPrepareErrorKind::LayoutMismatch {
                    resource: PureDecodeGraphV1C05H2DMetadataResource::PinnedHostSlab,
                    expected_geometry_digest: exact.geometry_digest(),
                    actual_geometry_digest: changed_geometry.geometry_digest(),
                }
            ),
            "complete layout mismatch must precede digest or byte-length diagnostics",
        );
        assert_eq!(
            validate_exact_metadata_resource(
                provenance,
                PureDecodeGraphV1C05H2DMetadataResource::DeviceSlab,
                exact,
                changed_geometry.geometry_digest(),
                exact.total_bytes(),
            ),
            Err(
                PureDecodeGraphV1C05H2DMetadataOwnerPrepareErrorKind::GeometryDigestMismatch {
                    resource: PureDecodeGraphV1C05H2DMetadataResource::DeviceSlab,
                    expected_geometry_digest: exact.geometry_digest(),
                    actual_geometry_digest: changed_geometry.geometry_digest(),
                }
            ),
        );
        assert_eq!(
            validate_exact_metadata_resource(
                provenance,
                PureDecodeGraphV1C05H2DMetadataResource::HostSlab,
                exact,
                exact.geometry_digest(),
                exact.total_bytes() - 1,
            ),
            Err(
                PureDecodeGraphV1C05H2DMetadataOwnerPrepareErrorKind::PayloadByteLenMismatch {
                    resource: PureDecodeGraphV1C05H2DMetadataResource::HostSlab,
                    expected_byte_len: exact.total_bytes(),
                    actual_byte_len: exact.total_bytes() - 1,
                }
            ),
        );
    }

    fn first_context() -> TestResult<(CudaContext, riley_cuda::CudaStream)> {
        let runtime = CudaRuntime::initialize()?;
        assert!(runtime.device_count() > 0, "remote runner has no CUDA GPU");
        let context = runtime.device(0)?.create_context()?;
        let stream = context.create_stream()?;
        assert!(context.allocation_stats()?.is_zero());
        Ok((context, stream))
    }

    /// Captures exact C07 metadata allocations and verifies close rewraps them.
    ///
    /// This intentionally does not enter a replay or Llama execution path. It
    /// establishes only that the C07 host/pinned/device ownership chain can
    /// create and release the one-node C05 graph without allocation leakage.
    #[test]
    #[ignore = "requires a remote CUDA GPU"]
    fn c07_26_captures_exact_metadata_owner_and_recovers_cold_resources() -> TestResult {
        let (context, stream) = first_context()?;
        let layout = layout(1, 1, 3, 5);
        let signature_binding = signature_binding(layout);
        let mut prepared =
            PreparedLlamaBatchMetadata::prepare(LlamaBatchMetadataConfig::new(1, 1, 1, 1, 1)?)?;
        let token_ids = [0x0102_0304_u32];
        let physical_block_ids = [0_u32];
        let valid_tokens = [1_u16];
        let rows = [LlamaBatchRow::new(
            0xfeed_beef,
            LlamaBatchRowKind::Decode,
            &token_ids,
            1,
            LlamaBatchBlockTable::new(
                BLOCK_TABLE_V1_VERSION,
                &physical_block_ids,
                &valid_tokens,
                1,
            ),
            Some(0),
        )];
        let metadata = prepared.pack(&rows)?;
        let mut host_slab = PureDecodeGraphV1ExactHostSlab::prepare(layout)?;
        let pinned = PureDecodeGraphV1ExactPinnedHostSlab::prepare(&context, layout)?;
        let device = PureDecodeGraphV1ExactDeviceSlab::prepare(&context, layout)?;
        let stable_allocations = context.allocation_stats()?;
        let mut cold_resources = PureDecodeGraphV1C05H2DColdResources::try_new(
            signature_binding,
            stream,
            pinned,
            device,
        )
        .expect("matching exact C07 resources must bind to their complete signature");

        for iteration in 0_u8..2 {
            let header = [0xA0_u8, 0xA1, iteration];
            let control_status = [0xC0_u8, 0xC1, 0xC2, 0xC3, iteration];
            let host_lease = match host_slab
                .write_exact_v1_leased(&metadata, &header, &control_status)
                .map_err(|error| {
                    std::io::Error::other(format!("C07 exact host write failed: {error:?}"))
                })? {
                PureDecodeGraphV1ExactHostSlabWrite::Written(lease) => lease,
                PureDecodeGraphV1ExactHostSlabWrite::Ineligible(reason) => {
                    return Err(format!("strict C07 M1 fixture was ineligible: {reason:?}").into());
                }
            };
            let owner: PureDecodeGraphV1C05ExactMetadataH2DOwner = cold_resources
                .instantiate_exact_owner(host_lease, GraphReplaySlot::new(26))
                .expect("exact C07 slab provenance must capture one C05 H2D owner");
            assert_eq!(owner.signature(), signature_binding.signature());
            assert_eq!(owner.layout(), layout);
            assert_eq!(owner.geometry_digest(), layout.geometry_digest());
            assert_eq!(owner.payload_byte_len(), layout.total_bytes());
            assert_eq!(owner.replay_slot(), GraphReplaySlot::new(26));
            {
                let rewrite_lease = match host_slab
                    .write_exact_v1_leased(&metadata, &header, &control_status)
                    .map_err(|error| {
                        std::io::Error::other(format!("C07 exact host rewrite failed: {error:?}"))
                    })? {
                    PureDecodeGraphV1ExactHostSlabWrite::Written(lease) => lease,
                    PureDecodeGraphV1ExactHostSlabWrite::Ineligible(reason) => {
                        return Err(format!(
                            "strict C07 M1 rewrite fixture was ineligible: {reason:?}"
                        )
                        .into());
                    }
                };
                assert_eq!(rewrite_lease.layout(), layout);
            }
            cold_resources = owner.close()?;
            assert_eq!(cold_resources.signature(), signature_binding.signature());
            assert_eq!(cold_resources.layout(), layout);
            assert_eq!(cold_resources.geometry_digest(), layout.geometry_digest());
            assert_eq!(cold_resources.payload_byte_len(), layout.total_bytes());
            assert_eq!(
                context.allocation_stats()?,
                stable_allocations,
                "C07-26 capture/close changed cold CUDA allocation accounting on iteration {iteration}",
            );
        }

        cold_resources.close()?;
        drop(host_slab);
        assert!(context.allocation_stats()?.is_zero());
        context.close()?;
        Ok(())
    }

    /// Verifies that C05's Rust-side preflight leaves typed C07 resources
    /// recoverable before native capture ownership starts.
    #[test]
    #[ignore = "requires a remote CUDA GPU"]
    fn c07_26_recovers_exact_resources_after_c05_preflight_rejection() -> TestResult {
        let runtime = CudaRuntime::initialize()?;
        assert!(runtime.device_count() > 0, "remote runner has no CUDA GPU");
        let device = runtime.device(0)?;
        let resource_context = device.create_context()?;
        let stream_context = device.create_context()?;
        let stream = stream_context.create_stream()?;
        let layout = layout(1, 1, 3, 5);
        let signature_binding = signature_binding(layout);
        let mut prepared =
            PreparedLlamaBatchMetadata::prepare(LlamaBatchMetadataConfig::new(1, 1, 1, 1, 1)?)?;
        let token_ids = [0x0102_0304_u32];
        let physical_block_ids = [0_u32];
        let valid_tokens = [1_u16];
        let rows = [LlamaBatchRow::new(
            0xfeed_beef,
            LlamaBatchRowKind::Decode,
            &token_ids,
            1,
            LlamaBatchBlockTable::new(
                BLOCK_TABLE_V1_VERSION,
                &physical_block_ids,
                &valid_tokens,
                1,
            ),
            Some(0),
        )];
        let metadata = prepared.pack(&rows)?;
        let mut host_slab = PureDecodeGraphV1ExactHostSlab::prepare(layout)?;
        let pinned = PureDecodeGraphV1ExactPinnedHostSlab::prepare(&resource_context, layout)?;
        let device = PureDecodeGraphV1ExactDeviceSlab::prepare(&resource_context, layout)?;
        let cold_resources = PureDecodeGraphV1C05H2DColdResources::try_new(
            signature_binding,
            stream,
            pinned,
            device,
        )
        .expect("layout matching must not inspect C05 stream context yet");
        let header = [0xA0_u8, 0xA1, 0xA2];
        let control_status = [0xC0_u8, 0xC1, 0xC2, 0xC3, 0xC4];
        let host_lease = match host_slab
            .write_exact_v1_leased(&metadata, &header, &control_status)
            .map_err(|error| {
                std::io::Error::other(format!("C07 exact host write failed: {error:?}"))
            })? {
            PureDecodeGraphV1ExactHostSlabWrite::Written(lease) => lease,
            PureDecodeGraphV1ExactHostSlabWrite::Ineligible(reason) => {
                return Err(format!("strict C07 M1 fixture was ineligible: {reason:?}").into());
            }
        };
        let error = match cold_resources
            .instantiate_exact_owner(host_lease, GraphReplaySlot::new(26))
        {
            Ok(_) => {
                return Err("C05 unexpectedly captured resources from another CUDA context".into());
            }
            Err(error) => error,
        };
        assert!(
            matches!(
                error.kind(),
                PureDecodeGraphV1C05H2DMetadataOwnerPrepareErrorKind::CaptureBegin(_)
            ),
            "C05 preflight rejection must preserve its capture-begin error"
        );
        let recovered = error
            .into_resources()
            .expect("C05 Rust preflight must return the untouched typed C07 resource bundle");
        assert_eq!(recovered.signature(), signature_binding.signature());
        assert_eq!(recovered.layout(), layout);
        assert_eq!(recovered.geometry_digest(), layout.geometry_digest());
        assert_eq!(recovered.payload_byte_len(), layout.total_bytes());
        recovered.close()?;
        drop(host_slab);
        assert!(resource_context.allocation_stats()?.is_zero());
        assert!(stream_context.allocation_stats()?.is_zero());
        stream_context.close()?;
        resource_context.close()?;
        Ok(())
    }
}
