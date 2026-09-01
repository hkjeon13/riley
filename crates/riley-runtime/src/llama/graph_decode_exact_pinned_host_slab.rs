//! Cold-owned pinned staging storage for one exact V1 C07 metadata slab.
//!
//! C07-14 receives a successful C07-13 host-slab lease and synchronously
//! records its exact bytes into one cold-prepared pinned allocation. It yields
//! only a read-only pinned lease; device storage, command submission, graph
//! capture, and execution remain outside this boundary.

use std::error;
use std::fmt;

use riley_cuda::{CudaContext, CudaError, CudaPinnedHostBuffer, CudaResult};

use super::graph_decode_exact_host_slab::PureDecodeGraphV1ExactHostSlabLease;
use super::graph_decode_layout::{
    PureDecodeGraphMetadataGeometryDigest, PureDecodeGraphMetadataLayout,
};

/// Closed failure while staging one successful exact V1 host-slab lease.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum PureDecodeGraphV1ExactPinnedHostSlabStageError {
    /// The successful host lease belongs to a different cold layout.
    LayoutMismatch {
        /// Geometry identity retained by this pinned cold owner.
        pinned_geometry_digest: PureDecodeGraphMetadataGeometryDigest,
        /// Geometry identity carried by the successful host lease.
        source_geometry_digest: PureDecodeGraphMetadataGeometryDigest,
    },
    /// Recording the exact host bytes into pinned storage failed.
    PinnedWrite(CudaError),
}

impl fmt::Display for PureDecodeGraphV1ExactPinnedHostSlabStageError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LayoutMismatch {
                pinned_geometry_digest,
                source_geometry_digest,
            } => write!(
                formatter,
                "exact C07 pinned host-slab geometry {pinned_geometry_digest:?} does not match successful source geometry {source_geometry_digest:?}"
            ),
            Self::PinnedWrite(source) => write!(
                formatter,
                "could not stage exact C07 host-slab bytes into pinned storage: {source}"
            ),
        }
    }
}

impl error::Error for PureDecodeGraphV1ExactPinnedHostSlabStageError {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match self {
            Self::LayoutMismatch { .. } => None,
            Self::PinnedWrite(source) => Some(source),
        }
    }
}

/// Result of staging one successful exact V1 host-slab lease.
pub(crate) type PureDecodeGraphV1ExactPinnedHostSlabStageResult<T> =
    Result<T, PureDecodeGraphV1ExactPinnedHostSlabStageError>;

/// Read-only proof of one successfully staged exact V1 pinned payload.
///
/// The lease keeps the pinned owner mutably borrowed, so Rust prevents another
/// staging write, owner move, or explicit close until the lease ends. It proves
/// only that the synchronous pinned write returned success; it does not prove
/// any later transfer, device contents, graph readiness, or execution.
#[must_use]
pub(crate) struct PureDecodeGraphV1ExactPinnedHostSlabLease<'pinned> {
    layout: &'pinned PureDecodeGraphMetadataLayout,
    geometry_digest: &'pinned PureDecodeGraphMetadataGeometryDigest,
    pinned: &'pinned CudaPinnedHostBuffer,
}

impl<'pinned> PureDecodeGraphV1ExactPinnedHostSlabLease<'pinned> {
    /// Returns the exact cold layout retained by the pinned owner.
    #[must_use]
    pub(crate) const fn layout(&self) -> PureDecodeGraphMetadataLayout {
        *self.layout
    }

    /// Returns the cold geometry identity retained by the pinned owner.
    #[must_use]
    pub(crate) const fn geometry_digest(&self) -> PureDecodeGraphMetadataGeometryDigest {
        *self.geometry_digest
    }

    /// Returns the pinned storage owned by the same cold layout.
    #[must_use]
    pub(crate) const fn pinned_host_buffer(&self) -> &'pinned CudaPinnedHostBuffer {
        self.pinned
    }
}

/// One cold-owned pinned payload for an exact V1 C07 metadata layout.
///
/// Preparation allocates exactly `layout.total_bytes()` pinned bytes once. A
/// staging write reuses those bytes without allocating. This owner does not
/// expose mutable bytes or a raw address; future transfer ownership must accept
/// the read-only lease explicitly.
#[must_use]
pub(crate) struct PureDecodeGraphV1ExactPinnedHostSlab {
    layout: PureDecodeGraphMetadataLayout,
    geometry_digest: PureDecodeGraphMetadataGeometryDigest,
    pinned: CudaPinnedHostBuffer,
}

impl PureDecodeGraphV1ExactPinnedHostSlab {
    /// Cold-prepares one pinned payload with exactly this layout's byte length.
    ///
    /// This performs the sole pinned allocation for this owner. Later staging
    /// writes only reuse this allocation.
    pub(crate) fn prepare(
        context: &CudaContext,
        layout: PureDecodeGraphMetadataLayout,
    ) -> CudaResult<Self> {
        let pinned = context.allocate_pinned_host_buffer(layout.total_bytes())?;
        Ok(Self {
            layout,
            geometry_digest: layout.geometry_digest(),
            pinned,
        })
    }

    /// Returns the immutable exact cold layout owned by this pinned payload.
    #[must_use]
    pub(crate) const fn layout(&self) -> PureDecodeGraphMetadataLayout {
        self.layout
    }

    /// Returns the cold geometry identity retained with this pinned payload.
    #[must_use]
    pub(crate) const fn geometry_digest(&self) -> PureDecodeGraphMetadataGeometryDigest {
        self.geometry_digest
    }

    /// Returns the fixed payload length of this cold pinned allocation.
    #[must_use]
    pub(crate) const fn payload_byte_len(&self) -> u64 {
        self.pinned.byte_len()
    }

    /// Synchronously records one successful exact host lease into pinned storage.
    ///
    /// Admission compares complete layouts before the native pinned write. A
    /// mismatch performs no write and reports both diagnostic digests. On a
    /// matching layout this invokes one synchronous pinned write at offset zero
    /// and returns a read-only lease only after success. A failed pinned write
    /// returns no lease and makes no transactional claim about pinned contents.
    pub(crate) fn stage_from_host_lease<'pinned>(
        &'pinned mut self,
        source: PureDecodeGraphV1ExactHostSlabLease<'_>,
    ) -> PureDecodeGraphV1ExactPinnedHostSlabStageResult<
        PureDecodeGraphV1ExactPinnedHostSlabLease<'pinned>,
    > {
        if source.layout() != self.layout {
            return Err(
                PureDecodeGraphV1ExactPinnedHostSlabStageError::LayoutMismatch {
                    pinned_geometry_digest: self.geometry_digest,
                    source_geometry_digest: source.geometry_digest(),
                },
            );
        }
        self.pinned
            .write(0, source.bytes())
            .map_err(PureDecodeGraphV1ExactPinnedHostSlabStageError::PinnedWrite)?;
        Ok(self.successful_stage_lease())
    }

    /// Explicitly frees this pinned allocation after all leases have ended.
    pub(crate) fn close(self) -> CudaResult<()> {
        self.pinned.close()
    }

    /// Moves this exact pinned allocation into C05's by-value H2D owner.
    ///
    /// This narrow transfer is private to the C07 metadata graph-preparation
    /// boundary.  It deliberately transfers no layout authority: the caller
    /// must retain and later re-establish the original exact layout only after
    /// C05 has proved graph-resource release.
    pub(crate) fn into_c05_owned_graph_h2d_source(self) -> CudaPinnedHostBuffer {
        self.pinned
    }

    /// Rewraps a pinned allocation recovered from a known C05 H2D graph close.
    ///
    /// The input must originate from this owner's corresponding
    /// [`Self::into_c05_owned_graph_h2d_source`] call and may be supplied only
    /// after C05 has returned it following known native resource release.  It
    /// is not a general raw-buffer constructor.
    pub(crate) fn recover_from_c05_owned_graph_h2d_source(
        layout: PureDecodeGraphMetadataLayout,
        pinned: CudaPinnedHostBuffer,
    ) -> Self {
        Self {
            layout,
            geometry_digest: layout.geometry_digest(),
            pinned,
        }
    }

    fn successful_stage_lease(&self) -> PureDecodeGraphV1ExactPinnedHostSlabLease<'_> {
        PureDecodeGraphV1ExactPinnedHostSlabLease {
            layout: &self.layout,
            geometry_digest: &self.geometry_digest,
            pinned: &self.pinned,
        }
    }
}
