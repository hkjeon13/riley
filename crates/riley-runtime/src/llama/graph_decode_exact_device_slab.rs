//! Cold-owned device storage and geometry binding for one exact V1 C07 slab.
//!
//! C07-15 cold-prepares one opaque device allocation and binds it only to a
//! successful C07-14 pinned lease with identical complete layout geometry. The
//! resulting binding exposes neither bytes nor copy submission; transfer,
//! completion, graph capture, and execution remain outside this boundary.

use std::error;
use std::fmt;

use riley_cuda::{CudaContext, CudaDeviceBuffer, CudaPinnedHostBuffer, CudaResult};

use super::graph_decode_exact_pinned_host_slab::PureDecodeGraphV1ExactPinnedHostSlabLease;
use super::graph_decode_layout::{
    PureDecodeGraphMetadataGeometryDigest, PureDecodeGraphMetadataLayout,
};

/// Checks complete C07 metadata-layout equality for two cold slab owners.
///
/// The layout is the authority; a digest remains diagnostic/provenance data
/// and must never be used as a substitute for this complete comparison.
#[must_use]
pub(crate) fn pure_decode_graph_v1_exact_metadata_layouts_match(
    expected: PureDecodeGraphMetadataLayout,
    actual: PureDecodeGraphMetadataLayout,
) -> bool {
    expected == actual
}

/// Closed failure while binding one pinned exact V1 slab to device storage.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PureDecodeGraphV1ExactPinnedDeviceSlabBindingError {
    /// The successful pinned lease belongs to a different cold layout.
    LayoutMismatch {
        /// Geometry identity retained by this device cold owner.
        device_geometry_digest: PureDecodeGraphMetadataGeometryDigest,
        /// Geometry identity carried by the successful pinned lease.
        pinned_geometry_digest: PureDecodeGraphMetadataGeometryDigest,
    },
}

impl fmt::Display for PureDecodeGraphV1ExactPinnedDeviceSlabBindingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LayoutMismatch {
                device_geometry_digest,
                pinned_geometry_digest,
            } => write!(
                formatter,
                "exact C07 device-slab geometry {device_geometry_digest:?} does not match successful pinned geometry {pinned_geometry_digest:?}"
            ),
        }
    }
}

impl error::Error for PureDecodeGraphV1ExactPinnedDeviceSlabBindingError {}

/// Result of binding one pinned exact V1 slab to device storage.
pub(crate) type PureDecodeGraphV1ExactPinnedDeviceSlabBindingResult<T> =
    Result<T, PureDecodeGraphV1ExactPinnedDeviceSlabBindingError>;

/// Read-only geometry binding between one pinned lease and one device owner.
///
/// The binding retains both owners' borrows with independent lifetimes. Its
/// device borrow originates from `&mut` ownership, so Rust prevents rebind,
/// owner move, and explicit close while it is live. Its pinned reference keeps
/// the originating C07-14 owner borrowed as well. The binding proves layout
/// equality only; it does not submit a copy, prove device contents or transfer
/// completion, or establish graph readiness or execution.
#[must_use]
pub(crate) struct PureDecodeGraphV1ExactPinnedDeviceSlabBinding<'device, 'pinned> {
    layout: &'device PureDecodeGraphMetadataLayout,
    geometry_digest: &'device PureDecodeGraphMetadataGeometryDigest,
    device: &'device CudaDeviceBuffer,
    pinned: &'pinned CudaPinnedHostBuffer,
}

impl<'device, 'pinned> PureDecodeGraphV1ExactPinnedDeviceSlabBinding<'device, 'pinned> {
    /// Returns the exact cold layout shared by both borrowed owners.
    #[must_use]
    pub(crate) const fn layout(&self) -> PureDecodeGraphMetadataLayout {
        *self.layout
    }

    /// Returns the shared cold geometry identity.
    #[must_use]
    pub(crate) const fn geometry_digest(&self) -> PureDecodeGraphMetadataGeometryDigest {
        *self.geometry_digest
    }

    /// Returns the opaque device allocation owned by this exact layout.
    #[must_use]
    pub(crate) const fn device_buffer(&self) -> &'device CudaDeviceBuffer {
        self.device
    }

    /// Returns the successful pinned storage bound to this exact layout.
    #[must_use]
    pub(crate) const fn pinned_host_buffer(&self) -> &'pinned CudaPinnedHostBuffer {
        self.pinned
    }
}

/// One cold-owned opaque device allocation for an exact V1 C07 metadata slab.
///
/// Preparation allocates exactly `layout.total_bytes()` device bytes once. The
/// private allocation exposes no bytes or raw address; a future transfer owner
/// must consume the typed pinned/device binding explicitly.
#[must_use]
pub(crate) struct PureDecodeGraphV1ExactDeviceSlab {
    layout: PureDecodeGraphMetadataLayout,
    geometry_digest: PureDecodeGraphMetadataGeometryDigest,
    device: CudaDeviceBuffer,
}

impl PureDecodeGraphV1ExactDeviceSlab {
    /// Cold-prepares one opaque device allocation with this layout's byte length.
    ///
    /// This performs the sole device allocation for this owner. Binding a
    /// successful pinned lease performs no device allocation or copy.
    pub(crate) fn prepare(
        context: &CudaContext,
        layout: PureDecodeGraphMetadataLayout,
    ) -> CudaResult<Self> {
        let device = context.allocate_device_buffer(layout.total_bytes())?;
        Ok(Self {
            layout,
            geometry_digest: layout.geometry_digest(),
            device,
        })
    }

    /// Returns the immutable exact cold layout owned by this device allocation.
    #[must_use]
    pub(crate) const fn layout(&self) -> PureDecodeGraphMetadataLayout {
        self.layout
    }

    /// Returns the cold geometry identity retained with this device allocation.
    #[must_use]
    pub(crate) const fn geometry_digest(&self) -> PureDecodeGraphMetadataGeometryDigest {
        self.geometry_digest
    }

    /// Returns the fixed payload length of this cold device allocation.
    #[must_use]
    pub(crate) const fn payload_byte_len(&self) -> u64 {
        self.device.byte_len()
    }

    /// Binds one successful pinned lease to this same-layout device allocation.
    ///
    /// Admission compares complete layouts before creating the borrowed binding.
    /// A mismatch has no native side effect and reports both diagnostic digests.
    /// A matching result has no copy or other native operation; it carries only
    /// opaque pinned and device buffers with their common cold geometry.
    pub(crate) fn bind_pinned_host_lease<'device, 'pinned>(
        &'device mut self,
        source: PureDecodeGraphV1ExactPinnedHostSlabLease<'pinned>,
    ) -> PureDecodeGraphV1ExactPinnedDeviceSlabBindingResult<
        PureDecodeGraphV1ExactPinnedDeviceSlabBinding<'device, 'pinned>,
    > {
        if !pure_decode_graph_v1_exact_metadata_layouts_match(self.layout, source.layout()) {
            return Err(
                PureDecodeGraphV1ExactPinnedDeviceSlabBindingError::LayoutMismatch {
                    device_geometry_digest: self.geometry_digest,
                    pinned_geometry_digest: source.geometry_digest(),
                },
            );
        }
        Ok(PureDecodeGraphV1ExactPinnedDeviceSlabBinding {
            layout: &self.layout,
            geometry_digest: &self.geometry_digest,
            device: &self.device,
            pinned: source.pinned_host_buffer(),
        })
    }

    /// Explicitly frees this device allocation after all bindings have ended.
    pub(crate) fn close(self) -> CudaResult<()> {
        self.device.close()
    }

    /// Moves this exact device allocation into C05's by-value H2D owner.
    ///
    /// This narrow transfer is private to the C07 metadata graph-preparation
    /// boundary. It does not expose an address, bytes, command submission, or
    /// execution authority.
    pub(crate) fn into_c05_owned_graph_h2d_destination(self) -> CudaDeviceBuffer {
        self.device
    }

    /// Rewraps a device allocation recovered from a known C05 H2D graph close.
    ///
    /// The input must originate from this owner's corresponding
    /// [`Self::into_c05_owned_graph_h2d_destination`] call and may be supplied
    /// only after C05 has returned it following known native resource release.
    /// It is not a general raw-buffer constructor.
    pub(crate) fn recover_from_c05_owned_graph_h2d_destination(
        layout: PureDecodeGraphMetadataLayout,
        device: CudaDeviceBuffer,
    ) -> Self {
        Self {
            layout,
            geometry_digest: layout.geometry_digest(),
            device,
        }
    }

    /// Exposes the owned device buffer only to the internal CUDA parity test.
    ///
    /// Production C07 code must continue to reach the allocation through an
    /// exact pinned/device binding or a completed device-fresh lease.  The
    /// test probe exists solely to perform an independent D2H byte comparison
    /// after that lease has ended.
    #[cfg(test)]
    pub(crate) fn device_buffer_for_gpu_test(&mut self) -> &mut CudaDeviceBuffer {
        &mut self.device
    }
}
