//! Completed exact V1 C07 H2D authority from one batch-owning receipt.
//!
//! C07-17 consumes one caller-created batch while it invokes C07-16's exact
//! enqueue primitive, then retains that exact batch in a private receipt. Only
//! a successful completion of the receipt-owned batch releases the pinned
//! source and yields a device-fresh lease. It neither creates a stream/batch,
//! captures a graph, or executes graph work.

use riley_cuda::{CudaCommandBatch, CudaDeviceBuffer, CudaResult};

use super::graph_decode_exact_device_slab::PureDecodeGraphV1ExactPinnedDeviceSlabBinding;
use super::graph_decode_exact_h2d_submission::enqueue_pure_decode_graph_v1_exact_h2d;
use super::graph_decode_layout::{
    PureDecodeGraphMetadataGeometryDigest, PureDecodeGraphMetadataLayout,
};

/// Unforgeable proof that one exact C07 H2D copy entered its owned batch.
///
/// The receipt owns the caller's exact active batch and retains the
/// stream-lifetime binding. It proves enqueue only, not copy completion,
/// device freshness, graph readiness, or execution.
#[must_use]
pub(crate) struct PureDecodeGraphV1ExactH2DEnqueued<'stream, 'device, 'pinned> {
    binding: &'stream PureDecodeGraphV1ExactPinnedDeviceSlabBinding<'device, 'pinned>,
    batch: CudaCommandBatch<'stream>,
}

/// Read-only proof of one completed exact V1 H2D device payload.
///
/// This lease retains only the device owner borrow and the exact copied cold
/// geometry. The originating pinned source is intentionally released after
/// completion, so it may be reused or closed. The lease proves completed exact
/// H2D only; it does not establish graph readiness or execution.
#[must_use]
pub(crate) struct PureDecodeGraphV1ExactDeviceFreshLease<'device> {
    layout: PureDecodeGraphMetadataLayout,
    geometry_digest: PureDecodeGraphMetadataGeometryDigest,
    device: &'device CudaDeviceBuffer,
}

impl<'device> PureDecodeGraphV1ExactDeviceFreshLease<'device> {
    /// Returns the exact cold layout whose payload completed H2D.
    #[must_use]
    pub(crate) const fn layout(&self) -> PureDecodeGraphMetadataLayout {
        self.layout
    }

    /// Returns the completed payload's cold geometry identity.
    #[must_use]
    pub(crate) const fn geometry_digest(&self) -> PureDecodeGraphMetadataGeometryDigest {
        self.geometry_digest
    }

    /// Returns the opaque device allocation holding the completed payload.
    #[must_use]
    pub(crate) const fn device_buffer(&self) -> &'device CudaDeviceBuffer {
        self.device
    }
}

/// Enqueues one exact H2D into an owned batch and returns its receipt.
///
/// This consumes one caller-created active batch, scopes its only command
/// proxy, and delegates the copy to C07-16. A successful receipt owns that
/// exact batch, so completion cannot be redirected to another stream or batch.
/// On an enqueue failure, the original native copy error is returned unchanged;
/// the consumed batch follows its existing best-effort drop contract and no
/// recovery handle or fresh lease is returned.
pub(crate) fn submit_pure_decode_graph_v1_exact_h2d<'stream, 'device, 'pinned>(
    binding: &'stream PureDecodeGraphV1ExactPinnedDeviceSlabBinding<'device, 'pinned>,
    mut batch: CudaCommandBatch<'stream>,
) -> CudaResult<PureDecodeGraphV1ExactH2DEnqueued<'stream, 'device, 'pinned>>
where
    'device: 'stream,
    'pinned: 'stream,
{
    {
        let mut commands = batch.commands();
        enqueue_pure_decode_graph_v1_exact_h2d(binding, &mut commands)?;
    }
    Ok(PureDecodeGraphV1ExactH2DEnqueued { binding, batch })
}

/// Completes one exact enqueued H2D and returns its device-fresh lease.
///
/// The receipt can originate only from this module's C07-16 submission
/// transaction and owns that active batch. This consumes the receipt and
/// observes completion exactly once before constructing a lease. A completion
/// failure preserves the native error and yields no lease; native resource
/// retention after an ambiguous failure is owned by the existing CUDA
/// command-batch contract.
pub(crate) fn finish_pure_decode_graph_v1_exact_h2d<'stream, 'device, 'pinned>(
    submitted: PureDecodeGraphV1ExactH2DEnqueued<'stream, 'device, 'pinned>,
) -> CudaResult<PureDecodeGraphV1ExactDeviceFreshLease<'device>>
where
    'device: 'stream,
    'pinned: 'stream,
{
    let binding = submitted.binding;
    submitted.batch.finish()?;
    Ok(PureDecodeGraphV1ExactDeviceFreshLease {
        layout: binding.layout(),
        geometry_digest: binding.geometry_digest(),
        device: binding.device_buffer(),
    })
}
