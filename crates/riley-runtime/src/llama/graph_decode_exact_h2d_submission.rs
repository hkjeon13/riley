//! Exact V1 C07 H2D submission into one caller-owned active command batch.
//!
//! C07-16 borrows a successful C07-15 pinned/device geometry binding for the
//! command stream's lifetime and enqueues one exact full-slab transfer. It
//! neither creates nor completes the batch; completion, device freshness,
//! graph capture, and execution remain caller-owned later boundaries.

use riley_cuda::{CudaCommandStream, CudaResult};

use super::graph_decode_exact_device_slab::PureDecodeGraphV1ExactPinnedDeviceSlabBinding;

/// Enqueues one exact full-slab pinned-to-device transfer in an active batch.
///
/// The binding borrow lasts for the command stream's lifetime, keeping both
/// original owners borrowed until the caller completes its batch. This returns
/// the underlying submission result without translating it. `Ok(())` means
/// only that enqueue succeeded; it does not establish completion, device
/// contents, graph readiness, or execution.
pub(crate) fn enqueue_pure_decode_graph_v1_exact_h2d<'batch, 'stream, 'device, 'pinned>(
    binding: &'stream PureDecodeGraphV1ExactPinnedDeviceSlabBinding<'device, 'pinned>,
    commands: &mut CudaCommandStream<'batch, 'stream>,
) -> CudaResult<()>
where
    'device: 'stream,
    'pinned: 'stream,
{
    binding.device_buffer().copy_from_pinned_in_command_batch(
        0,
        binding.pinned_host_buffer(),
        0,
        binding.layout().total_bytes(),
        commands,
    )
}
