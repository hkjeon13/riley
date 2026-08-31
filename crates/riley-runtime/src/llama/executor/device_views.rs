//! Non-owning CUDA spans for prepared batch metadata inputs.
//!
//! This module binds already allocated metadata buffers to the CUDA batch
//! descriptor. It neither allocates, uploads, copies, nor owns streams.

use riley_cuda::{CudaBufferSpan, CudaDType, CudaDeviceBuffer, PackedBatchHostV1, PackedBatchV1};

use super::super::batch::LlamaPackedBatchMetadata;
use super::super::error::ExecutionSite;
use super::buffers::{PerOperationDeviceMetadata, U16_BYTES, U32_BYTES};
use super::error::{
    LlamaBatchExecutorError, LlamaBatchExecutorResource, LlamaBatchExecutorResult,
    cuda_error as batch_cuda,
};
use super::metadata::{ByteRegion, PackedIterationLayout};

/// Borrowed batch descriptor and optional input/output spans for one dispatch.
pub(crate) struct BatchDeviceViews<'a> {
    pub(crate) batch: PackedBatchV1<'a>,
    pub(crate) token_ids: Option<CudaBufferSpan<'a>>,
    pub(crate) output_token_indices: Option<CudaBufferSpan<'a>>,
}

/// Binds the separate cold-prepared metadata buffers for synchronous uploads.
pub(crate) fn per_operation_device_views<'a>(
    host_batch: PackedBatchHostV1<'a>,
    device: &'a PerOperationDeviceMetadata,
    packed: &LlamaPackedBatchMetadata<'_>,
    site: ExecutionSite,
) -> LlamaBatchExecutorResult<BatchDeviceViews<'a>> {
    let batch = PackedBatchV1::new(
        host_batch,
        device_span(
            &device.sequence_block_offsets,
            CudaDType::U32,
            packed.block_row_offsets().len() * U32_BYTES,
            site,
        )?,
        device_span(
            &device.physical_block_ids,
            CudaDType::U32,
            packed.physical_block_ids().len() * U32_BYTES,
            site,
        )?,
        device_span(
            &device.valid_tokens,
            CudaDType::U16,
            packed.valid_tokens().len() * U16_BYTES,
            site,
        )?,
        device_span(
            &device.row_sequence_slots,
            CudaDType::U32,
            packed.total_input_tokens() * U32_BYTES,
            site,
        )?,
        device_span(
            &device.row_positions,
            CudaDType::U32,
            packed.total_input_tokens() * U32_BYTES,
            site,
        )?,
    )
    .map_err(|source| batch_cuda(site, source))?;
    let output_token_indices = if packed.output_count() == 0 {
        None
    } else {
        let output = device.output_token_indices.as_ref().ok_or(
            LlamaBatchExecutorError::InvalidConfiguration {
                field: "output_token_indices",
                reason: "non-empty output has no cold-prepared device index buffer",
            },
        )?;
        Some(device_span(
            output,
            CudaDType::U32,
            packed.output_count() * U32_BYTES,
            site,
        )?)
    };
    Ok(BatchDeviceViews {
        batch,
        token_ids: None,
        output_token_indices,
    })
}

/// Binds regions of the one cold-prepared packed metadata slab.
pub(crate) fn packed_device_views<'a>(
    host_batch: PackedBatchHostV1<'a>,
    slab: &'a CudaDeviceBuffer,
    packed: &LlamaPackedBatchMetadata<'_>,
    layout: PackedIterationLayout,
    site: ExecutionSite,
) -> LlamaBatchExecutorResult<BatchDeviceViews<'a>> {
    layout.validate_capacity(usize::try_from(slab.byte_len()).map_err(|_| {
        LlamaBatchExecutorError::ArithmeticOverflow {
            resource: LlamaBatchExecutorResource::PackedIterationInput,
        }
    })?)?;
    let batch = PackedBatchV1::new(
        host_batch,
        device_span_region(slab, CudaDType::U32, layout.sequence_block_offsets, site)?,
        device_span_region(slab, CudaDType::U32, layout.physical_block_ids, site)?,
        device_span_region(slab, CudaDType::U16, layout.valid_tokens, site)?,
        device_span_region(slab, CudaDType::U32, layout.row_sequence_slots, site)?,
        device_span_region(slab, CudaDType::U32, layout.row_positions, site)?,
    )
    .map_err(|source| batch_cuda(site, source))?;
    let output_token_indices = if packed.output_count() == 0 {
        None
    } else {
        Some(device_span_region(
            slab,
            CudaDType::U32,
            layout.output_token_indices,
            site,
        )?)
    };
    Ok(BatchDeviceViews {
        batch,
        token_ids: Some(device_span_region(
            slab,
            CudaDType::U32,
            layout.token_ids,
            site,
        )?),
        output_token_indices,
    })
}

fn device_span(
    buffer: &CudaDeviceBuffer,
    dtype: CudaDType,
    byte_len: usize,
    site: ExecutionSite,
) -> LlamaBatchExecutorResult<CudaBufferSpan<'_>> {
    CudaBufferSpan::new(
        buffer,
        dtype,
        0,
        usize_u64(byte_len, LlamaBatchExecutorResource::HostWorkspace)?,
    )
    .map_err(|source| batch_cuda(site, source))
}

fn device_span_region(
    buffer: &CudaDeviceBuffer,
    dtype: CudaDType,
    region: ByteRegion,
    site: ExecutionSite,
) -> LlamaBatchExecutorResult<CudaBufferSpan<'_>> {
    CudaBufferSpan::new(
        buffer,
        dtype,
        usize_u64(
            region.offset,
            LlamaBatchExecutorResource::PackedIterationInput,
        )?,
        usize_u64(
            region.byte_len,
            LlamaBatchExecutorResource::PackedIterationInput,
        )?,
    )
    .map_err(|source| batch_cuda(site, source))
}

fn usize_u64(value: usize, resource: LlamaBatchExecutorResource) -> LlamaBatchExecutorResult<u64> {
    u64::try_from(value).map_err(|_| LlamaBatchExecutorError::ArithmeticOverflow { resource })
}
