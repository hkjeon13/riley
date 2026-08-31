//! Cold-owned batch-input buffers and allocation helpers.
//!
//! This module owns only metadata-input device and host storage. The enclosing
//! batch executor retains the model, KV, `RoPE`, output, forward, and dispatch
//! owners that use these buffers.

use riley_cuda::{CudaContext, CudaDeviceBuffer, CudaPinnedHostBuffer};

use super::super::batch::LlamaBatchMetadataConfig;
use super::super::error::{ExecutionSite, LlamaOp};
use super::error::{
    LlamaBatchExecutorError, LlamaBatchExecutorResource, LlamaBatchExecutorResult,
    checked_byte_len, cuda_error as allocation_cuda, record_close,
};
use super::host::allocate_zeroed_host_bytes;

pub(crate) const U32_BYTES: usize = 4;
pub(crate) const U16_BYTES: usize = 2;

pub(crate) struct PerOperationDeviceMetadata {
    pub(crate) sequence_block_offsets: CudaDeviceBuffer,
    pub(crate) physical_block_ids: CudaDeviceBuffer,
    pub(crate) valid_tokens: CudaDeviceBuffer,
    pub(crate) row_sequence_slots: CudaDeviceBuffer,
    pub(crate) row_positions: CudaDeviceBuffer,
    pub(crate) output_token_indices: Option<CudaDeviceBuffer>,
}

pub(crate) enum BatchDeviceInput {
    PerOperation(PerOperationDeviceMetadata),
    IterationBatch { slab: CudaDeviceBuffer },
}

pub(crate) struct PerOperationHostWorkspace {
    pub(crate) padded_tokens: Box<[u32]>,
    pub(crate) sequence_block_offsets: Box<[u8]>,
    pub(crate) physical_block_ids: Box<[u8]>,
    pub(crate) valid_tokens: Box<[u8]>,
    pub(crate) row_sequence_slots: Box<[u8]>,
    pub(crate) row_positions: Box<[u8]>,
    pub(crate) output_token_indices: Box<[u8]>,
}

pub(crate) struct IterationBatchHostWorkspace {
    pub(crate) bytes: Box<[u8]>,
    pub(crate) pinned: CudaPinnedHostBuffer,
}

pub(crate) enum BatchHostInput {
    PerOperation(PerOperationHostWorkspace),
    IterationBatch(IterationBatchHostWorkspace),
}

pub(crate) fn allocate_synchronous_device_input(
    context: &CudaContext,
    bounds: LlamaBatchMetadataConfig,
) -> LlamaBatchExecutorResult<BatchDeviceInput> {
    allocate_per_operation_device_metadata(context, bounds).map(BatchDeviceInput::PerOperation)
}

pub(crate) fn allocate_packed_device_input(
    context: &CudaContext,
    capacity: usize,
) -> LlamaBatchExecutorResult<BatchDeviceInput> {
    let slab = allocate_device(
        context,
        usize_u64(capacity, LlamaBatchExecutorResource::PackedIterationInput)?,
        ExecutionSite::global(LlamaOp::BatchMetadataUpload),
    )?;
    Ok(BatchDeviceInput::IterationBatch { slab })
}

pub(crate) fn allocate_synchronous_host_input(
    bounds: LlamaBatchMetadataConfig,
) -> LlamaBatchExecutorResult<BatchHostInput> {
    let offsets =
        bounds
            .max_rows()
            .checked_add(1)
            .ok_or(LlamaBatchExecutorError::ArithmeticOverflow {
                resource: LlamaBatchExecutorResource::SequenceBlockOffsets,
            })?;
    Ok(BatchHostInput::PerOperation(PerOperationHostWorkspace {
        padded_tokens: allocate_zeroed_u32(bounds.max_input_tokens())?,
        sequence_block_offsets: allocate_zeroed_host_bytes(offsets, U32_BYTES)?,
        physical_block_ids: allocate_zeroed_host_bytes(bounds.max_block_entries(), U32_BYTES)?,
        valid_tokens: allocate_zeroed_host_bytes(bounds.max_block_entries(), U16_BYTES)?,
        row_sequence_slots: allocate_zeroed_host_bytes(bounds.max_input_tokens(), U32_BYTES)?,
        row_positions: allocate_zeroed_host_bytes(bounds.max_input_tokens(), U32_BYTES)?,
        output_token_indices: allocate_zeroed_host_bytes(bounds.max_output_slots(), U32_BYTES)?,
    }))
}

pub(crate) fn allocate_packed_host_input(
    context: &CudaContext,
    capacity: usize,
) -> LlamaBatchExecutorResult<BatchHostInput> {
    let bytes = allocate_zeroed_host_bytes(capacity, 1)?;
    let pinned = context
        .allocate_pinned_host_buffer(usize_u64(
            capacity,
            LlamaBatchExecutorResource::PinnedIterationInput,
        )?)
        .map_err(|source| {
            allocation_cuda(ExecutionSite::global(LlamaOp::BatchMetadataUpload), source)
        })?;
    Ok(BatchHostInput::IterationBatch(
        IterationBatchHostWorkspace { bytes, pinned },
    ))
}

fn allocate_device(
    context: &CudaContext,
    byte_len: u64,
    site: ExecutionSite,
) -> LlamaBatchExecutorResult<CudaDeviceBuffer> {
    context
        .allocate_device_buffer(byte_len)
        .map_err(|source| allocation_cuda(site, source))
}

pub(crate) fn close_device_input(input: BatchDeviceInput) -> Option<LlamaBatchExecutorError> {
    let mut first = None;
    match input {
        BatchDeviceInput::PerOperation(metadata) => {
            let PerOperationDeviceMetadata {
                sequence_block_offsets,
                physical_block_ids,
                valid_tokens,
                row_sequence_slots,
                row_positions,
                output_token_indices,
            } = metadata;
            for (resource, result) in [
                (
                    LlamaBatchExecutorResource::SequenceBlockOffsets,
                    sequence_block_offsets.close(),
                ),
                (
                    LlamaBatchExecutorResource::PhysicalBlockIds,
                    physical_block_ids.close(),
                ),
                (
                    LlamaBatchExecutorResource::ValidTokens,
                    valid_tokens.close(),
                ),
                (
                    LlamaBatchExecutorResource::RowSequenceSlots,
                    row_sequence_slots.close(),
                ),
                (
                    LlamaBatchExecutorResource::RowPositions,
                    row_positions.close(),
                ),
            ] {
                record_close(&mut first, resource, result);
            }
            if let Some(buffer) = output_token_indices {
                record_close(
                    &mut first,
                    LlamaBatchExecutorResource::OutputTokenIndices,
                    buffer.close(),
                );
            }
        }
        BatchDeviceInput::IterationBatch { slab } => record_close(
            &mut first,
            LlamaBatchExecutorResource::PackedIterationInput,
            slab.close(),
        ),
    }
    first
}

pub(crate) fn close_host_input(input: BatchHostInput) -> Option<LlamaBatchExecutorError> {
    let mut first = None;
    if let BatchHostInput::IterationBatch(host) = input {
        record_close(
            &mut first,
            LlamaBatchExecutorResource::PinnedIterationInput,
            host.pinned.close(),
        );
    }
    first
}

fn usize_u64(value: usize, resource: LlamaBatchExecutorResource) -> LlamaBatchExecutorResult<u64> {
    u64::try_from(value).map_err(|_| LlamaBatchExecutorError::ArithmeticOverflow { resource })
}

fn allocate_per_operation_device_metadata(
    context: &CudaContext,
    bounds: LlamaBatchMetadataConfig,
) -> LlamaBatchExecutorResult<PerOperationDeviceMetadata> {
    let offsets =
        bounds
            .max_rows()
            .checked_add(1)
            .ok_or(LlamaBatchExecutorError::ArithmeticOverflow {
                resource: LlamaBatchExecutorResource::SequenceBlockOffsets,
            })?;
    let allocate = |elements: usize,
                    element_bytes: usize,
                    resource: LlamaBatchExecutorResource|
     -> LlamaBatchExecutorResult<CudaDeviceBuffer> {
        let bytes = checked_byte_len(elements, element_bytes, resource)?;
        allocate_device(
            context,
            usize_u64(bytes, resource)?,
            ExecutionSite::global(LlamaOp::BatchMetadataUpload),
        )
    };
    Ok(PerOperationDeviceMetadata {
        sequence_block_offsets: allocate(
            offsets,
            U32_BYTES,
            LlamaBatchExecutorResource::SequenceBlockOffsets,
        )?,
        physical_block_ids: allocate(
            bounds.max_block_entries(),
            U32_BYTES,
            LlamaBatchExecutorResource::PhysicalBlockIds,
        )?,
        valid_tokens: allocate(
            bounds.max_block_entries(),
            U16_BYTES,
            LlamaBatchExecutorResource::ValidTokens,
        )?,
        row_sequence_slots: allocate(
            bounds.max_input_tokens(),
            U32_BYTES,
            LlamaBatchExecutorResource::RowSequenceSlots,
        )?,
        row_positions: allocate(
            bounds.max_input_tokens(),
            U32_BYTES,
            LlamaBatchExecutorResource::RowPositions,
        )?,
        output_token_indices: if bounds.max_output_slots() == 0 {
            None
        } else {
            Some(allocate(
                bounds.max_output_slots(),
                U32_BYTES,
                LlamaBatchExecutorResource::OutputTokenIndices,
            )?)
        },
    })
}

fn allocate_zeroed_u32(elements: usize) -> LlamaBatchExecutorResult<Box<[u32]>> {
    let requested_bytes = checked_byte_len(
        elements,
        U32_BYTES,
        LlamaBatchExecutorResource::HostWorkspace,
    )?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(elements)
        .map_err(|_| LlamaBatchExecutorError::HostAllocation {
            resource: LlamaBatchExecutorResource::HostWorkspace,
            requested_bytes: requested_bytes as u64,
        })?;
    values.resize(elements, 0);
    Ok(values.into_boxed_slice())
}
