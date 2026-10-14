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
    checked_byte_len, cuda_error as allocation_cuda, record_close, usize_u64,
};
use super::host::allocate_zeroed_host_bytes;
use super::metadata::sequence_block_offset_count;

pub(crate) const U32_BYTES: usize = 4;
pub(crate) const U16_BYTES: usize = 2;

/// Trailing U32 words of the device row-position buffer: active rows, output rows.
pub(crate) const STEP_COUNT_WORDS: usize = 2;

pub(crate) struct PerOperationDeviceMetadata {
    pub(crate) sequence_block_offsets: CudaDeviceBuffer,
    pub(crate) physical_block_ids: CudaDeviceBuffer,
    pub(crate) valid_tokens: CudaDeviceBuffer,
    pub(crate) row_sequence_slots: CudaDeviceBuffer,
    pub(crate) row_positions: CudaDeviceBuffer,
    pub(crate) output_token_indices: Option<CudaDeviceBuffer>,
    /// Token ids of the iteration when uploads are staged into the command
    /// batch (the forward's own token buffer is then not written).
    pub(crate) token_ids: Option<CudaDeviceBuffer>,
}

pub(crate) enum BatchDeviceInput {
    PerOperation(PerOperationDeviceMetadata),
    IterationBatch { slab: CudaDeviceBuffer },
}

/// Owned copy of the bytes last uploaded for one kind of step metadata.
pub(crate) type UploadedBytes = Vec<u8>;

pub(crate) struct PerOperationHostWorkspace {
    pub(crate) padded_tokens: Box<[u32]>,
    pub(crate) sequence_block_offsets: Box<[u8]>,
    pub(crate) physical_block_ids: Box<[u8]>,
    pub(crate) valid_tokens: Box<[u8]>,
    pub(crate) row_sequence_slots: Box<[u8]>,
    pub(crate) row_positions: Box<[u8]>,
    pub(crate) output_token_indices: Box<[u8]>,
    /// Step counts last written to the device (active rows, output rows);
    /// a step with the same counts skips that copy.
    pub(crate) uploaded_step_counts: Option<[u32; STEP_COUNT_WORDS]>,
    /// Row kinds (one U32 per token row, 1 = prompt token) for the
    /// tensor-core prompt attention.
    pub(crate) row_kinds: Box<[u8]>,
    /// Bytes last copied to the device for the metadata that often repeats
    /// from one step to the next (block offsets, block ids, row slots, output
    /// indices); a step whose bytes are the same skips that copy. Emptied
    /// before a copy starts, so a failed copy is never taken as current.
    pub(crate) uploaded: [UploadedBytes; UPLOADED_METADATA_KINDS],
    /// Staged step uploads: the pinned bytes of this iteration and where each
    /// run of them goes. `None` keeps one synchronized upload per buffer.
    pub(crate) staging: Option<CudaPinnedHostBuffer>,
    pub(crate) staged: [StagedCopy; MAX_STAGED_COPIES],
    pub(crate) staged_count: usize,
    /// Results the command batch copied back into `staging`, after the upload
    /// region: the greedy selections (`staged_greedy_bytes` of them, at
    /// `staged_output_offset`) and the embedding error report (32 bytes after
    /// the greedy capacity). Cells, because the iteration body records them
    /// while it only shares the workspace.
    pub(crate) staged_output_offset: u64,
    pub(crate) staged_report_offset: u64,
    pub(crate) staged_greedy_bytes: std::cell::Cell<usize>,
    pub(crate) staged_report: std::cell::Cell<bool>,
}

/// Bytes of the device embedding error report (`RileyCudaEmbeddingErrorReport`).
pub(crate) const EMBEDDING_REPORT_BYTES: usize = 32;

/// Device buffer a staged run of bytes is copied to.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum StagedTarget {
    Tokens,
    BlockOffsets,
    BlockIds,
    ValidTokens,
    RowSlots,
    RowPositions,
    OutputIndices,
}

/// One copy from the pinned staging buffer into a device buffer.
#[derive(Clone, Copy, Debug)]
pub(crate) struct StagedCopy {
    pub(crate) target: StagedTarget,
    pub(crate) destination_offset: u64,
    pub(crate) staging_offset: u64,
    pub(crate) byte_len: u64,
}

/// Tokens, five metadata arrays, the step counts, the row kinds and the
/// output indices.
pub(crate) const MAX_STAGED_COPIES: usize = 9;

pub(crate) const UPLOADED_METADATA_KINDS: usize = 4;
pub(crate) const UPLOADED_BLOCK_OFFSETS: usize = 0;
pub(crate) const UPLOADED_BLOCK_IDS: usize = 1;
pub(crate) const UPLOADED_ROW_SLOTS: usize = 2;
pub(crate) const UPLOADED_OUTPUT_INDICES: usize = 3;

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
    staged: bool,
) -> LlamaBatchExecutorResult<BatchDeviceInput> {
    let mut metadata = allocate_per_operation_device_metadata(context, bounds)?;
    if staged {
        let bytes = checked_byte_len(
            bounds.max_input_tokens(),
            U32_BYTES,
            LlamaBatchExecutorResource::HostWorkspace,
        )?;
        metadata.token_ids = Some(allocate_device(
            context,
            usize_u64(bytes, LlamaBatchExecutorResource::HostWorkspace)?,
            ExecutionSite::global(LlamaOp::Embedding),
        )?);
    }
    Ok(BatchDeviceInput::PerOperation(metadata))
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

/// Pinned bytes one iteration can stage: every array at its bound, each run
/// rounded up to eight bytes.
fn staged_upload_capacity(bounds: LlamaBatchMetadataConfig) -> LlamaBatchExecutorResult<usize> {
    let offsets = sequence_block_offset_count(bounds.max_rows())?;
    let runs = [
        (bounds.max_input_tokens(), U32_BYTES),
        (offsets, U32_BYTES),
        (bounds.max_block_entries(), U32_BYTES),
        (bounds.max_block_entries(), U16_BYTES),
        (bounds.max_input_tokens(), U32_BYTES),
        (bounds.max_input_tokens(), U32_BYTES),
        (STEP_COUNT_WORDS, U32_BYTES),
        // Row kinds, then the generated-row list (count and rows).
        (bounds.max_input_tokens() + 1 + bounds.max_rows(), U32_BYTES),
        (bounds.max_output_slots(), U32_BYTES),
    ];
    let mut total = 0_usize;
    for (elements, element_bytes) in runs {
        let bytes = checked_byte_len(
            elements,
            element_bytes,
            LlamaBatchExecutorResource::HostWorkspace,
        )?;
        total = total.checked_add(bytes.div_ceil(8) * 8).ok_or(
            LlamaBatchExecutorError::InvalidConfiguration {
                field: "staged_step_uploads",
                reason: "staged upload capacity overflows",
            },
        )?;
    }
    Ok(total.max(8))
}

pub(crate) fn allocate_synchronous_host_input(
    context: &CudaContext,
    bounds: LlamaBatchMetadataConfig,
    staged: bool,
) -> LlamaBatchExecutorResult<BatchHostInput> {
    let staged_output_offset = staged_upload_capacity(bounds)?;
    let staged_greedy_capacity = checked_byte_len(
        bounds.max_output_slots(),
        2 * U32_BYTES,
        LlamaBatchExecutorResource::HostWorkspace,
    )?
    .div_ceil(8)
        * 8;
    let staged_report_offset = staged_output_offset + staged_greedy_capacity;
    let staging = if staged {
        let capacity = staged_report_offset + EMBEDDING_REPORT_BYTES;
        Some(
            context
                .allocate_pinned_host_buffer(usize_u64(
                    capacity,
                    LlamaBatchExecutorResource::PinnedIterationInput,
                )?)
                .map_err(|source| {
                    allocation_cuda(ExecutionSite::global(LlamaOp::BatchMetadataUpload), source)
                })?,
        )
    } else {
        None
    };
    let offsets = sequence_block_offset_count(bounds.max_rows())?;
    Ok(BatchHostInput::PerOperation(PerOperationHostWorkspace {
        padded_tokens: allocate_zeroed_u32(bounds.max_input_tokens())?,
        sequence_block_offsets: allocate_zeroed_host_bytes(offsets, U32_BYTES)?,
        physical_block_ids: allocate_zeroed_host_bytes(bounds.max_block_entries(), U32_BYTES)?,
        valid_tokens: allocate_zeroed_host_bytes(bounds.max_block_entries(), U16_BYTES)?,
        row_sequence_slots: allocate_zeroed_host_bytes(bounds.max_input_tokens(), U32_BYTES)?,
        row_positions: allocate_zeroed_host_bytes(bounds.max_input_tokens(), U32_BYTES)?,
        output_token_indices: allocate_zeroed_host_bytes(bounds.max_output_slots(), U32_BYTES)?,
        uploaded_step_counts: None,
        row_kinds: allocate_zeroed_host_bytes(
            bounds.max_input_tokens() + 1 + bounds.max_rows(),
            U32_BYTES,
        )?,
        uploaded: [
            Vec::with_capacity(offsets * U32_BYTES),
            Vec::with_capacity(bounds.max_block_entries() * U32_BYTES),
            Vec::with_capacity(bounds.max_input_tokens() * U32_BYTES),
            Vec::with_capacity(bounds.max_output_slots() * U32_BYTES),
        ],
        staging,
        staged: [StagedCopy {
            target: StagedTarget::Tokens,
            destination_offset: 0,
            staging_offset: 0,
            byte_len: 0,
        }; MAX_STAGED_COPIES],
        staged_count: 0,
        staged_output_offset: staged_output_offset as u64,
        staged_report_offset: staged_report_offset as u64,
        staged_greedy_bytes: std::cell::Cell::new(0),
        staged_report: std::cell::Cell::new(false),
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
                token_ids,
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
            if let Some(buffer) = token_ids {
                record_close(
                    &mut first,
                    LlamaBatchExecutorResource::HostWorkspace,
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
    match input {
        BatchHostInput::IterationBatch(host) => record_close(
            &mut first,
            LlamaBatchExecutorResource::PinnedIterationInput,
            host.pinned.close(),
        ),
        BatchHostInput::PerOperation(host) => {
            if let Some(staging) = host.staging {
                record_close(
                    &mut first,
                    LlamaBatchExecutorResource::PinnedIterationInput,
                    staging.close(),
                );
            }
        }
    }
    first
}

fn allocate_per_operation_device_metadata(
    context: &CudaContext,
    bounds: LlamaBatchMetadataConfig,
) -> LlamaBatchExecutorResult<PerOperationDeviceMetadata> {
    let offsets = sequence_block_offset_count(bounds.max_rows())?;
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
        // Two U32 words after the positions hold the step's active row count
        // and output row count for recorded steps that read them on the
        // device; one row-kind word per token row follows them, then the
        // generated-row list of the dense prompt attention (count and rows).
        row_positions: allocate(
            2 * bounds.max_input_tokens() + STEP_COUNT_WORDS + 1 + bounds.max_rows(),
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
        token_ids: None,
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
