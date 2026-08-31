//! Checked packed-metadata layout descriptors for one batch iteration.
//!
//! This module calculates host-side byte regions and cold capacity without
//! owning device buffers or deciding how a prepared batch is dispatched.

use super::super::batch::{LlamaBatchMetadataConfig, LlamaPackedBatchMetadata};
use super::error::{LlamaBatchExecutorError, LlamaBatchExecutorResource, LlamaBatchExecutorResult};

const U16_BYTES: usize = 2;
const U32_BYTES: usize = 4;
const PACKED_ITERATION_ALIGNMENT: usize = U32_BYTES;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ByteRegion {
    pub(crate) offset: usize,
    pub(crate) byte_len: usize,
}

impl ByteRegion {
    pub(crate) fn end(self) -> LlamaBatchExecutorResult<usize> {
        self.offset
            .checked_add(self.byte_len)
            .ok_or(LlamaBatchExecutorError::ArithmeticOverflow {
                resource: LlamaBatchExecutorResource::PackedIterationInput,
            })
    }
}

/// Dynamic, densely packed source layout for one iteration-batch upload.
/// Every U32 region is four-byte aligned and the U16 region is two-byte
/// aligned. Padding bytes are deterministic zeroes and are copied with the
/// single contiguous transfer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct PackedIterationLayout {
    pub(crate) token_ids: ByteRegion,
    pub(crate) sequence_block_offsets: ByteRegion,
    pub(crate) physical_block_ids: ByteRegion,
    pub(crate) valid_tokens: ByteRegion,
    pub(crate) row_sequence_slots: ByteRegion,
    pub(crate) row_positions: ByteRegion,
    pub(crate) output_token_indices: ByteRegion,
    pub(crate) total_bytes: usize,
}

impl PackedIterationLayout {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn checked(
        dense_rows: usize,
        sequence_block_offset_count: usize,
        physical_block_count: usize,
        valid_token_count: usize,
        active_rows: usize,
        position_count: usize,
        output_count: usize,
    ) -> LlamaBatchExecutorResult<Self> {
        let mut cursor = 0_usize;
        let token_ids = push_region(
            &mut cursor,
            dense_rows,
            U32_BYTES,
            U32_BYTES,
            LlamaBatchExecutorResource::PackedIterationInput,
        )?;
        let sequence_block_offsets = push_region(
            &mut cursor,
            sequence_block_offset_count,
            U32_BYTES,
            U32_BYTES,
            LlamaBatchExecutorResource::SequenceBlockOffsets,
        )?;
        let physical_block_ids = push_region(
            &mut cursor,
            physical_block_count,
            U32_BYTES,
            U32_BYTES,
            LlamaBatchExecutorResource::PhysicalBlockIds,
        )?;
        let valid_tokens = push_region(
            &mut cursor,
            valid_token_count,
            U16_BYTES,
            U16_BYTES,
            LlamaBatchExecutorResource::ValidTokens,
        )?;
        let row_sequence_slots = push_region(
            &mut cursor,
            active_rows,
            U32_BYTES,
            U32_BYTES,
            LlamaBatchExecutorResource::RowSequenceSlots,
        )?;
        let row_positions = push_region(
            &mut cursor,
            position_count,
            U32_BYTES,
            U32_BYTES,
            LlamaBatchExecutorResource::RowPositions,
        )?;
        let output_token_indices = push_region(
            &mut cursor,
            output_count,
            U32_BYTES,
            U32_BYTES,
            LlamaBatchExecutorResource::OutputTokenIndices,
        )?;
        let total_bytes = align_up(
            cursor,
            PACKED_ITERATION_ALIGNMENT,
            LlamaBatchExecutorResource::PackedIterationInput,
        )?;
        Ok(Self {
            token_ids,
            sequence_block_offsets,
            physical_block_ids,
            valid_tokens,
            row_sequence_slots,
            row_positions,
            output_token_indices,
            total_bytes,
        })
    }

    pub(crate) fn for_batch(
        packed: &LlamaPackedBatchMetadata<'_>,
        dense_rows: usize,
    ) -> LlamaBatchExecutorResult<Self> {
        Self::checked(
            dense_rows,
            packed.block_row_offsets().len(),
            packed.physical_block_ids().len(),
            packed.valid_tokens().len(),
            packed.total_input_tokens(),
            packed.position_ids().len(),
            packed.output_count(),
        )
    }

    pub(crate) fn capacity(bounds: LlamaBatchMetadataConfig) -> LlamaBatchExecutorResult<Self> {
        let offsets = bounds.max_rows().checked_add(1).ok_or(
            LlamaBatchExecutorError::ArithmeticOverflow {
                resource: LlamaBatchExecutorResource::SequenceBlockOffsets,
            },
        )?;
        Self::checked(
            bounds.max_input_tokens(),
            offsets,
            bounds.max_block_entries(),
            bounds.max_block_entries(),
            bounds.max_input_tokens(),
            bounds.max_input_tokens(),
            bounds.max_output_slots(),
        )
    }

    pub(crate) fn validate_capacity(self, capacity: usize) -> LlamaBatchExecutorResult<()> {
        if self.total_bytes > capacity {
            return Err(LlamaBatchExecutorError::InvalidBatch {
                field: "packed_iteration_input",
                reason: "dynamic packed input exceeds the cold-prepared slab",
            });
        }
        Ok(())
    }
}

fn align_up(
    value: usize,
    alignment: usize,
    resource: LlamaBatchExecutorResource,
) -> LlamaBatchExecutorResult<usize> {
    debug_assert!(alignment.is_power_of_two());
    value
        .checked_add(alignment - 1)
        .map(|rounded| rounded & !(alignment - 1))
        .ok_or(LlamaBatchExecutorError::ArithmeticOverflow { resource })
}

fn push_region(
    cursor: &mut usize,
    elements: usize,
    element_bytes: usize,
    alignment: usize,
    resource: LlamaBatchExecutorResource,
) -> LlamaBatchExecutorResult<ByteRegion> {
    let offset = align_up(*cursor, alignment, resource)?;
    let byte_len = elements
        .checked_mul(element_bytes)
        .ok_or(LlamaBatchExecutorError::ArithmeticOverflow { resource })?;
    *cursor = offset
        .checked_add(byte_len)
        .ok_or(LlamaBatchExecutorError::ArithmeticOverflow { resource })?;
    Ok(ByteRegion { offset, byte_len })
}
