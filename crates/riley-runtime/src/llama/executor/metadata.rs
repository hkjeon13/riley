//! Checked packed-metadata layout descriptors for one batch iteration.
//!
//! This module calculates host-side byte regions and cold capacity without
//! owning device buffers or deciding how a prepared batch is dispatched.

use super::super::batch::{
    LLAMA_BATCH_METADATA_V1_VERSION, LlamaBatchMetadataConfig, LlamaPackedBatchMetadata,
};
use super::error::{
    LlamaBatchExecutorError, LlamaBatchExecutorResource, LlamaBatchExecutorResult, checked_byte_len,
};

const U16_BYTES: usize = 2;
const U32_BYTES: usize = 4;
const PACKED_ITERATION_ALIGNMENT: usize = U32_BYTES;

/// Returns the CSR block-row offset count required for one bounded batch.
pub(in crate::llama) fn sequence_block_offset_count(
    maximum_rows: usize,
) -> LlamaBatchExecutorResult<usize> {
    maximum_rows
        .checked_add(1)
        .ok_or(LlamaBatchExecutorError::ArithmeticOverflow {
            resource: LlamaBatchExecutorResource::SequenceBlockOffsets,
        })
}

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
        let offsets = sequence_block_offset_count(bounds.max_rows())?;
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

    /// Checks a CUDA ABI slab capacity before binding or writing packed metadata.
    pub(crate) fn validate_u64_capacity(self, capacity: u64) -> LlamaBatchExecutorResult<()> {
        let capacity =
            usize::try_from(capacity).map_err(|_| LlamaBatchExecutorError::ArithmeticOverflow {
                resource: LlamaBatchExecutorResource::PackedIterationInput,
            })?;
        self.validate_capacity(capacity)
    }
}

/// Validates one packed batch against executor-independent host bounds.
///
/// The owner supplies its immutable model and profile-position-limit scalars so
/// this helper remains independent of prepared CUDA resources and dispatch.
#[allow(clippy::large_types_passed_by_value)]
pub(in crate::llama) fn validate_for_execution(
    packed: LlamaPackedBatchMetadata<'_>,
    vocabulary_size: usize,
    maximum_position_count: u64,
    bounds: LlamaBatchMetadataConfig,
    profile_position_limit: Option<u64>,
) -> LlamaBatchExecutorResult<()> {
    if packed.schema_version() != LLAMA_BATCH_METADATA_V1_VERSION {
        return Err(LlamaBatchExecutorError::InvalidBatch {
            field: "schema_version",
            reason: "packed metadata version differs from the executor contract",
        });
    }
    if packed.total_input_tokens() > bounds.max_input_tokens()
        || packed.row_count() > bounds.max_rows()
        || packed.physical_block_ids().len() > bounds.max_block_entries()
        || packed.output_count() > bounds.max_output_slots()
    {
        return Err(LlamaBatchExecutorError::InvalidBatch {
            field: "capacity",
            reason: "packed metadata exceeds the executor's cold bounds",
        });
    }
    for (position, &token_id) in packed.input_token_ids().iter().enumerate() {
        if usize::try_from(token_id)
            .ok()
            .is_none_or(|token| token >= vocabulary_size)
        {
            return Err(LlamaBatchExecutorError::TokenOutOfRange {
                position,
                token_id,
                vocabulary_size,
            });
        }
    }
    for (&row, &position) in packed
        .row_sequence_slots()
        .iter()
        .zip(packed.position_ids())
    {
        match profile_position_limit {
            Some(profile_maximum) if u64::from(position) >= profile_maximum => {
                return Err(LlamaBatchExecutorError::PositionOutOfRange {
                    row: usize::try_from(row).unwrap_or(usize::MAX),
                    position,
                    maximum: usize::try_from(profile_maximum).unwrap_or(usize::MAX),
                });
            }
            _ => {}
        }
        if u64::from(position) >= maximum_position_count {
            return Err(LlamaBatchExecutorError::PositionOutOfRange {
                row: usize::try_from(row).unwrap_or(usize::MAX),
                position,
                maximum: usize::try_from(maximum_position_count).unwrap_or(usize::MAX),
            });
        }
    }
    Ok(())
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
    let byte_len = checked_byte_len(elements, element_bytes, resource)?;
    *cursor = offset
        .checked_add(byte_len)
        .ok_or(LlamaBatchExecutorError::ArithmeticOverflow { resource })?;
    Ok(ByteRegion { offset, byte_len })
}

/// Writes the seven packed-batch source arrays into one preallocated slab.
///
/// The caller retains ownership of the slab and decides whether its bytes are
/// copied synchronously or through a pinned host transfer.
pub(crate) fn pack_iteration_input(
    packed: &LlamaPackedBatchMetadata<'_>,
    dense_rows: usize,
    layout: PackedIterationLayout,
    destination: &mut [u8],
) -> LlamaBatchExecutorResult<()> {
    layout.validate_capacity(destination.len())?;
    if packed.total_input_tokens() > dense_rows {
        return Err(LlamaBatchExecutorError::InvalidBatch {
            field: "dense_rows",
            reason: "active input rows exceed the selected packed token region",
        });
    }
    destination[..layout.total_bytes].fill(0);
    let active_token_bytes = checked_byte_len(
        packed.total_input_tokens(),
        U32_BYTES,
        LlamaBatchExecutorResource::PackedIterationInput,
    )?;
    encode_u32_region(
        packed.input_token_ids(),
        destination,
        ByteRegion {
            offset: layout.token_ids.offset,
            byte_len: active_token_bytes,
        },
        LlamaBatchExecutorResource::PackedIterationInput,
    )?;
    encode_u32_region(
        packed.block_row_offsets(),
        destination,
        layout.sequence_block_offsets,
        LlamaBatchExecutorResource::SequenceBlockOffsets,
    )?;
    encode_u32_region(
        packed.physical_block_ids(),
        destination,
        layout.physical_block_ids,
        LlamaBatchExecutorResource::PhysicalBlockIds,
    )?;
    encode_u16_region(
        packed.valid_tokens(),
        destination,
        layout.valid_tokens,
        LlamaBatchExecutorResource::ValidTokens,
    )?;
    encode_u32_region(
        packed.row_sequence_slots(),
        destination,
        layout.row_sequence_slots,
        LlamaBatchExecutorResource::RowSequenceSlots,
    )?;
    encode_u32_region(
        packed.position_ids(),
        destination,
        layout.row_positions,
        LlamaBatchExecutorResource::RowPositions,
    )?;
    encode_u32_region(
        packed.output_token_indices(),
        destination,
        layout.output_token_indices,
        LlamaBatchExecutorResource::OutputTokenIndices,
    )?;
    Ok(())
}

/// Encodes native-endian `u32` values into an already-sized byte prefix.
pub(crate) fn encode_u32(source: &[u32], destination: &mut [u8]) {
    for (value, bytes) in source.iter().zip(destination.chunks_exact_mut(U32_BYTES)) {
        bytes.copy_from_slice(&value.to_ne_bytes());
    }
}

fn encode_u32_region(
    source: &[u32],
    destination: &mut [u8],
    region: ByteRegion,
    resource: LlamaBatchExecutorResource,
) -> LlamaBatchExecutorResult<()> {
    let bytes = checked_region_slice_mut(
        destination,
        source.len(),
        U32_BYTES,
        region,
        resource,
        "U32 region length does not match its host source",
    )?;
    encode_u32(source, bytes);
    Ok(())
}

fn encode_u16_region(
    source: &[u16],
    destination: &mut [u8],
    region: ByteRegion,
    resource: LlamaBatchExecutorResource,
) -> LlamaBatchExecutorResult<()> {
    let bytes = checked_region_slice_mut(
        destination,
        source.len(),
        U16_BYTES,
        region,
        resource,
        "U16 region length does not match its host source",
    )?;
    encode_u16(source, bytes);
    Ok(())
}

fn checked_region_slice_mut<'a>(
    destination: &'a mut [u8],
    source_len: usize,
    element_bytes: usize,
    region: ByteRegion,
    resource: LlamaBatchExecutorResource,
    mismatch_reason: &'static str,
) -> LlamaBatchExecutorResult<&'a mut [u8]> {
    let expected = checked_byte_len(source_len, element_bytes, resource)?;
    if region.byte_len != expected {
        return Err(LlamaBatchExecutorError::InvalidConfiguration {
            field: "packed_iteration_layout",
            reason: mismatch_reason,
        });
    }
    region_slice_mut(destination, region, resource)
}

/// Encodes native-endian `u16` values into an already-sized byte prefix.
pub(crate) fn encode_u16(source: &[u16], destination: &mut [u8]) {
    for (value, bytes) in source.iter().zip(destination.chunks_exact_mut(U16_BYTES)) {
        bytes.copy_from_slice(&value.to_ne_bytes());
    }
}

fn region_slice_mut(
    bytes: &mut [u8],
    region: ByteRegion,
    resource: LlamaBatchExecutorResource,
) -> LlamaBatchExecutorResult<&mut [u8]> {
    bytes
        .get_mut(region.offset..region.end()?)
        .ok_or(LlamaBatchExecutorError::ArithmeticOverflow { resource })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_region_slice_preserves_error_precedence() {
        let mut destination = [0_u8; U32_BYTES];
        assert!(matches!(
            checked_region_slice_mut(
                &mut destination,
                usize::MAX,
                U32_BYTES,
                ByteRegion {
                    offset: U32_BYTES,
                    byte_len: 0,
                },
                LlamaBatchExecutorResource::RowPositions,
                "test mismatch",
            ),
            Err(LlamaBatchExecutorError::ArithmeticOverflow {
                resource: LlamaBatchExecutorResource::RowPositions,
            })
        ));

        assert!(matches!(
            checked_region_slice_mut(
                &mut destination,
                1,
                U32_BYTES,
                ByteRegion {
                    offset: usize::MAX,
                    byte_len: U32_BYTES,
                },
                LlamaBatchExecutorResource::RowPositions,
                "test mismatch",
            ),
            Err(LlamaBatchExecutorError::ArithmeticOverflow {
                resource: LlamaBatchExecutorResource::PackedIterationInput,
            })
        ));

        assert!(matches!(
            checked_region_slice_mut(
                &mut destination,
                1,
                U32_BYTES,
                ByteRegion {
                    offset: U32_BYTES,
                    byte_len: U16_BYTES,
                },
                LlamaBatchExecutorResource::RowPositions,
                "test mismatch",
            ),
            Err(LlamaBatchExecutorError::InvalidConfiguration {
                field: "packed_iteration_layout",
                reason: "test mismatch",
            })
        ));

        assert!(matches!(
            checked_region_slice_mut(
                &mut destination,
                1,
                U32_BYTES,
                ByteRegion {
                    offset: U32_BYTES,
                    byte_len: U32_BYTES,
                },
                LlamaBatchExecutorResource::RowPositions,
                "test mismatch",
            ),
            Err(LlamaBatchExecutorError::ArithmeticOverflow {
                resource: LlamaBatchExecutorResource::RowPositions,
            })
        ));
    }

    #[test]
    fn typed_region_encoders_preserve_mismatch_reason_without_writing() {
        let mut u32_destination = [0xA5_u8; U32_BYTES];
        assert!(matches!(
            encode_u32_region(
                &[7],
                &mut u32_destination,
                ByteRegion {
                    offset: 0,
                    byte_len: U16_BYTES,
                },
                LlamaBatchExecutorResource::RowPositions,
            ),
            Err(LlamaBatchExecutorError::InvalidConfiguration {
                field: "packed_iteration_layout",
                reason: "U32 region length does not match its host source",
            })
        ));
        assert_eq!(u32_destination, [0xA5; U32_BYTES]);

        let mut u16_destination = [0x5A_u8; U16_BYTES];
        assert!(matches!(
            encode_u16_region(
                &[7],
                &mut u16_destination,
                ByteRegion {
                    offset: 0,
                    byte_len: U32_BYTES,
                },
                LlamaBatchExecutorResource::ValidTokens,
            ),
            Err(LlamaBatchExecutorError::InvalidConfiguration {
                field: "packed_iteration_layout",
                reason: "U16 region length does not match its host source",
            })
        ));
        assert_eq!(u16_destination, [0x5A; U16_BYTES]);
    }

    #[test]
    fn packed_layout_u64_capacity_preserves_conversion_and_layout_errors() {
        let layout =
            PackedIterationLayout::checked(1, 1, 0, 0, 0, 0, 0).expect("small packed layout");
        let exact_capacity = u64::try_from(layout.total_bytes).expect("native capacity fits ABI");
        layout
            .validate_u64_capacity(exact_capacity)
            .expect("exact capacity is accepted");
        assert!(matches!(
            layout.validate_u64_capacity(exact_capacity - 1),
            Err(LlamaBatchExecutorError::InvalidBatch {
                field: "packed_iteration_input",
                reason: "dynamic packed input exceeds the cold-prepared slab",
            })
        ));

        let maximum_capacity = layout.validate_u64_capacity(u64::MAX);
        match usize::try_from(u64::MAX) {
            Ok(_) => maximum_capacity.expect("maximum ABI capacity fits the native width"),
            Err(_) => assert!(matches!(
                maximum_capacity,
                Err(LlamaBatchExecutorError::ArithmeticOverflow {
                    resource: LlamaBatchExecutorResource::PackedIterationInput,
                })
            )),
        }
    }
}
