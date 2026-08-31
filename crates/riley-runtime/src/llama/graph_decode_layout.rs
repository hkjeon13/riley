//! Fixed-offset pure-decode graph metadata layout geometry.
//!
//! This C07-1 descriptor calculates cold slab regions only. It neither
//! allocates or addresses storage, nor packs bytes, validates a dynamic batch,
//! constructs an existing batch ABI object, or selects and runs a graph.

use std::error;
use std::fmt;

use super::graph::PURE_DECODE_GRAPH_BUCKETS;

const U16_BYTES: u64 = 2;
const U32_BYTES: u64 = 4;
const BASE_ALIGNMENT: u64 = U32_BYTES;
const FIELD_COUNT: usize = 9;

/// Version of the C07 fixed pure-decode metadata layout geometry.
pub(crate) const PURE_DECODE_GRAPH_METADATA_LAYOUT_SCHEMA_VERSION: u32 = 1;

/// Result of fixed pure-decode metadata layout calculation.
pub(crate) type PureDecodeGraphMetadataLayoutResult<T> =
    Result<T, PureDecodeGraphMetadataLayoutError>;

/// Canonical fixed-region order for pure-decode graph metadata.
///
/// The enum order is the future signature-hash order. This slice deliberately
/// specifies geometry only; it does not define field contents or sentinels.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub(crate) enum PureDecodeGraphMetadataField {
    /// Versioned fixed header bytes.
    Header = 1,
    /// One fixed-width token identifier per bucket row.
    TokenIds = 2,
    /// One fixed-width absolute position per bucket row.
    PositionIds = 3,
    /// One fixed-width row-to-sequence slot per bucket row.
    RowSequenceSlots = 4,
    /// Fixed CSR row-offset descriptors for every bucket row plus the tail.
    SequenceBlockOffsets = 5,
    /// Fixed-capacity physical block identifiers.
    PhysicalBlockIds = 6,
    /// Fixed-capacity valid-token counts aligned with physical block IDs.
    ValidTokens = 7,
    /// One fixed-width output-token index per bucket row.
    OutputTokenIndices = 8,
    /// Versioned fixed control and status bytes.
    ControlStatus = 9,
}

impl PureDecodeGraphMetadataField {
    /// Canonical field sequence for fixed layout and future signature hashing.
    pub(crate) const ALL: [Self; FIELD_COUNT] = [
        Self::Header,
        Self::TokenIds,
        Self::PositionIds,
        Self::RowSequenceSlots,
        Self::SequenceBlockOffsets,
        Self::PhysicalBlockIds,
        Self::ValidTokens,
        Self::OutputTokenIndices,
        Self::ControlStatus,
    ];

    const fn index(self) -> usize {
        match self {
            Self::Header => 0,
            Self::TokenIds => 1,
            Self::PositionIds => 2,
            Self::RowSequenceSlots => 3,
            Self::SequenceBlockOffsets => 4,
            Self::PhysicalBlockIds => 5,
            Self::ValidTokens => 6,
            Self::OutputTokenIndices => 7,
            Self::ControlStatus => 8,
        }
    }

    const fn id(self) -> &'static str {
        match self {
            Self::Header => "header",
            Self::TokenIds => "token-ids",
            Self::PositionIds => "position-ids",
            Self::RowSequenceSlots => "row-sequence-slots",
            Self::SequenceBlockOffsets => "sequence-block-offsets",
            Self::PhysicalBlockIds => "physical-block-ids",
            Self::ValidTokens => "valid-tokens",
            Self::OutputTokenIndices => "output-token-indices",
            Self::ControlStatus => "control-status",
        }
    }
}

type PureDecodeGraphMetadataFieldSpec = (PureDecodeGraphMetadataField, u64, u64);

/// Explicit cold dimensions for one exact pure-decode graph metadata slab.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct PureDecodeGraphMetadataLayoutSpec {
    bucket_rows: u32,
    block_entry_capacity: u64,
    header_bytes: u64,
    control_status_bytes: u64,
}

impl PureDecodeGraphMetadataLayoutSpec {
    /// Creates an unvalidated cold layout specification.
    #[must_use]
    pub(crate) const fn new(
        bucket_rows: u32,
        block_entry_capacity: u64,
        header_bytes: u64,
        control_status_bytes: u64,
    ) -> Self {
        Self {
            bucket_rows,
            block_entry_capacity,
            header_bytes,
            control_status_bytes,
        }
    }
}

/// One non-owning fixed byte region in the cold metadata slab geometry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct PureDecodeGraphMetadataRegion {
    offset: u64,
    byte_len: u64,
    alignment: u64,
}

impl PureDecodeGraphMetadataRegion {
    const ZERO: Self = Self {
        offset: 0,
        byte_len: 0,
        alignment: 1,
    };

    /// Returns the fixed offset from an appropriately aligned slab base.
    #[must_use]
    pub(crate) const fn offset(self) -> u64 {
        self.offset
    }

    /// Returns the fixed byte capacity of this region.
    #[must_use]
    pub(crate) const fn byte_len(self) -> u64 {
        self.byte_len
    }

    /// Returns the required alignment of this region's offset.
    #[must_use]
    pub(crate) const fn alignment(self) -> u64 {
        self.alignment
    }
}

/// Closed failure from cold fixed-layout geometry calculation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub(crate) enum PureDecodeGraphMetadataLayoutError {
    /// The requested row count is not one exact C07 graph bucket.
    UnsupportedBucket {
        /// Requested fixed bucket row count.
        bucket_rows: u32,
    },
    /// The fixed block-entry capacity cannot represent one block per row.
    BlockEntryCapacityTooSmall {
        /// Exact graph bucket row count.
        bucket_rows: u32,
        /// Configured cold block-entry capacity.
        block_entry_capacity: u64,
    },
    /// One opaque fixed region omitted its required capacity.
    ZeroSizedOpaqueRegion {
        /// Header or control/status field with zero capacity.
        field: PureDecodeGraphMetadataField,
    },
    /// Checked fixed-layout arithmetic could not be represented.
    ArithmeticOverflow {
        /// Field whose offset or byte capacity calculation overflowed.
        field: PureDecodeGraphMetadataField,
    },
}

impl fmt::Display for PureDecodeGraphMetadataLayoutError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedBucket { bucket_rows } => {
                write!(
                    formatter,
                    "unsupported pure-decode graph bucket {bucket_rows}"
                )
            }
            Self::BlockEntryCapacityTooSmall {
                bucket_rows,
                block_entry_capacity,
            } => write!(
                formatter,
                "pure-decode graph bucket {bucket_rows} needs at least {bucket_rows} block entries, got {block_entry_capacity}"
            ),
            Self::ZeroSizedOpaqueRegion { field } => {
                write!(
                    formatter,
                    "fixed {} region has zero byte capacity",
                    field.id()
                )
            }
            Self::ArithmeticOverflow { field } => {
                write!(
                    formatter,
                    "fixed {} region arithmetic overflowed",
                    field.id()
                )
            }
        }
    }
}

impl error::Error for PureDecodeGraphMetadataLayoutError {}

/// Immutable fixed-offset geometry for one exact pure-decode graph bucket.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct PureDecodeGraphMetadataLayout {
    bucket_rows: u32,
    block_entry_capacity: u64,
    regions: [PureDecodeGraphMetadataRegion; FIELD_COUNT],
    total_bytes: u64,
}

impl PureDecodeGraphMetadataLayout {
    /// Calculates fixed geometry for one exact C07 bucket and cold capacities.
    ///
    /// The input bucket must already be one exact catalog member; this
    /// descriptor does not round an active row count upward.
    pub(crate) fn try_new(
        spec: PureDecodeGraphMetadataLayoutSpec,
    ) -> PureDecodeGraphMetadataLayoutResult<Self> {
        let bucket_rows = validate_cold_spec(spec)?;
        let fields = fixed_field_specs(spec, bucket_rows)?;
        let (regions, total_bytes) = fixed_regions(fields)?;
        Ok(Self {
            bucket_rows: spec.bucket_rows,
            block_entry_capacity: spec.block_entry_capacity,
            regions,
            total_bytes,
        })
    }

    /// Returns the version of this fixed layout geometry.
    #[must_use]
    pub(crate) const fn schema_version() -> u32 {
        PURE_DECODE_GRAPH_METADATA_LAYOUT_SCHEMA_VERSION
    }

    /// Returns the exact C07 graph bucket represented by this layout.
    #[must_use]
    pub(crate) const fn bucket_rows(self) -> u32 {
        self.bucket_rows
    }

    /// Returns the fixed cold block-entry capacity.
    #[must_use]
    pub(crate) const fn block_entry_capacity(self) -> u64 {
        self.block_entry_capacity
    }

    /// Returns the fixed region for one canonical field.
    #[must_use]
    pub(crate) const fn region(
        self,
        field: PureDecodeGraphMetadataField,
    ) -> PureDecodeGraphMetadataRegion {
        self.regions[field.index()]
    }

    /// Returns all regions in canonical field order.
    #[must_use]
    pub(crate) const fn regions(&self) -> &[PureDecodeGraphMetadataRegion; FIELD_COUNT] {
        &self.regions
    }

    /// Returns the alignment required of the slab base.
    #[must_use]
    pub(crate) const fn required_base_alignment() -> u64 {
        BASE_ALIGNMENT
    }

    /// Returns the final aligned fixed slab byte capacity.
    #[must_use]
    pub(crate) const fn total_bytes(self) -> u64 {
        self.total_bytes
    }
}

fn validate_cold_spec(
    spec: PureDecodeGraphMetadataLayoutSpec,
) -> PureDecodeGraphMetadataLayoutResult<u64> {
    if !PURE_DECODE_GRAPH_BUCKETS.contains(&spec.bucket_rows) {
        return Err(PureDecodeGraphMetadataLayoutError::UnsupportedBucket {
            bucket_rows: spec.bucket_rows,
        });
    }
    let bucket_rows = u64::from(spec.bucket_rows);
    if spec.block_entry_capacity < bucket_rows {
        return Err(
            PureDecodeGraphMetadataLayoutError::BlockEntryCapacityTooSmall {
                bucket_rows: spec.bucket_rows,
                block_entry_capacity: spec.block_entry_capacity,
            },
        );
    }
    if spec.header_bytes == 0 {
        return Err(PureDecodeGraphMetadataLayoutError::ZeroSizedOpaqueRegion {
            field: PureDecodeGraphMetadataField::Header,
        });
    }
    if spec.control_status_bytes == 0 {
        return Err(PureDecodeGraphMetadataLayoutError::ZeroSizedOpaqueRegion {
            field: PureDecodeGraphMetadataField::ControlStatus,
        });
    }
    Ok(bucket_rows)
}

fn fixed_field_specs(
    spec: PureDecodeGraphMetadataLayoutSpec,
    bucket_rows: u64,
) -> PureDecodeGraphMetadataLayoutResult<[PureDecodeGraphMetadataFieldSpec; FIELD_COUNT]> {
    let sequence_block_offset_count = bucket_rows.checked_add(1).ok_or(
        PureDecodeGraphMetadataLayoutError::ArithmeticOverflow {
            field: PureDecodeGraphMetadataField::SequenceBlockOffsets,
        },
    )?;
    let token_bytes = checked_byte_len(
        bucket_rows,
        U32_BYTES,
        PureDecodeGraphMetadataField::TokenIds,
    )?;
    let position_bytes = checked_byte_len(
        bucket_rows,
        U32_BYTES,
        PureDecodeGraphMetadataField::PositionIds,
    )?;
    let row_sequence_slot_bytes = checked_byte_len(
        bucket_rows,
        U32_BYTES,
        PureDecodeGraphMetadataField::RowSequenceSlots,
    )?;
    let sequence_block_offset_bytes = checked_byte_len(
        sequence_block_offset_count,
        U32_BYTES,
        PureDecodeGraphMetadataField::SequenceBlockOffsets,
    )?;
    let physical_block_id_bytes = checked_byte_len(
        spec.block_entry_capacity,
        U32_BYTES,
        PureDecodeGraphMetadataField::PhysicalBlockIds,
    )?;
    let valid_token_bytes = checked_byte_len(
        spec.block_entry_capacity,
        U16_BYTES,
        PureDecodeGraphMetadataField::ValidTokens,
    )?;
    let output_token_index_bytes = checked_byte_len(
        bucket_rows,
        U32_BYTES,
        PureDecodeGraphMetadataField::OutputTokenIndices,
    )?;
    Ok([
        (
            PureDecodeGraphMetadataField::Header,
            spec.header_bytes,
            U32_BYTES,
        ),
        (
            PureDecodeGraphMetadataField::TokenIds,
            token_bytes,
            U32_BYTES,
        ),
        (
            PureDecodeGraphMetadataField::PositionIds,
            position_bytes,
            U32_BYTES,
        ),
        (
            PureDecodeGraphMetadataField::RowSequenceSlots,
            row_sequence_slot_bytes,
            U32_BYTES,
        ),
        (
            PureDecodeGraphMetadataField::SequenceBlockOffsets,
            sequence_block_offset_bytes,
            U32_BYTES,
        ),
        (
            PureDecodeGraphMetadataField::PhysicalBlockIds,
            physical_block_id_bytes,
            U32_BYTES,
        ),
        (
            PureDecodeGraphMetadataField::ValidTokens,
            valid_token_bytes,
            U16_BYTES,
        ),
        (
            PureDecodeGraphMetadataField::OutputTokenIndices,
            output_token_index_bytes,
            U32_BYTES,
        ),
        (
            PureDecodeGraphMetadataField::ControlStatus,
            spec.control_status_bytes,
            U32_BYTES,
        ),
    ])
}

fn fixed_regions(
    fields: [PureDecodeGraphMetadataFieldSpec; FIELD_COUNT],
) -> PureDecodeGraphMetadataLayoutResult<([PureDecodeGraphMetadataRegion; FIELD_COUNT], u64)> {
    let mut regions = [PureDecodeGraphMetadataRegion::ZERO; FIELD_COUNT];
    let mut cursor = 0_u64;
    for (field, byte_len, alignment) in fields {
        regions[field.index()] = push_region(&mut cursor, byte_len, alignment, field)?;
    }
    let total_bytes = align_up(
        cursor,
        BASE_ALIGNMENT,
        PureDecodeGraphMetadataField::ControlStatus,
    )?;
    Ok((regions, total_bytes))
}

fn checked_byte_len(
    element_count: u64,
    element_bytes: u64,
    field: PureDecodeGraphMetadataField,
) -> PureDecodeGraphMetadataLayoutResult<u64> {
    element_count
        .checked_mul(element_bytes)
        .ok_or(PureDecodeGraphMetadataLayoutError::ArithmeticOverflow { field })
}

fn push_region(
    cursor: &mut u64,
    byte_len: u64,
    alignment: u64,
    field: PureDecodeGraphMetadataField,
) -> PureDecodeGraphMetadataLayoutResult<PureDecodeGraphMetadataRegion> {
    let offset = align_up(*cursor, alignment, field)?;
    *cursor = offset
        .checked_add(byte_len)
        .ok_or(PureDecodeGraphMetadataLayoutError::ArithmeticOverflow { field })?;
    Ok(PureDecodeGraphMetadataRegion {
        offset,
        byte_len,
        alignment,
    })
}

fn align_up(
    value: u64,
    alignment: u64,
    field: PureDecodeGraphMetadataField,
) -> PureDecodeGraphMetadataLayoutResult<u64> {
    debug_assert!(alignment.is_power_of_two());
    let remainder = value % alignment;
    if remainder == 0 {
        Ok(value)
    } else {
        value
            .checked_add(alignment - remainder)
            .ok_or(PureDecodeGraphMetadataLayoutError::ArithmeticOverflow { field })
    }
}

#[cfg(test)]
mod tests {
    use super::{
        PURE_DECODE_GRAPH_METADATA_LAYOUT_SCHEMA_VERSION, PureDecodeGraphMetadataField,
        PureDecodeGraphMetadataLayout, PureDecodeGraphMetadataLayoutError,
        PureDecodeGraphMetadataLayoutSpec,
    };

    const fn spec(
        bucket_rows: u32,
        block_entry_capacity: u64,
        header_bytes: u64,
        control_status_bytes: u64,
    ) -> PureDecodeGraphMetadataLayoutSpec {
        PureDecodeGraphMetadataLayoutSpec::new(
            bucket_rows,
            block_entry_capacity,
            header_bytes,
            control_status_bytes,
        )
    }

    #[test]
    fn exact_catalog_buckets_have_stable_fixed_geometry() {
        for bucket_rows in [1, 2, 4, 8, 16, 32] {
            let block_entry_capacity = u64::from(bucket_rows);
            let layout = PureDecodeGraphMetadataLayout::try_new(spec(
                bucket_rows,
                block_entry_capacity,
                5,
                5,
            ))
            .expect("exact C07 bucket must prepare fixed geometry");
            assert_eq!(
                layout,
                PureDecodeGraphMetadataLayout::try_new(spec(
                    bucket_rows,
                    block_entry_capacity,
                    5,
                    5,
                ))
                .expect("same cold spec must remain deterministic")
            );
            assert_eq!(
                PureDecodeGraphMetadataLayout::schema_version(),
                PURE_DECODE_GRAPH_METADATA_LAYOUT_SCHEMA_VERSION
            );
            assert_eq!(layout.bucket_rows(), bucket_rows);
            assert_eq!(layout.block_entry_capacity(), block_entry_capacity);
        }
    }

    #[test]
    fn layout_uses_canonical_fixed_order_alignment_and_final_padding() {
        let layout = PureDecodeGraphMetadataLayout::try_new(spec(8, 16, 5, 5))
            .expect("representable cold layout");
        let expected = [
            (PureDecodeGraphMetadataField::Header, 0, 5, 4),
            (PureDecodeGraphMetadataField::TokenIds, 8, 32, 4),
            (PureDecodeGraphMetadataField::PositionIds, 40, 32, 4),
            (PureDecodeGraphMetadataField::RowSequenceSlots, 72, 32, 4),
            (
                PureDecodeGraphMetadataField::SequenceBlockOffsets,
                104,
                36,
                4,
            ),
            (PureDecodeGraphMetadataField::PhysicalBlockIds, 140, 64, 4),
            (PureDecodeGraphMetadataField::ValidTokens, 204, 32, 2),
            (PureDecodeGraphMetadataField::OutputTokenIndices, 236, 32, 4),
            (PureDecodeGraphMetadataField::ControlStatus, 268, 5, 4),
        ];

        assert_eq!(
            PureDecodeGraphMetadataField::ALL,
            expected.map(|(field, _, _, _)| field)
        );
        for (index, (field, offset, byte_len, alignment)) in expected.into_iter().enumerate() {
            let region = layout.region(field);
            assert_eq!(layout.regions()[index], region);
            assert_eq!(region.offset(), offset);
            assert_eq!(region.byte_len(), byte_len);
            assert_eq!(region.alignment(), alignment);
            assert_eq!(region.offset() % region.alignment(), 0);
            if index != 0 {
                let previous = layout.regions()[index - 1];
                assert!(previous.offset() + previous.byte_len() <= region.offset());
            }
        }
        assert_eq!(PureDecodeGraphMetadataLayout::required_base_alignment(), 4);
        assert_eq!(layout.total_bytes(), 276);
        assert_eq!(
            layout.total_bytes() % PureDecodeGraphMetadataLayout::required_base_alignment(),
            0
        );
    }

    #[test]
    fn layout_rejects_non_exact_buckets_and_invalid_cold_capacities() {
        for bucket_rows in [0, 3, 33, u32::MAX] {
            assert_eq!(
                PureDecodeGraphMetadataLayout::try_new(spec(bucket_rows, 32, 1, 1)),
                Err(PureDecodeGraphMetadataLayoutError::UnsupportedBucket { bucket_rows })
            );
        }
        assert_eq!(
            PureDecodeGraphMetadataLayout::try_new(spec(8, 7, 1, 1)),
            Err(
                PureDecodeGraphMetadataLayoutError::BlockEntryCapacityTooSmall {
                    bucket_rows: 8,
                    block_entry_capacity: 7,
                }
            )
        );
        assert_eq!(
            PureDecodeGraphMetadataLayout::try_new(spec(8, 0, 0, 0)),
            Err(
                PureDecodeGraphMetadataLayoutError::BlockEntryCapacityTooSmall {
                    bucket_rows: 8,
                    block_entry_capacity: 0,
                }
            )
        );
        assert_eq!(
            PureDecodeGraphMetadataLayout::try_new(spec(8, 8, 0, 1)),
            Err(PureDecodeGraphMetadataLayoutError::ZeroSizedOpaqueRegion {
                field: PureDecodeGraphMetadataField::Header,
            })
        );
        assert_eq!(
            PureDecodeGraphMetadataLayout::try_new(spec(8, 8, 1, 0)),
            Err(PureDecodeGraphMetadataLayoutError::ZeroSizedOpaqueRegion {
                field: PureDecodeGraphMetadataField::ControlStatus,
            })
        );
    }

    #[test]
    fn layout_fails_closed_on_fixed_field_multiplication_and_offset_overflow() {
        assert_eq!(
            PureDecodeGraphMetadataLayout::try_new(spec(1, u64::MAX, 1, 1)),
            Err(PureDecodeGraphMetadataLayoutError::ArithmeticOverflow {
                field: PureDecodeGraphMetadataField::PhysicalBlockIds,
            })
        );
        assert_eq!(
            PureDecodeGraphMetadataLayout::try_new(spec(1, 1, u64::MAX - 3, 1)),
            Err(PureDecodeGraphMetadataLayoutError::ArithmeticOverflow {
                field: PureDecodeGraphMetadataField::TokenIds,
            })
        );
    }
}
